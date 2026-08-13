import asyncio
import io
import time
from pathlib import Path
from typing import Any

import orjson
from gemini_webapi import GeminiClient
from gemini_webapi.constants import GRPC, AccountStatus
from gemini_webapi.types import (
    Candidate,
    DeepResearchPlan,
    GeneratedVideo,
    ModelOutput,
    RPCData,
)
from gemini_webapi.utils import extract_json_from_response, get_nested_value
from loguru import logger

from app.models import AppMessage
from app.utils import g_config
from app.utils.helper import (
    add_tag,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

_SESSION_VALIDATION_PROMPT = "Reply with exactly OK."
_AUTH_FAILURE_TEXT_PATTERNS = (
    "are you signed in",
    "sign in",
    "signed in",
    "log in",
    "logged in",
)


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        self._cfg_impersonate: str | None = kwargs.pop("impersonate", None)
        super().__init__(**kwargs)
        self.id = client_id
        self._initialized = False
        # Chat id of the last conversation this client opened. Its kind is not verified here -
        # it is simply the most recent cid we saw, whether that conversation is persistent or
        # temporary. Callers use it in temporary mode, where Google closes the open window as
        # soon as another conversation is created, making this the only cid that can still be
        # continuable. Deliberately in-memory and cleared on every (re)initialization: after an
        # auto-close, restart or redeploy we can no longer vouch for any window.
        self.latest_chat_cid: str | None = None

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """
        Inject default configuration values from global settings.
        """
        config = g_config.gemini
        init_kwargs: dict[str, Any] = {
            "timeout": config.timeout,
            "watchdog_timeout": config.watchdog_timeout,
            "auto_refresh": config.auto_refresh,
            "refresh_interval": config.refresh_interval,
            "auto_close": config.auto_close,
            "close_delay": config.close_delay,
            "verbose": config.verbose,
        }
        if self._cfg_impersonate is not None:
            init_kwargs["impersonate"] = self._cfg_impersonate
        try:
            await super().init(**init_kwargs)
            self._initialized = True
            self.latest_chat_cid = None
        except Exception:
            self._initialized = False
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    async def validate_session(self, timeout: int = 60) -> bool:
        """Probe the session with a minimal temporary chat.

        Detects silently-degraded sessions (signed-out / "are you signed in"
        bodies) that pass library init but would fail on the first real request.
        Uses `temporary=True` so NO conversation is persisted — standing fact:
        conversations on this account cannot be deleted.
        """
        try:
            response = await asyncio.wait_for(
                self.generate_content(_SESSION_VALIDATION_PROMPT, temporary=True),
                timeout=timeout,
            )
            response_text = getattr(response, "text", "") or ""
        except Exception as e:
            logger.warning(f"Session validation probe failed for client {self.id}: {e}")
            return False
        normalized = response_text.strip().lower()
        if not normalized or any(pattern in normalized for pattern in _AUTH_FAILURE_TEXT_PATTERNS):
            logger.warning(
                f"Session validation probe returned suspicious content for client "
                f"{self.id}: {response_text[:120]!r}"
            )
            return False
        logger.info(f"Session validation probe succeeded for client {self.id}.")
        return True

    async def fetch_videos_from_turns(self, cid: str, limit: int = 10) -> list[GeneratedVideo]:
        """Recover generated videos from the conversation-turns RPC.

        Video generation is asynchronous server-side: the live response usually
        carries only the placeholder text, while the actual video metadata lands
        in the finalized turn. The pinned dependency's extraction path
        (candidate [12][59]) does not match the observed payload shape
        (candidate[12][0]["60"][0][0][0][0] with url list at item[7]), so we
        crawl the raw turns payload directly.
        """
        resp = await self._batch_execute(
            [
                RPCData(
                    rpcid=GRPC.LIST_CONVERSATION_TURNS,
                    payload=orjson.dumps([cid, limit, None, 1, [1], [4], None, 1]).decode("utf-8"),
                )
            ]
        )
        parts = extract_json_from_response(resp.text)
        videos: list[GeneratedVideo] = []
        seen: set[str] = set()
        for part in parts:
            body = get_nested_value(part, [2])
            if not body:
                continue
            try:
                part_body = orjson.loads(body) if isinstance(body, str) else body
            except orjson.JSONDecodeError:
                continue
            for turn in get_nested_value(part_body, [0]) or []:
                for cand in get_nested_value(turn, [3, 0]) or []:
                    if not isinstance(cand, list) or len(cand) <= 12:
                        continue
                    blocks = cand[12]
                    if not isinstance(blocks, list):
                        continue
                    for block in blocks:
                        item = None
                        if isinstance(block, dict) and isinstance(block.get("60"), list):
                            nested = block["60"]
                            try:
                                item = get_nested_value(nested, [0, 0, 0, 0])
                            except Exception:
                                item = None
                        if not isinstance(item, list) or len(item) < 12:
                            continue
                        filename = get_nested_value(item, [2], "")
                        mime = get_nested_value(item, [11], "")
                        if not (str(filename).endswith(".mp4") or "video" in str(mime)):
                            continue
                        urls = get_nested_value(item, [7], []) or []
                        video_url = next(
                            (
                                u
                                for u in urls
                                if isinstance(u, str)
                                and "contribution.usercontent.google.com/download" in u
                                and "filename=video.mp4" in u
                            ),
                            None,
                        )
                        thumbnail = next(
                            (
                                u
                                for u in urls
                                if isinstance(u, str) and "lh3.googleusercontent.com" in u
                            ),
                            None,
                        )
                        if not video_url or video_url in seen:
                            continue
                        seen.add(video_url)
                        videos.append(
                            GeneratedVideo(
                                url=video_url,
                                thumbnail=thumbnail or "",
                                cid=cid,
                                client_ref=self,
                                proxy=self.proxy,
                            )
                        )
        return videos

    async def fetch_last_model_turn(self, cid: str, limit: int = 10) -> ModelOutput | None:
        """Fetch the newest completed model turn from the conversation-turns RPC.

        Used for stream recovery: when a stream dies mid-response, Gemini still
        finalizes the complete turn server-side. Returns the newest completed
        model turn (full text + thoughts), or None when no finalized turn is
        present yet — callers should poll. Extraction mirrors the dependency's
        own read_chat shape: text at candidate[1][0], thoughts at [37][0][0],
        completion indicator at [8][0] == 2.
        """
        resp = await self._batch_execute(
            [
                RPCData(
                    rpcid=GRPC.LIST_CONVERSATION_TURNS,
                    payload=orjson.dumps([cid, limit, None, 1, [1], [4], None, 1]).decode("utf-8"),
                )
            ]
        )
        parts = extract_json_from_response(resp.text)
        for part in parts:
            body = get_nested_value(part, [2])
            if not body:
                continue
            try:
                part_body = orjson.loads(body) if isinstance(body, str) else body
            except orjson.JSONDecodeError:
                continue
            for turn in get_nested_value(part_body, [0]) or []:
                rid = get_nested_value(turn, [0, 1], "")
                candidates = get_nested_value(turn, [3, 0]) or []
                for candidate_data in candidates:
                    if not isinstance(candidate_data, list):
                        continue
                    rcid = get_nested_value(candidate_data, [0], "")
                    if not rcid:
                        continue
                    if get_nested_value(candidate_data, [8, 0]) != 2:
                        continue
                    text = get_nested_value(candidate_data, [1, 0], "") or ""
                    thoughts = get_nested_value(candidate_data, [37, 0, 0]) or ""
                    if not (text or thoughts):
                        continue
                    candidate = Candidate(
                        rcid=rcid,
                        text=text,
                        text_delta=text,
                        thoughts=thoughts,
                        thoughts_delta=thoughts,
                    )
                    return ModelOutput(metadata=[cid, rid], candidates=[candidate])
        return None

    async def fetch_deep_research_report(
        self,
        plan: DeepResearchPlan,
        poll_interval: float = 10.0,
        timeout: float = 600.0,
        min_report_chars: int = 1500,
    ) -> str | None:
        """Poll the conversation-turns RPC for a completed deep-research report.

        Fallback for accounts where ``plan.research_id`` never materializes and
        the dependency's ``wait_for_deep_research`` raises before any poll
        (research_mixin.py:333). Research still completes server-side; the full
        report text lands in the completed turn's research block
        (candidate[30][0][4], measured 19,787-26,356 chars on this account —
        the plain text slot candidate[1][0] only carries a short completion
        note). Same crawl shape as ``fetch_last_model_turn`` (rcid + completion
        gate [8][0]==2 are both required), extended to extract the research
        block. Returns the longest report-scale completed-turn text once it has
        stayed stable across a confirmation poll, or None on timeout (the
        caller decides the HTTP mapping). ``min_report_chars`` = 1500: measured
        progress notes are 142-169 chars and plan text 144, so 1500 separates
        reports (>10x) without a false positive.
        """
        cid = plan.cid or ""
        if not cid:
            logger.warning("Deep research fallback: plan.cid is missing; cannot poll turns RPC.")
            return None
        deadline = time.monotonic() + timeout
        best_text: str | None = None
        best_rcid = ""
        stable_polls = 0
        while time.monotonic() < deadline:
            try:
                report_texts = await self._crawl_report_texts(cid, limit=10)
            except Exception as e:
                logger.warning(
                    f"Deep research fallback: turns RPC failed for {cid} "
                    f"({type(e).__name__}: {e}); retrying until timeout."
                )
                report_texts = []
            current_rcid, current_text = self._longest_report_scale(report_texts, min_report_chars)
            elapsed = timeout - max(0.0, deadline - time.monotonic())
            logger.info(
                f"Deep research fallback poll (cid={cid}): elapsed={elapsed:.0f}s, "
                f"longest_report_scale={len(current_text) if current_text else 0} chars, "
                f"best={len(best_text) if best_text else 0} chars"
            )
            if current_text is not None and (
                best_text is None or len(current_text) > len(best_text)
            ):
                best_text = current_text
                best_rcid = current_rcid
                stable_polls = 0
            elif best_text is not None:
                if current_rcid != best_rcid:
                    logger.info(
                        f"Deep research fallback: report turn {best_rcid} superseded by "
                        f"{current_rcid} (shorter); settling on longest seen."
                    )
                    return best_text
                stable_polls += 1
                if stable_polls >= 1:
                    logger.info(
                        f"Deep research fallback: report crawl settled on {len(best_text)} "
                        f"chars (rcid {best_rcid})."
                    )
                    return best_text
            await asyncio.sleep(poll_interval)
        logger.warning(
            f"Deep research fallback: timed out after {timeout}s (best "
            f"{len(best_text) if best_text else 0} chars); research may still "
            f"complete in Gemini web history."
        )
        return None

    async def _crawl_report_texts(self, cid: str, limit: int = 10) -> list[tuple[str, str]]:
        """Crawl completed model turns and return (rcid, report-scale text) pairs."""
        resp = await self._batch_execute(
            [
                RPCData(
                    rpcid=GRPC.LIST_CONVERSATION_TURNS,
                    payload=orjson.dumps([cid, limit, None, 1, [1], [4], None, 1]).decode("utf-8"),
                )
            ]
        )
        texts: list[tuple[str, str]] = []
        parts = extract_json_from_response(resp.text)
        for part in parts:
            body = get_nested_value(part, [2])
            if not body:
                continue
            try:
                part_body = orjson.loads(body) if isinstance(body, str) else body
            except orjson.JSONDecodeError:
                continue
            for turn in get_nested_value(part_body, [0]) or []:
                candidates = get_nested_value(turn, [3, 0]) or []
                for candidate_data in candidates:
                    if not isinstance(candidate_data, list):
                        continue
                    rcid = get_nested_value(candidate_data, [0], "")
                    if not rcid:
                        continue
                    if get_nested_value(candidate_data, [8, 0]) != 2:
                        continue
                    block = get_nested_value(candidate_data, [30], None)
                    report_text = ""
                    if isinstance(block, list) and block and isinstance(block[0], list):
                        head = block[0]
                        if len(head) > 4 and isinstance(head[4], str):
                            report_text = head[4]
                    plain_text = get_nested_value(candidate_data, [1, 0], "") or ""
                    candidate_text = (
                        report_text if len(report_text) >= len(plain_text) else plain_text
                    )
                    if candidate_text:
                        texts.append((rcid, candidate_text))
        return texts

    @staticmethod
    def _longest_report_scale(
        texts: list[tuple[str, str]], min_report_chars: int
    ) -> tuple[str, str | None]:
        """Pick the longest completed-turn text at or above the report threshold."""
        longest_rcid = ""
        longest_text: str | None = None
        longest_len = 0
        for rcid, text in texts:
            if len(text) >= min_report_chars and len(text) > longest_len:
                longest_rcid = rcid
                longest_text = text
                longest_len = len(text)
        return longest_rcid, longest_text

    def is_healthy(self) -> bool:
        """
        Check if the client is healthy.

        A client is healthy if it is active (running or initialized with auto-close)
        and the account status is available.
        """
        is_active = self._running or (self.auto_close and self._initialized)
        return is_active and self.account_status == AccountStatus.AVAILABLE

    @staticmethod
    async def _process_content_item(
        item: Any, role: str, tempdir: Path | None
    ) -> tuple[str | None, Path | str | None]:
        """
        Process a single content item (text, image_url, file, input_audio).
        Returns a tuple of (text_fragment, file_path).
        """
        if item.type == "text":
            item_text = getattr(item, "text", "") or ""
            if item_text or role == "tool":
                return item_text, None
        elif item.type == "image_url":
            if item_media_url := getattr(item, "url", None):
                return None, await save_url_to_tempfile(item_media_url, tempdir)
            raise ValueError(f"{item.type} cannot be empty")
        elif item.type == "file":
            if not (file_data := getattr(item, "file_data", None)):
                raise ValueError("File must contain 'file_data'")
            filename = getattr(item, "filename", "") or ""
            return None, await save_file_to_tempfile(file_data, filename, tempdir)
        elif item.type == "input_audio":
            if file_data := getattr(item, "file_data", None):
                return None, await save_file_to_tempfile(file_data, "audio.wav", tempdir)
            raise ValueError("input_audio must contain 'file_data' key")
        return None, None

    @staticmethod
    async def _extract_content_and_files(
        message: AppMessage, tempdir: Path | None
    ) -> tuple[list[str], list[Path | str]]:
        """
        Extract text fragments and files from message content.
        """
        files: list[Path | str] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            if message.content or message.role == "tool":
                text_fragments.append(message.content or "")
        elif isinstance(message.content, list):
            for item in message.content:
                text, file = await GeminiClientWrapper._process_content_item(
                    item, message.role, tempdir
                )
                if text is not None:
                    text_fragments.append(text)
                if file is not None:
                    files.append(file)
        elif message.content is None and message.role == "tool":
            text_fragments.append("")
        elif message.content is not None:
            raise ValueError(f"Unsupported message content type: {type(message.content)}")

        return text_fragments, files

    @staticmethod
    def _format_tool_results(
        text_fragments: list[str], tool_name: str | None, wrap_tool: bool
    ) -> list[str]:
        """
        Format tool results into the PascalCase technical protocol blocks.
        """
        tool_name = tool_name or "unknown"
        combined_content = "\n".join(text_fragments).strip()
        res_block = (
            f"[Result:{tool_name}]\n[ToolResult]\n{combined_content}\n[/ToolResult]\n[/Result]"
        )
        return [f"[ToolResults]\n{res_block}\n[/ToolResults]"] if wrap_tool else [res_block]

    @staticmethod
    def _format_tool_calls(message: AppMessage) -> str | None:
        """
        Format tool calls into the PascalCase technical protocol blocks.
        """
        if not message.tool_calls:
            return None

        tool_blocks: list[str] = []
        for call in message.tool_calls:
            params_text = call.function.arguments.strip()
            formatted_params = ""
            if params_text:
                try:
                    parsed_params = orjson.loads(params_text)
                    if isinstance(parsed_params, dict):
                        for k, v in parsed_params.items():
                            val_str = v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                            formatted_params += (
                                f"[CallParameter:{k}]\n```\n{val_str}\n```\n[/CallParameter]\n"
                            )
                    else:
                        formatted_params += f"```\n{params_text}\n```\n"
                except orjson.JSONDecodeError:
                    formatted_params += f"```\n{params_text}\n```\n"

            tool_blocks.append(f"[Call:{call.function.name}]\n{formatted_params}[/Call]")

        return "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]" if tool_blocks else None

    @staticmethod
    async def process_message(
        message: AppMessage,
        tempdir: Path | None = None,
        tagged: bool = True,
        wrap_tool: bool = True,
    ) -> tuple[str, list[Path | str]]:
        """
        Process a Message into Gemini API format using the PascalCase technical protocol.
        Extracts text, handles files, and appends ToolCalls/ToolResults blocks.
        """
        text_fragments, files = await GeminiClientWrapper._extract_content_and_files(
            message, tempdir
        )

        if message.role == "tool":
            text_fragments = GeminiClientWrapper._format_tool_results(
                text_fragments, message.name, wrap_tool
            )

        if tool_section := GeminiClientWrapper._format_tool_calls(message):
            text_fragments.append(tool_section)

        model_input = "\n".join(fragment for fragment in text_fragments if fragment is not None)

        if (model_input or message.role == "tool") and tagged:
            model_input = add_tag(message.role, model_input)

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[AppMessage], tempdir: Path | None = None
    ) -> tuple[str, list[str | Path | bytes | io.BytesIO]]:
        conversation: list[str] = []
        files: list[str | Path | bytes | io.BytesIO] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "tool":
                tool_blocks: list[str] = []
                while i < len(messages) and messages[i].role == "tool":
                    part, part_files = await GeminiClientWrapper.process_message(
                        messages[i], tempdir, tagged=False, wrap_tool=False
                    )
                    tool_blocks.append(part)
                    files.extend(part_files)
                    i += 1

                combined_tool_content = "\n".join(tool_blocks)
                wrapped_content = f"[ToolResults]\n{combined_tool_content}\n[/ToolResults]"
                conversation.append(add_tag("tool", wrapped_content))
            else:
                input_part, files_part = await GeminiClientWrapper.process_message(
                    msg, tempdir, tagged=True
                )
                conversation.append(input_part)
                files.extend(files_part)
                i += 1

        conversation.append(add_tag("assistant", "", unclose=True))
        return "\n".join(conversation), files
