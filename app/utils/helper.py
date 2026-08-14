import base64
import hashlib
import html
import ipaddress
import mimetypes
import re
import reprlib
import socket
import struct
import tempfile
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import orjson
from curl_cffi import CurlFollow, CurlHttpVersion, requests
from loguru import logger
from pydantic import BaseModel

from app.models import (
    AppContentItem,
    AppMessage,
    AppMessageRole,
    AppToolCall,
    AppToolCallFunction,
    ChatCompletionMessage,
    ChatCompletionNamedToolChoice,
    ImageGeneration,
    StructuredOutputRequirement,
    ToolChoiceFunction,
    ToolChoiceTypes,
    VideoGeneration,
)
from app.utils import g_config

MAX_REMOTE_FETCH_BYTES = 20 * 1024 * 1024
_HTML_SNIFF_PREFIXES = (b"<!doctype", b"<html", b"<script")

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None

VALID_TAG_ROLES = {"user", "assistant", "system", "tool"}
TOOL_WRAP_HINT = (
    "\n\n### SYSTEM: TOOL CALLING PROTOCOL (MANDATORY) ###\n"
    "If tool execution is required, you MUST adhere to this EXACT protocol. No exceptions.\n\n"
    "1. OUTPUT RESTRICTION: Your response MUST contain ONLY the [ToolCalls] block. Conversational filler, preambles, or concluding remarks are STRICTLY PROHIBITED.\n"
    "2. WRAPPING LOGIC: Every parameter value MUST be enclosed in a markdown code block. Use 3 backticks (```) by default. If the value contains backticks, the outer fence MUST be longer than any sequence inside (e.g., ````).\n"
    "3. TAG SYMMETRY: All tags MUST be balanced and closed in the exact reverse order of opening. Incomplete or unclosed blocks are strictly prohibited.\n\n"
    "REQUIRED SYNTAX:\n"
    "[ToolCalls]\n"
    "[Call:tool_name]\n"
    "[CallParameter:parameter_name]\n"
    "```\n"
    "value\n"
    "```\n"
    "[/CallParameter]\n"
    "[/Call]\n"
    "[/ToolCalls]\n\n"
    "CRITICAL: Do NOT mix natural language with protocol tags. Either respond naturally OR provide the protocol block alone. There is no middle ground."
)
STRUCTURED_JSON_WRAP_HINT = (
    "\n\n### SYSTEM: STRUCTURED JSON PROTOCOL (MANDATORY) ###\n"
    "Return ONLY one markdown code block containing a single strict JSON document that conforms to the provided JSON Schema.\n"
    "Use ```json by default. If the JSON contains backticks, the outer fence MUST be longer than any backtick sequence inside (e.g., ````json).\n"
    "REQUIRED SYNTAX:\n"
    "```json\n"
    '{"field":"value"}\n'
    "```\n\n"
    "CRITICAL: Do NOT mix natural language with the fenced JSON block. Provide the protocol block alone. There is no middle ground."
)
TOOL_BLOCK_RE = re.compile(
    r"\\?\[ToolCalls\\?](.*?)\\?\[\\?/ToolCalls\\?]",
    re.DOTALL | re.IGNORECASE,
)
TOOL_CALL_RE = re.compile(
    r"\\?\[Call\\?:(?P<name>[^]]+)\\?](?P<body>.*?)\\?\[\\?/Call\\?]",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_BLOCK_RE = re.compile(
    r"\\?\[ToolResults\\?](.*?)\\?\[\\?/ToolResults\\?]",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_ITEM_RE = re.compile(
    r"\\?\[Result\\?:(?P<name>[^]]+)\\?](?P<body>.*?)\\?\[\\?/Result\\?]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_ARG_RE = re.compile(
    r"\\?\[CallParameter\\?:(?P<name>[^]]+)\\?](?P<body>.*?)\\?\[\\?/CallParameter\\?]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_RESULT_RE = re.compile(
    r"\\?\[ToolResult\\?](.*?)\\?\[\\?/ToolResult\\?]",
    re.DOTALL | re.IGNORECASE,
)
CONTROL_TOKEN_RE = re.compile(r"\\?<\\?\|im\\?_(?:start|end)\\?\|\\?>", re.IGNORECASE)
CHATML_START_RE = re.compile(r"\\?<\\?\|im\\?_start\\?\|\\?>(\w+)\n?", re.IGNORECASE)
CHATML_END_RE = re.compile(r"\\?<\\?\|im\\?_end\\?\|\\?>", re.IGNORECASE)
COMMONMARK_UNESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
PARAM_FENCE_RE = re.compile(r"^(?P<fence>`{3,})")
TOOL_HINT_STRIPPED = TOOL_WRAP_HINT.strip()
_hint_lines = [line.strip() for line in TOOL_WRAP_HINT.split("\n") if line.strip()]
TOOL_HINT_LINE_START = _hint_lines[0] if _hint_lines else ""
TOOL_HINT_LINE_END = _hint_lines[-1] if _hint_lines else ""
TOOL_HINT_START_ESC = re.escape(TOOL_HINT_LINE_START) if TOOL_HINT_LINE_START else ""
TOOL_HINT_END_ESC = re.escape(TOOL_HINT_LINE_END) if TOOL_HINT_LINE_END else ""

HINT_FULL_RE = (
    re.compile(rf"\n?{TOOL_HINT_START_ESC}:?.*?{TOOL_HINT_END_ESC}\n?", re.DOTALL | re.IGNORECASE)
    if TOOL_HINT_START_ESC and TOOL_HINT_END_ESC
    else None
)
HINT_START_RE = (
    re.compile(rf"\n?{TOOL_HINT_START_ESC}:?\s*", re.IGNORECASE) if TOOL_HINT_START_ESC else None
)
HINT_END_RE = (
    re.compile(rf"\s*{TOOL_HINT_END_ESC}\n?", re.IGNORECASE) if TOOL_HINT_END_ESC else None
)

# --- Streaming Specific Patterns ---
_START_PATTERNS = {
    "TOOL": r"\\?\[ToolCalls\\?]",
    "ORPHAN": r"\\?\[Call\\?:[^]]+\\?]",
    "RESP": r"\\?\[ToolResults\\?]",
    "ARG": r"\\?\[CallParameter\\?:[^]]+\\?]",
    "RESULT": r"\\?\[ToolResult\\?]",
    "ITEM": r"\\?\[Result\\?:[^]]+\\?]",
    "TAG": r"\\?<\\?\|im\\?_start\\?\|\\?>",
}

_PROTOCOL_ENDS = r"\\?\[\\?/(?:ToolCalls|Call|ToolResults|CallParameter|ToolResult|Result)\\?]"
_TAG_END = r"\\?<\\?\|im\\?_end\\?\|\\?>"

if TOOL_HINT_START_ESC and TOOL_HINT_END_ESC:
    _START_PATTERNS["HINT"] = rf"\n?{TOOL_HINT_START_ESC}:?\s*"

_master_parts = [f"(?P<{name}_START>{pattern})" for name, pattern in _START_PATTERNS.items()]
_master_parts.extend((f"(?P<PROTOCOL_EXIT>{_PROTOCOL_ENDS})", f"(?P<TAG_EXIT>{_TAG_END})"))
if TOOL_HINT_START_ESC and TOOL_HINT_END_ESC:
    _master_parts.append(f"(?P<HINT_EXIT>{TOOL_HINT_END_ESC}\n?)")

STREAM_MASTER_RE = re.compile("|".join(_master_parts), re.IGNORECASE)
STREAM_TAIL_RE = re.compile(
    r"(?:\\|\\?\[[^]]*|\\?<\\?\|?i?m?\\?_?(?:s?t?a?r?t?|e?n?d?)\\?\|?\\?>?)$",
    re.IGNORECASE,
)


def add_tag(role: str, content: str, unclose: bool = False) -> str:
    """Surround content with ChatML role tags."""
    if role not in VALID_TAG_ROLES:
        logger.warning(f"Unknown role: {role}, returning content without tags")
        return content

    return f"<|im_start|>{role}\n{content}" + ("" if unclose else "\n<|im_end|>")


def normalize_llm_text(s: str) -> str:
    """
    Safely normalize LLM-generated text for both display and hashing.
    Includes: HTML unescaping, NFC normalization, and line ending standardization.
    """
    if not s:
        return ""

    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    return s.replace("\r\n", "\n").replace("\r", "\n")


def unescape_text(s: str) -> str:
    """Remove CommonMark backslash escapes from LLM-generated text."""
    return COMMONMARK_UNESCAPE_RE.sub(r"\1", s) if s else ""


def strip_markdown_fence(s: str) -> str:
    """
    Remove one outer Markdown code fence layer for protected LLM payloads.

    The fence length is detected from the opening fence so tool parameters and
    structured JSON can safely contain shorter backtick sequences inside.
    """
    s = s.strip()
    if not s:
        return ""

    match = PARAM_FENCE_RE.match(s)
    if not match or not s.endswith(match.group("fence")):
        return s

    fence = match.group("fence")
    lines = s.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == fence:
        return "\n".join(lines[1:-1])

    return s[len(fence) : -len(fence)].strip()


def _parse_tool_argument_value(raw_value: str) -> JsonValue:
    """
    Convert a tagged tool argument into the most specific JSON-compatible value.

    JSON literals, arrays, and objects are preserved so downstream clients receive
    strict argument types, while plain text values remain strings for compatibility.
    """
    value = strip_markdown_fence(raw_value)
    if not value:
        return ""

    try:
        parsed_value: Any = orjson.loads(value)
    except orjson.JSONDecodeError:
        return value

    return parsed_value


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens heuristically based on character count."""
    return len(text) // 3 if text else 0


async def save_file_to_tempfile(
    file_in_base64: str | bytes, file_name: str = "", tempdir: Path | None = None
) -> Path:
    """Decode base64 file data and save to a temporary file."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file_name).suffix if file_name else ".bin", dir=tempdir
    ) as tmp:
        tmp.write(base64.b64decode(file_in_base64))
        return Path(tmp.name)


def reject_unsafe_url(url: str) -> None:
    """Reject remote URLs that could target internal/private networks (SSRF guard).

    Allows only http/https. When `gemini.allow_private_url_fetch` is false (default),
    any resolved address that is loopback, RFC1918-private, link-local, reserved,
    multicast or unspecified is refused. DNS-rebinding TOCTOU between resolve and
    fetch is a known residual; the opt-out knob exists for localhost-served images.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")

    if g_config.gemini.allow_private_url_fetch:
        return

    try:
        addrinfos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}") from e

    for addrinfo in addrinfos:
        ip = ipaddress.ip_address(addrinfo[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                f"Refusing to fetch private/reserved address {ip} for host {host!r} "
                "(set gemini.allow_private_url_fetch=true to override)"
            )


def looks_like_html(content_type: str | None, body: bytes) -> bool:
    """True when a fetched body is HTML — a classic SSRF/CSRF smuggling signal."""
    ctype = (content_type or "").lower()
    if "html" in ctype:
        return True
    head = body[:2048].lstrip().lower()
    return any(head.startswith(prefix) for prefix in _HTML_SNIFF_PREFIXES)


async def save_url_to_tempfile(url: str, tempdir: Path | None = None) -> Path:
    """Download content from a URL and save to a temporary file."""
    data: bytes | None = None
    suffix: str | None = None
    if url.startswith("data:"):
        metadata_part = url.split(",")[0]
        mime_type = metadata_part.split(":")[1].split(";")[0]
        data = base64.b64decode(url.split(",")[1])
        suffix = mimetypes.guess_extension(mime_type) or (
            f".{mime_type.split('/')[1]}" if "/" in mime_type else ".bin"
        )
    else:
        reject_unsafe_url(url)
        async with requests.AsyncSession(
            impersonate="chrome",
            allow_redirects=CurlFollow.SAFE,
            http_version=CurlHttpVersion.NONE,
            timeout=g_config.gemini.url_fetch_timeout,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = await resp.acontent()
            content_type = resp.headers.get("content-type")
        if len(data) > MAX_REMOTE_FETCH_BYTES:
            raise ValueError(f"Remote fetch exceeded {MAX_REMOTE_FETCH_BYTES} bytes: {url}")
        if looks_like_html(content_type, data):
            raise ValueError(f"Refusing to save fetched content that looks like HTML: {url}")
        if content_type:
            suffix = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if not suffix:
            suffix = Path(urlparse(url).path).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempdir) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def strip_tagged_blocks(text: str) -> str:
    """
    Remove ChatML role blocks (<|im_start|>role...<|im_end|>).
    Role 'tool' blocks are removed entirely; others have markers stripped but content preserved.
    """
    if not text:
        return text

    result = []
    idx = 0
    while idx < len(text):
        match_start = CHATML_START_RE.search(text, idx)
        if not match_start:
            result.append(text[idx:])
            break

        result.append(text[idx : match_start.start()])
        role = match_start.group(1).lower()
        content_start = match_start.end()

        match_end = CHATML_END_RE.search(text, content_start)
        if not match_end:
            if role != "tool":
                result.append(text[content_start:])
            break

        if role != "tool":
            result.append(text[content_start : match_end.start()])
        idx = match_end.end()

    return "".join(result)


def strip_system_hints(text: str) -> str:
    """Remove system hints, ChatML tags, and technical protocol markers from text."""
    if not text:
        return text

    t_unescaped = unescape_text(text)

    cleaned = t_unescaped.replace(TOOL_WRAP_HINT, "").replace(TOOL_HINT_STRIPPED, "")

    if HINT_FULL_RE:
        cleaned = HINT_FULL_RE.sub("", cleaned)
    if HINT_START_RE:
        cleaned = HINT_START_RE.sub("", cleaned)
    if HINT_END_RE:
        cleaned = HINT_END_RE.sub("", cleaned)

    cleaned = strip_tagged_blocks(cleaned)
    cleaned = CONTROL_TOKEN_RE.sub("", cleaned)
    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = TOOL_CALL_RE.sub("", cleaned)
    cleaned = RESPONSE_BLOCK_RE.sub("", cleaned)
    cleaned = RESPONSE_ITEM_RE.sub("", cleaned)
    cleaned = TAGGED_ARG_RE.sub("", cleaned)
    return TAGGED_RESULT_RE.sub("", cleaned)


def _process_tools_internal(text: str, extract: bool = True) -> tuple[str, list[AppToolCall]]:
    """
    Extract tool metadata and return text stripped of technical markers.
    Tagged arguments preserve JSON-compatible types and receive deterministic call IDs.
    """
    if not text:
        return text, []

    tool_calls: list[AppToolCall] = []

    def _create_tool_call(name: str, raw_args: str) -> None:
        if not extract:
            return
        if not name:
            logger.warning("Encountered tool_call without a function name.")
            return

        name = unescape_text(name.strip())
        raw_args = unescape_text(raw_args)

        arg_matches = TAGGED_ARG_RE.findall(raw_args)
        if arg_matches:
            args_dict = {
                arg_name.strip(): _parse_tool_argument_value(arg_value)
                for arg_name, arg_value in arg_matches
            }
            arguments = orjson.dumps(args_dict).decode("utf-8")
            logger.debug(f"Successfully parsed {len(args_dict)} arguments for tool: {name}")
        else:
            cleaned_raw = raw_args.strip()
            if not cleaned_raw:
                logger.debug(f"Successfully parsed 0 arguments for tool: {name}")
            else:
                logger.warning(
                    f"Malformed arguments for tool '{name}'. Text found but no valid tags: {reprlib.repr(cleaned_raw)}"
                )
            arguments = "{}"

        index = len(tool_calls)
        seed = f"{name}:{arguments}:{index}".encode()
        call_id = f"call_{hashlib.sha256(seed).hexdigest()[:24]}"

        tool_calls.append(
            AppToolCall(
                id=call_id,
                type="function",
                function=AppToolCallFunction(name=name, arguments=arguments),
            )
        )

    for match in TOOL_CALL_RE.finditer(text):
        _create_tool_call(match.group(1), match.group(2))

    cleaned = strip_system_hints(text)
    return cleaned, tool_calls


def remove_tool_call_blocks(text: str) -> str:
    """Strip tool call blocks from text for display."""
    cleaned, _ = _process_tools_internal(text, extract=False)
    return cleaned


def extract_tool_calls(text: str) -> tuple[str, list[AppToolCall]]:
    """Extract tool calls and return cleaned text."""
    return _process_tools_internal(text, extract=True)


def text_from_message(message: AppMessage) -> str:
    """Concatenate text and tool arguments from a message for token estimation."""
    base_text = ""
    if isinstance(message.content, str):
        base_text = message.content
    elif isinstance(message.content, list):
        base_text = "\n".join(
            item.text or "" for item in message.content if getattr(item, "type", "") == "text"
        )
    elif message.content is None:
        base_text = ""

    if message.tool_calls:
        tool_arg_text = "".join(call.function.arguments or "" for call in message.tool_calls)
        base_text = f"{base_text}\n{tool_arg_text}" if base_text else tool_arg_text

    return base_text


def extract_image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Return image dimensions (width, height) if PNG or JPEG headers are present."""
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        try:
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        except struct.error:
            return None, None

    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        idx = 2
        length = len(data)
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while idx < length:
            if data[idx] != 0xFF:
                idx += 1
                continue
            while idx < length and data[idx] == 0xFF:
                idx += 1
            if idx >= length:
                break
            marker = data[idx]
            idx += 1
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            if idx + 1 >= length:
                break
            segment_length = (data[idx] << 8) + data[idx + 1]
            idx += 2
            if segment_length < 2:
                break
            if marker in sof_markers:
                if idx + 4 < length:
                    height = (data[idx + 1] << 8) + data[idx + 2]
                    width = (data[idx + 3] << 8) + data[idx + 4]
                    return int(width), int(height)
                break
            idx += segment_length - 2
    return None, None


def detect_image_extension(data: bytes) -> str | None:
    """Detect image extension from magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    return ".webp" if data.startswith(b"RIFF") and data[8:12] == b"WEBP" else None


def dump_model(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model into a JSON-compatible dict with None values excluded."""
    return model.model_dump(mode="json", exclude_none=True)


def serialize_tools_for_response(tools: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Serialize tool objects into clean dictionary representations without None values."""
    if not tools:
        return []
    result: list[dict[str, Any]] = []
    for t in tools:
        if hasattr(t, "model_dump"):
            result.append(t.model_dump(exclude_none=True))
        elif hasattr(t, "dict"):
            result.append(t.dict(exclude_none=True))
        elif isinstance(t, dict):
            result.append({k: v for k, v in t.items() if v is not None})
        else:
            result.append(t)
    return result


def serialize_tool_choice_for_response(tool_choice: Any) -> Any:
    """Serialize tool choice object into a clean dictionary or string representation."""
    if tool_choice is None:
        return "auto"
    if hasattr(tool_choice, "model_dump"):
        return tool_choice.model_dump(exclude_none=True)
    if hasattr(tool_choice, "dict"):
        return tool_choice.dict(exclude_none=True)
    return tool_choice


def calculate_usage(
    messages: list[AppMessage],
    assistant_text: str | None,
    tool_calls: list[AppToolCall] | None,
    thoughts: str | None = None,
) -> tuple[int, int, int, int]:
    """Calculate prompt, completion, total and reasoning tokens consistently."""
    prompt_tokens = sum(estimate_tokens(text_from_message(msg)) for msg in messages)
    tool_args_text = ""
    if tool_calls:
        for call in tool_calls:
            tool_args_text += call.function.arguments or ""

    completion_basis = assistant_text or ""
    if tool_args_text:
        completion_basis = (
            f"{completion_basis}\n{tool_args_text}" if completion_basis else tool_args_text
        )

    completion_tokens = estimate_tokens(completion_basis)
    reasoning_tokens = estimate_tokens(thoughts) if thoughts else 0
    total_completion_tokens = completion_tokens + reasoning_tokens

    return (
        prompt_tokens,
        total_completion_tokens,
        prompt_tokens + total_completion_tokens,
        reasoning_tokens,
    )


def normalize_app_message_role(role_name: str) -> AppMessageRole:
    """Normalize and validate input role string to a valid AppMessage role."""
    roles: dict[str, AppMessageRole] = {
        "developer": "system",
        "function": "tool",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
        "system": "system",
    }
    return roles.get(role_name, "system")


def convert_to_app_messages(messages: list[ChatCompletionMessage]) -> list[AppMessage]:
    """Convert OpenAI ChatCompletionMessage list into AppMessage format."""
    app_messages: list[AppMessage] = []
    for msg in messages:
        app_content: str | list[AppContentItem] | None = None
        if isinstance(msg.content, str):
            app_content = msg.content
        elif isinstance(msg.content, list):
            app_content = []
            for item in msg.content:
                if item.type == "text":
                    app_content.append(AppContentItem(type="text", text=item.text))
                elif item.type == "image_url":
                    media_dict = getattr(item, "image_url", None)
                    url = media_dict.get("url") if media_dict else None
                    app_content.append(AppContentItem(type="image_url", url=url))
                elif item.type == "file":
                    file_dict = getattr(item, "file", None)
                    filename = file_dict.get("filename") if file_dict else None
                    file_data = file_dict.get("file_data") if file_dict else None
                    app_content.append(
                        AppContentItem(type="file", filename=filename, file_data=file_data)
                    )
                elif item.type == "input_audio":
                    audio_dict = getattr(item, "input_audio", None)
                    audio_data = audio_dict.get("data") if audio_dict else None
                    app_content.append(
                        AppContentItem(
                            type="input_audio",
                            file_data=audio_data,
                            raw_data=audio_dict,
                        )
                    )
                elif item.type in ("refusal", "reasoning"):
                    text_val = getattr(item, "text", None) or getattr(item, item.type, None)
                    app_content.append(AppContentItem(type=item.type, text=text_val))

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                AppToolCall(
                    id=tc.id,
                    type="function",
                    function=AppToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        role = normalize_app_message_role(msg.role)

        app_messages.append(
            AppMessage(
                role=role,
                content=app_content,
                tool_calls=tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                reasoning_content=getattr(msg, "reasoning_content", None),
            )
        )
    return app_messages


def canonicalize_structured_output(
    visible_output: str, structured_requirement: StructuredOutputRequirement
) -> str | None:
    """Parse raw or fenced structured JSON and return its canonical JSON representation."""
    candidate = strip_markdown_fence(visible_output)
    try:
        structured_payload = orjson.loads(candidate)
    except orjson.JSONDecodeError:
        logger.warning(
            f"Failed to decode JSON for structured response (schema={structured_requirement.schema_name})."
        )
        return None

    canonical_output = orjson.dumps(structured_payload).decode("utf-8")
    logger.debug(f"Structured response fulfilled (schema={structured_requirement.schema_name}).")
    return canonical_output


def process_llm_output(
    thoughts: str | None,
    raw_text: str,
    structured_requirement: StructuredOutputRequirement | None,
) -> tuple[str | None, str, str, list[AppToolCall]]:
    """
    Post-process Gemini output to extract tool calls, unwrap structured JSON fences, and prepare clean text for display and storage.
    Returns: (thoughts, visible_text, storage_output, tool_calls)
    """
    if thoughts:
        thoughts = thoughts.strip()

    visible_output, tool_calls = extract_tool_calls(raw_text)
    if tool_calls:
        logger.debug(f"Detected {len(tool_calls)} tool call(s) in model output.")

    visible_output = visible_output.strip()
    storage_output = visible_output

    if (
        structured_requirement
        and visible_output
        and (
            canonical_output := canonicalize_structured_output(
                visible_output, structured_requirement
            )
        )
    ):
        visible_output = canonical_output
        storage_output = canonical_output

    return thoughts, visible_output, storage_output, tool_calls


def extract_tool_info(tool: Any) -> tuple[str, str, dict[str, Any] | None]:
    """Extract (name, description, parameters) from any tool representation."""
    if hasattr(tool, "function") and tool.function is not None:
        fn = tool.function
        if isinstance(fn, dict):
            name = fn.get("name", "")
            description = fn.get("description") or "No description provided."
            parameters = fn.get("parameters")
        else:
            name = getattr(fn, "name", "")
            description = getattr(fn, "description", None) or "No description provided."
            parameters = getattr(fn, "parameters", None)
        return name, description, parameters

    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            fn = tool["function"]
            return (
                fn.get("name", ""),
                fn.get("description") or "No description provided.",
                fn.get("parameters"),
            )
        return (
            tool.get("name", ""),
            tool.get("description") or "No description provided.",
            tool.get("parameters"),
        )

    name = getattr(tool, "name", "")
    description = getattr(tool, "description", None) or "No description provided."
    parameters = getattr(tool, "parameters", None)
    return name, description, parameters


def extract_named_tool_choice(tool_choice: Any) -> str | None:
    """Extract target function name from any named tool choice representation."""
    if isinstance(tool_choice, ChatCompletionNamedToolChoice):
        return tool_choice.function.name
    if isinstance(tool_choice, ToolChoiceFunction):
        return tool_choice.name
    if isinstance(tool_choice, dict):
        if "function" in tool_choice and isinstance(tool_choice["function"], dict):
            return tool_choice["function"].get("name")
        return tool_choice.get("name")
    return None


def build_tool_prompt(
    tools: Sequence[Any],
    tool_choice: (
        Literal["none", "auto", "required"]
        | ChatCompletionNamedToolChoice
        | ToolChoiceFunction
        | ToolChoiceTypes
        | None
    ),
) -> str:
    """Generate a system prompt describing available tools and the PascalCase protocol."""
    if not tools:
        return ""

    lines: list[str] = [
        "SYSTEM INTERFACE: You have access to the following technical tools. You MUST invoke them when necessary to fulfill the request, strictly adhering to the provided JSON schemas."
    ]

    for tool in tools:
        name, description, parameters = extract_tool_info(tool)
        if not name:
            continue
        lines.append(f"Tool `{name}`: {description}")
        if parameters:
            schema_text = orjson.dumps(parameters, option=orjson.OPT_SORT_KEYS).decode("utf-8")
            lines.extend(("Arguments JSON schema:", schema_text))
        else:
            lines.append("Arguments JSON schema: {}")

    if tool_choice == "none":
        lines.append(
            "For this request you must not call any tool. Provide the best possible natural language answer."
        )
    elif tool_choice == "required":
        lines.append(
            "You must call at least one tool before responding to the user. Do not provide a final user-facing answer until a tool call has been issued."
        )
    elif (target_name := extract_named_tool_choice(tool_choice)) is not None:
        lines.append(
            f"You are required to call the tool named `{target_name}`. Do not call any other tool."
        )

    lines.append(TOOL_WRAP_HINT)

    return "\n".join(lines)


def build_image_generation_instruction(
    tools: list[ImageGeneration] | None,
    tool_choice: ToolChoiceFunction | None,
) -> str | None:
    """Construct explicit guidance so Gemini emits images when requested."""
    has_forced_choice = tool_choice is not None and tool_choice.type == "image_generation"
    primary = tools[0] if tools else None

    if not has_forced_choice and primary is None:
        return None

    instructions: list[str] = [
        "IMAGE GENERATION ENABLED: When an image is requested, you MUST return a real generated image directly.",
        "1. For new requests, generate new images matching the description immediately.",
        "2. For edits to existing images, apply changes and return a new generated version.",
        "3. CRITICAL: Provide ZERO text explanation, prologue, or apologies. Do not describe the creation process.",
        "4. NEVER send placeholder text or descriptions like 'Generating image...' without an actual image attachment.",
    ]

    if has_forced_choice:
        instructions.append(
            "Image generation was explicitly requested. You MUST return at least one generated image. Any response without an image will be treated as a failure."
        )

    return "\n\n".join(instructions)


def append_tool_hint_to_last_user_message(messages: list[AppMessage]) -> None:
    """Ensure the last user message carries the tool wrap hint."""
    for msg in reversed(messages):
        if msg.role != "user" or msg.content is None:
            continue

        if isinstance(msg.content, str):
            if TOOL_HINT_STRIPPED not in msg.content:
                msg.content = f"{msg.content}\n{TOOL_WRAP_HINT}"
            return

        if isinstance(msg.content, list):
            for part in reversed(msg.content):
                if getattr(part, "type", None) != "text":
                    continue
                text_value = getattr(part, "text", "") or ""
                if TOOL_HINT_STRIPPED in text_value:
                    return
                part.text = f"{text_value}\n{TOOL_WRAP_HINT}"
                return

            messages_text = TOOL_WRAP_HINT.strip()
            msg.content.append(AppContentItem(type="text", text=messages_text))
            return
