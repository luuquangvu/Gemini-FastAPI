import asyncio
import base64
import contextlib
import hashlib
import io
import reprlib
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import ModelInvalidError
from gemini_webapi.types.image import GeneratedImage, Image
from gemini_webapi.types.video import GeneratedMedia, GeneratedVideo
from loguru import logger

from app.models import (
    AppContentItem,
    AppMessage,
    AppToolCall,
    AppToolCallFunction,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    ConversationInStore,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    ImageGeneration,
    ImageGenerationCall,
    ModelData,
    ModelListResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseFormatTextJSONSchemaConfig,
    ResponseFunctionToolCall,
    ResponseInputMessage,
    ResponseOutputContent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
    ResponseUsage,
    StructuredOutputRequirement,
    SummaryTextContent,
    ToolChoiceFunction,
    ToolChoiceTypes,
    VideoGeneration,
)
from app.server.middleware import (
    get_media_store_dir,
    get_media_token,
    get_temp_dir,
    verify_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from app.utils import g_config
from app.utils.config import ChatMode, get_reloaded_models, reload_models_if_changed
from app.utils.helper import (
    STREAM_MASTER_RE,
    STREAM_TAIL_RE,
    STRUCTURED_JSON_WRAP_HINT,
    append_tool_hint_to_last_user_message,
    build_image_generation_instruction,
    build_tool_prompt,
    build_video_generation_instruction,
    calculate_usage,
    convert_to_app_messages,
    detect_image_extension,
    dump_model,
    extract_image_dimensions,
    normalize_app_message_role,
    normalize_llm_text,
    process_llm_output,
    serialize_tool_choice_for_response,
    serialize_tools_for_response,
    strip_system_hints,
)

MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)
# Google's temporary chat mode accepts a smaller payload than a normal chat, so tighten
# the guardrail further on top of the standard 10% safety margin.
TEMPORARY_MAX_CHARS_PER_REQUEST = int(MAX_CHARS_PER_REQUEST * 0.9)

router = APIRouter()
_AVAILABLE_MODELS_CACHE: list[ModelData] | None = None
_AVAILABLE_MODELS_FINGERPRINT: tuple[str, ...] | None = None
_AVAILABLE_MODELS_CACHE_LOCK = asyncio.Lock()


type ProcessedImageData = tuple[str, int | None, int | None, str, str]
type ProcessedMediaData = dict[str, tuple[str, str]]
type ProcessedImageResult = tuple[Literal["image"], Image, ProcessedImageData]
type ProcessedMediaResult = tuple[
    Literal["media"], GeneratedVideo | GeneratedMedia, ProcessedMediaData
]


# --- Helper Functions ---


async def _image_to_base64(
    image: Image, temp_dir: Path
) -> tuple[str, int | None, int | None, str, str]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions, filename, and hash."""
    if isinstance(image, GeneratedImage):
        try:
            saved_path = await image.save(path=str(temp_dir), full_size=True)
        except Exception as e:
            logger.warning(
                f"Failed to download full-size GeneratedImage, retrying with default size: {e}"
            )
            saved_path = await image.save(path=str(temp_dir), full_size=False)
    else:
        saved_path = await image.save(path=str(temp_dir))

    if not saved_path:
        raise ValueError("Failed to save generated image")

    original_path = Path(saved_path)
    data = original_path.read_bytes()
    suffix = original_path.suffix

    if not suffix:
        detected_ext = detect_image_extension(data)
        suffix = detected_ext or (".png" if isinstance(image, GeneratedImage) else ".jpg")

    random_name = f"img_{uuid.uuid4().hex}{suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    width, height = extract_image_dimensions(data)
    filename = random_name
    file_hash = hashlib.sha256(data).hexdigest()
    return base64.b64encode(data).decode("ascii"), width, height, filename, file_hash


async def _media_to_local_file(
    media: GeneratedVideo | GeneratedMedia, temp_dir: Path
) -> dict[str, tuple[str, str]]:
    """Persist media and return dict mapping type to (filename, hash)"""
    try:
        saved_paths = await media.save(path=str(temp_dir))
        if not saved_paths:
            logger.warning("No files saved from media object.")
            return {}
    except Exception as e:
        logger.error(f"Failed to save media: {e}")
        return {}

    default_extensions = {
        "video": ".mp4",
        "audio": ".mp3",
        "video_thumbnail": ".jpg",
        "audio_thumbnail": ".jpg",
    }

    results = {}
    path_map = {}

    for mtype, spath in saved_paths.items():
        if not spath:
            continue
        try:
            original_path = Path(spath)
            if not original_path.exists():
                if spath in path_map:
                    results[mtype] = path_map[spath]
                continue

            if spath in path_map:
                results[mtype] = path_map[spath]
                continue

            data = original_path.read_bytes()
            suffix = original_path.suffix or (
                default_extensions.get(mtype) or (".mp4" if "video" in mtype else ".mp3")
            )

            random_name = f"media_{uuid.uuid4().hex}{suffix}"
            new_path = temp_dir / random_name
            original_path.rename(new_path)

            fhash = hashlib.sha256(data).hexdigest()
            results[mtype] = (random_name, fhash)
            path_map[spath] = (random_name, fhash)
        except Exception as e:
            logger.warning(f"Error processing {mtype} at {spath}: {e}")

    return results


def _create_responses_standard_payload(
    response_id: str,
    created_time: int,
    model_name: str,
    detected_tool_calls: list[AppToolCall] | None,
    image_call_items: list[ImageGenerationCall],
    response_contents: list[ResponseOutputContent],
    usage: ResponseUsage,
    request: ResponseCreateRequest,
    full_thoughts: str | None = None,
    message_id: str | None = None,
    reason_id: str | None = None,
) -> ResponseCreateResponse:
    """Unified factory for building ResponseCreateResponse objects."""
    message_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    reason_id = reason_id or f"rs_{uuid.uuid4().hex[:24]}"
    now_ts = int(datetime.now(tz=UTC).timestamp())

    output_items: list[Any] = []
    if full_thoughts:
        output_items.append(
            ResponseReasoningItem(
                id=reason_id,
                type="reasoning",
                status="completed",
                summary=[SummaryTextContent(type="summary_text", text=full_thoughts)],
            )
        )

    if response_contents or not (detected_tool_calls or image_call_items):
        output_items.append(
            ResponseOutputMessage(
                id=message_id,
                type="message",
                status="completed",
                role="assistant",
                content=response_contents,
            )
        )

    if detected_tool_calls:
        output_items.extend(
            [
                ResponseFunctionToolCall(
                    id=call.id,
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                    status="completed",
                )
                for call in detected_tool_calls
            ]
        )

    output_items.extend(image_call_items)

    text_config = ResponseTextConfig()
    if request.response_format and request.response_format.get("type") == "json_schema":
        text_config.format = ResponseFormatTextJSONSchemaConfig()

    return ResponseCreateResponse(
        id=response_id,
        object="response",
        created_at=created_time,
        completed_at=now_ts,
        model=model_name,
        output=output_items,
        status="completed",
        usage=usage,
        metadata=request.metadata or {},
        tools=serialize_tools_for_response(request.tools),
        tool_choice=serialize_tool_choice_for_response(request.tool_choice),
        text=text_config,
    )


def _create_chat_completion_standard_payload(
    completion_id: str,
    created_time: int,
    model_name: str,
    visible_output: str | None,
    tool_calls: list[AppToolCall] | None,
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"],
    usage: dict,
    reasoning_content: str | None = None,
) -> ChatCompletionResponse:
    """Unified factory for building Chat Completion response objects."""
    tc_converted = None
    if tool_calls:
        tc_converted = [
            ChatCompletionMessageToolCall(
                id=tc.id,
                type="function",
                function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments),
            )
            for tc in tool_calls
        ]

    message = ChatCompletionMessage(
        role="assistant",
        content=visible_output or None,
        tool_calls=tc_converted,
        reasoning_content=reasoning_content or None,
    )

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(**usage),
    )


def _persist_conversation(
    db: LMDBConversationStore,
    model_name: str,
    client: GeminiClientWrapper,
    metadata: list[str | None],
    messages: list[AppMessage],
    storage_output: str | None,
    tool_calls: list[AppToolCall] | None,
) -> str | None:
    """Unified logic to save conversation history to LMDB."""
    if _use_temporary_chat_mode():
        # This turn's conversation is now the last chat this client opened; any window it
        # replaced has been closed by Google and must not be replayed again. Recorded before
        # the store so a persistence failure cannot leave a closed window looking reusable.
        client.latest_chat_cid = _cid_of(metadata)

    try:
        current_assistant_message = AppMessage(
            role="assistant",
            content=storage_output or None,
            tool_calls=tool_calls or None,
            reasoning_content=None,
        )
        full_history = [*messages, current_assistant_message]

        # A temporary chat stays continuable while its window is the live one, so its metadata
        # is worth keeping and reusing just like a normal chat's.
        db.store(
            client_id=client.id,
            model=model_name,
            messages=full_history,
            metadata=metadata,
        )
        logger.debug("Conversation saved to LMDB.")
        return "success"
    except Exception as e:
        logger.warning(f"Failed to save {len(messages) + 1} messages to LMDB: {e}")
        return None


def _build_structured_requirement(
    response_format: dict[str, Any] | None,
) -> StructuredOutputRequirement | None:
    """Translate OpenAI-style response_format into helper-managed fenced JSON instructions."""
    if not response_format or not isinstance(response_format, dict):
        return None

    if response_format.get("type") != "json_schema":
        logger.warning(
            f"Unsupported response_format type requested: {reprlib.repr(response_format)}"
        )
        return None

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        logger.warning(
            f"Invalid json_schema payload in response_format: {reprlib.repr(response_format)}"
        )
        return None

    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        logger.warning(
            f"Missing `schema` object in response_format payload: {reprlib.repr(response_format)}"
        )
        return None

    schema_name = json_schema.get("name") or "response"
    strict = json_schema.get("strict", True)

    pretty_schema = orjson.dumps(schema, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    instruction_parts = [
        STRUCTURED_JSON_WRAP_HINT,
        f'Schema name: "{schema_name}"',
        "JSON Schema:",
        pretty_schema,
    ]
    if strict:
        instruction_parts.insert(
            1,
            "Strict schema adherence is required: the JSON must conform exactly to the schema.",
        )

    instruction = "\n\n".join(instruction_parts)
    return StructuredOutputRequirement(
        schema_name=schema_name,
        schema=schema,
        instruction=instruction,
        raw_format=response_format,
    )


def _extract_last_user_text(messages: list[ChatCompletionMessage]) -> str:
    """Return the text of the last user message (string or text parts)."""
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.text or ""
                for part in content
                if getattr(part, "type", None) == "text" and part.text
            )
    return ""


def _prepare_messages_for_model(
    source_messages: list[AppMessage],
    tools: Sequence[Any] | None,
    tool_choice: Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoice
    | ToolChoiceFunction
    | ToolChoiceTypes
    | None,
    extra_instructions: list[str] | None = None,
    inject_system_defaults: bool = True,
) -> list[AppMessage]:
    """Return a copy of messages enriched with tool instructions when needed."""
    prepared = [msg.model_copy(deep=True) for msg in source_messages]

    # Resolve tool names for 'tool' messages by looking back at previous assistant tool calls
    tool_id_to_name = {}
    for msg in prepared:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_id_to_name[tc.id] = tc.function.name

    for msg in prepared:
        if msg.role == "tool" and not msg.name and msg.tool_call_id:
            msg.name = tool_id_to_name.get(msg.tool_call_id)

    instructions: list[str] = []
    tool_prompt_injected = False
    if inject_system_defaults and tools and (tool_prompt := build_tool_prompt(tools, tool_choice)):
        instructions.append(tool_prompt)
        tool_prompt_injected = True

    if extra_instructions:
        instructions.extend(instr for instr in extra_instructions if instr)
        logger.debug(
            f"Applied {len(extra_instructions)} extra instructions for tool/structured output."
        )

    if not instructions:
        if tools and tool_choice != "none" and not tool_prompt_injected:
            append_tool_hint_to_last_user_message(prepared)
        return prepared

    combined_instructions = "\n\n".join(instructions)
    if prepared and prepared[0].role == "system" and isinstance(prepared[0].content, str):
        existing = prepared[0].content or ""
        if combined_instructions not in existing:
            separator = "\n\n" if existing else ""
            prepared[0].content = f"{existing}{separator}{combined_instructions}"
    else:
        prepared.insert(0, AppMessage(role="system", content=combined_instructions))

    if tools and tool_choice != "none" and not tool_prompt_injected:
        append_tool_hint_to_last_user_message(prepared)

    return prepared


def _convert_responses_to_app_messages(
    items: Any,
) -> list[AppMessage]:
    """Convert Responses API input items into internal AppMessage objects, skipping incomplete tool calls."""
    messages: list[AppMessage] = []

    if isinstance(items, str):
        messages.append(AppMessage(role="user", content=items))
        logger.debug("Normalized Responses input: single string message.")
        return messages

    for item in items:
        if isinstance(item, (ResponseInputMessage, ResponseOutputMessage)):
            raw_role = getattr(item, "role", "user")
            role = normalize_app_message_role(raw_role)

            content = item.content
            if isinstance(content, str):
                messages.append(AppMessage(role=role, content=content))
            else:
                converted: list[AppContentItem] = []
                reasoning_parts: list[str] = []
                for part in content:
                    if part.type in ("input_text", "output_text"):
                        if text_value := getattr(part, "text", "") or "":
                            converted.append(AppContentItem(type="text", text=text_value))
                    elif part.type == "reasoning_text":
                        if text_value := getattr(part, "text", "") or "":
                            reasoning_parts.append(text_value)
                    elif part.type == "input_image":
                        if image_url := getattr(part, "image_url", None):
                            converted.append(AppContentItem(type="image_url", url=image_url))
                    elif part.type == "input_file":
                        file_url = getattr(part, "file_url", None)
                        file_data = getattr(part, "file_data", None)
                        if file_url or file_data:
                            converted.append(
                                AppContentItem(
                                    type="file",
                                    url=file_url,
                                    file_data=file_data,
                                    filename=getattr(part, "filename", None),
                                )
                            )
                reasoning_val = "\n\n".join(reasoning_parts) if reasoning_parts else None
                messages.append(
                    AppMessage(
                        role=role,
                        content=converted or None,
                        reasoning_content=reasoning_val,
                    )
                )

        elif isinstance(item, ResponseFunctionToolCall):
            call_id = item.call_id or item.id
            if not call_id or not item.name or item.arguments is None:
                logger.warning(
                    f"Skipping incomplete function_call input item: {reprlib.repr(item.model_dump(mode='json'))}"
                )
                continue
            messages.append(
                AppMessage(
                    role="assistant",
                    tool_calls=[
                        AppToolCall(
                            id=call_id,
                            type="function",
                            function=AppToolCallFunction(name=item.name, arguments=item.arguments),
                        )
                    ],
                )
            )
        elif isinstance(item, FunctionCallOutput):
            output_content = str(item.output) if isinstance(item.output, list) else item.output
            messages.append(
                AppMessage(
                    role="tool",
                    tool_call_id=item.call_id,
                    content=output_content,
                )
            )
        elif isinstance(item, ResponseReasoningItem):
            reasoning_val = None
            if item.content:
                reasoning_val = "\n\n".join(x.text for x in item.content if x.text)
            messages.append(
                AppMessage(
                    role="assistant",
                    reasoning_content=reasoning_val,
                )
            )
        elif isinstance(item, ImageGenerationCall):
            messages.append(
                AppMessage(
                    role="assistant",
                    content=item.result or None,
                )
            )

        else:
            if hasattr(item, "role"):
                raw_role = getattr(item, "role", "user")
                role = normalize_app_message_role(raw_role)
                messages.append(
                    AppMessage(
                        role=role,
                        content=str(getattr(item, "content", "")),
                    )
                )

    compacted_messages: list[AppMessage] = []
    for msg in messages:
        if not compacted_messages:
            compacted_messages.append(msg)
            continue

        last_msg = compacted_messages[-1]
        if last_msg.role == "assistant" and msg.role == "assistant":
            reasoning_parts = []
            if last_msg.reasoning_content:
                reasoning_parts.append(last_msg.reasoning_content)
            if msg.reasoning_content:
                reasoning_parts.append(msg.reasoning_content)

            merged_content = []
            if isinstance(last_msg.content, str):
                merged_content.append(AppContentItem(type="text", text=last_msg.content))
            elif isinstance(last_msg.content, list):
                merged_content.extend(last_msg.content)

            if isinstance(msg.content, str):
                merged_content.append(AppContentItem(type="text", text=msg.content))
            elif isinstance(msg.content, list):
                merged_content.extend(msg.content)

            merged_tools = []
            if last_msg.tool_calls:
                merged_tools.extend(last_msg.tool_calls)
            if msg.tool_calls:
                merged_tools.extend(msg.tool_calls)

            last_msg.reasoning_content = "\n\n".join(reasoning_parts) if reasoning_parts else None
            last_msg.content = merged_content or None
            last_msg.tool_calls = merged_tools or None
        else:
            compacted_messages.append(msg)

    logger.debug(f"Normalized Responses input: {len(compacted_messages)} message items.")
    return compacted_messages


def _convert_instructions_to_app_messages(
    instructions: str | list[ResponseInputMessage] | None,
) -> list[AppMessage]:
    """Normalize instructions payload into AppMessage objects."""
    if not instructions:
        return []

    if isinstance(instructions, str):
        return [AppMessage(role="system", content=instructions)]

    instruction_messages: list[AppMessage] = []
    for instruction in instructions:
        if instruction.type and instruction.type != "message":
            continue

        role = normalize_app_message_role(instruction.role)

        content = instruction.content
        if isinstance(content, str):
            instruction_messages.append(AppMessage(role=role, content=content))
        else:
            converted: list[AppContentItem] = []
            for part in content:
                if part.type in ("input_text", "output_text"):
                    if text_value := getattr(part, "text", "") or "":
                        converted.append(AppContentItem(type="text", text=text_value))
                elif part.type == "input_image":
                    if image_url := getattr(part, "image_url", None):
                        converted.append(AppContentItem(type="image_url", url=image_url))
                elif part.type == "input_file":
                    file_url = getattr(part, "file_url", None)
                    file_data = getattr(part, "file_data", None)
                    if file_url or file_data:
                        converted.append(
                            AppContentItem(
                                type="file",
                                url=file_url,
                                file_data=file_data,
                                filename=getattr(part, "filename", None),
                            )
                        )
            instruction_messages.append(AppMessage(role=role, content=converted or None))

    return instruction_messages


def _get_model_by_name(name: str) -> Model:
    """Retrieve a Model instance by name.

    Resolution order: custom model name → dependency Model enum (with its built-in
    normalized match) → configured `default_model` fallback → ValueError (400).
    """
    reload_models_if_changed()
    strategy = g_config.gemini.model_strategy
    custom_models = {m.model_name: m for m in get_reloaded_models() if m.model_name}

    if name in custom_models:
        return Model.from_dict(custom_models[name].model_dump())

    if strategy == "overwrite":
        raise ValueError(f"Model '{name}' not found in custom models (strategy='overwrite').")

    try:
        return Model.from_name(name)
    except ValueError:
        default_model = g_config.gemini.default_model
        if not default_model or default_model.strip().lower() == name.strip().lower():
            raise
        logger.warning(
            f"Model '{name}' not found; falling back to default_model '{default_model}'."
        )
        try:
            return _get_model_by_name(default_model)
        except ValueError:
            raise ValueError(
                f"Unknown model name: {name} (default_model '{default_model}' not found either)."
            ) from None


async def _build_available_models(pool: GeminiClientPool) -> list[ModelData]:
    """Build the available model list from configured models and currently running clients."""
    reload_models_if_changed()
    now = int(datetime.now(tz=UTC).timestamp())
    strategy = g_config.gemini.model_strategy
    models_data = []
    seen_model_ids = set()

    for model in get_reloaded_models():
        if model.model_name and model.model_name not in seen_model_ids:
            models_data.append(
                ModelData(
                    id=model.model_name,
                    created=now,
                    owned_by="custom",
                )
            )
            seen_model_ids.add(model.model_name)

    if strategy == "append":
        for client in pool.clients:
            if not client.running():
                continue

            if client_models := client.list_models():
                for model in client_models:
                    model_id = model.model_name or model.model_id
                    if model_id and model_id not in seen_model_ids:
                        models_data.append(
                            ModelData(
                                id=model_id,
                                created=now,
                                owned_by="google",
                            )
                        )
                        seen_model_ids.add(model_id)

    return models_data


async def refresh_available_models_cache(pool: GeminiClientPool) -> list[ModelData]:
    """Refresh and return the cached model list while clients are available."""
    global _AVAILABLE_MODELS_CACHE, _AVAILABLE_MODELS_FINGERPRINT

    async with _AVAILABLE_MODELS_CACHE_LOCK:
        models = await _build_available_models(pool)
        _AVAILABLE_MODELS_CACHE = models
        _AVAILABLE_MODELS_FINGERPRINT = tuple(
            m.model_name for m in get_reloaded_models() if m.model_name
        )
        logger.info(f"Cached {len(models)} available model(s).")
        return list(models)


async def _get_available_models(pool: GeminiClientPool) -> list[ModelData]:
    """Return cached available models, refreshing when the custom model list changed."""
    reload_models_if_changed()
    if _AVAILABLE_MODELS_CACHE is not None:
        current_fingerprint = tuple(m.model_name for m in get_reloaded_models() if m.model_name)
        if current_fingerprint == _AVAILABLE_MODELS_FINGERPRINT:
            return list(_AVAILABLE_MODELS_CACHE)
        logger.info("Custom model list changed; refreshing available-models cache.")

    return await refresh_available_models_cache(pool)


def _cid_of(metadata: list[str | None] | None) -> str | None:
    """Chat id from a metadata list, which stores it at index 0."""
    return metadata[0] if metadata else None


def _is_live_temporary_chat(conv: ConversationInStore, client: GeminiClientWrapper) -> bool:
    """Whether a stored chat can still be the open temporary window on this client.

    Google closes the previous temporary conversation as soon as another one is created, so at
    most one temporary window per client is alive at a time, and it can only be the last chat
    the client opened. Replaying an older one makes Google start a fresh chat and answer
    without the earlier context: a silent loss that raises no error, which is why this is
    checked up front rather than detected after the fact.

    `latest_chat_cid` is only that - the last cid this client saw. Its kind is not verified, so
    a match is a necessary condition rather than proof the window is temporary or still open.
    It lives in memory and is cleared on every client (re)initialization, so an auto-close,
    restart or redeploy invalidates every stored window: once the client that opened a window
    is gone, there is nothing left to vouch for it.
    """
    latest = client.latest_chat_cid
    if not latest:
        logger.debug(f"Client {client.id} has no chat on record; starting a fresh conversation.")
        return False

    cid = _cid_of(conv.metadata)
    if cid and cid == latest:
        return True

    logger.debug(
        f"Stored chat {cid!r} is not the latest ({latest!r}) on client {client.id}; a newer "
        "conversation has closed it, so replaying the full history in a fresh conversation."
    )
    return False


async def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[AppMessage],
    temporary: bool = False,
) -> tuple[
    ChatSession | None, GeminiClientWrapper | None, list[AppMessage], ConversationInStore | None
]:
    """Find an existing chat session matching the longest suitable history prefix."""
    if len(messages) < 2:
        return None, None, messages, None

    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]
        if search_history[-1].role in {"assistant", "system", "tool"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = await pool.acquire(conv.client_id)
                    # Checked after acquiring: acquire may restart a closed client, which clears
                    # the tracked cid and is exactly what invalidates a temporary window.
                    if temporary and not _is_live_temporary_chat(conv, client):
                        # Every prefix of one conversation carries the same cid, so if the
                        # longest match is not the live window, no shorter one will be either.
                        break
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    logger.debug(
                        f"Match found at prefix length {search_end}/{len(messages)}. Client: {conv.client_id}"
                    )
                    return session, client, remain, conv
            except Exception as e:
                logger.warning(
                    f"Error checking LMDB for reusable session at length {search_end}: {e}"
                )
                break
        search_end -= 1

    logger.debug(f"No reusable session found for {len(messages)} messages.")
    return None, None, messages, None


def _use_temporary_chat_mode() -> bool:
    """Whether requests should be sent through Google's temporary chat mode."""
    return g_config.gemini.chat_mode == ChatMode.TEMPORARY


def _effective_max_chars_per_request(temporary: bool) -> int:
    """Return the payload guardrail for the active chat mode."""
    return TEMPORARY_MAX_CHARS_PER_REQUEST if temporary else MAX_CHARS_PER_REQUEST


async def _send_with_split(
    session: ChatSession,
    text: str,
    files: list[Any] | None = None,
    stream: bool = False,
    temporary: bool = False,
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send text to Gemini with configured generation options, using an attachment if too long."""
    limit = _effective_max_chars_per_request(temporary)
    if len(text) <= limit:
        try:
            if stream:
                return session.send_message_stream(
                    text,
                    files=files,
                    temporary=temporary,
                    extended_thinking=g_config.gemini.extended_thinking,
                )
            return await session.send_message(
                text,
                files=files,
                temporary=temporary,
                extended_thinking=g_config.gemini.extended_thinking,
            )
        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}")
            raise

    logger.info(
        f"Message length ({len(text)}) exceeds limit ({limit}). Converting text to file attachment."
    )
    file_obj = io.BytesIO(text.encode("utf-8"))
    file_obj.name = "message.txt"
    try:
        final_files: list[Any] = list(files) if files else []
        final_files.insert(0, file_obj)
        instruction = (
            "The user's input exceeds the character limit and is provided in the attached file `message.txt`.\n\n"
            "**System Instruction:**\n"
            "1. Read the content of `message.txt`.\n"
            "2. Treat that content as the **primary** user prompt for this turn.\n"
            "3. Execute the instructions or answer the questions found *inside* that file immediately.\n"
        )
        if stream:
            return session.send_message_stream(
                instruction,
                files=final_files,
                temporary=temporary,
                extended_thinking=g_config.gemini.extended_thinking,
            )
        return await session.send_message(
            instruction,
            files=final_files,
            temporary=temporary,
            extended_thinking=g_config.gemini.extended_thinking,
        )
    except Exception as e:
        logger.error(f"Error sending large text as file to Gemini: {e}")
        raise


async def _restream(
    first: ModelOutput, rest: AsyncGenerator[ModelOutput]
) -> AsyncGenerator[ModelOutput]:
    """Re-emit an already-consumed first chunk, then delegate to the remainder."""
    yield first
    async for chunk in rest:
        yield chunk


STREAM_HEARTBEAT_INTERVAL = 5.0
INPUT_PREPROCESS_TIMEOUT_SECONDS = min(float(g_config.gemini.timeout), 60.0)


def _trace_id_from_headers(headers: Any) -> str:
    """Trace id for per-request tracing: x-obp-request-id -> x-request-id -> \"-\"."""
    for name in ("x-obp-request-id", "x-request-id"):
        if value := headers.get(name):
            return value
    return "-"


async def _process_conversation_with_timeout(msgs: list[AppMessage], tmp_dir: Path) -> tuple:
    """Run the input preprocess pipeline under a hard time bound (<=60s)."""
    return await asyncio.wait_for(
        GeminiClientWrapper.process_conversation(msgs, tmp_dir),
        timeout=INPUT_PREPROCESS_TIMEOUT_SECONDS,
    )


async def _stream_with_idle_timeout(
    generator: AsyncGenerator[ModelOutput],
    timeout_seconds: float,
) -> AsyncGenerator[ModelOutput | None]:
    """Iterate a chunk stream with a per-chunk deadline and idle heartbeats.

    Chunks are yielded unchanged. When no chunk arrives within
    STREAM_HEARTBEAT_INTERVAL seconds, yields None (the heartbeat sentinel -
    callers translate it to an SSE comment, never a data event). Raises
    asyncio.TimeoutError if a single chunk is stalled past timeout_seconds
    (the deadline resets per chunk), handing the failure to the caller's
    recovery path.
    """
    loop = asyncio.get_running_loop()
    next_task: asyncio.Task | None = None
    try:
        while True:
            deadline = loop.time() + timeout_seconds
            remaining = timeout_seconds
            while remaining > 0:
                if next_task is None:
                    next_task = asyncio.create_task(anext(generator))
                done, _ = await asyncio.wait(
                    {next_task},
                    timeout=min(STREAM_HEARTBEAT_INTERVAL, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    try:
                        chunk = next_task.result()
                    except StopAsyncIteration:
                        return
                    next_task = None
                    yield chunk
                    break
                remaining = deadline - loop.time()
                yield None
            else:
                raise TimeoutError(
                    f"stream chunk stalled past {timeout_seconds:.0f}s (per-chunk timeout)"
                )
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
        if next_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await next_task
        with contextlib.suppress(Exception):
            await generator.aclose()


async def _send_and_await_first_chunk(
    session: ChatSession,
    text: str,
    *,
    files: list[Any],
    stream: bool,
    temporary: bool,
) -> AsyncGenerator[ModelOutput] | ModelOutput:
    """Send to Gemini, pulling the first streamed chunk so start-of-stream errors surface here.

    `send_message_stream` is an async generator function: calling it runs none of its body, so
    without this the request would not reach Google until the caller iterates - by which point
    the HTTP response has already been committed and a failure can no longer be recovered.
    """
    output = await _send_with_split(session, text, files=files, stream=stream, temporary=temporary)
    if not stream:
        return output

    generator = cast(AsyncGenerator[ModelOutput], output)
    try:
        first = await anext(generator)
    except StopAsyncIteration:
        return generator  # already exhausted, so iterating it again simply yields nothing
    except BaseException:
        await generator.aclose()
        raise
    return _restream(first, generator)


async def _send_with_internal_fallback(
    *,
    pool: GeminiClientPool,
    db: LMDBConversationStore,
    model: Model,
    session: ChatSession,
    client: GeminiClientWrapper,
    current_input: str,
    files: list[Any],
    full_prepared_messages: list[AppMessage],
    stored_conversation: ConversationInStore | None,
    tmp_dir: Path,
    stream: bool,
    temporary: bool,
) -> tuple[AsyncGenerator[ModelOutput] | ModelOutput, ChatSession, GeminiClientWrapper]:
    """Send the request, replaying the full history in a fresh chat if reused metadata is dead.

    Streaming is recovered as well as non-streaming: the first chunk is pulled here, which is
    where Google reports a rejected chat, so the retry happens before any response is committed
    to the client. The cost is that response headers wait for Google's first chunk.
    """
    try:
        output = await _send_and_await_first_chunk(
            session, current_input, files=files, stream=stream, temporary=temporary
        )
        return output, session, client
    except ModelInvalidError:
        if stored_conversation is None:
            raise

        # Drop the dead metadata so the next request does not rediscover and re-fail on it.
        try:
            if db.evict(stored_conversation):
                logger.info("Evicted stale conversation metadata after Google rejected it.")
        except Exception as evict_exc:
            logger.warning(f"Failed to evict stale conversation metadata: {evict_exc}")

        logger.warning(
            "Metadata-backed chat reuse failed; retrying with internal history replay in a fresh chat."
        )
        fallback_client = await pool.acquire()
        fallback_session = fallback_client.start_chat(model=model)
        try:
            fallback_input, fallback_files = await _process_conversation_with_timeout(
                full_prepared_messages, tmp_dir
            )
        except TimeoutError:
            logger.warning(
                f"Input preprocessing timed out after {INPUT_PREPROCESS_TIMEOUT_SECONDS:.0f}s "
                "during metadata-replay fallback."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Input preprocessing timed out.",
            ) from None
        # Keep the caller's streaming mode: the endpoints reject a ModelOutput when the client
        # asked for a stream, so downgrading here would turn a recovery into a 502.
        output = await _send_and_await_first_chunk(
            fallback_session,
            fallback_input,
            files=list(fallback_files),
            stream=stream,
            temporary=temporary,
        )
        return output, fallback_session, fallback_client


class StreamingOutputFilter:
    """
    Filter to suppress technical protocol markers, tool calls, and system hints from the stream.
    Uses a stack-based state machine to handle nested fragmented markers.
    """

    def __init__(self):
        self.buffer = ""
        self.stack = ["NORMAL"]
        self.current_role = ""

    @property
    def state(self):
        return self.stack[-1]

    def _is_outputting(self) -> bool:
        """Determines if the current state allows yielding text to the stream."""
        if self.state == "POST_BLOCK":
            return False
        return self.state == "NORMAL" or (self.state == "IN_BLOCK" and self.current_role != "tool")

    def process(self, chunk: str) -> str:
        self.buffer += chunk
        output = []

        while self.buffer:
            if self.state == "IN_TAG_HEADER":
                nl_idx = self.buffer.find("\n")
                if nl_idx == -1:
                    break

                self.current_role = self.buffer[:nl_idx].strip().lower()
                self.buffer = self.buffer[nl_idx + 1 :]
                self.stack[-1] = "IN_BLOCK"
                continue
            if self.state == "POST_BLOCK":
                stripped = self.buffer.lstrip()
                if not stripped:
                    break
                self.buffer = stripped
                self.stack[-1] = "NORMAL"

            match = STREAM_MASTER_RE.search(self.buffer)
            if not match:
                if tail_match := STREAM_TAIL_RE.search(self.buffer):
                    yield_len = len(self.buffer) - len(tail_match.group(0))
                    if yield_len > 0:
                        if self._is_outputting():
                            output.append(self.buffer[:yield_len])
                        self.buffer = self.buffer[yield_len:]
                else:
                    if self._is_outputting():
                        output.append(self.buffer)
                    self.buffer = ""
                break

            start, end = match.span()
            matched_group = match.lastgroup
            pre_text = self.buffer[:start]

            if self._is_outputting():
                output.append(pre_text)

            if matched_group and matched_group.endswith("_START"):
                m_type = matched_group.split("_")[0]
                if m_type == "TAG":
                    self.stack.append("IN_TAG_HEADER")
                else:
                    self.stack.append(f"IN_{m_type}")
            elif matched_group in ("PROTOCOL_EXIT", "TAG_EXIT", "HINT_EXIT"):
                if len(self.stack) > 1:
                    self.stack.pop()
                else:
                    self.stack = ["NORMAL"]

                if self.state == "NORMAL" and matched_group in (
                    "PROTOCOL_EXIT",
                    "HINT_EXIT",
                ):
                    self.stack[-1] = "POST_BLOCK"

                if self.state in ("NORMAL", "POST_BLOCK"):
                    self.current_role = ""

            self.buffer = self.buffer[end:]

        return "".join(output)

    def flush(self) -> str:
        """Release remaining buffer content and perform final cleanup at stream end."""
        res = ""
        if self._is_outputting():
            res = self.buffer
            if tail_match := STREAM_TAIL_RE.search(res):
                res = res[: -len(tail_match.group(0))]

        self.buffer = ""
        self.stack = ["NORMAL"]
        self.current_role = ""
        return strip_system_hints(res)


# --- Media Processing Helpers ---


async def _process_image_item(image: Image) -> ProcessedImageResult | None:
    """Process an image item by converting it to base64 and returning a typed image result tuple."""
    try:
        media_store = get_media_store_dir()
        return "image", image, await _image_to_base64(image, media_store)
    except Exception as exc:
        logger.warning(f"Background image processing failed: {exc}")
        return None


async def _process_media_item(
    media_item: GeneratedVideo | GeneratedMedia,
) -> ProcessedMediaResult | None:
    """Process a media item by saving it to local files and returning a typed media result tuple."""
    try:
        media_store = get_media_store_dir()
        return "media", media_item, await _media_to_local_file(media_item, media_store)
    except Exception as exc:
        logger.warning(f"Background media processing failed: {exc}")
        return None


# --- Response Builders & Streaming ---


async def _recover_stream_output(
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    prev_rcid: str,
    recover_timeout: int,
) -> ModelOutput | None:
    """Recover the full model turn from the conversation-turns RPC after a truncated stream.

    Gemini finalizes the turn server-side even when the stream dies mid-response.
    Polls until a completed turn with a NEW rcid appears (rec_rcid != prev_rcid,
    parity with the pinned dependency's own recovery gate), or the timeout
    elapses.
    """
    cid = getattr(session, "cid", "") or ""
    if not cid or recover_timeout <= 0:
        return None
    deadline = time.monotonic() + recover_timeout
    while time.monotonic() < deadline:
        try:
            recovered = await client_wrapper.fetch_last_model_turn(cid)
        except Exception as e:
            logger.warning(f"[Recovery] Turns RPC failed for {cid}: {type(e).__name__}: {e}")
            break
        if recovered is not None:
            rec_rcid = recovered.rcid or ""
            if rec_rcid and (not prev_rcid or rec_rcid != prev_rcid):
                logger.info(f"[Recovery] Full turn recovered for {cid} (rcid {rec_rcid})")
                return recovered
            logger.debug(
                f"[Recovery] Turn not ready for {cid} (rcid {rec_rcid!r}, prev {prev_rcid!r}); waiting..."
            )
        await asyncio.sleep(10)
    logger.warning(f"[Recovery] Timed out after {recover_timeout}s for {cid}")
    return None


def _create_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    completion_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response.
    Reconciles manual delta accumulation with the model's final authoritative state.
    Emits typed image and media results as incremental markdown deltas.
    """

    async def generate_stream():
        full_text = ""
        full_thoughts = ""
        has_started = False
        all_outputs: list[ModelOutput] = []
        suppressor = StreamingOutputFilter()
        prev_rcid = getattr(session, "rcid", "") or ""
        t0 = time.monotonic()
        logged_first_chunk = False
        logged_first_text = False

        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        async def _make_async_gen(item: ModelOutput) -> AsyncGenerator[ModelOutput]:
            yield item

        def make_chunk(delta_content: dict) -> str:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, **delta_content}],
            }
            return f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        try:
            if hasattr(resp_or_stream, "__aiter__"):
                generator = cast(AsyncGenerator[ModelOutput], resp_or_stream)
            else:
                generator = _make_async_gen(cast(ModelOutput, resp_or_stream))

            generator = _stream_with_idle_timeout(generator, float(g_config.gemini.timeout))

            async for chunk in generator:
                if chunk is None:
                    yield ": ping\n\n"
                    continue
                all_outputs.append(chunk)
                if not has_started:
                    if not logged_first_chunk:
                        logger.info(
                            f"First upstream chunk after {(time.monotonic() - t0) * 1000:.0f} ms"
                        )
                        logged_first_chunk = True
                    yield make_chunk(
                        {"delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                    )
                    has_started = True

                if t_delta := chunk.thoughts_delta:
                    full_thoughts += t_delta
                    yield make_chunk(
                        {"delta": {"reasoning_content": t_delta}, "finish_reason": None}
                    )

                if text_delta := chunk.text_delta:
                    if not logged_first_text:
                        logger.info(
                            f"First text delta after {(time.monotonic() - t0) * 1000:.0f} ms"
                        )
                        logged_first_text = True
                    full_text += text_delta
                    if not structured_requirement and (
                        visible_delta := suppressor.process(text_delta)
                    ):
                        yield make_chunk(
                            {"delta": {"content": visible_delta}, "finish_reason": None}
                        )

                for img in chunk.images or []:
                    if img.url and img.url not in seen_image_urls:
                        seen_image_urls.add(img.url)
                        media_tasks.append(asyncio.create_task(_process_image_item(img)))

                m_list = (chunk.videos or []) + (chunk.media or [])
                for m in m_list:
                    p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                    if p_url and p_url not in seen_media_urls:
                        seen_media_urls.add(p_url)
                        media_tasks.append(asyncio.create_task(_process_media_item(m)))
        except Exception as e:
            logger.error(
                f"Error during streaming after {(time.monotonic() - t0) * 1000:.0f} ms: {e}"
            )
            if not structured_requirement and (remainder := suppressor.flush()):
                yield make_chunk({"delta": {"content": remainder}, "finish_reason": None})
            recovered = await _recover_stream_output(
                client_wrapper, session, prev_rcid, g_config.gemini.recovery_timeout
            )
            if recovered is None:
                yield f"data: {orjson.dumps({'error': {'message': f'Streaming error occurred: {e}', 'type': 'server_error', 'param': None, 'code': None}}).decode('utf-8')}\n\n"
                return
            logger.info(f"Stream truncated; recovered full turn text for {session.cid}.")
            rec_thoughts = recovered.thoughts or ""
            if rec_thoughts.startswith(full_thoughts) and len(rec_thoughts) > len(full_thoughts):
                t_tail = rec_thoughts[len(full_thoughts) :]
                full_thoughts = rec_thoughts
                yield make_chunk({"delta": {"reasoning_content": t_tail}, "finish_reason": None})
            rec_text = recovered.text or ""
            if rec_text.startswith(full_text) and len(rec_text) > len(full_text):
                text_tail = rec_text[len(full_text) :]
                full_text = rec_text
                if not structured_requirement and (visible_tail := suppressor.process(text_tail)):
                    yield make_chunk({"delta": {"content": visible_tail}, "finish_reason": None})

        if all_outputs:
            final_chunk = all_outputs[-1]
            if final_chunk.thoughts:
                f_thoughts = final_chunk.thoughts
                ft_len, ct_len = len(f_thoughts), len(full_thoughts)
                if ft_len > ct_len and f_thoughts.startswith(full_thoughts):
                    drift_t = f_thoughts[ct_len:]
                    full_thoughts = f_thoughts
                    yield make_chunk(
                        {"delta": {"reasoning_content": drift_t}, "finish_reason": None}
                    )

            if final_chunk.text:
                f_text = final_chunk.text
                f_len, c_len = len(f_text), len(full_text)
                if f_len > c_len and f_text.startswith(full_text):
                    drift = f_text[c_len:]
                    full_text = f_text
                    if not structured_requirement and (visible_drift := suppressor.process(drift)):
                        yield make_chunk(
                            {"delta": {"content": visible_drift}, "finish_reason": None}
                        )

        if not structured_requirement and (remaining_text := suppressor.flush()):
            yield make_chunk({"delta": {"content": remaining_text}, "finish_reason": None})

        _, visible_output, storage_output, detected_tool_calls = process_llm_output(
            normalize_llm_text(full_thoughts or ""),
            normalize_llm_text(full_text or ""),
            structured_requirement,
        )
        if structured_requirement and visible_output:
            yield make_chunk({"delta": {"content": visible_output}, "finish_reason": None})

        seen_hashes = {}
        seen_media_hashes = {}
        media_store = get_media_store_dir()

        if media_tasks:
            logger.debug(f"Waiting for {len(media_tasks)} background media tasks with heartbeat...")
            while media_tasks:
                done, pending = await asyncio.wait(
                    media_tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
                )
                media_tasks = list(pending)

                if not done:
                    yield ": ping\n\n"
                    continue

                for task in done:
                    res = task.result()
                    if not res:
                        continue

                    rtype, original_item, media_data = res
                    if rtype == "image":
                        _, _, _, fname, fhash = media_data
                        if fhash in seen_hashes:
                            (media_store / fname).unlink(missing_ok=True)
                            fname = seen_hashes[fhash]
                        else:
                            seen_hashes[fhash] = fname

                        img_url = f"{base_url}media/{fname}?token={get_media_token(fname)}"
                        title = getattr(original_item, "title", "Image")
                        md = f"![{title}]({img_url})"
                        storage_output += f"\n\n{md}"
                        yield make_chunk({"delta": {"content": f"\n\n{md}"}, "finish_reason": None})

                    elif rtype == "media":
                        m_dict = cast(ProcessedMediaData, media_data)
                        if not m_dict:
                            continue

                        m_urls = {}
                        for mtype, (random_name, fhash) in m_dict.items():
                            if fhash in seen_media_hashes:
                                existing_name = seen_media_hashes[fhash]
                                if random_name != existing_name:
                                    (media_store / random_name).unlink(missing_ok=True)
                                m_urls[mtype] = (
                                    f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                                )
                            else:
                                seen_media_hashes[fhash] = random_name
                                m_urls[mtype] = (
                                    f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                                )

                        title = getattr(original_item, "title", "Media")
                        video_url = m_urls.get("video")
                        audio_url = m_urls.get("audio")
                        current_thumb = m_urls.get("video_thumbnail") or m_urls.get(
                            "audio_thumbnail"
                        )

                        md_parts = []
                        if video_url:
                            md_parts.append(
                                f"[![{title}]({current_thumb})]({video_url})"
                                if current_thumb
                                else f"[{title}]({video_url})"
                            )
                        if audio_url:
                            md_parts.append(
                                f"[![{title} - Audio]({current_thumb})]({audio_url})"
                                if current_thumb
                                else f"[{title} - Audio]({audio_url})"
                            )

                        if md_parts:
                            md = "\n\n".join(md_parts)
                            storage_output += f"\n\n{md}"
                            yield make_chunk(
                                {"delta": {"content": f"\n\n{md}"}, "finish_reason": None}
                            )

        if detected_tool_calls:
            for idx, call in enumerate(detected_tool_calls):
                tc_dict = {
                    "index": idx,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }

                yield make_chunk(
                    {
                        "delta": {
                            "tool_calls": [tc_dict],
                        },
                        "finish_reason": None,
                    }
                )

        p_tok, c_tok, t_tok, r_tok = calculate_usage(
            messages, storage_output, detected_tool_calls, full_thoughts
        )
        usage = CompletionUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            completion_tokens_details={"reasoning_tokens": r_tok},
        )
        _persist_conversation(
            db,
            model.model_name,
            client_wrapper,
            session.metadata,
            messages,
            storage_output,
            detected_tool_calls,
        )
        yield make_chunk(
            {
                "delta": {},
                "finish_reason": "tool_calls" if detected_tool_calls else "stop",
                "usage": dump_model(usage),
            }
        )
        yield "data: [DONE]\n\n"
        logger.info(f"Stream finished after {(time.monotonic() - t0) * 1000:.0f} ms")

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_responses_real_streaming_response(
    resp_or_stream: AsyncGenerator[ModelOutput] | ModelOutput,
    response_id: str,
    created_time: int,
    model_name: str,
    messages: list[AppMessage],
    db: LMDBConversationStore,
    model: Model,
    client_wrapper: GeminiClientWrapper,
    session: ChatSession,
    request: ResponseCreateRequest,
    base_url: str,
    structured_requirement: StructuredOutputRequirement | None = None,
) -> StreamingResponse:
    """
    Create a real-time streaming response for the Responses API.
    Ensures final accumulated text and thoughts are synchronized and follow the formal event stream spec.
    Emits typed image and media results as incremental response output text events.
    """
    base_event = {
        "id": response_id,
        "object": "response",
        "created_at": created_time,
        "model": model_name,
    }

    async def generate_stream():
        seq = 0

        def make_event(etype: str, data: dict) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return f"event: {etype}\ndata: {orjson.dumps(data).decode()}\n\n"

        yield make_event(
            "response.created",
            {
                **base_event,
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_time,
                    "model": model_name,
                    "status": "in_progress",
                    "metadata": request.metadata or {},
                    "input": None,
                    "tools": serialize_tools_for_response(request.tools),
                    "tool_choice": serialize_tool_choice_for_response(request.tool_choice),
                    "output": [],
                    "usage": None,
                },
            },
        )
        yield make_event(
            "response.in_progress",
            {
                **base_event,
                "type": "response.in_progress",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_time,
                    "model": model_name,
                    "status": "in_progress",
                    "metadata": request.metadata or {},
                    "output": [],
                },
            },
        )

        full_text = ""
        full_thoughts = ""
        media_tasks = []
        seen_media_urls = set()
        seen_image_urls = set()

        all_outputs: list[ModelOutput] = []

        thought_item_id = f"rs_{uuid.uuid4().hex[:24]}"
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"

        thought_open, message_open = False, False
        next_output_index = 0
        thought_index = 0
        message_index = 0
        suppressor = StreamingOutputFilter()
        prev_rcid = getattr(session, "rcid", "") or ""
        t0 = time.monotonic()
        logged_first_chunk = False
        logged_first_text = False

        try:
            if hasattr(resp_or_stream, "__aiter__"):
                generator = cast(AsyncGenerator[ModelOutput], resp_or_stream)
            else:

                async def _make_async_gen(item: ModelOutput) -> AsyncGenerator[ModelOutput]:
                    yield item

                generator = _make_async_gen(cast(ModelOutput, resp_or_stream))

            generator = _stream_with_idle_timeout(generator, float(g_config.gemini.timeout))

            async for chunk in generator:
                if chunk is None:
                    yield ": ping\n\n"
                    continue
                all_outputs.append(chunk)
                if not logged_first_chunk:
                    logger.info(
                        f"First upstream chunk after {(time.monotonic() - t0) * 1000:.0f} ms "
                        f"(responses)"
                    )
                    logged_first_chunk = True

                if chunk.thoughts_delta:
                    if not thought_open:
                        thought_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": thought_index,
                                "item": dump_model(
                                    ResponseReasoningItem(
                                        id=thought_item_id,
                                        type="reasoning",
                                        status="in_progress",
                                        summary=[],
                                    )
                                ),
                            },
                        )

                        yield make_event(
                            "response.reasoning_summary_part.added",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_part.added",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "part": dump_model(SummaryTextContent(text="")),
                            },
                        )
                        thought_open = True

                    full_thoughts += chunk.thoughts_delta
                    yield make_event(
                        "response.reasoning_summary_text.delta",
                        {
                            **base_event,
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "delta": chunk.thoughts_delta,
                        },
                    )

                if chunk.text_delta:
                    if not logged_first_text:
                        logger.info(
                            f"First text delta after {(time.monotonic() - t0) * 1000:.0f} ms "
                            f"(responses)"
                        )
                        logged_first_text = True
                    full_text += chunk.text_delta
                    if thought_open:
                        yield make_event(
                            "response.reasoning_summary_text.done",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_text.done",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "text": full_thoughts,
                            },
                        )
                        yield make_event(
                            "response.reasoning_summary_part.done",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_part.done",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "part": dump_model(SummaryTextContent(text=full_thoughts)),
                            },
                        )
                        yield make_event(
                            "response.output_item.done",
                            {
                                **base_event,
                                "type": "response.output_item.done",
                                "output_index": thought_index,
                                "item": dump_model(
                                    ResponseReasoningItem(
                                        id=thought_item_id,
                                        type="reasoning",
                                        status="completed",
                                        summary=[SummaryTextContent(text=full_thoughts)],
                                    )
                                ),
                            },
                        )
                        thought_open = False

                    if not structured_requirement:
                        if not message_open:
                            message_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": dump_model(
                                        ResponseOutputMessage(
                                            id=message_item_id,
                                            type="message",
                                            status="in_progress",
                                            role="assistant",
                                            content=[],
                                        )
                                    ),
                                },
                            )

                            yield make_event(
                                "response.content_part.added",
                                {
                                    **base_event,
                                    "type": "response.content_part.added",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": dump_model(
                                        ResponseOutputText(type="output_text", text="")
                                    ),
                                },
                            )
                            message_open = True

                        if visible := suppressor.process(chunk.text_delta):
                            yield make_event(
                                "response.output_text.delta",
                                {
                                    **base_event,
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": visible,
                                    "logprobs": [],
                                },
                            )

                for img in chunk.images or []:
                    if img.url and img.url not in seen_image_urls:
                        seen_image_urls.add(img.url)
                        media_tasks.append(asyncio.create_task(_process_image_item(img)))

                m_list = (chunk.videos or []) + (chunk.media or [])
                for m in m_list:
                    p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
                    if p_url and p_url not in seen_media_urls:
                        seen_media_urls.add(p_url)
                        media_tasks.append(asyncio.create_task(_process_media_item(m)))

        except Exception as e:
            logger.error(
                f"Error during streaming after {(time.monotonic() - t0) * 1000:.0f} ms: {e}"
            )
            remaining = "" if structured_requirement else suppressor.flush()
            if remaining and message_open:
                yield make_event(
                    "response.output_text.delta",
                    {
                        **base_event,
                        "type": "response.output_text.delta",
                        "item_id": message_item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": remaining,
                        "logprobs": [],
                    },
                )
            recovered = await _recover_stream_output(
                client_wrapper, session, prev_rcid, g_config.gemini.recovery_timeout
            )
            if recovered is None:
                yield make_event(
                    "error",
                    {
                        **base_event,
                        "type": "error",
                        "error": {"message": f"Streaming error occurred: {e}"},
                    },
                )
                return
            logger.info(f"Stream truncated; recovered full turn text for {session.cid}.")
            rec_thoughts = recovered.thoughts or ""
            if rec_thoughts.startswith(full_thoughts) and len(rec_thoughts) > len(full_thoughts):
                t_tail = rec_thoughts[len(full_thoughts) :]
                full_thoughts = rec_thoughts
                if not thought_open:
                    thought_index = next_output_index
                    next_output_index += 1
                    yield make_event(
                        "response.output_item.added",
                        {
                            **base_event,
                            "type": "response.output_item.added",
                            "output_index": thought_index,
                            "item": dump_model(
                                ResponseReasoningItem(
                                    id=thought_item_id,
                                    type="reasoning",
                                    status="in_progress",
                                    summary=[],
                                )
                            ),
                        },
                    )
                    yield make_event(
                        "response.reasoning_summary_part.added",
                        {
                            **base_event,
                            "type": "response.reasoning_summary_part.added",
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "part": dump_model(SummaryTextContent(text="")),
                        },
                    )
                    thought_open = True
                yield make_event(
                    "response.reasoning_summary_text.delta",
                    {
                        **base_event,
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": thought_item_id,
                        "output_index": thought_index,
                        "summary_index": 0,
                        "delta": t_tail,
                    },
                )
            rec_text = recovered.text or ""
            if rec_text.startswith(full_text) and len(rec_text) > len(full_text):
                text_tail = rec_text[len(full_text) :]
                full_text = rec_text
                if not structured_requirement:
                    if not message_open:
                        message_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": message_index,
                                "item": dump_model(
                                    ResponseOutputMessage(
                                        id=message_item_id,
                                        type="message",
                                        status="in_progress",
                                        role="assistant",
                                        content=[],
                                    )
                                ),
                            },
                        )
                        yield make_event(
                            "response.content_part.added",
                            {
                                **base_event,
                                "type": "response.content_part.added",
                                "item_id": message_item_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "part": dump_model(ResponseOutputText(type="output_text", text="")),
                            },
                        )
                        message_open = True
                    if visible := suppressor.process(text_tail):
                        yield make_event(
                            "response.output_text.delta",
                            {
                                **base_event,
                                "type": "response.output_text.delta",
                                "item_id": message_item_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "delta": visible,
                                "logprobs": [],
                            },
                        )

        if all_outputs:
            last = all_outputs[-1]
            if last.thoughts:
                l_thoughts = last.thoughts
                lt_len, ct_len = len(l_thoughts), len(full_thoughts)
                if lt_len > ct_len and l_thoughts.startswith(full_thoughts):
                    drift_t = l_thoughts[ct_len:]
                    full_thoughts = l_thoughts
                    if not thought_open:
                        thought_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": thought_index,
                                "item": dump_model(
                                    ResponseReasoningItem(
                                        id=thought_item_id,
                                        type="reasoning",
                                        status="in_progress",
                                        summary=[],
                                    )
                                ),
                            },
                        )
                        yield make_event(
                            "response.reasoning_summary_part.added",
                            {
                                **base_event,
                                "type": "response.reasoning_summary_part.added",
                                "item_id": thought_item_id,
                                "output_index": thought_index,
                                "summary_index": 0,
                                "part": dump_model(SummaryTextContent(text="")),
                            },
                        )
                        thought_open = True

                    yield make_event(
                        "response.reasoning_summary_text.delta",
                        {
                            **base_event,
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": thought_item_id,
                            "output_index": thought_index,
                            "summary_index": 0,
                            "delta": drift_t,
                        },
                    )

            if last.text:
                l_text = last.text
                l_len, c_len = len(l_text), len(full_text)
                if l_len > c_len and l_text.startswith(full_text):
                    drift = l_text[c_len:]
                    full_text = l_text
                    if not structured_requirement and (visible := suppressor.process(drift)):
                        if not message_open:
                            message_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": dump_model(
                                        ResponseOutputMessage(
                                            id=message_item_id,
                                            type="message",
                                            status="in_progress",
                                            role="assistant",
                                            content=[],
                                        )
                                    ),
                                },
                            )
                            yield make_event(
                                "response.content_part.added",
                                {
                                    **base_event,
                                    "type": "response.content_part.added",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": dump_model(
                                        ResponseOutputText(type="output_text", text="")
                                    ),
                                },
                            )
                            message_open = True

                        yield make_event(
                            "response.output_text.delta",
                            {
                                **base_event,
                                "type": "response.output_text.delta",
                                "item_id": message_item_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "delta": visible,
                                "logprobs": [],
                            },
                        )

        remaining = "" if structured_requirement else suppressor.flush()
        if remaining and message_open:
            yield make_event(
                "response.output_text.delta",
                {
                    **base_event,
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "delta": remaining,
                    "logprobs": [],
                },
            )

        if thought_open:
            yield make_event(
                "response.reasoning_summary_text.done",
                {
                    **base_event,
                    "type": "response.reasoning_summary_text.done",
                    "item_id": thought_item_id,
                    "output_index": thought_index,
                    "summary_index": 0,
                    "text": full_thoughts,
                },
            )
            yield make_event(
                "response.reasoning_summary_part.done",
                {
                    **base_event,
                    "type": "response.reasoning_summary_part.done",
                    "item_id": thought_item_id,
                    "output_index": thought_index,
                    "summary_index": 0,
                    "part": dump_model(SummaryTextContent(text=full_thoughts)),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": thought_index,
                    "item": dump_model(
                        ResponseReasoningItem(
                            id=thought_item_id,
                            type="reasoning",
                            status="completed",
                            summary=[SummaryTextContent(text=full_thoughts)],
                        )
                    ),
                },
            )

        _, assistant_text, storage_output, detected_tool_calls = process_llm_output(
            normalize_llm_text(full_thoughts or ""),
            normalize_llm_text(full_text or ""),
            structured_requirement,
        )

        if structured_requirement and assistant_text and not message_open:
            message_index = next_output_index
            next_output_index += 1
            yield make_event(
                "response.output_item.added",
                {
                    **base_event,
                    "type": "response.output_item.added",
                    "output_index": message_index,
                    "item": dump_model(
                        ResponseOutputMessage(
                            id=message_item_id,
                            type="message",
                            status="in_progress",
                            role="assistant",
                            content=[],
                        )
                    ),
                },
            )
            yield make_event(
                "response.content_part.added",
                {
                    **base_event,
                    "type": "response.content_part.added",
                    "item_id": message_item_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "part": dump_model(ResponseOutputText(type="output_text", text="")),
                },
            )
            message_open = True
            yield make_event(
                "response.output_text.delta",
                {
                    **base_event,
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "delta": assistant_text,
                    "logprobs": [],
                },
            )

        image_items = []
        seen_hashes = {}
        seen_media_hashes = {}
        media_store = get_media_store_dir()

        if media_tasks:
            logger.debug(
                f"Waiting for {len(media_tasks)} background media tasks in Responses with heartbeat..."
            )
            while media_tasks:
                done, pending = await asyncio.wait(
                    media_tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED
                )
                media_tasks = list(pending)

                if not done:
                    yield ": ping\n\n"
                    continue

                for task in done:
                    res = task.result()
                    if not res:
                        continue

                    rtype, original_item, media_data = res
                    if rtype == "image":
                        b64, w, h, fname, fhash = media_data
                        if fhash in seen_hashes:
                            (media_store / fname).unlink(missing_ok=True)
                            b64, w, h, fname = seen_hashes[fhash]
                        else:
                            seen_hashes[fhash] = (b64, w, h, fname)

                        parts = fname.rsplit(".", 1)
                        img_id = parts[0]
                        fmt = parts[1] if len(parts) > 1 else "png"

                        img_item = ImageGenerationCall(
                            id=img_id,
                            result=b64,
                            output_format=fmt,
                            size=f"{w}x{h}" if w and h else None,
                        )

                        img_link = (
                            f"![{fname}]({base_url}media/{fname}?token={get_media_token(fname)})"
                        )
                        md_to_add = f"\n\n{img_link}"

                        img_index = next_output_index
                        next_output_index += 1
                        yield make_event(
                            "response.output_item.added",
                            {
                                **base_event,
                                "type": "response.output_item.added",
                                "output_index": img_index,
                                "item": dump_model(img_item),
                            },
                        )
                        yield make_event(
                            "response.output_item.done",
                            {
                                **base_event,
                                "type": "response.output_item.done",
                                "output_index": img_index,
                                "item": dump_model(img_item),
                            },
                        )

                        if not message_open:
                            message_index = next_output_index
                            next_output_index += 1
                            yield make_event(
                                "response.output_item.added",
                                {
                                    **base_event,
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": dump_model(
                                        ResponseOutputMessage(
                                            id=message_item_id,
                                            type="message",
                                            status="in_progress",
                                            role="assistant",
                                            content=[],
                                        )
                                    ),
                                },
                            )
                            yield make_event(
                                "response.content_part.added",
                                {
                                    **base_event,
                                    "type": "response.content_part.added",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": dump_model(
                                        ResponseOutputText(type="output_text", text="")
                                    ),
                                },
                            )
                            message_open = True

                        yield make_event(
                            "response.output_text.delta",
                            {
                                **base_event,
                                "type": "response.output_text.delta",
                                "item_id": message_item_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "delta": md_to_add,
                                "logprobs": [],
                            },
                        )
                        assistant_text += md_to_add
                        storage_output += md_to_add
                        image_items.append(img_item)

                    elif rtype == "media":
                        m_dict = cast(ProcessedMediaData, media_data)
                        if not m_dict:
                            continue

                        m_urls = {}
                        for mtype, (random_name, fhash) in m_dict.items():
                            if fhash in seen_media_hashes:
                                existing_name = seen_media_hashes[fhash]
                                if random_name != existing_name:
                                    (media_store / random_name).unlink(missing_ok=True)
                                m_urls[mtype] = (
                                    f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                                )
                            else:
                                seen_media_hashes[fhash] = random_name
                                m_urls[mtype] = (
                                    f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                                )

                        title = getattr(original_item, "title", "Media")
                        video_url = m_urls.get("video")
                        audio_url = m_urls.get("audio")
                        current_thumb = m_urls.get("video_thumbnail") or m_urls.get(
                            "audio_thumbnail"
                        )

                        md_parts = []
                        if video_url:
                            md_parts.append(
                                f"[![{title}]({current_thumb})]({video_url})"
                                if current_thumb
                                else f"[{title}]({video_url})"
                            )
                        if audio_url:
                            md_parts.append(
                                f"[![{title} - Audio]({current_thumb})]({audio_url})"
                                if current_thumb
                                else f"[{title} - Audio]({audio_url})"
                            )

                        if md_parts:
                            media_md = "\n\n".join(md_parts)
                            md_to_add = f"\n\n{media_md}"

                            if not message_open:
                                message_index = next_output_index
                                next_output_index += 1
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        **base_event,
                                        "type": "response.output_item.added",
                                        "output_index": message_index,
                                        "item": dump_model(
                                            ResponseOutputMessage(
                                                id=message_item_id,
                                                type="message",
                                                status="in_progress",
                                                role="assistant",
                                                content=[],
                                            )
                                        ),
                                    },
                                )
                                yield make_event(
                                    "response.content_part.added",
                                    {
                                        **base_event,
                                        "type": "response.content_part.added",
                                        "item_id": message_item_id,
                                        "output_index": message_index,
                                        "content_index": 0,
                                        "part": dump_model(
                                            ResponseOutputText(type="output_text", text="")
                                        ),
                                    },
                                )
                                message_open = True

                            yield make_event(
                                "response.output_text.delta",
                                {
                                    **base_event,
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": md_to_add,
                                    "logprobs": [],
                                },
                            )
                            assistant_text += md_to_add
                            storage_output += md_to_add

        final_response_contents: list[ResponseOutputContent] = []
        if message_open:
            if assistant_text:
                final_response_contents = [
                    ResponseOutputText(type="output_text", text=assistant_text)
                ]
            else:
                final_response_contents = [ResponseOutputText(type="output_text", text="")]

            yield make_event(
                "response.output_text.done",
                {
                    **base_event,
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_index,
                    "content_index": 0,
                },
            )
            yield make_event(
                "response.content_part.done",
                {
                    **base_event,
                    "type": "response.content_part.done",
                    "item_id": message_item_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "part": dump_model(ResponseOutputText(type="output_text", text=assistant_text)),
                },
            )

            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": message_index,
                    "item": dump_model(
                        ResponseOutputMessage(
                            id=message_item_id,
                            type="message",
                            status="completed",
                            role="assistant",
                            content=final_response_contents,
                        )
                    ),
                },
            )

        for call in detected_tool_calls:
            tc_index = next_output_index
            next_output_index += 1
            tc_item = ResponseFunctionToolCall(
                id=call.id,
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
                status="completed",
            )
            yield make_event(
                "response.output_item.added",
                {
                    **base_event,
                    "type": "response.output_item.added",
                    "output_index": tc_index,
                    "item": dump_model(tc_item),
                },
            )
            yield make_event(
                "response.output_item.done",
                {
                    **base_event,
                    "type": "response.output_item.done",
                    "output_index": tc_index,
                    "item": dump_model(tc_item),
                },
            )

        p_tok, c_tok, t_tok, r_tok = calculate_usage(
            messages, storage_output, detected_tool_calls, full_thoughts
        )
        usage = ResponseUsage(
            input_tokens=p_tok,
            output_tokens=c_tok,
            total_tokens=t_tok,
            output_tokens_details={"reasoning_tokens": r_tok},
        )
        payload = _create_responses_standard_payload(
            response_id,
            created_time,
            model_name,
            detected_tool_calls,
            image_items,
            final_response_contents,
            usage,
            request,
            full_thoughts,
            message_item_id,
            thought_item_id,
        )
        _persist_conversation(
            db,
            model.model_name,
            client_wrapper,
            session.metadata,
            messages,
            storage_output,
            detected_tool_calls,
        )

        yield make_event(
            "response.completed",
            {
                **base_event,
                "type": "response.completed",
                "response": dump_model(payload),
            },
        )

        yield "data: [DONE]\n\n"
        logger.info(f"Responses stream finished after {(time.monotonic() - t0) * 1000:.0f} ms")

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# --- Main Router Endpoints ---


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    pool = GeminiClientPool()
    models = await _get_available_models(pool)
    return ModelListResponse(data=models)


@router.get("/v1/gems")
async def list_gems(api_key: str = Depends(verify_api_key)):
    pool = GeminiClientPool()
    try:
        client = await pool.acquire()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    try:
        gems = await client.fetch_gems()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
    return {
        "object": "list",
        "data": [
            {
                "id": gem.id,
                "name": gem.name,
                "description": getattr(gem, "description", None),
            }
            for gem in gems.values()
        ],
    }


@router.post("/v1/chat/completions", response_model_exclude_none=True)
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    base_url = str(raw_request.base_url)
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not request.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages required.")

    app_messages = convert_to_app_messages(request.messages)

    trace_id = _trace_id_from_headers(raw_request.headers)
    prompt_chars = sum(len(str(m.content)) for m in app_messages if m.content)
    logger.info(
        f"Request accepted: trace={trace_id} model={request.model} "
        f"stream={bool(request.stream)} prompt_chars={prompt_chars}"
    )

    if request.deep_research:
        if request.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Streaming is not supported for deep research.",
            )
        prompt = _extract_last_user_text(request.messages)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deep research requires a non-empty user text message.",
            )
        research_timeout = request.deep_research_timeout or 600
        try:
            client = await pool.acquire()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e
        fallback_report: str | None = None
        try:
            research_plan = await client.create_deep_research_plan(prompt)
            start_output = await client.start_deep_research(research_plan)
            if (
                not research_plan.research_id
                and start_output.deep_research_plan
                and start_output.deep_research_plan.research_id
            ):
                research_plan.research_id = start_output.deep_research_plan.research_id
            if research_plan.research_id:
                result = await client.wait_for_deep_research(
                    research_plan, timeout=research_timeout
                )
                result.start_output = start_output
            else:
                logger.warning(
                    "Deep research: plan.research_id is missing on this account; "
                    f"falling back to turns-RPC report crawl (cid={research_plan.cid}, "
                    f"timeout={research_timeout}s)."
                )
                fallback_report = await client.fetch_deep_research_report(
                    research_plan, timeout=research_timeout
                )
                if fallback_report is None:
                    raise RuntimeError(
                        f"Deep research report not available over HTTP after "
                        f"{research_timeout}s; the research completes in your Gemini "
                        f"web history (cid={research_plan.cid})."
                    )
                result = None
        except Exception as e:
            logger.error(f"Deep research error: {e}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

        research_text = ""
        if result is not None:
            if result.final_output is not None:
                research_text = result.final_output.text or ""
            if not research_text:
                research_text = result.plan.response_text or ""
        else:
            research_text = fallback_report or ""
        if result is None or result.done:
            visible_output = research_text
        else:
            eta = result.plan.eta_text or ""
            visible_output = (
                f"Deep research started (research_id={result.plan.research_id}, {eta}). "
                f"The full report continues in your Gemini web history.\n\n{research_text}"
            )

        research_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(datetime.now(tz=UTC).timestamp())
        p_tok, c_tok, t_tok, r_tok = calculate_usage(app_messages, visible_output, None, "")
        usage = {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "completion_tokens_details": {"reasoning_tokens": r_tok},
        }
        return _create_chat_completion_standard_payload(
            research_id,
            created_time,
            request.model,
            visible_output,
            None,
            "stop",
            usage,
            "",
        )

    structured_requirement = _build_structured_requirement(request.response_format)
    extra_instr = [structured_requirement.instruction] if structured_requirement else None

    msgs = _prepare_messages_for_model(
        app_messages,
        request.tools,
        request.tool_choice,
        extra_instr,
    )

    use_temporary = _use_temporary_chat_mode()
    session, client, remain, stored_conv = await _find_reusable_session(
        db, pool, model, msgs, temporary=use_temporary
    )

    if session:
        if not remain:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")

        input_msgs = _prepare_messages_for_model(
            remain,
            request.tools,
            request.tool_choice,
            extra_instr,
            False,
        )
        try:
            m_input, files = await _process_conversation_with_timeout(input_msgs, tmp_dir)
        except TimeoutError:
            logger.warning(
                f"Input preprocessing timed out after {INPUT_PREPROCESS_TIMEOUT_SECONDS:.0f}s "
                f"(reused session)."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Input preprocessing timed out.",
            ) from None

        logger.debug(
            f"Reused session {reprlib.repr(session.metadata)} - sending {len(input_msgs)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model)
            m_input, files = await _process_conversation_with_timeout(msgs, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    if session is None or client is None:
        logger.error("No Gemini session or client available after preparing conversation.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No available Gemini client."
        )

    try:
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream, session, client = await _send_with_internal_fallback(
            pool=pool,
            db=db,
            model=model,
            session=session,
            client=client,
            current_input=m_input,
            files=files,
            full_prepared_messages=msgs,
            stored_conversation=stored_conv,
            tmp_dir=tmp_dir,
            stream=bool(request.stream),
            temporary=use_temporary,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        if isinstance(resp_or_stream, ModelOutput):
            logger.error("Expected a streaming response from Gemini but got a complete output.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Streaming response unavailable."
            )
        return _create_real_streaming_response(
            resp_or_stream,
            completion_id,
            created_time,
            request.model,
            msgs,
            db,
            model,
            client,
            session,
            base_url,
            structured_requirement,
        )

    if not isinstance(resp_or_stream, ModelOutput):
        logger.error("Expected a complete output from Gemini but got a stream.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Unexpected streaming response."
        )

    thoughts, visible_output, storage_output, tool_calls = process_llm_output(
        normalize_llm_text(resp_or_stream.thoughts or ""),
        normalize_llm_text(resp_or_stream.text or ""),
        structured_requirement,
    )

    images = resp_or_stream.images or []
    media_items: list[GeneratedVideo | GeneratedMedia] = (resp_or_stream.videos or []) + (
        resp_or_stream.media or []
    )
    unique_media = []
    seen_urls = set()
    for m in media_items:
        v_url = getattr(m, "url", None)
        a_url = getattr(m, "mp3_url", None)
        primary_url = v_url or a_url
        if primary_url and primary_url not in seen_urls:
            unique_media.append(m)
            seen_urls.add(primary_url)

    tasks = [_process_image_item(img) for img in images] + [
        _process_media_item(m) for m in unique_media
    ]
    results = await asyncio.gather(*tasks)

    image_markdown = ""
    media_markdown = ""
    seen_hashes = {}
    seen_media_hashes = {}
    media_store = get_media_store_dir()

    for res in results:
        if not res:
            continue

        if res[0] == "image":
            original_item = res[1]
            media_data = res[2]
            _, _, _, fname, fhash = media_data
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                fname = seen_hashes[fhash]
            else:
                seen_hashes[fhash] = fname

            img_url = f"{base_url}media/{fname}?token={get_media_token(fname)}"
            title = getattr(original_item, "title", "Image")
            image_markdown += f"\n\n![{title}]({img_url})"

        elif res[0] == "media":
            original_item = res[1]
            media_data = res[2]
            m_dict = media_data
            if not m_dict:
                continue

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    existing_name = seen_media_hashes[fhash]
                    if random_name != existing_name:
                        (media_store / random_name).unlink(missing_ok=True)
                    m_urls[mtype] = (
                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                    )
                else:
                    seen_media_hashes[fhash] = random_name
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

            title = getattr(original_item, "title", "Media")
            video_url = m_urls.get("video")
            audio_url = m_urls.get("audio")
            current_thumb = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            md_parts = []
            if video_url:
                md_parts.append(
                    f"[![{title}]({current_thumb})]({video_url})"
                    if current_thumb
                    else f"[{title}]({video_url})"
                )
            if audio_url:
                md_parts.append(
                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                    if current_thumb
                    else f"[{title} - Audio]({audio_url})"
                )

            if md_parts:
                media_markdown += f"\n\n{'\n\n'.join(md_parts)}"

    if image_markdown:
        visible_output += image_markdown
        storage_output += image_markdown

    if media_markdown:
        visible_output += media_markdown
        storage_output += media_markdown

    p_tok, c_tok, t_tok, r_tok = calculate_usage(app_messages, storage_output, tool_calls, thoughts)
    usage = {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": t_tok,
        "completion_tokens_details": {"reasoning_tokens": r_tok},
    }
    payload = _create_chat_completion_standard_payload(
        completion_id,
        created_time,
        request.model,
        visible_output,
        tool_calls or None,
        "tool_calls" if tool_calls else "stop",
        usage,
        thoughts,
    )
    _persist_conversation(
        db,
        model.model_name,
        client,
        session.metadata,
        msgs,
        storage_output,
        tool_calls,
    )
    return payload


@router.post("/v1/responses", response_model_exclude_none=True)
async def create_response(
    request: ResponseCreateRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    base_url = str(raw_request.base_url)
    base_messages = _convert_responses_to_app_messages(request.input)
    trace_id = _trace_id_from_headers(raw_request.headers)
    prompt_chars = sum(len(str(m.content)) for m in base_messages if m.content)
    logger.info(
        f"Request accepted: trace={trace_id} model={request.model} "
        f"stream={bool(request.stream)} prompt_chars={prompt_chars}"
    )
    structured_requirement = _build_structured_requirement(request.response_format)
    extra_instr = [structured_requirement.instruction] if structured_requirement else []

    standard_tools, image_tools, video_tools = [], [], []
    if request.tools:
        for t in request.tools:
            if isinstance(t, FunctionTool):
                standard_tools.append(t)
            elif isinstance(t, ImageGeneration):
                image_tools.append(t)
            elif isinstance(t, VideoGeneration):
                video_tools.append(t)
            elif isinstance(t, dict):
                if t.get("type") == "function":
                    standard_tools.append(FunctionTool.model_validate(t))
                elif t.get("type") == "image_generation":
                    image_tools.append(ImageGeneration.model_validate(t))
                elif t.get("type") == "video_generation":
                    video_tools.append(VideoGeneration.model_validate(t))

    img_instr = build_image_generation_instruction(
        image_tools,
        request.tool_choice if isinstance(request.tool_choice, ToolChoiceFunction) else None,
    )
    if img_instr:
        extra_instr.append(img_instr)
    video_instr = build_video_generation_instruction(
        video_tools,
        request.tool_choice if isinstance(request.tool_choice, ToolChoiceFunction) else None,
    )
    if video_instr:
        extra_instr.append(video_instr)
    preface = _convert_instructions_to_app_messages(request.instructions)
    conv_messages = [*preface, *base_messages] if preface else base_messages
    model_tool_choice = (
        request.tool_choice
        if isinstance(request.tool_choice, (str, ChatCompletionNamedToolChoice))
        else None
    )

    messages = _prepare_messages_for_model(
        conv_messages,
        standard_tools or None,
        model_tool_choice,
        extra_instr or None,
    )
    pool, db = GeminiClientPool(), LMDBConversationStore()
    try:
        model = _get_model_by_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    use_temporary = _use_temporary_chat_mode()
    session, client, remain, stored_conv = await _find_reusable_session(
        db, pool, model, messages, temporary=use_temporary
    )
    if session:
        msgs = _prepare_messages_for_model(
            remain,
            request.tools,
            request.tool_choice,
            None,
            False,
        )
        if not msgs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new messages.")
        try:
            m_input, files = await _process_conversation_with_timeout(msgs, tmp_dir)
        except TimeoutError:
            logger.warning(
                f"Input preprocessing timed out after {INPUT_PREPROCESS_TIMEOUT_SECONDS:.0f}s "
                "(reused session)."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Input preprocessing timed out.",
            ) from None
        logger.debug(
            f"Reused session {reprlib.repr(session.metadata)} - sending {len(msgs)} prepared messages."
        )
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model)
            m_input, files = await _process_conversation_with_timeout(messages, tmp_dir)
        except Exception as e:
            logger.error(f"Error in preparing conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            ) from e

    response_id = f"resp_{uuid.uuid4().hex}"
    created_time = int(datetime.now(tz=UTC).timestamp())

    if session is None or client is None:
        logger.error("No Gemini session or client available after preparing conversation.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No available Gemini client."
        )

    try:
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(m_input)}, files count: {len(files)}"
        )
        resp_or_stream, session, client = await _send_with_internal_fallback(
            pool=pool,
            db=db,
            model=model,
            session=session,
            client=client,
            current_input=m_input,
            files=files,
            full_prepared_messages=messages,
            stored_conversation=stored_conv,
            tmp_dir=tmp_dir,
            stream=bool(request.stream),
            temporary=use_temporary,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    if request.stream:
        if isinstance(resp_or_stream, ModelOutput):
            logger.error("Expected a streaming response from Gemini but got a complete output.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Streaming response unavailable."
            )
        return _create_responses_real_streaming_response(
            resp_or_stream,
            response_id,
            created_time,
            request.model,
            messages,
            db,
            model,
            client,
            session,
            request,
            base_url,
            structured_requirement,
        )

    if not isinstance(resp_or_stream, ModelOutput):
        logger.error("Expected a complete output from Gemini but got a stream.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Unexpected streaming response."
        )

    thoughts, assistant_text, storage_output, tool_calls = process_llm_output(
        normalize_llm_text(resp_or_stream.thoughts or ""),
        normalize_llm_text(resp_or_stream.text or ""),
        structured_requirement,
    )
    images = resp_or_stream.images or []
    videos = resp_or_stream.videos or []
    video_wait = request.video_wait_timeout or 600
    video_requested = (
        isinstance(request.tool_choice, ToolChoiceTypes)
        and request.tool_choice.type == "video_generation"
    ) or bool(video_tools)
    if video_requested and not videos:
        deadline = time.monotonic() + video_wait
        logger.debug(f"Video generation in progress, polling turns up to {video_wait}s.")
        while time.monotonic() < deadline:
            try:
                videos = await client.fetch_videos_from_turns(session.cid)
            except Exception as e:
                logger.warning(f"Failed to fetch videos from turns: {e}")
                break
            if videos:
                break
            await asyncio.sleep(15)
    if (
        isinstance(request.tool_choice, ToolChoiceTypes)
        and request.tool_choice.type == "image_generation"
    ) and not images:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No images returned.")
    if (
        isinstance(request.tool_choice, ToolChoiceTypes)
        and request.tool_choice.type == "video_generation"
    ) and not videos:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"No videos returned within {video_wait}s. Video generation is "
                "asynchronous; the video may still complete in Gemini web history."
            ),
        )

    unique_media = []
    seen_urls = set()
    for m in videos + (resp_or_stream.media or []):
        p_url = getattr(m, "url", None) or getattr(m, "mp3_url", None)
        if p_url and p_url not in seen_urls:
            unique_media.append(m)
            seen_urls.add(p_url)

    tasks = [_process_image_item(img) for img in images] + [
        _process_media_item(m) for m in unique_media
    ]
    results = await asyncio.gather(*tasks)

    contents, img_calls = [], []
    seen_hashes = {}
    seen_media_hashes = {}
    media_markdown = ""
    media_store = get_media_store_dir()

    for res in results:
        if not res:
            continue

        if res[0] == "image":
            media_data = res[2]
            b64, w, h, fname, fhash = media_data
            if fhash in seen_hashes:
                (media_store / fname).unlink(missing_ok=True)
                b64, w, h, fname = seen_hashes[fhash]
            else:
                seen_hashes[fhash] = (b64, w, h, fname)

            parts = fname.rsplit(".", 1)
            img_id = parts[0]
            fmt = parts[1] if len(parts) > 1 else "png"
            img_calls.append(
                ImageGenerationCall(
                    id=img_id, result=b64, output_format=fmt, size=f"{w}x{h}" if w and h else None
                )
            )

        elif res[0] == "media":
            original_item = res[1]
            media_data = res[2]
            m_dict = media_data
            if not m_dict:
                continue

            m_urls = {}
            for mtype, (random_name, fhash) in m_dict.items():
                if fhash in seen_media_hashes:
                    existing_name = seen_media_hashes[fhash]
                    if random_name != existing_name:
                        (media_store / random_name).unlink(missing_ok=True)
                    m_urls[mtype] = (
                        f"{base_url}media/{existing_name}?token={get_media_token(existing_name)}"
                    )
                else:
                    seen_media_hashes[fhash] = random_name
                    m_urls[mtype] = (
                        f"{base_url}media/{random_name}?token={get_media_token(random_name)}"
                    )

            title = getattr(original_item, "title", "Media")
            video_url = m_urls.get("video")
            audio_url = m_urls.get("audio")
            current_thumb = m_urls.get("video_thumbnail") or m_urls.get("audio_thumbnail")

            md_parts = []
            if video_url:
                md_parts.append(
                    f"[![{title}]({current_thumb})]({video_url})"
                    if current_thumb
                    else f"[{title}]({video_url})"
                )
            if audio_url:
                md_parts.append(
                    f"[![{title} - Audio]({current_thumb})]({audio_url})"
                    if current_thumb
                    else f"[{title} - Audio]({audio_url})"
                )

            if md_parts:
                media_markdown += f"\n\n{'\n\n'.join(md_parts)}"

    if assistant_text:
        contents.append(ResponseOutputText(type="output_text", text=assistant_text))

    image_markdown = ""
    for ic in img_calls:
        img_url = f"{base_url}media/{ic.id}.{ic.output_format}?token={get_media_token(f'{ic.id}.{ic.output_format}')}"
        image_markdown += f"\n\n![{ic.id}]({img_url})"

    if image_markdown:
        storage_output += image_markdown
        contents.append(ResponseOutputText(type="output_text", text=image_markdown))

    if media_markdown:
        storage_output += media_markdown
        contents.append(ResponseOutputText(type="output_text", text=media_markdown))

    if not contents:
        contents.append(ResponseOutputText(type="output_text", text=""))

    p_tok, c_tok, t_tok, r_tok = calculate_usage(messages, storage_output, tool_calls, thoughts)
    usage = ResponseUsage(
        input_tokens=p_tok,
        output_tokens=c_tok,
        total_tokens=t_tok,
        output_tokens_details={"reasoning_tokens": r_tok},
    )
    payload = _create_responses_standard_payload(
        response_id,
        created_time,
        request.model,
        tool_calls,
        img_calls,
        contents,
        usage,
        request,
        thoughts,
    )
    _persist_conversation(
        db,
        model.model_name,
        client,
        session.metadata,
        messages,
        storage_output,
        tool_calls,
    )
    return payload
