"""Gemini REST API v1beta native endpoints (ported from fork fujunchao, adapted to our tree).

Endpoints (Google Gemini API compatible):
- GET  /v1beta/models             — list models
- GET  /v1beta/models/{model}     — get one model
- POST /v1beta/models/{model}:generateContent       — non-streaming generation
- POST /v1beta/models/{model}:streamGenerateContent  — streaming (SSE)

Reuses OUR chat.py helpers; no chat.py changes were made (C1 parallel-safety).
Adaptation deltas vs the fork (see .metacog/fork-evals/fujunchao.md §5):
- helpers live in app/utils/helper.py as calculate_usage / process_llm_output /
  normalize_llm_text (no _-prefix, no chat.py copies)
- image store helpers are get_media_store_dir / get_media_token; media served at
  /media/{fname}?token= (not /images/)
- _find_reusable_session returns a 4-tuple here (session, client, remain, conv)
- _persist_conversation takes the client wrapper and no thoughts argument here
- _get_available_models requires a pool: pass GeminiClientPool() (a singleton that
  lifespan already initialized)
- GeminiClientWrapper.extract_output does not exist here; use normalize_llm_text
- internal message types are AppMessage/AppContentItem/AppToolCall(+Function),
  tools are FunctionTool (flat schema)
- Gemini fileData (Files API URI) parts are logged and skipped: our pipeline only
  accepts base64 file_data items, and a Google fileUri cannot be fetched locally.
"""

from __future__ import annotations

import io
import reprlib
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import ModelOutput
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models import (
    AppContentItem,
    AppMessage,
    AppToolCall,
    AppToolCallFunction,
    FunctionTool,
    ToolChoiceFunction,
)
from app.models.gemini_models import (
    GeminiCandidate,
    GeminiContent,
    GeminiErrorDetail,
    GeminiErrorResponse,
    GeminiFunctionCall,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiModelInfo,
    GeminiModelListResponse,
    GeminiPart,
    GeminiUsageMetadata,
)

# 从 chat.py 导入已有辅助函数(不修改 chat.py)
from app.server.chat import (
    StreamingOutputFilter,
    _build_structured_requirement,
    _find_reusable_session,
    _get_available_models,
    _image_to_base64,
    _persist_conversation,
    _prepare_messages_for_model,
    _resolve_model_name,
    _send_with_split,
)
from app.server.middleware import (
    get_media_store_dir,
    get_media_token,
    get_temp_dir,
    verify_gemini_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore
from app.utils.helper import calculate_usage, normalize_llm_text, process_llm_output

router = APIRouter()


def add_gemini_exception_handlers(app: FastAPI) -> None:
    """Register Google-style HTTP and validation errors for /v1beta routes."""

    @app.exception_handler(StarletteHTTPException)
    async def gemini_http_exception_handler(request: Request, exc: StarletteHTTPException):
        if not request.url.path.startswith("/v1beta/"):
            return await http_exception_handler(request, exc)

        grpc_status = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            429: "RESOURCE_EXHAUSTED",
            503: "UNAVAILABLE",
        }.get(exc.status_code, "INTERNAL")
        err = _to_gemini_error(exc.status_code, str(exc.detail), grpc_status)
        return JSONResponse(status_code=exc.status_code, content=err.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def gemini_validation_exception_handler(request: Request, exc: RequestValidationError):
        """Convert Gemini-route 422 validation errors into Google API error format."""
        if request.url.path.startswith("/v1beta/"):
            detail = str(exc.errors()) if exc.errors() else str(exc)
            err = GeminiErrorResponse(
                error=GeminiErrorDetail(
                    code=400,
                    message=f"Invalid request: {detail}",
                    status="INVALID_ARGUMENT",
                )
            )
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))
        return await request_validation_exception_handler(request, exc)


# ---------------------------------------------------------------------------
# Gemini ↔ 内部格式转换函数
# ---------------------------------------------------------------------------


def _gemini_contents_to_messages(
    contents: list[GeminiContent],
    system_instruction: Any | None = None,
) -> list[AppMessage]:
    """Convert Gemini contents + systemInstruction into internal AppMessage list.

    Handles:
    - role="model" + functionCall → assistant message + tool_calls
    - role="user"/"function" + functionResponse → tool messages
    - multiple functionResponse parts → multiple tool messages
    - multi-modal parts keep original order (text/image interleaved)
    """
    messages: list[AppMessage] = []

    if system_instruction:
        sys_parts = (
            system_instruction.parts
            if hasattr(system_instruction, "parts")
            else (system_instruction.get("parts") or [])
        )
        if sys_texts := [p.text for p in sys_parts if p.text]:
            messages.append(AppMessage(role="system", content="\n".join(sys_texts)))

    # Track the previous assistant message's tool_call IDs for functionResponse mapping
    last_tool_call_ids: dict[str, str] = {}  # function_name → call_id

    for content in contents:
        role = content.role or "user"
        parts = content.parts or []

        internal_role = cast(Literal["system", "user", "assistant", "tool"], role)
        if role == "model":
            internal_role = "assistant"
        elif role == "function":
            internal_role = "tool"

        text_fragments: list[str] = []
        content_items: list[AppContentItem] = []
        tool_calls: list[AppToolCall] = []
        function_responses: list[tuple[str | None, str]] = []  # (name, content_json)

        for part in parts:
            if part.text is not None:
                text_fragments.append(part.text)

            if part.inlineData:
                data_url = f"data:{part.inlineData.mimeType};base64,{part.inlineData.data}"
                content_items.append(AppContentItem(type="image_url", url=data_url))

            if part.fileData:
                # Our pipeline only accepts base64 file_data items; a Google Files API
                # fileUri cannot be fetched locally. Log and skip (disclosed in C1 report).
                logger.warning(
                    "[Gemini API] Skipping fileData part "
                    f"(unsupported): {reprlib.repr(part.fileData.fileUri)}"
                )

            if part.functionCall:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                tool_calls.append(
                    AppToolCall(
                        id=call_id,
                        type="function",
                        function=AppToolCallFunction(
                            name=part.functionCall.name,
                            arguments=(
                                orjson.dumps(part.functionCall.args).decode("utf-8")
                                if part.functionCall.args
                                else "{}"
                            ),
                        ),
                    )
                )
                last_tool_call_ids[part.functionCall.name] = call_id

            if part.functionResponse:
                resp_content = orjson.dumps(part.functionResponse.response).decode("utf-8")
                function_responses.append((part.functionResponse.name, resp_content))

        if function_responses:
            # functionResponse → tool messages regardless of original role
            for fn_name, fn_content in function_responses:
                call_id = last_tool_call_ids.get(fn_name or "", f"call_{uuid.uuid4().hex[:24]}")
                messages.append(
                    AppMessage(
                        role="tool",
                        content=fn_content,
                        name=fn_name,
                        tool_call_id=call_id,
                    )
                )
            if text_fragments and internal_role != "tool":
                messages.append(AppMessage(role=internal_role, content="\n".join(text_fragments)))
        elif tool_calls:
            msg_content = "\n".join(text_fragments) if text_fragments else None
            messages.append(
                AppMessage(role="assistant", content=msg_content, tool_calls=tool_calls)
            )
        elif content_items or text_fragments:
            if content_items:
                # keep original interleaved text/image order
                ordered_items: list[AppContentItem] = []
                text_idx, media_idx = 0, 0
                for part in parts:
                    if part.text is not None and text_idx < len(text_fragments):
                        ordered_items.append(
                            AppContentItem(type="text", text=text_fragments[text_idx])
                        )
                        text_idx += 1
                    elif (part.inlineData or part.fileData) and media_idx < len(content_items):
                        ordered_items.append(content_items[media_idx])
                        media_idx += 1
                messages.append(AppMessage(role=internal_role, content=ordered_items))
            else:
                messages.append(AppMessage(role=internal_role, content="\n".join(text_fragments)))
        else:
            # empty parts: still create a message to keep conversation structure
            messages.append(AppMessage(role=internal_role, content=""))

    return messages


def _gemini_tools_to_internal(
    tools: list[Any] | None,
    tool_config: Any | None = None,
) -> tuple[
    list[FunctionTool] | None,
    Literal["none", "auto", "required"] | ToolChoiceFunction | None,
]:
    """Convert Gemini tools + toolConfig into internal FunctionTool list and tool_choice."""
    if not tools:
        return None, None

    internal_tools: list[FunctionTool] = []
    for tool in tools:
        internal_tools.extend(
            FunctionTool(
                type="function",
                name=decl.name,
                description=decl.description,
                parameters=decl.parameters,
            )
            for decl in tool.functionDeclarations or []
        )
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceFunction | None = None
    if tool_config and tool_config.functionCallingConfig:
        mode = tool_config.functionCallingConfig.mode.upper()
        if mode == "NONE":
            tool_choice = cast(Literal["none", "auto", "required"], "none")
        elif mode == "ANY":
            tool_choice = cast(Literal["none", "auto", "required"], "required")
        else:  # AUTO
            tool_choice = "auto"

    return internal_tools or None, tool_choice


def _to_gemini_response(
    visible_text: str | None,
    tool_calls: list[Any],
    thoughts: str | None,
    usage_tuple: tuple[int, int, int, int],
    model_name: str,
    image_parts: list[GeminiPart] | None = None,
) -> GeminiGenerateContentResponse:
    """Convert the internal processing result into a Gemini API response."""
    parts: list[GeminiPart] = []

    if thoughts:
        parts.append(GeminiPart(text=thoughts, thought=True))

    if visible_text:
        parts.append(GeminiPart(text=visible_text))

    if image_parts:
        parts.extend(image_parts)

    for tc in tool_calls:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        fn_args_raw = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        try:
            fn_args = orjson.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
        except orjson.JSONDecodeError:
            fn_args = {}
        parts.append(GeminiPart(functionCall=GeminiFunctionCall(name=fn_name, args=fn_args)))

    finish_reason = "STOP"
    p_tok, c_tok, t_tok, r_tok = usage_tuple

    candidate = GeminiCandidate(
        content=GeminiContent(role="model", parts=parts),
        finishReason=finish_reason,
        index=0,
    )

    usage_meta = GeminiUsageMetadata(
        promptTokenCount=p_tok,
        candidatesTokenCount=c_tok - r_tok,
        totalTokenCount=t_tok,
        thoughtsTokenCount=r_tok if r_tok > 0 else None,
    )

    return GeminiGenerateContentResponse(
        candidates=[candidate],
        usageMetadata=usage_meta,
        modelVersion=model_name,
    )


def _to_gemini_error(status_code: int, message: str, grpc_status: str) -> GeminiErrorResponse:
    """Build a Google API standard error response."""
    return GeminiErrorResponse(
        error=GeminiErrorDetail(
            code=status_code,
            message=message,
            status=grpc_status,
        )
    )


def _validate_gemini_request(request: GeminiGenerateContentRequest) -> str | None:
    """Reject only structures that would otherwise be silently dropped from the prompt."""
    if not request.contents:
        return "contents is required and cannot be empty."

    for content in request.contents:
        if not content.parts:
            return "Each content entry must contain at least one part."
        if any(part.fileData is not None for part in content.parts):
            return "fileData is not supported; provide the data using inlineData instead."
    if request.systemInstruction and any(
        part.fileData is not None for part in request.systemInstruction.parts
    ):
        return "fileData is not supported; provide the data using inlineData instead."
    return None


def _strip_model_prefix(model: str) -> str:
    """Strip a leading 'models/' prefix if present."""
    return model[len("models/") :] if model.startswith("models/") else model


def _model_data_to_gemini_info(model_data: Any) -> GeminiModelInfo:
    """Convert internal ModelData into GeminiModelInfo."""
    return GeminiModelInfo(
        name=f"models/{model_data.id}",
        displayName=model_data.id,
        description=f"Gemini model: {model_data.id}",
        supportedGenerationMethods=["generateContent", "streamGenerateContent"],
    )


# ---------------------------------------------------------------------------
# 路由端点
# ---------------------------------------------------------------------------


@router.get("/v1beta/models")
async def gemini_list_models(api_key: str = Depends(verify_gemini_api_key)):
    """List available models (Gemini API format)."""
    models = await _get_available_models(GeminiClientPool())

    logger.info(f"[Gemini API] Retrieved {len(models)} models")
    if not models:
        logger.warning("[Gemini API] Model list is empty")

    gemini_models = [_model_data_to_gemini_info(m) for m in models]
    return GeminiModelListResponse(models=gemini_models)


@router.get("/v1beta/models/{model:path}")
async def gemini_get_model(model: str, api_key: str = Depends(verify_gemini_api_key)):
    """Get one model's info (Gemini API format)."""
    model_name = _strip_model_prefix(model)
    try:
        _resolve_model_name(GeminiClientPool(), model_name)
    except ValueError as exc:
        err = _to_gemini_error(404, str(exc), "NOT_FOUND")
        return JSONResponse(status_code=404, content=err.model_dump(mode="json"))

    all_models = await _get_available_models(GeminiClientPool())
    for m in all_models:
        if m.id == model_name:
            return _model_data_to_gemini_info(m)

    # Model exists but not listed (e.g. resolved directly from gemini-webapi constants)
    return GeminiModelInfo(
        name=f"models/{model_name}",
        displayName=model_name,
        description=f"Gemini model: {model_name}",
        supportedGenerationMethods=["generateContent", "streamGenerateContent"],
    )


@router.post("/v1beta/models/{model:path}:generateContent")
async def gemini_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """Non-streaming content generation (Gemini API format)."""
    model_name = _strip_model_prefix(model)

    try:
        model_obj = _resolve_model_name(GeminiClientPool(), model_name)
    except ValueError as exc:
        err = _to_gemini_error(400, str(exc), "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    if validation_error := _validate_gemini_request(request):
        err = _to_gemini_error(400, validation_error, "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    messages = _gemini_contents_to_messages(request.contents, request.systemInstruction)

    internal_tools, tool_choice = _gemini_tools_to_internal(request.tools, request.toolConfig)

    structured_requirement = None
    if request.generationConfig:
        gen_cfg = request.generationConfig
        schema = gen_cfg.responseSchema or gen_cfg.responseJsonSchema
        if gen_cfg.responseMimeType == "application/json" and schema:
            structured_requirement = _build_structured_requirement(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )

    extra_instr = [structured_requirement.instruction] if structured_requirement else None

    msgs = _prepare_messages_for_model(messages, internal_tools, tool_choice, extra_instr)

    pool, db = GeminiClientPool(), LMDBConversationStore()

    session, client, remain, _conv = await _find_reusable_session(db, pool, model_obj, msgs)

    if session:
        if not remain:
            err = _to_gemini_error(400, "No new messages to send.", "INVALID_ARGUMENT")
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

        input_msgs = _prepare_messages_for_model(
            remain, internal_tools, tool_choice, extra_instr, False
        )
        m_input, files = await GeminiClientWrapper.process_conversation(input_msgs, tmp_dir)
        logger.debug(
            f"[Gemini API] Reusing session {reprlib.repr(session.metadata)}"
            f" - sending {len(input_msgs)} message(s)."
        )
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model_obj)
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.exception("[Gemini API] Failed to prepare session")
            err = _to_gemini_error(503, str(e), "UNAVAILABLE")
            return JSONResponse(status_code=503, content=err.model_dump(mode="json"))

    try:
        assert session is not None
        assert client is not None
        logger.debug(
            f"[Gemini API] Client: {client.id}, input len: {len(m_input)}, files: {len(files)}"
        )
        resp = await _send_with_split(
            session, m_input, files=cast("list[Path | str | io.BytesIO]", files), stream=False
        )
    except Exception as e:
        logger.exception("[Gemini API] Gemini call failed")
        err = _to_gemini_error(502, str(e), "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    try:
        assert isinstance(resp, ModelOutput)
        thoughts = normalize_llm_text(resp.thoughts or "")
        raw_clean = normalize_llm_text(resp.text or "")
    except Exception:
        logger.exception("[Gemini API] Output parsing failed")
        err = _to_gemini_error(502, "Malformed response.", "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    thoughts, visible_output, storage_output, tool_calls = process_llm_output(
        thoughts, raw_clean, structured_requirement
    )

    # Images: collect Gemini images → inlineData parts + markdown URL for LMDB persistence
    image_parts: list[GeminiPart] = []
    seen_hashes: set[str] = set()
    image_store = get_media_store_dir()
    base_url = str(raw_request.base_url).rstrip("/")
    for image in resp.images or []:
        try:
            b64_str, _w, _h, fname, file_hash = await _image_to_base64(image, image_store)
            if file_hash in seen_hashes:
                (image_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(file_hash)
            suffix = Path(fname).suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix, "image/png")
            image_parts.append(
                GeminiPart(inlineData=GeminiInlineData(mimeType=mime_type, data=b64_str))
            )
            token = get_media_token(fname)
            img_url = f"{base_url}/media/{fname}?token={token}"
            storage_output += f"\n\n![{fname}]({img_url})"
        except Exception as exc:
            logger.warning(f"[Gemini API] Failed to process image: {exc}")

    usage_tuple = calculate_usage(messages, visible_output, tool_calls, thoughts)

    _persist_conversation(
        db,
        model_obj,
        client,
        session.metadata,
        msgs,
        storage_output,
        tool_calls,
    )

    return _to_gemini_response(
        visible_output, tool_calls, thoughts, usage_tuple, model_name, image_parts
    )


@router.post("/v1beta/models/{model:path}:streamGenerateContent")
async def gemini_stream_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """Streaming content generation (Gemini API format, SSE)."""
    model_name = _strip_model_prefix(model)

    try:
        model_obj = _resolve_model_name(GeminiClientPool(), model_name)
    except ValueError as exc:
        err = _to_gemini_error(400, str(exc), "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    if validation_error := _validate_gemini_request(request):
        err = _to_gemini_error(400, validation_error, "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    messages = _gemini_contents_to_messages(request.contents, request.systemInstruction)
    internal_tools, tool_choice = _gemini_tools_to_internal(request.tools, request.toolConfig)

    structured_requirement = None
    if request.generationConfig:
        gen_cfg = request.generationConfig
        schema = gen_cfg.responseSchema or gen_cfg.responseJsonSchema
        if gen_cfg.responseMimeType == "application/json" and schema:
            structured_requirement = _build_structured_requirement(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )

    extra_instr = [structured_requirement.instruction] if structured_requirement else None
    msgs = _prepare_messages_for_model(messages, internal_tools, tool_choice, extra_instr)

    pool, db = GeminiClientPool(), LMDBConversationStore()
    session, client, remain, _conv = await _find_reusable_session(db, pool, model_obj, msgs)

    if session:
        if not remain:
            err = _to_gemini_error(400, "No new messages to send.", "INVALID_ARGUMENT")
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

        input_msgs = _prepare_messages_for_model(
            remain, internal_tools, tool_choice, extra_instr, False
        )
        m_input, files = await GeminiClientWrapper.process_conversation(input_msgs, tmp_dir)
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model_obj)
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.exception("[Gemini API] Failed to prepare streaming session")
            err = _to_gemini_error(503, str(e), "UNAVAILABLE")
            return JSONResponse(status_code=503, content=err.model_dump(mode="json"))

    try:
        assert session is not None
        assert client is not None
        generator = await _send_with_split(
            session, m_input, files=cast("list[Path | str | io.BytesIO]", files), stream=True
        )
    except Exception as e:
        logger.exception("[Gemini API] Gemini streaming call failed")
        err = _to_gemini_error(502, str(e), "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    return _create_gemini_streaming_response(
        generator=generator,
        model_name=model_name,
        messages=msgs,
        original_messages=messages,
        db=db,
        model=model_obj,
        client_wrapper=client,
        session=session,
        structured_requirement=structured_requirement,
        base_url=str(raw_request.base_url).rstrip("/"),
    )


def _create_gemini_streaming_response(
    generator,
    model_name: str,
    messages: list[AppMessage],
    original_messages: list[AppMessage],
    db: LMDBConversationStore,
    model,
    client_wrapper: GeminiClientWrapper,
    session,
    structured_requirement=None,
    base_url: str = "",
) -> StreamingResponse:
    """Create a Gemini-format SSE streaming response."""

    async def generate_stream():
        full_thoughts, full_text = "", ""
        last_chunk: ModelOutput | None = None
        all_images: list[Any] = []  # images from all chunks (url-deduped)
        seen_image_urls: set[str] = set()
        suppressor = StreamingOutputFilter()

        try:
            async for chunk in generator:
                last_chunk = chunk

                if chunk.images:
                    for img in chunk.images:
                        if img.url not in seen_image_urls:
                            all_images.append(img)
                            seen_image_urls.add(img.url)

                if t_delta := chunk.thoughts_delta:
                    full_thoughts += t_delta
                    think_resp = GeminiGenerateContentResponse(
                        candidates=[
                            GeminiCandidate(
                                content=GeminiContent(
                                    role="model",
                                    parts=[GeminiPart(text=t_delta, thought=True)],
                                ),
                                index=0,
                            )
                        ],
                    )
                    yield f"data: {orjson.dumps(think_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

                if text_delta := chunk.text_delta:
                    full_text += text_delta
                    if visible_delta := suppressor.process(text_delta):
                        chunk_resp = GeminiGenerateContentResponse(
                            candidates=[
                                GeminiCandidate(
                                    content=GeminiContent(
                                        role="model",
                                        parts=[GeminiPart(text=visible_delta)],
                                    ),
                                    index=0,
                                )
                            ],
                        )
                        yield f"data: {orjson.dumps(chunk_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

        except Exception as e:
            logger.exception(f"[Gemini API] Streaming error: {e}")
            err_resp = _to_gemini_error(500, "Streaming error occurred.", "INTERNAL")
            yield f"data: {orjson.dumps(err_resp.model_dump(mode='json')).decode('utf-8')}\n\n"
            return

        # Use the final chunk's full text if available
        if last_chunk is not None:
            if last_chunk.text:
                full_text = last_chunk.text
            if last_chunk.thoughts:
                full_thoughts = last_chunk.thoughts

        if remaining_text := suppressor.flush():
            chunk_resp = GeminiGenerateContentResponse(
                candidates=[
                    GeminiCandidate(
                        content=GeminiContent(
                            role="model",
                            parts=[GeminiPart(text=remaining_text)],
                        ),
                        index=0,
                    )
                ],
            )
            yield f"data: {orjson.dumps(chunk_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

        # --- post-processing: protective layer so the SSE tail survives errors ---
        try:
            _thoughts, visible_output, storage_output, tool_calls = process_llm_output(
                full_thoughts, full_text, structured_requirement
            )

            image_store = get_media_store_dir()
            seen_hashes: set[str] = set()
            for image in all_images:
                try:
                    b64_str, _w, _h, fname, file_hash = await _image_to_base64(image, image_store)
                    if file_hash in seen_hashes:
                        (image_store / fname).unlink(missing_ok=True)
                        continue
                    seen_hashes.add(file_hash)
                    suffix = Path(fname).suffix.lower()
                    mime_map = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                        ".gif": "image/gif",
                    }
                    mime_type = mime_map.get(suffix, "image/png")
                    img_chunk = GeminiGenerateContentResponse(
                        candidates=[
                            GeminiCandidate(
                                content=GeminiContent(
                                    role="model",
                                    parts=[
                                        GeminiPart(
                                            inlineData=GeminiInlineData(
                                                mimeType=mime_type, data=b64_str
                                            )
                                        )
                                    ],
                                ),
                                index=0,
                            )
                        ],
                    )
                    yield f"data: {orjson.dumps(img_chunk.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"
                    token = get_media_token(fname)
                    img_url = f"{base_url}/media/{fname}?token={token}"
                    storage_output += f"\n\n![{fname}]({img_url})"
                except Exception as exc:
                    logger.warning(f"[Gemini API] Failed to process streaming image: {exc}")
            # Final chunk (finishReason + usageMetadata)
            usage_tuple = calculate_usage(original_messages, visible_output, tool_calls, _thoughts)
            p_tok, c_tok, t_tok, r_tok = usage_tuple

            final_parts: list[GeminiPart] = []
            if tool_calls:
                for tc in tool_calls:
                    tc_any: Any = tc
                    fn = (
                        tc_any.function
                        if hasattr(tc_any, "function")
                        else tc_any.get("function", {})
                    )
                    fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    fn_args_raw = (
                        fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                    )
                    try:
                        fn_args = (
                            orjson.loads(fn_args_raw)
                            if isinstance(fn_args_raw, str)
                            else fn_args_raw
                        )
                    except orjson.JSONDecodeError:
                        fn_args = {}
                    final_parts.append(
                        GeminiPart(functionCall=GeminiFunctionCall(name=fn_name, args=fn_args))
                    )

            final_resp = GeminiGenerateContentResponse(
                candidates=[
                    GeminiCandidate(
                        content=GeminiContent(role="model", parts=final_parts)
                        if final_parts
                        else None,
                        finishReason="STOP",
                        index=0,
                    )
                ],
                usageMetadata=GeminiUsageMetadata(
                    promptTokenCount=p_tok,
                    candidatesTokenCount=c_tok - r_tok,
                    totalTokenCount=t_tok,
                    thoughtsTokenCount=r_tok if r_tok > 0 else None,
                ),
                modelVersion=model_name,
            )
            yield f"data: {orjson.dumps(final_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

            _persist_conversation(
                db,
                model.model_name,
                client_wrapper,
                session.metadata,
                messages,
                storage_output,
                tool_calls,
            )
        except Exception as exc:
            logger.exception(f"[Gemini API] Post-processing error: {exc}")
            err_resp = _to_gemini_error(500, "Post-processing error.", "INTERNAL")
            yield f"data: {orjson.dumps(err_resp.model_dump(mode='json')).decode('utf-8')}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
