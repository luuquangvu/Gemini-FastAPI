# ruff: noqa: N815  # camelCase field names are the Gemini REST wire format
"""Gemini REST API 原生格式的 Pydantic 数据模型。

覆盖 generateContent / streamGenerateContent / models.list / models.get 端点所需的
请求体和响应体结构, 遵循 Google Gemini REST API v1beta 规范。
(Adapted from fork fujunchao — unchanged, per C1 contract.)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 请求模型 — 内部 Part 类型
# ---------------------------------------------------------------------------


class GeminiInlineData(BaseModel):
    """内嵌二进制数据(图片等)。"""

    mimeType: str
    data: str  # base64


class GeminiFileData(BaseModel):
    """通过 Files API 上传的文件引用。"""

    mimeType: str
    fileUri: str


class GeminiFunctionCall(BaseModel):
    """模型发起的函数调用。"""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class GeminiFunctionResponse(BaseModel):
    """用户提交的函数执行结果。"""

    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class GeminiPart(BaseModel):
    """Content 中的一个 part, 可能是文本/思考、内嵌数据、函数调用或函数响应。"""

    text: str | None = None
    thought: bool | None = None
    inlineData: GeminiInlineData | None = None
    functionCall: GeminiFunctionCall | None = None
    functionResponse: GeminiFunctionResponse | None = None
    fileData: GeminiFileData | None = None


# ---------------------------------------------------------------------------
# 请求模型 — 顶层结构
# ---------------------------------------------------------------------------


class GeminiContent(BaseModel):
    """一条消息内容(用户 / 模型 / 函数角色)。"""

    role: Literal["user", "model", "function"] | None = None
    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiSystemInstruction(BaseModel):
    """系统指令(顶层, 不在 contents 中)。"""

    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiFunctionDeclaration(BaseModel):
    """函数声明。"""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GeminiTool(BaseModel):
    """工具集合。"""

    functionDeclarations: list[GeminiFunctionDeclaration] = Field(default_factory=list)


class GeminiFunctionCallingConfig(BaseModel):
    """函数调用配置。"""

    mode: Literal["AUTO", "NONE", "ANY"] = "AUTO"
    allowedFunctionNames: list[str] | None = None


class GeminiToolConfig(BaseModel):
    """工具配置。"""

    functionCallingConfig: GeminiFunctionCallingConfig | None = None


class GeminiSafetySetting(BaseModel):
    """安全设置。"""

    category: str
    threshold: str


class GeminiThinkingConfig(BaseModel):
    """思考配置(控制模型推理行为)。"""

    includeThoughts: bool | None = None
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None  # OFF | LOW | MEDIUM | HIGH


class GeminiGenerationConfig(BaseModel):
    """生成参数。"""

    temperature: float | None = None
    topP: float | None = None
    topK: int | None = None
    maxOutputTokens: int | None = None
    stopSequences: list[str] | None = None
    responseMimeType: str | None = None
    responseSchema: dict[str, Any] | None = None
    responseJsonSchema: dict[str, Any] | None = None
    candidateCount: int | None = None
    thinkingConfig: GeminiThinkingConfig | None = None


class GeminiGenerateContentRequest(BaseModel):
    """generateContent / streamGenerateContent 请求体。"""

    contents: list[GeminiContent] = Field(default_factory=list)
    systemInstruction: GeminiSystemInstruction | None = None
    tools: list[GeminiTool] | None = None
    toolConfig: GeminiToolConfig | None = None
    safetySettings: list[GeminiSafetySetting] | None = None
    generationConfig: GeminiGenerationConfig | None = None
    cachedContent: str | None = None


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class GeminiSafetyRating(BaseModel):
    """安全评级。"""

    category: str
    probability: str


class GeminiCitationSource(BaseModel):
    """单条引用来源。"""

    startIndex: int | None = None
    endIndex: int | None = None
    uri: str | None = None
    license: str | None = None


class GeminiCitationMetadata(BaseModel):
    """引用元数据。"""

    citationSources: list[GeminiCitationSource] = Field(default_factory=list)


class GeminiGroundingChunkWeb(BaseModel):
    """溯源来源网页信息。"""

    uri: str | None = None
    title: str | None = None


class GeminiGroundingChunk(BaseModel):
    """溯源来源块。"""

    web: GeminiGroundingChunkWeb | None = None


class GeminiSearchEntryPoint(BaseModel):
    """搜索入口点。"""

    renderedContent: str | None = None
    sdkBlob: str | None = None


class GeminiGroundingSupport(BaseModel):
    """溯源支撑片段。"""

    segment: dict[str, Any] | None = None
    groundingChunkIndices: list[int] = Field(default_factory=list)
    confidenceScores: list[float] = Field(default_factory=list)


class GeminiGroundingMetadata(BaseModel):
    """溯源元数据(Google Search 等)。"""

    webSearchQueries: list[str] = Field(default_factory=list)
    groundingChunks: list[GeminiGroundingChunk] = Field(default_factory=list)
    searchEntryPoint: GeminiSearchEntryPoint | None = None
    groundingSupports: list[GeminiGroundingSupport] = Field(default_factory=list)


class GeminiCandidate(BaseModel):
    """生成候选项。"""

    content: GeminiContent | None = None
    finishReason: str | None = None
    index: int = 0
    safetyRatings: list[GeminiSafetyRating] = Field(default_factory=list)
    citationMetadata: GeminiCitationMetadata | None = None
    groundingMetadata: GeminiGroundingMetadata | None = None
    tokenCount: int | None = None
    avgLogprobs: float | None = None


class GeminiUsageMetadata(BaseModel):
    """用量统计。"""

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0
    thoughtsTokenCount: int | None = None
    cachedContentTokenCount: int | None = None
    toolUsePromptTokenCount: int | None = None


class GeminiGenerateContentResponse(BaseModel):
    """generateContent / streamGenerateContent 响应体。"""

    candidates: list[GeminiCandidate] = Field(default_factory=list)
    usageMetadata: GeminiUsageMetadata | None = None
    modelVersion: str | None = None


# ---------------------------------------------------------------------------
# models.list / models.get 响应
# ---------------------------------------------------------------------------


class GeminiModelInfo(BaseModel):
    """单个模型信息。"""

    name: str
    version: str | None = None
    displayName: str | None = None
    description: str | None = None
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = None
    supportedGenerationMethods: list[str] = Field(default_factory=list)


class GeminiModelListResponse(BaseModel):
    """models.list 响应体。"""

    models: list[GeminiModelInfo] = Field(default_factory=list)
    nextPageToken: str | None = None


# ---------------------------------------------------------------------------
# 错误响应
# ---------------------------------------------------------------------------


class GeminiErrorDetail(BaseModel):
    """Google API 标准错误详情。"""

    code: int
    message: str
    status: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class GeminiErrorResponse(BaseModel):
    """Google API 标准错误包装。"""

    error: GeminiErrorDetail
