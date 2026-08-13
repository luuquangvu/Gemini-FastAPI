# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ [English](README.md) | 中文 ]

将 Gemini 网页端模型封装为兼容 OpenAI API 的 API Server。基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 实现。

**无需 API Key，免费通过 API 调用 Gemini 网页端模型！**

## 功能特性

- **无需 Google API Key**：只需网页 Cookie，即可免费通过 API 调用 Gemini 模型。
- **内置 Google 搜索**：API 已内置 Gemini 网页端的搜索能力，模型响应更加准确。
- **会话持久化**：基于 LMDB 存储，支持多轮对话历史记录。
- **多模态支持**：可处理文本、图片及文件上传。
- **多账户负载均衡**：支持多账户分发请求，可为每个账户单独配置代理。

## 快速开始

**如需 Docker 部署，请参见下方 [Docker 部署](#docker-部署) 部分。**

### 前置条件

- Python 3.13
- 拥有网页版 Gemini 访问权限的 Google 账号 (开启 **[Gemini Apps 应用活动](https://myactivity.google.com/product/gemini)** 以获得最佳会话持久化体验)
- 从 Gemini 网页获取的 `secure_1psid` 和 `secure_1psidts` Cookie

### 安装

#### 使用 uv (推荐)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### 使用 pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### 配置

编辑 `config/config.yaml` 并提供至少一组凭证：

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # 可选代理 URL (null/空值则保持直连)
      impersonate: null # 可选浏览器指纹模拟 (null 则使用库的默认值)
```

> [!NOTE]
> 详细说明请参见下方 [配置](#配置说明) 部分。

### 启动服务

```bash
# 使用 uv
uv run python run.py

# 直接用 Python
python run.py
```

服务默认启动在 `http://localhost:8000`。

## API 接口

本服务器提供了一系列接口，重点支持 OpenAI 兼容协议。

### OpenAI 兼容接口

这些接口遵循 OpenAI 的 API 规范，允许你将 Gemini 作为 **Drop-in 替代方案** 直接接入现有的 AI 应用。

- **`GET /v1/models`**: 列出所有可用的 Gemini 模型。
- **`POST /v1/chat/completions`**: 统一聊天对话接口。
  - **流式传输**: 设置 `stream: true` 即可实时接收增量响应 (Stream Delta)。
  - **多模态支持**: 支持在消息中包含文本、图片以及文件上传。
  - **工具调用**: 支持通过 `tools` 参数进行函数调用 (Function Calling)。
  - **结构化输出**: 支持 `response_format`，可严格遵循 JSON Schema。

### 高级接口

- **`POST /v1/responses`**: 用于复杂交互模式的专用接口，支持分步输出、生成图片及工具调用等更丰富的响应项。

### 实用工具接口

- **`GET /health`**: 健康检查接口。返回服务器、已配置的 Gemini 客户端以及对话存储的状态。
- **`GET /media/{filename}`**: 用于分发生成的媒体内容的内部接口。需要有效的 Token（API 返回的图片 URL 中已自动包含该 Token）。

## Docker 部署

### 直接运行

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -e CONFIG_SERVER__API_KEY="your-api-key-here" \
  -e CONFIG_GEMINI__CLIENTS__0__ID="client-a" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts" \
  -e GEMINI_COOKIE_PATH="/app/cache" \
  ghcr.io/nativu5/gemini-fastapi
```

> [!TIP]
> 需要代理时可添加 `CONFIG_GEMINI__CLIENTS__0__PROXY`；省略该变量将保持直连。
>
> `GEMINI_COOKIE_PATH` 指定容器内保存刷新后 Cookie 的目录。将其挂载（例如 `-v $(pwd)/cache:/app/cache`）可以在容器重建或重启后保留这些 Cookie，避免频繁重新认证。

### 使用 Docker Compose

创建 `docker-compose.yml` 文件：

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config  # Uncomment to use a custom config file
      # - ./certs:/app/certs    # Uncomment to enable HTTPS with your certs
      - ./data:/app/data
      - ./cache:/app/cache
    environment:
      - CONFIG_SERVER__HOST=0.0.0.0
      - CONFIG_SERVER__PORT=8000
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
      - GEMINI_COOKIE_PATH=/app/cache # must match the cache volume mount above
    restart: on-failure:3 # Avoid retrying too many times
```

然后运行：

```bash
docker compose up -d
```

> [!IMPORTANT]
> 请务必挂载 `/app/data` 卷以保证对话数据在容器重启后持久化。
> 同时挂载 `/app/cache`（或与 `GEMINI_COOKIE_PATH` 对应的目录）以保存刷新后的 Cookie，这样在容器重建/重启后无需频繁重新认证。

## 配置说明

服务器读取 `config/config.yaml` 配置文件。

各项配置说明请参见 [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) 文件中的注释。

### 环境变量覆盖

> [!TIP]
> 该功能适用于 Docker 部署和生产环境，可将敏感信息与配置文件分离。

你可以通过带有 `CONFIG_` 前缀的环境变量覆盖任意配置项，嵌套键用双下划线（`__`）分隔，例如：

```bash
# 覆盖服务器设置
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# 覆盖 Client 0 的用户凭据
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# 覆盖 Client 0 的代理设置
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"

# 覆盖 Client 0 的浏览器指纹模拟
export CONFIG_GEMINI__CLIENTS__0__IMPERSONATE="chrome"


# 覆盖对话存储大小限制
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB
```

### 客户端 ID 与会话重用

会话在保存时会绑定创建它的客户端 ID。请在配置中保持这些 `id` 值稳定，
这样在更新 Cookie 列表时依然可以复用旧会话。

### Gemini 凭据

> [!WARNING]
> 请妥善保管这些凭据，切勿提交到版本控制。这些 Cookie 可访问你的 Google 账号。

使用 Gemini-FastAPI 需提取 Gemini 会话 Cookie：

1. 在无痕/隐私窗口打开 [Gemini](https://gemini.google.com/) 并登录
2. 打开开发者工具（F12）
3. 进入 **Application** → **Storage** → **Cookies**
4. 查找并复制以下值：
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!IMPORTANT]
> **请开启 [Gemini Apps 应用活动](https://myactivity.google.com/product/gemini)** 以确保稳定的会话持久化。
>
> 虽然在没有开启该设置的情况下，连续的聊天过程可能暂时正常，但任何瞬时错误、TLS 会话重启或服务器重启都可能导致 Google 端过期的会话元数据。如果该设置被禁用，模型将 **完全丢失多轮对话的上下文**，导致即使本地 LMDB 中存有历史记录，旧对话也将无法继续。

### 代理设置

每个客户端条目可以配置不同的代理，从而规避速率限制。省略 `proxy` 字段或将其设置为 `null` 或空字符串以保持直连。

### 浏览器指纹模拟

每个客户端可以通过 `impersonate` 参数设置 `curl_cffi` 使用的 TLS/HTTP 指纹。

- 设置为 `null`（默认）则使用库的默认值。
- 可设为 [`curl_cffi` 的 `BrowserTypeLiteral`](https://github.com/lexiforest/curl_cffi) 支持的任意值。
- 启动时会校验该值；无效值会阻止服务启动。

```yaml
gemini:
  clients:
    - id: "client-a"
      impersonate: "chrome" # 使用 Chrome 指纹
    - id: "client-b"
      impersonate: null # 使用库默认值
```

### 会话模式

你可以控制请求使用普通的 Google 会话，还是 Google 的临时会话模式：

```yaml
gemini:
  chat_mode: "normal" # "normal"（普通）或 "temporary"（临时）
  max_chars_per_request: 1000000
```

设置为 `temporary` 时，对话不会保存到 Google 账号中。只要 Google 尚未关闭该临时窗口，
临时会话仍然可以继续对话，因此会话重用与会话存储的行为与普通模式完全一致。

当已存储的会话无法再被延续时——例如切换了 `chat_mode`，或 Google 已关闭该临时窗口——
服务会回退到将完整对话历史重放到一个全新的会话中，从而重建上下文，而不是丢失上下文。

每个账号在 Google 侧最多只保留一个处于开启状态的临时窗口：一旦创建新的会话，上一个临时
会话就会被关闭。因此只有最近一次开启的临时会话仍可继续对话。服务会按客户端记录该会话，
并且**只**重用它；任何更早的临时会话都会以完整历史重放到全新会话中。这里没有需要调节的
超时时间——该规则直接依据 Google 的实际行为，而不是靠猜测过期时长。

该记录刻意只保存在内存中，因此只要客户端会话被重新初始化——例如因闲置触发 `auto_close`、
服务重启或重新部署——它同样会被清空。发生上述情况后，服务无法再确认任何窗口仍然有效，
所有已存储的临时会话都会改为重放，而不是重用。

以上规则**仅**在临时模式下生效：普通会话在用户手动删除之前会一直由 Google 保留，
因此其元数据可以长期重用，并且不受重启影响。

> [!WARNING]
> Google 可能在任意时刻、且不作任何提示地关闭临时会话窗口，包括在对话进行到一半时。
> 此时模型可能直接返回不含既有上下文的回复，而不会抛出错误，因此上下文丢失可能是静默的。
> 只要服务能够识别出该会话已失效，就会将完整历史重放到新会话中，但这种识别并非总能成功。
> 对于较长或对上下文较敏感的对话，建议使用 `normal`；`temporary` 的连续性应视为尽力而为。

由于临时会话可接受的负载更小，服务会在标准安全余量的基础上再收紧 10%，
因此有效输入上限为 `max_chars_per_request` 的 81%（而非 90%）。
两种模式下，超出有效上限的输入仍会以 `message.txt` 附件的形式发送。

环境变量等价写法：

```bash
export CONFIG_GEMINI__CHAT_MODE="temporary"
```

### 自定义模型

你可以在 `config/config.yaml` 中或通过环境变量定义自定义模型。

#### YAML 配置

```yaml
gemini:
  model_strategy: "append" # "append" (默认 + 自定义) 或 "overwrite" (仅限自定义)
  models:
    - model_name: "xxx"
      model_header:
        x-goog-ext-525001261-jspb: '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8],null,null,1,null,null,1,1,"EA3C5672-E422-4A5F-BE26-B5B57D3B9AC3"]'
        x-goog-ext-73010989-jspb: "[0]"
        x-goog-ext-73010990-jspb: "[0,0,0]"
```

#### 环境变量

你可以通过 `CONFIG_GEMINI__MODELS` 以 JSON 字符串或列表结构的形式提供模型。这为通过 shell 或在自动化环境（例如 Docker）中覆盖设置提供了一种灵活的方式，而无需修改配置文件。

```bash
export CONFIG_GEMINI__MODEL_STRATEGY="overwrite"
export CONFIG_GEMINI__MODELS='[{"model_name": "xxx", "model_header": {"x-goog-ext-525001261-jspb": "[1,null,null,null,\"fbb127bbb056c959\",null,null,0,\[4,5,6,8\],null,null,1,null,null,1,1,\"EA3C5672-E422-4A5F-BE26-B5B57D3B9AC3\"]", "x-goog-ext-73010989-jspb": "[0]", "x-goog-ext-73010990-jspb": "[0,0,0]"}}]'
```

## 鸣谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - 底层 Gemini Web API 客户端
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - 本项目最初基于此仓库，经过深度重构与工程化改进，现已成为独立项目，并增加了多轮会话复用等新特性。在此表示特别感谢。

## 免责声明

本项目与 Google 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 Google 服务条款。使用风险自负。
