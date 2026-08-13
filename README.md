# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ English | [中文](README.zh.md) ]

Web-based Gemini models wrapped into an OpenAI-compatible API. Powered by [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API).

**Call Gemini's web-based models via API without an API Key, completely free!**

## Features

- **No Google API Key Required**: Use web cookies to freely access Gemini's models via API.
- **Google Search Included**: Get up-to-date answers using web-based Gemini's search capabilities.
- **Conversation Persistence**: LMDB-based storage supporting multi-turn conversations.
- **Multi-modal Support**: Support for handling text, images, and file uploads.
- **Multi-account Load Balancing**: Distribute requests across multiple accounts with per-account proxy settings.

## Quick Start

**For Docker deployment, see the [Docker Deployment](#docker-deployment) section below.**

### Prerequisites

- Python 3.13
- Google account with Gemini access on web (Enable **[Gemini Apps activity](https://myactivity.google.com/product/gemini)** for best conversation persistence)
- `secure_1psid` and `secure_1psidts` cookies from Gemini web interface

### Installation

#### Using uv (Recommended)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### Using pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### Basic Configuration

Edit `config/config.yaml` and provide at least one credential pair:

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # Optional proxy URL (null/empty keeps direct connection)
      impersonate: null # Optional browser impersonation target (null uses library default)
```

> [!NOTE]
> For details, refer to the [Configuration](#configuration) section below.

### Running the Server

```bash
# Using uv
uv run python run.py

# Using Python directly
python run.py
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

The server provides several endpoints, including OpenAI-compatible ones.

### OpenAI-Compatible Endpoints

These endpoints are designed to be compatible with OpenAI's API structure, allowing you to use Gemini as a drop-in replacement.

- **`GET /v1/models`**: Lists all supported Gemini models.
- **`POST /v1/chat/completions`**: Unified chat interface.
  - **Streaming**: Set `stream: true` to receive real-time delta chunks.
  - **Multi-modal**: Supports text, images, and file uploads.
  - **Tool Calling**: Supports function calling via the `tools` parameter.
  - **Structured Output**: Supports `response_format` for JSON schema enforcement.

### Advanced Endpoints

- **`POST /v1/responses`**: An alternative endpoint for complex interaction patterns, supporting rich output items including generated images and tool calls.

### Utility Endpoints

- **`GET /health`**: Health check endpoint. Returns the status of the server, configured Gemini clients, and conversation storage.
- **`GET /media/{filename}`**: Internal endpoint to serve generated media. Requires a valid token (automatically included in image URLs returned by the API).

## Docker Deployment

### Run with Options

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
> Add `CONFIG_GEMINI__CLIENTS__N__PROXY` only if you need a proxy; omit the variable to keep direct connections.
>
> `GEMINI_COOKIE_PATH` points to the directory inside the container where refreshed cookies are stored. Bind-mounting it (e.g. `-v $(pwd)/cache:/app/cache`) preserves those cookies across container rebuilds/recreations so you rarely need to re-authenticate.

### Run with Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config      # Uncomment to use a custom config file
      # - ./certs:/app/certs        # Uncomment to enable HTTPS with your certs
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

Then run:

```bash
docker compose up -d
```

> [!IMPORTANT]
> Make sure to mount the `/app/data` volume to persist conversation data between container restarts.
> Also mount `/app/cache` so refreshed cookies (including rotated 1PSIDTS values) survive container rebuilds/recreates without re-auth.

## Configuration

The server reads a YAML configuration file located at `config/config.yaml`.

For details on each configuration option, refer to the comments in the [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) file.

### Environment Variable Overrides

> [!TIP]
> This feature is particularly useful for Docker deployments and production environments where you want to keep sensitive credentials separate from configuration files.

You can override any configuration option using environment variables with the `CONFIG_` prefix. Use double underscores (`__`) to represent nested keys, for example:

```bash
# Override server settings
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# Override Gemini credentials for client 0
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# Override optional proxy settings for client 0
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"

# Override browser impersonation for client 0
export CONFIG_GEMINI__CLIENTS__0__IMPERSONATE="chrome"


# Override conversation storage size limit
export CONFIG_STORAGE__MAX_SIZE=1073741824  # 1 GB

# Override conversation retention (days before automatic cleanup, 0 disables cleanup)
export CONFIG_STORAGE__RETENTION_DAYS=7
```

> [!NOTE]
> The LMDB map size (`storage.max_size`, default 1 GB) is fixed when the store
> opens. When the data file grows to that ceiling every write fails with
> `MDB_MAP_FULL` — the map never grows on its own. If that happens, raise
> `max_size` and **restart** the server; existing data is preserved. The
> 6-hourly cleanup task (main.py) reclaims space once conversations are older
> than `retention_days` (default 7).

### Client IDs and Conversation Reuse

Conversations are stored with the ID of the client that generated them.
Keep these identifiers stable in your configuration so that sessions remain valid
when you update the cookie list.

### Gemini Credentials

> [!WARNING]
> Keep these credentials secure and never commit them to version control. These cookies provide access to your Google account.

To use Gemini-FastAPI, you need to extract your Gemini session cookies:

1. Open [Gemini](https://gemini.google.com/) in a private/incognito browser window and sign in
2. Open Developer Tools (F12)
3. Navigate to **Application** → **Storage** → **Cookies**
4. Find and copy the values for:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!IMPORTANT]
> **Enable [Gemini Apps activity](https://myactivity.google.com/product/gemini)** to ensure stable conversation persistence.
>
> While active chat turns may work temporarily without it, any transient error, TLS session restart, or server reboot can cause Google to expire the conversation metadata. If this setting is disabled, the model will **completely lose the context of your multi-turn conversation**, making old threads unreachable even if they are stored in your local LMDB.

<!-- Keeps the two callouts as separate blockquotes (markdownlint MD028). -->

> [!TIP]
> For detailed instructions, refer to the [HanaokaYuzu/Gemini-API authentication guide](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication).

### Proxy Settings

Each client entry can be configured with a different proxy to work around rate limits. Omit the `proxy` field or set it to `null` or an empty string to keep a direct connection.

### Browser Impersonation

Each client can optionally set an `impersonate` value to control the TLS/HTTP fingerprint used by `curl_cffi`.

- Set to `null` (default) to use the library's default.
- Set to any value supported by [`curl_cffi`'s `BrowserTypeLiteral`](https://github.com/lexiforest/curl_cffi).
- The value is validated at startup; an invalid value will prevent the server from starting.

```yaml
gemini:
  clients:
    - id: "client-a"
      impersonate: "chrome" # Use Chrome fingerprint
    - id: "client-b"
      impersonate: null # Use library default
```

### Chat Session Mode

You can control whether requests use normal Google chats or Google's temporary chat mode:

```yaml
gemini:
  chat_mode: "normal" # "normal" or "temporary"
  max_chars_per_request: 1000000
```

With `temporary`, conversations are not saved to the Google account. A temporary chat is still
continuable for as long as Google keeps the window open, so session reuse and conversation
storage work exactly as they do in normal mode.

When a stored chat can no longer be continued - after changing `chat_mode`, or once Google has
closed a temporary window - the server falls back to replaying the full conversation history
into a fresh chat, so the context is rebuilt rather than lost.

Google keeps at most one temporary window open per account and closes the previous one as soon
as a new conversation is created, so only the most recently opened temporary chat is still
continuable. The server tracks that chat per client and reuses **only** it; any older temporary
conversation is replayed in full into a fresh chat instead. There is no timeout to tune - the
rule follows Google's actual behaviour rather than guessing at an expiry.

That tracking is deliberately in-memory, so it is also cleared whenever the client session
restarts - an `auto_close` after inactivity, a server restart, or a redeploy. After any of
those, no window can be vouched for and every stored temporary conversation is replayed rather
than reused.

This applies **only** in temporary mode. A normal chat is kept by Google until you delete it,
so its metadata stays reusable indefinitely and across restarts.

> [!WARNING]
> Google can close a temporary chat window at any time, without notice and mid-conversation.
> When that happens the reply may come back without the earlier context instead of raising an
> error, so the loss can be silent. The server replays the full history into a fresh chat when
> it can detect the chat is gone, but detection is not guaranteed. Prefer `normal` for long or
> context-sensitive conversations, and treat `temporary` as best-effort continuity.

Because temporary chats accept a smaller payload, the server applies an additional 10% reduction
on top of the standard safety margin, so the effective input limit becomes 81% of
`max_chars_per_request` instead of 90%. Input exceeding the effective limit is still sent as a
`message.txt` attachment in both modes.

Environment variable equivalent:

```bash
export CONFIG_GEMINI__CHAT_MODE="temporary"
```

### Custom Models

You can define custom models in `config/config.yaml` or via environment variables.

#### YAML Configuration

```yaml
gemini:
  model_strategy: "append" # "append" (default + custom) or "overwrite" (custom only)
  models:
    - model_name: "xxx"
      model_header:
        x-goog-ext-525001261-jspb: '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8],null,null,1,null,null,1,1,"EA3C5672-E422-4A5F-BE26-B5B57D3B9AC3"]'
        x-goog-ext-73010989-jspb: "[0]"
        x-goog-ext-73010990-jspb: "[0,0,0]"
```

#### Environment Variables

You can supply models as a JSON string or list structure via `CONFIG_GEMINI__MODELS`. This provides a flexible way to override settings via the shell or in automated environments (e.g. Docker) without modifying the configuration file.

```bash
export CONFIG_GEMINI__MODEL_STRATEGY="overwrite"
export CONFIG_GEMINI__MODELS='[{"model_name": "xxx", "model_header": {"x-goog-ext-525001261-jspb": "[1,null,null,null,\"fbb127bbb056c959\",null,null,0,\[4,5,6,8\],null,null,1,null,null,1,1,\"EA3C5672-E422-4A5F-BE26-B5B57D3B9AC3\"]", "x-goog-ext-73010989-jspb": "[0]", "x-goog-ext-73010990-jspb": "[0,0,0]"}}]'
```

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
