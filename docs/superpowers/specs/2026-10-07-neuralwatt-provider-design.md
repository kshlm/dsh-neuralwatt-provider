# Neuralwatt Provider Plugin — Design

**Package:** `dsh-llm-neuralwatt` (repo folder `dsh-neuralwatt-provider`)
**Plugin `name`:** `llm-neuralwatt`
**Provider route:** `neuralwatt` (selectable as `neuralwatt/<model>`)
**Date:** 2026-10-07
**Status:** Approved design

## Goal

A DeepSeek Harness plugin (`dsh-plugin`) that implements an LLM provider for
[Neuralwatt](https://docs.neuralwatt.com/api/overview). It must:

1. **Dynamically populate models** from Neuralwatt's `GET /v1/models` endpoint (no
   hand-written model list).
2. **Set each model's capabilities correctly** — input modalities (text / image),
   context window, max output tokens, reasoning efforts — from the endpoint's rich
   `metadata`.
3. **Allow the base URL to be overwritten** (config + env fallback), so it works
   against a proxy/gateway or self-hosted deployment.

Neuralwatt is **OpenAI-compatible** (base `https://api.neuralwatt.com/v1`), so the
provider speaks `/chat/completions` (SSE) for inference and `/models` for discovery.

## Non-goals

- No live/periodic model refresh — **boot snapshot only** (fetch once at load).
- No filtering of tier variants (`-flex`/`-speed`/`-fast`); expose **all** models,
  skipping only ones flagged `deprecated: true`.
- No Anthropic-Messages or Responses protocol — only OpenAI-compatible
  chat completions.
- No reimplementation of the `llm-pi-ai` streaming stack; this plugin owns the
  `neuralwatt` route directly.

## Architecture

A self-contained `LlmAdapter` that **owns** the `neuralwatt` route and registers
itself on the harness LLM seam (`ctx.llm.registerAdapter(['neuralwatt'], adapter)`).
This is the documented "LLM adapter" integration path, standing entirely on the
public `@deepseek-ai/dsh-llm` contract.

- **Boot:** `apply` → resolve effective config → start `GET {baseURL}/models`
  (memoized, non-blocking) → map each model's `metadata` to capabilities →
  `ctx.llm.registerAdapter(['neuralwatt'], adapter)`.
- **Select:** `listModels` / `resolveModel` serve the mapped catalog with correct
  `name`, `inputModalities`, `context.contextWindow`, `defaultMaxTokens`, `reasoning`.
- **Chat:** `stream()` → `POST {baseURL}/chat/completions` (stream) → SSE →
  `StreamChunk`.

> **Install note:** this replaces the hand-written `neuralwatt` block under
> `llm-pi-ai.providers` in `cordis.patch.yml`. Remove that block when installing, or
> the two `neuralwatt` routes collide.

### Why a standalone adapter (not config injection into `llm-pi-ai`)

`llm-pi-ai`'s `PiAiAdapter` and its resolved-profile construction (`resolveProfiles`,
`buildProvider`) are **not** cleanly importable from outside the package, so wrapping
it requires reaching into private internals. A standalone adapter instead uses only
the public `LlmAdapter` + `registerAdapter` seam, gives exact control of capability
mapping, and is independently testable. The streaming translation is bounded and the
protocol is documented.

## Components

| File | Purpose |
|---|---|
| `src/index.ts` | Plugin entry: `name`, `inject: ['llm']`, `Config`, `apply()` → register adapter. |
| `src/config.ts` | Schemastery `Config` schema + effective-config resolution (baseURL / API-key with env fallback). |
| `src/models.ts` | `/v1/models` response types, discovery fetch, **capability mapping**. |
| `src/adapter.ts` | `NeuralwattAdapter extends LlmAdapter` — `providerInfo`, `listModels`, `resolveModel`, `prepareCall`, `stream`. |
| `src/openai.ts` | Build the chat-completions request; parse SSE → `StreamChunk`. |
| `src/credentials.ts` | `resolveApiKey` via `ctx.credentials` → `process.env` fallback. |
| `src/errors.ts` | `LlmError` helpers + stable error codes. |
| `test/*.test.ts` | Unit tests (mapping, SSE, config, adapter against a mock server). |

### Config surface (`src/config.ts`)

Each field has a built-in default, an optional env fallback, and **config wins over
env** (env wins over the built-in default):

| Field | Default | Env fallback | Meaning |
|---|---|---|---|
| `route` | `neuralwatt` | — | Provider route key / selector id. |
| `displayName` | `Neuralwatt` | — | Human-readable provider name. |
| `baseURL` | `https://api.neuralwatt.com/v1` | `NEURALWATT_BASE_URL` | Endpoint root; **overridable**. |
| `apiKeyEnv` | `NEURALWATT_API_KEY` | — | The env-var *name* holding the API key (never the secret). |

Resolution: `baseURL = config.baseURL ?? env.NEURALWATT_BASE_URL ?? 'https://api.neuralwatt.com/v1'`.
Fields are left **optional in the schema** (no schema-level default) so an explicit
config value, an env value, and the built-in default resolve in that precedence order.

## Capability mapping (`src/models.ts`) — core requirement

`GET {baseURL}/models` → `{ object: 'list', data: [ { id, metadata } ] }`. Per model,
**skip** entries with `metadata.deprecated === true`; preserve endpoint order.

| Harness field | Neuralwatt source |
|---|---|
| `id` | `entry.id` |
| `name` | `metadata.display_name` ?? `id` |
| `description` | `metadata.description` |
| `inputModalities` | `['text']` + (`metadata.capabilities.vision` ? `['image']` : `[]`) |
| `context.contextWindow` | `metadata.limits.max_context_length` ?? `entry.max_model_len` ?? `262144` |
| `defaultMaxTokens` | `metadata.limits.max_output_tokens` ?? `32768` |
| `reasoning.efforts` | `metadata.reasoning.accepted_efforts`, normalized via `effort_aliases` (`max→xhigh`, `high→xhigh`, `minimal→low`) → `ReasoningEffortId` + display names |
| `reasoning.defaultEffort` | `metadata.reasoning.default_effort`; omit `reasoning` entirely when `capabilities.reasoning` is false |

Tool support (`capabilities.tools`) gates whether `tools` are forwarded and
`tool-call` blocks emitted. Text-only models advertise `['text']`; vision models
advertise `['text','image']`. The seam's modality vocabulary is `text | image`, so
`capabilities.audio_input` is not claimed as a modality.

Fallbacks keep the mapping total: an absent `metadata` still yields a usable
text-only model with default capacities rather than dropping the entry.

## Data flow (request)

`GenerateOptions` → build an OpenAI chat-completions body:

- `messages`: `system` → system slot; conversation `messages` mapped to
  `system`/`user`/`assistant`/`tool` roles. `ImageBlock` content → OpenAI
  `image_url` data URLs, resolved through the attachment service
  (`ctx.get('attachments')` + `resolveImageAccess`), mirroring `llm-pi-ai`.
- `tools`: `ToolSchema[]` → OpenAI `tools` (function declarations).
- `reasoning_effort`: from `GenerateOptions.reasoningEffort`.
- `max_tokens`, `temperature`, `stop`: passed through.
- `stream: true`, `stream_options: { include_usage: true }`.

Request headers: `content-type: application/json`, `authorization: Bearer <key>`,
`...attributionHeaders()`; `options.signal` forwarded. URL: `POST {baseURL}/chat/completions`.

SSE → `StreamChunk` translation:

| SSE | StreamChunk |
|---|---|
| `delta.content` | `block-start` (first) + `text-delta` |
| `delta.reasoning` / `reasoning_content` | `reasoning-delta` |
| `delta.tool_calls[].function.{name,arguments}` | `block-start tool-call` + `tool-call-delta` (incremental `argumentsDelta`) |
| final `usage` chunk | `usage` (`inputTokens`/`outputTokens`/`totalTokens`/cache/reasoning) |
| `finish_reason` | `finish` (`stop` / `tool-calls` / `max-tokens`) |

Each `block-start` is paired with a `block-end` carrying the assembled `ContentBlock`.
`usage` is emitted before the terminal `finish`.

## Error handling

- **Discovery failure at boot:** log a warning and register with an **empty** catalog;
  `listModels` returns `[]`, `resolveModel` throws `UNKNOWN_MODEL` with an actionable
  message (check baseURL / API key / network). Never silently guess capabilities.
- **HTTP non-2xx** → `LlmError(msg, 'PROVIDER_HTTP_ERROR')`.
- **Missing / unusable key** → `LlmError(..., 'MISSING_CREDENTIAL')`; validate with
  `assertUsableApiKey` and never echo the secret.
- **Malformed SSE / payload** → `LlmError(..., 'PROVIDER_PROTOCOL_ERROR')`.
- **Unknown model** → `LlmError(..., 'UNKNOWN_MODEL')`.
- Map `finish_reason: length` (and zero-output `length` usage) to the `max-tokens`
  finish reason so context overflow is reported correctly.
- Every request merges `attributionHeaders()` and forwards `options.signal`.

## Testing

- **`models.test.ts`** — capability mapping fixtures: vision→image, context/output
  sizes, reasoning-effort aliasing + default, deprecated skip, `display_name`
  fallback, text-only model.
- **`openai-stream.test.ts`** — SSE→`StreamChunk`: text, reasoning, tool-call deltas,
  usage, finish reasons, error/abort normalization.
- **`config.test.ts`** — baseURL resolution precedence (config > env > default),
  `apiKeyEnv` default.
- **`adapter.test.ts`** — mock OpenAI-compatible server; full `stream()` chunk
  sequence, request headers (`authorization`, `attribution`), signal forwarding.
- **Manual:** `dsh plugin add dsh-llm-neuralwatt` into a profile, boot, confirm the
  model picker lists Neuralwatt models with correct context/modalities and a chat
  round-trips.

## Packaging

- `package.json`: `type: module`, `main: lib/index.js`, `types: lib/types/index.d.ts`.
- `tsconfig.json`; build with `tsc`.
- Peer deps: `@deepseek-ai/dsh-llm`, `@deepseek-ai/cordis`, `@deepseek-ai/schemastery`.
- Install: `dsh plugin --profile <p> add dsh-llm-neuralwatt`; during development a
  `cordis.yml` `insert` entry pointing at the built `lib/index.js`.
