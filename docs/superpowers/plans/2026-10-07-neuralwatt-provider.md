# Neuralwatt Provider Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `dsh-llm-neuralwatt`, a DeepSeek Harness plugin that implements a Neuralwatt LLM provider: it dynamically populates models from `GET /v1/models`, sets each model's capabilities correctly (input modalities, context window, max output, reasoning), and lets the base URL be overridden.

**Architecture:** A self-contained `LlmAdapter` subclass owning the `neuralwatt` provider route. On load it discovers models once (boot snapshot) by fetching `GET {baseURL}/models` and mapping Neuralwatt's `metadata` to harness model descriptors. On request it speaks Neuralwatt's OpenAI-compatible `POST {baseURL}/chat/completions` (SSE) and translates the stream into the harness `StreamChunk` protocol. Pure mapping/codec logic lives in dependency-light modules so it is unit-testable without a live harness.

**Tech Stack:** TypeScript (ESM), `tsc` build, `vitest` tests, `@deepseek-ai/dsh-llm` (LlmAdapter/StreamChunk/LlmError), `@deepseek-ai/dsh-brand` (branded ids), `@deepseek-ai/schemastery` (Config schema), `@deepseek-ai/cordis` (plugin Context), `@deepseek-ai/dsh-attachment` (image refs). All `@deepseek-ai/*` resolve from npm.

## Global Constraints

- Provider route key default: `neuralwatt`; plugin `name`: `llm-neuralwatt`; package name: `dsh-llm-neuralwatt`.
- Base URL resolution precedence: `config.baseURL` > `env.NEURALWATT_BASE_URL` > `https://api.neuralwatt.com/v1`. The env var is `NEURALWATT_BASE_URL`.
- API key env-var name default: `NEURALWATT_API_KEY` (config field `apiKeyEnv` names the var; never store or echo the secret).
- Model refresh: **boot snapshot only** — fetch `/v1/models` once (memoized); no timers, no periodic refresh.
- Model filtering: expose **all** models, skipping only entries with `metadata.deprecated === true`.
- Modalities vocabulary is `text | image`. Advertise `['text']`, plus `'image'` iff `metadata.capabilities.vision` is truthy. Never claim `audio`.
- Context default `262144`; max-output default `32768` (used only when the endpoint omits the value).
- Every provider HTTP request must merge `attributionHeaders()` and forward `options.signal`.
- Errors surface as `LlmError` with stable codes: `PROVIDER_HTTP_ERROR`, `PROVIDER_PROTOCOL_ERROR`, `MISSING_CREDENTIAL`, `UNKNOWN_MODEL`.
- Build output: `lib/index.js` + `lib/types/**`; `package.json` `type: module`.
- Commit after every green test cycle (`git -c commit.gpgsign=false …` — GPG signing is unavailable in this environment).

---

## File Structure

| File | Responsibility |
|---|---|
| `package.json` | Package metadata, deps, `main`/`types`, scripts. |
| `tsconfig.json` | Strict ESM TypeScript build config. |
| `vitest.config.ts` | Test runner config. |
| `src/config.ts` | `Config` schema + `resolveEffectiveConfig()` (baseURL/apiKey/env precedence). |
| `src/models.ts` | `/v1/models` response types + `mapModel`/`mapModels` capability mapping + `discoverModels` fetch. |
| `src/errors.ts` | `LlmError` constructor helpers + stable codes. |
| `src/openai.ts` | `buildChatRequestBody()` (messages/tools/params + image data-URL) and `createSseParser()` (SSE→`StreamChunk`). |
| `src/credentials.ts` | `makeResolveApiKey()` — credentials seam → env fallback. |
| `src/adapter.ts` | `NeuralwattAdapter extends LlmAdapter` — `providerInfo`/`listModels`/`resolveModel`/`prepareCall`/`stream`. |
| `src/index.ts` | Plugin entry: `name`, `inject`, `Config`, `apply()` → register adapter + wire credentials/image access. |
| `test/*.test.ts` | Unit + mock-server tests. |
| `README.md` | Install + config docs. |

**Interface contracts (referenced by every task):**

- `config.ts`: `Config { route?: string; displayName?: string; baseURL?: string; apiKeyEnv?: string }`; `EffectiveConfig { route: string; displayName: string; baseURL: string; apiKeyEnv: string }`; `resolveEffectiveConfig(config, env?): EffectiveConfig`; consts `DEFAULT_BASE_URL`, `DEFAULT_ROUTE`, `DEFAULT_DISPLAY_NAME`, `DEFAULT_API_KEY_ENV`.
- `models.ts`: `ModelDescriptor { id; name; description?; inputModalities: ModelModality[]; contextWindow: number; maxTokens: number; tools: boolean; reasoning?: { efforts: {id:string;name:string}[]; defaultEffort?: string } }`; `mapModel(entry): ModelDescriptor | null`; `mapModels(res): ModelDescriptor[]`; `discoverModels(baseURL, apiKey?, fetchFn?, signal?): Promise<ModelDescriptor[]>`.
- `errors.ts`: `httpError(status, detail)`, `protocolError(detail)`, `missingCredential(ref)`, `unknownModel(route, model, detail?)`, `discoveryError(detail)` → all return `LlmError`.
- `openai.ts`: `buildChatRequestBody(options: GenerateOptions, hooks: { imageDataUrl?: (attachment) => Promise<string|undefined> }, defaults: { maxOutputTokens?: number }): ChatRequestBody`; `createSseParser(): SseParser` where `SseParser { push(text: string): StreamChunk[]; end(): StreamChunk[] }`.
- `credentials.ts`: `ResolveApiKey = () => Promise<string|undefined>`; `makeResolveApiKey(opts: { apiKeyEnv: string; resolveCredential?: (ref: string) => Promise<string|undefined>; env?: NodeJS.ProcessEnv }): ResolveApiKey`.
- `adapter.ts`: `NeuralwattAdapter` ctor `(opts: { route; displayName; baseURL; apiKeyEnv; resolveApiKey: ResolveApiKey; imageDataUrl?: (attachment)=>Promise<string|undefined>; discover?: () => Promise<ModelDescriptor[]> })`.

---

### Task 0: Project scaffold + dependency install

**Files:**
- Create: `package.json`, `tsconfig.json`, `vitest.config.ts`, `.gitignore`

**Interfaces:**
- Produces: the build/test toolchain every later task runs (`pnpm test`, `pnpm build`, `pnpm exec tsc --noEmit`).

- [ ] **Step 1: Write `package.json`**

```json
{
  "name": "dsh-llm-neuralwatt",
  "version": "0.1.0",
  "description": "Neuralwatt LLM provider plugin for DeepSeek Harness — dynamic model discovery with capability mapping over an OpenAI-compatible API",
  "type": "module",
  "main": "lib/index.js",
  "types": "lib/types/index.d.ts",
  "exports": {
    ".": { "types": "./lib/types/index.d.ts", "default": "./lib/index.js" },
    "./package.json": "./package.json"
  },
  "files": ["lib", "README.md"],
  "license": "MIT",
  "scripts": {
    "build": "tsc -p tsconfig.json",
    "typecheck": "tsc -p tsconfig.json --noEmit",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@deepseek-ai/dsh-brand": "^0.2.0-rc.2",
    "@deepseek-ai/dsh-llm": "^0.2.0-rc.2",
    "@deepseek-ai/schemastery": "^3.18.4"
  },
  "peerDependencies": {
    "@deepseek-ai/cordis": "~4.0.4",
    "@deepseek-ai/dsh-attachment": "^0.2.0-rc.2"
  },
  "devDependencies": {
    "@deepseek-ai/cordis": "~4.0.4",
    "@deepseek-ai/dsh-attachment": "^0.2.0-rc.2",
    "typescript": "^5.6.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "declaration": true,
    "declarationDir": "lib/types",
    "outDir": "lib",
    "rootDir": "src",
    "skipLibCheck": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*.ts"]
}
```

- [ ] **Step 3: Write `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'
export default defineConfig({
  test: { include: ['test/**/*.test.ts'], environment: 'node' },
})
```

- [ ] **Step 4: Write `.gitignore`**

```
node_modules/
lib/
*.tsbuildinfo
```

- [ ] **Step 5: Install dependencies and verify the seam resolves**

Run: `pnpm install`
Expected: installs with no peer-resolution errors.

Run: `pnpm exec tsc --noEmit --skipLibCheck -e "import type { LlmAdapter } from '@deepseek-ai/dsh-llm'"` (or a one-line `src/_smoke.ts` importing `LlmAdapter`, `StreamChunk`, `brandString`, then delete it).
Expected: `@deepseek-ai/dsh-llm`, `@deepseek-ai/dsh-brand`, `@deepseek-ai/schemastery` all resolve types.

- [ ] **Step 6: Commit**

```bash
git -c commit.gpgsign=false add package.json tsconfig.json vitest.config.ts .gitignore pnpm-lock.yaml
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "chore: scaffold dsh-llm-neuralwatt package"
```

---

### Task 1: Effective-config resolution (`src/config.ts`)

**Files:**
- Create: `src/config.ts`
- Test: `test/config.test.ts`

**Interfaces:**
- Consumes: none.
- Produces: `resolveEffectiveConfig(config, env?): EffectiveConfig`, `Config` (Schemastery schema + TS type), `DEFAULT_BASE_URL`, `DEFAULT_ROUTE`, `DEFAULT_DISPLAY_NAME`, `DEFAULT_API_KEY_ENV`.

- [ ] **Step 1: Write the failing test `test/config.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import {
  DEFAULT_API_KEY_ENV,
  DEFAULT_BASE_URL,
  DEFAULT_DISPLAY_NAME,
  DEFAULT_ROUTE,
  resolveEffectiveConfig,
} from '../src/config.ts'

describe('resolveEffectiveConfig', () => {
  it('uses built-in defaults when nothing is set', () => {
    const eff = resolveEffectiveConfig({}, {})
    expect(eff).toEqual({
      route: DEFAULT_ROUTE,
      displayName: DEFAULT_DISPLAY_NAME,
      baseURL: DEFAULT_BASE_URL,
      apiKeyEnv: DEFAULT_API_KEY_ENV,
    })
  })

  it('prefers config.baseURL over env over default', () => {
    expect(resolveEffectiveConfig({ baseURL: 'https://cfg.example/v1' }, { NEURALWATT_BASE_URL: 'https://env.example/v1' }).baseURL)
      .toBe('https://cfg.example/v1')
    expect(resolveEffectiveConfig({}, { NEURALWATT_BASE_URL: 'https://env.example/v1' }).baseURL)
      .toBe('https://env.example/v1')
    expect(resolveEffectiveConfig({}, {}).baseURL).toBe(DEFAULT_BASE_URL)
  })

  it('honors route, displayName, apiKeyEnv overrides', () => {
    const eff = resolveEffectiveConfig({ route: 'nw', displayName: 'NW', apiKeyEnv: 'MY_KEY' }, {})
    expect(eff.route).toBe('nw')
    expect(eff.displayName).toBe('NW')
    expect(eff.apiKeyEnv).toBe('MY_KEY')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/config.test.ts`
Expected: FAIL — `Cannot find module '../src/config.ts'`.

- [ ] **Step 3: Write minimal implementation `src/config.ts`**

```ts
import Schema from '@deepseek-ai/schemastery'

/** Plugin configuration; every field is optional and resolved against env + defaults. */
export interface Config {
  /** Provider route key and selector id. */
  route?: string
  /** Human-readable provider name. */
  displayName?: string
  /** Endpoint root; overrides the env fallback and the built-in default. */
  baseURL?: string
  /** Name of the environment variable that holds the API key (never the secret). */
  apiKeyEnv?: string
}

export const Config: Schema<Config> = Schema.object({
  route: Schema.string(),
  displayName: Schema.string(),
  baseURL: Schema.string(),
  apiKeyEnv: Schema.string(),
})

export const DEFAULT_BASE_URL = 'https://api.neuralwatt.com/v1'
export const DEFAULT_ROUTE = 'neuralwatt'
export const DEFAULT_DISPLAY_NAME = 'Neuralwatt'
export const DEFAULT_API_KEY_ENV = 'NEURALWATT_API_KEY'

/** Fully-resolved, non-optional provider configuration. */
export interface EffectiveConfig {
  route: string
  displayName: string
  baseURL: string
  apiKeyEnv: string
}

/** Resolve config with precedence: config > env > built-in default. */
export function resolveEffectiveConfig(
  config: Config,
  env: NodeJS.ProcessEnv = process.env,
): EffectiveConfig {
  return {
    route: config.route ?? DEFAULT_ROUTE,
    displayName: config.displayName ?? DEFAULT_DISPLAY_NAME,
    baseURL: config.baseURL ?? env.NEURALWATT_BASE_URL ?? DEFAULT_BASE_URL,
    apiKeyEnv: config.apiKeyEnv ?? DEFAULT_API_KEY_ENV,
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/config.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/config.ts test/config.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: effective-config resolution with baseURL override"
```

---

### Task 2: Capability mapping (`src/models.ts`)

**Files:**
- Create: `src/models.ts`
- Test: `test/models.test.ts`

**Interfaces:**
- Consumes: none (pure).
- Produces: `NeuralwattModelEntry`, `NeuralwattModelsResponse`, `ModelDescriptor`, `mapModel(entry): ModelDescriptor | null`, `mapModels(res): ModelDescriptor[]`, `discoverModels(baseURL, apiKey?, fetchFn?, signal?): Promise<ModelDescriptor[]>`, `DEFAULT_CONTEXT_WINDOW`, `DEFAULT_MAX_TOKENS`.

- [ ] **Step 1: Write the failing test `test/models.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { mapModel, mapModels, discoverModels, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS } from '../src/models.ts'

const visionEntry = {
  id: 'qwen-3.8-27b',
  max_model_len: 262128,
  metadata: {
    display_name: 'Qwen 3.8 27B',
    description: 'dense 27B VL model',
    deprecated: false,
    capabilities: { tools: true, vision: true, reasoning: true },
    limits: { max_context_length: 262128, max_output_tokens: 131072 },
    reasoning: {
      accepted_efforts: ['max', 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'],
      default_effort: 'xhigh',
      effort_aliases: { max: 'xhigh', high: 'xhigh', minimal: 'low' },
    },
  },
}
const textOnlyEntry = {
  id: 'nw-small',
  metadata: {
    display_name: 'NW Small',
    capabilities: { tools: false, vision: false, reasoning: false },
    limits: { max_context_length: 32768, max_output_tokens: 4096 },
  },
}
const deprecatedEntry = { id: 'old-model', metadata: { deprecated: true, display_name: 'Old' } }
const bareEntry = { id: 'bare-model' }

describe('mapModel', () => {
  it('maps vision + reasoning + capacities', () => {
    const m = mapModel(visionEntry)!
    expect(m.id).toBe('qwen-3.8-27b')
    expect(m.name).toBe('Qwen 3.8 27B')
    expect(m.description).toBe('dense 27B VL model')
    expect(m.inputModalities).toEqual(['text', 'image'])
    expect(m.contextWindow).toBe(262128)
    expect(m.maxTokens).toBe(131072)
    expect(m.tools).toBe(true)
    // aliases folded: max/high -> xhigh, minimal -> low, 'none' -> off
    expect(m.reasoning!.defaultEffort).toBe('xhigh')
    const ids = m.reasoning!.efforts.map((e) => e.id)
    expect(ids).toEqual(['xhigh', 'medium', 'low', 'off'])
  })

  it('maps a text-only model with no reasoning', () => {
    const m = mapModel(textOnlyEntry)!
    expect(m.inputModalities).toEqual(['text'])
    expect(m.contextWindow).toBe(32768)
    expect(m.maxTokens).toBe(4096)
    expect(m.tools).toBe(false)
    expect(m.reasoning).toBeUndefined()
  })

  it('drops deprecated models', () => {
    expect(mapModel(deprecatedEntry)).toBeNull()
  })

  it('falls back to defaults for a bare entry', () => {
    const m = mapModel(bareEntry)!
    expect(m.name).toBe('bare-model')
    expect(m.inputModalities).toEqual(['text'])
    expect(m.contextWindow).toBe(DEFAULT_CONTEXT_WINDOW)
    expect(m.maxTokens).toBe(DEFAULT_MAX_TOKENS)
  })
})

describe('mapModels', () => {
  it('keeps endpoint order and filters deprecated', () => {
    const list = mapModels({ data: [visionEntry, deprecatedEntry, textOnlyEntry] })
    expect(list.map((m) => m.id)).toEqual(['qwen-3.8-27b', 'nw-small'])
  })
})

describe('discoverModels', () => {
  it('fetches /models with bearer auth and maps the list', async () => {
    const calls: { url: string; auth?: string }[] = []
    const fetchFn = async (url: any, init: any) => {
      calls.push({ url: String(url), auth: init?.headers?.Authorization })
      return new Response(JSON.stringify({ object: 'list', data: [visionEntry] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }
    const models = await discoverModels('https://api.example/v1', 'sk-test', fetchFn as any)
    expect(models.map((m) => m.id)).toEqual(['qwen-3.8-27b'])
    expect(calls[0].url).toBe('https://api.example/v1/models')
    expect(calls[0].auth).toBe('Bearer sk-test')
  })

  it('omits auth header when no key is supplied', async () => {
    let auth: string | undefined
    const fetchFn = async (_url: any, init: any) => {
      auth = init?.headers?.Authorization
      return new Response(JSON.stringify({ data: [] }), { status: 200 })
    }
    await discoverModels('https://api.example/v1', undefined, fetchFn as any)
    expect(auth).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/models.test.ts`
Expected: FAIL — `Cannot find module '../src/models.ts'`.

- [ ] **Step 3: Write minimal implementation `src/models.ts`**

```ts
import type { ModelModality } from '@deepseek-ai/dsh-llm'

export const DEFAULT_CONTEXT_WINDOW = 262144
export const DEFAULT_MAX_TOKENS = 32768

/** Raw `/v1/models` entry (the fields this plugin reads). */
export interface NeuralwattModelEntry {
  id: string
  max_model_len?: number
  metadata?: {
    display_name?: string
    description?: string
    deprecated?: boolean
    capabilities?: { tools?: boolean; vision?: boolean; reasoning?: boolean }
    limits?: { max_context_length?: number; max_output_tokens?: number }
    reasoning?: {
      accepted_efforts?: string[]
      default_effort?: string
      effort_aliases?: Record<string, string>
    }
  }
}

export interface NeuralwattModelsResponse {
  object?: string
  data: NeuralwattModelEntry[]
}

export interface ModelReasoningEffort {
  id: string
  name: string
}

/** Harness-facing model descriptor produced by the mapping. */
export interface ModelDescriptor {
  id: string
  name: string
  description?: string
  inputModalities: ModelModality[]
  contextWindow: number
  maxTokens: number
  tools: boolean
  reasoning?: { efforts: ModelReasoningEffort[]; defaultEffort?: string }
}

/** Canonical effort ids the harness seam understands, mapped from Neuralwatt names/aliases. */
const EFFORT_NAME: Record<string, string> = {
  off: 'Off',
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
}

/** Fold one effort through `effort_aliases`, then normalize to a canonical id. */
function normalizeEffort(raw: string, aliases: Record<string, string> = {}): string | null {
  const folded = aliases[raw] ?? raw
  if (folded === 'none' || folded === 'off') return 'off'
  if (folded in EFFORT_NAME) return folded
  return null
}

/** Map one endpoint entry to a descriptor; null when the model is deprecated. */
export function mapModel(entry: NeuralwattModelEntry): ModelDescriptor | null {
  const md = entry.metadata
  if (md?.deprecated === true) return null

  const caps = md?.capabilities ?? {}
  const limits = md?.limits ?? {}
  const inputModalities: ModelModality[] = ['text']
  if (caps.vision === true) inputModalities.push('image')

  const contextWindow =
    limits.max_context_length ?? entry.max_model_len ?? DEFAULT_CONTEXT_WINDOW
  const maxTokens = limits.max_output_tokens ?? DEFAULT_MAX_TOKENS

  const descriptor: ModelDescriptor = {
    id: entry.id,
    name: md?.display_name ?? entry.id,
    inputModalities,
    contextWindow,
    maxTokens,
    tools: caps.tools === true,
  }
  if (md?.description !== undefined) descriptor.description = md.description

  if (caps.reasoning === true) {
    const aliases = md?.reasoning?.effort_aliases ?? {}
    const seen = new Set<string>()
    const efforts: ModelReasoningEffort[] = []
    for (const raw of md?.reasoning?.accepted_efforts ?? []) {
      const id = normalizeEffort(raw, aliases)
      if (id === null || seen.has(id)) continue
      seen.add(id)
      efforts.push({ id, name: EFFORT_NAME[id] })
    }
    if (efforts.length > 0) {
      const defaultRaw = md?.reasoning?.default_effort
      const defaultEffort = defaultRaw === undefined ? undefined : normalizeEffort(defaultRaw, aliases) ?? undefined
      descriptor.reasoning = defaultEffort === undefined ? { efforts } : { efforts, defaultEffort }
    }
  }
  return descriptor
}

/** Map a full `/models` response, preserving endpoint order and dropping deprecated entries. */
export function mapModels(response: NeuralwattModelsResponse): ModelDescriptor[] {
  const out: ModelDescriptor[] = []
  for (const entry of response.data ?? []) {
    const mapped = mapModel(entry)
    if (mapped !== null) out.push(mapped)
  }
  return out
}

/** Fetch and map `GET {baseURL}/models`. Pass `fetchFn` for tests. */
export async function discoverModels(
  baseURL: string,
  apiKey?: string,
  fetchFn: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<ModelDescriptor[]> {
  const url = `${baseURL.replace(/\/+$/, '')}/models`
  const headers: Record<string, string> = { accept: 'application/json' }
  if (apiKey !== undefined && apiKey.length > 0) headers.authorization = `Bearer ${apiKey}`
  const res = await fetchFn(url, { method: 'GET', headers, ...(signal ? { signal } : {}) })
  if (!res.ok) {
    throw new Error(`Neuralwatt /models returned ${res.status}`)
  }
  const body = (await res.json()) as NeuralwattModelsResponse
  return mapModels(body)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/models.test.ts`
Expected: PASS (all mapping + discovery tests).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/models.ts test/models.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: Neuralwatt model capability mapping"
```

---

### Task 3: Error helpers (`src/errors.ts`)

**Files:**
- Create: `src/errors.ts`
- Test: `test/errors.test.ts`

**Interfaces:**
- Consumes: `LlmError` from `@deepseek-ai/dsh-llm`.
- Produces: `httpError(status, detail)`, `protocolError(detail)`, `missingCredential(ref)`, `unknownModel(route, model, detail?)`, `discoveryError(detail)`; `Codes` const.

- [ ] **Step 1: Write the failing test `test/errors.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { LlmError } from '@deepseek-ai/dsh-llm'
import { Codes, discoveryError, httpError, missingCredential, protocolError, unknownModel } from '../src/errors.ts'

describe('error helpers', () => {
  it('builds a provider HTTP error with a stable code', () => {
    const e = httpError(503, 'upstream down')
    expect(e).toBeInstanceOf(LlmError)
    expect(e.code).toBe(Codes.PROVIDER_HTTP_ERROR)
    expect(e.message).toContain('503')
    expect(e.message).toContain('upstream down')
  })
  it('builds a protocol error', () => {
    expect(protocolError('bad sse').code).toBe(Codes.PROVIDER_PROTOCOL_ERROR)
  })
  it('builds a missing-credential error naming the ref', () => {
    const e = missingCredential('NEURALWATT_API_KEY')
    expect(e.code).toBe(Codes.MISSING_CREDENTIAL)
    expect(e.message).toContain('NEURALWATT_API_KEY')
  })
  it('builds an unknown-model error', () => {
    const e = unknownModel('neuralwatt', 'nope', 'not in catalog')
    expect(e.code).toBe(Codes.UNKNOWN_MODEL)
    expect(e.message).toContain('nope')
  })
  it('builds a discovery error', () => {
    expect(discoveryError('boom').code).toBe(Codes.PROVIDER_HTTP_ERROR)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/errors.test.ts`
Expected: FAIL — `Cannot find module '../src/errors.ts'`.

- [ ] **Step 3: Write minimal implementation `src/errors.ts`**

```ts
import { LlmError } from '@deepseek-ai/dsh-llm'

export const Codes = {
  PROVIDER_HTTP_ERROR: 'PROVIDER_HTTP_ERROR',
  PROVIDER_PROTOCOL_ERROR: 'PROVIDER_PROTOCOL_ERROR',
  MISSING_CREDENTIAL: 'MISSING_CREDENTIAL',
  UNKNOWN_MODEL: 'UNKNOWN_MODEL',
} as const

export function httpError(status: number, detail: string): LlmError {
  return new LlmError(`Neuralwatt API error: ${status} ${detail}`, Codes.PROVIDER_HTTP_ERROR)
}

export function protocolError(detail: string): LlmError {
  return new LlmError(`Neuralwatt protocol error: ${detail}`, Codes.PROVIDER_PROTOCOL_ERROR)
}

export function missingCredential(ref: string): LlmError {
  return new LlmError(
    `Neuralwatt: no API key; this route resolves ${ref}, which is not set — store ${ref} through the credentials service or export it`,
    Codes.MISSING_CREDENTIAL,
  )
}

export function unknownModel(route: string, model: string, detail?: string): LlmError {
  return new LlmError(
    `Neuralwatt route "${route}" has no model "${model}"${detail === undefined ? '' : ` (${detail})`}`,
    Codes.UNKNOWN_MODEL,
  )
}

export function discoveryError(detail: string): LlmError {
  return new LlmError(`Neuralwatt model discovery failed: ${detail}`, Codes.PROVIDER_HTTP_ERROR)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/errors.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/errors.ts test/errors.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: LlmError helpers with stable codes"
```

---

### Task 4: Chat request builder (`src/openai.ts` — request half)

**Files:**
- Create: `src/openai.ts`
- Test: `test/openai-request.test.ts`

**Interfaces:**
- Consumes: `GenerateOptions`, `RequestMessage`, `ContentBlock`, `ImageAttachmentRef`, `textOnlyImageText`, `offloadedImageText` from `@deepseek-ai/dsh-llm`.
- Produces: `ChatRequestBody`, `BuildHooks`, `buildChatRequestBody(options, hooks, defaults)`. (Task 5 adds `createSseParser` to the same file.)

- [ ] **Step 1: Write the failing test `test/openai-request.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { buildChatRequestBody } from '../src/openai.ts'

const baseOptions = {
  provider: 'neuralwatt',
  model: 'qwen-3.8-27b',
  messages: [
    { role: 'system', content: 'be brief' },
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'hello' }] },
    { role: 'tool', toolCallId: 'call-1', content: [{ type: 'text', text: 'result' }] },
  ],
} as any

describe('buildChatRequestBody', () => {
  it('maps roles, params, and stream options', () => {
    const body = buildChatRequestBody(
      { ...baseOptions, temperature: 0.5, maxTokens: 100, stop: ['x'], reasoningEffort: 'xhigh' } as any,
      {},
      {},
    )
    expect(body.model).toBe('qwen-3.8-27b')
    expect(body.stream).toBe(true)
    expect(body.stream_options).toEqual({ include_usage: true })
    expect(body.temperature).toBe(0.5)
    expect(body.max_tokens).toBe(100)
    expect(body.stop).toEqual(['x'])
    expect(body.reasoning_effort).toBe('xhigh')
    expect(body.messages[0]).toEqual({ role: 'system', content: 'be brief' })
    expect(body.messages[1]).toEqual({ role: 'user', content: [{ type: 'text', text: 'hi' }] })
    expect(body.messages[3]).toEqual({ role: 'tool', tool_call_id: 'call-1', content: 'result' })
  })

  it('materializes max_tokens from defaults when caller omits it', () => {
    const body = buildChatRequestBody(baseOptions as any, {}, { maxOutputTokens: 2048 })
    expect(body.max_tokens).toBe(2048)
  })

  it('maps tools to OpenAI function declarations', () => {
    const body = buildChatRequestBody(
      { ...baseOptions, tools: [{ name: 'bash', description: 'run', parameters: { type: 'object' } }] } as any,
      {},
      {},
    )
    expect(body.tools).toEqual([
      { type: 'function', function: { name: 'bash', description: 'run', parameters: { type: 'object' } } },
    ])
  })

  it('maps a resolved image to an image_url part', async () => {
    const opts = {
      ...baseOptions,
      messages: [{ role: 'user', content: [{ type: 'image', attachment: { id: 'a' } }] }],
    } as any
    const body = await buildChatRequestBody(opts, { imageDataUrl: async () => 'data:image/png;base64,AAA' }, {})
    expect(body.messages[0].content).toEqual([{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } }])
  })

  it('falls back to text placeholder when the image cannot be resolved', async () => {
    const opts = {
      ...baseOptions,
      messages: [{ role: 'user', content: [{ type: 'image', attachment: { id: 'a', name: 'pic.png' } }] }],
    } as any
    const body = await buildChatRequestBody(opts, { imageDataUrl: async () => undefined }, {})
    expect(body.messages[0].content[0].type).toBe('text')
    expect(typeof body.messages[0].content[0].text).toBe('string')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/openai-request.test.ts`
Expected: FAIL — `Cannot find module '../src/openai.ts'`.

- [ ] **Step 3: Write the request half of `src/openai.ts`**

```ts
import type { GenerateOptions, ImageAttachmentRef, ToolSchema } from '@deepseek-ai/dsh-llm'
import { offloadedImageText, textOnlyImageText } from '@deepseek-ai/dsh-llm'

export interface BuildHooks {
  /** Resolve an image attachment to a data URL, or undefined when unavailable. */
  imageDataUrl?: (attachment: ImageAttachmentRef) => Promise<string | undefined>
}

export interface ChatRequestBody {
  model: string
  messages: unknown[]
  stream: true
  stream_options: { include_usage: true }
  tools?: unknown[]
  max_tokens?: number
  temperature?: number
  stop?: string[]
  reasoning_effort?: string
}

interface TextPart { type: 'text'; text: string }
interface ImageUrlPart { type: 'image_url'; image_url: { url: string } }

function toolDef(t: ToolSchema) {
  return { type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }
}

async function mapContent(
  content: readonly unknown[],
  hooks: BuildHooks,
): Promise<Array<TextPart | ImageUrlPart>> {
  const parts: Array<TextPart | ImageUrlPart> = []
  for (const raw of content) {
    const block = raw as { type: string; text?: string; attachment?: ImageAttachmentRef; offloaded?: true }
    if (block.type === 'text') {
      parts.push({ type: 'text', text: block.text ?? '' })
    } else if (block.type === 'image') {
      const ref = block.attachment!
      if (block.offloaded === true) {
        parts.push({ type: 'text', text: offloadedImageText(ref) })
        continue
      }
      const dataUrl = hooks.imageDataUrl ? await hooks.imageDataUrl(ref) : undefined
      if (dataUrl !== undefined) parts.push({ type: 'image_url', image_url: { url: dataUrl } })
      else parts.push({ type: 'text', text: textOnlyImageText(ref) })
    } else if (block.type === 'reasoning' || block.type === 'tool-addition' || block.type === 'tool-removal') {
      // Not provider-visible request content; skip.
    } else {
      parts.push({ type: 'text', text: '' })
    }
  }
  return parts
}

async function mapMessage(msg: any, hooks: BuildHooks): Promise<unknown> {
  const role = msg.role
  const content = Array.isArray(msg.content) ? msg.content : [{ type: 'text', text: msg.content ?? '' }]
  if (role === 'tool') {
    const parts = await mapContent(content, hooks)
    return { role: 'tool', tool_call_id: msg.toolCallId ?? msg.tool_call_id, content: parts.map((p: any) => p.text ?? '').join('\n') }
  }
  if (role === 'assistant' && Array.isArray(msg.toolCalls) && msg.toolCalls.length > 0) {
    return {
      role: 'assistant',
      content: (await mapContent(content, hooks)) as unknown[],
      tool_calls: msg.toolCalls.map((c: any) => ({
        id: c.id,
        type: 'function',
        function: { name: c.name, arguments: c.arguments ?? '' },
      })),
    }
  }
  return { role, content: await mapContent(content, hooks) }
}

/** Build an OpenAI chat-completions request body from harness GenerateOptions. */
export async function buildChatRequestBody(
  options: GenerateOptions,
  hooks: BuildHooks,
  defaults: { maxOutputTokens?: number },
): Promise<ChatRequestBody> {
  const messages: unknown[] = []
  if (options.system !== undefined && options.system.length > 0) {
    messages.push({ role: 'system', content: options.system })
  }
  for (const msg of options.messages) messages.push(await mapMessage(msg, hooks))

  const body: ChatRequestBody = {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
  }
  const tools = options.tools
  if (tools !== undefined && tools.length > 0) body.tools = tools.map(toolDef)
  const maxTokens = options.maxTokens ?? defaults.maxOutputTokens
  if (maxTokens !== undefined) body.max_tokens = maxTokens
  if (options.temperature !== undefined) body.temperature = options.temperature
  if (options.stop !== undefined && options.stop.length > 0) body.stop = options.stop
  if (options.reasoningEffort !== undefined) body.reasoning_effort = String(options.reasoningEffort)
  return body
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/openai-request.test.ts`
Expected: PASS (5 tests). (If `textOnlyImageText`/`offloadedImageText` require a richer `ImageAttachmentRef` than the test's `{ id }`, widen the test fixtures minimally to satisfy the type, keeping the assertions.)

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/openai.ts test/openai-request.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: OpenAI chat-completions request builder"
```

---

### Task 5: SSE → StreamChunk parser (`src/openai.ts` — response half)

**Files:**
- Modify: `src/openai.ts` (append `createSseParser`)
- Test: `test/openai-stream.test.ts`

**Interfaces:**
- Consumes: `StreamChunk`, `ContentBlockType`, `FinishReason`, `TokenUsage`, `ToolCallId`, `brandString` (`@deepseek-ai/dsh-brand`).
- Produces: `SseParser { push(text: string): StreamChunk[]; end(): StreamChunk[] }`, `createSseParser(): SseParser`.

- [ ] **Step 1: Write the failing test `test/openai-stream.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { createSseParser } from '../src/openai.ts'

const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`

describe('createSseParser', () => {
  it('streams text then finishes with usage', () => {
    const p = createSseParser()
    const out = [
      ...p.push(sse({ choices: [{ delta: { content: 'Hel' } }] })),
      ...p.push(sse({ choices: [{ delta: { content: 'lo' } }] })),
      ...p.push(sse({ choices: [{ delta: {}, finish_reason: 'stop' }] })),
      ...p.push(sse({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } })),
      ...p.push('data: [DONE]\n\n'),
      ...p.end(),
    ]
    expect(out[0]).toMatchObject({ type: 'block-start', index: 0, blockType: 'text' })
    const text = out.filter((c: any) => c.type === 'text-delta').map((c: any) => c.text).join('')
    expect(text).toBe('Hello')
    expect(out.some((c: any) => c.type === 'block-end' && c.block?.type === 'text' && c.block.text === 'Hello')).toBe(true)
    const usage = out.find((c: any) => c.type === 'usage')!
    expect(usage.usage).toMatchObject({ inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    const finish = out[out.length - 1]
    expect(finish).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })

  it('assembles a streamed tool call', () => {
    const p = createSseParser()
    const out = [
      ...p.push(sse({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'bash', arguments: '' } }] } }] })),
      ...p.push(sse({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"cmd"' } }] } }] })),
      ...p.push(sse({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ':1}' } }] } }] })),
      ...p.push(sse({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })),
      ...p.end(),
    ]
    const end = out.find((c: any) => c.type === 'block-end' && c.block?.type === 'tool-call')!
    expect(end.block.arguments).toBe('{"cmd":1}')
    expect(end.block.name).toBe('bash')
    expect(out[out.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'tool-calls' } })
  })

  it('emits reasoning deltas and maps length to max-tokens', () => {
    const p = createSseParser()
    const out = [
      ...p.push(sse({ choices: [{ delta: { reasoning_content: 'think' } }] })),
      ...p.push(sse({ choices: [{ delta: {}, finish_reason: 'length' }] })),
      ...p.end(),
    ]
    expect(out.some((c: any) => c.type === 'reasoning-delta' && c.text === 'think')).toBe(true)
    expect(out[out.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'max-tokens' } })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/openai-stream.test.ts`
Expected: FAIL — `createSseParser` is not exported from `../src/openai.ts`.

- [ ] **Step 3: Append `createSseParser` to `src/openai.ts`**

```ts
import { brandString } from '@deepseek-ai/dsh-brand'
import type { ContentBlock, ContentBlockType, FinishReason, StreamChunk, TokenUsage, ToolCallId } from '@deepseek-ai/dsh-llm'

export interface SseParser {
  push(text: string): StreamChunk[]
  end(): StreamChunk[]
}

interface ToolAcc { index: number; id: string; name: string; args: string }

/** Incremental OpenAI SSE → harness StreamChunk translator. */
export function createSseParser(): SseParser {
  let buf = ''
  let nextIndex = 0
  let textBlock: { index: number; text: string } | null = null
  let reasonBlock: { index: number; text: string } | null = null
  const toolBlocks = new Map<number, ToolAcc>()
  let finishReason: FinishReason = { kind: 'stop' }
  let usage: TokenUsage | null = null

  function finishFrom(raw: string | undefined): FinishReason {
    if (raw === 'tool_calls') return { kind: 'tool-calls' }
    if (raw === 'length') return { kind: 'max-tokens' }
    return { kind: 'stop' }
  }

  function handleData(payload: string, out: StreamChunk[]): void {
    if (payload === '[DONE]') return
    let evt: any
    try {
      evt = JSON.parse(payload)
    } catch {
      throw new Error(`unparseable SSE payload: ${payload.slice(0, 80)}`)
    }
    const choice = evt?.choices?.[0]
    const delta = choice?.delta
    if (typeof delta?.content === 'string' && delta.content.length > 0) {
      if (textBlock === null) {
        textBlock = { index: nextIndex++, text: '' }
        out.push({ type: 'block-start', index: textBlock.index, blockType: 'text' })
      }
      textBlock.text += delta.content
      out.push({ type: 'text-delta', index: textBlock.index, text: delta.content })
    }
    const reasoning = delta?.reasoning_content ?? delta?.reasoning
    if (typeof reasoning === 'string' && reasoning.length > 0) {
      if (reasonBlock === null) {
        reasonBlock = { index: nextIndex++, text: '' }
        out.push({ type: 'block-start', index: reasonBlock.index, blockType: 'reasoning' })
      }
      reasonBlock.text += reasoning
      out.push({ type: 'reasoning-delta', index: reasonBlock.index, text: reasoning })
    }
    if (Array.isArray(delta?.tool_calls)) {
      for (const tc of delta.tool_calls) {
        const sseIndex = typeof tc.index === 'number' ? tc.index : 0
        let acc = toolBlocks.get(sseIndex)
        if (acc === undefined) {
          acc = { index: nextIndex++, id: '', name: '', args: '' }
          toolBlocks.set(sseIndex, acc)
          out.push({ type: 'block-start', index: acc.index, blockType: 'tool-call' })
        }
        if (typeof tc.id === 'string') acc.id += tc.id
        const fn = tc.function ?? {}
        if (typeof fn.name === 'string') acc.name += fn.name
        const argDelta = typeof fn.arguments === 'string' ? fn.arguments : ''
        acc.args += argDelta
        out.push({
          type: 'tool-call-delta',
          index: acc.index,
          id: brandString<ToolCallId>(acc.id),
          ...(acc.name.length > 0 ? { name: acc.name } : {}),
          argumentsDelta: argDelta,
        })
      }
    }
    if (choice?.finish_reason !== undefined && choice.finish_reason !== null) {
      finishReason = finishFrom(choice.finish_reason)
    }
    if (evt?.usage !== undefined && evt.usage !== null) {
      const u = evt.usage
      const tokenUsage: TokenUsage = {
        inputTokens: u.prompt_tokens ?? 0,
        outputTokens: u.completion_tokens ?? 0,
      }
      if (u.total_tokens !== undefined) tokenUsage.totalTokens = u.total_tokens
      const cached = u.prompt_tokens_details?.cached_tokens
      if (cached !== undefined) tokenUsage.cacheReadTokens = cached
      const reasoningTokens = u.completion_tokens_details?.reasoning_tokens
      if (reasoningTokens !== undefined) tokenUsage.reasoningTokens = reasoningTokens
      usage = tokenUsage
    }
  }

  return {
    push(text: string): StreamChunk[] {
      buf += text
      const out: StreamChunk[] = []
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '')
        buf = buf.slice(nl + 1)
        if (line.startsWith('data:')) handleData(line.slice(5).trim(), out)
      }
      return out
    },
    end(): StreamChunk[] {
      const out: StreamChunk[] = []
      const blocks: { index: number; block: ContentBlock }[] = []
      if (reasonBlock !== null) blocks.push({ index: reasonBlock.index, block: { type: 'reasoning', text: reasonBlock.text } })
      if (textBlock !== null) blocks.push({ index: textBlock.index, block: { type: 'text', text: textBlock.text } })
      for (const acc of toolBlocks.values()) {
        blocks.push({ index: acc.index, block: { type: 'tool-call', id: brandString<ToolCallId>(acc.id), name: acc.name, arguments: acc.args } })
      }
      blocks.sort((a, b) => a.index - b.index)
      for (const b of blocks) out.push({ type: 'block-end', index: b.index, block: b.block })
      if (usage !== null) out.push({ type: 'usage', usage })
      out.push({ type: 'finish', reason: finishReason })
      return out
    },
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/openai-stream.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/openai.ts test/openai-stream.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: SSE to StreamChunk translator"
```

---

### Task 6: Credential resolution (`src/credentials.ts`)

**Files:**
- Create: `src/credentials.ts`
- Test: `test/credentials.test.ts`

**Interfaces:**
- Consumes: `assertUsableApiKey` from `@deepseek-ai/dsh-llm`.
- Produces: `ResolveApiKey = () => Promise<string|undefined>`; `makeResolveApiKey(opts): ResolveApiKey`.

- [ ] **Step 1: Write the failing test `test/credentials.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { makeResolveApiKey } from '../src/credentials.ts'

describe('makeResolveApiKey', () => {
  it('prefers the credentials seam over env', async () => {
    const resolve = makeResolveApiKey({
      apiKeyEnv: 'NEURALWATT_API_KEY',
      resolveCredential: async () => 'sk-from-store',
      env: { NEURALWATT_API_KEY: 'sk-from-env' },
    })
    expect(await resolve()).toBe('sk-from-store')
  })
  it('falls back to env and trims', async () => {
    const resolve = makeResolveApiKey({
      apiKeyEnv: 'NEURALWATT_API_KEY',
      resolveCredential: async () => undefined,
      env: { NEURALWATT_API_KEY: '  sk-env  ' },
    })
    expect(await resolve()).toBe('sk-env')
  })
  it('returns undefined when unset', async () => {
    const resolve = makeResolveApiKey({ apiKeyEnv: 'NEURALWATT_API_KEY', env: {} })
    expect(await resolve()).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/credentials.test.ts`
Expected: FAIL — `Cannot find module '../src/credentials.ts'`.

- [ ] **Step 3: Write minimal implementation `src/credentials.ts`**

```ts
import { assertUsableApiKey } from '@deepseek-ai/dsh-llm'

export type ResolveApiKey = () => Promise<string | undefined>

export interface ResolveApiKeyOptions {
  /** Environment-variable name that holds the API key. */
  apiKeyEnv: string
  /** Optional credentials-seam resolver (returns the stored secret or undefined). */
  resolveCredential?: (ref: string) => Promise<string | undefined>
  /** Environment source; defaults to process.env. */
  env?: NodeJS.ProcessEnv
}

/** Resolve the API key: credentials seam first, then env. Undefined when unset. */
export function makeResolveApiKey(opts: ResolveApiKeyOptions): ResolveApiKey {
  const env = opts.env ?? process.env
  return async () => {
    const fromStore = opts.resolveCredential ? await opts.resolveCredential(opts.apiKeyEnv) : undefined
    const raw = fromStore !== undefined && fromStore.length > 0 ? fromStore : env[opts.apiKeyEnv]
    if (raw === undefined || raw.trim().length === 0) return undefined
    return assertUsableApiKey(raw, 'dsh-llm-neuralwatt', opts.apiKeyEnv)
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/credentials.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/credentials.ts test/credentials.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: API-key resolution with env fallback"
```

---

### Task 7: The adapter (`src/adapter.ts`)

**Files:**
- Create: `src/adapter.ts`
- Test: `test/adapter.test.ts`

**Interfaces:**
- Consumes: `LlmAdapter`, `attributionHeaders`, `LlmError`, `brandString`; `mapModel`-shaped `ModelDescriptor` (from `models.ts`); `buildChatRequestBody`, `createSseParser` (from `openai.ts`); `makeResolveApiKey` types (from `credentials.ts`); `httpError`/`missingCredential`/`unknownModel`/`discoveryError` (from `errors.ts`).
- Produces: `NeuralwattAdapter` class + `NeuralwattAdapterOptions`.

- [ ] **Step 1: Write the failing test `test/adapter.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import { NeuralwattAdapter } from '../src/adapter.ts'
import type { ModelDescriptor } from '../src/models.ts'

const descriptor: ModelDescriptor = {
  id: 'qwen-3.8-27b',
  name: 'Qwen 3.8 27B',
  inputModalities: ['text', 'image'],
  contextWindow: 262128,
  maxTokens: 131072,
  tools: true,
  reasoning: { efforts: [{ id: 'xhigh', name: 'XHigh' }, { id: 'off', name: 'Off' }], defaultEffort: 'xhigh' },
}

function makeAdapter(overrides: Partial<ConstructorParameters<typeof NeuralwattAdapter>[0]> = {}) {
  return new NeuralwattAdapter({
    route: 'neuralwatt',
    displayName: 'Neuralwatt',
    baseURL: 'https://api.example/v1',
    apiKeyEnv: 'NEURALWATT_API_KEY',
    resolveApiKey: async () => 'sk-test',
    discover: async () => [descriptor],
    ...overrides,
  })
}

describe('NeuralwattAdapter', () => {
  it('exposes provider info and the discovered model list', async () => {
    const a = makeAdapter()
    expect(a.providerInfo('neuralwatt')).toEqual({ id: 'neuralwatt', name: 'Neuralwatt' })
    const models = await a.listModels('neuralwatt')
    expect(models).toEqual([
      { provider: 'neuralwatt', id: 'qwen-3.8-27b', name: 'Qwen 3.8 27B', description: undefined, inputModalities: ['text', 'image'] },
    ])
  })

  it('resolves a model with context, max tokens, and reasoning', async () => {
    const a = makeAdapter()
    const info = await a.resolveModel('neuralwatt', 'qwen-3.8-27b')
    expect(info.context).toEqual({ contextWindow: 262128 })
    expect(info.defaultMaxTokens).toBe(131072)
    expect(info.inputModalities).toEqual(['text', 'image'])
    expect(info.reasoning!.defaultEffort).toBe('xhigh')
    expect(info.reasoning!.efforts.map((e: any) => e.id)).toEqual(['xhigh', 'off'])
  })

  it('throws UNKNOWN_MODEL for an unlisted model', async () => {
    const a = makeAdapter()
    await expect(a.resolveModel('neuralwatt', 'nope')).rejects.toMatchObject({ code: 'UNKNOWN_MODEL' })
  })

  it('streams a completion from a mock OpenAI server', async () => {
    const sse = ['data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n', 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n', 'data: [DONE]\n\n'].join('')
    const calls: { url: string; auth?: string; attribution?: string }[] = []
    const fetchFn = async (url: any, init: any) => {
      calls.push({ url: String(url), auth: init?.headers?.authorization ?? init?.headers?.Authorization, attribution: init?.headers?.['x-dsh-attribution'] })
      return new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    const a = makeAdapter({ fetchFn: fetchFn as any })
    const chunks: any[] = []
    for await (const c of a.stream({ provider: 'neuralwatt', model: 'qwen-3.8-27b', messages: [{ role: 'user', content: 'hi' }] } as any)) chunks.push(c)
    expect(calls[0].url).toBe('https://api.example/v1/chat/completions')
    expect(calls[0].auth).toBe('Bearer sk-test')
    expect(chunks.some((c) => c.type === 'text-delta' && c.text === 'Hi')).toBe(true)
    expect(chunks[chunks.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })

  it('surfaces HTTP errors as PROVIDER_HTTP_ERROR', async () => {
    const fetchFn = async () => new Response('nope', { status: 500 })
    const a = makeAdapter({ fetchFn: fetchFn as any })
    await expect(async () => {
      for await (const _ of a.stream({ provider: 'neuralwatt', model: 'qwen-3.8-27b', messages: [] } as any)) { /* drain */ }
    }).rejects.toMatchObject({ code: 'PROVIDER_HTTP_ERROR' })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/adapter.test.ts`
Expected: FAIL — `Cannot find module '../src/adapter.ts'`.

- [ ] **Step 3: Write minimal implementation `src/adapter.ts`**

```ts
import { LlmAdapter, attributionHeaders, type GenerateOptions, type LlmModelInfo, type LlmProviderInfo, type LlmResolvedModelInfo, type StreamChunk } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type { ModelDescriptor } from './models.ts'
import { discoverModels } from './models.ts'
import { buildChatRequestBody, createSseParser } from './openai.ts'
import type { ResolveApiKey } from './credentials.ts'
import { httpError, missingCredential, protocolError, unknownModel } from './errors.ts'

export interface NeuralwattAdapterOptions {
  route: string
  displayName: string
  baseURL: string
  apiKeyEnv: string
  resolveApiKey: ResolveApiKey
  imageDataUrl?: (attachment: ImageAttachmentRef) => Promise<string | undefined>
  /** Injected model source; defaults to a memoized discoverModels boot snapshot. */
  discover?: () => Promise<ModelDescriptor[]>
  /** Injected fetch for tests. */
  fetchFn?: typeof fetch
}

export class NeuralwattAdapter extends LlmAdapter {
  private readonly opts: NeuralwattAdapterOptions
  private readonly fetchFn: typeof fetch
  private catalogPromise: Promise<ModelDescriptor[]> | undefined
  private catalogError: string | undefined

  constructor(opts: NeuralwattAdapterOptions) {
    super()
    this.opts = opts
    this.fetchFn = opts.fetchFn ?? fetch
  }

  /** Memoized boot snapshot (or the injected discover). */
  private catalog(): Promise<ModelDescriptor[]> {
    if (this.catalogPromise === undefined) {
      this.catalogPromise = (this.opts.discover ?? (async () => {
        const key = await this.opts.resolveApiKey()
        return discoverModels(this.opts.baseURL, key, this.fetchFn)
      }))().catch((err: unknown) => {
        this.catalogError = err instanceof Error ? err.message : String(err)
        return [] as ModelDescriptor[]
      })
    }
    return this.catalogPromise
  }

  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.opts.displayName }
  }

  async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const models = await this.catalog()
    return models.map((m) => ({
      provider,
      id: m.id,
      name: m.name,
      ...(m.description === undefined ? {} : { description: m.description }),
      inputModalities: [...m.inputModalities],
    }))
  }

  async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const models = await this.catalog()
    const found = models.find((m) => m.id === model)
    if (found === undefined) throw unknownModel(provider, model, this.catalogError)
    return {
      provider,
      id: found.id,
      name: found.name,
      ...(found.description === undefined ? {} : { description: found.description }),
      inputModalities: [...found.inputModalities],
      context: { contextWindow: found.contextWindow },
      defaultMaxTokens: found.maxTokens,
      ...(found.reasoning === undefined
        ? {}
        : {
            reasoning: {
              efforts: found.reasoning.efforts.map((e) => ({ id: e.id as any, name: e.name })),
              ...(found.reasoning.defaultEffort === undefined ? {} : { defaultEffort: found.reasoning.defaultEffort as any }),
            },
          }),
    }
  }

  async prepareCall(provider: string, model: string, signal?: AbortSignal) {
    const info = await this.resolveModel(provider, model, signal)
    return { model: info, stream: (options: GenerateOptions) => this.stream(options) }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const key = await this.opts.resolveApiKey()
    if (key === undefined) throw missingCredential(this.opts.apiKeyEnv)
    const body = await buildChatRequestBody(
      options,
      { ...(this.opts.imageDataUrl ? { imageDataUrl: this.opts.imageDataUrl } : {}) },
      {},
    )
    const url = `${this.opts.baseURL.replace(/\/+$/, '')}/chat/completions`
    const res = await this.fetchFn(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${key}`,
        ...attributionHeaders(),
      },
      body: JSON.stringify(body),
      ...(options.signal ? { signal: options.signal } : {}),
    })
    if (!res.ok) throw httpError(res.status, await res.text().catch(() => ''))
    if (res.body === null) throw protocolError('empty response body')

    const parser = createSseParser()
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const chunk of parser.push(decoder.decode(value, { stream: true }))) yield chunk
      }
      for (const chunk of parser.push(decoder.decode())) yield chunk
      for (const chunk of parser.end()) yield chunk
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') throw err
      throw protocolError(err instanceof Error ? err.message : String(err))
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/adapter.test.ts`
Expected: PASS (5 tests). (Adjust `listModels` expectation if `description: undefined` is omitted rather than present — assert on `id`/`name`/`inputModalities` fields explicitly.)

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add src/adapter.ts test/adapter.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: NeuralwattAdapter over OpenAI-compatible chat completions"
```

---

### Task 8: Plugin entry (`src/index.ts`)

**Files:**
- Create: `src/index.ts`
- Test: `test/plugin.test.ts`

**Interfaces:**
- Consumes: `resolveEffectiveConfig`, `Config` (config.ts); `makeResolveApiKey` (credentials.ts); `NeuralwattAdapter` (adapter.ts); `resolveImageAttachmentAccess` (`@deepseek-ai/dsh-llm`); `Context` (`@deepseek-ai/cordis`).
- Produces: plugin module exporting `name`, `inject`, `Config`, `apply`.

- [ ] **Step 1: Write the failing test `test/plugin.test.ts`**

```ts
import { describe, expect, it } from 'vitest'
import * as plugin from '../src/index.ts'

describe('plugin module', () => {
  it('exports the plugin contract', () => {
    expect(plugin.name).toBe('llm-neuralwatt')
    expect(plugin.inject).toEqual(['llm'])
    expect(typeof plugin.apply).toBe('function')
    expect(plugin.Config).toBeDefined()
  })

  it('registers the neuralwatt adapter on apply', () => {
    const registered: { routes: string[] }[] = []
    const ctx: any = {
      get: () => undefined,
      llm: { registerAdapter: (routes: string[]) => { registered.push({ routes }); return { replace() {} } } },
    }
    plugin.apply(ctx, { baseURL: 'https://x.example/v1' })
    expect(registered[0].routes).toEqual(['neuralwatt'])
  })

  it('uses the configured route key', () => {
    const registered: { routes: string[] }[] = []
    const ctx: any = {
      get: () => undefined,
      llm: { registerAdapter: (routes: string[]) => { registered.push({ routes }); return { replace() {} } } },
    }
    plugin.apply(ctx, { route: 'nw' })
    expect(registered[0].routes).toEqual(['nw'])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run test/plugin.test.ts`
Expected: FAIL — `Cannot find module '../src/index.ts'`.

- [ ] **Step 3: Write minimal implementation `src/index.ts`**

```ts
import type { Context } from '@deepseek-ai/cordis'
import { resolveImageAttachmentAccess } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { Config, resolveEffectiveConfig, type Config as PluginConfig } from './config.ts'
import { makeResolveApiKey } from './credentials.ts'
import { NeuralwattAdapter } from './adapter.ts'

export const name = 'llm-neuralwatt'
export const inject = ['llm']
export { Config }
export type { PluginConfig as ConfigType }

async function readDataUrl(path: string): Promise<string | undefined> {
  try {
    const { readFile } = await import('node:fs/promises')
    const bytes = await readFile(path)
    return `data:image/png;base64,${Buffer.from(bytes).toString('base64')}`
  } catch {
    return undefined
  }
}

export function apply(ctx: Context, config: PluginConfig): void {
  const eff = resolveEffectiveConfig(config, process.env)

  const resolveApiKey = makeResolveApiKey({
    apiKeyEnv: eff.apiKeyEnv,
    resolveCredential: async (ref) => {
      const credentials = ctx.get('credentials') as { resolve(r: string): Promise<{ value?: string } | undefined> } | undefined
      return (await credentials?.resolve(ref))?.value
    },
    env: process.env,
  })

  const imageDataUrl = async (ref: ImageAttachmentRef): Promise<string | undefined> => {
    const attachments = ctx.get('attachments') as any
    const fs = ctx.get('fs') as any
    if (attachments === undefined) return undefined
    const access = resolveImageAttachmentAccess(
      attachments,
      (hostPath: string) => fs?.processPathFromHostPath?.(hostPath) ?? hostPath,
      ref,
    )
    return access === undefined ? undefined : readDataUrl(access.readonlyPath)
  }

  const adapter = new NeuralwattAdapter({
    route: eff.route,
    displayName: eff.displayName,
    baseURL: eff.baseURL,
    apiKeyEnv: eff.apiKeyEnv,
    resolveApiKey,
    imageDataUrl,
  })
  ctx.llm.registerAdapter([eff.route], adapter)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run test/plugin.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite + typecheck + build**

Run: `pnpm test && pnpm exec tsc -p tsconfig.json --noEmit && pnpm build`
Expected: all tests green, `lib/index.js` + `lib/types/` emitted.

- [ ] **Step 6: Commit**

```bash
git -c commit.gpgsign=false add src/index.ts test/plugin.test.ts
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "feat: llm-neuralwatt plugin entry registering the adapter"
```

---

### Task 9: README + packaging + manual verification

**Files:**
- Create: `README.md`
- Modify: `package.json` (verify `files`, `exports`, `peerDependencies`)

**Interfaces:**
- Consumes: everything above.
- Produces: an installable package + operator docs.

- [ ] **Step 1: Write `README.md`**

```markdown
# dsh-llm-neuralwatt

Neuralwatt LLM provider plugin for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

It dynamically populates models from Neuralwatt's `GET /v1/models` endpoint and sets
each model's capabilities — input modalities (text / image), context window, max output
tokens, and reasoning efforts — from the endpoint metadata. It speaks Neuralwatt's
OpenAI-compatible `/chat/completions` API.

## Install

    dsh plugin --profile <profile> add dsh-llm-neuralwatt

Then remove any hand-written `neuralwatt` block under `llm-pi-ai.providers` in your
profile's `cordis.patch.yml` (the two routes would otherwise collide).

## Configuration

Configure under the plugin (`llm-neuralwatt`) in `cordis.patch.yml`:

    - id: llm-neuralwatt
      name: dsh-llm-neuralwatt
      config:
        baseURL: https://api.neuralwatt.com/v1   # optional; override here
        apiKeyEnv: NEURALWATT_API_KEY            # optional; env var *name*
        route: neuralwatt                        # optional
        displayName: Neuralwatt                  # optional

| Field | Default | Env fallback | Meaning |
|---|---|---|---|
| `route` | `neuralwatt` | — | Provider route key / selector id. |
| `displayName` | `Neuralwatt` | — | Provider display name. |
| `baseURL` | `https://api.neuralwatt.com/v1` | `NEURALWATT_BASE_URL` | Endpoint root (overridable). |
| `apiKeyEnv` | `NEURALWATT_API_KEY` | — | Env-var *name* holding the API key. |

Precedence: `config` > env > built-in default. Set the API key by exporting
`NEURALWATT_API_KEY` (or whatever `apiKeyEnv` names) or storing it via the credentials
service.

## Behavior

- Models are fetched **once at load** (boot snapshot) from `GET {baseURL}/models`.
- All models are exposed; only entries marked `deprecated` are skipped.
- Vision models advertise `image` input; text-only models advertise `text`.
```

- [ ] **Step 2: Verify package metadata**

Confirm `package.json` `files: ["lib", "README.md"]`, `main: lib/index.js`, `types: lib/types/index.d.ts`, and peer/deps match Task 0. Fix any mismatch.

- [ ] **Step 3: Build and smoke-load**

Run: `pnpm build && node -e "import('./lib/index.js').then(m=>console.log(m.name, m.inject))"`
Expected: prints `llm-neuralwatt [ 'llm' ]`.

- [ ] **Step 4: Manual install verification (optional, needs a dsh profile)**

Run: `dsh plugin --profile <profile> add ./` (or publish then `add dsh-llm-neuralwatt`), boot the profile, open the model picker, and confirm Neuralwatt models appear with correct context/modalities and a chat round-trips.

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add README.md package.json
git -c commit.gpgsign=false -c user.name="dsh-neuralwatt" -c user.email="dev@localhost" commit -m "docs: README and packaging for dsh-llm-neuralwatt"
```

---

## Self-Review

**Spec coverage:**
- Dynamically populate models from `/v1/models` → Task 2 (`discoverModels`/`mapModels`) + Task 7 (memoized boot snapshot).
- Set capabilities (text/image, context, max output, reasoning) → Task 2 mapping table + Task 7 `resolveModel`.
- Allow baseURL override → Task 1 precedence + Task 8 wiring + README.
- Boot snapshot only → Task 7 `catalog()` memoization (no timers).
- All models, skip deprecated → Task 2 `mapModel` returns null on deprecated.
- OpenAI-compatible chat + SSE → Task 4 + Task 5.
- Error handling (stable codes, attribution headers, signal) → Task 3 + Task 7.
- Packaging + install → Task 0 + Task 9.

**Placeholder scan:** No TBD/TODO; every code step carries concrete code and exact commands. Image data-URL uses a concrete `data:image/png;base64` encoding with a documented fallback to `textOnlyImageText`/`offloadedImageText` (not a TODO).

**Type consistency:** `ModelDescriptor`, `EffectiveConfig`, `ResolveApiKey`, `SseParser`, `NeuralwattAdapterOptions` names and shapes are used identically across Tasks 1–8. `resolveModel`/`listModels`/`stream`/`prepareCall` signatures match `LlmAdapter`. One caveat to confirm at implementation: `resolveModel`'s third `signal` parameter is accepted for interface parity (Task 7 implements `(provider, model)` and the base default passes through).
