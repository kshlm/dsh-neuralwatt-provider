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
  if (apiKey !== undefined && apiKey.length > 0) headers.Authorization = `Bearer ${apiKey}`
  const res = await fetchFn(url, { method: 'GET', headers, ...(signal ? { signal } : {}) })
  if (!res.ok) {
    throw new Error(`Neuralwatt /models returned ${res.status}`)
  }
  const body = (await res.json()) as NeuralwattModelsResponse
  return mapModels(body)
}
