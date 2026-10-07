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
