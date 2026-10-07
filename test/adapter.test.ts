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

  it('forwards the abort signal and attribution headers', async () => {
    const controller = new AbortController()
    let seenSignal: AbortSignal | undefined
    let seenAttribution: string | undefined
    const fetchFn = async (_url: any, init: any) => {
      seenSignal = init?.signal
      seenAttribution = init?.headers?.['user-agent'] ?? init?.headers?.['User-Agent']
      return new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    const a = makeAdapter({ fetchFn: fetchFn as any })
    for await (const _ of a.stream({ provider: 'neuralwatt', model: 'qwen-3.8-27b', messages: [], signal: controller.signal } as any)) { /* drain */ }
    expect(seenSignal).toBe(controller.signal)
    expect(typeof seenAttribution).toBe('string')
    expect(seenAttribution!.length).toBeGreaterThan(0)
  })

  it('surfaces HTTP errors as PROVIDER_HTTP_ERROR', async () => {
    const fetchFn = async () => new Response('nope', { status: 500 })
    const a = makeAdapter({ fetchFn: fetchFn as any })
    await expect(async () => {
      for await (const _ of a.stream({ provider: 'neuralwatt', model: 'qwen-3.8-27b', messages: [] } as any)) { /* drain */ }
    }).rejects.toMatchObject({ code: 'PROVIDER_HTTP_ERROR' })
  })

  it('throws MISSING_CREDENTIAL when no key resolves', async () => {
    const a = makeAdapter({ resolveApiKey: async () => undefined })
    await expect(async () => {
      for await (const _ of a.stream({ provider: 'neuralwatt', model: 'qwen-3.8-27b', messages: [] } as any)) { /* drain */ }
    }).rejects.toMatchObject({ code: 'MISSING_CREDENTIAL' })
  })
})
