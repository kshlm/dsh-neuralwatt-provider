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
  it('maps roles, params, and stream options', async () => {
    const body = await buildChatRequestBody(
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

  it('materializes max_tokens from defaults when caller omits it', async () => {
    const body = await buildChatRequestBody(baseOptions as any, {}, { maxOutputTokens: 2048 })
    expect(body.max_tokens).toBe(2048)
  })

  it('maps tools to OpenAI function declarations', async () => {
    const body = await buildChatRequestBody(
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

  it('maps an assistant tool-call content block to tool_calls', async () => {
    const opts = {
      ...baseOptions,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'calling' },
            { type: 'tool-call', id: 'call-9', name: 'bash', arguments: '{"cmd":"ls"}' },
          ],
        },
      ],
    } as any
    const body = await buildChatRequestBody(opts, {}, {})
    expect(body.messages[0].content).toEqual([{ type: 'text', text: 'calling' }])
    expect(body.messages[0].tool_calls).toEqual([
      { id: 'call-9', type: 'function', function: { name: 'bash', arguments: '{"cmd":"ls"}' } },
    ])
  })
})
