import { describe, expect, it } from 'vitest'
import { createSseParser } from '../src/openai.ts'

const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`

describe('createSseParser', () => {
  it('streams text then finishes with usage', () => {
    const p = createSseParser()
    const t1 = { choices: [{ delta: { content: 'Hel' } }] }
    const t2 = { choices: [{ delta: { content: 'lo' } }] }
    const t3 = { choices: [{ delta: {}, finish_reason: 'stop' }] }
    const t4 = { choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } }
    const out = [
      ...p.push(sse(t1)),
      ...p.push(sse(t2)),
      ...p.push(sse(t3)),
      ...p.push(sse(t4)),
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
    const e1 = { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'bash', arguments: '' } }] } }] }
    const e2 = { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"cmd"' } }] } }] }
    const e3 = { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ':1}' } }] } }] }
    const e4 = { choices: [{ delta: {}, finish_reason: 'tool_calls' }] }
    const out = [
      ...p.push(sse(e1)),
      ...p.push(sse(e2)),
      ...p.push(sse(e3)),
      ...p.push(sse(e4)),
      ...p.end(),
    ]
    const end = out.find((c: any) => c.type === 'block-end' && c.block?.type === 'tool-call')!
    expect(end.block.arguments).toBe('{"cmd":1}')
    expect(end.block.name).toBe('bash')
    expect(out[out.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'tool-calls' } })
  })

  it('emits reasoning deltas and maps length to max-tokens', () => {
    const p = createSseParser()
    const r1 = { choices: [{ delta: { reasoning_content: 'think' } }] }
    const r2 = { choices: [{ delta: {}, finish_reason: 'length' }] }
    const out = [
      ...p.push(sse(r1)),
      ...p.push(sse(r2)),
      ...p.end(),
    ]
    expect(out.some((c: any) => c.type === 'reasoning-delta' && c.text === 'think')).toBe(true)
    expect(out[out.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'max-tokens' } })
  })
})
