#!/usr/bin/env node
/**
 * Local end-to-end smoke test for @kshlm/dsh-neuralwatt-provider.
 *
 * Unlike the vitest suite (which injects a fake fetch), this boots the built
 * plugin against a real local HTTP server that speaks Neuralwatt's wire API:
 *   GET  /v1/models            -> model catalog with metadata
 *   POST /v1/chat/completions  -> OpenAI-style SSE stream
 *
 * It exercises the same path a dsh profile takes: apply() -> adapter ->
 * listModels() (boot snapshot) -> stream() (real fetch + SSE translation).
 *
 * Usage:
 *   pnpm build && node scripts/local-smoke.mjs
 */
import { createServer } from 'node:http'
import assert from 'node:assert/strict'
import { once } from 'node:events'

const apiKey = 'sk-local-smoke'
process.env.NEURALWATT_API_KEY = apiKey

// ---------------------------------------------------------------------------
// Mock Neuralwatt server
// ---------------------------------------------------------------------------
const catalog = {
  object: 'list',
  data: [
    {
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
    },
    { id: 'retired-model', metadata: { display_name: 'Retired', deprecated: true } },
  ],
}

const seen = []
const server = createServer((req, res) => {
  seen.push({ method: req.method, url: req.url, auth: req.headers.authorization, ua: req.headers['user-agent'] })
  if (req.method === 'GET' && req.url === '/v1/models') {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify(catalog))
    return
  }
  if (req.method === 'POST' && req.url === '/v1/chat/completions') {
    let body = ''
    req.on('data', (c) => { body += c })
    req.on('end', () => {
      seen.at(-1).body = JSON.parse(body)
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      const ev = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`)
      ev({ choices: [{ delta: { reasoning_content: 'hmm' } }] })
      ev({ choices: [{ delta: { content: 'Hi' } }] })
      ev({ choices: [{ delta: {}, finish_reason: 'stop' }] })
      ev({ choices: [], usage: { prompt_tokens: 7, completion_tokens: 1, total_tokens: 8 } })
      res.write('data: [DONE]\n\n')
      res.end()
    })
    return
  }
  res.writeHead(404).end('not found')
})
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const baseURL = `http://127.0.0.1:${server.address().port}/v1`
console.log(`mock Neuralwatt server on ${baseURL}`)

// ---------------------------------------------------------------------------
// Boot the built plugin with a minimal fake harness context
// ---------------------------------------------------------------------------
const { apply } = await import(new URL('../lib/index.js', import.meta.url))
let adapter
apply(
  { get: () => undefined, llm: { registerAdapter: (_routes, a) => { adapter = a } } },
  { baseURL, displayName: 'Neuralwatt' },
)
assert.ok(adapter, 'apply() registered an adapter')

const models = await adapter.listModels('neuralwatt')
console.log('discovered models:', models.map((m) => `${m.id} [${m.inputModalities.join('+')}]`).join(', '))
assert.deepEqual(models.map((m) => m.id), ['qwen-3.8-27b'], 'deprecated models are filtered')

const info = await adapter.resolveModel('neuralwatt', 'qwen-3.8-27b')
console.log('resolved:', JSON.stringify({ context: info.context, maxTokens: info.defaultMaxTokens, reasoning: info.reasoning }))
assert.equal(info.context.contextWindow, 262128)
assert.equal(info.defaultMaxTokens, 131072)
assert.deepEqual(info.reasoning.efforts.map((e) => e.id), ['xhigh', 'medium', 'low', 'off'])

const chunks = []
for await (const chunk of adapter.stream({
  provider: 'neuralwatt',
  model: 'qwen-3.8-27b',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  maxTokens: 64,
})) {
  chunks.push(chunk)
}
const text = chunks.filter((c) => c.type === 'text-delta').map((c) => c.text).join('')
const finish = chunks.at(-1)
const usage = chunks.find((c) => c.type === 'usage')
console.log('stream text:', JSON.stringify(text), '| finish:', JSON.stringify(finish.reason), '| usage:', JSON.stringify(usage.usage))
assert.equal(text, 'Hi')
assert.deepEqual(finish.reason, { kind: 'stop' })
assert.deepEqual(usage.usage, { inputTokens: 7, outputTokens: 1, totalTokens: 8 })

const chat = seen.find((r) => r.url === '/v1/chat/completions')
assert.equal(chat.auth, `Bearer ${apiKey}`)
assert.ok(chat.ua, 'attribution user-agent header is sent')
assert.equal(chat.body.model, 'qwen-3.8-27b')
assert.equal(chat.body.stream, true)
assert.equal(chat.body.max_tokens, 64)
console.log('request headers/body verified:', JSON.stringify({ auth: chat.auth, ua: chat.ua, model: chat.body.model, max_tokens: chat.body.max_tokens }))

server.close()
console.log('\n✅ local smoke test passed')
