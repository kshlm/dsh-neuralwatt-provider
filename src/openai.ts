import type { GenerateOptions, ToolSchema } from '@deepseek-ai/dsh-llm'
import { fileHandleText, offloadedImageText, textOnlyImageText } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, FinishReason, StreamChunk, TokenUsage, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { brandString } from '@deepseek-ai/dsh-brand'

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
type Part = TextPart | ImageUrlPart

/** Loose view over harness content blocks; unknown tags are skipped. */
interface LooseBlock {
  type?: string
  text?: string
  attachment?: ImageAttachmentRef
  offloaded?: true
  id?: unknown
  name?: string
  arguments?: string
}

interface ToolCallShape {
  id: string
  name: string
  arguments: string
}

function toolDef(t: ToolSchema) {
  return { type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }
}

async function mapContent(content: readonly unknown[], hooks: BuildHooks): Promise<Part[]> {
  const parts: Part[] = []
  for (const raw of content) {
    const block = raw as LooseBlock
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
    } else if (block.type === 'file') {
      // Providers never receive file blocks natively; send the deterministic handle text.
      parts.push({ type: 'text', text: fileHandleText(block.attachment as never, undefined) })
    }
    // reasoning / tool-call / tool-addition / tool-removal are not provider-visible
    // request content here (tool calls are lifted into `tool_calls`), and unknown
    // merge-extensible tags are skipped.
  }
  return parts
}

/** Collect assistant tool calls from `tool-call` blocks and any legacy `toolCalls` array. */
function collectToolCalls(msg: any): ToolCallShape[] {
  const calls: ToolCallShape[] = []
  if (Array.isArray(msg.content)) {
    for (const raw of msg.content) {
      const block = raw as LooseBlock
      if (block.type === 'tool-call') {
        calls.push({ id: String(block.id ?? ''), name: block.name ?? '', arguments: block.arguments ?? '' })
      }
    }
  }
  if (Array.isArray(msg.toolCalls)) {
    for (const c of msg.toolCalls) {
      calls.push({ id: String(c.id ?? ''), name: c.name ?? '', arguments: c.arguments ?? '' })
    }
  }
  return calls
}

function textOf(parts: Part[]): string {
  return parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n')
}

async function mapMessage(msg: any, hooks: BuildHooks): Promise<unknown> {
  const rawContent = msg.content
  const isString = typeof rawContent === 'string'
  const blocks: unknown[] = Array.isArray(rawContent)
    ? rawContent
    : [{ type: 'text', text: isString ? rawContent : '' }]
  // OpenAI has no `developer` role; the system slot carries that instruction text.
  const role = msg.role === 'developer' ? 'system' : msg.role
  const parts = await mapContent(blocks, hooks)

  if (role === 'tool') {
    return { role: 'tool', tool_call_id: msg.toolCallId ?? msg.tool_call_id, content: textOf(parts) }
  }
  const toolCalls = collectToolCalls(msg)
  if (role === 'assistant' && toolCalls.length > 0) {
    return {
      role: 'assistant',
      content: parts.length > 0 ? parts : '',
      tool_calls: toolCalls.map((c) => ({
        id: c.id,
        type: 'function',
        function: { name: c.name, arguments: c.arguments },
      })),
    }
  }
  if (isString) return { role, content: textOf(parts) }
  return { role, content: parts }
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
