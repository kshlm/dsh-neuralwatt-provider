import {
  LlmAdapter,
  attributionHeaders,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type PreparedAdapterCall,
  type ReasoningEffortId,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type { ModelDescriptor } from './models.js'
import { discoverModels } from './models.js'
import { buildChatRequestBody, createSseParser } from './openai.js'
import type { ResolveApiKey } from './credentials.js'
import { httpError, missingCredential, protocolError, unknownModel } from './errors.js'

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

  async resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
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
              efforts: found.reasoning.efforts.map((e) => ({ id: e.id as ReasoningEffortId, name: e.name })),
              ...(found.reasoning.defaultEffort === undefined
                ? {}
                : { defaultEffort: found.reasoning.defaultEffort as ReasoningEffortId }),
            },
          }),
    }
  }

  async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    const info = await this.resolveModel(provider, model, signal)
    return { model: info, stream: (options: GenerateOptions) => this.stream(options) }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const key = await this.opts.resolveApiKey()
    if (key === undefined) throw missingCredential(this.opts.apiKeyEnv)

    // Materialize the route's own output cap only when callers omit one.
    const models = await this.catalog()
    const found = models.find((m) => m.id === options.model)
    const defaults: { maxOutputTokens?: number } = found === undefined ? {} : { maxOutputTokens: found.maxTokens }

    const body = await buildChatRequestBody(
      options,
      { ...(this.opts.imageDataUrl ? { imageDataUrl: this.opts.imageDataUrl } : {}) },
      defaults,
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
