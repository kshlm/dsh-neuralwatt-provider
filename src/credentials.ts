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
