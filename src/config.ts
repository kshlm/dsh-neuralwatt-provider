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
