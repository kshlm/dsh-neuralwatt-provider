import { LlmError } from '@deepseek-ai/dsh-llm'

export const Codes = {
  PROVIDER_HTTP_ERROR: 'PROVIDER_HTTP_ERROR',
  PROVIDER_PROTOCOL_ERROR: 'PROVIDER_PROTOCOL_ERROR',
  MISSING_CREDENTIAL: 'MISSING_CREDENTIAL',
  UNKNOWN_MODEL: 'UNKNOWN_MODEL',
} as const

export function httpError(status: number, detail: string): LlmError {
  return new LlmError(`Neuralwatt API error: ${status} ${detail}`, Codes.PROVIDER_HTTP_ERROR, { status })
}

export function protocolError(detail: string): LlmError {
  return new LlmError(`Neuralwatt protocol error: ${detail}`, Codes.PROVIDER_PROTOCOL_ERROR)
}

export function missingCredential(ref: string): LlmError {
  return new LlmError(
    `Neuralwatt: no API key; this route resolves ${ref}, which is not set — store ${ref} through the credentials service or export it`,
    Codes.MISSING_CREDENTIAL,
  )
}

export function unknownModel(route: string, model: string, detail?: string): LlmError {
  return new LlmError(
    `Neuralwatt route "${route}" has no model "${model}"${detail === undefined ? '' : ` (${detail})`}`,
    Codes.UNKNOWN_MODEL,
  )
}

export function discoveryError(detail: string): LlmError {
  return new LlmError(`Neuralwatt model discovery failed: ${detail}`, Codes.PROVIDER_HTTP_ERROR)
}
