import { describe, expect, it } from 'vitest'
import { LlmError } from '@deepseek-ai/dsh-llm'
import { Codes, discoveryError, httpError, missingCredential, protocolError, unknownModel } from '../src/errors.ts'

describe('error helpers', () => {
  it('builds a provider HTTP error with a stable code', () => {
    const e = httpError(503, 'upstream down')
    expect(e).toBeInstanceOf(LlmError)
    expect(e.code).toBe(Codes.PROVIDER_HTTP_ERROR)
    expect(e.message).toContain('503')
    expect(e.message).toContain('upstream down')
  })
  it('builds a protocol error', () => {
    expect(protocolError('bad sse').code).toBe(Codes.PROVIDER_PROTOCOL_ERROR)
  })
  it('builds a missing-credential error naming the ref', () => {
    const e = missingCredential('NEURALWATT_API_KEY')
    expect(e.code).toBe(Codes.MISSING_CREDENTIAL)
    expect(e.message).toContain('NEURALWATT_API_KEY')
  })
  it('builds an unknown-model error', () => {
    const e = unknownModel('neuralwatt', 'nope', 'not in catalog')
    expect(e.code).toBe(Codes.UNKNOWN_MODEL)
    expect(e.message).toContain('nope')
  })
  it('builds a discovery error', () => {
    expect(discoveryError('boom').code).toBe(Codes.PROVIDER_HTTP_ERROR)
  })
})
