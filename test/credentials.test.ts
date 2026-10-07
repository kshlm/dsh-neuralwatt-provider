import { describe, expect, it } from 'vitest'
import { makeResolveApiKey } from '../src/credentials.ts'

describe('makeResolveApiKey', () => {
  it('prefers the credentials seam over env', async () => {
    const resolve = makeResolveApiKey({
      apiKeyEnv: 'NEURALWATT_API_KEY',
      resolveCredential: async () => 'sk-from-store',
      env: { NEURALWATT_API_KEY: 'sk-from-env' },
    })
    expect(await resolve()).toBe('sk-from-store')
  })
  it('falls back to env and trims', async () => {
    const resolve = makeResolveApiKey({
      apiKeyEnv: 'NEURALWATT_API_KEY',
      resolveCredential: async () => undefined,
      env: { NEURALWATT_API_KEY: '  sk-env  ' },
    })
    expect(await resolve()).toBe('sk-env')
  })
  it('returns undefined when unset', async () => {
    const resolve = makeResolveApiKey({ apiKeyEnv: 'NEURALWATT_API_KEY', env: {} })
    expect(await resolve()).toBeUndefined()
  })
})
