import { describe, expect, it } from 'vitest'
import * as plugin from '../src/index.ts'

describe('plugin module', () => {
  it('exports the plugin contract', () => {
    expect(plugin.name).toBe('llm-neuralwatt')
    expect(plugin.inject).toEqual(['llm'])
    expect(typeof plugin.apply).toBe('function')
    expect(plugin.Config).toBeDefined()
  })

  it('registers the neuralwatt adapter on apply', () => {
    const registered: { routes: string[] }[] = []
    const ctx: any = {
      get: () => undefined,
      llm: { registerAdapter: (routes: string[]) => { registered.push({ routes }); return { replace() {} } } },
    }
    plugin.apply(ctx, { baseURL: 'https://x.example/v1' })
    expect(registered[0].routes).toEqual(['neuralwatt'])
  })

  it('uses the configured route key', () => {
    const registered: { routes: string[] }[] = []
    const ctx: any = {
      get: () => undefined,
      llm: { registerAdapter: (routes: string[]) => { registered.push({ routes }); return { replace() {} } } },
    }
    plugin.apply(ctx, { route: 'nw' })
    expect(registered[0].routes).toEqual(['nw'])
  })
})
