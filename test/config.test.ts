import { describe, expect, it } from 'vitest'
import {
  DEFAULT_API_KEY_ENV,
  DEFAULT_BASE_URL,
  DEFAULT_DISPLAY_NAME,
  DEFAULT_ROUTE,
  resolveEffectiveConfig,
} from '../src/config.ts'

describe('resolveEffectiveConfig', () => {
  it('uses built-in defaults when nothing is set', () => {
    const eff = resolveEffectiveConfig({}, {})
    expect(eff).toEqual({
      route: DEFAULT_ROUTE,
      displayName: DEFAULT_DISPLAY_NAME,
      baseURL: DEFAULT_BASE_URL,
      apiKeyEnv: DEFAULT_API_KEY_ENV,
    })
  })

  it('prefers config.baseURL over env over default', () => {
    expect(resolveEffectiveConfig({ baseURL: 'https://cfg.example/v1' }, { NEURALWATT_BASE_URL: 'https://env.example/v1' }).baseURL)
      .toBe('https://cfg.example/v1')
    expect(resolveEffectiveConfig({}, { NEURALWATT_BASE_URL: 'https://env.example/v1' }).baseURL)
      .toBe('https://env.example/v1')
    expect(resolveEffectiveConfig({}, {}).baseURL).toBe(DEFAULT_BASE_URL)
  })

  it('honors route, displayName, apiKeyEnv overrides', () => {
    const eff = resolveEffectiveConfig({ route: 'nw', displayName: 'NW', apiKeyEnv: 'MY_KEY' }, {})
    expect(eff.route).toBe('nw')
    expect(eff.displayName).toBe('NW')
    expect(eff.apiKeyEnv).toBe('MY_KEY')
  })
})
