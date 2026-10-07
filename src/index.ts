import type { Context } from '@deepseek-ai/cordis'
import { resolveImageAttachmentAccess } from '@deepseek-ai/dsh-llm'
import type { AttachmentStore, ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { Config, resolveEffectiveConfig, type Config as PluginConfig } from './config.js'
import { makeResolveApiKey } from './credentials.js'
import { NeuralwattAdapter } from './adapter.js'

export const name = 'llm-neuralwatt'
export const inject = ['llm']
export { Config }
export type { PluginConfig as ConfigType }

async function readDataUrl(path: string, mediaType: string): Promise<string | undefined> {
  try {
    const { readFile } = await import('node:fs/promises')
    const bytes = await readFile(path)
    return `data:${mediaType};base64,${Buffer.from(bytes).toString('base64')}`
  } catch {
    return undefined
  }
}

export function apply(ctx: Context, config: PluginConfig): void {
  const eff = resolveEffectiveConfig(config, process.env)

  const resolveApiKey = makeResolveApiKey({
    apiKeyEnv: eff.apiKeyEnv,
    resolveCredential: async (ref) => {
      const credentials = ctx.get('credentials') as { resolve(r: string): Promise<{ value?: string } | undefined> } | undefined
      return (await credentials?.resolve(ref))?.value
    },
    env: process.env,
  })

  const imageDataUrl = async (ref: ImageAttachmentRef): Promise<string | undefined> => {
    const attachments = ctx.get('attachments') as AttachmentStore | undefined
    if (attachments === undefined) return undefined
    const fs = ctx.get('fs') as { processPathFromHostPath?: (hostPath: string) => string | undefined } | undefined
    try {
      const access = resolveImageAttachmentAccess(
        attachments,
        (hostPath) => fs?.processPathFromHostPath?.(hostPath) ?? hostPath,
        ref,
      )
      return access === undefined ? undefined : readDataUrl(access.readonlyPath, ref.mediaType)
    } catch {
      return undefined
    }
  }

  const adapter = new NeuralwattAdapter({
    route: eff.route,
    displayName: eff.displayName,
    baseURL: eff.baseURL,
    apiKeyEnv: eff.apiKeyEnv,
    resolveApiKey,
    imageDataUrl,
  })
  ctx.llm.registerAdapter([eff.route], adapter)
}
