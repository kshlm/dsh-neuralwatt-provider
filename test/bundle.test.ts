import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * `dsh plugin add` only installs the package as a profile layer when its
 * package.json declares `dsh.bundle`; without it dsh warns "declares no
 * dsh.bundle" and activates no configuration layer (see docs
 * develop/basic/publish). These tests pin that contract.
 */
const root = dirname(dirname(fileURLToPath(import.meta.url)))
const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))

describe('dsh.bundle manifest', () => {
  it('declares a bundle patch so the plugin installs as a profile layer', () => {
    expect(typeof manifest.dsh?.bundle?.patch).toBe('string')
    expect((manifest.dsh.bundle.patch as string).length).toBeGreaterThan(0)
  })

  it('declared patch file exists in the repo', () => {
    const patch: string = manifest.dsh.bundle.patch
    expect(() => readFileSync(join(root, patch), 'utf8')).not.toThrow()
  })

  it('ships the patch file in `files`', () => {
    const patch: string = manifest.dsh.bundle.patch.replace(/^\.\//, '')
    expect(manifest.files).toContain(patch)
  })

  it('bundle patch inserts the plugin by package name', () => {
    const patch = readFileSync(join(root, manifest.dsh.bundle.patch), 'utf8')
    expect(patch).toMatch(/-\s*insert:/)
    expect(patch).toMatch(/\bid:\s*llm-neuralwatt\b/)
    // Rows must reference the installed package by name, not a source path,
    // so Node resolution finds the installed code.
    expect(patch).toMatch(/\bname:\s*dsh-llm-neuralwatt\s*$/m)
  })
})
