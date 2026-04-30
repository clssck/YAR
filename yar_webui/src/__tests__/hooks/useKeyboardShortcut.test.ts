import { describe, expect, test } from 'bun:test'
import { BREAKPOINTS } from '../../hooks/useBreakpoint'
import { formatShortcut } from '../../hooks/useKeyboardShortcut'

describe('BREAKPOINTS', () => {
  test('has correct Tailwind default values', () => {
    expect(BREAKPOINTS.sm).toBe(640)
    expect(BREAKPOINTS.md).toBe(768)
    expect(BREAKPOINTS.lg).toBe(1024)
    expect(BREAKPOINTS.xl).toBe(1280)
    expect(BREAKPOINTS['2xl']).toBe(1536)
  })
})

describe('formatShortcut', () => {
  test('formats single key', () => {
    expect(formatShortcut('k', undefined, false)).toBe('K')
  })

  test('formats key with ctrl modifier (non-Mac)', () => {
    expect(formatShortcut('k', { ctrl: true }, false)).toBe('Ctrl+K')
  })

  test('formats key with meta modifier (non-Mac)', () => {
    expect(formatShortcut('k', { meta: true }, false)).toBe('Win+K')
  })

  test('formats key with meta modifier (Mac)', () => {
    expect(formatShortcut('k', { meta: true }, true)).toBe('⌘K')
  })

  test('formats key with multiple modifiers (Mac)', () => {
    expect(formatShortcut('k', { meta: true, shift: true }, true)).toBe('⇧⌘K')
  })

  test('formats key with all modifiers (Mac)', () => {
    expect(formatShortcut('k', { ctrl: true, alt: true, shift: true, meta: true }, true)).toBe(
      '⌃⌥⇧⌘K'
    )
  })

  test('formats key with multiple modifiers (non-Mac)', () => {
    expect(formatShortcut('k', { ctrl: true, shift: true }, false)).toBe('Ctrl+Shift+K')
  })

  test('preserves special key names', () => {
    expect(formatShortcut('Escape', undefined, false)).toBe('Escape')
    expect(formatShortcut('Enter', undefined, false)).toBe('Enter')
    expect(formatShortcut('ArrowUp', undefined, false)).toBe('ArrowUp')
  })

  test('lowercases single character', () => {
    expect(formatShortcut('K', undefined, false)).toBe('K')
    expect(formatShortcut('a', undefined, false)).toBe('A')
  })
})
