import { useEffect, useId, useRef } from 'react'
import { type ShortcutCategory, useShortcutStore } from '@/stores/shortcuts'

export type KeyModifiers = {
  ctrl?: boolean
  alt?: boolean
  shift?: boolean
  meta?: boolean // Cmd on Mac, Win on Windows
}

export type ShortcutConfig = {
  key: string
  modifiers?: KeyModifiers
  callback: (event: KeyboardEvent) => void
  /** Prevent default browser behavior */
  preventDefault?: boolean
  /** Stop event propagation */
  stopPropagation?: boolean
  /** Only trigger when no input/textarea is focused */
  ignoreInputs?: boolean
  /** Description for help display (i18n key) */
  description?: string
  /** Category for grouping in help modal */
  category?: ShortcutCategory
}

/**
 * Check if event matches the shortcut configuration
 */
function matchesShortcut(
  event: KeyboardEvent,
  config: ShortcutConfig,
): boolean {
  const { key, modifiers = {} } = config

  // Check key (case-insensitive)
  if (event.key.toLowerCase() !== key.toLowerCase()) {
    return false
  }

  // Check modifiers
  const { ctrl = false, alt = false, shift = false, meta = false } = modifiers

  if (event.ctrlKey !== ctrl) return false
  if (event.altKey !== alt) return false
  if (event.shiftKey !== shift) return false
  if (event.metaKey !== meta) return false

  return true
}

/**
 * Check if an input element is currently focused
 */
function isInputFocused(): boolean {
  const activeElement = document.activeElement
  if (!activeElement) return false

  const tagName = activeElement.tagName.toLowerCase()
  if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
    return true
  }

  // Check for contenteditable
  if (activeElement.getAttribute('contenteditable') === 'true') {
    return true
  }

  return false
}

/**
 * Hook for registering a single keyboard shortcut
 *
 * @example
 * ```tsx
 * useKeyboardShortcut({
 *   key: 'k',
 *   modifiers: { meta: true },
 *   callback: () => setSearchOpen(true),
 *   description: 'shortcutHelp.focusSearch',
 *   category: 'global'
 * })
 * ```
 */
export function useKeyboardShortcut(config: ShortcutConfig) {
  const configRef = useRef(config)
  configRef.current = config
  const id = useId()

  // Register with shortcut store for help modal
  useEffect(() => {
    const { description, category, key, modifiers } = config
    if (description && category) {
      useShortcutStore.getState().register({
        id,
        key,
        modifiers,
        description,
        category,
      })
      return () => useShortcutStore.getState().unregister(id)
    }
  }, [id, config])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const currentConfig = configRef.current

      // Check if we should ignore when input is focused
      if (currentConfig.ignoreInputs !== false && isInputFocused()) {
        return
      }

      if (matchesShortcut(event, currentConfig)) {
        if (currentConfig.preventDefault !== false) {
          event.preventDefault()
        }
        if (currentConfig.stopPropagation) {
          event.stopPropagation()
        }
        currentConfig.callback(event)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
}

/**
 * Hook for registering multiple keyboard shortcuts
 *
 * @example
 * ```tsx
 * useKeyboardShortcuts([
 *   { key: 'k', modifiers: { meta: true }, callback: openSearch },
 *   { key: 'Escape', callback: closeModal },
 *   { key: 's', callback: focusSearch, ignoreInputs: true },
 * ])
 * ```
 */
export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]) {
  const shortcutsRef = useRef(shortcuts)
  shortcutsRef.current = shortcuts

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      for (const config of shortcutsRef.current) {
        // Check if we should ignore when input is focused
        if (config.ignoreInputs !== false && isInputFocused()) {
          continue
        }

        if (matchesShortcut(event, config)) {
          if (config.preventDefault !== false) {
            event.preventDefault()
          }
          if (config.stopPropagation) {
            event.stopPropagation()
          }
          config.callback(event)
          return // Only trigger first matching shortcut
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
}

/**
 * Format a shortcut for display (e.g., "⌘K" or "Ctrl+K")
 */
export function formatShortcut(
  key: string,
  modifiers?: KeyModifiers,
  useMacSymbols = typeof navigator !== 'undefined' &&
    /Mac|iPhone|iPad/.test(navigator.userAgent),
): string {
  const parts: string[] = []

  if (modifiers?.ctrl) parts.push(useMacSymbols ? '⌃' : 'Ctrl')
  if (modifiers?.alt) parts.push(useMacSymbols ? '⌥' : 'Alt')
  if (modifiers?.shift) parts.push(useMacSymbols ? '⇧' : 'Shift')
  if (modifiers?.meta) parts.push(useMacSymbols ? '⌘' : 'Win')

  // Format special keys
  const keyDisplay = key.length === 1 ? key.toUpperCase() : key

  if (useMacSymbols) {
    return parts.join('') + keyDisplay
  }

  parts.push(keyDisplay)
  return parts.join('+')
}

/**
 * Hook that returns a formatted shortcut string
 */
export function useShortcutDisplay(
  key: string,
  modifiers?: KeyModifiers,
): string {
  return formatShortcut(key, modifiers)
}

/**
 * Create a shortcut handler that can be passed to onClick handlers
 */
export function createShortcutHandler(
  shortcut: Omit<ShortcutConfig, 'callback'>,
  callback: () => void,
): ShortcutConfig {
  return {
    ...shortcut,
    callback,
  }
}
