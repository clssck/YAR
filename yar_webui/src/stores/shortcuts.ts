import { create } from 'zustand'

export type ShortcutCategory =
  | 'global'
  | 'documents'
  | 'graph'
  | 'retrieval'
  | 'navigation'

export type RegisteredShortcut = {
  id: string
  key: string
  modifiers?: {
    ctrl?: boolean
    alt?: boolean
    shift?: boolean
    meta?: boolean
  }
  description: string
  category: ShortcutCategory
}

type ShortcutStore = {
  shortcuts: Map<string, RegisteredShortcut>
  register: (shortcut: RegisteredShortcut) => void
  unregister: (id: string) => void
  getByCategory: (category: ShortcutCategory) => RegisteredShortcut[]
  getAll: () => RegisteredShortcut[]
}

export const useShortcutStore = create<ShortcutStore>((set, get) => ({
  shortcuts: new Map(),

  register: (shortcut) =>
    set((state) => {
      const newShortcuts = new Map(state.shortcuts)
      newShortcuts.set(shortcut.id, shortcut)
      return { shortcuts: newShortcuts }
    }),

  unregister: (id) =>
    set((state) => {
      const newShortcuts = new Map(state.shortcuts)
      newShortcuts.delete(id)
      return { shortcuts: newShortcuts }
    }),

  getByCategory: (category) => {
    const { shortcuts } = get()
    return Array.from(shortcuts.values()).filter((s) => s.category === category)
  },

  getAll: () => {
    const { shortcuts } = get()
    return Array.from(shortcuts.values())
  },
}))

/**
 * Format a shortcut for display (e.g., "⌘K" or "Ctrl+K")
 */
export function formatShortcutDisplay(
  key: string,
  modifiers?: RegisteredShortcut['modifiers'],
  useMacSymbols = typeof navigator !== 'undefined' &&
    /Mac|iPhone|iPad/.test(navigator.userAgent),
): string {
  const parts: string[] = []

  if (modifiers?.ctrl) parts.push(useMacSymbols ? '⌃' : 'Ctrl')
  if (modifiers?.alt) parts.push(useMacSymbols ? '⌥' : 'Alt')
  if (modifiers?.shift) parts.push(useMacSymbols ? '⇧' : 'Shift')
  if (modifiers?.meta) parts.push(useMacSymbols ? '⌘' : 'Ctrl')

  // Format special keys
  let keyDisplay = key.length === 1 ? key.toUpperCase() : key
  if (key === '?') keyDisplay = '?'
  if (key === 'Escape') keyDisplay = useMacSymbols ? '⎋' : 'Esc'
  if (key === 'Enter') keyDisplay = useMacSymbols ? '⏎' : 'Enter'

  if (useMacSymbols) {
    return parts.join('') + keyDisplay
  }

  parts.push(keyDisplay)
  return parts.join('+')
}
