import { KeyboardIcon } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/Dialog'
import { useKeyboardShortcut } from '@/hooks/useKeyboardShortcut'
import {
  formatShortcutDisplay,
  type RegisteredShortcut,
  type ShortcutCategory,
  useShortcutStore,
} from '@/stores/shortcuts'

const CATEGORY_ORDER: ShortcutCategory[] = ['global', 'navigation', 'documents', 'graph', 'retrieval']

const CATEGORY_LABELS: Record<ShortcutCategory, string> = {
  global: 'shortcutHelp.category.global',
  navigation: 'shortcutHelp.category.navigation',
  documents: 'shortcutHelp.category.documents',
  graph: 'shortcutHelp.category.graph',
  retrieval: 'shortcutHelp.category.retrieval',
}

function ShortcutRow({ shortcut }: { shortcut: RegisteredShortcut }) {
  const { t } = useTranslation()
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{t(shortcut.description)}</span>
      <kbd className="ml-4 inline-flex h-6 min-w-[24px] items-center justify-center rounded border border-border bg-muted px-1.5 font-mono text-xs font-medium text-foreground">
        {formatShortcutDisplay(shortcut.key, shortcut.modifiers)}
      </kbd>
    </div>
  )
}

function ShortcutSection({ category, shortcuts }: { category: ShortcutCategory; shortcuts: RegisteredShortcut[] }) {
  const { t } = useTranslation()

  if (shortcuts.length === 0) return null

  return (
    <div className="mb-4">
      <h3 className="mb-2 text-sm font-semibold text-foreground">
        {t(CATEGORY_LABELS[category])}
      </h3>
      <div className="space-y-0.5">
        {shortcuts.map((shortcut) => (
          <ShortcutRow key={shortcut.id} shortcut={shortcut} />
        ))}
      </div>
    </div>
  )
}

export function KeyboardShortcutHelp() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const shortcuts = useShortcutStore((state) => state.shortcuts)

  // Register "?" shortcut to open help
  useKeyboardShortcut({
    key: '?',
    modifiers: { shift: true },
    callback: useCallback(() => setOpen(true), []),
    description: 'shortcutHelp.openHelp',
    ignoreInputs: true,
  })

  // Register Escape to close
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [open])

  // Group shortcuts by category
  const shortcutsByCategory = CATEGORY_ORDER.reduce(
    (acc, category) => {
      acc[category] = Array.from(shortcuts.values()).filter((s) => s.category === category)
      return acc
    },
    {} as Record<ShortcutCategory, RegisteredShortcut[]>
  )

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0"
          title={t('shortcutHelp.title')}
        >
          <KeyboardIcon className="h-4 w-4" />
          <span className="sr-only">{t('shortcutHelp.title')}</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[80vh] max-w-lg overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyboardIcon className="h-5 w-5" />
            {t('shortcutHelp.title')}
          </DialogTitle>
        </DialogHeader>
        <div className="mt-4">
          {CATEGORY_ORDER.map((category) => (
            <ShortcutSection
              key={category}
              category={category}
              shortcuts={shortcutsByCategory[category]}
            />
          ))}
          {shortcuts.size === 0 && (
            <p className="text-center text-sm text-muted-foreground py-8">
              {t('shortcutHelp.noShortcuts')}
            </p>
          )}
        </div>
        <div className="mt-4 border-t pt-4">
          <p className="text-xs text-muted-foreground text-center">
            {t('shortcutHelp.hint')}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default KeyboardShortcutHelp
