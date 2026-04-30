import { X } from 'lucide-react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

export const ONBOARDING_KEY = 'yar-graph-onboarding-complete'
export const ONBOARDING_RESET_EVENT = 'yar-onboarding-reset'

interface OnboardingStep {
  id: string
  title: string
  description: string
  position: 'top-left' | 'bottom-left' | 'top-right' | 'bottom-right'
}

interface OnboardingState {
  completed: boolean
  lastStep: number
}

const readState = (): OnboardingState | null => {
  try {
    const raw = localStorage.getItem(ONBOARDING_KEY)
    if (!raw) return null
    // Backward compatibility with the previous boolean-only flag
    if (raw === 'true') return { completed: true, lastStep: 0 }
    const parsed = JSON.parse(raw) as Partial<OnboardingState>
    if (typeof parsed?.completed === 'boolean' && typeof parsed?.lastStep === 'number') {
      return parsed as OnboardingState
    }
    return null
  } catch {
    return null
  }
}

const writeState = (state: OnboardingState) => {
  try {
    localStorage.setItem(ONBOARDING_KEY, JSON.stringify(state))
  } catch {
    // localStorage may be unavailable (private mode, quota); fail silently
  }
}

/**
 * Non-blocking onboarding tour for first-time graph users.
 *
 * Initial visibility is derived synchronously from localStorage so the
 * dialog appears in the same paint as its parent (no FOUC, no arbitrary
 * mount delay). Replay is exposed through KeyboardShortcutHelp; this
 * component does not render any persistent affordance when dismissed.
 */
const OnboardingHints = () => {
  const { t } = useTranslation()
  const [isVisible, setIsVisible] = useState(() => {
    const state = readState()
    return !state?.completed
  })
  const [currentStep, setCurrentStep] = useState(() => readState()?.lastStep ?? 0)
  const dialogRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const titleId = useId()
  const descId = useId()

  // External replay handshake (dispatched by KeyboardShortcutHelp)
  useEffect(() => {
    const handleReset = () => {
      setCurrentStep(0)
      setIsVisible(true)
    }
    window.addEventListener(ONBOARDING_RESET_EVENT, handleReset)
    return () => window.removeEventListener(ONBOARDING_RESET_EVENT, handleReset)
  }, [])

  const steps: OnboardingStep[] = [
    {
      id: 'search',
      title: t('graphPanel.onboarding.searchTitle', 'Search Nodes'),
      description: t(
        'graphPanel.onboarding.searchDesc',
        'Use Cmd+K (or Ctrl+K) to quickly search and focus on any node in the graph.'
      ),
      position: 'top-left'
    },
    {
      id: 'controls',
      title: t('graphPanel.onboarding.controlsTitle', 'Graph Controls'),
      description: t(
        'graphPanel.onboarding.controlsDesc',
        'Use the control panel to change layouts, zoom, and toggle settings. Press +/- to zoom, 0 to reset.'
      ),
      position: 'bottom-left'
    },
    {
      id: 'interactions',
      title: t('graphPanel.onboarding.interactionsTitle', 'Node Interactions'),
      description: t(
        'graphPanel.onboarding.interactionsDesc',
        'Click any node to see its properties. Press P to toggle the panel, Escape to deselect.'
      ),
      position: 'top-right'
    }
  ]

  const handleSkip = useCallback(() => {
    writeState({ completed: true, lastStep: 0 })
    setIsVisible(false)
  }, [])

  const handleNext = useCallback(() => {
    setCurrentStep((prev) => {
      const next = prev + 1
      if (next >= steps.length) {
        writeState({ completed: true, lastStep: 0 })
        setIsVisible(false)
        return 0
      }
      writeState({ completed: false, lastStep: next })
      return next
    })
  }, [steps.length])

  // Focus management: capture previous focus, focus dialog on appear,
  // restore previous focus on dismiss.
  useEffect(() => {
    if (!isVisible) return
    const previous = document.activeElement
    previousFocusRef.current = previous instanceof HTMLElement ? previous : null
    // Focus the dialog container so screen readers announce it and keyboard
    // users can immediately Tab into the actions.
    dialogRef.current?.focus()
    return () => {
      previousFocusRef.current?.focus()
    }
  }, [isVisible])

  // Esc to dismiss; Tab cycles within the panel when focus is inside it.
  useEffect(() => {
    if (!isVisible) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        handleSkip()
        return
      }
      if (e.key !== 'Tab') return
      const root = dialogRef.current
      if (!root) return
      const active = document.activeElement
      // Soft trap: only intercept Tab when focus is already inside the panel.
      // Click on the graph moves focus elsewhere; Tab there behaves normally.
      if (!root.contains(active) && active !== root) return
      const focusable = Array.from(
        root.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      ).filter((el) => !el.hasAttribute('aria-hidden'))
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey) {
        if (active === first || active === root) {
          e.preventDefault()
          last.focus()
        }
      } else {
        if (active === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isVisible, handleSkip])

  if (!isVisible) return null

  const step = steps[currentStep]
  const isLast = currentStep === steps.length - 1

  // Desktop position lookup. On mobile (<sm) the panel is bottom-centered.
  const desktopPosition = {
    'top-left': 'sm:top-14 sm:left-4',
    'bottom-left': 'sm:bottom-20 sm:left-4',
    'top-right': 'sm:top-14 sm:right-4',
    'bottom-right': 'sm:bottom-20 sm:right-4'
  }

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="false"
      aria-labelledby={titleId}
      aria-describedby={descId}
      tabIndex={-1}
      className={cn(
        'absolute z-30 w-72 max-w-[calc(100vw-2rem)] bg-background/95 backdrop-blur-lg rounded-lg border-2 border-primary/20 shadow-xl p-4 outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'animate-in fade-in slide-in-from-bottom-2 duration-300',
        // Mobile: bottom-centered
        'bottom-4 left-1/2 -translate-x-1/2',
        // Desktop reset + position
        'sm:left-auto sm:right-auto sm:translate-x-0 sm:bottom-auto sm:top-auto',
        desktopPosition[step.position]
      )}
    >
      <button
        type="button"
        onClick={handleSkip}
        className="hover:bg-primary/10 absolute top-2 right-2 rounded-full p-1 transition-colors"
        aria-label={t('graphPanel.onboarding.close', 'Close onboarding')}
      >
        <X className="text-muted-foreground h-4 w-4" />
      </button>

      {/* Step indicator */}
      <div className="mb-3 flex gap-1" aria-hidden="true">
        {steps.map((s, idx) => (
          <div
            key={s.id}
            className={cn(
              'h-1 flex-1 rounded-full transition-colors',
              idx <= currentStep ? 'bg-primary' : 'bg-primary/20'
            )}
          />
        ))}
      </div>

      <h4 id={titleId} className="mb-1.5 text-sm font-semibold">
        {step.title}
      </h4>
      <p id={descId} className="text-muted-foreground mb-4 text-xs leading-relaxed">
        {step.description}
      </p>

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={handleSkip}
          className="text-muted-foreground hover:text-foreground text-xs transition-colors"
        >
          {t('graphPanel.onboarding.skip', "Don't show again")}
        </button>
        <Button size="sm" onClick={handleNext}>
          {isLast
            ? t('graphPanel.onboarding.done', 'Got it!')
            : t('graphPanel.onboarding.next', 'Next')}
        </Button>
      </div>
    </div>
  )
}

export default OnboardingHints
