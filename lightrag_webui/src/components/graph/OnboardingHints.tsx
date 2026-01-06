import { HelpCircle, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

const ONBOARDING_KEY = 'lightrag-graph-onboarding-complete'

interface OnboardingStep {
  id: string
  title: string
  description: string
  position: 'top-left' | 'bottom-left' | 'top-right' | 'bottom-right'
  targetSelector?: string
}

/**
 * Simple onboarding hints for first-time graph users.
 * Shows 3 tips about search, layout, and interactions.
 */
const OnboardingHints = () => {
  const { t } = useTranslation()
  const [isVisible, setIsVisible] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)

  // Check if onboarding has been completed
  useEffect(() => {
    const completed = localStorage.getItem(ONBOARDING_KEY)
    if (!completed) {
      // Show after a short delay for better UX
      const timer = setTimeout(() => setIsVisible(true), 1500)
      return () => clearTimeout(timer)
    }
  }, [])

  const steps: OnboardingStep[] = [
    {
      id: 'search',
      title: t('graphPanel.onboarding.searchTitle', 'Search Nodes'),
      description: t(
        'graphPanel.onboarding.searchDesc',
        'Use Cmd+K (or Ctrl+K) to quickly search and focus on any node in the graph.'
      ),
      position: 'top-left',
    },
    {
      id: 'controls',
      title: t('graphPanel.onboarding.controlsTitle', 'Graph Controls'),
      description: t(
        'graphPanel.onboarding.controlsDesc',
        'Use the control panel to change layouts, zoom, and toggle settings. Press +/- to zoom, 0 to reset.'
      ),
      position: 'bottom-left',
    },
    {
      id: 'interactions',
      title: t('graphPanel.onboarding.interactionsTitle', 'Node Interactions'),
      description: t(
        'graphPanel.onboarding.interactionsDesc',
        'Click any node to see its properties. Press P to toggle the panel, Escape to deselect.'
      ),
      position: 'top-right',
    },
  ]

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1)
    } else {
      // Mark as complete
      localStorage.setItem(ONBOARDING_KEY, 'true')
      setIsVisible(false)
    }
  }, [currentStep, steps.length])

  const handleSkip = useCallback(() => {
    localStorage.setItem(ONBOARDING_KEY, 'true')
    setIsVisible(false)
  }, [])

  const handleReplay = useCallback(() => {
    setCurrentStep(0)
    setIsVisible(true)
  }, [])

  if (!isVisible) {
    // Show help button to replay
    return (
      <Button
        size="icon"
        variant="ghost"
        className="absolute top-2 right-2 z-20 h-8 w-8 bg-background/60 backdrop-blur-lg hover:bg-background/80"
        onClick={handleReplay}
        tooltip={t('graphPanel.onboarding.showHints', 'Show tips')}
      >
        <HelpCircle className="h-4 w-4 text-muted-foreground" />
      </Button>
    )
  }

  const step = steps[currentStep]

  // Position classes based on step
  const positionClasses = {
    'top-left': 'top-14 left-4',
    'bottom-left': 'bottom-20 left-4',
    'top-right': 'top-14 right-4',
    'bottom-right': 'bottom-20 right-4',
  }

  return (
    <div
      className={cn(
        'absolute z-30 w-72 bg-background/95 backdrop-blur-lg rounded-lg border-2 border-primary/20 shadow-xl p-4',
        'animate-in fade-in slide-in-from-bottom-2 duration-300',
        positionClasses[step.position]
      )}
    >
      {/* Close button */}
      <button
        onClick={handleSkip}
        className="absolute top-2 right-2 p-1 rounded-full hover:bg-primary/10 transition-colors"
        aria-label="Skip onboarding"
      >
        <X className="h-4 w-4 text-muted-foreground" />
      </button>

      {/* Step indicator */}
      <div className="flex gap-1 mb-3">
        {steps.map((_, index) => (
          <div
            key={index}
            className={cn(
              'h-1 flex-1 rounded-full transition-colors',
              index <= currentStep ? 'bg-primary' : 'bg-primary/20'
            )}
          />
        ))}
      </div>

      {/* Content */}
      <h4 className="font-semibold text-sm mb-1.5">{step.title}</h4>
      <p className="text-xs text-muted-foreground mb-4 leading-relaxed">{step.description}</p>

      {/* Actions */}
      <div className="flex justify-between items-center">
        <button
          onClick={handleSkip}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {t('graphPanel.onboarding.skip', "Don't show again")}
        </button>
        <Button size="sm" onClick={handleNext}>
          {currentStep < steps.length - 1
            ? t('graphPanel.onboarding.next', 'Next')
            : t('graphPanel.onboarding.done', 'Got it!')}
        </Button>
      </div>
    </div>
  )
}

export default OnboardingHints
