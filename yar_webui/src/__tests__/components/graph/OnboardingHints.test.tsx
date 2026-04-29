/**
 * Tests for OnboardingHints — the first-time graph user tour.
 *
 * Behaviour under test:
 *  - localStorage gate (legacy + structured value)
 *  - rAF-deferred mount
 *  - Esc / Skip / Done dismiss + persistence
 *  - external replay event
 *  - dialog ARIA attributes (role/labelledby/describedby)
 */
import '../../setup'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

// Defensive mock so that t(key, fallback) returns the fallback string
// regardless of mock leakage from other test files (Bun's mock.module is
// process-wide). Our component uses inline fallbacks as ground truth.
mock.module('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}))

import { fireEvent, render, waitFor } from '@testing-library/react'
import OnboardingHints, {
  ONBOARDING_KEY,
  ONBOARDING_RESET_EVENT,
} from '@/components/graph/OnboardingHintsImpl'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  localStorage.clear()
})

describe('OnboardingHints', () => {
  test('renders nothing when structured state marks completed', async () => {
    localStorage.setItem(
      ONBOARDING_KEY,
      JSON.stringify({ completed: true, lastStep: 0 }),
    )
    const { container } = render(<OnboardingHints />)
    // Dialog is gated synchronously; no async tick needed.
    expect(container.querySelector('[role="dialog"]')).toBeNull()
  })

  test('renders nothing when legacy "true" flag is present', async () => {
    localStorage.setItem(ONBOARDING_KEY, 'true')
    const { container } = render(<OnboardingHints />)
    // Dialog is gated synchronously; no async tick needed.
    expect(container.querySelector('[role="dialog"]')).toBeNull()
  })

  test('renders dialog with correct ARIA wiring when state is absent', async () => {
    const { container } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    const dialog = container.querySelector('[role="dialog"]') as HTMLElement
    const labelledBy = dialog.getAttribute('aria-labelledby')
    const describedBy = dialog.getAttribute('aria-describedby')
    expect(labelledBy).toBeTruthy()
    expect(describedBy).toBeTruthy()
    expect(dialog.querySelector(`#${labelledBy}`)?.textContent).toBe(
      'Search Nodes',
    )
    expect(dialog.querySelector(`#${describedBy}`)?.textContent).toContain(
      'Cmd+K',
    )
    // Non-blocking — graph remains interactive
    expect(dialog.getAttribute('aria-modal')).toBe('false')
  })

  test('Esc dismisses and persists completed state', async () => {
    const { container } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).toBeNull()
    })
    const stored = JSON.parse(
      localStorage.getItem(ONBOARDING_KEY) ?? 'null',
    ) as { completed: boolean; lastStep: number } | null
    expect(stored?.completed).toBe(true)
  })

  test('clicking Skip dismisses and persists completed state', async () => {
    const { container, getByText } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    fireEvent.click(getByText("Don't show again"))
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).toBeNull()
    })
    const stored = JSON.parse(
      localStorage.getItem(ONBOARDING_KEY) ?? 'null',
    ) as { completed: boolean; lastStep: number } | null
    expect(stored?.completed).toBe(true)
  })

  test('Next advances steps and final step writes completed', async () => {
    const { container, getByRole } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })

    // Step 1 → click Next
    let nextBtn = getByRole('button', { name: 'Next' })
    fireEvent.click(nextBtn)
    await waitFor(() => {
      expect(container.querySelector('h4')?.textContent).toBe('Graph Controls')
    })
    let stored = JSON.parse(localStorage.getItem(ONBOARDING_KEY) ?? 'null') as {
      completed: boolean
      lastStep: number
    } | null
    expect(stored).toEqual({ completed: false, lastStep: 1 })

    // Step 2 → click Next
    nextBtn = getByRole('button', { name: 'Next' })
    fireEvent.click(nextBtn)
    await waitFor(() => {
      expect(container.querySelector('h4')?.textContent).toBe(
        'Node Interactions',
      )
    })
    stored = JSON.parse(localStorage.getItem(ONBOARDING_KEY) ?? 'null') as {
      completed: boolean
      lastStep: number
    } | null
    expect(stored).toEqual({ completed: false, lastStep: 2 })

    // Last step shows "Got it!" and dismisses
    const doneBtn = getByRole('button', { name: 'Got it!' })
    fireEvent.click(doneBtn)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).toBeNull()
    })
    stored = JSON.parse(localStorage.getItem(ONBOARDING_KEY) ?? 'null') as {
      completed: boolean
      lastStep: number
    } | null
    expect(stored?.completed).toBe(true)
  })

  test('replay event re-shows the panel from step 0 even after completion', async () => {
    localStorage.setItem(
      ONBOARDING_KEY,
      JSON.stringify({ completed: true, lastStep: 0 }),
    )
    const { container } = render(<OnboardingHints />)
    // Dialog is gated synchronously; no async tick needed.
    expect(container.querySelector('[role="dialog"]')).toBeNull()

    fireEvent(window, new Event(ONBOARDING_RESET_EVENT))
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    expect(container.querySelector('h4')?.textContent).toBe('Search Nodes')
  })

  test('resumes at lastStep when state is interrupted (not completed)', async () => {
    localStorage.setItem(
      ONBOARDING_KEY,
      JSON.stringify({ completed: false, lastStep: 1 }),
    )
    const { container } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    expect(container.querySelector('h4')?.textContent).toBe('Graph Controls')
  })

  test('dialog contains focusable Skip and action buttons', async () => {
    const { container } = render(<OnboardingHints />)
    await waitFor(() => {
      expect(container.querySelector('[role="dialog"]')).not.toBeNull()
    })
    const dialog = container.querySelector('[role="dialog"]') as HTMLElement
    const buttons = dialog.querySelectorAll('button')
    // Close X, Skip, Next — three actionable controls
    expect(buttons.length).toBe(3)
  })
})
