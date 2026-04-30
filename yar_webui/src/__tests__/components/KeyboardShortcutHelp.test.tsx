/**
 * Tests for KeyboardShortcutHelp — focuses on the new replay-onboarding
 * footer action. Other shortcut listing behaviour is exercised indirectly
 * through the underlying useShortcutStore.
 */
import '../setup'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

// Defensive mock so that t(key, fallback) returns the fallback string,
// regardless of mock leakage from other test files.
mock.module('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

import { fireEvent, render, waitFor } from '@testing-library/react'
import { ONBOARDING_KEY, ONBOARDING_RESET_EVENT } from '@/components/graph/OnboardingHints'
import KeyboardShortcutHelp from '@/components/KeyboardShortcutHelp'

const openHelpDialog = (root: HTMLElement) => {
  // Trigger is the first button in the DialogTrigger; query structurally
  // rather than by text so the assertion is robust against i18n state.
  const trigger = root.querySelector('button')
  if (!trigger) throw new Error('trigger not found')
  fireEvent.click(trigger)
}

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  localStorage.clear()
})

describe('KeyboardShortcutHelp', () => {
  test('replay button clears onboarding key and dispatches reset event', async () => {
    localStorage.setItem(ONBOARDING_KEY, JSON.stringify({ completed: true, lastStep: 0 }))

    let dispatched = 0
    const handler = () => {
      dispatched += 1
    }
    window.addEventListener(ONBOARDING_RESET_EVENT, handler)

    try {
      const { container } = render(<KeyboardShortcutHelp />)
      openHelpDialog(container)

      // Radix Dialog renders into a portal; query the document root.
      let replayBtn: HTMLButtonElement | null = null
      await waitFor(() => {
        replayBtn = document.body.querySelector('button.mt-3.w-full') as HTMLButtonElement | null
        expect(replayBtn).not.toBeNull()
        expect(replayBtn?.textContent?.trim()).toBe('Replay onboarding tour')
      })

      fireEvent.click(replayBtn!)

      expect(localStorage.getItem(ONBOARDING_KEY)).toBeNull()
      expect(dispatched).toBe(1)
    } finally {
      window.removeEventListener(ONBOARDING_RESET_EVENT, handler)
    }
  })

  test('replay button still dispatches event when no onboarding state exists', async () => {
    let dispatched = 0
    const handler = () => {
      dispatched += 1
    }
    window.addEventListener(ONBOARDING_RESET_EVENT, handler)

    try {
      const { container } = render(<KeyboardShortcutHelp />)
      openHelpDialog(container)

      let replayBtn: HTMLButtonElement | null = null
      await waitFor(() => {
        replayBtn = document.body.querySelector('button.mt-3.w-full') as HTMLButtonElement | null
        expect(replayBtn).not.toBeNull()
      })

      fireEvent.click(replayBtn!)
      expect(dispatched).toBe(1)
    } finally {
      window.removeEventListener(ONBOARDING_RESET_EVENT, handler)
    }
  })
})
