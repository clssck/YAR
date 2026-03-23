/**
 * Test setup for Bun test runner with happy-dom
 */

import { afterEach } from 'bun:test'
import { GlobalRegistrator } from '@happy-dom/global-registrator'
import { cleanup } from '@testing-library/react'

// Register happy-dom globals (window, document, etc.)
GlobalRegistrator.register()

if (document.doctype == null) {
  const doctype = document.implementation.createDocumentType('html', '', '')
  document.insertBefore(doctype, document.documentElement)
}

const IGNORED_TEST_MESSAGES = [
  "KaTeX doesn't work in quirks mode.",
  'i18next is maintained with support from Locize',
]

const silenceKnownTestNoise = (
  method: 'log' | 'info' | 'warn' | 'error',
) => {
  const original = console[method].bind(console)
  console[method] = ((...args: unknown[]) => {
    const first = args[0]
    if (
      typeof first === 'string' &&
      IGNORED_TEST_MESSAGES.some((message) => first.includes(message))
    ) {
      return
    }
    original(...args)
  }) as typeof console[typeof method]
}

silenceKnownTestNoise('log')
silenceKnownTestNoise('info')
silenceKnownTestNoise('warn')
silenceKnownTestNoise('error')

await import('../i18n')

// Cleanup after each test to prevent DOM pollution
afterEach(() => {
  cleanup()
})

// Mock matchMedia for responsive tests
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
})

// Mock ResizeObserver
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
window.ResizeObserver = ResizeObserverMock

// Mock IntersectionObserver
class IntersectionObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
window.IntersectionObserver =
  IntersectionObserverMock as unknown as typeof IntersectionObserver
