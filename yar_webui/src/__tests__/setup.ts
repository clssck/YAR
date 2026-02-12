/**
 * Test setup for Bun test runner with happy-dom
 */

import { afterEach } from 'bun:test'
import { GlobalRegistrator } from '@happy-dom/global-registrator'
import { cleanup } from '@testing-library/react'

// Register happy-dom globals (window, document, etc.)
GlobalRegistrator.register()

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
