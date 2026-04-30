import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { StoreApi, UseBoundStore } from 'zustand'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function randomColor() {
  const digits = '0123456789abcdef'
  let code = '#'
  for (let i = 0; i < 6; i++) {
    code += digits.charAt(Math.floor(Math.random() * 16))
  }
  return code
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Returns a shallow copy of `obj` with the listed keys removed. Use instead of
 * destructure-and-rest patterns when the only purpose of the destructure is to
 * exclude keys from the rest spread.
 */
export function omit<T extends object, K extends keyof T>(obj: T, keys: readonly K[]): Omit<T, K> {
  const next = { ...obj }
  for (const key of keys) {
    delete next[key]
  }
  return next
}

/**
 * Pairs each item in an append-only string list with a stable React key derived
 * from content + per-content occurrence index. Lets duplicate strings coexist
 * without falling back to the array index as a key.
 */
export function withStableKeys(items: string[]): Array<{ key: string; value: string }> {
  const seen = new Map<string, number>()
  return items.map((value) => {
    const occurrence = (seen.get(value) ?? 0) + 1
    seen.set(value, occurrence)
    return { key: `${value}\u0000${occurrence}`, value }
  })
}

/**
 * Creates a throttled function that limits how often the original function can be called
 * @param fn The function to throttle
 * @param delay The delay in milliseconds
 * @returns A throttled version of the function
 */
export function throttle<T extends (...args: never[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function (this: unknown, ...args: Parameters<T>) {
    const now = Date.now()
    const remaining = delay - (now - lastCall)

    if (remaining <= 0) {
      // If enough time has passed, execute the function immediately
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      lastCall = now
      fn.apply(this, args)
    } else if (!timeoutId) {
      // If not enough time has passed, set a timeout to execute after the remaining time
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        timeoutId = null
        fn.apply(this, args)
      }, remaining)
    }
  }
}

type WithSelectors<S> = S extends { getState: () => infer T }
  ? S & { use: { [K in keyof T]: () => T[K] } }
  : never

export const createSelectors = <S extends UseBoundStore<StoreApi<object>>>(_store: S) => {
  type State = ReturnType<S['getState']>
  const store = _store as WithSelectors<typeof _store>
  store.use = {} as { [K in keyof State]: () => State[K] }
  for (const k of Object.keys(store.getState())) {
    const key = k as keyof State & string
    ;(store.use as Record<string, () => unknown>)[key] = () =>
      store((s) => (s as Record<string, unknown>)[key])
  }

  return store
}
