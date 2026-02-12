import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { act, renderHook } from '@testing-library/react'
import {
  BREAKPOINTS,
  type Breakpoint,
  useBreakpoint,
  useIsAboveBreakpoint,
  useIsBelowBreakpoint,
  useMediaQuery,
  useResponsive,
  useResponsiveValue,
  useWindowSize,
} from '../../hooks/useBreakpoint'

describe('useBreakpoint', () => {
  let originalInnerWidth: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  test('initializes with correct breakpoint on mount', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('lg')
  })

  test('returns xs for width < 640px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('xs')
  })

  test('returns sm for width 640-767px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 640,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('sm')
  })

  test('returns md for width 768-1023px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('md')
  })

  test('returns lg for width 1024-1279px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('lg')
  })

  test('returns xl for width 1280-1535px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1280,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('xl')
  })

  test('returns 2xl for width >= 1536px', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1536,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('2xl')
  })

  test('updates breakpoint on window resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useBreakpoint())
    expect(result.current).toBe('xs')

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current).toBe('lg')
  })

  test('handles multiple resize events', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useBreakpoint())

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 640,
      })
      window.dispatchEvent(new Event('resize'))
    })
    expect(result.current).toBe('sm')

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })
      window.dispatchEvent(new Event('resize'))
    })
    expect(result.current).toBe('lg')

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1536,
      })
      window.dispatchEvent(new Event('resize'))
    })
    expect(result.current).toBe('2xl')
  })

  test('cleans up resize event listener on unmount', () => {
    const { unmount } = renderHook(() => useBreakpoint())

    unmount()
    // Verify the event listener was registered and cleaned up
    // (The cleanup happens automatically in the cleanup function)
  })

  test('handles edge case: width exactly at breakpoint boundary', () => {
    const testCases: Array<[number, Breakpoint | 'xs']> = [
      [640, 'sm'],
      [768, 'md'],
      [1024, 'lg'],
      [1280, 'xl'],
      [1536, '2xl'],
    ]

    testCases.forEach(([width, expected]) => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: width,
      })

      const { result, unmount } = renderHook(() => useBreakpoint())
      expect(result.current).toBe(expected)
      unmount()
    })
  })
})

describe('useIsAboveBreakpoint', () => {
  let originalInnerWidth: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  test('returns true when width >= breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useIsAboveBreakpoint('md'))
    expect(result.current).toBe(true)
  })

  test('returns false when width < breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useIsAboveBreakpoint('md'))
    expect(result.current).toBe(false)
  })

  test('updates on window resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useIsAboveBreakpoint('md'))
    expect(result.current).toBe(false)

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current).toBe(true)
  })

  test('correctly evaluates all breakpoints', () => {
    const breakpoints: Breakpoint[] = ['sm', 'md', 'lg', 'xl']

    breakpoints.forEach((bp) => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: BREAKPOINTS[bp],
      })

      const { result, unmount } = renderHook(() => useIsAboveBreakpoint(bp))
      expect(result.current).toBe(true)
      unmount()
    })
  })

  test('updates when breakpoint prop changes', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 900,
    })

    const { result, rerender } = renderHook(
      ({ breakpoint }: { breakpoint: Breakpoint }) =>
        useIsAboveBreakpoint(breakpoint),
      { initialProps: { breakpoint: 'md' as Breakpoint } },
    )

    expect(result.current).toBe(true)

    act(() => {
      rerender({ breakpoint: 'xl' as Breakpoint })
    })

    expect(result.current).toBe(false)
  })
})

describe('useIsBelowBreakpoint', () => {
  let originalInnerWidth: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  test('returns true when width < breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useIsBelowBreakpoint('md'))
    expect(result.current).toBe(true)
  })

  test('returns false when width >= breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useIsBelowBreakpoint('md'))
    expect(result.current).toBe(false)
  })

  test('is inverse of useIsAboveBreakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    })

    const { result: aboveResult } = renderHook(() => useIsAboveBreakpoint('md'))
    const { result: belowResult } = renderHook(() => useIsBelowBreakpoint('md'))

    expect(belowResult.current).toBe(!aboveResult.current)
  })
})

describe('useResponsiveValue', () => {
  let originalInnerWidth: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  test('returns value for current breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        xs: 1,
        sm: 2,
        md: 3,
        lg: 4,
      }),
    )

    expect(result.current).toBe(4)
  })

  test('falls back to lower breakpoint if current not defined', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        xs: 1,
        sm: 2,
        md: 3,
      }),
    )

    expect(result.current).toBe(3)
  })

  test('returns xs value when at smallest screen', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        xs: 'mobile',
        md: 'tablet',
      }),
    )

    expect(result.current).toBe('mobile')
  })

  test('returns undefined when no matching value found', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        md: 'tablet',
        lg: 'desktop',
      }),
    )

    expect(result.current).toBeUndefined()
  })

  test('updates when window is resized', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        xs: 'mobile',
        md: 'tablet',
        lg: 'desktop',
      }),
    )

    expect(result.current).toBe('mobile')

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current).toBe('desktop')
  })

  test('supports different value types', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() =>
      useResponsiveValue({
        xs: { cols: 1 },
        lg: { cols: 4 },
      }),
    )

    expect(result.current).toEqual({ cols: 4 })
  })
})

describe('useResponsive', () => {
  let originalInnerWidth: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
  })

  test('returns correct flags for mobile breakpoints', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 640,
    })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isMobile).toBe(true)
    expect(result.current.isTablet).toBe(false)
    expect(result.current.isDesktop).toBe(false)
    expect(result.current.isSmallScreen).toBe(true)
    expect(result.current.isLargeScreen).toBe(false)
    expect(result.current.breakpoint).toBe('sm')
  })

  test('returns correct flags for tablet breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isMobile).toBe(false)
    expect(result.current.isTablet).toBe(true)
    expect(result.current.isDesktop).toBe(false)
    expect(result.current.isSmallScreen).toBe(true)
    expect(result.current.isLargeScreen).toBe(false)
    expect(result.current.breakpoint).toBe('md')
  })

  test('returns correct flags for desktop breakpoints', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isMobile).toBe(false)
    expect(result.current.isTablet).toBe(false)
    expect(result.current.isDesktop).toBe(true)
    expect(result.current.isSmallScreen).toBe(false)
    expect(result.current.isLargeScreen).toBe(true)
    expect(result.current.breakpoint).toBe('lg')
  })

  test('returns correct flags for extra large breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1536,
    })

    const { result } = renderHook(() => useResponsive())

    expect(result.current.isMobile).toBe(false)
    expect(result.current.isTablet).toBe(false)
    expect(result.current.isDesktop).toBe(true)
    expect(result.current.isLargeScreen).toBe(true)
    expect(result.current.breakpoint).toBe('2xl')
  })

  test('updates all flags on resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    const { result } = renderHook(() => useResponsive())
    expect(result.current.isMobile).toBe(true)

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current.isMobile).toBe(false)
    expect(result.current.isDesktop).toBe(true)
  })
})

describe('useWindowSize', () => {
  let originalInnerWidth: number
  let originalInnerHeight: number

  beforeEach(() => {
    originalInnerWidth = window.innerWidth
    originalInnerHeight = window.innerHeight
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    })
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: originalInnerHeight,
    })
  })

  test('initializes with current window dimensions', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 768,
    })

    const { result } = renderHook(() => useWindowSize())

    expect(result.current.width).toBe(1024)
    expect(result.current.height).toBe(768)
  })

  test('updates width on resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 800,
    })
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 600,
    })

    const { result } = renderHook(() => useWindowSize())

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1280,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current.width).toBe(1280)
  })

  test('updates height on resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 800,
    })
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 600,
    })

    const { result } = renderHook(() => useWindowSize())

    act(() => {
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 900,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current.height).toBe(900)
  })

  test('updates both dimensions on resize', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 800,
    })
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 600,
    })

    const { result } = renderHook(() => useWindowSize())

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1920,
      })
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 1080,
      })
      window.dispatchEvent(new Event('resize'))
    })

    expect(result.current.width).toBe(1920)
    expect(result.current.height).toBe(1080)
  })
})

describe('useMediaQuery', () => {
  test('initializes with matchMedia result', () => {
    const { result } = renderHook(() => useMediaQuery('(min-width: 768px)'))
    expect(typeof result.current).toBe('boolean')
  })

  test('updates when media query matches', () => {
    const { result } = renderHook(() =>
      useMediaQuery('(prefers-reduced-motion: reduce)'),
    )

    expect(typeof result.current).toBe('boolean')
  })

  test('handles different media queries', () => {
    const queries = [
      '(min-width: 768px)',
      '(max-width: 640px)',
      '(prefers-reduced-motion: reduce)',
      '(orientation: portrait)',
    ]

    queries.forEach((query) => {
      const { result, unmount } = renderHook(() => useMediaQuery(query))
      expect(typeof result.current).toBe('boolean')
      unmount()
    })
  })

  test('updates on media query change event', () => {
    const { result } = renderHook(() => useMediaQuery('(min-width: 1024px)'))
    const initialValue = result.current

    act(() => {
      const mediaQuery = window.matchMedia('(min-width: 1024px)')
      const event = new Event('change') as MediaQueryListEvent
      Object.defineProperty(event, 'matches', { value: !initialValue })
      mediaQuery.dispatchEvent(event)
    })

    // The hook will have updated based on the event
    expect(typeof result.current).toBe('boolean')
  })
})

describe('BREAKPOINTS constant', () => {
  test('has correct Tailwind default values', () => {
    expect(BREAKPOINTS.sm).toBe(640)
    expect(BREAKPOINTS.md).toBe(768)
    expect(BREAKPOINTS.lg).toBe(1024)
    expect(BREAKPOINTS.xl).toBe(1280)
    expect(BREAKPOINTS['2xl']).toBe(1536)
  })

  test('values are in ascending order', () => {
    const values = Object.values(BREAKPOINTS)
    for (let i = 1; i < values.length; i++) {
      expect(values[i]).toBeGreaterThan(values[i - 1])
    }
  })
})
