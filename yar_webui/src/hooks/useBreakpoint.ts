import { useEffect, useState } from 'react'

/**
 * Tailwind CSS default breakpoints (in pixels)
 * These match Tailwind's default configuration
 */
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const

export type Breakpoint = keyof typeof BREAKPOINTS
export type BreakpointOrBelow = Breakpoint | 'xs'

/**
 * Get the current breakpoint based on window width
 */
function getCurrentBreakpoint(width: number): BreakpointOrBelow {
  if (width >= BREAKPOINTS['2xl']) return '2xl'
  if (width >= BREAKPOINTS.xl) return 'xl'
  if (width >= BREAKPOINTS.lg) return 'lg'
  if (width >= BREAKPOINTS.md) return 'md'
  if (width >= BREAKPOINTS.sm) return 'sm'
  return 'xs'
}

/**
 * Hook that returns the current Tailwind breakpoint
 *
 * @example
 * ```tsx
 * const breakpoint = useBreakpoint()
 * // Returns: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl'
 *
 * return breakpoint === 'xs' ? <MobileView /> : <DesktopView />
 * ```
 */
export function useBreakpoint(): BreakpointOrBelow {
  const [breakpoint, setBreakpoint] = useState<BreakpointOrBelow>(() => {
    if (typeof window === 'undefined') return 'lg' // SSR fallback
    return getCurrentBreakpoint(window.innerWidth)
  })

  useEffect(() => {
    const handleResize = () => {
      setBreakpoint(getCurrentBreakpoint(window.innerWidth))
    }

    // Set initial value
    handleResize()

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return breakpoint
}

/**
 * Hook that returns true if the current viewport is at or above the specified breakpoint
 *
 * @example
 * ```tsx
 * const isDesktop = useIsAboveBreakpoint('lg')
 * const isMobile = !useIsAboveBreakpoint('md')
 * ```
 */
export function useIsAboveBreakpoint(breakpoint: Breakpoint): boolean {
  const [isAbove, setIsAbove] = useState(() => {
    if (typeof window === 'undefined') return true // SSR fallback
    return window.innerWidth >= BREAKPOINTS[breakpoint]
  })

  useEffect(() => {
    const handleResize = () => {
      setIsAbove(window.innerWidth >= BREAKPOINTS[breakpoint])
    }

    handleResize()

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [breakpoint])

  return isAbove
}

/**
 * Hook that returns true if the current viewport is below the specified breakpoint
 *
 * @example
 * ```tsx
 * const isMobile = useIsBelowBreakpoint('md')
 * ```
 */
export function useIsBelowBreakpoint(breakpoint: Breakpoint): boolean {
  return !useIsAboveBreakpoint(breakpoint)
}

/**
 * Hook that returns responsive values based on current breakpoint
 *
 * @example
 * ```tsx
 * const columns = useResponsiveValue({
 *   xs: 1,
 *   sm: 2,
 *   md: 3,
 *   lg: 4,
 * })
 * ```
 */
export function useResponsiveValue<T>(
  values: Partial<Record<BreakpointOrBelow, T>>,
): T | undefined {
  const breakpoint = useBreakpoint()

  // Find the value for current breakpoint or the largest smaller breakpoint
  const breakpointOrder: BreakpointOrBelow[] = [
    'xs',
    'sm',
    'md',
    'lg',
    'xl',
    '2xl',
  ]
  const currentIndex = breakpointOrder.indexOf(breakpoint)

  // Look for value at current or lower breakpoints
  for (let i = currentIndex; i >= 0; i--) {
    const bp = breakpointOrder[i]
    if (values[bp] !== undefined) {
      return values[bp]
    }
  }

  return undefined
}

/**
 * Hook that provides common responsive helpers
 *
 * @example
 * ```tsx
 * const { isMobile, isTablet, isDesktop } = useResponsive()
 *
 * return isMobile ? <MobileNav /> : <DesktopNav />
 * ```
 */
export function useResponsive() {
  const breakpoint = useBreakpoint()

  return {
    breakpoint,
    isMobile: breakpoint === 'xs' || breakpoint === 'sm',
    isTablet: breakpoint === 'md',
    isDesktop:
      breakpoint === 'lg' || breakpoint === 'xl' || breakpoint === '2xl',
    isSmallScreen:
      breakpoint === 'xs' || breakpoint === 'sm' || breakpoint === 'md',
    isLargeScreen:
      breakpoint === 'lg' || breakpoint === 'xl' || breakpoint === '2xl',
  }
}

/**
 * Hook that returns the window dimensions
 * Use sparingly - prefer breakpoint-based logic when possible
 */
export function useWindowSize() {
  const [size, setSize] = useState(() => {
    if (typeof window === 'undefined') {
      return { width: 1024, height: 768 } // SSR fallback
    }
    return { width: window.innerWidth, height: window.innerHeight }
  })

  useEffect(() => {
    const handleResize = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight })
    }

    handleResize()

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return size
}

/**
 * Media query hook for custom queries
 *
 * @example
 * ```tsx
 * const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
 * const isPortrait = useMediaQuery('(orientation: portrait)')
 * ```
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    const mediaQuery = window.matchMedia(query)
    setMatches(mediaQuery.matches)

    const handler = (event: MediaQueryListEvent) => setMatches(event.matches)
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [query])

  return matches
}
