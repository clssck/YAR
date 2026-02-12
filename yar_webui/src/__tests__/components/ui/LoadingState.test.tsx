import { describe, expect, test } from 'bun:test'
import { render } from '@testing-library/react'
import LoadingState, {
  PulsingDot,
  Skeleton,
  SkeletonText,
} from '@/components/ui/LoadingState'

describe('LoadingState', () => {
  describe('Rendering', () => {
    test('renders with default props', () => {
      const { container } = render(<LoadingState />)
      const output = container.querySelector('output')
      expect(output).toBeTruthy()
    })

    test('renders spinner by default', () => {
      const { container } = render(<LoadingState />)
      const spinner = container.querySelector('svg[class*="animate-spin"]')
      expect(spinner).toBeTruthy()
    })

    test('hides spinner when showSpinner is false', () => {
      const { container } = render(<LoadingState showSpinner={false} />)
      const spinner = container.querySelector('svg[class*="animate-spin"]')
      expect(spinner).toBeFalsy()
    })

    test('renders custom message', () => {
      const { container } = render(<LoadingState message="Loading data..." />)
      const message = container.querySelector('span:not(.sr-only)')
      expect(message?.textContent).toBe('Loading data...')
    })

    test('renders without message when not provided', () => {
      const { container } = render(<LoadingState showSpinner={false} />)
      const spans = container.querySelectorAll('span')
      // Only aria-live span should exist
      expect(spans.length).toBe(1)
    })
  })

  describe('Variants', () => {
    test('renders inline variant by default', () => {
      const { container } = render(<LoadingState variant="inline" />)
      const output = container.querySelector('output')
      expect(output?.className).toContain('gap-2')
    })

    test('renders overlay variant', () => {
      const { container } = render(<LoadingState variant="overlay" />)
      const output = container.querySelector('output')
      expect(output?.className).toContain('fixed')
      expect(output?.className).toContain('inset-0')
      expect(output?.className).toContain('z-50')
    })

    test('renders centered variant', () => {
      const { container } = render(<LoadingState variant="centered" />)
      const output = container.querySelector('output')
      expect(output?.className).toContain('justify-center')
      expect(output?.className).toContain('flex-col')
    })

    test('renders minimal variant', () => {
      const { container } = render(<LoadingState variant="minimal" />)
      const output = container.querySelector('output')
      expect(output?.className).toContain('gap-1.5')
    })
  })

  describe('Sizes', () => {
    test('renders small spinner', () => {
      const { container } = render(<LoadingState size="sm" />)
      const spinner = container.querySelector('svg')
      expect(spinner?.className).toContain('h-3')
      expect(spinner?.className).toContain('w-3')
    })

    test('renders default size spinner', () => {
      const { container } = render(<LoadingState size="default" />)
      const spinner = container.querySelector('svg')
      expect(spinner?.className).toContain('h-4')
      expect(spinner?.className).toContain('w-4')
    })

    test('renders large spinner', () => {
      const { container } = render(<LoadingState size="lg" />)
      const spinner = container.querySelector('svg')
      expect(spinner?.className).toContain('h-6')
      expect(spinner?.className).toContain('w-6')
    })

    test('applies text size for small variant', () => {
      const { container } = render(<LoadingState size="sm" message="Loading" />)
      const span = container.querySelector('span:not(.sr-only)')
      expect(span?.className).toContain('text-xs')
    })

    test('applies text size for default variant', () => {
      const { container } = render(
        <LoadingState size="default" message="Loading" />,
      )
      const span = container.querySelector('span:not(.sr-only)')
      expect(span?.className).toContain('text-sm')
    })

    test('applies text size for large variant', () => {
      const { container } = render(<LoadingState size="lg" message="Loading" />)
      const span = container.querySelector('span:not(.sr-only)')
      expect(span?.className).toContain('text-base')
    })
  })

  describe('Progress Bar', () => {
    test('shows progress bar when progress is provided', () => {
      const { container } = render(<LoadingState progress={50} />)
      const progressBar = container.querySelector('[class*="bg-secondary"]')
      expect(progressBar).toBeTruthy()
    })

    test('hides progress bar when progress is undefined', () => {
      const { container } = render(<LoadingState />)
      const progressBar = container.querySelector('[class*="bg-secondary"]')
      expect(progressBar).toBeFalsy()
    })

    test('hides progress bar when progress is negative', () => {
      const { container } = render(<LoadingState progress={-1} />)
      const progressBar = container.querySelector('[class*="bg-secondary"]')
      expect(progressBar).toBeFalsy()
    })

    test('displays progress percentage', () => {
      const { container } = render(<LoadingState progress={75} />)
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('75%'),
      )
      expect(percentage?.textContent).toBe('75%')
    })

    test('rounds progress percentage', () => {
      const { container } = render(<LoadingState progress={33.7} />)
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('34%'),
      )
      expect(percentage?.textContent).toBe('34%')
    })

    test('displays 0% when progress is 0', () => {
      const { container } = render(<LoadingState progress={0} />)
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('0%'),
      )
      expect(percentage?.textContent).toBe('0%')
    })

    test('displays 100% when progress is 100', () => {
      const { container } = render(<LoadingState progress={100} />)
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('100%'),
      )
      expect(percentage?.textContent).toBe('100%')
    })
  })

  describe('Accessibility', () => {
    test('sets aria-busy to true', () => {
      const { container } = render(<LoadingState />)
      const output = container.querySelector('output')
      expect(output?.getAttribute('aria-busy')).toBe('true')
    })

    test('uses provided ariaLabel', () => {
      const { container } = render(
        <LoadingState ariaLabel="Loading documents" />,
      )
      const output = container.querySelector('output')
      expect(output?.getAttribute('aria-label')).toBe('Loading documents')
    })

    test('uses message as aria-label when ariaLabel not provided', () => {
      const { container } = render(<LoadingState message="Please wait" />)
      const output = container.querySelector('output')
      expect(output?.getAttribute('aria-label')).toBe('Please wait')
    })

    test('uses default aria-label when neither ariaLabel nor message provided', () => {
      const { container } = render(<LoadingState />)
      const output = container.querySelector('output')
      expect(output?.getAttribute('aria-label')).toBe('Loading')
    })

    test('hides spinner from screen readers', () => {
      const { container } = render(<LoadingState />)
      const spinner = container.querySelector('svg')
      expect(spinner?.getAttribute('aria-hidden')).toBe('true')
    })

    test('includes live region for status updates', () => {
      const { container } = render(<LoadingState />)
      const liveRegion = container.querySelector('[aria-live="polite"]')
      expect(liveRegion).toBeTruthy()
      expect(liveRegion?.className).toContain('sr-only')
    })

    test('announces loading status in live region', () => {
      const { container } = render(<LoadingState />)
      const liveRegion = container.querySelector('[aria-live="polite"]')
      expect(liveRegion?.textContent).toBe('Loading...')
    })

    test('announces progress in live region', () => {
      const { container } = render(<LoadingState progress={50} />)
      const liveRegion = container.querySelector('[aria-live="polite"]')
      expect(liveRegion?.textContent).toBe('Loading: 50% complete')
    })

    test('output element is semantic HTML', () => {
      const { container } = render(<LoadingState />)
      const output = container.querySelector('output')
      expect(output).toBeTruthy()
      expect(output?.tagName.toLowerCase()).toBe('output')
    })
  })

  describe('Custom Props', () => {
    test('accepts custom className', () => {
      const { container } = render(<LoadingState className="custom-class" />)
      const output = container.querySelector('output')
      expect(output?.className).toContain('custom-class')
    })

    test('merges custom className with variant classes', () => {
      const { container } = render(
        <LoadingState variant="centered" className="custom-class" />,
      )
      const output = container.querySelector('output')
      expect(output?.className).toContain('justify-center')
      expect(output?.className).toContain('custom-class')
    })

    test('accepts additional HTML attributes', () => {
      const { container } = render(
        <LoadingState data-testid="loading" role="status" />,
      )
      const output = container.querySelector('output')
      expect(output?.getAttribute('data-testid')).toBe('loading')
      expect(output?.getAttribute('role')).toBe('status')
    })
  })

  describe('Complex Scenarios', () => {
    test('renders with message and progress', () => {
      const { container } = render(
        <LoadingState message="Uploading files..." progress={60} />,
      )
      const message = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('Uploading'),
      )
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('60%'),
      )
      expect(message?.textContent).toBe('Uploading files...')
      expect(percentage?.textContent).toBe('60%')
    })

    test('renders overlay with message and progress', () => {
      const { container } = render(
        <LoadingState
          variant="overlay"
          message="Processing..."
          progress={30}
          showSpinner={true}
        />,
      )
      const output = container.querySelector('output')
      expect(output?.className).toContain('fixed')

      const message = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('Processing'),
      )
      const percentage = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('30%'),
      )
      expect(message?.textContent).toBe('Processing...')
      expect(percentage?.textContent).toBe('30%')
    })

    test('renders minimal variant without spinner', () => {
      const { container } = render(
        <LoadingState
          variant="minimal"
          message="Syncing..."
          showSpinner={false}
        />,
      )
      const spinner = container.querySelector('svg[class*="animate-spin"]')
      expect(spinner).toBeFalsy()

      const message = Array.from(container.querySelectorAll('span')).find(
        (span) => span.textContent?.includes('Syncing'),
      )
      expect(message?.textContent).toBe('Syncing...')
    })
  })
})

describe('Skeleton', () => {
  describe('Rendering', () => {
    test('renders skeleton element', () => {
      const { container } = render(<Skeleton />)
      const skeleton = container.querySelector('div[class*="animate-pulse"]')
      expect(skeleton).toBeTruthy()
    })

    test('applies default skeleton styles', () => {
      const { container } = render(<Skeleton />)
      const skeleton = container.querySelector('div')
      expect(skeleton?.className).toContain('animate-pulse')
      expect(skeleton?.className).toContain('rounded-md')
      expect(skeleton?.className).toContain('bg-muted')
    })

    test('accepts custom className', () => {
      const { container } = render(<Skeleton className="h-12 w-full" />)
      const skeleton = container.querySelector('div')
      expect(skeleton?.className).toContain('h-12')
      expect(skeleton?.className).toContain('w-full')
    })

    test('merges custom className with default styles', () => {
      const { container } = render(<Skeleton className="custom-class" />)
      const skeleton = container.querySelector('div')
      expect(skeleton?.className).toContain('animate-pulse')
      expect(skeleton?.className).toContain('custom-class')
    })

    test('hides skeleton from screen readers', () => {
      const { container } = render(<Skeleton />)
      const skeleton = container.querySelector('div')
      expect(skeleton?.getAttribute('aria-hidden')).toBe('true')
    })

    test('accepts additional HTML attributes', () => {
      const { container } = render(<Skeleton data-testid="skeleton" />)
      const skeleton = container.querySelector('div')
      expect(skeleton?.getAttribute('data-testid')).toBe('skeleton')
    })
  })
})

describe('SkeletonText', () => {
  describe('Rendering', () => {
    test('renders with default 3 lines', () => {
      const { container } = render(<SkeletonText />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      expect(skeletons.length).toBe(3)
    })

    test('renders custom number of lines', () => {
      const { container } = render(<SkeletonText lines={5} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      expect(skeletons.length).toBe(5)
    })

    test('renders single line', () => {
      const { container } = render(<SkeletonText lines={1} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      expect(skeletons.length).toBe(1)
    })

    test('renders container with proper spacing', () => {
      const { container } = render(<SkeletonText />)
      const wrapper = container.querySelector('div[class*="space-y"]')
      expect(wrapper?.className).toContain('space-y-2')
    })

    test('last line has reduced width', () => {
      const { container } = render(<SkeletonText lines={3} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      const lastSkeleton = skeletons[skeletons.length - 1]
      expect(lastSkeleton?.className).toContain('w-3/4')
    })

    test('all lines except last have full width', () => {
      const { container } = render(<SkeletonText lines={3} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      expect(skeletons[0]?.className).toContain('w-full')
      expect(skeletons[1]?.className).toContain('w-full')
      expect(skeletons[2]?.className).toContain('w-3/4')
    })

    test('all lines have consistent height', () => {
      const { container } = render(<SkeletonText lines={4} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      skeletons.forEach((skeleton) => {
        expect(skeleton?.className).toContain('h-4')
      })
    })

    test('accepts custom className', () => {
      const { container } = render(<SkeletonText className="custom-wrapper" />)
      const wrapper = container.querySelector('div[class*="space-y"]')
      expect(wrapper?.className).toContain('custom-wrapper')
    })

    test('hides container from screen readers', () => {
      const { container } = render(<SkeletonText />)
      const wrapper = container.querySelector('div[class*="space-y"]')
      expect(wrapper?.getAttribute('aria-hidden')).toBe('true')
    })

    test('generates unique keys for each skeleton', () => {
      const { container } = render(<SkeletonText lines={3} />)
      const skeletons = container.querySelectorAll(
        'div[class*="animate-pulse"]',
      )
      expect(skeletons.length).toBe(3)
    })
  })
})

describe('PulsingDot', () => {
  describe('Rendering', () => {
    test('renders pulsing dot element', () => {
      const { container } = render(<PulsingDot />)
      const dot = container.querySelector('span[class*="animate-pulse"]')
      expect(dot).toBeTruthy()
    })

    test('applies correct styles', () => {
      const { container } = render(<PulsingDot />)
      const dot = container.querySelector('span')
      expect(dot?.className).toContain('inline-block')
      expect(dot?.className).toContain('h-2')
      expect(dot?.className).toContain('w-2')
      expect(dot?.className).toContain('rounded-full')
      expect(dot?.className).toContain('bg-primary')
      expect(dot?.className).toContain('animate-pulse')
    })

    test('hides dot from screen readers', () => {
      const { container } = render(<PulsingDot />)
      const dot = container.querySelector('span')
      expect(dot?.getAttribute('aria-hidden')).toBe('true')
    })

    test('accepts custom className', () => {
      const { container } = render(<PulsingDot className="custom-dot" />)
      const dot = container.querySelector('span')
      expect(dot?.className).toContain('custom-dot')
    })

    test('merges custom className with default styles', () => {
      const { container } = render(<PulsingDot className="ml-2" />)
      const dot = container.querySelector('span')
      expect(dot?.className).toContain('animate-pulse')
      expect(dot?.className).toContain('ml-2')
    })

    test('maintains inline display', () => {
      const { container } = render(<PulsingDot />)
      const dot = container.querySelector('span')
      expect(dot?.className).toContain('inline-block')
    })
  })
})

describe('Integration Tests', () => {
  test('LoadingState with all features', () => {
    const { container } = render(
      <LoadingState
        variant="centered"
        size="lg"
        message="Initializing system..."
        progress={45}
        showSpinner={true}
        ariaLabel="System initialization"
      />,
    )

    const output = container.querySelector('output')
    expect(output?.getAttribute('aria-label')).toBe('System initialization')
    expect(output?.getAttribute('aria-busy')).toBe('true')

    const spinner = container.querySelector('svg')
    expect(spinner?.className).toContain('h-6')

    const message = Array.from(container.querySelectorAll('span')).find(
      (span) => span.textContent?.includes('Initializing'),
    )
    expect(message?.textContent).toBe('Initializing system...')

    const percentage = Array.from(container.querySelectorAll('span')).find(
      (span) => span.textContent?.includes('45%'),
    )
    expect(percentage?.textContent).toBe('45%')
  })

  test('SkeletonText with custom lines for different content types', () => {
    const { container: titleContainer } = render(<SkeletonText lines={1} />)
    const { container: paragraphContainer } = render(<SkeletonText lines={5} />)

    const titleSkeletons = titleContainer.querySelectorAll(
      'div[class*="animate-pulse"]',
    )
    const paragraphSkeletons = paragraphContainer.querySelectorAll(
      'div[class*="animate-pulse"]',
    )

    expect(titleSkeletons.length).toBe(1)
    expect(paragraphSkeletons.length).toBe(5)
  })

  test('Multiple loading indicators together', () => {
    const { container } = render(
      <div>
        <LoadingState variant="inline" message="Loading..." size="sm" />
        <PulsingDot />
        <Skeleton className="h-32 w-full mt-4" />
      </div>,
    )

    const output = container.querySelector('output')
    const dot = container.querySelector('span[class*="animate-pulse"]')
    const skeleton = container.querySelector('div[class*="bg-muted"]')

    expect(output).toBeTruthy()
    expect(dot).toBeTruthy()
    expect(skeleton).toBeTruthy()
  })
})
