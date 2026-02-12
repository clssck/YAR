import { cva, type VariantProps } from 'class-variance-authority'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import Progress from './Progress'

const loadingContainerVariants = cva('flex items-center', {
  variants: {
    variant: {
      inline: 'gap-2',
      overlay:
        'fixed inset-0 z-50 bg-background/80 backdrop-blur-sm justify-center flex-col gap-3',
      centered: 'justify-center flex-col gap-3 py-8',
      minimal: 'gap-1.5',
    },
    size: {
      sm: '',
      default: '',
      lg: '',
    },
  },
  defaultVariants: {
    variant: 'inline',
    size: 'default',
  },
})

const spinnerSizeMap = {
  sm: 'h-3 w-3',
  default: 'h-4 w-4',
  lg: 'h-6 w-6',
}

const textSizeMap = {
  sm: 'text-xs',
  default: 'text-sm',
  lg: 'text-base',
}

export interface LoadingStateProps
  extends React.OutputHTMLAttributes<HTMLOutputElement>,
    VariantProps<typeof loadingContainerVariants> {
  /** Message to display while loading */
  message?: string
  /** Progress percentage (0-100) - shows progress bar when provided */
  progress?: number
  /** Whether to show the spinner */
  showSpinner?: boolean
  /** Accessible label for screen readers */
  ariaLabel?: string
}

/**
 * Flexible loading indicator with optional message and progress bar.
 * Supports multiple display modes: inline, overlay, centered, minimal.
 */
export default function LoadingState({
  message,
  progress,
  variant = 'inline',
  size = 'default',
  showSpinner = true,
  ariaLabel,
  className,
  ...props
}: LoadingStateProps) {
  const sizeKey = size ?? 'default'
  const hasProgress = progress !== undefined && progress >= 0

  return (
    <output
      className={cn(loadingContainerVariants({ variant, size }), className)}
      aria-label={ariaLabel ?? message ?? 'Loading'}
      aria-busy="true"
      {...props}
    >
      {showSpinner && (
        <Loader2
          className={cn(
            'animate-spin text-muted-foreground',
            spinnerSizeMap[sizeKey],
          )}
          aria-hidden="true"
        />
      )}

      {message && (
        <span className={cn('text-muted-foreground', textSizeMap[sizeKey])}>
          {message}
        </span>
      )}

      {hasProgress && (
        <div className="w-full max-w-xs">
          <Progress value={progress} className="h-1.5" />
          <span
            className={cn(
              'text-muted-foreground mt-1 block text-center',
              textSizeMap[sizeKey],
            )}
          >
            {Math.round(progress)}%
          </span>
        </div>
      )}

      {/* Screen reader only live region */}
      <span className="sr-only" aria-live="polite">
        {hasProgress
          ? `Loading: ${Math.round(progress)}% complete`
          : 'Loading...'}
      </span>
    </output>
  )
}

/**
 * Skeleton placeholder for content that's loading.
 * Use this for layout-preserving loading states.
 */
export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-muted', className)}
      aria-hidden="true"
      {...props}
    />
  )
}

/**
 * Skeleton loader for text content
 */
export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number
  className?: string
}) {
  return (
    <div className={cn('space-y-2', className)} aria-hidden="true">
      {Array.from({ length: lines }).map((_, i, arr) => (
        <Skeleton
          key={`line-${i}-${arr.length}`}
          className={cn('h-4', i === lines - 1 ? 'w-3/4' : 'w-full')}
        />
      ))}
    </div>
  )
}

/**
 * Pulsing dot indicator for subtle loading feedback
 */
export function PulsingDot({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        'inline-block h-2 w-2 rounded-full bg-primary animate-pulse',
        className,
      )}
      aria-hidden="true"
    />
  )
}
