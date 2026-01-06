import { useState, useEffect, useCallback } from 'react'
import { RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import Button from './Button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './Tooltip'

/**
 * Format a timestamp as relative time (e.g., "2 min ago", "just now")
 */
function formatRelativeTime(timestamp: number): string {
  const now = Date.now()
  const diffMs = now - timestamp
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHour = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHour / 24)

  if (diffSec < 5) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  if (diffMin < 60) return `${diffMin} min ago`
  if (diffHour < 24) return `${diffHour}h ago`
  if (diffDay === 1) return 'yesterday'
  return `${diffDay}d ago`
}

/**
 * Format timestamp as full date/time for tooltip
 */
function formatFullTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString()
}

export interface LastUpdatedProps {
  /** Timestamp of last update (ms since epoch) */
  timestamp: number | null
  /** Callback to trigger a refresh */
  onRefresh?: () => void
  /** Whether a refresh is currently in progress */
  isRefreshing?: boolean
  /** Label prefix (default: "Updated") */
  label?: string
  /** Additional class for the container */
  className?: string
  /** Interval to update the relative time display (ms, default: 10000) */
  refreshInterval?: number
  /** Whether to show the refresh button */
  showRefreshButton?: boolean
}

/**
 * Displays a relative timestamp with optional refresh button.
 * Auto-updates the display to keep relative time current.
 */
export default function LastUpdated({
  timestamp,
  onRefresh,
  isRefreshing = false,
  label = 'Updated',
  className,
  refreshInterval = 10000,
  showRefreshButton = true,
}: LastUpdatedProps) {
  const [, setTick] = useState(0)

  // Auto-update the display periodically
  useEffect(() => {
    if (!timestamp) return

    const interval = setInterval(() => {
      setTick((t) => t + 1)
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [timestamp, refreshInterval])

  const handleRefresh = useCallback(() => {
    if (!isRefreshing && onRefresh) {
      onRefresh()
    }
  }, [isRefreshing, onRefresh])

  if (!timestamp) {
    return (
      <div className={cn('flex items-center gap-1.5 text-xs text-muted-foreground', className)}>
        <span>Never updated</span>
        {showRefreshButton && onRefresh && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="h-6 w-6 p-0"
          >
            <RefreshCw className={cn('h-3 w-3', isRefreshing && 'animate-spin')} />
            <span className="sr-only">Refresh</span>
          </Button>
        )}
      </div>
    )
  }

  const relativeTime = formatRelativeTime(timestamp)
  const fullTime = formatFullTime(timestamp)

  return (
    <div className={cn('flex items-center gap-1.5 text-xs text-muted-foreground', className)}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-default">
              {label} {relativeTime}
            </span>
          </TooltipTrigger>
          <TooltipContent>
            <p>{fullTime}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {showRefreshButton && onRefresh && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="h-6 w-6 p-0"
              >
                <RefreshCw className={cn('h-3 w-3', isRefreshing && 'animate-spin')} />
                <span className="sr-only">Refresh now</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Refresh now</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  )
}
