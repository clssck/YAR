import { cva, type VariantProps } from 'class-variance-authority'
import {
  CheckCircle,
  Clock,
  AlertCircle,
  Loader2,
  Info,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const statusBadgeVariants = cva(
  'inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      status: {
        success: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400',
        warning: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
        error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
        info: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
        pending: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
        processing: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
      },
      size: {
        sm: 'text-[10px] px-2 py-0.5',
        default: 'text-xs px-2.5 py-0.5',
        lg: 'text-sm px-3 py-1',
      },
    },
    defaultVariants: {
      status: 'info',
      size: 'default',
    },
  }
)

const statusIcons: Record<string, LucideIcon> = {
  success: CheckCircle,
  warning: AlertCircle,
  error: XCircle,
  info: Info,
  pending: Clock,
  processing: Loader2,
}

export interface StatusBadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof statusBadgeVariants> {
  /** Text label displayed in the badge */
  label: string
  /** Whether to show the status icon */
  showIcon?: boolean
  /** Custom icon to override the default status icon */
  icon?: LucideIcon
  /** Whether the processing icon should animate (spin) */
  animate?: boolean
}

/**
 * Accessible status badge with text label and optional icon.
 * Provides visual status indication that doesn't rely solely on color.
 */
export default function StatusBadge({
  label,
  status,
  size,
  showIcon = true,
  icon: CustomIcon,
  animate = true,
  className,
  ...props
}: StatusBadgeProps) {
  const Icon = CustomIcon ?? (status ? statusIcons[status] : null)
  const isAnimated = animate && status === 'processing'

  return (
    <span
      className={cn(statusBadgeVariants({ status, size }), className)}
      role="status"
      aria-label={`Status: ${label}`}
      {...props}
    >
      {showIcon && Icon && (
        <Icon
          className={cn('h-3 w-3 shrink-0', isAnimated && 'animate-spin')}
          aria-hidden="true"
        />
      )}
      <span>{label}</span>
    </span>
  )
}

/**
 * Pre-configured status badges for common document states
 */
export const DocumentStatusBadge = {
  Processed: () => <StatusBadge status="success" label="Processed" />,
  Processing: () => <StatusBadge status="processing" label="Processing" />,
  Pending: () => <StatusBadge status="pending" label="Pending" />,
  Failed: () => <StatusBadge status="error" label="Failed" />,
  Preprocessed: () => <StatusBadge status="info" label="Preprocessed" />,
}
