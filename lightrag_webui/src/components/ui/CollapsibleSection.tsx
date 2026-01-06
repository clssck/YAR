import { useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import Badge from './Badge'

export interface CollapsibleSectionProps {
  /** Section header title */
  title: string
  /** Optional badge to show count or status */
  badge?: string | number
  /** Whether section is initially open (uncontrolled mode) */
  defaultOpen?: boolean
  /** Controlled open state */
  open?: boolean
  /** Callback when open state changes */
  onOpenChange?: (open: boolean) => void
  /** Section content */
  children: ReactNode
  /** Additional class for the container */
  className?: string
  /** Additional class for the header */
  headerClassName?: string
  /** Additional class for the content */
  contentClassName?: string
}

export default function CollapsibleSection({
  title,
  badge,
  defaultOpen = true,
  open: controlledOpen,
  onOpenChange,
  children,
  className,
  headerClassName,
  contentClassName,
}: CollapsibleSectionProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)

  // Support both controlled and uncontrolled modes
  const isControlled = controlledOpen !== undefined
  const isOpen = isControlled ? controlledOpen : internalOpen

  const handleToggle = () => {
    const newState = !isOpen
    if (!isControlled) {
      setInternalOpen(newState)
    }
    onOpenChange?.(newState)
  }

  return (
    <div className={cn('border-b border-border/50 last:border-b-0', className)}>
      <button
        type="button"
        onClick={handleToggle}
        className={cn(
          'flex w-full items-center justify-between py-2.5 px-1 text-sm font-medium',
          'text-foreground/90 hover:text-foreground transition-colors',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-sm',
          headerClassName
        )}
        aria-expanded={isOpen}
      >
        <span className="flex items-center gap-2">
          {title}
          {badge !== undefined && (
            <Badge variant="secondary" className="px-1.5 py-0 text-[10px] font-normal">
              {badge}
            </Badge>
          )}
        </span>
        <ChevronDown
          className={cn(
            'h-4 w-4 text-muted-foreground transition-transform duration-200',
            isOpen && 'rotate-180'
          )}
        />
      </button>
      <div
        className={cn(
          'grid transition-[grid-template-rows] duration-200 ease-out',
          isOpen ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
        )}
      >
        <div className="overflow-hidden">
          <div className={cn('pb-3 pt-1', contentClassName)}>{children}</div>
        </div>
      </div>
    </div>
  )
}
