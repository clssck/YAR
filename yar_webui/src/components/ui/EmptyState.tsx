import { FileUp, FolderOpen, type LucideIcon, MessageSquare, Network, Search } from 'lucide-react'
import Button from '@/components/ui/Button'
import { Card, CardDescription, CardTitle } from '@/components/ui/Card'
import { cn } from '@/lib/utils'

export interface EmptyStateProps extends React.ComponentPropsWithoutRef<typeof Card> {
  /** Main title */
  title: string
  /** Description text */
  description?: string
  /** Icon to display */
  icon?: LucideIcon
  /** Primary action button */
  action?: {
    label: string
    onClick: () => void
    variant?: 'default' | 'outline' | 'secondary'
  }
  /** Secondary action (text link style) */
  secondaryAction?: {
    label: string
    onClick: () => void
  }
  /** Size variant */
  size?: 'sm' | 'default' | 'lg'
}

const sizeStyles = {
  sm: {
    container: 'p-8 space-y-4',
    iconContainer: 'p-4',
    icon: 'size-6',
    title: 'text-base',
    description: 'text-xs',
  },
  default: {
    container: 'p-16 space-y-6',
    iconContainer: 'p-6',
    icon: 'size-10',
    title: 'text-lg',
    description: 'text-sm',
  },
  lg: {
    container: 'p-20 space-y-8',
    iconContainer: 'p-8',
    icon: 'size-14',
    title: 'text-xl',
    description: 'text-base',
  },
}

/**
 * Empty state component for when there's no content to display.
 * Provides clear guidance on next actions.
 */
export default function EmptyState({
  title,
  description,
  icon: Icon = FolderOpen,
  action,
  secondaryAction,
  size = 'default',
  className,
  ...props
}: EmptyStateProps) {
  const styles = sizeStyles[size]

  return (
    <Card
      className={cn(
        'flex w-full h-full flex-col items-center justify-center bg-transparent border-none shadow-none',
        styles.container,
        className
      )}
      {...props}
    >
      <div
        className={cn(
          'shrink-0 rounded-2xl bg-gradient-to-br from-muted/80 to-muted/40 ring-1 ring-border/50',
          styles.iconContainer
        )}
      >
        <Icon className={cn('text-muted-foreground/70', styles.icon)} aria-hidden="true" />
      </div>

      <div className="flex flex-col items-center gap-2 text-center max-w-sm">
        <CardTitle className={cn('font-semibold', styles.title)}>{title}</CardTitle>
        {description && (
          <CardDescription className={cn('text-muted-foreground/80', styles.description)}>
            {description}
          </CardDescription>
        )}
      </div>

      {(action || secondaryAction) && (
        <div className="flex flex-col items-center gap-2 mt-2">
          {action && (
            <Button variant={action.variant ?? 'default'} onClick={action.onClick}>
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <button
              type="button"
              onClick={secondaryAction.onClick}
              className="text-sm text-muted-foreground hover:text-foreground underline-offset-4 hover:underline transition-colors"
            >
              {secondaryAction.label}
            </button>
          )}
        </div>
      )}
    </Card>
  )
}

/**
 * Pre-configured empty states for common use cases
 */

export function EmptyDocuments({
  onUpload,
  className,
}: {
  onUpload: () => void
  className?: string
}) {
  return (
    <EmptyState
      icon={FileUp}
      title="No documents yet"
      description="Upload documents to start building your knowledge graph"
      action={{ label: 'Upload Documents', onClick: onUpload }}
      secondaryAction={{ label: 'Or drag files here', onClick: onUpload }}
      className={className}
    />
  )
}

export function EmptyGraph({
  onLoadData,
  className,
}: {
  onLoadData?: () => void
  className?: string
}) {
  return (
    <EmptyState
      icon={Network}
      title="No graph data"
      description="Process documents to generate the knowledge graph"
      action={onLoadData ? { label: 'Load Graph', onClick: onLoadData } : undefined}
      className={className}
    />
  )
}

export function EmptySearchResults({
  query,
  onClear,
  className,
}: {
  query: string
  onClear: () => void
  className?: string
}) {
  return (
    <EmptyState
      icon={Search}
      title="No results found"
      description={`No matches for "${query}". Try a different search term.`}
      action={{ label: 'Clear Search', onClick: onClear, variant: 'outline' }}
      size="sm"
      className={className}
    />
  )
}

export function EmptyChat({
  onStartQuery,
  className,
}: {
  onStartQuery?: () => void
  className?: string
}) {
  return (
    <EmptyState
      icon={MessageSquare}
      title="Start a conversation"
      description="Ask questions about your knowledge base"
      action={onStartQuery ? { label: 'Ask a Question', onClick: onStartQuery } : undefined}
      className={className}
    />
  )
}
