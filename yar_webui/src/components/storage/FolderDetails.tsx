import { useQuery } from '@tanstack/react-query'
import {
  FileIcon,
  FolderIcon,
  GripVerticalIcon,
  HardDriveIcon,
  Loader2Icon,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { S3ObjectInfo } from '@/api/yar'
import { s3FolderStats } from '@/api/yar'
import { ScrollArea } from '@/components/ui/ScrollArea'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/Sheet'
import { cn } from '@/lib/utils'

interface FolderDetailsProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  folderPath: string | null
  folderName: string
}

// Storage key for persisted width
const FOLDER_WIDTH_KEY = 'yar-folder-viewer-width'
const DEFAULT_WIDTH = 500
const MIN_WIDTH = 350
const MAX_WIDTH = 900

// Format bytes to human readable size
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`
}

// Format ISO date to localized string
function formatDate(isoDate: string | null): string {
  if (!isoDate) return '-'
  try {
    return new Date(isoDate).toLocaleString()
  } catch {
    return isoDate
  }
}

// Extract display name from full key
function getDisplayName(key: string, prefix: string): string {
  const relative = key.startsWith(prefix) ? key.slice(prefix.length) : key
  return relative.endsWith('/') ? relative.slice(0, -1) : relative
}

export default function FolderDetails({
  open,
  onOpenChange,
  folderPath,
  folderName,
}: FolderDetailsProps) {
  const { t } = useTranslation()

  // Resizable width state
  const [width, setWidth] = useState(() => {
    const saved =
      typeof window !== 'undefined'
        ? localStorage.getItem(FOLDER_WIDTH_KEY)
        : null
    const parsed = saved ? parseInt(saved, 10) : NaN
    return Number.isFinite(parsed) && parsed >= MIN_WIDTH && parsed <= MAX_WIDTH
      ? parsed
      : DEFAULT_WIDTH
  })
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const widthRef = useRef(width)

  useEffect(() => {
    widthRef.current = width
  }, [width])

  // Fetch folder statistics
  const {
    data: stats,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['s3', 'folder-stats', folderPath],
    queryFn: () => s3FolderStats(folderPath ?? '', 15),
    enabled: open && !!folderPath,
  })

  // Handle resize start
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      setIsResizing(true)
      resizeRef.current = { startX: e.clientX, startWidth: width }
    },
    [width],
  )

  // Handle resize move
  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeRef.current) return
      const delta = resizeRef.current.startX - e.clientX
      const newWidth = Math.min(
        MAX_WIDTH,
        Math.max(MIN_WIDTH, resizeRef.current.startWidth + delta),
      )
      setWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
      localStorage.setItem(FOLDER_WIDTH_KEY, String(widthRef.current))
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing])

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex flex-col p-0"
        style={{ width: `${width}px`, maxWidth: '90vw' }}
      >
        {/* Resize handle */}
        <div
          aria-hidden="true"
          className={cn(
            'absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/50 transition-colors z-50 group',
            isResizing && 'bg-primary/50',
          )}
          onMouseDown={handleResizeStart}
        >
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVerticalIcon className="h-6 w-6 text-muted-foreground" />
          </div>
        </div>

        <div className="p-6 pb-0 flex-shrink-0">
          <SheetHeader>
            <div className="flex items-center gap-2 pr-8">
              <FolderIcon className="h-5 w-5 text-yellow-500" />
              <SheetTitle className="truncate">{folderName}</SheetTitle>
            </div>
            <SheetDescription>{folderPath}</SheetDescription>
          </SheetHeader>
        </div>

        <div className="flex-1 mt-4 min-h-0 overflow-hidden px-6 pb-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : isError ? (
            <div className="flex flex-col items-center justify-center h-full text-destructive gap-2">
              <p>
                {error instanceof Error
                  ? error.message
                  : 'Failed to load folder stats'}
              </p>
            </div>
          ) : stats ? (
            <ScrollArea className="h-full">
              <div className="space-y-6">
                {/* Stats cards */}
                <div className="grid grid-cols-2 gap-3">
                  <StatCard
                    icon={<HardDriveIcon className="h-4 w-4" />}
                    label={t(
                      'storagePanel.folderDetails.totalSize',
                      'Total Size',
                    )}
                    value={formatBytes(stats.total_size)}
                  />
                  <StatCard
                    icon={<FileIcon className="h-4 w-4" />}
                    label={t('storagePanel.folderDetails.objects', 'Objects')}
                    value={stats.object_count.toLocaleString()}
                  />
                  <StatCard
                    icon={<FolderIcon className="h-4 w-4" />}
                    label={t(
                      'storagePanel.folderDetails.subfolders',
                      'Subfolders',
                    )}
                    value={stats.folder_count.toLocaleString()}
                  />
                  <StatCard
                    icon={<FileIcon className="h-4 w-4" />}
                    label={t(
                      'storagePanel.folderDetails.lastModified',
                      'Last Modified',
                    )}
                    value={formatDate(stats.last_modified)}
                    small
                  />
                </div>

                {/* Preview list */}
                {stats.preview.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2 text-muted-foreground">
                      {t(
                        'storagePanel.folderDetails.preview',
                        'Contents Preview',
                      )}
                    </h4>
                    <div className="border rounded-lg divide-y">
                      {stats.preview.map((obj: S3ObjectInfo) => (
                        <div
                          key={obj.key}
                          className="flex items-center justify-between px-3 py-2 text-sm"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <FileIcon className="h-4 w-4 text-blue-500 flex-shrink-0" />
                            <span className="truncate">
                              {getDisplayName(obj.key, stats.prefix)}
                            </span>
                          </div>
                          <span className="text-muted-foreground flex-shrink-0 ml-2">
                            {formatBytes(obj.size)}
                          </span>
                        </div>
                      ))}
                      {stats.object_count > stats.preview.length && (
                        <div className="px-3 py-2 text-sm text-muted-foreground text-center">
                          {t(
                            'storagePanel.folderDetails.andMore',
                            'and {{count}} more...',
                            {
                              count: stats.object_count - stats.preview.length,
                            },
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {stats.object_count === 0 && (
                  <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                    <FolderIcon className="h-12 w-12 mb-2 opacity-50" />
                    <p>
                      {t(
                        'storagePanel.folderDetails.empty',
                        'This folder is empty',
                      )}
                    </p>
                  </div>
                )}
              </div>
            </ScrollArea>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  )
}

// Small stat card component
function StatCard({
  icon,
  label,
  value,
  small,
}: {
  icon: React.ReactNode
  label: string
  value: string
  small?: boolean
}) {
  return (
    <div className="border rounded-lg p-3">
      <div className="flex items-center gap-1.5 text-muted-foreground mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className={cn('font-medium', small ? 'text-sm' : 'text-lg')}>
        {value}
      </div>
    </div>
  )
}
