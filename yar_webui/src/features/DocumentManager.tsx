import {
  AlertCircle,
  AlertTriangle,
  ArrowDownIcon,
  ArrowUpIcon,
  Brain,
  CheckCircle2,
  CheckSquareIcon,
  ChevronDown,
  ChevronRight,
  FileText,
  Info,
  Loader2,
  RefreshCwIcon,
  RotateCcw,
  Search,
  Shell,
  XIcon
} from 'lucide-react'
import React, { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQueryClient } from '@tanstack/react-query'
import {
  type DocStatus,
  type DocStatusResponse,
  type PaginationInfo,
  type PropertyValue
} from '@/api/yar'
import ClearDocumentsDialog from '@/components/documents/ClearDocumentsDialog'
import DeleteDocumentsDialog from '@/components/documents/DeleteDocumentsDialog'
import PipelineStatusDialog from '@/components/documents/PipelineStatusDialog'
import UploadDocumentsDialog from '@/components/documents/UploadDocumentsDialog'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import Checkbox from '@/components/ui/Checkbox'
import { EmptyDocuments } from '@/components/ui/EmptyState'
import Input from '@/components/ui/Input'
import LastUpdated from '@/components/ui/LastUpdated'
import PaginationControls from '@/components/ui/PaginationControls'
import DocumentStatusBadge from '@/components/ui/DocumentStatusBadge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { useResponsive } from '@/hooks/useBreakpoint'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { getCountValue } from '@/utils/documentUtils'
import { useDocumentListQuery } from '@/features/documents/useDocumentListQuery'
import { usePipelineMutations } from '@/features/documents/usePipelineMutations'

type StatusFilter = DocStatus | 'all'

// Timeline stages for document processing
const TIMELINE_STAGES = ['pending', 'preprocessed', 'processing', 'processed'] as const

// Stable empty fallbacks for the query-derived state. Hoisted so the
// `?? EMPTY_X` expressions return the same reference every render and don't
// invalidate downstream useMemo/useEffect deps.
const EMPTY_DOCS: DocStatusResponse[] = []
const EMPTY_STATUS_COUNTS: Record<string, number> = { all: 0 }

interface StatusTimelineProps {
  currentStatus: DocStatus
}

function StatusTimeline({ currentStatus }: StatusTimelineProps) {
  const { t } = useTranslation()
  const getStageIndex = (status: DocStatus): number => {
    if (status === 'failed') return -1 // Failed can happen at any stage
    return TIMELINE_STAGES.indexOf(status as (typeof TIMELINE_STAGES)[number])
  }

  const currentIndex = getStageIndex(currentStatus)
  const isFailed = currentStatus === 'failed'

  const stageLabels: Record<string, string> = {
    pending: t('documentPanel.documentManager.status.pending', 'Pending'),
    preprocessed: t('documentPanel.documentManager.status.preprocessed', 'Preprocessed'),
    processing: t('documentPanel.documentManager.status.processing', 'Processing'),
    processed: t('documentPanel.documentManager.status.completed', 'Processed')
  }

  return (
    <div className="flex min-w-[140px] flex-col gap-1">
      <div className="text-muted-foreground mb-1 text-xs font-medium">
        {t('documentPanel.documentManager.timeline.title', 'Processing Timeline')}
      </div>
      <div className="flex items-center gap-1">
        {TIMELINE_STAGES.map((stage, index) => {
          const isPast = currentIndex > index
          const isCurrent = currentIndex === index
          const isAfter = currentIndex < index

          return (
            <React.Fragment key={stage}>
              {/* Stage circle */}
              <div
                className={cn(
                  'w-3 h-3 rounded-full border-2 flex-shrink-0 transition-colors',
                  isPast && 'bg-emerald-500 border-emerald-500',
                  isCurrent && !isFailed && 'bg-blue-500 border-blue-500 animate-pulse',
                  isFailed &&
                    stage === TIMELINE_STAGES[Math.max(0, currentIndex)] &&
                    'bg-red-500 border-red-500',
                  isAfter && 'bg-transparent border-muted-foreground/30'
                )}
                title={stageLabels[stage]}
              />
              {/* Connector line (not after last) */}
              {index < TIMELINE_STAGES.length - 1 && (
                <div
                  className={cn(
                    'h-0.5 w-3 flex-shrink-0',
                    isPast ? 'bg-emerald-500' : 'bg-muted-foreground/20'
                  )}
                />
              )}
            </React.Fragment>
          )
        })}
      </div>
      {/* Stage labels */}
      <div className="text-muted-foreground mt-0.5 flex justify-between text-[10px]">
        <span>{stageLabels.pending}</span>
        <span>{stageLabels.processed}</span>
      </div>
      {/* Current status indicator */}
      <div
        className={cn(
          'text-xs font-medium mt-1 px-2 py-0.5 rounded-full text-center',
          currentStatus === 'processed' &&
            'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400',
          currentStatus === 'processing' &&
            'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
          currentStatus === 'preprocessed' &&
            'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
          currentStatus === 'pending' &&
            'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
          currentStatus === 'failed' &&
            'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
        )}
      >
        {currentStatus === 'failed'
          ? t('documentPanel.documentManager.status.failed', 'Failed')
          : stageLabels[currentStatus]}
      </div>
    </div>
  )
}
const getDisplayFileName = (doc: DocStatusResponse, maxLength = 20): string => {
  // Check if file_path exists and is a non-empty string
  if (!doc.file_path || typeof doc.file_path !== 'string' || doc.file_path.trim() === '') {
    return doc.id
  }

  // Try to extract filename from path
  const parts = doc.file_path.split('/')
  const fileName = parts[parts.length - 1]

  // Ensure extracted filename is valid
  if (!fileName || fileName.trim() === '') {
    return doc.id
  }

  // If filename is longer than maxLength, truncate it and add ellipsis
  return fileName.length > maxLength ? `${fileName.slice(0, maxLength)}...` : fileName
}

const formatMetadata = (metadata: Record<string, PropertyValue>): string => {
  const formattedMetadata: Record<string, PropertyValue> = { ...metadata }

  if (
    formattedMetadata.processing_start_time &&
    typeof formattedMetadata.processing_start_time === 'number'
  ) {
    const date = new Date(formattedMetadata.processing_start_time * 1000)
    if (!Number.isNaN(date.getTime())) {
      formattedMetadata.processing_start_time = date.toLocaleString()
    }
  }

  if (
    formattedMetadata.processing_end_time &&
    typeof formattedMetadata.processing_end_time === 'number'
  ) {
    const date = new Date(formattedMetadata.processing_end_time * 1000)
    if (!Number.isNaN(date.getTime())) {
      formattedMetadata.processing_end_time = date.toLocaleString()
    }
  }

  // Format JSON and remove outer braces and indentation
  const jsonStr = JSON.stringify(formattedMetadata, null, 2)
  const lines = jsonStr.split('\n')
  // Remove first line ({) and last line (}), and remove leading indentation (2 spaces)
  return lines
    .slice(1, -1)
    .map((line) => line.replace(/^ {2}/, ''))
    .join('\n')
}

// Type definitions for sort field and direction
type SortField = 'created_at' | 'updated_at' | 'id' | 'file_path'
type SortDirection = 'asc' | 'desc'

export default function DocumentManager() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // Server state from stores. health/pipelineBusy still drive the global UI shell.
  const health = useBackendState.use.health()
  const pipelineBusy = useBackendState.use.pipelineBusy()

  const currentTab = useSettingsStore.use.currentTab()
  const showFileName = useSettingsStore.use.showFileName()
  const setShowFileName = useSettingsStore.use.setShowFileName()
  const documentsPageSize = useSettingsStore.use.documentsPageSize()
  const setDocumentsPageSize = useSettingsStore.use.setDocumentsPageSize()

  // Local UI state. Page/filter/sort drive the queryKey; everything else is presentation.
  const [showPipelineStatus, setShowPipelineStatus] = useState(false)
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const deferredSearchQuery = useDeferredValue(searchQuery)
  const { isMobile, isTablet, isDesktop } = useResponsive()

  const [page, setPage] = useState(1)
  const [sortField, setSortField] = useState<SortField>('updated_at')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [pageByStatus, setPageByStatus] = useState<Record<StatusFilter, number>>({
    all: 1,
    processed: 1,
    preprocessed: 1,
    processing: 1,
    pending: 1,
    failed: 1
  })

  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const isSelectionMode = selectedDocIds.length > 0
  const [expandedErrorIds, setExpandedErrorIds] = useState<Set<string>>(new Set())

  // Mutation boost: when set to a future timestamp, useDocumentListQuery polls every 2s
  // until that time. Set by usePipelineMutations on success.
  const [boostUntil, setBoostUntil] = useState<number | null>(null)

  const documentsQuery = useDocumentListQuery({
    page,
    pageSize: documentsPageSize,
    statusFilter,
    sortField,
    sortDirection,
    boostUntil,
    enabled: currentTab === 'documents' && health
  })

  const { scan: scanMutation, retry: retryMutation } = usePipelineMutations({ setBoostUntil })

  // Derive everything previously held in local state from the query.
  const currentPageDocs = documentsQuery.data?.documents ?? EMPTY_DOCS
  const pagination: PaginationInfo = documentsQuery.data?.pagination ?? {
    page,
    page_size: documentsPageSize,
    total_count: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  }
  const statusCounts: Record<string, number> =
    documentsQuery.data?.status_counts ?? EMPTY_STATUS_COUNTS
  const lastFetchTime = documentsQuery.dataUpdatedAt > 0 ? documentsQuery.dataUpdatedAt : null
  const isRefreshing = documentsQuery.isFetching
  const isRetrying = retryMutation.isPending

  // Handle checkbox change for individual documents
  const handleDocumentSelect = useCallback((docId: string, checked: boolean) => {
    setSelectedDocIds((prev) => {
      if (checked) {
        return [...prev, docId]
      } else {
        return prev.filter((id) => id !== docId)
      }
    })
  }, [])

  // Toggle expanded error details for a document
  const toggleErrorExpanded = useCallback((docId: string) => {
    setExpandedErrorIds((prev) => {
      const next = new Set(prev)
      if (next.has(docId)) {
        next.delete(docId)
      } else {
        next.add(docId)
      }
      return next
    })
  }, [])

  // Handle deselect all documents
  const handleDeselectAll = useCallback(() => {
    setSelectedDocIds([])
  }, [])

  // Handle sort column click
  const handleSort = (field: SortField) => {
    let actualField = field

    // When clicking the first column, determine the actual sort field based on showFileName
    if (field === 'id') {
      actualField = showFileName ? 'file_path' : 'id'
    }

    const newDirection = sortField === actualField && sortDirection === 'desc' ? 'asc' : 'desc'

    setSortField(actualField)
    setSortDirection(newDirection)

    // Reset page to 1 when sorting changes
    setPage(1)

    // Reset all status filters' page memory since sorting affects all
    setPageByStatus({
      all: 1,
      processed: 1,
      preprocessed: 1,
      processing: 1,
      pending: 1,
      failed: 1
    })
  }

  // Filter documents based on search query
  const filteredDocs = useMemo(() => {
    if (!deferredSearchQuery.trim()) {
      return currentPageDocs
    }
    const query = deferredSearchQuery.toLowerCase()
    return currentPageDocs.filter(
      (doc) =>
        doc.id.toLowerCase().includes(query) ||
        doc.file_path.toLowerCase().includes(query) ||
        doc.content_summary?.toLowerCase().includes(query)
    )
  }, [currentPageDocs, deferredSearchQuery])

  // Define a new type that includes status information
  type DocStatusWithStatus = DocStatusResponse & { status: DocStatus }

  const filteredAndSortedDocs = useMemo<DocStatusWithStatus[] | null>(() => {
    if (currentPageDocs.length === 0) return null
    return filteredDocs.map((doc) => ({
      ...doc,
      status: doc.status as DocStatus
    }))
  }, [filteredDocs, currentPageDocs.length])

  // Calculate current page selection state (after filteredAndSortedDocs is defined)
  const currentPageDocIds = useMemo(() => {
    return filteredAndSortedDocs?.map((doc) => doc.id) || []
  }, [filteredAndSortedDocs])

  const selectedCurrentPageCount = useMemo(() => {
    return currentPageDocIds.filter((id) => selectedDocIds.includes(id)).length
  }, [currentPageDocIds, selectedDocIds])

  const isCurrentPageFullySelected = useMemo(() => {
    return currentPageDocIds.length > 0 && selectedCurrentPageCount === currentPageDocIds.length
  }, [currentPageDocIds, selectedCurrentPageCount])

  const hasCurrentPageSelection = useMemo(() => {
    return selectedCurrentPageCount > 0
  }, [selectedCurrentPageCount])

  // Handle select current page
  const handleSelectCurrentPage = useCallback(() => {
    setSelectedDocIds(currentPageDocIds)
  }, [currentPageDocIds])

  // Get selection button properties
  const getSelectionButtonProps = useCallback(() => {
    if (!hasCurrentPageSelection) {
      return {
        text: t('documentPanel.selectDocuments.selectCurrentPage', {
          count: currentPageDocIds.length
        }),
        action: handleSelectCurrentPage,
        icon: CheckSquareIcon
      }
    } else if (isCurrentPageFullySelected) {
      return {
        text: t('documentPanel.selectDocuments.deselectAll', {
          count: currentPageDocIds.length
        }),
        action: handleDeselectAll,
        icon: XIcon
      }
    } else {
      return {
        text: t('documentPanel.selectDocuments.selectCurrentPage', {
          count: currentPageDocIds.length
        }),
        action: handleSelectCurrentPage,
        icon: CheckSquareIcon
      }
    }
  }, [
    hasCurrentPageSelection,
    isCurrentPageFullySelected,
    currentPageDocIds.length,
    handleSelectCurrentPage,
    handleDeselectAll,
    t
  ])

  const processedCount = getCountValue(statusCounts, 'PROCESSED', 'processed') || 0
  const preprocessedCount = getCountValue(statusCounts, 'PREPROCESSED', 'preprocessed') || 0
  const processingCount = getCountValue(statusCounts, 'PROCESSING', 'processing') || 0
  const pendingCount = getCountValue(statusCounts, 'PENDING', 'pending') || 0
  const failedCount = getCountValue(statusCounts, 'FAILED', 'failed') || 0

  // Stats items configuration for modern dashboard
  const statItems = useMemo(
    () => [
      {
        id: 'all' as StatusFilter,
        label: t('documentPanel.documentManager.status.all'),
        icon: FileText,
        count: statusCounts.all || 0,
        color: 'text-muted-foreground',
        activeColor: 'text-foreground'
      },
      {
        id: 'processed' as StatusFilter,
        label: t('documentPanel.documentManager.status.completed'),
        icon: CheckCircle2,
        count: processedCount,
        color: 'text-emerald-500',
        activeColor: 'text-emerald-600 dark:text-emerald-500'
      },
      {
        id: 'preprocessed' as StatusFilter,
        label: t('documentPanel.documentManager.status.preprocessed'),
        icon: Brain,
        count: preprocessedCount,
        color: 'text-purple-500',
        activeColor: 'text-purple-600 dark:text-purple-500'
      },
      {
        id: 'processing' as StatusFilter,
        label: t('documentPanel.documentManager.status.processing'),
        icon: Loader2,
        count: processingCount,
        color: 'text-blue-500',
        activeColor: 'text-blue-600 dark:text-blue-500',
        spin: true
      },
      {
        id: 'pending' as StatusFilter,
        label: t('documentPanel.documentManager.status.pending'),
        icon: Loader2,
        count: pendingCount,
        color: 'text-amber-500',
        activeColor: 'text-amber-600 dark:text-amber-500',
        spin: true
      },
      {
        id: 'failed' as StatusFilter,
        label: t('documentPanel.documentManager.status.failed'),
        icon: AlertCircle,
        count: failedCount,
        color: 'text-red-500',
        activeColor: 'text-red-600 dark:text-red-500'
      }
    ],
    [t, statusCounts, processedCount, preprocessedCount, processingCount, pendingCount, failedCount]
  )

  // Store previous status counts
  const prevStatusCounts = useRef({
    processed: 0,
    preprocessed: 0,
    processing: 0,
    pending: 0,
    failed: 0
  })

  // Reference to the card content element
  const cardContentRef = useRef<HTMLDivElement>(null)

  // Reset polling cadence whenever the pipeline busy flag flips: we want the next
  // refetch to happen immediately rather than waiting out the current interval.
  useEffect(() => {
    if (currentTab === 'documents' && health) {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    }
  }, [pipelineBusy, currentTab, health, queryClient])

  // Trigger a backend health check whenever the status counts shift, so the
  // global pipeline indicator stays in sync with the actively-changing documents.
  useEffect(() => {
    const newStatusCounts = {
      processed: getCountValue(statusCounts, 'PROCESSED', 'processed') || 0,
      preprocessed: getCountValue(statusCounts, 'PREPROCESSED', 'preprocessed') || 0,
      processing: getCountValue(statusCounts, 'PROCESSING', 'processing') || 0,
      pending: getCountValue(statusCounts, 'PENDING', 'pending') || 0,
      failed: getCountValue(statusCounts, 'FAILED', 'failed') || 0
    }

    const hasStatusCountChange = (
      Object.keys(newStatusCounts) as Array<keyof typeof newStatusCounts>
    ).some((status) => newStatusCounts[status] !== prevStatusCounts.current[status])

    if (hasStatusCountChange) {
      useBackendState.getState().check()
    }

    prevStatusCounts.current = newStatusCounts
  }, [statusCounts])

  // Handle page change - only update state
  const handlePageChange = useCallback(
    (newPage: number) => {
      if (newPage === page) return

      setPageByStatus((prev) => ({ ...prev, [statusFilter]: newPage }))
      setPage(newPage)
    },
    [page, statusFilter]
  )

  // Handle page size change - update store + reset all filters back to page 1.
  const handlePageSizeChange = useCallback(
    (newPageSize: number) => {
      if (newPageSize === documentsPageSize) return
      setDocumentsPageSize(newPageSize)
      setPageByStatus({
        all: 1,
        processed: 1,
        preprocessed: 1,
        processing: 1,
        pending: 1,
        failed: 1
      })
      setPage(1)
    },
    [documentsPageSize, setDocumentsPageSize]
  )

  // Handle status filter change - update local page intent.
  const handleStatusFilterChange = useCallback(
    (newStatusFilter: StatusFilter) => {
      if (newStatusFilter === statusFilter) return
      setPageByStatus((prev) => ({ ...prev, [statusFilter]: page }))
      setStatusFilter(newStatusFilter)
      setPage(pageByStatus[newStatusFilter])
    },
    [statusFilter, page, pageByStatus]
  )

  // Handle documents deleted callback. Drop selection and let the query refresh.
  const handleDocumentsDeleted = useCallback(async () => {
    setSelectedDocIds([])
    useBackendState.getState().resetHealthCheckTimerDelayed(1000)
    queryClient.invalidateQueries({ queryKey: ['documents'] })
  }, [queryClient])

  // Handle documents cleared callback. Reset to page 1 and force a refetch.
  const handleDocumentsCleared = useCallback(async () => {
    setSelectedDocIds([])
    setPage(1)
    setPageByStatus({
      all: 1,
      processed: 1,
      preprocessed: 1,
      processing: 1,
      pending: 1,
      failed: 1
    })
    queryClient.invalidateQueries({ queryKey: ['documents'] })
  }, [queryClient])

  // Handle showFileName change - switch sort field if currently sorting by first column
  useEffect(() => {
    // Only switch if currently sorting by the first column (id or file_path)
    if (sortField === 'id' || sortField === 'file_path') {
      const newSortField = showFileName ? 'file_path' : 'id'
      if (sortField !== newSortField) {
        setSortField(newSortField)
      }
    }
  }, [showFileName, sortField])

  // Reset selection state when page, status filter, or sort changes.
  // The actual data fetch lives in useDocumentListQuery which keys off the same params.
  useEffect(() => {
    setSelectedDocIds([])
  }, [page, statusFilter, sortField, sortDirection])

  return (
    <TooltipProvider delayDuration={300}>
      <Card className="flex h-full min-h-0 flex-col !overflow-hidden !rounded-none">
        <CardHeader className="border-border/50 border-b px-6 py-4">
          <CardTitle className="text-xl font-semibold tracking-tight">
            {t('documentPanel.documentManager.title')}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col overflow-auto">
          {/* Search Bar */}
          <div className="relative mb-4">
            <Search className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
            <Input
              type="text"
              placeholder={t(
                'documentPanel.documentManager.searchPlaceholder',
                'Search documents...'
              )}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pr-9 pl-9"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="text-muted-foreground hover:text-foreground absolute top-1/2 right-3 h-4 w-4 -translate-y-1/2"
              >
                <XIcon className="h-4 w-4" />
              </button>
            )}
          </div>
          {/* Search is client-side and only filters the current page.
            Warn when there are additional pages so users don't assume
            a no-match means the corpus has nothing. */}
          {searchQuery && pagination.total_pages > 1 && (
            <div
              role="note"
              className="-mt-2 mb-4 flex items-start gap-2 rounded-md border border-amber-300/50 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800/50 dark:bg-amber-950/40 dark:text-amber-200"
            >
              <span aria-hidden="true">⚠</span>
              <span>
                {t(
                  'documentPanel.documentManager.searchScopedToPage',
                  'Search only matches documents on the current page ({{page}} of {{pages}}). Increase the page size or paginate to see matches across the full list.',
                  {
                    page: pagination.page,
                    pages: pagination.total_pages
                  }
                )}
              </span>
            </div>
          )}

          {/* Status Filter Chips */}
          <div className="mb-4 flex flex-wrap gap-2">
            {statItems.map((item) => {
              const isActive = statusFilter === item.id
              const Icon = item.icon
              const showIcon = item.count > 0 || item.id === 'all'

              return (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => handleStatusFilterChange(item.id)}
                  aria-pressed={isActive}
                  className={cn(
                    'inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all cursor-pointer border select-none',
                    isActive
                      ? 'bg-primary text-primary-foreground border-primary ring-2 ring-primary/20'
                      : 'bg-muted/50 text-muted-foreground border-transparent hover:bg-muted hover:text-foreground'
                  )}
                >
                  {showIcon && (
                    <Icon
                      className={cn(
                        'h-3.5 w-3.5',
                        isActive ? 'text-primary-foreground' : item.color,
                        item.spin && item.count > 0 && 'animate-spin'
                      )}
                    />
                  )}
                  <span>{item.label}</span>
                  <span
                    className={cn(
                      'px-1.5 py-0.5 text-xs rounded-full',
                      isActive
                        ? 'bg-primary-foreground/20 text-primary-foreground'
                        : 'bg-muted-foreground/20 text-muted-foreground'
                    )}
                  >
                    {item.count}
                  </span>
                  {isActive && item.id !== 'all' && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleStatusFilterChange('all')
                      }}
                      className="hover:bg-primary-foreground/20 ml-0.5 rounded-full p-0.5"
                    >
                      <XIcon className="h-3 w-3" />
                    </button>
                  )}
                </button>
              )
            })}
          </div>

          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => scanMutation.mutate()}
                side="bottom"
                tooltip={t('documentPanel.documentManager.scanTooltip')}
                size="sm"
              >
                <RefreshCwIcon /> {t('documentPanel.documentManager.scanButton')}
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowPipelineStatus(true)}
                side="bottom"
                tooltip={t('documentPanel.documentManager.pipelineStatusTooltip')}
                size="sm"
                className={cn(pipelineBusy && 'pipeline-busy')}
              >
                <Shell className="h-4 w-4" />{' '}
                {t('documentPanel.documentManager.pipelineStatusButton')}
              </Button>
              {failedCount > 0 && (
                <Button
                  variant="outline"
                  onClick={() => retryMutation.mutate()}
                  side="bottom"
                  tooltip={t(
                    'documentPanel.documentManager.retryFailedTooltip',
                    'Retry all failed documents'
                  )}
                  size="sm"
                  disabled={pipelineBusy || isRetrying || processingCount > 0}
                  className={cn(
                    'text-destructive border-destructive/50 hover:bg-destructive/10',
                    isRetrying && 'animate-pulse'
                  )}
                >
                  <RotateCcw className={cn('h-4 w-4', isRetrying && 'animate-spin')} />
                  {isRetrying
                    ? t('documentPanel.documentManager.retryingButton', 'Retrying...')
                    : t('documentPanel.documentManager.retryFailedButton', {
                        count: failedCount,
                        defaultValue: `Retry ${failedCount} Failed`
                      })}
                </Button>
              )}
            </div>

            {/* Pagination Controls in the middle */}
            {pagination.total_pages > 1 && (
              <PaginationControls
                currentPage={pagination.page}
                totalPages={pagination.total_pages}
                pageSize={pagination.page_size}
                totalCount={pagination.total_count}
                onPageChange={handlePageChange}
                onPageSizeChange={handlePageSizeChange}
                isLoading={isRefreshing}
                compact={true}
              />
            )}

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowFileName(!showFileName)}
                className={cn(
                  'transition-all border',
                  showFileName
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90 border-primary shadow-sm'
                    : 'text-muted-foreground border-dashed border-border/60 hover:border-solid hover:text-foreground hover:bg-accent/50'
                )}
                side="bottom"
                tooltip={
                  showFileName
                    ? t('documentPanel.documentManager.hideButton')
                    : t('documentPanel.documentManager.showButton')
                }
              >
                {showFileName ? (
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                ) : (
                  <FileText className="mr-2 h-4 w-4" />
                )}
                {t('documentPanel.documentManager.columns.fileName')}
              </Button>
              {isSelectionMode && (
                <DeleteDocumentsDialog
                  selectedDocIds={selectedDocIds}
                  onDocumentsDeleted={handleDocumentsDeleted}
                />
              )}
              {isSelectionMode && hasCurrentPageSelection ? (
                (() => {
                  const buttonProps = getSelectionButtonProps()
                  const IconComponent = buttonProps.icon
                  return (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={buttonProps.action}
                      side="bottom"
                      tooltip={buttonProps.text}
                    >
                      <IconComponent className="h-4 w-4" />
                      {buttonProps.text}
                    </Button>
                  )
                })()
              ) : !isSelectionMode ? (
                <ClearDocumentsDialog onDocumentsCleared={handleDocumentsCleared} />
              ) : null}
              <UploadDocumentsDialog
                onDocumentsUploaded={() =>
                  queryClient.invalidateQueries({ queryKey: ['documents'] })
                }
                open={uploadDialogOpen}
                onOpenChange={setUploadDialogOpen}
              />
              <PipelineStatusDialog
                open={showPipelineStatus}
                onOpenChange={setShowPipelineStatus}
              />
            </div>
          </div>

          <Card className="mb-2 flex min-h-0 flex-1 flex-col rounded-md border">
            <CardHeader className="flex-none px-4 py-2">
              <div className="flex items-center justify-between">
                <CardTitle>{t('documentPanel.documentManager.uploadedTitle')}</CardTitle>
                <LastUpdated
                  timestamp={lastFetchTime}
                  onRefresh={() => documentsQuery.refetch()}
                  isRefreshing={isRefreshing}
                />
              </div>
              <CardDescription aria-hidden="true" className="hidden">
                {t('documentPanel.documentManager.uploadedDescription')}
              </CardDescription>
            </CardHeader>

            <CardContent className="relative flex-1 p-0" ref={cardContentRef}>
              {pagination.total_count === 0 && (
                <div className="absolute inset-0 p-0">
                  <EmptyDocuments onUpload={() => setUploadDialogOpen(true)} />
                </div>
              )}
              {pagination.total_count > 0 && (
                <div className="absolute inset-0 flex flex-col p-0">
                  <div className="absolute inset-[-1px] flex flex-col overflow-hidden rounded-md border border-gray-200 p-0 dark:border-gray-700">
                    {/* Mobile: Card Layout */}
                    {isMobile && (
                      <div className="space-y-2 overflow-auto p-2">
                        {filteredAndSortedDocs?.map((doc) => (
                          <div
                            key={doc.id}
                            className={cn(
                              'p-3 rounded-lg border bg-card transition-colors',
                              selectedDocIds.includes(doc.id) &&
                                'ring-2 ring-primary/50 bg-primary/5'
                            )}
                          >
                            {/* Header: Name + Checkbox */}
                            <div className="mb-2 flex items-start justify-between gap-2">
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-medium">
                                  {showFileName ? getDisplayFileName(doc, 25) : doc.id}
                                </div>
                                {showFileName && (
                                  <div className="text-muted-foreground truncate font-mono text-xs">
                                    {doc.id}
                                  </div>
                                )}
                              </div>
                              <Checkbox
                                checked={selectedDocIds.includes(doc.id)}
                                onCheckedChange={(checked) =>
                                  handleDocumentSelect(doc.id, checked === true)
                                }
                              />
                            </div>

                            {/* Status + Summary */}
                            <div className="mb-2 flex items-center gap-2">
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help">
                                    {doc.status === 'processed' && (
                                      <DocumentStatusBadge.Processed />
                                    )}
                                    {doc.status === 'preprocessed' && (
                                      <DocumentStatusBadge.Preprocessed />
                                    )}
                                    {doc.status === 'processing' && (
                                      <DocumentStatusBadge.Processing />
                                    )}
                                    {doc.status === 'pending' && <DocumentStatusBadge.Pending />}
                                    {doc.status === 'failed' && <DocumentStatusBadge.Failed />}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent side="top" align="start" className="p-2">
                                  <StatusTimeline currentStatus={doc.status} />
                                </TooltipContent>
                              </Tooltip>
                              {doc.error_msg && (
                                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                              )}
                            </div>

                            {/* Summary */}
                            {doc.content_summary && (
                              <p className="text-muted-foreground mb-2 line-clamp-2 text-xs">
                                {doc.content_summary}
                              </p>
                            )}

                            {/* Stats Row */}
                            <div className="text-muted-foreground flex flex-wrap gap-x-4 gap-y-1 text-xs">
                              {doc.content_length != null && (
                                <span>
                                  {t('documentPanel.documentManager.tokensUnit', {
                                    count: doc.content_length
                                  })}
                                </span>
                              )}
                              {doc.chunks_count != null && (
                                <span>
                                  {t('documentPanel.documentManager.chunksUnit', {
                                    count: doc.chunks_count
                                  })}
                                </span>
                              )}
                              <span>{new Date(doc.updated_at).toLocaleDateString()}</span>
                            </div>

                            {/* Error Message (if failed) */}
                            {doc.error_msg && (
                              <div className="bg-destructive/10 text-destructive mt-2 rounded p-2 text-xs">
                                <pre className="break-words whitespace-pre-wrap">
                                  {doc.error_msg}
                                </pre>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Tablet/Desktop: Table Layout */}
                    {!isMobile && (
                      <Table className="w-full">
                        <TableHeader className="bg-background sticky top-0 z-10 shadow-sm">
                          <TableRow className="bg-card/95 supports-[backdrop-filter]:bg-card/75 border-b shadow-[inset_0_-1px_0_rgba(0,0,0,0.1)] backdrop-blur">
                            <TableHead
                              onClick={() => handleSort('id')}
                              className="cursor-pointer select-none hover:bg-gray-200 dark:hover:bg-gray-800"
                            >
                              <div className="flex items-center">
                                {showFileName
                                  ? t('documentPanel.documentManager.columns.fileName')
                                  : t('documentPanel.documentManager.columns.id')}
                                {((sortField === 'id' && !showFileName) ||
                                  (sortField === 'file_path' && showFileName)) && (
                                  <span className="ml-1">
                                    {sortDirection === 'asc' ? (
                                      <ArrowUpIcon size={14} />
                                    ) : (
                                      <ArrowDownIcon size={14} />
                                    )}
                                  </span>
                                )}
                              </div>
                            </TableHead>
                            <TableHead>
                              {t('documentPanel.documentManager.columns.summary')}
                            </TableHead>
                            <TableHead>
                              {t('documentPanel.documentManager.columns.status')}
                            </TableHead>
                            <TableHead>
                              {t('documentPanel.documentManager.columns.length')}
                            </TableHead>
                            <TableHead>
                              {t('documentPanel.documentManager.columns.chunks')}
                            </TableHead>
                            {/* S3 Key: Hidden on tablet, shown on desktop */}
                            {isDesktop && (
                              <TableHead>
                                {t('documentPanel.documentManager.columns.s3Key')}
                              </TableHead>
                            )}
                            <TableHead
                              onClick={() => handleSort('created_at')}
                              className="cursor-pointer select-none hover:bg-gray-200 dark:hover:bg-gray-800"
                            >
                              <div className="flex items-center">
                                {t('documentPanel.documentManager.columns.created')}
                                {sortField === 'created_at' && (
                                  <span className="ml-1">
                                    {sortDirection === 'asc' ? (
                                      <ArrowUpIcon size={14} />
                                    ) : (
                                      <ArrowDownIcon size={14} />
                                    )}
                                  </span>
                                )}
                              </div>
                            </TableHead>
                            <TableHead
                              onClick={() => handleSort('updated_at')}
                              className="cursor-pointer select-none hover:bg-gray-200 dark:hover:bg-gray-800"
                            >
                              <div className="flex items-center">
                                {t('documentPanel.documentManager.columns.updated')}
                                {sortField === 'updated_at' && (
                                  <span className="ml-1">
                                    {sortDirection === 'asc' ? (
                                      <ArrowUpIcon size={14} />
                                    ) : (
                                      <ArrowDownIcon size={14} />
                                    )}
                                  </span>
                                )}
                              </div>
                            </TableHead>
                            <TableHead className="w-16 text-center">
                              {t('documentPanel.documentManager.columns.select')}
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody className="overflow-auto text-sm">
                          {filteredAndSortedDocs?.map((doc) => {
                            const hasExpandableDetails =
                              doc.error_msg ||
                              (doc.metadata && Object.keys(doc.metadata).length > 0) ||
                              doc.track_id
                            const isExpanded = expandedErrorIds.has(doc.id)
                            const columnCount = isDesktop ? 9 : 8

                            return (
                              <React.Fragment key={doc.id}>
                                <TableRow
                                  className={cn(hasExpandableDetails && isExpanded && 'border-b-0')}
                                >
                                  <TableCell
                                    className={cn(
                                      'truncate font-mono overflow-visible',
                                      isTablet ? 'max-w-[180px]' : 'max-w-[250px]'
                                    )}
                                  >
                                    {showFileName ? (
                                      <>
                                        <Tooltip>
                                          <TooltipTrigger asChild>
                                            <div className="cursor-help truncate">
                                              {getDisplayFileName(doc, isTablet ? 20 : 30)}
                                            </div>
                                          </TooltipTrigger>
                                          <TooltipContent
                                            side="top"
                                            className="max-w-[600px] break-words whitespace-pre-wrap"
                                          >
                                            {doc.file_path}
                                          </TooltipContent>
                                        </Tooltip>
                                        <div className="text-muted-foreground text-xs">
                                          {doc.id}
                                        </div>
                                      </>
                                    ) : (
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <div className="cursor-help truncate">{doc.id}</div>
                                        </TooltipTrigger>
                                        <TooltipContent
                                          side="top"
                                          className="max-w-[600px] break-words whitespace-pre-wrap"
                                        >
                                          {doc.file_path}
                                        </TooltipContent>
                                      </Tooltip>
                                    )}
                                  </TableCell>
                                  <TableCell
                                    className={cn(
                                      'truncate overflow-visible',
                                      isTablet ? 'max-w-[120px]' : 'max-w-xs min-w-45'
                                    )}
                                  >
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <div className="cursor-help truncate">
                                          {doc.content_summary}
                                        </div>
                                      </TooltipTrigger>
                                      <TooltipContent
                                        side="top"
                                        className="max-w-[600px] break-words whitespace-pre-wrap"
                                      >
                                        {doc.content_summary}
                                      </TooltipContent>
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell>
                                    <div className="flex items-center gap-1">
                                      {/* Status badge with timeline tooltip */}
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <span className="cursor-help">
                                            {doc.status === 'processed' && (
                                              <DocumentStatusBadge.Processed />
                                            )}
                                            {doc.status === 'preprocessed' && (
                                              <DocumentStatusBadge.Preprocessed />
                                            )}
                                            {doc.status === 'processing' && (
                                              <DocumentStatusBadge.Processing />
                                            )}
                                            {doc.status === 'pending' && (
                                              <DocumentStatusBadge.Pending />
                                            )}
                                            {doc.status === 'failed' && (
                                              <DocumentStatusBadge.Failed />
                                            )}
                                          </span>
                                        </TooltipTrigger>
                                        <TooltipContent side="top" align="start" className="p-2">
                                          <StatusTimeline currentStatus={doc.status} />
                                        </TooltipContent>
                                      </Tooltip>

                                      {/* Clickable expand button for documents with details */}
                                      {hasExpandableDetails && (
                                        <button
                                          type="button"
                                          onClick={() => toggleErrorExpanded(doc.id)}
                                          className={cn(
                                            'ml-1 p-1 rounded-md transition-colors hover:bg-muted',
                                            doc.error_msg
                                              ? 'text-yellow-500 hover:text-yellow-600'
                                              : 'text-blue-500 hover:text-blue-600'
                                          )}
                                          title={
                                            isExpanded
                                              ? t(
                                                  'documentPanel.documentManager.collapseDetails',
                                                  'Collapse details'
                                                )
                                              : t(
                                                  'documentPanel.documentManager.expandDetails',
                                                  'View details'
                                                )
                                          }
                                        >
                                          {doc.error_msg ? (
                                            <AlertTriangle className="h-4 w-4" />
                                          ) : (
                                            <Info className="h-4 w-4" />
                                          )}
                                          {isExpanded ? (
                                            <ChevronDown className="ml-0.5 inline-block h-3 w-3" />
                                          ) : (
                                            <ChevronRight className="ml-0.5 inline-block h-3 w-3" />
                                          )}
                                        </button>
                                      )}
                                    </div>
                                  </TableCell>
                                  <TableCell>{doc.content_length ?? '-'}</TableCell>
                                  <TableCell>{doc.chunks_count ?? '-'}</TableCell>
                                  {/* S3 Key: Hidden on tablet, shown on desktop */}
                                  {isDesktop && (
                                    <TableCell className="max-w-[150px] truncate overflow-visible">
                                      {doc.s3_key ? (
                                        <Tooltip>
                                          <TooltipTrigger asChild>
                                            <div className="text-muted-foreground cursor-help truncate font-mono text-xs">
                                              {doc.s3_key.split('/').slice(-2).join('/')}
                                            </div>
                                          </TooltipTrigger>
                                          <TooltipContent
                                            side="top"
                                            className="max-w-[600px] break-words whitespace-pre-wrap"
                                          >
                                            {doc.s3_key}
                                          </TooltipContent>
                                        </Tooltip>
                                      ) : (
                                        <span className="text-muted-foreground">—</span>
                                      )}
                                    </TableCell>
                                  )}
                                  <TableCell className="truncate">
                                    {new Date(doc.created_at).toLocaleString()}
                                  </TableCell>
                                  <TableCell className="truncate">
                                    {new Date(doc.updated_at).toLocaleString()}
                                  </TableCell>
                                  <TableCell className="text-center">
                                    <Checkbox
                                      checked={selectedDocIds.includes(doc.id)}
                                      onCheckedChange={(checked) =>
                                        handleDocumentSelect(doc.id, checked === true)
                                      }
                                      className="mx-auto"
                                    />
                                  </TableCell>
                                </TableRow>

                                {/* Expanded details row */}
                                {hasExpandableDetails && isExpanded && (
                                  <TableRow className="bg-muted/30 hover:bg-muted/40">
                                    <TableCell colSpan={columnCount} className="px-4 py-3">
                                      <div className="space-y-2 text-sm">
                                        {doc.track_id && (
                                          <div className="flex items-center gap-2">
                                            <span className="text-muted-foreground font-medium">
                                              {t('documentPanel.documentManager.trackIdLabel')}
                                            </span>
                                            <code className="bg-muted rounded px-2 py-0.5 font-mono text-xs">
                                              {doc.track_id}
                                            </code>
                                          </div>
                                        )}
                                        {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                                          <div>
                                            <span className="text-muted-foreground font-medium">
                                              {t('documentPanel.documentManager.metadataLabel')}
                                            </span>
                                            <pre className="bg-muted mt-1 overflow-x-auto rounded p-2 font-mono text-xs">
                                              {formatMetadata(doc.metadata)}
                                            </pre>
                                          </div>
                                        )}
                                        {doc.error_msg && (
                                          <div>
                                            <span className="text-destructive font-medium">
                                              {t('documentPanel.documentManager.errorLabel')}
                                            </span>
                                            <pre className="bg-destructive/10 border-destructive/20 text-destructive mt-1 rounded border p-2 font-mono text-xs break-words whitespace-pre-wrap">
                                              {doc.error_msg}
                                            </pre>
                                          </div>
                                        )}
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                )}
                              </React.Fragment>
                            )
                          })}
                        </TableBody>
                      </Table>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </TooltipProvider>
  )
}
