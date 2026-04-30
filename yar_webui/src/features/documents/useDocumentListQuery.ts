import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  type DocStatus,
  type DocumentsRequest,
  getDocumentsPaginated,
  type PaginatedDocsResponse
} from '@/api/yar'
import { useBackendState } from '@/stores/state'
import { hasActiveDocumentsStatus } from '@/utils/documentUtils'

type SortField = DocumentsRequest['sort_field']
type SortDirection = DocumentsRequest['sort_direction']
type StatusFilter = DocStatus | 'all'

interface DocumentListQueryArgs {
  page: number
  pageSize: number
  statusFilter: StatusFilter
  sortField: SortField
  sortDirection: SortDirection
  /** Local boost: when set, the query refetches every 2s for `boostUntil - Date.now()` ms. */
  boostUntil: number | null
  /** Whether the documents tab is currently visible. */
  enabled: boolean
}

const POLL_FAST_MS = 2_000
const POLL_ACTIVE_MS = 5_000
const POLL_IDLE_MS = 30_000

/**
 * Document list query with pagination, sorting, status filtering, and adaptive polling.
 *
 * Replaces DocumentManager's hand-rolled circuit breaker, classifyError(), withTimeout(),
 * and the manual setInterval polling engine. TanStack Query handles retry, abort-on-unmount,
 * and stale-while-revalidate for free.
 *
 * Polling cadence (after the query has succeeded once):
 *   - boost window:        2s   (set by mutations to confirm a state transition quickly)
 *   - active processing:   5s   (any docs in PROCESSING/PENDING/PREPROCESSED)
 *   - idle:               30s
 *
 * The effective interval is recomputed on every successful fetch — TanStack Query
 * passes the latest data to `refetchInterval` so this stays reactive without effects.
 */
export function useDocumentListQuery(args: DocumentListQueryArgs) {
  const { t } = useTranslation()
  const documentListVersion = useBackendState.use.documentListVersion()

  const query = useQuery<PaginatedDocsResponse>({
    queryKey: [
      'documents',
      {
        page: args.page,
        pageSize: args.pageSize,
        statusFilter: args.statusFilter,
        sortField: args.sortField,
        sortDirection: args.sortDirection,
        version: documentListVersion
      }
    ],
    queryFn: ({ signal }) =>
      getDocumentsPaginated(
        {
          status_filter: args.statusFilter === 'all' ? null : args.statusFilter,
          page: args.page,
          page_size: args.pageSize,
          sort_field: args.sortField,
          sort_direction: args.sortDirection
        },
        { signal }
      ),
    enabled: args.enabled,
    placeholderData: keepPreviousData,
    refetchOnWindowFocus: false,
    retry: 3,
    retryDelay: (attempt) => Math.min(2 ** attempt * 1000, 30_000),
    refetchInterval: (q) => {
      if (!args.enabled) return false
      if (args.boostUntil !== null && Date.now() < args.boostUntil) {
        return POLL_FAST_MS
      }
      const counts = q.state.data?.status_counts ?? {}
      return hasActiveDocumentsStatus(counts) ? POLL_ACTIVE_MS : POLL_IDLE_MS
    }
  })

  if (query.isError && query.errorUpdateCount === 1) {
    toast.error(
      t('documentPanel.documentManager.errors.loadFailed', {
        error: query.error instanceof Error ? query.error.message : String(query.error)
      })
    )
  }

  return query
}
