import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { reprocessFailedDocuments, scanNewDocuments } from '@/api/yar'
import { useBackendState } from '@/stores/state'

interface UsePipelineMutationsArgs {
  /**
   * Called with `Date.now() + 15_000` when a mutation succeeds. The owning component
   * passes this through to `useDocumentListQuery` so the list polls every 2s for the
   * next 15 seconds, then resumes the active/idle cadence.
   */
  setBoostUntil: (boostUntil: number | null) => void
}

const BOOST_DURATION_MS = 15_000

/**
 * Pipeline mutations for the document manager: scan-new and retry-failed.
 *
 * Replaces the manual scanDocuments/retryFailedDocuments callbacks in DocumentManager,
 * which interleaved their own setTimeout-based recovery cadence with the polling engine.
 * Now: each mutation invalidates the documents query (drops cache), pings the backend
 * health-check timer, and sets a 15s boost window so the list refetches at 2s cadence
 * until the new state settles.
 */
export function usePipelineMutations({ setBoostUntil }: UsePipelineMutationsArgs) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const startBoost = () => {
    useBackendState.getState().resetHealthCheckTimerDelayed(1000)
    queryClient.invalidateQueries({ queryKey: ['documents'] })
    setBoostUntil(Date.now() + BOOST_DURATION_MS)
    window.setTimeout(() => setBoostUntil(null), BOOST_DURATION_MS)
  }

  const scan = useMutation({
    mutationFn: () => scanNewDocuments(),
    onSuccess: ({ status, message }) => {
      toast.message(message || status)
      startBoost()
    },
    onError: (err) => {
      toast.error(
        t('documentPanel.documentManager.errors.scanFailed', {
          error: err instanceof Error ? err.message : String(err)
        })
      )
    }
  })

  const retry = useMutation({
    mutationFn: () => reprocessFailedDocuments(),
    onSuccess: ({ status, message }) => {
      if (status === 'reprocessing_started') {
        toast.success(
          t('documentPanel.documentManager.retrySuccess', 'Retrying failed documents...')
        )
        startBoost()
      } else {
        toast.error(
          message || t('documentPanel.documentManager.retryFailed', 'Failed to retry documents')
        )
      }
    },
    onError: (err) => {
      toast.error(
        t('documentPanel.documentManager.errors.retryFailed', {
          error: err instanceof Error ? err.message : String(err)
        })
      )
    }
  })

  return { scan, retry }
}
