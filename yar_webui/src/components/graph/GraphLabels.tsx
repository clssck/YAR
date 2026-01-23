import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getPopularLabels } from '@/api/yar'
import LabelSelector from '@/components/graph/LabelSelector'
import Button from '@/components/ui/Button'
import { controlButtonVariant, popularLabelsDefaultLimit } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

const GraphLabels = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const setQueryLabel = useSettingsStore.use.setQueryLabel()
  const dropdownRefreshTrigger = useSettingsStore.use.searchLabelDropdownRefreshTrigger()
  const pipelineBusy = useBackendState.use.pipelineBusy()
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Suppress unused variable warnings for store subscriptions we need for reactivity
  void dropdownRefreshTrigger
  void pipelineBusy

  // Initialize search history on component mount
  useEffect(() => {
    const initializeHistory = async () => {
      const history = SearchHistoryManager.getHistory()

      if (history.length === 0) {
        try {
          const popularLabels = await getPopularLabels(popularLabelsDefaultLimit)
          await SearchHistoryManager.initializeWithDefaults(popularLabels)
        } catch (error) {
          console.error('Failed to initialize search history:', error)
        }
      }
    }

    initializeHistory()
  }, [])

  // Refresh handler - clears graph cache and triggers reload
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true)
    try {
      useGraphStore.getState().setTypeColorMap(new Map<string, string>())
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      useGraphStore.getState().setLastSuccessfulQueryLabel('')
      useGraphStore.getState().incrementGraphDataVersion()
    } finally {
      setIsRefreshing(false)
    }
  }, [])

  // Handle label selection from dropdown
  const handleLabelChange = useCallback(
    (newLabel: string) => {
      setQueryLabel(newLabel)
      // Reset fetch flag to trigger graph update with new label
      useGraphStore.getState().setGraphDataFetchAttempted(false)
    },
    [setQueryLabel]
  )

  return (
    <div className="flex items-center">
      <Button
        size="icon"
        variant={controlButtonVariant}
        onClick={handleRefresh}
        tooltip={t('graphPanel.graphLabels.refreshGlobalTooltip')}
        className="mr-2"
        disabled={isRefreshing}
      >
        <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
      </Button>
      <LabelSelector value={label} onChange={handleLabelChange} disabled={isRefreshing} />
    </div>
  )
}

export default GraphLabels
