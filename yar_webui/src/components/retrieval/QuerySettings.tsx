import { RotateCcw, Scale, Search, Zap } from 'lucide-react'
import { useCallback, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import type { QueryMode, QueryRequest } from '@/api/yar'
import Button from '@/components/ui/Button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/Card'
import Checkbox from '@/components/ui/Checkbox'
import CollapsibleSection from '@/components/ui/CollapsibleSection'
import Input from '@/components/ui/Input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/Tooltip'
import UserPromptInputWithHistory from '@/components/ui/UserPromptInputWithHistory'
import { useSettingsStore } from '@/stores/settings'

// Quick preset configurations
const PRESETS = {
  fast: {
    mode: 'local' as QueryMode,
    top_k: 20,
    chunk_top_k: 10,
    max_entity_tokens: 3000,
    max_relation_tokens: 4000,
    max_total_tokens: 15000,
    enable_rerank: false,
  },
  balanced: {
    mode: 'mix' as QueryMode,
    top_k: 40,
    chunk_top_k: 20,
    max_entity_tokens: 6000,
    max_relation_tokens: 8000,
    max_total_tokens: 30000,
    enable_rerank: false,
  },
  thorough: {
    mode: 'hybrid' as QueryMode,
    top_k: 60,
    chunk_top_k: 30,
    max_entity_tokens: 10000,
    max_relation_tokens: 12000,
    max_total_tokens: 50000,
    enable_rerank: false,
  },
}

export default function QuerySettings() {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)
  const userPromptHistory = useSettingsStore((state) => state.userPromptHistory)
  const storageConfig = useSettingsStore((state) => state.storageConfig)

  // Check if reranker is configured (rerank_binding is not null/undefined/'null')
  const isRerankerAvailable =
    storageConfig?.rerank_binding != null &&
    storageConfig.rerank_binding !== 'null' &&
    storageConfig.rerank_binding !== ''

  const handleChange = useCallback(
    (key: keyof QueryRequest, value: QueryRequest[keyof QueryRequest]) => {
      useSettingsStore.getState().updateQuerySettings({ [key]: value })
    },
    [],
  )

  const handleSelectFromHistory = useCallback(
    (prompt: string) => {
      handleChange('user_prompt', prompt)
    },
    [handleChange],
  )

  const handleDeleteFromHistory = useCallback(
    (index: number) => {
      const newHistory = [...userPromptHistory]
      newHistory.splice(index, 1)
      useSettingsStore.getState().setUserPromptHistory(newHistory)
    },
    [userPromptHistory],
  )

  // Default values for reset functionality
  const defaultValues = useMemo(
    () => ({
      mode: 'mix' as QueryMode,
      top_k: 40,
      chunk_top_k: 20,
      max_entity_tokens: 6000,
      max_relation_tokens: 8000,
      max_total_tokens: 30000,
      citation_mode: 'none' as 'none' | 'inline' | 'footnotes',
      citation_threshold: 0.7,
    }),
    [],
  )

  const handleReset = useCallback(
    (key: keyof typeof defaultValues) => {
      handleChange(key, defaultValues[key])
    },
    [handleChange, defaultValues],
  )

  // Apply preset configuration
  const applyPreset = useCallback(
    (preset: keyof typeof PRESETS) => {
      const config = PRESETS[preset]
      Object.entries(config).forEach(([key, value]) => {
        handleChange(key as keyof QueryRequest, value)
      })
    },
    [handleChange],
  )

  // Reset button component
  const ResetButton = ({
    onClick,
    title,
  }: {
    onClick: () => void
    title: string
  }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onClick}
            className="mr-1 p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={title}
          >
            <RotateCcw className="h-3 w-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>{title}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )

  return (
    <Card className="flex shrink-0 flex-col w-[280px]">
      <CardHeader className="px-4 pt-4 pb-2">
        <CardTitle>
          {t('retrievePanel.querySettings.parametersTitle')}
        </CardTitle>
        <CardDescription className="sr-only">
          {t('retrievePanel.querySettings.parametersDescription')}
        </CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-1 overflow-auto px-2 pr-2 pb-2">
            {/* Quick Presets */}
            <div className="flex gap-1 mb-2">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-7 text-xs gap-1"
                onClick={() => applyPreset('fast')}
                tooltip={t(
                  'retrievePanel.querySettings.presets.fastTooltip',
                  'Quick responses with fewer results',
                )}
              >
                <Zap className="h-3 w-3" />
                {t('retrievePanel.querySettings.presets.fast', 'Fast')}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-7 text-xs gap-1"
                onClick={() => applyPreset('balanced')}
                tooltip={t(
                  'retrievePanel.querySettings.presets.balancedTooltip',
                  'Balanced speed and quality',
                )}
              >
                <Scale className="h-3 w-3" />
                {t('retrievePanel.querySettings.presets.balanced', 'Balanced')}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-7 text-xs gap-1"
                onClick={() => applyPreset('thorough')}
                tooltip={t(
                  'retrievePanel.querySettings.presets.thoroughTooltip',
                  'Comprehensive search with more context',
                )}
              >
                <Search className="h-3 w-3" />
                {t('retrievePanel.querySettings.presets.thorough', 'Thorough')}
              </Button>
            </div>

            {/* Retrieval Section - Always Open */}
            <CollapsibleSection
              title={t(
                'retrievePanel.querySettings.sections.retrieval',
                'Retrieval',
              )}
              defaultOpen={true}
            >
              <div className="flex flex-col gap-2">
                {/* Query Mode */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="query_mode_select"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.queryMode')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.queryModeTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Select
                    value={querySettings.mode}
                    onValueChange={(v) => handleChange('mode', v as QueryMode)}
                  >
                    <SelectTrigger
                      id="query_mode_select"
                      className="hover:bg-primary/5 h-8 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0 flex-1 text-left [&>span]:break-all [&>span]:line-clamp-1"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="naive">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.naive',
                          )}
                        </SelectItem>
                        <SelectItem value="local">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.local',
                          )}
                        </SelectItem>
                        <SelectItem value="global">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.global',
                          )}
                        </SelectItem>
                        <SelectItem value="hybrid">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.hybrid',
                          )}
                        </SelectItem>
                        <SelectItem value="mix">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.mix',
                          )}
                        </SelectItem>
                        <SelectItem value="bypass">
                          {t(
                            'retrievePanel.querySettings.queryModeOptions.bypass',
                          )}
                        </SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <ResetButton
                    onClick={() => handleReset('mode')}
                    title="Reset to default (Mix)"
                  />
                </div>

                {/* Top K */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="top_k"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.topK')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.topKTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="top_k"
                    type="number"
                    value={querySettings.top_k ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange(
                        'top_k',
                        value === '' ? '' : Number.parseInt(value, 10) || 0,
                      )
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (
                        value === '' ||
                        Number.isNaN(Number.parseInt(value, 10))
                      ) {
                        handleChange('top_k', 40)
                      }
                    }}
                    min={1}
                    placeholder={t(
                      'retrievePanel.querySettings.topKPlaceholder',
                    )}
                    className="h-8 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('top_k')}
                    title="Reset to default"
                  />
                </div>

                {/* Chunk Top K */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="chunk_top_k"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.chunkTopK')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.chunkTopKTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="chunk_top_k"
                    type="number"
                    value={querySettings.chunk_top_k ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange(
                        'chunk_top_k',
                        value === '' ? '' : Number.parseInt(value, 10) || 0,
                      )
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (
                        value === '' ||
                        Number.isNaN(Number.parseInt(value, 10))
                      ) {
                        handleChange('chunk_top_k', 20)
                      }
                    }}
                    min={1}
                    placeholder={t(
                      'retrievePanel.querySettings.chunkTopKPlaceholder',
                    )}
                    className="h-8 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('chunk_top_k')}
                    title="Reset to default"
                  />
                </div>
              </div>
            </CollapsibleSection>

            {/* Token Budget Section - Collapsed by default */}
            <CollapsibleSection
              title={t(
                'retrievePanel.querySettings.sections.tokenBudget',
                'Token Budget',
              )}
              defaultOpen={false}
            >
              <div className="flex flex-col gap-2">
                {/* Max Entity Tokens */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="max_entity_tokens"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.maxEntityTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>
                        {t(
                          'retrievePanel.querySettings.maxEntityTokensTooltip',
                        )}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_entity_tokens"
                    type="number"
                    value={querySettings.max_entity_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange(
                        'max_entity_tokens',
                        value === '' ? '' : Number.parseInt(value, 10) || 0,
                      )
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (
                        value === '' ||
                        Number.isNaN(Number.parseInt(value, 10))
                      ) {
                        handleChange('max_entity_tokens', 6000)
                      }
                    }}
                    min={1}
                    placeholder={t(
                      'retrievePanel.querySettings.maxEntityTokensPlaceholder',
                    )}
                    className="h-8 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_entity_tokens')}
                    title="Reset to default"
                  />
                </div>

                {/* Max Relation Tokens */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="max_relation_tokens"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.maxRelationTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>
                        {t(
                          'retrievePanel.querySettings.maxRelationTokensTooltip',
                        )}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_relation_tokens"
                    type="number"
                    value={querySettings.max_relation_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange(
                        'max_relation_tokens',
                        value === '' ? '' : Number.parseInt(value, 10) || 0,
                      )
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (
                        value === '' ||
                        Number.isNaN(Number.parseInt(value, 10))
                      ) {
                        handleChange('max_relation_tokens', 8000)
                      }
                    }}
                    min={1}
                    placeholder={t(
                      'retrievePanel.querySettings.maxRelationTokensPlaceholder',
                    )}
                    className="h-8 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_relation_tokens')}
                    title="Reset to default"
                  />
                </div>

                {/* Max Total Tokens */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="max_total_tokens"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.maxTotalTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>
                        {t('retrievePanel.querySettings.maxTotalTokensTooltip')}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_total_tokens"
                    type="number"
                    value={querySettings.max_total_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange(
                        'max_total_tokens',
                        value === '' ? '' : Number.parseInt(value, 10) || 0,
                      )
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (
                        value === '' ||
                        Number.isNaN(Number.parseInt(value, 10))
                      ) {
                        handleChange('max_total_tokens', 30000)
                      }
                    }}
                    min={1}
                    placeholder={t(
                      'retrievePanel.querySettings.maxTotalTokensPlaceholder',
                    )}
                    className="h-8 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_total_tokens')}
                    title="Reset to default"
                  />
                </div>
              </div>
            </CollapsibleSection>

            {/* Output Section */}
            <CollapsibleSection
              title={t('retrievePanel.querySettings.sections.output', 'Output')}
              defaultOpen={false}
            >
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label
                          htmlFor="stream"
                          className="flex-1 ml-1 cursor-help text-xs"
                        >
                          {t('retrievePanel.querySettings.streamResponse')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>
                          {t(
                            'retrievePanel.querySettings.streamResponseTooltip',
                          )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-6 cursor-pointer"
                    id="stream"
                    checked={querySettings.stream}
                    onCheckedChange={(checked) =>
                      handleChange('stream', checked)
                    }
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label
                          htmlFor="show_references_section"
                          className="flex-1 ml-1 cursor-help text-xs"
                        >
                          {t(
                            'retrievePanel.querySettings.showReferencesSection',
                            'Show References',
                          )}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>
                          {t(
                            'retrievePanel.querySettings.showReferencesSectionTooltip',
                            'Show or hide the References section at the bottom of responses',
                          )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-6 cursor-pointer"
                    id="show_references_section"
                    checked={querySettings.show_references_section ?? true}
                    onCheckedChange={(checked) =>
                      handleChange('show_references_section', checked)
                    }
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label
                          htmlFor="only_need_context"
                          className="flex-1 ml-1 cursor-help text-xs"
                        >
                          {t('retrievePanel.querySettings.onlyNeedContext')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>
                          {t(
                            'retrievePanel.querySettings.onlyNeedContextTooltip',
                          )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-6 cursor-pointer"
                    id="only_need_context"
                    checked={querySettings.only_need_context}
                    onCheckedChange={(checked) => {
                      handleChange('only_need_context', checked)
                      if (checked) {
                        handleChange('only_need_prompt', false)
                      }
                    }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label
                          htmlFor="only_need_prompt"
                          className="flex-1 ml-1 cursor-help text-xs"
                        >
                          {t('retrievePanel.querySettings.onlyNeedPrompt')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>
                          {t(
                            'retrievePanel.querySettings.onlyNeedPromptTooltip',
                          )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-6 cursor-pointer"
                    id="only_need_prompt"
                    checked={querySettings.only_need_prompt}
                    onCheckedChange={(checked) => {
                      handleChange('only_need_prompt', checked)
                      if (checked) {
                        handleChange('only_need_context', false)
                      }
                    }}
                  />
                </div>
              </div>
            </CollapsibleSection>

            {/* Advanced Section */}
            <CollapsibleSection
              title={t(
                'retrievePanel.querySettings.sections.advanced',
                'Advanced',
              )}
              defaultOpen={false}
            >
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label
                          htmlFor="enable_rerank"
                          className={`flex-1 ml-1 cursor-help text-xs ${!isRerankerAvailable ? 'opacity-50' : ''}`}
                        >
                          {t('retrievePanel.querySettings.enableRerank')}
                          {!isRerankerAvailable && (
                            <span className="ml-1 text-muted-foreground">
                              (
                              {t(
                                'retrievePanel.querySettings.notConfigured',
                                'not configured',
                              )}
                              )
                            </span>
                          )}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>
                          {isRerankerAvailable
                            ? t(
                                'retrievePanel.querySettings.enableRerankTooltip',
                              )
                            : t(
                                'retrievePanel.querySettings.rerankNotConfiguredTooltip',
                                'No reranker model configured. Set RERANK_BINDING in server config to enable.',
                              )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className={`mr-6 ${isRerankerAvailable ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
                    id="enable_rerank"
                    checked={querySettings.enable_rerank}
                    onCheckedChange={(checked) =>
                      handleChange('enable_rerank', checked)
                    }
                    disabled={!isRerankerAvailable}
                  />
                </div>

                {/* User Prompt */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label
                        htmlFor="user_prompt"
                        className="ml-1 cursor-help text-xs"
                      >
                        {t('retrievePanel.querySettings.userPrompt')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>
                        {t('retrievePanel.querySettings.userPromptTooltip')}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <UserPromptInputWithHistory
                  id="user_prompt"
                  value={querySettings.user_prompt || ''}
                  onChange={(value) => handleChange('user_prompt', value)}
                  onSelectFromHistory={handleSelectFromHistory}
                  onDeleteFromHistory={handleDeleteFromHistory}
                  history={userPromptHistory}
                  placeholder={t(
                    'retrievePanel.querySettings.userPromptPlaceholder',
                  )}
                  className="h-8"
                />
              </div>
            </CollapsibleSection>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
