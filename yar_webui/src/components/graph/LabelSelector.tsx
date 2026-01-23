import { Check, ChevronsUpDown, Loader2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getPopularLabels, searchLabels } from '@/api/yar'
import Button from '@/components/ui/Button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/Command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { useDebounce } from '@/hooks/useDebounce'
import { controlButtonVariant, searchLabelsDefaultLimit } from '@/lib/constants'
import { cn } from '@/lib/utils'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

interface LabelSelectorProps {
  value: string
  onChange: (label: string) => void
  disabled?: boolean
}

const POPULAR_LABELS_LIMIT = 50
const RECENT_LABELS_LIMIT = 5

/**
 * LabelSelector - A performant label dropdown using Command + Popover pattern
 *
 * Key design decisions to prevent UI freezes:
 * 1. Popular labels loaded ONCE on mount (not on every dropdown open)
 * 2. Search API calls debounced by 300ms
 * 3. No dependency cascades - selection doesn't trigger refetch
 * 4. Limited rendered items per section
 */
const LabelSelector = ({ value, onChange, disabled = false }: LabelSelectorProps) => {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [popularLabels, setPopularLabels] = useState<string[]>([])
  const [searchResults, setSearchResults] = useState<string[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [isLoadingPopular, setIsLoadingPopular] = useState(false)

  const debouncedSearchTerm = useDebounce(searchTerm, 300)

  // Load popular labels ONCE on mount
  useEffect(() => {
    let cancelled = false

    const loadPopularLabels = async () => {
      setIsLoadingPopular(true)
      try {
        const labels = await getPopularLabels(POPULAR_LABELS_LIMIT)
        if (!cancelled) {
          setPopularLabels(labels.filter((l) => l !== '*'))
        }
      } catch (error) {
        console.error('Failed to load popular labels:', error)
      } finally {
        if (!cancelled) {
          setIsLoadingPopular(false)
        }
      }
    }

    loadPopularLabels()

    return () => {
      cancelled = true
    }
  }, [])

  // Search API when user types (debounced)
  useEffect(() => {
    if (!debouncedSearchTerm.trim()) {
      setSearchResults([])
      return
    }

    let cancelled = false

    const performSearch = async () => {
      setIsSearching(true)
      try {
        const results = await searchLabels(debouncedSearchTerm.trim(), searchLabelsDefaultLimit)
        if (!cancelled) {
          setSearchResults(results.filter((l) => l !== '*'))
        }
      } catch (error) {
        console.error('Failed to search labels:', error)
        if (!cancelled) {
          setSearchResults([])
        }
      } finally {
        if (!cancelled) {
          setIsSearching(false)
        }
      }
    }

    performSearch()

    return () => {
      cancelled = true
    }
  }, [debouncedSearchTerm])

  // Get recent searches from localStorage (synchronous, fast)
  const recentLabels = useMemo(() => {
    const recent = SearchHistoryManager.getRecentSearches(RECENT_LABELS_LIMIT)
    return recent.map((item) => item.label).filter((l) => l !== '*')
  }, [])

  const handleSelect = useCallback(
    (label: string) => {
      onChange(label)
      SearchHistoryManager.addToHistory(label)
      setOpen(false)
      setSearchTerm('')
    },
    [onChange]
  )

  // Clear search when dropdown closes
  const handleOpenChange = useCallback((newOpen: boolean) => {
    setOpen(newOpen)
    if (!newOpen) {
      setSearchTerm('')
      setSearchResults([])
    }
  }, [])

  // Determine what to show based on search state
  const showSearchResults = debouncedSearchTerm.trim().length > 0
  const showRecentAndPopular = !showSearchResults

  // Deduplicate labels across sections
  const displayedLabels = useMemo(() => {
    const seen = new Set<string>()
    const result = {
      recent: [] as string[],
      popular: [] as string[],
      search: [] as string[],
    }

    if (showSearchResults) {
      // When searching, only show search results
      for (const label of searchResults) {
        if (!seen.has(label)) {
          seen.add(label)
          result.search.push(label)
        }
      }
    } else {
      // When not searching, show recent + popular
      for (const label of recentLabels) {
        if (!seen.has(label)) {
          seen.add(label)
          result.recent.push(label)
        }
      }
      for (const label of popularLabels) {
        if (!seen.has(label)) {
          seen.add(label)
          result.popular.push(label)
        }
      }
    }

    return result
  }, [showSearchResults, recentLabels, popularLabels, searchResults])

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <Button
          variant={controlButtonVariant}
          role="combobox"
          aria-expanded={open}
          className="justify-between min-w-[80px] max-w-[200px] px-2 h-8"
          disabled={disabled}
          tooltip={t('graphPanel.graphLabels.selectLabelTooltip')}
        >
          <span className="truncate text-sm">{value || '*'}</span>
          <ChevronsUpDown className="ml-1 h-3 w-3 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[250px] p-0"
        align="start"
        sideOffset={8}
        collisionPadding={5}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <Command shouldFilter={false}>
          <div className="relative">
            <CommandInput
              placeholder={t('graphPanel.graphLabels.searchPlaceholder')}
              value={searchTerm}
              onValueChange={setSearchTerm}
            />
            {isSearching && (
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            )}
          </div>
          <CommandList>
            {/* Always show wildcard at top */}
            <CommandGroup>
              <CommandItem value="" onSelect={() => handleSelect('*')} className="cursor-pointer">
                <span className="font-mono">*</span>
                <span className="ml-2 text-xs text-muted-foreground">
                  {t('graphPanel.graphLabels.allLabels')}
                </span>
                <Check
                  className={cn('ml-auto h-4 w-4', value === '*' ? 'opacity-100' : 'opacity-0')}
                />
              </CommandItem>
            </CommandGroup>

            {/* Search results */}
            {showSearchResults &&
              (displayedLabels.search.length > 0 ? (
                <CommandGroup heading={t('graphPanel.graphLabels.searchResults')}>
                  {displayedLabels.search.map((label) => (
                    <CommandItem
                      key={label}
                      value=""
                      onSelect={() => handleSelect(label)}
                      className="cursor-pointer truncate"
                    >
                      {label}
                      <Check
                        className={cn(
                          'ml-auto h-4 w-4 shrink-0',
                          value === label ? 'opacity-100' : 'opacity-0'
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              ) : (
                !isSearching && <CommandEmpty>{t('graphPanel.graphLabels.noResults')}</CommandEmpty>
              ))}

            {/* Recent + Popular when not searching */}
            {showRecentAndPopular && (
              <>
                {displayedLabels.recent.length > 0 && (
                  <CommandGroup heading={t('graphPanel.graphLabels.recent')}>
                    {displayedLabels.recent.map((label) => (
                      <CommandItem
                        key={label}
                        value=""
                        onSelect={() => handleSelect(label)}
                        className="cursor-pointer truncate"
                      >
                        {label}
                        <Check
                          className={cn(
                            'ml-auto h-4 w-4 shrink-0',
                            value === label ? 'opacity-100' : 'opacity-0'
                          )}
                        />
                      </CommandItem>
                    ))}
                  </CommandGroup>
                )}

                {isLoadingPopular ? (
                  <CommandGroup heading={t('graphPanel.graphLabels.popular')}>
                    <CommandItem disabled className="justify-center">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </CommandItem>
                  </CommandGroup>
                ) : (
                  displayedLabels.popular.length > 0 && (
                    <CommandGroup heading={t('graphPanel.graphLabels.popular')}>
                      {displayedLabels.popular.map((label) => (
                        <CommandItem
                          key={label}
                          value=""
                          onSelect={() => handleSelect(label)}
                          className="cursor-pointer truncate"
                        >
                          {label}
                          <Check
                            className={cn(
                              'ml-auto h-4 w-4 shrink-0',
                              value === label ? 'opacity-100' : 'opacity-0'
                            )}
                          />
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  )
                )}
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export default LabelSelector
