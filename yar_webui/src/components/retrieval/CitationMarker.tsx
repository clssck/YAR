/**
 * CitationMarker Component
 *
 * Renders citation markers (e.g., [1]) as interactive hover cards
 * showing source metadata like document title, section, page, and excerpt.
 */

import { ExternalLinkIcon, FileTextIcon, LinkIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { CitationSource } from '@/api/yar'
import { getDocumentUrl } from '@/api/yar'
import Badge from '@/components/ui/Badge'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/HoverCard'
import { cn } from '@/lib/utils'
import { isSafeUrl } from '@/utils/url'

interface CitationMarkerProps {
  /** The citation marker text, e.g., "[1]" or "[1,2]" */
  marker: string
  /** Reference IDs this marker cites */
  referenceIds: string[]
  /** Confidence score (0-1) */
  confidence: number
  /** Source metadata for hover card */
  sources: CitationSource[]
}

/**
 * Get confidence level and styling based on score
 */
function getConfidenceLevel(confidence: number) {
  if (confidence >= 0.8) {
    return {
      level: 'high',
      color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      markerBg: 'bg-green-100/80 dark:bg-green-900/40',
      markerBorder: 'border-green-300 dark:border-green-700'
    }
  }
  if (confidence >= 0.6) {
    return {
      level: 'medium',
      color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
      markerBg: 'bg-yellow-100/80 dark:bg-yellow-900/40',
      markerBorder: 'border-yellow-300 dark:border-yellow-700'
    }
  }
  return {
    level: 'low',
    color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    markerBg: 'bg-red-100/80 dark:bg-red-900/40',
    markerBorder: 'border-red-300 dark:border-red-700'
  }
}

/**
 * Interactive citation marker with hover card showing source metadata
 */
export function CitationMarker({ marker, referenceIds, confidence, sources }: CitationMarkerProps) {
  const { t } = useTranslation()
  // Find sources matching our reference IDs (deduplicated by reference_id)
  const matchingSources = sources
    .filter((s) => referenceIds.includes(s.reference_id))
    .filter(
      (source, index, arr) => arr.findIndex((s) => s.reference_id === source.reference_id) === index
    )

  // Confidence styling
  const confidenceInfo = getConfidenceLevel(confidence)
  const confidenceLabel = t(
    `retrievePanel.citation.confidence.${confidenceInfo.level}`,
    confidenceInfo.level.charAt(0).toUpperCase() + confidenceInfo.level.slice(1)
  )

  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className={cn(
            'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md border',
            'text-primary hover:text-primary/80 cursor-pointer font-medium text-sm',
            'mx-0.5 transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-primary/20',
            confidenceInfo.markerBg,
            confidenceInfo.markerBorder
          )}
          title={t('retrievePanel.citation.clickToView', 'Click to view source')}
        >
          <LinkIcon className="h-3 w-3 opacity-70" />
          <span>{marker}</span>
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top" align="center">
        <div className="space-y-3">
          {matchingSources.map((source) => (
            <div key={source.reference_id} className="space-y-2">
              {/* Document title */}
              <div className="flex items-start gap-2">
                <FileTextIcon className="text-muted-foreground mt-0.5 h-4 w-4 shrink-0" />
                <h4 className="text-sm leading-tight font-semibold">
                  {source.document_title ||
                    t('retrievePanel.citation.untitled', 'Untitled Document')}
                </h4>
              </div>

              {/* Section title */}
              {source.section_title && (
                <p className="text-muted-foreground pl-6 text-xs">
                  {t('retrievePanel.citation.section', 'Section')}: {source.section_title}
                </p>
              )}

              {/* Page range */}
              {source.page_range && (
                <p className="text-muted-foreground pl-6 text-xs">
                  {t('retrievePanel.citation.pages', 'Pages')}: {source.page_range}
                </p>
              )}

              {/* Excerpt */}
              {source.excerpt && (
                <blockquote className="border-muted text-muted-foreground line-clamp-3 border-l-2 pl-6 text-xs italic">
                  "{source.excerpt}"
                </blockquote>
              )}

              {/* File path with optional link */}
              {(() => {
                const docUrl = getDocumentUrl(source)
                return docUrl ? (
                  <a
                    href={isSafeUrl(docUrl) ? docUrl : '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary/80 group flex items-center gap-1 truncate pl-6 text-xs"
                    title={t('retrievePanel.citation.openDocument', 'Open document in new tab')}
                  >
                    <ExternalLinkIcon className="h-3 w-3 shrink-0 opacity-70 group-hover:opacity-100" />
                    <span className="truncate underline underline-offset-2">
                      {source.file_path}
                    </span>
                  </a>
                ) : (
                  <p
                    className="text-muted-foreground/70 truncate pl-6 text-xs"
                    title={source.file_path}
                  >
                    {source.file_path}
                  </p>
                )
              })()}
            </div>
          ))}

          {/* Confidence badge with text label */}
          <div className="flex items-center justify-between border-t pt-2">
            <span className="text-muted-foreground text-xs">
              {t('retrievePanel.citation.matchConfidence', 'Match confidence')}
            </span>
            <Badge variant="outline" className={confidenceInfo.color}>
              {confidenceLabel} ({(confidence * 100).toFixed(0)}%)
            </Badge>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}

/**
 * Parses text containing citation markers and returns React elements
 * with interactive CitationMarker components.
 *
 * @param text - Text that may contain [n] or [n,m] patterns
 * @param sources - Array of citation sources for hover card metadata
 * @param markers - Array of citation markers with position and confidence data
 * @returns Array of React elements (strings and CitationMarker components)
 */
export function renderTextWithCitations(
  text: string,
  sources: CitationSource[],
  markers: Array<{
    marker: string
    reference_ids: string[]
    confidence: number
  }>
): React.ReactNode[] {
  // Match citation patterns like [1], [2], [1,2], etc.
  const citationPattern = /\[(\d+(?:,\d+)*)\]/g
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null = citationPattern.exec(text)

  while (match !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    // Parse reference IDs from the marker
    const markerText = match[0]
    const refIds = match[1].split(',').map((id) => id.trim())

    // Find matching marker data for confidence
    const markerData = markers.find((m) => m.marker === markerText)
    const confidence = markerData?.confidence ?? 0.5

    // Add the citation marker component
    parts.push(
      <CitationMarker
        key={`citation-${match.index}`}
        marker={markerText}
        referenceIds={refIds}
        confidence={confidence}
        sources={sources}
      />
    )

    lastIndex = match.index + match[0].length
    match = citationPattern.exec(text)
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts
}

/**
 * Citation summary component shown at the end of messages with citations
 */
interface CitationSummaryProps {
  /** Number of unique sources cited */
  sourceCount: number
  /** Total number of citation markers */
  citationCount: number
}

export function CitationSummary({ sourceCount, citationCount }: CitationSummaryProps) {
  const { t } = useTranslation()

  if (sourceCount === 0) return null

  return (
    <div className="border-muted text-muted-foreground mt-3 flex items-center gap-2 border-t pt-2 text-xs">
      <LinkIcon className="h-3.5 w-3.5" />
      <span>
        {t('retrievePanel.citation.summary', 'Cited from {{count}} source(s)', {
          count: sourceCount
        })}
        {citationCount > sourceCount && (
          <span className="ml-1 opacity-70">
            ({citationCount} {t('retrievePanel.citation.references', 'references')})
          </span>
        )}
      </span>
    </div>
  )
}
