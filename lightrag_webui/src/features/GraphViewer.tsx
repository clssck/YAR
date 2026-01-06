// import { MiniMap } from '@react-sigma/minimap'
import { SigmaContainer, useRegisterEvents, useSigma } from '@react-sigma/core'
import { createEdgeCurveProgram, EdgeCurvedArrowProgram } from '@sigma/edge-curve'
import { createNodeBorderProgram, NodeBorderProgram } from '@sigma/node-border'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Sigma } from 'sigma'
import { EdgeArrowProgram, NodeCircleProgram, NodePointProgram } from 'sigma/rendering'
import type { Settings as SigmaSettings } from 'sigma/settings'

import FocusOnNode from '@/components/graph/FocusOnNode'
import FullScreenControl from '@/components/graph/FullScreenControl'
import GraphControl from '@/components/graph/GraphControl'
import GraphLabels from '@/components/graph/GraphLabels'
import LayoutsControl from '@/components/graph/LayoutsControl'
import Legend from '@/components/graph/Legend'
import LegendButton from '@/components/graph/LegendButton'
import OnboardingHints from '@/components/graph/OnboardingHints'
import OrphanConnectionControl from '@/components/graph/OrphanConnectionControl'
import PropertiesView from '@/components/graph/PropertiesView'
import Settings from '@/components/graph/Settings'
import SettingsDisplay from '@/components/graph/SettingsDisplay'
import ZoomControl from '@/components/graph/ZoomControl'
import LoadingState, { Skeleton } from '@/components/ui/LoadingState'
import Separator from '@/components/ui/Separator'
import { useResponsive } from '@/hooks/useBreakpoint'
import { cn } from '@/lib/utils'

import { labelColorDarkTheme, labelColorLightTheme } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'

import '@react-sigma/core/lib/style.css'
import '@react-sigma/graph-search/lib/style.css'

// Create a distinct program for orphan nodes - thick dashed-style border
// Using a very thick outer ring to make orphans instantly recognizable
const OrphanNodeProgram = createNodeBorderProgram({
  borders: [
    // Outer thick border ring (white gap)
    { color: { attribute: 'color' }, size: { value: 0.4, mode: 'relative' } },
    // Inner contrasting ring
    { color: { value: '#374151' }, size: { value: 0.15, mode: 'relative' } },
  ],
})

// Function to create sigma settings based on theme
const createSigmaSettings = (isDarkTheme: boolean): Partial<SigmaSettings> => ({
  allowInvalidContainer: true,
  defaultNodeType: 'default',
  defaultEdgeType: 'curvedNoArrow',
  renderEdgeLabels: false,
  edgeProgramClasses: {
    arrow: EdgeArrowProgram,
    curvedArrow: EdgeCurvedArrowProgram,
    curvedNoArrow: createEdgeCurveProgram(),
  },
  nodeProgramClasses: {
    default: NodeBorderProgram,
    circle: NodeCircleProgram,
    point: NodePointProgram,
    // Orphan nodes (degree = 0) render as triangles for visual distinction
    orphan: OrphanNodeProgram,
  },
  labelGridCellSize: 60,
  labelRenderedSizeThreshold: 12,
  enableEdgeEvents: true,
  labelColor: {
    color: isDarkTheme ? labelColorDarkTheme : labelColorLightTheme,
    attribute: 'labelColor',
  },
  edgeLabelColor: {
    color: isDarkTheme ? labelColorDarkTheme : labelColorLightTheme,
    attribute: 'labelColor',
  },
  edgeLabelSize: 8,
  labelSize: 12,
  // minEdgeThickness: 2
  // labelFont: 'Lato, sans-serif'
})

// Keep focus logic isolated to avoid re-rendering the whole viewer during hover/selection churn
const FocusSync = () => {
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const moveToSelectedNode = useGraphStore.use.moveToSelectedNode()

  const autoFocusedNode = useMemo(() => focusedNode ?? selectedNode, [focusedNode, selectedNode])

  return <FocusOnNode node={autoFocusedNode} move={moveToSelectedNode} />
}


const GraphEvents = () => {
  const registerEvents = useRegisterEvents()
  const sigma = useSigma()
  const [draggedNode, setDraggedNode] = useState<string | null>(null)

  useEffect(() => {
    // Register the events
    registerEvents({
      downNode: (e) => {
        setDraggedNode(e.node)
        sigma.getGraph().setNodeAttribute(e.node, 'highlighted', true)
      },
      // On mouse move, if the drag mode is enabled, we change the position of the draggedNode
      mousemovebody: (e) => {
        if (!draggedNode) return
        // Get new position of node
        const pos = sigma.viewportToGraph(e)
        sigma.getGraph().setNodeAttribute(draggedNode, 'x', pos.x)
        sigma.getGraph().setNodeAttribute(draggedNode, 'y', pos.y)

        // Prevent sigma to move camera:
        e.preventSigmaDefault()
        e.original.preventDefault()
        e.original.stopPropagation()
      },
      // On mouse up, we reset the autoscale and the dragging mode
      mouseup: () => {
        if (draggedNode) {
          setDraggedNode(null)
          sigma.getGraph().removeNodeAttribute(draggedNode, 'highlighted')
        }
      },
      // Disable the autoscale at the first down interaction
      mousedown: (e) => {
        // Only set custom BBox if it's a drag operation (mouse button is pressed)
        const mouseEvent = e.original as MouseEvent
        if (mouseEvent.buttons !== 0 && !sigma.getCustomBBox()) {
          sigma.setCustomBBox(sigma.getBBox())
        }
      },
    })
  }, [registerEvents, sigma, draggedNode])

  return null
}

const GraphViewer = () => {
  // DEBUG: Uncomment next 2 lines to disable GraphViewer for testing
  // return <div className="flex items-center justify-center h-full text-muted-foreground">Graph viewer temporarily disabled for debugging</div>

  const { t } = useTranslation()
  const { isMobile, isTablet } = useResponsive()
  const [isThemeSwitching, setIsThemeSwitching] = useState(false)
  const sigmaRef = useRef<Sigma | null>(null)
  const prevTheme = useRef<string>('')

  const isFetching = useGraphStore.use.isFetching()
  const graphIsEmpty = useGraphStore.use.graphIsEmpty()
  const sigmaGraph = useGraphStore.use.sigmaGraph()

  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const enableNodeDrag = useSettingsStore.use.enableNodeDrag()
  const showLegend = useSettingsStore.use.showLegend()
  const theme = useSettingsStore.use.theme()

  // Graph stats for display
  const nodeCount = sigmaGraph?.order ?? 0
  const edgeCount = sigmaGraph?.size ?? 0

  // Compact mode for mobile/tablet
  const isCompact = isMobile || isTablet

  // Memoize sigma settings to prevent unnecessary re-creation
  const memoizedSigmaSettings = useMemo(() => {
    const isDarkTheme = theme === 'dark'
    return createSigmaSettings(isDarkTheme)
  }, [theme])

  // Initialize sigma settings based on theme with theme switching protection
  useEffect(() => {
    // Detect theme change
    const isThemeChange = prevTheme.current && prevTheme.current !== theme
    if (isThemeChange) {
      setIsThemeSwitching(true)
      console.log('Theme switching detected:', prevTheme.current, '->', theme)

      // Reset theme switching state after a short delay
      const timer = setTimeout(() => {
        setIsThemeSwitching(false)
        console.log('Theme switching completed')
      }, 150)

      return () => clearTimeout(timer)
    }
    prevTheme.current = theme
    console.log('Initialized sigma settings for theme:', theme)
  }, [theme])

  // Clean up sigma instance when component unmounts
  useEffect(() => {
    return () => {
      // TAB is mount twice in vite dev mode, this is a workaround

      const sigma = useGraphStore.getState().sigmaInstance
      if (sigma) {
        try {
          // Destroy sigma，and clear WebGL context
          sigma.kill()
          useGraphStore.getState().setSigmaInstance(null)
          console.log('Cleared sigma instance on Graphviewer unmount')
        } catch (error) {
          console.error('Error cleaning up sigma instance:', error)
        }
      }
    }
  }, [])

  // Note: There was a useLayoutEffect hook here to set up the sigma instance and graph data,
  // but testing showed it wasn't executing or having any effect, while the backup mechanism
  // in GraphControl was sufficient. This code was removed to simplify implementation

  // Always render SigmaContainer but control its visibility with CSS
  return (
    <div className="relative h-full w-full overflow-hidden">
      <SigmaContainer
        settings={memoizedSigmaSettings}
        className="!bg-background !size-full overflow-hidden"
        ref={sigmaRef}
      >
        <GraphControl />

        {enableNodeDrag && <GraphEvents />}

        <FocusSync />

        {/* Label selector - top left (hidden on mobile) */}
        {!isCompact && (
          <div className="absolute top-2 left-2 flex items-start gap-2">
            <GraphLabels />
          </div>
        )}

        {/* Control panel - bottom left (horizontal on mobile, vertical on desktop) */}
        <div
          className={cn(
            'bg-background/60 absolute backdrop-blur-lg border-2 rounded-xl',
            isCompact
              ? 'bottom-2 left-1/2 -translate-x-1/2 flex flex-row items-center gap-0.5 p-1'
              : 'bottom-2 left-2 flex flex-col p-1'
          )}
        >
          {/* Layout group */}
          <LayoutsControl />

          {/* Separator - horizontal on mobile, vertical on desktop */}
          <Separator
            orientation={isCompact ? 'vertical' : 'horizontal'}
            className={cn(isCompact ? 'h-6 mx-0.5' : 'my-1')}
          />

          {/* View group */}
          <ZoomControl />
          <FullScreenControl />

          {/* Separator and Filter group - desktop only */}
          {!isCompact && (
            <>
              <Separator className="my-1" />
              <LegendButton />
              <OrphanConnectionControl />
            </>
          )}

          {/* Separator */}
          <Separator
            orientation={isCompact ? 'vertical' : 'horizontal'}
            className={cn(isCompact ? 'h-6 mx-0.5' : 'my-1')}
          />

          {/* Settings */}
          <Settings />
        </div>

        {/* Properties panel - right side on desktop, bottom sheet on mobile (handled in PropertiesView) */}
        {showPropertyPanel && (
          <div className={cn('absolute z-10', isMobile ? '' : 'top-2 right-2')}>
            <PropertiesView />
          </div>
        )}

        {/* Legend - bottom right (hidden on mobile) */}
        {showLegend && !isCompact && (
          <div className="absolute bottom-10 right-2 z-0">
            <Legend className="bg-background/60 backdrop-blur-lg" />
          </div>
        )}

        <SettingsDisplay />

        {/* Onboarding hints for first-time users */}
        {!isFetching && !graphIsEmpty && <OnboardingHints />}

        {/* Graph stats display - top right on mobile, bottom right on desktop */}
        {!isFetching && !graphIsEmpty && nodeCount > 0 && (
          <div
            className={cn(
              'absolute z-0 bg-background/60 backdrop-blur-lg rounded-lg px-2 py-1 text-xs text-muted-foreground border',
              isCompact ? 'top-2 right-2' : 'bottom-2 right-2 px-3 py-1.5'
            )}
          >
            <span className="font-medium">{nodeCount.toLocaleString()}</span>
            <span className={cn('mx-1', isCompact && 'hidden sm:inline')}>
              {t('graphPanel.stats.nodes', 'nodes')}
            </span>
            {!isCompact && <span className="text-muted-foreground/50 mx-1">|</span>}
            {isCompact && <span className="text-muted-foreground/50 mx-0.5">/</span>}
            <span className="font-medium">{edgeCount.toLocaleString()}</span>
            <span className={cn('mx-1', isCompact && 'hidden sm:inline')}>
              {t('graphPanel.stats.edges', 'edges')}
            </span>
          </div>
        )}
      </SigmaContainer>

      {/* Loading overlay - shown when data is loading or theme is switching */}
      {(isFetching || isThemeSwitching) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm z-10">
          {/* Skeleton graph placeholder */}
          <div className="relative w-64 h-64 mb-6">
            {/* Skeleton nodes in a pattern */}
            <Skeleton className="absolute top-8 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full" />
            <Skeleton className="absolute top-20 left-8 w-6 h-6 rounded-full opacity-70" />
            <Skeleton className="absolute top-20 right-8 w-6 h-6 rounded-full opacity-70" />
            <Skeleton className="absolute bottom-20 left-12 w-5 h-5 rounded-full opacity-50" />
            <Skeleton className="absolute bottom-20 right-12 w-5 h-5 rounded-full opacity-50" />
            <Skeleton className="absolute bottom-8 left-1/2 -translate-x-1/2 w-7 h-7 rounded-full opacity-60" />
            {/* Skeleton edges */}
            <Skeleton className="absolute top-16 left-1/4 w-16 h-0.5 rotate-45 opacity-30" />
            <Skeleton className="absolute top-16 right-1/4 w-16 h-0.5 -rotate-45 opacity-30" />
            <Skeleton className="absolute top-1/2 left-1/2 -translate-x-1/2 w-24 h-0.5 opacity-30" />
          </div>
          <LoadingState
            variant="centered"
            size="lg"
            message={
              isThemeSwitching
                ? t('graphPanel.loading.switchingTheme', 'Switching theme...')
                : t('graphPanel.loading.fetchingData', 'Loading graph data...')
            }
          />
        </div>
      )}
    </div>
  )
}

export default GraphViewer
