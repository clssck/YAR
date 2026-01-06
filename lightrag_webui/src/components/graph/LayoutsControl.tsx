import { useSigma } from '@react-sigma/core'
import { useLayoutCirclepack } from '@react-sigma/layout-circlepack'
import { useLayoutCircular } from '@react-sigma/layout-circular'
import type {
  LayoutHook,
  LayoutWorkerHook,
  WorkerLayoutControlProps,
} from '@react-sigma/layout-core'
import { useLayoutForce, useWorkerLayoutForce } from '@react-sigma/layout-force'
import { useLayoutForceAtlas2, useWorkerLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import { useLayoutNoverlap, useWorkerLayoutNoverlap } from '@react-sigma/layout-noverlap'
import { useLayoutRandom } from '@react-sigma/layout-random'
import { GripIcon, Loader2, PlayIcon } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { animateNodes } from 'sigma/utils'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/Command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { controlButtonVariant } from '@/lib/constants'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'

type LayoutName =
  | 'Circular'
  | 'Circlepack'
  | 'Random'
  | 'Noverlaps'
  | 'Force Directed'
  | 'Force Atlas'

// Extend WorkerLayoutControlProps to include mainLayout
interface ExtendedWorkerLayoutControlProps extends WorkerLayoutControlProps {
  mainLayout: LayoutHook
  layoutName?: string
  onRunningChange?: (isRunning: boolean) => void
}

const WorkerLayoutControl = ({
  layout,
  autoRunFor,
  mainLayout,
  layoutName,
  onRunningChange,
}: ExtendedWorkerLayoutControlProps) => {
  const sigma = useSigma()
  // Use local state to track animation running status
  const [isRunning, setIsRunning] = useState(false)
  // Track iteration count for progress display
  const [iterationCount, setIterationCount] = useState(0)
  // Timer reference for animation
  const animationTimerRef = useRef<number | null>(null)
  const { t } = useTranslation()

  // Notify parent when running state changes
  useEffect(() => {
    onRunningChange?.(isRunning)
  }, [isRunning, onRunningChange])

  // Function to update node positions using the layout algorithm
  const updatePositions = useCallback(() => {
    if (!sigma) return

    try {
      const graph = sigma.getGraph()
      if (!graph || graph.order === 0) return

      // Use mainLayout to get positions, similar to refreshLayout function
      const positions = mainLayout.positions()

      // Animate nodes to new positions
      animateNodes(graph, positions, { duration: 300 })

      // Increment iteration count
      setIterationCount((prev) => prev + 1)
    } catch (error) {
      console.error('Error updating positions:', error)
      // Stop animation if there's an error
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
        setIsRunning(false)
        setIterationCount(0)
      }
    }
  }, [sigma, mainLayout])

  // Improved click handler that uses our own animation timer
  const handleClick = useCallback(() => {
    if (isRunning) {
      // Stop the animation
      console.log('Stopping layout animation')
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
      }

      // Try to kill the layout algorithm if it's running
      try {
        if (typeof layout.kill === 'function') {
          layout.kill()
          console.log('Layout algorithm killed')
        } else if (typeof layout.stop === 'function') {
          layout.stop()
          console.log('Layout algorithm stopped')
        }
      } catch (error) {
        console.error('Error stopping layout algorithm:', error)
      }

      // Show completion toast with layout name
      toast.success(
        layoutName
          ? t('graphPanel.sideBar.layoutsControl.layoutStoppedNamed', '{{name}} stopped', {
              name: layoutName,
            })
          : t('graphPanel.sideBar.layoutsControl.layoutStopped', 'Layout stopped')
      )

      setIsRunning(false)
      setIterationCount(0)
    } else {
      // Start the animation
      console.log('Starting layout animation')

      // Reset iteration count
      setIterationCount(0)

      // Initial position update
      updatePositions()

      // Set up interval for continuous updates
      animationTimerRef.current = window.setInterval(() => {
        updatePositions()
      }, 200) // Reduced interval to create overlapping animations for smoother transitions

      setIsRunning(true)

      // Set a timeout to automatically stop the animation after 3 seconds
      setTimeout(() => {
        if (animationTimerRef.current) {
          console.log('Auto-stopping layout animation after 3 seconds')
          window.clearInterval(animationTimerRef.current)
          animationTimerRef.current = null
          setIsRunning(false)
          setIterationCount(0)

          // Try to stop the layout algorithm
          try {
            if (typeof layout.kill === 'function') {
              layout.kill()
            } else if (typeof layout.stop === 'function') {
              layout.stop()
            }
          } catch (error) {
            console.error('Error stopping layout algorithm:', error)
          }

          // Show completion toast with layout name
          toast.success(
            layoutName
              ? t('graphPanel.sideBar.layoutsControl.layoutCompleteNamed', '{{name}} complete', {
                  name: layoutName,
                })
              : t('graphPanel.sideBar.layoutsControl.layoutComplete', 'Layout complete'),
            { duration: 2000 }
          )
        }
      }, 3000)
    }
  }, [isRunning, layout, updatePositions, t, layoutName])

  /**
   * Init component when Sigma or component settings change.
   */
  useEffect(() => {
    if (!sigma) {
      console.log('No sigma instance available')
      return
    }

    // Auto-run if specified
    let timeout: number | null = null
    if (autoRunFor !== undefined && autoRunFor > -1 && sigma.getGraph().order > 0) {
      console.log('Auto-starting layout animation')

      // Initial position update
      updatePositions()

      // Set up interval for continuous updates
      animationTimerRef.current = window.setInterval(() => {
        updatePositions()
      }, 200) // Reduced interval to create overlapping animations for smoother transitions

      setIsRunning(true)

      // Set a timeout to stop it if autoRunFor > 0
      if (autoRunFor > 0) {
        timeout = window.setTimeout(() => {
          console.log('Auto-stopping layout animation after timeout')
          if (animationTimerRef.current) {
            window.clearInterval(animationTimerRef.current)
            animationTimerRef.current = null
          }
          setIsRunning(false)
        }, autoRunFor)
      }
    }

    // Cleanup function
    return () => {
      // console.log('Cleaning up WorkerLayoutControl')
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
      }
      if (timeout) {
        window.clearTimeout(timeout)
      }
      setIsRunning(false)
    }
  }, [autoRunFor, sigma, updatePositions])

  return (
    <div className="relative">
      <Button
        size="icon"
        onClick={handleClick}
        tooltip={
          isRunning
            ? `${t('graphPanel.sideBar.layoutsControl.stopAnimation')} (${iterationCount})`
            : t('graphPanel.sideBar.layoutsControl.startAnimation')
        }
        variant={controlButtonVariant}
        className={cn(isRunning && 'ring-2 ring-primary ring-offset-1 ring-offset-background')}
      >
        {isRunning ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <PlayIcon />
        )}
      </Button>
      {/* Running indicator badge */}
      {isRunning && (
        <Badge
          variant="default"
          className="absolute -top-2 -right-2 h-5 min-w-5 px-1 text-[10px] font-medium animate-pulse"
        >
          {iterationCount}
        </Badge>
      )}
    </div>
  )
}

/**
 * Component that controls the layout of the graph.
 */
const LayoutsControl = () => {
  const sigma = useSigma()
  const { t } = useTranslation()
  const [layout, setLayout] = useState<LayoutName>('Circular')
  const [opened, setOpened] = useState<boolean>(false)
  const [isLayoutRunning, setIsLayoutRunning] = useState(false)

  const maxIterations = useSettingsStore.use.graphLayoutMaxIterations()

  const layoutCircular = useLayoutCircular()
  const layoutCirclepack = useLayoutCirclepack()
  const layoutRandom = useLayoutRandom()
  const layoutNoverlap = useLayoutNoverlap({
    maxIterations: maxIterations,
    settings: {
      margin: 5,
      expansion: 1.1,
      gridSize: 1,
      ratio: 1,
      speed: 3,
    },
  })
  // Add parameters for Force Directed layout to improve convergence
  const layoutForce = useLayoutForce({
    maxIterations: maxIterations,
    settings: {
      attraction: 0.0003, // Lower attraction force to reduce oscillation
      repulsion: 0.02, // Lower repulsion force to reduce oscillation
      gravity: 0.02, // Increase gravity to make nodes converge to center faster
      inertia: 0.4, // Lower inertia to add damping effect
      maxMove: 100, // Limit maximum movement per step to prevent large jumps
    },
  })
  const layoutForceAtlas2 = useLayoutForceAtlas2({ iterations: maxIterations })
  const workerNoverlap = useWorkerLayoutNoverlap()
  const workerForce = useWorkerLayoutForce()
  const workerForceAtlas2 = useWorkerLayoutForceAtlas2()

  const layouts = useMemo(() => {
    return {
      Circular: {
        layout: layoutCircular,
      },
      Circlepack: {
        layout: layoutCirclepack,
      },
      Random: {
        layout: layoutRandom,
      },
      Noverlaps: {
        layout: layoutNoverlap,
        worker: workerNoverlap,
      },
      'Force Directed': {
        layout: layoutForce,
        worker: workerForce,
      },
      'Force Atlas': {
        layout: layoutForceAtlas2,
        worker: workerForceAtlas2,
      },
    } as { [key: string]: { layout: LayoutHook; worker?: LayoutWorkerHook } }
  }, [
    layoutCirclepack,
    layoutCircular,
    layoutForce,
    layoutForceAtlas2,
    layoutNoverlap,
    layoutRandom,
    workerForce,
    workerNoverlap,
    workerForceAtlas2,
  ])

  const runLayout = useCallback(
    (newLayout: LayoutName) => {
      console.debug('Running layout:', newLayout)
      const { positions } = layouts[newLayout].layout

      try {
        const graph = sigma.getGraph()
        if (!graph) {
          console.error('No graph available')
          return
        }

        const pos = positions()
        console.log('Positions calculated, animating nodes')
        animateNodes(graph, pos, { duration: 400 })
        setLayout(newLayout)
      } catch (error) {
        console.error('Error running layout:', error)
      }
    },
    [layouts, sigma]
  )

  return (
    <div className="relative">
      {/* Layout running indicator badge - shown at top of control panel */}
      {isLayoutRunning && (
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap">
          <Badge
            variant="secondary"
            className="text-[10px] px-2 py-0.5 animate-pulse bg-primary/10 text-primary border-primary/20"
          >
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            {t('graphPanel.sideBar.layoutsControl.running', 'Layout running...')}
          </Badge>
        </div>
      )}
      <div>
        {layouts[layout] && 'worker' in layouts[layout] && layouts[layout].worker && (
          <WorkerLayoutControl
            layout={layouts[layout].worker}
            mainLayout={layouts[layout].layout}
            layoutName={layout}
            onRunningChange={setIsLayoutRunning}
          />
        )}
      </div>
      <div>
        <Popover open={opened} onOpenChange={setOpened}>
          <PopoverTrigger asChild>
            <Button
              size="icon"
              variant={controlButtonVariant}
              onClick={() => setOpened((e: boolean) => !e)}
              tooltip={`${t('graphPanel.sideBar.layoutsControl.layoutGraph')}: ${t(`graphPanel.sideBar.layoutsControl.layouts.${layout}`)}`}
            >
              <GripIcon />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            side="right"
            align="start"
            sideOffset={8}
            collisionPadding={5}
            sticky="always"
            className="p-1 min-w-auto"
          >
            <Command>
              <CommandList>
                <CommandGroup>
                  {Object.keys(layouts).map((name) => (
                    <CommandItem
                      onSelect={() => {
                        runLayout(name as LayoutName)
                      }}
                      key={name}
                      className={cn(
                        'cursor-pointer text-xs',
                        name === layout && 'bg-accent'
                      )}
                    >
                      {t(`graphPanel.sideBar.layoutsControl.layouts.${name}`)}
                      {name === layout && (
                        <span className="ml-auto text-muted-foreground">✓</span>
                      )}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  )
}

export default LayoutsControl
