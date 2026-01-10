import { GitBranchPlus, Link, Scissors, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { PropertyValue } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Text from '@/components/ui/Text'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/Tooltip'
import { useResponsive } from '@/hooks/useBreakpoint'
import useLightragGraph from '@/hooks/useLightragGraph'
import { cn } from '@/lib/utils'
import { type RawEdgeType, type RawNodeType, useGraphStore } from '@/stores/graph'
import EditablePropertyRow from './EditablePropertyRow'

// Type-safe helpers to extract values from PropertyValue union type
const asString = (value: PropertyValue | undefined): string | undefined => {
  if (typeof value === 'string') return value
  if (value == null) return undefined
  return String(value)
}

const asNumber = (value: PropertyValue | undefined): number => {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isNaN(parsed) ? 0 : parsed
  }
  return 0
}

/**
 * Component that view properties of elements in graph.
 */
const PropertiesView = () => {
  const { getNode, getEdge } = useLightragGraph()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()
  // Subscribe to trigger re-render when graph data updates
  const graphDataVersion = useGraphStore.use.graphDataVersion()
  void graphDataVersion

  const { isMobile } = useResponsive()
  const [currentElement, setCurrentElement] = useState<NodeType | EdgeType | null>(null)
  const [currentType, setCurrentType] = useState<'node' | 'edge' | null>(null)

  // Close handler to deselect current element
  const handleClose = () => {
    if (focusedNode || selectedNode) {
      useGraphStore.getState().setSelectedNode(null, false)
      useGraphStore.getState().setFocusedNode(null)
    } else if (focusedEdge || selectedEdge) {
      useGraphStore.getState().setSelectedEdge(null)
      useGraphStore.getState().setFocusedEdge(null)
    }
  }

  // This effect will run when selection changes or when graph data is updated
  useEffect(() => {
    let type: 'node' | 'edge' | null = null
    let element: RawNodeType | RawEdgeType | null = null
    if (focusedNode) {
      type = 'node'
      element = getNode(focusedNode)
    } else if (selectedNode) {
      type = 'node'
      element = getNode(selectedNode)
    } else if (focusedEdge) {
      type = 'edge'
      element = getEdge(focusedEdge, true)
    } else if (selectedEdge) {
      type = 'edge'
      element = getEdge(selectedEdge, true)
    }

    if (element) {
      if (type === 'node') {
        setCurrentElement(refineNodeProperties(element as RawNodeType))
      } else {
        setCurrentElement(refineEdgeProperties(element as RawEdgeType))
      }
      setCurrentType(type)
    } else {
      setCurrentElement(null)
      setCurrentType(null)
    }
  }, [focusedNode, selectedNode, focusedEdge, selectedEdge, getNode, getEdge])

  if (!currentElement) {
    return null
  }

  // Mobile: bottom sheet style, Desktop: side panel
  const containerClasses = cn(
    'bg-background/90 rounded-lg border-2 p-3 text-xs backdrop-blur-lg shadow-lg',
    isMobile
      ? 'fixed bottom-0 left-0 right-0 max-h-[50vh] rounded-b-none border-b-0 animate-in slide-in-from-bottom duration-200'
      : 'max-w-sm'
  )

  return (
    <div className={cn(containerClasses, 'relative')}>
      {/* Close button */}
      <button
        type="button"
        onClick={handleClose}
        className="absolute top-2 right-2 z-10 p-1 rounded-full hover:bg-primary/10 transition-colors"
        aria-label="Close properties panel"
      >
        <X className="h-4 w-4 text-muted-foreground" />
      </button>
      <div className={cn(isMobile && 'overflow-y-auto max-h-[calc(50vh-2rem)]')}>
        {currentType === 'node' ? (
          <NodePropertiesView node={currentElement as NodeType} />
        ) : (
          <EdgePropertiesView edge={currentElement as EdgeType} />
        )}
      </div>
    </div>
  )
}

type NodeType = RawNodeType & {
  relationships: {
    type: string
    id: string
    label: string
  }[]
}

type EdgeType = RawEdgeType & {
  sourceNode?: RawNodeType
  targetNode?: RawNodeType
}

const refineNodeProperties = (node: RawNodeType): NodeType => {
  const state = useGraphStore.getState()
  const relationships = []

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasNode(node.id)) {
        console.warn('Node not found in sigmaGraph:', node.id)
        return {
          ...node,
          relationships: [],
        }
      }

      const edges = state.sigmaGraph.edges(node.id)

      for (const edgeId of edges) {
        if (!state.sigmaGraph.hasEdge(edgeId)) continue

        const edge = state.rawGraph.getEdge(edgeId, true)
        if (edge) {
          const isTarget = node.id === edge.source
          const neighbourId = isTarget ? edge.target : edge.source

          if (!state.sigmaGraph.hasNode(neighbourId)) continue

          const neighbour = state.rawGraph.getNode(neighbourId)
          if (neighbour) {
            relationships.push({
              type: 'Neighbour',
              id: neighbourId,
              label: asString(neighbour.properties.entity_id) || neighbour.labels.join(', '),
            })
          }
        }
      }
    } catch (error) {
      console.error('Error refining node properties:', error)
    }
  }

  return {
    ...node,
    relationships,
  }
}

const refineEdgeProperties = (edge: RawEdgeType): EdgeType => {
  const state = useGraphStore.getState()
  let sourceNode: RawNodeType | undefined
  let targetNode: RawNodeType | undefined

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasEdge(edge.dynamicId)) {
        console.warn('Edge not found in sigmaGraph:', edge.id, 'dynamicId:', edge.dynamicId)
        return {
          ...edge,
          sourceNode: undefined,
          targetNode: undefined,
        }
      }

      if (state.sigmaGraph.hasNode(edge.source)) {
        sourceNode = state.rawGraph.getNode(edge.source)
      }

      if (state.sigmaGraph.hasNode(edge.target)) {
        targetNode = state.rawGraph.getNode(edge.target)
      }
    } catch (error) {
      console.error('Error refining edge properties:', error)
    }
  }

  return {
    ...edge,
    sourceNode,
    targetNode,
  }
}

const PropertyRow = ({
  name,
  value,
  onClick,
  tooltip,
  nodeId,
  edgeId,
  dynamicId,
  entityId,
  entityType,
  sourceId,
  targetId,
  isEditable = false,
  truncate,
}: {
  name: string
  value: PropertyValue
  onClick?: () => void
  tooltip?: string
  nodeId?: string
  entityId?: string
  edgeId?: string
  dynamicId?: string
  entityType?: 'node' | 'edge'
  sourceId?: string
  targetId?: string
  isEditable?: boolean
  truncate?: string
}) => {
  const { t } = useTranslation()

  const getPropertyNameTranslation = (name: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${name}`
    const translation = t(translationKey)
    return translation === translationKey ? name : translation
  }

  // Utility function to convert <SEP> to newlines
  const formatValueWithSeparators = (val: PropertyValue): string => {
    if (typeof val === 'string') {
      return val.replace(/<SEP>/g, ';\n')
    }
    return JSON.stringify(val, null, 2)
  }

  // Format the value to convert <SEP> to newlines
  const formattedValue = formatValueWithSeparators(value)
  let formattedTooltip = tooltip || formatValueWithSeparators(value)

  // If this is source_id field and truncate info exists, append it to the tooltip
  if (name === 'source_id' && truncate) {
    formattedTooltip += `\n(Truncated: ${truncate})`
  }

  // Use EditablePropertyRow for editable fields (description, entity_id and entity_type)
  if (
    isEditable &&
    (name === 'description' ||
      name === 'entity_id' ||
      name === 'entity_type' ||
      name === 'keywords')
  ) {
    return (
      <EditablePropertyRow
        name={name}
        value={value}
        onClick={onClick}
        nodeId={nodeId}
        entityId={entityId}
        edgeId={edgeId}
        dynamicId={dynamicId}
        entityType={entityType}
        sourceId={sourceId}
        targetId={targetId}
        isEditable={true}
        tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
      />
    )
  }

  // For non-editable fields, use the regular Text component
  return (
    <div className="flex items-center gap-2">
      <span className="text-primary/60 tracking-wide whitespace-nowrap">
        {getPropertyNameTranslation(name)}
        {name === 'source_id' && truncate && <sup className="text-red-500">†</sup>}
      </span>
      :
      <Text
        className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis"
        tooltipClassName="max-w-96 -translate-x-13"
        text={formattedValue}
        tooltip={formattedTooltip}
        side="left"
        onClick={onClick}
      />
    </div>
  )
}

const NodePropertiesView = ({ node }: { node: NodeType }) => {
  const { t } = useTranslation()

  const handleExpandNode = () => {
    useGraphStore.getState().triggerNodeExpand(node.id)
  }

  const handlePruneNode = () => {
    useGraphStore.getState().triggerNodePrune(node.id)
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Header with title and action buttons */}
      <div className="flex justify-between items-center pr-6">
        <h3 className="text-sm font-semibold tracking-wide text-blue-600 dark:text-blue-400">
          {t('graphPanel.propertiesView.node.title')}
        </h3>
        <div className="flex gap-2">
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6 border border-gray-300 hover:bg-gray-200 dark:border-gray-600 dark:hover:bg-gray-700"
            onClick={handleExpandNode}
            tooltip={t('graphPanel.propertiesView.node.expandNode')}
          >
            <GitBranchPlus className="h-3.5 w-3.5 text-gray-600 dark:text-gray-300" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6 border border-gray-300 hover:bg-gray-200 dark:border-gray-600 dark:hover:bg-gray-700"
            onClick={handlePruneNode}
            tooltip={t('graphPanel.propertiesView.node.pruneNode')}
          >
            <Scissors className="h-3.5 w-3.5 text-gray-600 dark:text-gray-300" />
          </Button>
        </div>
      </div>

      {/* Node Info Section */}
      <div className="bg-blue-50/50 dark:bg-blue-950/30 rounded-md p-2 border border-blue-100 dark:border-blue-900/50">
        <PropertyRow name={t('graphPanel.propertiesView.node.id')} value={String(node.id)} />
        <PropertyRow
          name={t('graphPanel.propertiesView.node.labels')}
          value={node.labels.join(', ')}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(node.id, true)
          }}
        />
        {/* Degree row - styled consistently with PropertyRow */}
        <div className="flex items-center gap-2">
          <span className="text-primary/60 tracking-wide whitespace-nowrap">
            {t('graphPanel.propertiesView.node.degree')}
          </span>
          :
          <span className="flex items-center gap-2">
            <Text
              className="hover:bg-primary/20 rounded p-1"
              text={String(node.degree)}
              tooltip={t(
                'graphPanel.propertiesView.node.degreeTooltip',
                'Visible connections in this graph'
              )}
              side="left"
            />
            {asNumber(node.properties?.db_degree) > node.degree && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    variant="outline"
                    className="text-xs px-1.5 py-0 text-amber-600 border-amber-400 cursor-help animate-pulse"
                  >
                    +{asNumber(node.properties.db_degree) - node.degree}{' '}
                    {t('graphPanel.propertiesView.node.hidden', 'hidden')}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="left" className="max-w-64">
                  {t(
                    'graphPanel.propertiesView.node.hiddenConnectionsTooltip',
                    'This node has {{count}} additional connections in the database that are not currently visible. Click "Load connections" below to expand.',
                    { count: asNumber(node.properties.db_degree) - node.degree }
                  )}
                </TooltipContent>
              </Tooltip>
            )}
          </span>
        </div>
      </div>

      {/* Load Hidden Connections button for nodes with hidden database connections */}
      {asNumber(node.properties?.db_degree) > node.degree && (
        <Button
          size="sm"
          variant="outline"
          className="w-full text-amber-600 border-amber-400 hover:bg-amber-50 dark:hover:bg-amber-950 font-medium"
          onClick={handleExpandNode}
          tooltip={t(
            'graphPanel.propertiesView.node.loadConnectionsTooltip',
            'Fetch and display {{count}} hidden connections from the database',
            { count: asNumber(node.properties.db_degree) - node.degree }
          )}
        >
          <Link className="h-4 w-4 mr-2" />
          {t('graphPanel.propertiesView.node.loadConnections', {
            count: asNumber(node.properties.db_degree) - node.degree,
          })}
        </Button>
      )}

      {/* Properties Section */}
      <div>
        <h4 className="text-xs font-semibold tracking-wide text-amber-600 dark:text-amber-400 mb-1.5 px-1">
          {t('graphPanel.propertiesView.node.properties')}
        </h4>
        <div className="bg-amber-50/50 dark:bg-amber-950/30 rounded-md p-2 border border-amber-100 dark:border-amber-900/50 max-h-48 overflow-auto">
          {Object.keys(node.properties)
            .sort()
            .map((name) => {
              if (name === 'created_at' || name === 'truncate' || name === 'db_degree') return null // Hide internal properties
              return (
                <PropertyRow
                  key={name}
                  name={name}
                  value={node.properties[name]}
                  nodeId={String(node.id)}
                  entityId={asString(node.properties.entity_id)}
                  entityType="node"
                  isEditable={
                    name === 'description' || name === 'entity_id' || name === 'entity_type'
                  }
                  truncate={asString(node.properties.truncate)}
                />
              )
            })}
        </div>
      </div>

      {/* Relationships Section */}
      {node.relationships.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold tracking-wide text-emerald-600 dark:text-emerald-400 mb-1.5 px-1">
            {t('graphPanel.propertiesView.node.relationships')}
          </h4>
          <div className="bg-emerald-50/50 dark:bg-emerald-950/30 rounded-md p-2 border border-emerald-100 dark:border-emerald-900/50 max-h-32 overflow-auto">
            {node.relationships.map(({ type, id, label }) => {
              return (
                <PropertyRow
                  key={id}
                  name={type}
                  value={label}
                  onClick={() => {
                    useGraphStore.getState().setSelectedNode(id, true)
                  }}
                />
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

const EdgePropertiesView = ({ edge }: { edge: EdgeType }) => {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <h3 className="text-sm font-semibold tracking-wide text-purple-600 dark:text-purple-400 pr-6">
        {t('graphPanel.propertiesView.edge.title')}
      </h3>

      {/* Edge Info Section */}
      <div className="bg-purple-50/50 dark:bg-purple-950/30 rounded-md p-2 border border-purple-100 dark:border-purple-900/50">
        <PropertyRow name={t('graphPanel.propertiesView.edge.id')} value={edge.id} />
        {edge.type && (
          <PropertyRow name={t('graphPanel.propertiesView.edge.type')} value={edge.type} />
        )}
        <PropertyRow
          name={t('graphPanel.propertiesView.edge.source')}
          value={edge.sourceNode ? edge.sourceNode.labels.join(', ') : edge.source}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.source, true)
          }}
        />
        <PropertyRow
          name={t('graphPanel.propertiesView.edge.target')}
          value={edge.targetNode ? edge.targetNode.labels.join(', ') : edge.target}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.target, true)
          }}
        />
      </div>

      {/* Properties Section */}
      <div>
        <h4 className="text-xs font-semibold tracking-wide text-amber-600 dark:text-amber-400 mb-1.5 px-1">
          {t('graphPanel.propertiesView.edge.properties')}
        </h4>
        <div className="bg-amber-50/50 dark:bg-amber-950/30 rounded-md p-2 border border-amber-100 dark:border-amber-900/50 max-h-48 overflow-auto">
          {Object.keys(edge.properties)
            .sort()
            .map((name) => {
              if (name === 'created_at' || name === 'truncate') return null // Hide created_at and truncate properties
              return (
                <PropertyRow
                  key={name}
                  name={name}
                  value={edge.properties[name]}
                  edgeId={String(edge.id)}
                  dynamicId={String(edge.dynamicId)}
                  entityType="edge"
                  sourceId={asString(edge.sourceNode?.properties.entity_id) || edge.source}
                  targetId={asString(edge.targetNode?.properties.entity_id) || edge.target}
                  isEditable={name === 'description' || name === 'keywords'}
                  truncate={asString(edge.properties.truncate)}
                />
              )
            })}
        </div>
      </div>
    </div>
  )
}

export default PropertiesView
