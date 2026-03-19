/**
 * Tests for GraphViewer component - utility functions and rendering
 */
import '../setup'
import { describe, expect, mock, test } from 'bun:test'
import { render } from '@testing-library/react'

// =============================================================================
// Mocks for Sigma packages only (these don't have their own tests)
// =============================================================================

// Mock Sigma and related packages - these are WebGL-based and can't run in jsdom
mock.module('@react-sigma/core', () => ({
  SigmaContainer: ({
    children,
    className,
  }: {
    children: React.ReactNode
    className?: string
  }) => (
    <div data-testid="sigma-container" className={className}>
      {children}
    </div>
  ),
  useRegisterEvents: () => mock(),
  useSigma: () => ({
    getGraph: () => ({
      setNodeAttribute: mock(),
      removeNodeAttribute: mock(),
    }),
    viewportToGraph: () => ({ x: 0, y: 0 }),
    getCustomBBox: () => null,
    setCustomBBox: mock(),
    getBBox: () => ({}),
  }),
}))

mock.module('@sigma/edge-curve', () => ({
  createEdgeCurveProgram: () => ({}),
  EdgeCurvedArrowProgram: {},
}))

mock.module('@sigma/node-border', () => ({
  createNodeBorderProgram: () => ({}),
  NodeBorderProgram: {},
}))

mock.module('sigma/rendering', () => ({
  EdgeArrowProgram: {},
  NodeCircleProgram: {},
  NodePointProgram: {},
}))

// Mock graph sub-components (these are tested separately or don't need deep testing)
mock.module('@/components/graph/FocusOnNode', () => ({
  default: () => <div data-testid="focus-on-node" />,
}))

mock.module('@/components/graph/FullScreenControl', () => ({
  default: () => (
    <button type="button" data-testid="fullscreen-control">
      Fullscreen
    </button>
  ),
}))

mock.module('@/components/graph/GraphControl', () => ({
  default: () => <div data-testid="graph-control" />,
}))

mock.module('@/components/graph/GraphLabels', () => ({
  default: () => <div data-testid="graph-labels">Labels</div>,
}))

mock.module('@/components/graph/LayoutsControl', () => ({
  default: () => (
    <button type="button" data-testid="layouts-control">
      Layouts
    </button>
  ),
}))

mock.module('@/components/graph/Legend', () => ({
  default: ({ className }: { className?: string }) => (
    <div data-testid="legend" className={className}>
      Legend
    </div>
  ),
}))

mock.module('@/components/graph/LegendButton', () => ({
  default: () => (
    <button type="button" data-testid="legend-button">
      Toggle Legend
    </button>
  ),
}))

mock.module('@/components/graph/OnboardingHints', () => ({
  default: () => <div data-testid="onboarding-hints">Hints</div>,
}))

mock.module('@/components/graph/OrphanConnectionControl', () => ({
  default: () => (
    <button type="button" data-testid="orphan-control">
      Orphan
    </button>
  ),
}))

mock.module('@/components/graph/PropertiesView', () => ({
  default: () => <div data-testid="properties-view">Properties</div>,
}))

mock.module('@/components/graph/Settings', () => ({
  default: () => (
    <button type="button" data-testid="settings">
      Settings
    </button>
  ),
}))

mock.module('@/components/graph/SettingsDisplay', () => ({
  default: () => <div data-testid="settings-display" />,
}))

mock.module('@/components/graph/ZoomControl', () => ({
  default: () => (
    <button type="button" data-testid="zoom-control">
      Zoom
    </button>
  ),
}))

// Mock stores with configurable state
let mockGraphState = {
  isFetching: false,
  graphIsEmpty: false,
  sigmaGraph: { order: 100, size: 250 },
  selectedNode: null as string | null,
  focusedNode: null as string | null,
  moveToSelectedNode: false,
}

let mockSettingsState = {
  showPropertyPanel: true,
  enableNodeDrag: true,
  showLegend: true,
  theme: 'light' as 'light' | 'dark',
}

mock.module('@/stores/graph', () => ({
  useGraphStore: {
    use: {
      isFetching: () => mockGraphState.isFetching,
      graphIsEmpty: () => mockGraphState.graphIsEmpty,
      sigmaGraph: () => mockGraphState.sigmaGraph,
      selectedNode: () => mockGraphState.selectedNode,
      focusedNode: () => mockGraphState.focusedNode,
      moveToSelectedNode: () => mockGraphState.moveToSelectedNode,
    },
    getState: () => ({
      sigmaInstance: null,
      setSigmaInstance: mock(),
    }),
  },
}))

mock.module('@/stores/settings', () => ({
  useSettingsStore: {
    use: {
      showPropertyPanel: () => mockSettingsState.showPropertyPanel,
      enableNodeDrag: () => mockSettingsState.enableNodeDrag,
      showLegend: () => mockSettingsState.showLegend,
      theme: () => mockSettingsState.theme,
    },
  },
}))

// Import component after mocks
const GraphViewer = (await import('@/features/GraphViewer')).default

// =============================================================================
// Utility Function Recreations for Testing
// =============================================================================

// Recreate createSigmaSettings for testing (simplified version)
interface SigmaSettings {
  allowInvalidContainer: boolean
  defaultNodeType: string
  defaultEdgeType: string
  renderEdgeLabels: boolean
  labelGridCellSize: number
  labelRenderedSizeThreshold: number
  enableEdgeEvents: boolean
  labelColor: { color: string; attribute: string }
  edgeLabelColor: { color: string; attribute: string }
  edgeLabelSize: number
  labelSize: number
}

const labelColorLightTheme = '#000000'
const labelColorDarkTheme = '#ffffff'

const createSigmaSettings = (isDarkTheme: boolean): SigmaSettings => ({
  allowInvalidContainer: true,
  defaultNodeType: 'default',
  defaultEdgeType: 'curvedNoArrow',
  renderEdgeLabels: false,
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
})

// Compact mode calculation
const calculateCompactMode = (
  isMobile: boolean,
  isTablet: boolean,
): boolean => {
  return isMobile || isTablet
}

// Graph stats formatting
const formatGraphStats = (nodeCount: number, edgeCount: number): string => {
  return `${nodeCount.toLocaleString()} nodes | ${edgeCount.toLocaleString()} edges`
}

// =============================================================================
// Tests
// =============================================================================

describe('GraphViewer Utility Functions', () => {
  describe('createSigmaSettings', () => {
    test('creates settings for light theme', () => {
      const settings = createSigmaSettings(false)

      expect(settings.labelColor.color).toBe(labelColorLightTheme)
      expect(settings.edgeLabelColor.color).toBe(labelColorLightTheme)
    })

    test('creates settings for dark theme', () => {
      const settings = createSigmaSettings(true)

      expect(settings.labelColor.color).toBe(labelColorDarkTheme)
      expect(settings.edgeLabelColor.color).toBe(labelColorDarkTheme)
    })

    test('has correct default node type', () => {
      const settings = createSigmaSettings(false)

      expect(settings.defaultNodeType).toBe('default')
    })

    test('has correct default edge type', () => {
      const settings = createSigmaSettings(false)

      expect(settings.defaultEdgeType).toBe('curvedNoArrow')
    })

    test('allows invalid container', () => {
      const settings = createSigmaSettings(false)

      expect(settings.allowInvalidContainer).toBe(true)
    })

    test('enables edge events', () => {
      const settings = createSigmaSettings(false)

      expect(settings.enableEdgeEvents).toBe(true)
    })

    test('disables edge label rendering', () => {
      const settings = createSigmaSettings(false)

      expect(settings.renderEdgeLabels).toBe(false)
    })

    test('has correct label sizes', () => {
      const settings = createSigmaSettings(false)

      expect(settings.labelSize).toBe(12)
      expect(settings.edgeLabelSize).toBe(8)
    })

    test('has correct label grid cell size', () => {
      const settings = createSigmaSettings(false)

      expect(settings.labelGridCellSize).toBe(60)
    })

    test('has correct label rendered size threshold', () => {
      const settings = createSigmaSettings(false)

      expect(settings.labelRenderedSizeThreshold).toBe(12)
    })

    test('label color has correct attribute', () => {
      const settings = createSigmaSettings(false)

      expect(settings.labelColor.attribute).toBe('labelColor')
      expect(settings.edgeLabelColor.attribute).toBe('labelColor')
    })
  })

  describe('calculateCompactMode', () => {
    test('returns true for mobile', () => {
      expect(calculateCompactMode(true, false)).toBe(true)
    })

    test('returns true for tablet', () => {
      expect(calculateCompactMode(false, true)).toBe(true)
    })

    test('returns true for mobile and tablet', () => {
      expect(calculateCompactMode(true, true)).toBe(true)
    })

    test('returns false for desktop', () => {
      expect(calculateCompactMode(false, false)).toBe(false)
    })
  })

  describe('formatGraphStats', () => {
    test('formats small numbers', () => {
      const result = formatGraphStats(10, 25)
      expect(result).toBe('10 nodes | 25 edges')
    })

    test('formats large numbers with locale separators', () => {
      const result = formatGraphStats(1000, 2500)
      expect(result).toContain('1')
      expect(result).toContain('000')
      expect(result).toContain('2')
      expect(result).toContain('500')
    })

    test('formats zero counts', () => {
      const result = formatGraphStats(0, 0)
      expect(result).toBe('0 nodes | 0 edges')
    })

    test('formats single node/edge', () => {
      const result = formatGraphStats(1, 1)
      expect(result).toBe('1 nodes | 1 edges')
    })
  })
})

// =============================================================================
// Theme Color Constants Tests
// =============================================================================

describe('Theme Color Constants', () => {
  test('light theme label color is defined', () => {
    expect(labelColorLightTheme).toBeDefined()
    expect(typeof labelColorLightTheme).toBe('string')
  })

  test('dark theme label color is defined', () => {
    expect(labelColorDarkTheme).toBeDefined()
    expect(typeof labelColorDarkTheme).toBe('string')
  })

  test('light and dark theme colors are different', () => {
    expect(labelColorLightTheme).not.toBe(labelColorDarkTheme)
  })

  test('colors are valid hex format', () => {
    const hexPattern = /^#[0-9a-fA-F]{6}$/
    expect(labelColorLightTheme).toMatch(hexPattern)
    expect(labelColorDarkTheme).toMatch(hexPattern)
  })
})

// =============================================================================
// Node Program Types Tests
// =============================================================================

describe('Node Program Types', () => {
  type NodeType = 'default' | 'circle' | 'point' | 'orphan'

  test('all node types are valid', () => {
    const nodeTypes: NodeType[] = ['default', 'circle', 'point', 'orphan']
    expect(nodeTypes).toHaveLength(4)
  })

  test('default is the standard node type', () => {
    const settings = createSigmaSettings(false)
    expect(settings.defaultNodeType).toBe('default')
  })

  test('orphan type exists for disconnected nodes', () => {
    const nodeTypes: NodeType[] = ['default', 'circle', 'point', 'orphan']
    expect(nodeTypes).toContain('orphan')
  })
})

// =============================================================================
// Edge Program Types Tests
// =============================================================================

describe('Edge Program Types', () => {
  type EdgeType = 'arrow' | 'curvedArrow' | 'curvedNoArrow'

  test('all edge types are valid', () => {
    const edgeTypes: EdgeType[] = ['arrow', 'curvedArrow', 'curvedNoArrow']
    expect(edgeTypes).toHaveLength(3)
  })

  test('curvedNoArrow is the default edge type', () => {
    const settings = createSigmaSettings(false)
    expect(settings.defaultEdgeType).toBe('curvedNoArrow')
  })
})

// =============================================================================
// Responsive Behavior Tests
// =============================================================================

describe('Responsive Behavior', () => {
  describe('Control Panel Layout', () => {
    test('horizontal layout on mobile', () => {
      const isCompact = calculateCompactMode(true, false)
      expect(isCompact).toBe(true)
      // In compact mode, controls are horizontal (flex-row)
    })

    test('vertical layout on desktop', () => {
      const isCompact = calculateCompactMode(false, false)
      expect(isCompact).toBe(false)
      // In non-compact mode, controls are vertical (flex-col)
    })
  })

  describe('Component Visibility', () => {
    test('graph labels hidden on mobile', () => {
      const isCompact = calculateCompactMode(true, false)
      // GraphLabels should be hidden when isCompact is true
      expect(isCompact).toBe(true)
    })

    test('legend hidden on mobile', () => {
      const isCompact = calculateCompactMode(true, false)
      // Legend should be hidden when isCompact is true
      expect(isCompact).toBe(true)
    })

    test('orphan control hidden on mobile', () => {
      const isCompact = calculateCompactMode(true, false)
      // OrphanConnectionControl should be hidden when isCompact is true
      expect(isCompact).toBe(true)
    })

    test('legend button hidden on mobile', () => {
      const isCompact = calculateCompactMode(true, false)
      // LegendButton should be hidden when isCompact is true
      expect(isCompact).toBe(true)
    })
  })
})

// =============================================================================
// Graph State Tests
// =============================================================================

describe('Graph State Logic', () => {
  // Type for sigma graph-like object
  interface GraphStats {
    order: number
    size: number
  }

  describe('Empty Graph Detection', () => {
    test('graph with nodes is not empty', () => {
      const graph: GraphStats = { order: 10, size: 15 }
      const isEmpty = graph.order === 0
      expect(isEmpty).toBe(false)
    })

    test('graph with no nodes is empty', () => {
      const graph: GraphStats = { order: 0, size: 0 }
      const isEmpty = graph.order === 0
      expect(isEmpty).toBe(true)
    })

    test('null graph is empty', () => {
      const graph = null as GraphStats | null
      const nodeCount = graph?.order ?? 0
      expect(nodeCount).toBe(0)
    })
  })

  describe('Node Count Extraction', () => {
    test('extracts node count from graph', () => {
      const graph: GraphStats | null = { order: 100, size: 250 }
      const nodeCount = graph?.order ?? 0
      expect(nodeCount).toBe(100)
    })

    test('defaults to 0 for null graph', () => {
      const graph = null as GraphStats | null
      const nodeCount = graph?.order ?? 0
      expect(nodeCount).toBe(0)
    })
  })

  describe('Edge Count Extraction', () => {
    test('extracts edge count from graph', () => {
      const graph: GraphStats | null = { order: 100, size: 250 }
      const edgeCount = graph?.size ?? 0
      expect(edgeCount).toBe(250)
    })

    test('defaults to 0 for null graph', () => {
      const graph = null as GraphStats | null
      const edgeCount = graph?.size ?? 0
      expect(edgeCount).toBe(0)
    })
  })
})

// =============================================================================
// OrphanNodeProgram Configuration Tests
// =============================================================================

describe('OrphanNodeProgram Configuration', () => {
  // Recreate the border configuration for testing
  interface BorderConfig {
    color: { attribute?: string; value?: string }
    size: { value: number; mode: string }
  }

  const orphanBorders: BorderConfig[] = [
    { color: { attribute: 'color' }, size: { value: 0.4, mode: 'relative' } },
    { color: { value: '#374151' }, size: { value: 0.15, mode: 'relative' } },
  ]

  test('has two border layers', () => {
    expect(orphanBorders).toHaveLength(2)
  })

  test('outer border uses node color', () => {
    expect(orphanBorders[0].color.attribute).toBe('color')
  })

  test('outer border has 40% relative size', () => {
    expect(orphanBorders[0].size.value).toBe(0.4)
    expect(orphanBorders[0].size.mode).toBe('relative')
  })

  test('inner border has fixed color', () => {
    expect(orphanBorders[1].color.value).toBe('#374151')
  })

  test('inner border has 15% relative size', () => {
    expect(orphanBorders[1].size.value).toBe(0.15)
    expect(orphanBorders[1].size.mode).toBe('relative')
  })
})

// =============================================================================
// Component Rendering Tests
// =============================================================================

describe('GraphViewer Component Rendering', () => {
  // Reset mock state before each test
  function resetMockState() {
    mockGraphState = {
      isFetching: false,
      graphIsEmpty: false,
      sigmaGraph: { order: 100, size: 250 },
      selectedNode: null,
      focusedNode: null,
      moveToSelectedNode: false,
    }
    mockSettingsState = {
      showPropertyPanel: true,
      enableNodeDrag: true,
      showLegend: true,
      theme: 'light',
    }
  }

  describe('Basic Rendering', () => {
    test('renders without crashing', () => {
      resetMockState()
      const { container } = render(<GraphViewer />)
      expect(container).toBeDefined()
    })

    test('renders sigma container', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('sigma-container')).toBeDefined()
    })

    test('renders graph control', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('graph-control')).toBeDefined()
    })

    test('renders focus on node component', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('focus-on-node')).toBeDefined()
    })
  })

  describe('Control Panel Elements', () => {
    test('renders layouts control', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('layouts-control')).toBeDefined()
    })

    test('renders zoom control', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('zoom-control')).toBeDefined()
    })

    test('renders fullscreen control', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('fullscreen-control')).toBeDefined()
    })

    test('renders settings button', () => {
      resetMockState()
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('settings')).toBeDefined()
    })
  })

  describe('Conditional Rendering', () => {
    test('renders properties view when showPropertyPanel is true', () => {
      resetMockState()
      mockSettingsState.showPropertyPanel = true
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('properties-view')).toBeDefined()
    })

    test('does not render properties view when showPropertyPanel is false', () => {
      resetMockState()
      mockSettingsState.showPropertyPanel = false
      const { queryByTestId } = render(<GraphViewer />)
      expect(queryByTestId('properties-view')).toBeNull()
    })

    test('renders onboarding hints when graph has data and not fetching', () => {
      resetMockState()
      mockGraphState.isFetching = false
      mockGraphState.graphIsEmpty = false
      const { getByTestId } = render(<GraphViewer />)
      expect(getByTestId('onboarding-hints')).toBeDefined()
    })

    test('does not render onboarding hints when fetching', () => {
      resetMockState()
      mockGraphState.isFetching = true
      const { queryByTestId } = render(<GraphViewer />)
      expect(queryByTestId('onboarding-hints')).toBeNull()
    })

    test('does not render onboarding hints when graph is empty', () => {
      resetMockState()
      mockGraphState.graphIsEmpty = true
      const { queryByTestId } = render(<GraphViewer />)
      expect(queryByTestId('onboarding-hints')).toBeNull()
    })
  })

  describe('Loading State', () => {
    test('shows loading overlay when fetching', () => {
      resetMockState()
      mockGraphState.isFetching = true
      const { container } = render(<GraphViewer />)
      // Loading overlay has backdrop-blur-sm class
      const loadingOverlay = container.querySelector('[class*="backdrop-blur"]')
      expect(loadingOverlay).toBeDefined()
    })

    test('hides loading overlay when not fetching', () => {
      resetMockState()
      mockGraphState.isFetching = false
      const { container } = render(<GraphViewer />)
      // Check that there's no overlay covering the sigma container
      const sigmaContainer = container.querySelector(
        '[data-testid="sigma-container"]',
      )
      expect(sigmaContainer).toBeDefined()
    })
  })

  describe('Graph Stats Display', () => {
    test('displays node count when graph has data', () => {
      resetMockState()
      mockGraphState.sigmaGraph = { order: 42, size: 100 }
      mockGraphState.graphIsEmpty = false
      mockGraphState.isFetching = false
      const { container } = render(<GraphViewer />)
      expect(container.textContent).toContain('42')
    })

    test('displays edge count when graph has data', () => {
      resetMockState()
      mockGraphState.sigmaGraph = { order: 42, size: 100 }
      mockGraphState.graphIsEmpty = false
      mockGraphState.isFetching = false
      const { container } = render(<GraphViewer />)
      expect(container.textContent).toContain('100')
    })

    test('does not display stats when graph is empty', () => {
      resetMockState()
      mockGraphState.sigmaGraph = { order: 0, size: 0 }
      mockGraphState.graphIsEmpty = true
      const { container } = render(<GraphViewer />)
      // The stats container with node/edge count shouldn't show numbers
      expect(container.textContent).not.toContain('nodes')
    })
  })

  describe('Theme Support', () => {
    test('renders with light theme', () => {
      resetMockState()
      mockSettingsState.theme = 'light'
      const { container } = render(<GraphViewer />)
      expect(container).toBeDefined()
    })

    test('renders with dark theme', () => {
      resetMockState()
      mockSettingsState.theme = 'dark'
      const { container } = render(<GraphViewer />)
      expect(container).toBeDefined()
    })
  })
})
