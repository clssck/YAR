/**
 * Component Demo Page
 * Visual showcase of all foundation UI components for testing and documentation
 */
import { useState } from 'react'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import CollapsibleSection from '@/components/ui/CollapsibleSection'
import EmptyState, {
  EmptyDocuments,
  EmptyGraph,
  EmptySearchResults,
} from '@/components/ui/EmptyState'
import LastUpdated from '@/components/ui/LastUpdated'
import LoadingState, {
  PulsingDot,
  Skeleton,
  SkeletonText,
} from '@/components/ui/LoadingState'
import StatusBadge, { DocumentStatusBadge } from '@/components/ui/StatusBadge'
import { useResponsive } from '@/hooks/useBreakpoint'
import {
  formatShortcut,
  useKeyboardShortcuts,
} from '@/hooks/useKeyboardShortcut'

export default function ComponentDemo() {
  const [collapsibleOpen, setCollapsibleOpen] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState(45)
  const [lastUpdate, setLastUpdate] = useState(Date.now() - 120000) // 2 min ago
  const [isRefreshing, setIsRefreshing] = useState(false)
  const { breakpoint, isMobile, isTablet, isDesktop } = useResponsive()

  // Demo keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 't',
      modifiers: { meta: true },
      callback: () =>
        toast('Keyboard shortcut triggered!', {
          description: '⌘T was pressed',
        }),
      description: 'Test shortcut',
    },
  ])

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await new Promise((r) => setTimeout(r, 1000))
    setLastUpdate(Date.now())
    setIsRefreshing(false)
    toast.success('Refreshed!')
  }

  return (
    <div
      className="container mx-auto p-6 space-y-8 max-w-4xl"
      data-testid="component-demo"
    >
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Component Demo</h1>
        <p className="text-muted-foreground">
          Visual showcase of foundation UI components. Current breakpoint:{' '}
          <code className="bg-muted px-1.5 py-0.5 rounded text-sm">
            {breakpoint}
          </code>
          {isMobile && ' (Mobile)'}
          {isTablet && ' (Tablet)'}
          {isDesktop && ' (Desktop)'}
        </p>
      </header>

      {/* StatusBadge Demo */}
      <Card data-testid="status-badge-demo">
        <CardHeader>
          <CardTitle>StatusBadge</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <StatusBadge label="Success" status="success" />
            <StatusBadge label="Warning" status="warning" />
            <StatusBadge label="Error" status="error" />
            <StatusBadge label="Info" status="info" />
            <StatusBadge label="Pending" status="pending" />
            <StatusBadge label="Processing" status="processing" />
          </div>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Document Status Presets:
            </p>
            <div className="flex flex-wrap gap-2">
              <DocumentStatusBadge.Processed />
              <DocumentStatusBadge.Processing />
              <DocumentStatusBadge.Pending />
              <DocumentStatusBadge.Failed />
              <DocumentStatusBadge.Preprocessed />
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Size Variants:</p>
            <div className="flex items-center gap-2">
              <StatusBadge label="Small" status="info" size="sm" />
              <StatusBadge label="Default" status="info" size="default" />
              <StatusBadge label="Large" status="info" size="lg" />
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Without Icon:</p>
            <StatusBadge label="No Icon" status="success" showIcon={false} />
          </div>
        </CardContent>
      </Card>

      {/* CollapsibleSection Demo */}
      <Card data-testid="collapsible-section-demo">
        <CardHeader>
          <CardTitle>CollapsibleSection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <CollapsibleSection title="Basic Section" defaultOpen={true}>
            <p className="text-sm text-muted-foreground">
              This section is open by default.
            </p>
          </CollapsibleSection>

          <CollapsibleSection title="With Badge Count" badge={5}>
            <p className="text-sm text-muted-foreground">
              This section has a badge showing a count.
            </p>
          </CollapsibleSection>

          <CollapsibleSection title="With String Badge" badge="new">
            <p className="text-sm text-muted-foreground">
              Badge can also be a string.
            </p>
          </CollapsibleSection>

          <CollapsibleSection
            title="Controlled Section"
            open={collapsibleOpen}
            onOpenChange={setCollapsibleOpen}
          >
            <p className="text-sm text-muted-foreground">
              This section is controlled. Current state:{' '}
              {collapsibleOpen ? 'Open' : 'Closed'}
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => setCollapsibleOpen(!collapsibleOpen)}
            >
              Toggle from outside
            </Button>
          </CollapsibleSection>

          <CollapsibleSection title="Starts Closed" defaultOpen={false}>
            <p className="text-sm text-muted-foreground">
              This section starts closed.
            </p>
          </CollapsibleSection>
        </CardContent>
      </Card>

      {/* LoadingState Demo */}
      <Card data-testid="loading-state-demo">
        <CardHeader>
          <CardTitle>LoadingState</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Inline (default):</p>
            <LoadingState message="Loading data..." />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">With Progress:</p>
            <LoadingState
              message="Uploading files..."
              progress={loadingProgress}
            />
            <input
              type="range"
              min="0"
              max="100"
              value={loadingProgress}
              onChange={(e) => setLoadingProgress(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Minimal:</p>
            <LoadingState variant="minimal" message="Saving..." />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Centered:</p>
            <div className="border rounded-lg h-32 relative">
              <LoadingState variant="centered" message="Loading graph..." />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Size Variants:</p>
            <div className="flex items-center gap-4">
              <LoadingState size="sm" message="Small" />
              <LoadingState size="default" message="Default" />
              <LoadingState size="lg" message="Large" />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Skeleton:</p>
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              SkeletonText (3 lines):
            </p>
            <SkeletonText lines={3} />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">PulsingDot:</p>
            <div className="flex items-center gap-2">
              <span>Status</span>
              <PulsingDot />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* LastUpdated Demo */}
      <Card data-testid="last-updated-demo">
        <CardHeader>
          <CardTitle>LastUpdated</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              With Refresh Button:
            </p>
            <LastUpdated
              timestamp={lastUpdate}
              onRefresh={handleRefresh}
              isRefreshing={isRefreshing}
            />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Custom Label:</p>
            <LastUpdated timestamp={Date.now() - 3600000} label="Synced" />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Without Refresh Button:
            </p>
            <LastUpdated
              timestamp={Date.now() - 60000}
              showRefreshButton={false}
            />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Never Updated:</p>
            <LastUpdated timestamp={null} onRefresh={handleRefresh} />
          </div>
        </CardContent>
      </Card>

      {/* EmptyState Demo */}
      <Card data-testid="empty-state-demo">
        <CardHeader>
          <CardTitle>EmptyState</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="border rounded-lg">
            <EmptyState
              title="No items found"
              description="Try adjusting your search or filter to find what you're looking for."
              action={{
                label: 'Clear Filters',
                onClick: () => toast('Filters cleared!'),
              }}
              size="sm"
            />
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Preset: EmptyDocuments
            </p>
            <div className="border rounded-lg">
              <EmptyDocuments onUpload={() => toast('Upload clicked!')} />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Preset: EmptyGraph</p>
            <div className="border rounded-lg">
              <EmptyGraph onLoadData={() => toast('Load graph clicked!')} />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Preset: EmptySearchResults
            </p>
            <div className="border rounded-lg">
              <EmptySearchResults
                query="test query"
                onClear={() => toast('Search cleared!')}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Keyboard Shortcuts Demo */}
      <Card data-testid="keyboard-shortcuts-demo">
        <CardHeader>
          <CardTitle>Keyboard Shortcuts</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Press{' '}
            <code className="bg-muted px-1.5 py-0.5 rounded">
              {formatShortcut('t', { meta: true })}
            </code>{' '}
            to trigger a toast notification.
          </p>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              formatShortcut examples:
            </p>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <code className="bg-muted px-1.5 py-0.5 rounded">
                  {formatShortcut('k', { meta: true })}
                </code>
                <span className="ml-2 text-muted-foreground">Search</span>
              </div>
              <div>
                <code className="bg-muted px-1.5 py-0.5 rounded">
                  {formatShortcut('s', { meta: true, shift: true })}
                </code>
                <span className="ml-2 text-muted-foreground">Save All</span>
              </div>
              <div>
                <code className="bg-muted px-1.5 py-0.5 rounded">
                  {formatShortcut('Escape')}
                </code>
                <span className="ml-2 text-muted-foreground">Close</span>
              </div>
              <div>
                <code className="bg-muted px-1.5 py-0.5 rounded">
                  {formatShortcut('Enter', { ctrl: true })}
                </code>
                <span className="ml-2 text-muted-foreground">Submit</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Responsive Hook Demo */}
      <Card data-testid="responsive-demo">
        <CardHeader>
          <CardTitle>useResponsive Hook</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">breakpoint:</span>
              <code className="ml-2 bg-muted px-1.5 py-0.5 rounded">
                {breakpoint}
              </code>
            </div>
            <div>
              <span className="text-muted-foreground">isMobile:</span>
              <code className="ml-2 bg-muted px-1.5 py-0.5 rounded">
                {String(isMobile)}
              </code>
            </div>
            <div>
              <span className="text-muted-foreground">isTablet:</span>
              <code className="ml-2 bg-muted px-1.5 py-0.5 rounded">
                {String(isTablet)}
              </code>
            </div>
            <div>
              <span className="text-muted-foreground">isDesktop:</span>
              <code className="ml-2 bg-muted px-1.5 py-0.5 rounded">
                {String(isDesktop)}
              </code>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-4">
            Resize the window to see values change.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
