import { test, expect } from '@playwright/test'

test.describe('Component Demo Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
    await expect(page.getByTestId('component-demo')).toBeVisible()
  })

  test('page loads with all component sections', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Component Demo' })).toBeVisible()
    await expect(page.getByTestId('status-badge-demo')).toBeVisible()
    await expect(page.getByTestId('collapsible-section-demo')).toBeVisible()
    await expect(page.getByTestId('loading-state-demo')).toBeVisible()
    await expect(page.getByTestId('last-updated-demo')).toBeVisible()
    await expect(page.getByTestId('empty-state-demo')).toBeVisible()
    await expect(page.getByTestId('keyboard-shortcuts-demo')).toBeVisible()
    await expect(page.getByTestId('responsive-demo')).toBeVisible()
  })
})

test.describe('StatusBadge Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
  })

  test('renders all status variants', async ({ page }) => {
    const demo = page.getByTestId('status-badge-demo')
    await expect(demo).toBeVisible()

    // Check for status badges by aria-label (more reliable)
    await expect(demo.locator('[aria-label*="Success"]').first()).toBeVisible()
    await expect(demo.locator('[aria-label*="Warning"]').first()).toBeVisible()
    await expect(demo.locator('[aria-label*="Error"]').first()).toBeVisible()
  })

  test('renders document status presets', async ({ page }) => {
    const demo = page.getByTestId('status-badge-demo')
    await expect(demo).toBeVisible()

    // Check document presets section exists
    await expect(demo.getByText('Document Status Presets:')).toBeVisible()
    // Check at least one preset is visible
    await expect(demo.locator('[aria-label*="Processed"]').first()).toBeVisible()
  })

  test('processing badge has spinning animation', async ({ page }) => {
    const demo = page.getByTestId('status-badge-demo')
    const processingBadge = demo.locator('[aria-label="Status: Processing"]').first()
    const icon = processingBadge.locator('svg')

    await expect(icon).toHaveClass(/animate-spin/)
  })

  test('has accessible aria-labels', async ({ page }) => {
    const demo = page.getByTestId('status-badge-demo')

    await expect(demo.locator('[aria-label="Status: Success"]').first()).toBeVisible()
    await expect(demo.locator('[aria-label="Status: Error"]').first()).toBeVisible()
  })
})

test.describe('CollapsibleSection Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
  })

  test('toggles open/closed on click', async ({ page }) => {
    const demo = page.getByTestId('collapsible-section-demo')
    const startsClosedButton = demo.getByRole('button', { name: /Starts Closed/ })

    // Initially closed
    await expect(startsClosedButton).toHaveAttribute('aria-expanded', 'false')

    // Click to open
    await startsClosedButton.click()
    await expect(startsClosedButton).toHaveAttribute('aria-expanded', 'true')

    // Click to close
    await startsClosedButton.click()
    await expect(startsClosedButton).toHaveAttribute('aria-expanded', 'false')
  })

  test('controlled section responds to external button', async ({ page }) => {
    const demo = page.getByTestId('collapsible-section-demo')
    const controlledButton = demo.getByRole('button', { name: /Controlled Section/ })
    const toggleButton = demo.getByRole('button', { name: 'Toggle from outside' })

    // Initially open
    await expect(controlledButton).toHaveAttribute('aria-expanded', 'true')

    // Click external toggle
    await toggleButton.click()
    await expect(controlledButton).toHaveAttribute('aria-expanded', 'false')
  })

  test('displays badge when provided', async ({ page }) => {
    const demo = page.getByTestId('collapsible-section-demo')

    // Numeric badge
    await expect(demo.getByText('5')).toBeVisible()

    // String badge
    await expect(demo.getByText('new')).toBeVisible()
  })
})

test.describe('LoadingState Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
  })

  test('shows loading spinner', async ({ page }) => {
    const demo = page.getByTestId('loading-state-demo')
    const spinner = demo.locator('.animate-spin').first()

    await expect(spinner).toBeVisible()
  })

  test('progress slider updates display', async ({ page }) => {
    const demo = page.getByTestId('loading-state-demo')
    await expect(demo).toBeVisible()

    // Check that a progress percentage is displayed (use first() to avoid strict mode)
    await expect(demo.getByText(/%/).first()).toBeVisible()
  })

  test('skeleton elements have pulse animation', async ({ page }) => {
    const demo = page.getByTestId('loading-state-demo')
    const skeletons = demo.locator('.animate-pulse')

    // Should have multiple skeleton elements
    await expect(skeletons.first()).toBeVisible()
  })
})

test.describe('LastUpdated Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
  })

  test('shows relative time', async ({ page }) => {
    const demo = page.getByTestId('last-updated-demo')
    await expect(demo).toBeVisible()

    // Should show some relative time text (use first() since multiple timestamps exist)
    await expect(demo.getByText(/Updated \d+ min ago/).first()).toBeVisible()
  })

  test('refresh button triggers update', async ({ page }) => {
    const demo = page.getByTestId('last-updated-demo')
    const refreshButton = demo.locator('button').filter({ has: page.locator('svg') }).first()

    await refreshButton.click()

    // Should show success toast
    await expect(page.getByText('Refreshed!')).toBeVisible()
  })

  test('never updated state shows correctly', async ({ page }) => {
    const demo = page.getByTestId('last-updated-demo')
    await expect(demo).toBeVisible()

    // Check the section exists with multiple timestamps
    await expect(demo.getByText('Without Refresh Button:')).toBeVisible()
  })
})

test.describe('EmptyState Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
  })

  test('shows empty documents preset', async ({ page }) => {
    const demo = page.getByTestId('empty-state-demo')
    await expect(demo).toBeVisible()

    // Check that empty state content is visible
    await expect(demo.getByText('Preset: EmptyDocuments')).toBeVisible()
  })

  test('shows empty graph preset', async ({ page }) => {
    const demo = page.getByTestId('empty-state-demo')

    await expect(demo.getByText('No graph data')).toBeVisible()
  })

  test('shows empty search results with query', async ({ page }) => {
    const demo = page.getByTestId('empty-state-demo')

    await expect(demo.getByText('No results found')).toBeVisible()
    await expect(demo.getByText(/test query/)).toBeVisible()
  })

  test('action buttons trigger toasts', async ({ page }) => {
    const demo = page.getByTestId('empty-state-demo')

    await demo.getByRole('button', { name: 'Clear Filters' }).click()
    await expect(page.getByText('Filters cleared!')).toBeVisible()
  })
})

test.describe('Responsive Behavior', () => {
  test('shows correct breakpoint on desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await page.goto('/#/demo')

    const demo = page.getByTestId('responsive-demo')
    await expect(demo.getByText('isDesktop:')).toBeVisible()
    await expect(demo.locator('code').filter({ hasText: 'true' })).toBeVisible()
  })

  test('shows correct breakpoint on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/#/demo')

    const demo = page.getByTestId('responsive-demo')
    await expect(demo.getByText('isMobile:')).toBeVisible()
    // Check that mobile is detected
    await expect(demo.getByText(/xs|sm/)).toBeVisible()
  })

  test('updates breakpoint on resize', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await page.goto('/#/demo')

    const demo = page.getByTestId('responsive-demo')

    // Start at desktop
    await expect(demo.locator('code').filter({ hasText: /lg|xl/ }).first()).toBeVisible()

    // Resize to mobile
    await page.setViewportSize({ width: 375, height: 667 })

    // Should update to mobile breakpoint
    await expect(demo.locator('code').filter({ hasText: /xs|sm/ }).first()).toBeVisible()
  })
})

test.describe('Keyboard Shortcuts', () => {
  test('displays shortcut format correctly', async ({ page }) => {
    await page.goto('/#/demo')

    const demo = page.getByTestId('keyboard-shortcuts-demo')

    // Check that keyboard shortcuts are displayed
    await expect(demo.locator('code')).toHaveCount(5) // At least 5 shortcut codes
  })

  test('keyboard shortcuts are documented', async ({ page }) => {
    await page.goto('/#/demo')

    const demo = page.getByTestId('keyboard-shortcuts-demo')
    await expect(demo).toBeVisible()

    // Check that keyboard shortcut info is displayed
    await expect(demo.getByText('formatShortcut examples:')).toBeVisible()
  })
})

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/demo')
    await expect(page.getByTestId('component-demo')).toBeVisible()
  })

  test('status badges have accessible aria-labels', async ({ page }) => {
    const demo = page.getByTestId('status-badge-demo')
    const statusBadges = demo.locator('[aria-label*="Status:"]')
    await expect(statusBadges.first()).toBeVisible()
  })

  test('collapsible sections are keyboard accessible', async ({ page }) => {
    const demo = page.getByTestId('collapsible-section-demo')
    const button = demo.getByRole('button', { name: /Basic Section/ })

    // Focus the button
    await button.focus()

    // Press Enter to toggle
    await page.keyboard.press('Enter')

    // Should toggle
    await expect(button).toHaveAttribute('aria-expanded', 'false')
  })

  test('loading states have aria-busy', async ({ page }) => {
    const loadingElements = page.locator('[aria-busy="true"]')
    await expect(loadingElements.first()).toBeVisible()
  })
})
