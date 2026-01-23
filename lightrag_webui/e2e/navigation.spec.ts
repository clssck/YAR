import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('renders main navigation tabs', async ({ page }) => {
    // Check that main navigation tabs are visible
    await expect(page.getByRole('tab', { name: /Documents/i })).toBeVisible()
    await expect(page.getByRole('tab', { name: /Knowledge Graph/i })).toBeVisible()
    await expect(page.getByRole('tab', { name: /Retrieval/i })).toBeVisible()
    await expect(page.getByRole('tab', { name: /API/i })).toBeVisible()
  })

  test('switches tabs on click', async ({ page }) => {
    // Start on documents tab (default)
    const documentsTab = page.getByRole('tab', { name: /Documents/i })
    const retrievalTab = page.getByRole('tab', { name: /Retrieval/i })
    const graphTab = page.getByRole('tab', { name: /Knowledge Graph/i })

    // Click retrieval tab
    await retrievalTab.click()
    await expect(retrievalTab).toHaveAttribute('data-state', 'active')

    // Click graph tab
    await graphTab.click()
    await expect(graphTab).toHaveAttribute('data-state', 'active')

    // Click documents tab
    await documentsTab.click()
    await expect(documentsTab).toHaveAttribute('data-state', 'active')
  })

  test('header has logo/brand', async ({ page }) => {
    // Check for brain icon (logo)
    const header = page.locator('header')
    await expect(header).toBeVisible()
    await expect(header.locator('svg').first()).toBeVisible()
  })

  test('theme toggle button is accessible', async ({ page }) => {
    // Theme toggle button (shows sun/moon icon with Light/Dark tooltip) should be in the header
    const themeButton = page.locator('header button').filter({ has: page.locator('svg.lucide-sun, svg.lucide-moon') })
    await expect(themeButton).toBeVisible()
  })

  test('keyboard shortcut help is accessible', async ({ page }) => {
    // Keyboard shortcut button should be visible (has title="Keyboard Shortcuts")
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await expect(shortcutButton).toBeVisible()
  })
})

test.describe('URL Routing', () => {
  test('navigates to documents tab by default', async ({ page }) => {
    await page.goto('/')
    const documentsTab = page.getByRole('tab', { name: /Documents/i })
    await expect(documentsTab).toHaveAttribute('data-state', 'active')
  })

  test('demo page is accessible', async ({ page }) => {
    await page.goto('/#/demo')
    await expect(page.getByTestId('component-demo')).toBeVisible()
  })
})
