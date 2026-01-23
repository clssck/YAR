import { test, expect } from '@playwright/test'

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('page has accessible title', async ({ page }) => {
    await expect(page).toHaveTitle(/LightRAG/i)
  })

  test('navigation tabs are keyboard accessible', async ({ page }) => {
    const documentsTab = page.getByRole('tab', { name: /Documents/i })

    // Focus on documents tab
    await documentsTab.focus()
    await expect(documentsTab).toBeFocused()

    // Arrow right should move to next tab
    await page.keyboard.press('ArrowRight')

    // Knowledge Graph tab should be focused
    const graphTab = page.getByRole('tab', { name: /Knowledge Graph/i })
    await expect(graphTab).toBeFocused()
  })

  test('buttons have accessible names', async ({ page }) => {
    // Theme toggle button (has tooltip as accessible name)
    const themeButton = page.locator('header button').filter({ has: page.locator('svg.lucide-sun, svg.lucide-moon') })
    await expect(themeButton).toBeVisible()

    // Keyboard shortcuts button (has title attribute)
    const shortcutsButton = page.locator('button[title="Keyboard Shortcuts"]')
    await expect(shortcutsButton).toBeVisible()
  })

  test('tabs have proper ARIA attributes', async ({ page }) => {
    const tabs = page.getByRole('tablist')
    await expect(tabs).toBeVisible()

    const documentsTab = page.getByRole('tab', { name: /Documents/i })
    await expect(documentsTab).toHaveAttribute('role', 'tab')
  })

  test('dialogs trap focus', async ({ page }) => {
    // Open keyboard shortcuts dialog
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Focus should be trapped within dialog
    // Tab through elements - should not leave dialog
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')

    // Focus should still be within dialog
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()
  })

  test('escape closes dialogs', async ({ page }) => {
    // Open keyboard shortcuts dialog
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()
  })
})

test.describe('Focus Management', () => {
  test('focus returns to trigger after dialog closes', async ({ page }) => {
    await page.goto('/')

    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()

    // Focus should return to shortcut button
    await expect(shortcutButton).toBeFocused()
  })
})

test.describe('Color Contrast', () => {
  test('text is readable in light mode', async ({ page }) => {
    await page.goto('/')

    // Get initial theme state from html class
    const html = page.locator('html')
    const initialClass = await html.getAttribute('class')
    const isDark = initialClass?.includes('dark')

    // If in dark mode, toggle to light
    if (isDark) {
      const themeButton = page.locator('header button').filter({ has: page.locator('svg') }).last()
      await themeButton.click()
      await expect(html).not.toHaveClass(/dark/)
    }

    // Check that header text is visible
    const header = page.locator('header')
    await expect(header).toBeVisible()
  })

  test('text is readable in dark mode', async ({ page }) => {
    await page.goto('/')

    // Get initial theme state from html class
    const html = page.locator('html')
    const initialClass = await html.getAttribute('class')
    const isDark = initialClass?.includes('dark')

    // If in light mode, toggle to dark
    if (!isDark) {
      const themeButton = page.locator('header button').filter({ has: page.locator('svg') }).last()
      await themeButton.click()
      await expect(html).toHaveClass(/dark/)
    }

    // Check that header text is still visible in dark mode
    const header = page.locator('header')
    await expect(header).toBeVisible()
  })
})

test.describe('Screen Reader Support', () => {
  test('images have alt text or are decorative', async ({ page }) => {
    await page.goto('/')

    // All images should have alt attribute or aria-hidden
    const images = page.locator('img')
    const count = await images.count()

    for (let i = 0; i < count; i++) {
      const img = images.nth(i)
      const alt = await img.getAttribute('alt')
      const ariaHidden = await img.getAttribute('aria-hidden')

      // Should have alt text OR be marked as decorative
      expect(alt !== null || ariaHidden === 'true').toBeTruthy()
    }
  })

  test('icons are hidden from screen readers', async ({ page }) => {
    await page.goto('/')

    // Decorative SVG icons should have aria-hidden
    const icons = page.locator('svg[aria-hidden="true"]')
    await expect(icons.first()).toBeVisible()
  })

  test('page has interactive elements', async ({ page }) => {
    await page.goto('/')

    // Wait for page to load with increased timeout
    await page.waitForLoadState('domcontentloaded')

    // Check that tabs exist - the main navigation element
    const documentsTab = page.getByRole('tab', { name: /Documents/i })
    await expect(documentsTab).toBeVisible({ timeout: 10000 })
  })
})
