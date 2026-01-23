import { test, expect } from '@playwright/test'

test.describe('Theme Toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('theme toggle button is visible', async ({ page }) => {
    // Theme toggle shows sun or moon icon in header
    const themeButton = page.locator('header button').filter({ has: page.locator('svg') }).last()
    await expect(themeButton).toBeVisible()
  })

  test('clicking theme toggle changes theme class', async ({ page }) => {
    // Get initial theme state from html class
    const html = page.locator('html')
    const initialClass = await html.getAttribute('class')
    const wasLight = initialClass?.includes('light') || !initialClass?.includes('dark')

    // Find and click theme toggle button (last button in header with svg)
    const themeButton = page.locator('header button').filter({ has: page.locator('svg') }).last()
    await themeButton.click()

    // Theme class should change
    if (wasLight) {
      await expect(html).toHaveClass(/dark/)
    } else {
      await expect(html).not.toHaveClass(/dark/)
    }
  })

  test('can toggle theme multiple times', async ({ page }) => {
    const html = page.locator('html')
    const themeButton = page.locator('header button').filter({ has: page.locator('svg') }).last()

    // Get initial state
    const initialClass = await html.getAttribute('class')
    const wasLight = initialClass?.includes('light') || !initialClass?.includes('dark')

    // First toggle
    await themeButton.click()
    if (wasLight) {
      await expect(html).toHaveClass(/dark/)
    } else {
      await expect(html).not.toHaveClass(/dark/)
    }

    // Second toggle - should return to initial
    await themeButton.click()
    if (wasLight) {
      await expect(html).not.toHaveClass(/dark/)
    } else {
      await expect(html).toHaveClass(/dark/)
    }
  })
})

test.describe('Keyboard Shortcuts Help', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('opens keyboard shortcuts dialog', async ({ page }) => {
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    // Should show keyboard shortcuts dialog
    await expect(page.getByRole('dialog')).toBeVisible()
  })

  test('shows available shortcuts', async ({ page }) => {
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    // Should list some keyboard shortcuts
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Should contain keyboard key indicators (kbd elements)
    await expect(dialog.locator('kbd').first()).toBeVisible()
  })

  test('shows category sections', async ({ page }) => {
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Should have category headers (Global, Navigation, etc.)
    await expect(dialog.getByText(/Global|Navigation|Documents|Graph|Chat/i).first()).toBeVisible()
  })

  test('can close with Escape key', async ({ page }) => {
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()
  })

  test('dialog title shows Keyboard Shortcuts', async ({ page }) => {
    const shortcutButton = page.locator('button[title="Keyboard Shortcuts"]')
    await shortcutButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('heading', { name: /Keyboard Shortcuts/i })).toBeVisible()
  })
})
