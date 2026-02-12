import { expect, test } from '@playwright/test'

test.describe('Theme Persistence', () => {
  test('theme is saved to localStorage', async ({ page }) => {
    await page.goto('/')

    // Get theme element
    const html = page.locator('html')

    // Toggle theme
    const themeButton = page
      .locator('header button')
      .filter({ has: page.locator('svg') })
      .last()
    await themeButton.click()

    // Wait for state to settle
    await page.waitForTimeout(500)

    // Verify localStorage was updated
    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage).not.toBeNull()
    expect(storage.state).toBeDefined()

    // Theme in storage should match current UI state
    const currentClass = await html.getAttribute('class')
    const isDark = currentClass?.includes('dark')

    expect(storage.state.theme).toBe(isDark ? 'dark' : 'light')
  })

  test('theme loads from localStorage on page load', async ({
    page,
    context,
  }) => {
    // First, set theme to dark via localStorage before visiting the page
    await context.addInitScript(() => {
      const existingStorage = localStorage.getItem('settings-storage')
      const data = existingStorage
        ? JSON.parse(existingStorage)
        : { state: {}, version: 26 }
      data.state.theme = 'dark'
      localStorage.setItem('settings-storage', JSON.stringify(data))
    })

    await page.goto('/')

    // Page should load with dark theme
    const html = page.locator('html')
    await expect(html).toHaveClass(/dark/)
  })

  test('theme survives page refresh', async ({ page }) => {
    await page.goto('/')

    // Set to dark mode
    const html = page.locator('html')
    const themeButton = page
      .locator('header button')
      .filter({ has: page.locator('svg') })
      .last()

    // Toggle until we're in dark mode
    const currentClass = await html.getAttribute('class')
    if (!currentClass?.includes('dark')) {
      await themeButton.click()
      await expect(html).toHaveClass(/dark/)
    }

    // Refresh the page
    await page.reload()

    // Should still be dark
    await expect(html).toHaveClass(/dark/)
  })
})

test.describe('Language Persistence', () => {
  test('language setting is stored', async ({ page }) => {
    await page.goto('/')

    // Wait for app to initialize and persist state
    await page.waitForTimeout(1000)

    // Interact with the page to trigger state persistence
    const themeButton = page
      .locator('header button')
      .filter({ has: page.locator('svg') })
      .last()
    await themeButton.click()
    await page.waitForTimeout(500)

    // Check localStorage has language setting
    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage).not.toBeNull()
    expect(storage.state.language).toBeDefined()
    expect(['en', 'zh'].includes(storage.state.language)).toBeTruthy()
  })
})

test.describe('Settings Storage Structure', () => {
  test('settings storage has expected structure', async ({ page }) => {
    await page.goto('/')

    // Wait for app to initialize
    await page.waitForTimeout(1000)

    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage).not.toBeNull()
    expect(storage.version).toBeGreaterThanOrEqual(1)
    expect(storage.state).toBeDefined()

    // Check for expected state properties
    const state = storage.state
    expect(state).toHaveProperty('theme')
    expect(state).toHaveProperty('language')
  })

  test('query settings are persisted', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()

    // Change mode
    const modeSelect = page.locator('#query_mode_select')
    await modeSelect.click()
    await page.getByRole('option', { name: /Local/i }).click()

    // Wait for persistence
    await page.waitForTimeout(500)

    // Check localStorage
    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage.state.querySettings).toBeDefined()
    expect(storage.state.querySettings.mode).toBe('local')
  })
})
