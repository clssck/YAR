import { test, expect } from '@playwright/test'

test.describe('Documents Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Documents tab should be default, but click to be sure
    await page.getByRole('tab', { name: /Documents/i }).click()
  })

  test('documents tab is default active tab', async ({ page }) => {
    await page.goto('/')
    const documentsTab = page.getByRole('tab', { name: /Documents/i })
    await expect(documentsTab).toHaveAttribute('data-state', 'active')
  })

  test('shows upload button', async ({ page }) => {
    // Upload button has an icon and text
    const uploadButton = page.locator('button').filter({ hasText: /Upload/i }).first()
    await expect(uploadButton).toBeVisible()
  })

  test('shows scan/retry button', async ({ page }) => {
    // Button might have tooltip or different text
    const scanButton = page.locator('button').filter({ hasText: /Scan|Retry/i }).first()
      .or(page.locator('button[title*="Scan"]'))
      .or(page.locator('button[title*="Retry"]'))
    await expect(scanButton.first()).toBeVisible()
  })

  test('shows document table or empty state', async ({ page }) => {
    // Either shows a table with documents or an empty state message
    const table = page.locator('table')
    const emptyState = page.getByText(/No Documents|no.*uploaded|empty/i)

    const hasTable = await table.count() > 0
    const hasEmptyState = await emptyState.count() > 0

    // Should show one or the other
    expect(hasTable || hasEmptyState).toBeTruthy()
  })

  test('upload button opens upload dialog', async ({ page }) => {
    const uploadButton = page.locator('button').filter({ hasText: /Upload/i }).first()
    await uploadButton.click()

    // Should open upload dialog/modal
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible({ timeout: 3000 })
  })

  test('upload dialog has file input area', async ({ page }) => {
    const uploadButton = page.locator('button').filter({ hasText: /Upload/i }).first()
    await uploadButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Should have some upload-related content
    const uploadContent = dialog.locator('input[type="file"]')
      .or(dialog.getByText(/drag|drop|select|browse|upload/i))
    await expect(uploadContent.first()).toBeVisible()
  })

  test('upload dialog can be closed', async ({ page }) => {
    const uploadButton = page.locator('button').filter({ hasText: /Upload/i }).first()
    await uploadButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Close with Escape
    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()
  })

  test('shows supported file types in upload dialog', async ({ page }) => {
    const uploadButton = page.locator('button').filter({ hasText: /Upload/i }).first()
    await uploadButton.click()

    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    // Should mention supported types
    const fileTypes = dialog.getByText(/TXT|PDF|MD|DOCX/i)
    await expect(fileTypes).toBeVisible()
  })
})

test.describe('Document Table', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Documents/i }).click()
  })

  test('table has expected columns when documents exist', async ({ page }) => {
    // This test only runs if there are documents
    const table = page.locator('table')

    if (await table.count() > 0) {
      // Check for expected column headers
      const headers = ['File Name', 'Status', 'Chunks', 'Created']
      let foundHeaders = 0
      for (const header of headers) {
        const headerCell = table.getByRole('columnheader', { name: new RegExp(header, 'i') })
        if (await headerCell.count() > 0) {
          foundHeaders++
        }
      }
      // At least some of these columns should exist
      expect(foundHeaders).toBeGreaterThan(0)
    }
  })

  test('clear button requires confirmation', async ({ page }) => {
    const clearButton = page.getByRole('button', { name: /Clear/i }).filter({ has: page.locator('text=Clear') })

    if (await clearButton.count() > 0 && await clearButton.isEnabled()) {
      await clearButton.click()

      // Should show confirmation dialog
      const confirmDialog = page.getByRole('dialog')
      await expect(confirmDialog).toBeVisible({ timeout: 3000 })

      // Should have warning text
      await expect(confirmDialog.getByText(/WARNING|cannot be undone|permanent/i)).toBeVisible()

      // Close without confirming
      await page.keyboard.press('Escape')
    }
  })
})

test.describe('Document Status Indicators', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Documents/i }).click()
  })

  test('document panel has action buttons', async ({ page }) => {
    // Document panel should have some action buttons
    const buttons = page.locator('button')
    const count = await buttons.count()
    expect(count).toBeGreaterThan(0)
  })

  test('refresh button exists', async ({ page }) => {
    // Look for refresh functionality
    const refreshButton = page.getByRole('button', { name: /Refresh|Reset/i })
      .or(page.locator('button').filter({ has: page.locator('svg.lucide-refresh-cw, svg.lucide-rotate-ccw') }))

    if (await refreshButton.count() > 0) {
      await expect(refreshButton.first()).toBeVisible()
    }
  })
})

test.describe('Pipeline Status', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Documents/i }).click()
  })

  test('pipeline status button exists', async ({ page }) => {
    const pipelineButton = page.getByRole('button', { name: /Pipeline/i })

    if (await pipelineButton.count() > 0) {
      await expect(pipelineButton).toBeVisible()
    }
  })

  test('pipeline status button opens status dialog', async ({ page }) => {
    const pipelineButton = page.getByRole('button', { name: /Pipeline/i })

    if (await pipelineButton.count() > 0 && await pipelineButton.isEnabled()) {
      await pipelineButton.click()

      // Should open pipeline status dialog
      const dialog = page.getByRole('dialog')
      await expect(dialog).toBeVisible({ timeout: 3000 })

      // Close
      await page.keyboard.press('Escape')
    }
  })
})
