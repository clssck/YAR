import { test, expect } from '@playwright/test'

test.describe('Query Settings - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
    // Wait for panel to load
    await expect(page.getByText(/Parameters/i).first()).toBeVisible()
  })

  test('top_k input accepts and retains values', async ({ page }) => {
    const topKInput = page.locator('#top_k')

    if (await topKInput.count() > 0) {
      await topKInput.fill('50')
      await expect(topKInput).toHaveValue('50')

      // Change to another value
      await topKInput.fill('100')
      await expect(topKInput).toHaveValue('100')
    }
  })

  test('chunk_top_k input accepts and retains values', async ({ page }) => {
    const chunkTopKInput = page.locator('#chunk_top_k')

    if (await chunkTopKInput.count() > 0) {
      await chunkTopKInput.fill('25')
      await expect(chunkTopKInput).toHaveValue('25')
    }
  })

  test('streaming toggle can be switched on and off', async ({ page }) => {
    // Expand the "Output" section first
    const outputSection = page.locator('button').filter({ hasText: /Output/i })
    await outputSection.click()

    // Find the streaming checkbox by its label association
    const streamingCheckbox = page.locator('#stream')
    await expect(streamingCheckbox).toBeVisible()

    const initialChecked = await streamingCheckbox.isChecked()

    // Toggle it
    await streamingCheckbox.click()

    // Verify state changed
    const newChecked = await streamingCheckbox.isChecked()
    expect(newChecked).not.toBe(initialChecked)

    // Toggle back
    await streamingCheckbox.click()
    expect(await streamingCheckbox.isChecked()).toBe(initialChecked)
  })

  test('enable rerank checkbox can be toggled', async ({ page }) => {
    // Expand the "Advanced" section first
    const advancedSection = page.locator('button').filter({ hasText: /Advanced/i })
    await advancedSection.click()

    const rerankCheckbox = page.locator('#enable_rerank')
    await expect(rerankCheckbox).toBeVisible()

    // Note: rerank may be disabled if no reranker is configured
    const isDisabled = await rerankCheckbox.isDisabled()
    if (!isDisabled) {
      const initialChecked = await rerankCheckbox.isChecked()
      await rerankCheckbox.click()
      expect(await rerankCheckbox.isChecked()).not.toBe(initialChecked)
    }
  })

  test('only need context checkbox can be toggled', async ({ page }) => {
    // Expand the "Output" section first
    const outputSection = page.locator('button').filter({ hasText: /Output/i })
    await outputSection.click()

    const contextCheckbox = page.locator('#only_need_context')
    await expect(contextCheckbox).toBeVisible()

    const initialChecked = await contextCheckbox.isChecked()
    await contextCheckbox.click()
    expect(await contextCheckbox.isChecked()).not.toBe(initialChecked)
  })

  test('only need prompt checkbox can be toggled', async ({ page }) => {
    // Expand the "Output" section first
    const outputSection = page.locator('button').filter({ hasText: /Output/i })
    await outputSection.click()

    const promptCheckbox = page.locator('#only_need_prompt')
    await expect(promptCheckbox).toBeVisible()

    const initialChecked = await promptCheckbox.isChecked()
    await promptCheckbox.click()
    expect(await promptCheckbox.isChecked()).not.toBe(initialChecked)
  })

  test('user prompt textarea accepts text input', async ({ page }) => {
    // Find user prompt input area
    const userPromptInput = page.locator('textarea').filter({ hasText: '' }).first()
      .or(page.locator('[placeholder*="prompt"]'))
      .or(page.locator('#user_prompt'))

    if (await userPromptInput.count() > 0) {
      await userPromptInput.fill('Custom instruction for the response')
      const value = await userPromptInput.inputValue()
      expect(value).toContain('Custom instruction')
    }
  })
})

test.describe('Mode Selector - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('selecting mode updates the selector display', async ({ page }) => {
    const modeSelect = page.locator('#query_mode_select')

    // Select Local mode
    await modeSelect.click()
    await page.getByRole('option', { name: /Local/i }).click()
    await expect(modeSelect).toContainText(/Local/i)

    // Select Global mode
    await modeSelect.click()
    await page.getByRole('option', { name: /Global/i }).click()
    await expect(modeSelect).toContainText(/Global/i)

    // Select Naive mode
    await modeSelect.click()
    await page.getByRole('option', { name: /Naive/i }).click()
    await expect(modeSelect).toContainText(/Naive/i)
  })

  test('mode selection persists to localStorage', async ({ page }) => {
    const modeSelect = page.locator('#query_mode_select')

    // Select Hybrid mode
    await modeSelect.click()
    await page.getByRole('option', { name: /Hybrid/i }).click()

    // Wait for persistence
    await page.waitForTimeout(500)

    // Check localStorage
    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage?.state?.querySettings?.mode).toBe('hybrid')
  })

  test('mode selection survives page reload', async ({ page }) => {
    const modeSelect = page.locator('#query_mode_select')

    // Select Bypass mode
    await modeSelect.click()
    await page.getByRole('option', { name: /Bypass/i }).click()
    await expect(modeSelect).toContainText(/Bypass/i)

    // Wait for persistence
    await page.waitForTimeout(500)

    // Reload page
    await page.reload()
    await page.getByRole('tab', { name: /Retrieval/i }).click()

    // Mode should still be Bypass
    const newModeSelect = page.locator('#query_mode_select')
    await expect(newModeSelect).toContainText(/Bypass/i)
  })
})

test.describe('Input Field - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('input accepts and displays typed text', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)

    await input.fill('What is the capital of France?')
    await expect(input).toHaveValue('What is the capital of France?')
  })

  test('input clears when cleared programmatically', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)

    await input.fill('Test query')
    await expect(input).toHaveValue('Test query')

    await input.fill('')
    await expect(input).toHaveValue('')
  })

  test('multiline input works with Shift+Enter', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)

    await input.fill('Line 1')
    await input.press('Shift+Enter')
    await input.pressSequentially('Line 2')
    await input.press('Shift+Enter')
    await input.pressSequentially('Line 3')

    const value = await input.inputValue()
    expect(value).toContain('Line 1')
    expect(value).toContain('Line 2')
    expect(value).toContain('Line 3')
    expect(value.split('\n').length).toBeGreaterThanOrEqual(3)
  })

  test('input with mode prefix is parsed correctly', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)

    // Type a query with mode prefix
    await input.fill('/local What are the main entities?')

    // Submit - even if no backend, should not show invalid mode error
    await input.press('Enter')

    // Should NOT show invalid mode error for valid prefix
    const invalidModeError = page.getByText(/Invalid query mode prefix/i)
    await expect(invalidModeError).not.toBeVisible()
  })
})

test.describe('Settings Persistence - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('multiple settings changes persist together', async ({ page }) => {
    // Change mode
    const modeSelect = page.locator('#query_mode_select')
    await modeSelect.click()
    await page.getByRole('option', { name: /Mix/i }).click()

    // Change top_k if available
    const topKInput = page.locator('#top_k')
    if (await topKInput.count() > 0) {
      await topKInput.fill('75')
    }

    // Wait for persistence
    await page.waitForTimeout(500)

    // Verify in localStorage
    const storage = await page.evaluate(() => {
      const data = localStorage.getItem('settings-storage')
      return data ? JSON.parse(data) : null
    })

    expect(storage?.state?.querySettings?.mode).toBe('mix')
  })

  test('settings survive tab switching', async ({ page }) => {
    // Set a specific mode
    const modeSelect = page.locator('#query_mode_select')
    await modeSelect.click()
    await page.getByRole('option', { name: /Global/i }).click()

    // Switch to Documents tab
    await page.getByRole('tab', { name: /Documents/i }).click()
    await expect(page.getByRole('tab', { name: /Documents/i })).toHaveAttribute('data-state', 'active')

    // Switch to Knowledge Graph tab
    await page.getByRole('tab', { name: /Knowledge Graph/i }).click()

    // Switch back to Retrieval
    await page.getByRole('tab', { name: /Retrieval/i }).click()

    // Mode should still be Global
    const newModeSelect = page.locator('#query_mode_select')
    await expect(newModeSelect).toContainText(/Global/i)
  })
})

test.describe('Clear and Export Buttons - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('clear button stays disabled without messages', async ({ page }) => {
    const clearButton = page.getByRole('button', { name: /Clear/i })

    // Should be disabled
    await expect(clearButton).toBeDisabled()

    // Type something in input (doesn't create messages)
    const input = page.getByPlaceholder(/Enter your query/i)
    await input.fill('Test query')

    // Clear should still be disabled (no messages yet)
    await expect(clearButton).toBeDisabled()
  })

  test('export button stays disabled without messages', async ({ page }) => {
    const exportButton = page.getByRole('button', { name: /Export/i })

    // Should be disabled
    await expect(exportButton).toBeDisabled()

    // Wait and verify still disabled
    await page.waitForTimeout(1000)
    await expect(exportButton).toBeDisabled()
  })
})

test.describe('Token Limit Inputs - Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('max entity tokens input accepts values', async ({ page }) => {
    const input = page.locator('#max_entity_tokens')

    if (await input.count() > 0) {
      await input.fill('8000')
      await expect(input).toHaveValue('8000')
    }
  })

  test('max relation tokens input accepts values', async ({ page }) => {
    const input = page.locator('#max_relation_tokens')

    if (await input.count() > 0) {
      await input.fill('10000')
      await expect(input).toHaveValue('10000')
    }
  })

  test('max total tokens input accepts values', async ({ page }) => {
    const input = page.locator('#max_total_tokens')

    if (await input.count() > 0) {
      await input.fill('40000')
      await expect(input).toHaveValue('40000')
    }
  })
})
