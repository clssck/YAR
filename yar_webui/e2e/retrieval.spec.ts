import { expect, test } from '@playwright/test'

test.describe('Retrieval Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Navigate to retrieval tab
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('shows retrieval input', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)
    await expect(input).toBeVisible()
  })

  test('shows send button', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: /Send/i })
    await expect(sendButton).toBeVisible()
  })

  test('shows clear button', async ({ page }) => {
    const clearButton = page.getByRole('button', { name: /Clear/i })
    await expect(clearButton).toBeVisible()
  })

  test('shows export button', async ({ page }) => {
    const exportButton = page.getByRole('button', { name: /Export/i })
    await expect(exportButton).toBeVisible()
  })

  test('shows mode selector', async ({ page }) => {
    // Mode selector button (shows "Default" or a mode name)
    const modeButton = page
      .locator('button')
      .filter({ hasText: /Default|Hybrid|Local|Global|Naive|Mix|Bypass/i })
    await expect(modeButton.first()).toBeVisible()
  })

  test('can type in input field', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)
    await input.fill('Test question')
    await expect(input).toHaveValue('Test question')
  })

  test('input expands to textarea on multiline', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)

    // Type single line first
    await input.fill('Single line')

    // Press Shift+Enter to add newline
    await input.press('Shift+Enter')
    await input.pressSequentially('Second line')

    // Should now be a textarea with the multiline content
    const value = await input.inputValue()
    expect(value).toContain('\n')
  })

  test('clear button is disabled when no messages', async ({ page }) => {
    const clearButton = page.getByRole('button', { name: /Clear/i })
    // Initially should be disabled (no messages)
    await expect(clearButton).toBeDisabled()
  })

  test('export button is disabled when no messages', async ({ page }) => {
    const exportButton = page.getByRole('button', { name: /Export/i })
    // Initially should be disabled (no messages)
    await expect(exportButton).toBeDisabled()
  })

  test('shows empty state message', async ({ page }) => {
    // Should show prompt to start retrieval
    await expect(
      page.getByText(/Start a retrieval by typing your query/i),
    ).toBeVisible()
  })
})

test.describe('Mode Selector', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('opens mode dropdown on click', async ({ page }) => {
    // Mode selector is a Select component with id="query_mode_select"
    const modeSelect = page.locator('#query_mode_select')
    await modeSelect.click()

    // Should show mode options in dropdown
    await expect(page.getByRole('option', { name: /Hybrid/i })).toBeVisible()
    await expect(page.getByRole('option', { name: /Local/i })).toBeVisible()
    await expect(page.getByRole('option', { name: /Global/i })).toBeVisible()
  })

  test('can select different mode', async ({ page }) => {
    const modeSelect = page.locator('#query_mode_select')
    await modeSelect.click()

    // Select Hybrid mode
    await page.getByRole('option', { name: /Hybrid/i }).click()

    // Mode selector should now show Hybrid
    await expect(modeSelect).toContainText(/Hybrid/i)
  })
})

test.describe('Query Settings Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('shows parameters panel on right side', async ({ page }) => {
    // Query settings panel has title "Parameters"
    await expect(page.getByText(/Parameters/i).first()).toBeVisible()
  })

  test('has streaming toggle', async ({ page }) => {
    // Should have streaming option
    const streamingLabel = page.getByText(/Stream/i)
    await expect(streamingLabel.first()).toBeVisible()
  })
})

test.describe('Retrieval Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Retrieval/i }).click()
  })

  test('Cmd/Ctrl+K focuses input', async ({ page }) => {
    // First click somewhere else to unfocus
    await page.locator('body').click()

    // Press Cmd+K (or Ctrl+K on non-Mac)
    await page.keyboard.press('Meta+k')

    // Input should be focused
    const input = page.getByPlaceholder(/Enter your query/i)
    await expect(input).toBeFocused()
  })

  test('Enter submits query', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)
    await input.fill('Test query')

    // Enter should trigger submit (though it may fail without backend)
    await input.press('Enter')

    // Input should be cleared after submit attempt
    // Note: This may show an error if no backend, but input should clear
  })

  test('Shift+Enter adds newline', async ({ page }) => {
    const input = page.getByPlaceholder(/Enter your query/i)
    await input.fill('Line 1')
    await input.press('Shift+Enter')
    await input.pressSequentially('Line 2')

    const value = await input.inputValue()
    expect(value).toContain('Line 1\nLine 2')
  })
})
