import { test, expect } from '@playwright/test'

/**
 * Test to verify that citations/references are not duplicated in responses
 */
test.describe('Citation Deduplication', () => {
  test.beforeAll(() => {
    // Increase timeout for LLM responses
    test.setTimeout(120000)
  })

  test('API response should not have duplicate references', async ({ request }) => {
    // Use non-streaming endpoint for simpler testing
    const response = await request.post('http://localhost:9621/query', {
      headers: { 'Content-Type': 'application/json' },
      data: {
        query: 'What is the process development strategy?',
        mode: 'hybrid',
        include_references: true,
      },
      timeout: 90000,
    })

    expect(response.ok()).toBeTruthy()
    const data = await response.json()

    // Check the response text for References section
    const responseText = data.response || ''
    console.log('Response length:', responseText.length)

    // Extract reference lines from the References section
    const refsMatch = responseText.match(/### References\s*\n([\s\S]*?)(?:\n\n|$)/i)
    if (refsMatch) {
      const refsSection = refsMatch[1]
      const refLines = refsSection.split('\n').filter((line: string) => line.trim().startsWith('-'))

      console.log('Found reference lines:', refLines.length)
      console.log('References:', refLines)

      // Count occurrences of each reference
      const refCounts = new Map<string, number>()
      for (const line of refLines) {
        const normalized = line.trim()
        refCounts.set(normalized, (refCounts.get(normalized) || 0) + 1)
      }

      // Check for duplicates
      for (const [ref, count] of refCounts) {
        console.log(`Reference "${ref.substring(0, 50)}..." appears ${count} times`)
        expect(count).toBe(1)
      }
    }

    // Also check that the references array (metadata) has unique reference_ids
    if (data.references) {
      const refIds = data.references.map((r: { reference_id: string }) => r.reference_id)
      const uniqueRefIds = [...new Set(refIds)]
      console.log(`References array: ${refIds.length} total, ${uniqueRefIds.length} unique`)
      expect(refIds.length).toBe(uniqueRefIds.length)
    }
  })

  test('streaming API should have unique sources in citations_metadata', async ({ request }) => {
    // Make a streaming request
    const response = await request.post('http://localhost:9621/query/stream', {
      headers: { 'Content-Type': 'application/json' },
      data: {
        query: 'process development strategy',
        mode: 'hybrid',
        stream: true,
        include_references: true,
        citation_mode: 'footnotes',
      },
      timeout: 90000,
    })

    expect(response.ok()).toBeTruthy()
    const body = await response.text()
    const lines = body.split('\n').filter((line) => line.trim())

    let foundCitationsMetadata = false

    // Find citations_metadata line
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line)

        // Check initial references array
        if (parsed.references) {
          const refIds = parsed.references.map((r: { reference_id: string }) => r.reference_id)
          const uniqueRefIds = [...new Set(refIds)]
          console.log(`Initial references: ${refIds.length} total, ${uniqueRefIds.length} unique`)
          expect(refIds.length).toBe(uniqueRefIds.length)
        }

        // Check citations_metadata.sources
        if (parsed.citations_metadata) {
          foundCitationsMetadata = true
          const sources = parsed.citations_metadata.sources || []
          const sourceRefIds = sources.map((s: { reference_id: string }) => s.reference_id)
          const uniqueSourceRefIds = [...new Set(sourceRefIds)]

          console.log(`Citation sources: ${sourceRefIds.length} total, ${uniqueSourceRefIds.length} unique`)
          expect(sourceRefIds.length).toBe(uniqueSourceRefIds.length)

          // Check footnotes for duplicates
          const footnotes = parsed.citations_metadata.footnotes || []
          const uniqueFootnotes = [...new Set(footnotes)]
          console.log(`Footnotes: ${footnotes.length} total, ${uniqueFootnotes.length} unique`)
          expect(footnotes.length).toBe(uniqueFootnotes.length)
        }
      } catch {
        // Skip non-JSON lines
      }
    }

    console.log('Found citations_metadata:', foundCitationsMetadata)
  })
})
