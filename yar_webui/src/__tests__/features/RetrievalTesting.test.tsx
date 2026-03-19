/**
 * Tests for RetrievalTesting component utility functions
 */
import '../setup'
import { describe, expect, test } from 'bun:test'
import {
  applyNonStreamResponse,
  deduplicateReferencesSection,
  detectLatexCompleteness,
  generateUniqueId,
  type NonStreamReference,
  parseCOTContent,
  renumberReferencesSequential,
  stripReferencesSection,
} from '@/utils/textProcessing'


// =============================================================================
// Tests
// =============================================================================

describe('RetrievalTesting Utility Functions', () => {
  describe('generateUniqueId', () => {
    test('generates non-empty string', () => {
      const id = generateUniqueId()
      expect(id).toBeDefined()
      expect(id.length).toBeGreaterThan(0)
    })

    test('generates unique IDs', () => {
      const ids = new Set<string>()
      for (let i = 0; i < 100; i++) {
        ids.add(generateUniqueId())
      }
      expect(ids.size).toBe(100)
    })

    test('generates UUID format when crypto available', () => {
      const id = generateUniqueId()
      // UUID format or fallback format
      expect(typeof id).toBe('string')
    })
  })

  describe('detectLatexCompleteness', () => {
    test('returns true for content without LaTeX', () => {
      expect(detectLatexCompleteness('Hello world')).toBe(true)
    })

    test('returns true for complete inline LaTeX', () => {
      expect(detectLatexCompleteness('Formula: $E = mc^2$ is famous')).toBe(
        true,
      )
    })

    test('returns true for complete block LaTeX', () => {
      expect(
        detectLatexCompleteness('Formula: $$E = mc^2$$ is displayed'),
      ).toBe(true)
    })

    test('returns false for unclosed inline LaTeX', () => {
      expect(detectLatexCompleteness('Formula: $E = mc^2 is incomplete')).toBe(
        false,
      )
    })

    test('returns false for unclosed block LaTeX', () => {
      expect(detectLatexCompleteness('Formula: $$E = mc^2 is incomplete')).toBe(
        false,
      )
    })

    test('returns true for multiple complete inline formulas', () => {
      expect(detectLatexCompleteness('$a$ and $b$ and $c$')).toBe(true)
    })

    test('returns true for multiple complete block formulas', () => {
      expect(detectLatexCompleteness('$$a$$ and $$b$$')).toBe(true)
    })

    test('returns true for mixed complete formulas', () => {
      expect(detectLatexCompleteness('Inline $x$ and block $$y$$')).toBe(true)
    })

    test('returns false for odd number of block delimiters', () => {
      expect(detectLatexCompleteness('$$a$$ and $$b')).toBe(false)
    })

    test('handles empty string', () => {
      expect(detectLatexCompleteness('')).toBe(true)
    })
  })

  describe('parseCOTContent', () => {
    test('returns original content when no think tags', () => {
      const result = parseCOTContent('Hello world')
      expect(result.isThinking).toBe(false)
      expect(result.thinkingContent).toBe('')
      expect(result.displayContent).toBe('Hello world')
      expect(result.hasValidThinkBlock).toBe(false)
    })

    test('detects active thinking state', () => {
      const result = parseCOTContent('<think>Analyzing the question...')
      expect(result.isThinking).toBe(true)
      expect(result.thinkingContent).toBe('Analyzing the question...')
      expect(result.displayContent).toBe('')
    })

    test('extracts thinking content from complete block', () => {
      const result = parseCOTContent('<think>My analysis</think>Final answer')
      expect(result.isThinking).toBe(false)
      expect(result.thinkingContent).toBe('My analysis')
      expect(result.displayContent).toBe('Final answer')
      expect(result.hasValidThinkBlock).toBe(true)
    })

    test('handles multiple think blocks', () => {
      const result = parseCOTContent(
        '<think>First thought</think>Middle<think>Second thought</think>Final',
      )
      expect(result.hasValidThinkBlock).toBe(true)
      expect(result.thinkingContent).toBe('Second thought')
      expect(result.displayContent).toBe('Final')
    })

    test('handles empty think block', () => {
      const result = parseCOTContent('<think></think>Answer')
      expect(result.isThinking).toBe(false)
      expect(result.displayContent).toBe('Answer')
    })

    test('handles whitespace in think block', () => {
      const result = parseCOTContent('<think>  spaced content  </think>Answer')
      expect(result.thinkingContent).toBe('spaced content')
    })

    test('detects incomplete think block during streaming', () => {
      const result = parseCOTContent('<think>Still thinking about')
      expect(result.isThinking).toBe(true)
      expect(result.thinkingContent).toBe('Still thinking about')
    })
  })

  describe('deduplicateReferencesSection', () => {
    test('removes duplicate references', () => {
      const input = `Some text

### References
- [1] Document.pdf
- [1] Document.pdf
- [2] Other.pdf
`
      const result = deduplicateReferencesSection(input)
      expect(result.match(/\[1\] Document\.pdf/g)?.length).toBe(1)
      expect(result).toContain('[2] Other.pdf')
    })

    test('preserves unique references', () => {
      const input = `Text

### References
- [1] First.pdf
- [2] Second.pdf
- [3] Third.pdf
`
      const result = deduplicateReferencesSection(input)
      expect(result).toContain('[1] First.pdf')
      expect(result).toContain('[2] Second.pdf')
      expect(result).toContain('[3] Third.pdf')
    })

    test('handles empty string', () => {
      expect(deduplicateReferencesSection('')).toBe('')
    })

    test('handles text without references section', () => {
      const input = 'Just some text without references'
      expect(deduplicateReferencesSection(input)).toBe(input)
    })

    test('handles References: format', () => {
      const input = `Text

References:
- [1] Doc.pdf
- [1] Doc.pdf
`
      const result = deduplicateReferencesSection(input)
      expect(result.match(/\[1\] Doc\.pdf/g)?.length).toBe(1)
    })

    test('handles ## References format', () => {
      const input = `Text

## References
- [1] Doc.pdf
- [1] Doc.pdf
`
      const result = deduplicateReferencesSection(input)
      expect(result.match(/\[1\] Doc\.pdf/g)?.length).toBe(1)
    })
  })

  describe('stripReferencesSection', () => {
    test('removes references section', () => {
      const input = `Main content here.

### References
- [1] Source.pdf
- [2] Another.pdf
`
      const result = stripReferencesSection(input)
      expect(result).toBe('Main content here.')
      expect(result).not.toContain('References')
      expect(result).not.toContain('[1]')
    })

    test('handles empty string', () => {
      expect(stripReferencesSection('')).toBe('')
    })

    test('returns content without references unchanged', () => {
      const input = 'Just text without references'
      expect(stripReferencesSection(input)).toBe(input)
    })

    test('handles multiple paragraphs before references', () => {
      const input = `Paragraph 1.

Paragraph 2.

### References
- [1] Source.pdf
`
      const result = stripReferencesSection(input)
      expect(result).toContain('Paragraph 1')
      expect(result).toContain('Paragraph 2')
      expect(result).not.toContain('References')
    })
  })

  describe('renumberReferencesSequential', () => {
    test('renumbers sparse references to sequential', () => {
      const input = 'See [5] and [9] for details.'
      const result = renumberReferencesSequential(input)
      expect(result).toBe('See [1] and [2] for details.')
    })

    test('preserves already sequential references', () => {
      const input = 'See [1] and [2] and [3].'
      const result = renumberReferencesSequential(input)
      expect(result).toBe('See [1] and [2] and [3].')
    })

    test('handles repeated references', () => {
      const input = 'First [5], again [5], then [10].'
      const result = renumberReferencesSequential(input)
      expect(result).toBe('First [1], again [1], then [2].')
    })

    test('handles empty string', () => {
      expect(renumberReferencesSequential('')).toBe('')
    })

    test('handles text without references', () => {
      const input = 'No references here'
      expect(renumberReferencesSequential(input)).toBe(input)
    })

    test('renumbers in order of first appearance', () => {
      const input = '[10] appears first, then [5], then [10] again.'
      const result = renumberReferencesSequential(input)
      expect(result).toBe('[1] appears first, then [2], then [1] again.')
    })

    test('handles single-digit and multi-digit numbers', () => {
      const input = '[1] [10] [100] [2]'
      const result = renumberReferencesSequential(input)
      expect(result).toBe('[1] [2] [3] [4]')
    })

    test('handles references in markdown context', () => {
      const input = `Based on [42], the answer is clear.

### References
- [42] Important source
`
      const result = renumberReferencesSequential(input)
      expect(result).toContain('[1]')
      expect(result).not.toContain('[42]')
    })
  })
})


describe('Non-stream reference propagation', () => {
  test('attaches references to assistant message and state', () => {
    const assistantMessage: { id: string; references?: NonStreamReference[] } = { id: 'assistant-1' }
    const messages: { id: string; references?: NonStreamReference[] }[] = [{ id: 'assistant-1' }, { id: 'assistant-2' }]
    const references: NonStreamReference[] = [
      { reference_id: '1', file_path: '/docs/source.pdf' },
    ]

    const updatedMessages = applyNonStreamResponse(
      assistantMessage,
      { response: 'Answer', references },
      messages,
    )

    expect(assistantMessage.references).toEqual(references)
    expect(updatedMessages[0].references).toEqual(references)
    expect(updatedMessages[1].references).toBeUndefined()
  })

  test('clears references when response references are null', () => {
    const assistantMessage = {
      id: 'assistant-1',
      references: [{ reference_id: 'existing', file_path: '/docs/old.pdf' }],
    }
    const messages = [
      {
        id: 'assistant-1',
        references: [{ reference_id: 'existing', file_path: '/docs/old.pdf' }],
      },
    ]

    const updatedMessages = applyNonStreamResponse(
      assistantMessage,
      { response: 'Answer', references: null },
      messages,
    )

    expect(assistantMessage.references).toBeUndefined()
    expect(updatedMessages[0].references).toBeUndefined()
  })
})
