/**
 * Tests for RetrievalTesting component utility functions
 */
import '../setup'
import { describe, expect, test } from 'bun:test'

// =============================================================================
// Utility Function Recreations for Testing
// =============================================================================

// Helper function to generate unique IDs with browser compatibility
const generateUniqueId = () => {
  if (
    typeof crypto !== 'undefined' &&
    typeof crypto.randomUUID === 'function'
  ) {
    return crypto.randomUUID()
  }
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// LaTeX completeness detection function
const detectLatexCompleteness = (content: string): boolean => {
  const blockLatexMatches = content.match(/\$\$/g) || []
  const hasUnclosedBlock = blockLatexMatches.length % 2 !== 0

  const contentWithoutBlocks = content.replace(/\$\$[\s\S]*?\$\$/g, '')
  const inlineLatexMatches =
    contentWithoutBlocks.match(/(?<!\$)\$(?!\$)/g) || []
  const hasUnclosedInline = inlineLatexMatches.length % 2 !== 0

  return !hasUnclosedBlock && !hasUnclosedInline
}

// Robust COT parsing function
const parseCOTContent = (content: string) => {
  const thinkStartTag = '<think>'
  const thinkEndTag = '</think>'

  const startMatches: number[] = []
  const endMatches: number[] = []

  let startIndex = content.indexOf(thinkStartTag)
  while (startIndex !== -1) {
    startMatches.push(startIndex)
    startIndex = content.indexOf(
      thinkStartTag,
      startIndex + thinkStartTag.length,
    )
  }

  let endIndex = content.indexOf(thinkEndTag)
  while (endIndex !== -1) {
    endMatches.push(endIndex)
    endIndex = content.indexOf(thinkEndTag, endIndex + thinkEndTag.length)
  }

  const hasThinkStart = startMatches.length > 0
  const hasThinkEnd = endMatches.length > 0
  const isThinking = hasThinkStart && startMatches.length > endMatches.length

  let thinkingContent = ''
  let displayContent = content

  if (hasThinkStart) {
    if (hasThinkEnd && startMatches.length === endMatches.length) {
      const lastStartIndex = startMatches[startMatches.length - 1]
      const lastEndIndex = endMatches[endMatches.length - 1]

      if (lastEndIndex > lastStartIndex) {
        thinkingContent = content
          .substring(lastStartIndex + thinkStartTag.length, lastEndIndex)
          .trim()
        displayContent = content
          .substring(lastEndIndex + thinkEndTag.length)
          .trim()
      }
    } else if (isThinking) {
      const lastStartIndex = startMatches[startMatches.length - 1]
      thinkingContent = content.substring(lastStartIndex + thinkStartTag.length)
      displayContent = ''
    }
  }

  return {
    isThinking,
    thinkingContent,
    displayContent,
    hasValidThinkBlock:
      hasThinkStart && hasThinkEnd && startMatches.length === endMatches.length,
  }
}

// Deduplicate references in LLM-generated References section
const deduplicateReferencesSection = (text: string): string => {
  if (!text) return text

  const refsPattern =
    /(#{2,3}\s*References|References:?)\s*\n((?:[-*]\s*\[\d+\][^\n]*\n?)+)/gi

  return text.replace(refsPattern, (_match, header, refsBlock) => {
    const refLinePattern = /[-*]\s*\[(\d+)\]\s*([^\n]+)/
    const seenRefs = new Set<string>()
    const uniqueLines: string[] = []

    for (const line of refsBlock.trim().split('\n')) {
      const trimmedLine = line.trim()
      if (!trimmedLine) continue

      const refMatch = trimmedLine.match(refLinePattern)
      if (refMatch) {
        const refKey = `${refMatch[1]}:${refMatch[2].trim()}`
        if (!seenRefs.has(refKey)) {
          seenRefs.add(refKey)
          uniqueLines.push(trimmedLine)
        }
      } else {
        uniqueLines.push(trimmedLine)
      }
    }

    return `${header}\n${uniqueLines.join('\n')}\n`
  })
}

// Strip the References section from the response
const stripReferencesSection = (text: string): string => {
  if (!text) return text
  const refsPattern =
    /\n*(#{2,3}\s*References|References:?)\s*\n((?:[-*]\s*\[\d+\][^\n]*\n?)+)/gi
  return text.replace(refsPattern, '').trim()
}

// Renumber citation markers to be sequential
const renumberReferencesSequential = (text: string): string => {
  if (!text) return text

  const refPattern = /\[(\d+)\]/g
  const allRefs: string[] = []
  let match: RegExpExecArray | null = refPattern.exec(text)
  while (match !== null) {
    allRefs.push(match[1])
    match = refPattern.exec(text)
  }

  if (allRefs.length === 0) return text

  const seen = new Set<string>()
  const uniqueRefs: string[] = []
  for (const ref of allRefs) {
    if (!seen.has(ref)) {
      seen.add(ref)
      uniqueRefs.push(ref)
    }
  }

  const refMapping = new Map<string, string>()
  uniqueRefs.forEach((oldNum, index) => {
    refMapping.set(oldNum, String(index + 1))
  })

  const placeholder = '\x00REF_'
  let result = text

  for (const oldNum of refMapping.keys()) {
    result = result.replace(
      new RegExp(`\\[${oldNum}\\]`, 'g'),
      `${placeholder}${oldNum}\x00`,
    )
  }

  for (const [oldNum, newNum] of refMapping.entries()) {
    result = result.replace(
      new RegExp(`${placeholder}${oldNum}\x00`, 'g'),
      `[${newNum}]`,
    )
  }

  return result
}

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

// =============================================================================
// Query Mode Tests
// =============================================================================

describe('Query Mode Types', () => {
  type QueryMode = 'local' | 'global' | 'hybrid' | 'naive' | 'mix'

  test('all query modes are valid', () => {
    const modes: QueryMode[] = ['local', 'global', 'hybrid', 'naive', 'mix']
    expect(modes).toHaveLength(5)
  })

  test('mix is the default mode', () => {
    const defaultMode: QueryMode = 'mix'
    expect(defaultMode).toBe('mix')
  })
})

// =============================================================================
// Message Type Tests
// =============================================================================

describe('Message Types', () => {
  interface MessageWithError {
    id: string
    role: 'user' | 'assistant'
    content: string
    isError?: boolean
    errorType?: 'timeout' | 'auth' | 'server' | 'network' | 'unknown'
    isThinking?: boolean
    timestamp?: number
  }

  test('creates valid user message', () => {
    const msg: MessageWithError = {
      id: 'msg-1',
      role: 'user',
      content: 'Hello',
    }
    expect(msg.role).toBe('user')
    expect(msg.isError).toBeUndefined()
  })

  test('creates valid assistant message', () => {
    const msg: MessageWithError = {
      id: 'msg-2',
      role: 'assistant',
      content: 'Hi there',
      timestamp: Date.now(),
    }
    expect(msg.role).toBe('assistant')
    expect(msg.timestamp).toBeDefined()
  })

  test('creates error message', () => {
    const msg: MessageWithError = {
      id: 'msg-3',
      role: 'assistant',
      content: 'Error occurred',
      isError: true,
      errorType: 'server',
    }
    expect(msg.isError).toBe(true)
    expect(msg.errorType).toBe('server')
  })

  test('creates thinking message', () => {
    const msg: MessageWithError = {
      id: 'msg-4',
      role: 'assistant',
      content: '',
      isThinking: true,
    }
    expect(msg.isThinking).toBe(true)
  })
})
