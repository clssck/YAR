// Helper function to generate unique IDs with browser compatibility
export const generateUniqueId = () => {
  // Use crypto.randomUUID() if available
  if (
    typeof crypto !== 'undefined' &&
    typeof crypto.randomUUID === 'function'
  ) {
    return crypto.randomUUID()
  }
  // Fallback to timestamp + random string for browsers without crypto.randomUUID
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// LaTeX completeness detection function
export const detectLatexCompleteness = (content: string): boolean => {
  // Check for unclosed block-level LaTeX formulas ($$...$$)
  const blockLatexMatches = content.match(/\$\$/g) || []
  const hasUnclosedBlock = blockLatexMatches.length % 2 !== 0

  // Check for unclosed inline LaTeX formulas ($...$, but not $$)
  // Remove all block formulas first to avoid interference
  const contentWithoutBlocks = content.replace(/\$\$[\s\S]*?\$\$/g, '')
  const inlineLatexMatches =
    contentWithoutBlocks.match(/(?<!\$)\$(?!\$)/g) || []
  const hasUnclosedInline = inlineLatexMatches.length % 2 !== 0

  // LaTeX is complete if there are no unclosed formulas
  return !hasUnclosedBlock && !hasUnclosedInline
}

// Robust COT parsing function to handle multiple think blocks and edge cases
export const parseCOTContent = (content: string) => {
  const thinkStartTag = '<think>'
  const thinkEndTag = '</think>'

  // Find all <think> and </think> tag positions
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

  // Analyze COT state
  const hasThinkStart = startMatches.length > 0
  const hasThinkEnd = endMatches.length > 0
  const isThinking = hasThinkStart && startMatches.length > endMatches.length

  let thinkingContent = ''
  let displayContent = content

  if (hasThinkStart) {
    if (hasThinkEnd && startMatches.length === endMatches.length) {
      // Complete thinking blocks: extract the last complete thinking content
      const lastStartIndex = startMatches[startMatches.length - 1]
      const lastEndIndex = endMatches[endMatches.length - 1]

      if (lastEndIndex > lastStartIndex) {
        thinkingContent = content
          .substring(lastStartIndex + thinkStartTag.length, lastEndIndex)
          .trim()

        // Remove all thinking blocks, keep only the final display content
        displayContent = content
          .substring(lastEndIndex + thinkEndTag.length)
          .trim()
      }
    } else if (isThinking) {
      // Currently thinking: extract current thinking content
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

/**
 * Deduplicate references in LLM-generated References section.
 * The LLM sometimes generates duplicate reference lines like:
 * - [2] Document.pdf
 * - [2] Document.pdf
 * This keeps only the first occurrence of each reference.
 */
export const deduplicateReferencesSection = (text: string): string => {
  if (!text) return text

  // Match References section (### References or ## References)
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

/**
 * Strip the References section from the response if user has disabled it.
 */
export const stripReferencesSection = (text: string): string => {
  if (!text) return text
  // Match References section and everything after it
  const refsPattern =
    /\n*(#{2,3}\s*References|References:?)\s*\n((?:[-*]\s*\[\d+\][^\n]*\n?)+)/gi
  return text.replace(refsPattern, '').trim()
}

/**
 * Renumber citation markers to be sequential (1, 2, 3...) instead of sparse (2, 5, 9...).
 * The LLM generates citations using original chunk indices which can be sparse.
 * This function renumbers them to be sequential for better UX.
 *
 * Uses placeholder tokens to avoid replacement collisions (e.g., [5]→[1] then [1]→[3]).
 */
export const renumberReferencesSequential = (text: string): string => {
  if (!text) return text

  // Find all unique reference numbers in the text
  const refPattern = /\[(\d+)\]/g
  const allRefs: string[] = []
  let match: RegExpExecArray | null = refPattern.exec(text)
  while (match !== null) {
    allRefs.push(match[1])
    match = refPattern.exec(text)
  }

  if (allRefs.length === 0) return text

  // Get unique reference numbers in order of first appearance
  const seen = new Set<string>()
  const uniqueRefs: string[] = []
  for (const ref of allRefs) {
    if (!seen.has(ref)) {
      seen.add(ref)
      uniqueRefs.push(ref)
    }
  }

  // Create mapping: old_number -> new_sequential_number
  const refMapping = new Map<string, string>()
  uniqueRefs.forEach((oldNum, index) => {
    refMapping.set(oldNum, String(index + 1))
  })

  // Two-pass replacement using placeholder tokens to avoid collisions
  // e.g., [5]→[1] then [1]→[3] would incorrectly change original [5] to [3]
  const placeholder = '\x00REF_'
  let result = text

  // First pass: replace all [n] with placeholder tokens
  for (const oldNum of refMapping.keys()) {
    result = result.replace(
      new RegExp(`\\[${oldNum}\\]`, 'g'),
      `${placeholder}${oldNum}\x00`,
    )
  }

  // Second pass: replace placeholder tokens with new sequential numbers
  for (const [oldNum, newNum] of refMapping.entries()) {
    result = result.replace(
      new RegExp(`${placeholder}${oldNum}\x00`, 'g'),
      `[${newNum}]`,
    )
  }

  return result
}

export type NonStreamReference = {
  reference_id: string
  file_path: string
}

export type NonStreamQueryResponse = {
  response: string
  references?: NonStreamReference[] | null
}

export const applyNonStreamResponse = (
  assistantMessage: { id: string; references?: NonStreamReference[] },
  response: NonStreamQueryResponse,
  messages: Array<{ id: string; references?: NonStreamReference[] }>,
) => {
  const references = response.references ?? undefined
  assistantMessage.references = references
  return messages.map((msg) =>
    msg.id === assistantMessage.id
      ? {
          ...msg,
          references,
        }
      : msg,
  )
}
