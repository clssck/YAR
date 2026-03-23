import type { QueryMode } from '@/api/yar'

export const MIN_RETRIEVAL_QUERY_LENGTH = 3
export const ALLOWED_RETRIEVAL_QUERY_MODES: QueryMode[] = [
  'naive',
  'local',
  'global',
  'hybrid',
  'mix',
  'bypass',
]

export type RetrievalInputValidationResult =
  | {
      ok: true
      effectiveMode: QueryMode | undefined
      trimmedQuery: string
    }
  | {
      ok: false
      error: 'invalid_prefix' | 'invalid_mode' | 'too_short'
    }

export const validateRetrievalInput = (
  inputValue: string,
  modeOverride: QueryMode | null,
  minLength: number = MIN_RETRIEVAL_QUERY_LENGTH,
): RetrievalInputValidationResult => {
  const prefixMatch = inputValue.match(/^\/(\w+)\s+([\s\S]+)/)

  if (/^\/\S+/.test(inputValue) && !prefixMatch) {
    return { ok: false, error: 'invalid_prefix' }
  }

  let effectiveMode: QueryMode | undefined = modeOverride ?? undefined
  let actualQuery = inputValue

  if (prefixMatch) {
    const mode = prefixMatch[1] as QueryMode
    const query = prefixMatch[2]

    if (!ALLOWED_RETRIEVAL_QUERY_MODES.includes(mode)) {
      return { ok: false, error: 'invalid_mode' }
    }

    effectiveMode = mode
    actualQuery = query
  }

  const trimmedQuery = actualQuery.trim()
  if (trimmedQuery.length < minLength) {
    return { ok: false, error: 'too_short' }
  }

  return {
    ok: true,
    effectiveMode,
    trimmedQuery,
  }
}
