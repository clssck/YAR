/**
 * Utility functions for document status management.
 */

// Utility functions defined outside component for better performance and to avoid dependency issues
export const getCountValue = (
  counts: Record<string, number>,
  ...keys: string[]
): number => {
  for (const key of keys) {
    const value = counts[key]
    if (typeof value === 'number') {
      return value
    }
  }
  return 0
}

export const hasActiveDocumentsStatus = (counts: Record<string, number>): boolean =>
  getCountValue(counts, 'PROCESSING', 'processing') > 0 ||
  getCountValue(counts, 'PENDING', 'pending') > 0 ||
  getCountValue(counts, 'PREPROCESSED', 'preprocessed') > 0
