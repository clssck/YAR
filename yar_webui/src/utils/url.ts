/**
 * Returns true if the URL uses a safe scheme (http or https).
 * Rejects javascript:, data:, and any other non-http(s) scheme.
 */
export const isSafeUrl = (url: string): boolean =>
  url.startsWith('https://') || url.startsWith('http://')
