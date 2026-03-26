/**
 * Returns true if the URL uses a safe scheme or is a same-origin relative path.
 * Rejects javascript:, data:, and any other non-http(s) absolute scheme.
 */
export const isSafeUrl = (url: string): boolean =>
  url.startsWith('https://') || url.startsWith('http://') || url.startsWith('/') || url.startsWith('./')
  || url.startsWith('../')
