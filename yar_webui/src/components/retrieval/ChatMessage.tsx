/**
 * Public re-export shim for the ChatMessage component.
 *
 * The implementation lives in `ChatMessageImpl` so consumer tests
 * (e.g. RetrievalTesting.render) can mock `@/components/retrieval/ChatMessage`
 * with a render-counting stub without preventing the dedicated ChatMessage
 * test suite from importing the real implementation directly.
 */
export * from './ChatMessageImpl'
