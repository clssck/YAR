/**
 * Public re-export shim for the YAR API client.
 *
 * The implementation lives in `yarImpl` so consumer tests
 * (e.g. RetrievalTesting.render, DocumentManager.render) can mock
 * `@/api/yar` to stub out HTTP without preventing the dedicated yar
 * test suite from importing the real implementation directly via
 * `@/api/yarImpl`.
 */
export * from './yarImpl'
