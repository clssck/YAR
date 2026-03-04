import '../setup'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

const navigateToLogin = mock(() => {})

mock.module('@/stores/settings', () => ({
  useSettingsStore: {
    getState: () => ({ apiKey: null }),
  },
}))

mock.module('@/services/navigation', () => ({
  navigationService: {
    navigateToLogin,
  },
}))

mock.module('@/lib/constants', () => ({
  backendBaseUrl: 'http://backend.test',
  popularLabelsDefaultLimit: 20,
  searchLabelsDefaultLimit: 20,
}))

import {
  type QueryRequest,
  type StreamReference,
  queryTextStream,
} from '@/api/yar'

const makeSuccessResponse = (body: string): Response =>
  new Response(body, {
    status: 200,
    headers: {
      'Content-Type': 'application/x-ndjson',
    },
  })

describe('queryTextStream NDJSON parsing', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
    localStorage.removeItem('YAR-API-TOKEN')
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
    navigateToLogin.mockReset()
  })

  test('processes response and references when both appear in one NDJSON object', async () => {
    const references: StreamReference[] = [
      {
        reference_id: '1',
        file_path: '/docs/a.pdf',
        document_title: 'Doc A',
        s3_key: null,
        excerpt: null,
        presigned_url: 'https://example.test/a.pdf',
      },
    ]

    const ndjson = `${JSON.stringify({ response: 'Answer chunk', references })}\n`
    globalThis.fetch = mock(async () => makeSuccessResponse(ndjson)) as typeof fetch

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []
    const onError = mock(() => {})

    await queryTextStream(
      { query: 'What is AI?', mode: 'mix', stream: true } as QueryRequest,
      (chunk) => chunks.push(chunk),
      onError,
      undefined,
      (refs) => receivedReferences.push(refs),
    )

    expect(chunks).toEqual(['Answer chunk'])
    expect(receivedReferences).toEqual([references])
    expect(onError).toHaveBeenCalledTimes(0)
  })

  test('processes references from the final buffered NDJSON object without trailing newline', async () => {
    const references: StreamReference[] = [
      {
        reference_id: '2',
        file_path: '/docs/b.pdf',
        document_title: 'Doc B',
        s3_key: null,
        excerpt: null,
        presigned_url: null,
      },
    ]

    const finalChunk = JSON.stringify({ response: 'Buffered answer', references })
    globalThis.fetch = mock(async () => makeSuccessResponse(finalChunk)) as typeof fetch

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []

    await queryTextStream(
      { query: 'Tell me more', mode: 'mix', stream: true } as QueryRequest,
      (chunk) => chunks.push(chunk),
      undefined,
      undefined,
      (refs) => receivedReferences.push(refs),
    )

    expect(chunks).toEqual(['Buffered answer'])
    expect(receivedReferences).toEqual([references])
  })

  test('reports malformed references payload in streamed lines and skips onReferences', async () => {
    const ndjson = `${JSON.stringify({ response: 'Chunk before bad refs', references: { invalid: true } })}\n`
    globalThis.fetch = mock(async () => makeSuccessResponse(ndjson)) as typeof fetch

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []
    const onError = mock(() => {})

    await queryTextStream(
      { query: 'What happened?', mode: 'mix', stream: true } as QueryRequest,
      (chunk) => chunks.push(chunk),
      onError,
      undefined,
      (refs) => receivedReferences.push(refs),
    )

    expect(chunks).toEqual(['Chunk before bad refs'])
    expect(receivedReferences).toEqual([])
    expect(onError).toHaveBeenCalledWith(
      'Protocol error: expected "references" to be an array',
    )
  })

  test('routes citation_error event from final buffered object to onError', async () => {
    const finalChunk = JSON.stringify({ citation_error: 'citation service unavailable' })
    globalThis.fetch = mock(async () => makeSuccessResponse(finalChunk)) as typeof fetch

    const onError = mock(() => {})

    await queryTextStream(
      { query: 'Need citations', mode: 'mix', stream: true } as QueryRequest,
      () => {},
      onError,
    )

    expect(onError).toHaveBeenCalledWith(
      'Citation error: citation service unavailable',
    )
  })
})
