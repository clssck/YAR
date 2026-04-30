import '../setup'
import { createServer } from 'node:http'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

const TEST_BACKEND_ORIGIN = 'http://127.0.0.1:45451'
const navigateToLogin = mock(() => {})

mock.module('@/services/navigation', () => ({
  navigationService: {
    navigateToLogin,
  },
}))

mock.module('@/lib/constants', () => ({
  backendBaseUrl: TEST_BACKEND_ORIGIN,
  popularLabelsDefaultLimit: 10,
  searchLabelsDefaultLimit: 10,
}))

import type { QueryRequest, StreamReference } from '@/api/yarImpl'

const { getDocumentUrl, queryTextStream } = await import('@/api/yarImpl')

const makeSuccessResponseBody = (body: string) => ({
  statusCode: 200,
  headers: {
    'Content-Type': 'application/x-ndjson',
    'Content-Length': String(Buffer.byteLength(body)),
  },
  body,
})

const startStreamServer = async (body: string) => {
  const server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', TEST_BACKEND_ORIGIN)
    response.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')

    if (request.method === 'OPTIONS') {
      response.writeHead(204)
      response.end()
      return
    }

    if (request.method === 'POST' && request.url === '/query/stream') {
      const mocked = makeSuccessResponseBody(body)
      response.writeHead(mocked.statusCode, mocked.headers)
      response.end(mocked.body)
      return
    }

    response.writeHead(404)
    response.end('not found')
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(45451, '127.0.0.1', () => resolve())
  })

  return {
    stop: () => new Promise<void>((resolve, reject) => {
      server.close((error) => {
        if (error) reject(error)
        else resolve()
      })
    }),
  }
}

describe('queryTextStream NDJSON parsing', () => {
  beforeEach(() => {
    localStorage.removeItem('YAR-API-TOKEN')
    window.happyDOM.setURL(`${TEST_BACKEND_ORIGIN}/webui/`)
  })

  afterEach(() => {
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
    const server = await startStreamServer(ndjson)

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []
    const onError = mock(() => {})

    try {
      await queryTextStream(
        { query: 'What is AI?', mode: 'mix', stream: true } as QueryRequest,
        (chunk) => chunks.push(chunk),
        onError,
        undefined,
        (refs) => receivedReferences.push(refs),
      )
    } finally {
      await server.stop()
    }

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
    const server = await startStreamServer(finalChunk)

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []

    try {
      await queryTextStream(
        { query: 'Tell me more', mode: 'mix', stream: true } as QueryRequest,
        (chunk) => chunks.push(chunk),
        undefined,
        undefined,
        (refs) => receivedReferences.push(refs),
      )
    } finally {
      await server.stop()
    }

    expect(chunks).toEqual(['Buffered answer'])
    expect(receivedReferences).toEqual([references])
  })

  test('reports malformed references payload in streamed lines and skips onReferences', async () => {
    const ndjson = `${JSON.stringify({ response: 'Chunk before bad refs', references: { invalid: true } })}\n`
    const server = await startStreamServer(ndjson)

    const chunks: string[] = []
    const receivedReferences: StreamReference[][] = []
    const onError = mock(() => {})

    try {
      await queryTextStream(
        { query: 'What happened?', mode: 'mix', stream: true } as QueryRequest,
        (chunk) => chunks.push(chunk),
        onError,
        undefined,
        (refs) => receivedReferences.push(refs),
      )
    } finally {
      await server.stop()
    }

    expect(chunks).toEqual(['Chunk before bad refs'])
    expect(receivedReferences).toEqual([])
    expect(onError).toHaveBeenCalledWith(
      'Protocol error: expected "references" to be an array',
    )
  })

  test('routes citation_error event from final buffered object to onError', async () => {
    const finalChunk = JSON.stringify({ citation_error: 'citation service unavailable' })
    const server = await startStreamServer(finalChunk)

    const onError = mock(() => {})

    try {
      await queryTextStream(
        { query: 'Need citations', mode: 'mix', stream: true } as QueryRequest,
        () => {},
        onError,
      )
    } finally {
      await server.stop()
    }

    expect(onError).toHaveBeenCalledWith(
      'Citation error: citation service unavailable',
    )
  })
})

describe('getDocumentUrl', () => {
  test('prefers presigned URLs when available', () => {
    expect(
      getDocumentUrl({
        s3_key: 'folder/report #1?.pdf',
        presigned_url: 'https://example.test/fallback.pdf?download=1',
      }),
    ).toBe('https://example.test/fallback.pdf?download=1')
  })

  test('falls back to encoded S3 proxy URLs when no presigned URL is present', () => {
    expect(
      getDocumentUrl({
        s3_key: 'folder/report #1?.pdf',
        presigned_url: null,
      }),
    ).toBe(`${TEST_BACKEND_ORIGIN}/s3/content/folder%2Freport%20%231%3F.pdf`)
  })

  test('returns null when no browser-safe document URL is available', () => {
    expect(
      getDocumentUrl({
        s3_key: null,
        presigned_url: null,
      }),
    ).toBeNull()
  })
})
