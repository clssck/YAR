import '../setup'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'

const mockGetDocumentsPaginated = mock(async (request?: unknown) => {
  void request
  return {
    documents: [
      {
        id: 'doc-1',
        file_path: '/docs/doc-1.txt',
        content_summary: 'Example document',
        content_length: 42,
        status: 'processed',
        created_at: '2024-01-01T00:00:00.000Z',
        updated_at: '2024-01-02T00:00:00.000Z',
        chunks_count: 1,
        metadata: {},
      },
    ],
    pagination: {
      page: 1,
      page_size: 10,
      total_count: 1,
      total_pages: 1,
      has_next: false,
      has_prev: false,
    },
    status_counts: { all: 1, processed: 1 },
  }
})

const mockScanNewDocuments = mock(async () => ({
  status: 'success',
  message: 'ok',
  track_id: 'track-1',
}))

const mockReprocessFailedDocuments = mock(async () => ({
  status: 'success',
  message: 'ok',
}))

const mockBackendState = {
  resetHealthCheckTimerDelayed: mock(() => {}),
  check: mock(async () => true),
}

const mockSettingsState = {
  currentTab: 'documents',
  showFileName: false,
  documentsPageSize: 10,
}

mock.module('@/api/yar', () => ({
  getDocumentsPaginated: mockGetDocumentsPaginated,
  scanNewDocuments: mockScanNewDocuments,
  reprocessFailedDocuments: mockReprocessFailedDocuments,
}))

mock.module('@/hooks/useBreakpoint', () => ({
  useResponsive: () => ({ isMobile: false, isTablet: false, isDesktop: true }),
}))

mock.module('@/stores/settings', () => ({
  useSettingsStore: {
    use: {
      currentTab: () => mockSettingsState.currentTab,
      showFileName: () => mockSettingsState.showFileName,
      setShowFileName: () => mock(() => {}),
      documentsPageSize: () => mockSettingsState.documentsPageSize,
      setDocumentsPageSize: () => mock(() => {}),
    },
    getState: () => ({ apiKey: null }),
  },
}))

mock.module('@/stores/state', () => ({
  useBackendState: {
    use: {
      health: () => true,
      pipelineBusy: () => false,
    },
    getState: () => mockBackendState,
  },
}))

mock.module('@/components/documents/ClearDocumentsDialog', () => ({ default: () => null }))
mock.module('@/components/documents/DeleteDocumentsDialog', () => ({ default: () => null }))
mock.module('@/components/documents/PipelineStatusDialog', () => ({ default: () => null }))
mock.module('@/components/documents/UploadDocumentsDialog', () => ({ default: () => null }))

const DocumentManager = (await import('@/features/DocumentManager')).default

describe('DocumentManager Rendering', () => {
  beforeEach(() => {
    mockGetDocumentsPaginated.mockClear()
    mockScanNewDocuments.mockClear()
    mockReprocessFailedDocuments.mockClear()
    mockBackendState.resetHealthCheckTimerDelayed.mockClear()
    mockBackendState.check.mockClear()
    mockSettingsState.currentTab = 'documents'
    mockSettingsState.showFileName = false
    mockSettingsState.documentsPageSize = 10
  })

  afterEach(() => {
    cleanup()
  })

  test('mounts and fetches paginated documents once on load', async () => {
    let rendered: ReturnType<typeof render>
    await act(async () => {
      rendered = render(<DocumentManager />)
    })

    await waitFor(() => {
      expect(mockGetDocumentsPaginated).toHaveBeenCalledTimes(1)
    })

    expect(mockGetDocumentsPaginated).toHaveBeenLastCalledWith(
      expect.objectContaining({
        page: 1,
        page_size: 10,
        sort_field: 'updated_at',
        sort_direction: 'desc',
      }),
    )
    expect(rendered!.getByText('doc-1')).toBeTruthy()
  })

  test('refetches page 1 when the active sort direction changes', async () => {
    let rendered: ReturnType<typeof render>
    await act(async () => {
      rendered = render(<DocumentManager />)
    })

    await waitFor(() => {
      expect(mockGetDocumentsPaginated).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(rendered!.getByText('Updated').closest('th') ?? rendered!.getByText('Updated'))

    await waitFor(() => {
      expect(mockGetDocumentsPaginated).toHaveBeenCalledTimes(2)
    })

    expect(mockGetDocumentsPaginated).toHaveBeenLastCalledWith(
      expect.objectContaining({
        page: 1,
        page_size: 10,
        sort_field: 'updated_at',
        sort_direction: 'asc',
      }),
    )
  })
})
