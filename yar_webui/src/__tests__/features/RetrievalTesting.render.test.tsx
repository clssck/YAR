import '../setup'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, fireEvent, render, waitFor } from '@testing-library/react'
import {
  memo,
  type FormEventHandler,
  type InputHTMLAttributes,
  type ReactNode,
  type TextareaHTMLAttributes,
} from 'react'
import type { CitationsMetadata } from '@/api/yar'

const mockQueryText = mock(async () => ({ response: '', references: [] }))
const mockQueryTextStream = mock(
  async (
    _request: unknown,
    onChunk: (chunk: string) => void,
    onError?: (error: string) => void,
    onCitations?: (metadata: CitationsMetadata) => void,
  ) => {
    onChunk('Partial answer')
    onError?.('Stream failed')
    onCitations?.({
      markers: [],
      sources: [],
      footnotes: [],
      uncited_count: 0,
    })
  },
)
const chatMessageRenderCounts = new Map<string, number>()

mock.module('sonner', () => ({
  toast: {
    success: mock(() => {}),
    error: mock(() => {}),
  },
}))

// Bun mock note: spreading the realYar namespace into the factory causes
// the mock to leak through `export *` and override the underlying yarImpl
// module too. So enumerate the specific exports downstream consumers
// (ChatMessageImpl getDocumentUrl, KeyboardShortcutHelp checkHealth via
// store side-effects) actually pull. The test never exercises these paths,
// so cheap stubs are sufficient.
mock.module('@/api/yar', () => ({
  queryText: mockQueryText,
  queryTextStream: mockQueryTextStream,
  checkHealth: () => Promise.resolve({ status: 'healthy' }),
  getDocumentUrl: () => null,
  getDocumentsPaginated: () => Promise.resolve({ documents: [], pagination: {} }),
  scanNewDocuments: () => Promise.resolve({ status: 'success' }),
  reprocessFailedDocuments: () => Promise.resolve({ status: 'success' }),
}))


mock.module('@/components/ui/Input', () => ({
  default: ({
    onChange,
    ...props
  }: InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} onInput={onChange as unknown as FormEventHandler<HTMLInputElement>} />
  ),
}))

mock.module('@/components/ui/Textarea', () => ({
  default: ({
    onChange,
    ...props
  }: TextareaHTMLAttributes<HTMLTextAreaElement>) => (
    <textarea
      {...props}
      onInput={onChange as unknown as FormEventHandler<HTMLTextAreaElement>}
    />
  ),
}))

mock.module('@/components/ui/Popover', () => ({
  Popover: ({ children }: { children?: ReactNode }) => <>{children}</>,
  PopoverTrigger: ({ children }: { children?: ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children?: ReactNode }) => <>{children}</>,
}))

mock.module('@/components/retrieval/QuerySettings', () => ({
  default: () => null,
}))

mock.module('@/components/retrieval/ChatMessage', () => ({
  ChatMessage: memo(
    ({
      message,
    }: {
      message: {
        id: string
        role: 'user' | 'assistant' | 'system'
        content: string
        displayContent?: string
        isError?: boolean
      }
    }) => {
      chatMessageRenderCounts.set(
        message.id,
        (chatMessageRenderCounts.get(message.id) ?? 0) + 1,
      )

      return (
        <div
          data-testid={`message-${message.role}`}
          data-message-id={message.id}
          data-error={message.isError ? 'true' : 'false'}
        >
          {message.displayContent ?? message.content}
        </div>
      )
    },
  ),
}))

const setElementValue = (
  element: HTMLInputElement | HTMLTextAreaElement,
  value: string,
) => {
  const prototype =
    element instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype
  const valueSetter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set
  valueSetter?.call(element, value)
  fireEvent.input(element, { target: { value } })
}

const submitQuery = async (
  rendered: ReturnType<typeof render>,
  query: string,
): Promise<void> => {
  const input = rendered.container.querySelector('#query-input') as
    | HTMLInputElement
    | HTMLTextAreaElement
    | null
  const form = rendered.container.querySelector('form') as HTMLFormElement | null

  expect(input).toBeTruthy()
  expect(form).toBeTruthy()

  await act(async () => {
    if (input) {
      setElementValue(input, query)
    }
  })
  await waitFor(() => {
    expect(input?.value).toBe(query)
  })
  await act(async () => {
    fireEvent.submit(form!)
  })
  await waitFor(() => {
    expect(mockQueryTextStream).toHaveBeenCalledTimes(1)
  })
}

const RetrievalTesting = (await import('@/features/RetrievalTesting')).default
const { useSettingsStore } = await import('@/stores/settings')

const defaultQuerySettings = { ...useSettingsStore.getState().querySettings }

describe('RetrievalTesting streaming errors', () => {
  beforeEach(() => {
    localStorage.clear()
    chatMessageRenderCounts.clear()
    mockQueryText.mockClear()
    mockQueryTextStream.mockClear()
    mockQueryTextStream.mockImplementation(
      async (
        _request: unknown,
        onChunk: (chunk: string) => void,
        onError?: (error: string) => void,
      ) => {
        onChunk('Partial answer')
        onError?.('Stream failed')
      },
    )
    useSettingsStore.setState({
      currentTab: 'retrieval',
      retrievalHistory: [],
      querySettings: {
        ...defaultQuerySettings,
        stream: true,
      },
    })
  })

  afterEach(() => {
    useSettingsStore.setState({
      retrievalHistory: [],
      querySettings: {
        ...defaultQuerySettings,
      },
    })
  })

  test('coalesces multi-chunk streaming into the final assistant content', async () => {
    mockQueryTextStream.mockImplementation(
      async (_request: unknown, onChunk: (chunk: string) => void) => {
        onChunk('<thi')
        onChunk('nk>Planning')
        onChunk(' across chunks</th')
        onChunk('ink>Final answer')
      },
    )

    const rendered = render(<RetrievalTesting />)
    await submitQuery(rendered, 'What is AI?')

    await waitFor(() => {
      const assistantMessage = rendered.container.querySelector(
        '[data-testid="message-assistant"]',
      )
      expect(assistantMessage?.textContent).toBe('Final answer')
    })

    const userMessage = rendered.container.querySelector(
      '[data-testid="message-user"]',
    )
    expect(userMessage).toBeTruthy()
    const userMessageId = userMessage?.getAttribute('data-message-id')
    expect(userMessageId).toBeTruthy()
    expect(chatMessageRenderCounts.get(userMessageId!)).toBe(1)
  })

  test('applies citations against buffered streamed content', async () => {
    mockQueryTextStream.mockImplementation(
      async (
        _request: unknown,
        onChunk: (chunk: string) => void,
        _onError?: (error: string) => void,
        onCitations?: (metadata: CitationsMetadata) => void,
      ) => {
        onChunk('Answer')
        onChunk(' with support')
        onCitations?.({
          markers: [
            {
              marker: '[1]',
              insert_position: 6,
              reference_ids: ['ref-1'],
              confidence: 0.91,
              text_preview: 'Answer',
            },
          ],
          sources: [],
          footnotes: ['[1] Source.pdf'],
          uncited_count: 0,
        })
      },
    )

    const rendered = render(<RetrievalTesting />)
    await submitQuery(rendered, 'Need sources')

    await waitFor(() => {
      const assistantMessage = rendered.container.querySelector(
        '[data-testid="message-assistant"]',
      )
      expect(assistantMessage?.textContent).toBe(
        'Answer[1] with support\n\n---\n\n**References:**\n[1] Source.pdf',
      )
    })
  })

  test('appends stream errors without duplicating partial assistant content', async () => {
    const rendered = render(<RetrievalTesting />)
    await submitQuery(rendered, 'What is AI?')

    await waitFor(() => {
      const assistantMessage = rendered.container.querySelector(
        '[data-testid="message-assistant"]',
      )
      expect(assistantMessage).toBeTruthy()
      expect(assistantMessage?.textContent).toContain('Partial answer')
      expect(assistantMessage?.textContent).toContain('Stream failed')
      expect(assistantMessage?.getAttribute('data-error')).toBe('true')
      expect(assistantMessage?.textContent?.match(/Partial answer/g)?.length).toBe(1)
    })
  })
})
