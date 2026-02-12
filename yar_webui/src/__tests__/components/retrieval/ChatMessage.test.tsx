/**
 * Tests for ChatMessage component
 */
import '../../setup'
import { describe, expect, mock, test } from 'bun:test'
import { render } from '@testing-library/react'
import type { MessageWithError } from '@/components/retrieval/ChatMessage'

// Mock react-i18next
mock.module('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}))

// Mock useTheme hook
mock.module('@/hooks/useTheme', () => ({
  default: () => ({
    theme: 'light',
  }),
}))

// Mock mermaid
mock.module('mermaid', () => ({
  default: {
    initialize: mock(),
    run: mock(),
  },
}))

// Mock react-syntax-highlighter
mock.module('react-syntax-highlighter', () => ({
  Prism: ({ children }: { children: string }) => <pre>{children}</pre>,
}))

mock.module('react-syntax-highlighter/dist/cjs/styles/prism', () => ({
  oneDark: {},
  oneLight: {},
}))

// Mock rehype plugins
mock.module('rehype-raw', () => ({
  default: () => {},
}))

mock.module('rehype-react', () => ({
  default: () => {},
}))

mock.module('remark-gfm', () => ({
  default: () => {},
}))

mock.module('remark-math', () => ({
  default: () => {},
}))

mock.module('@/utils/remarkFootnotes', () => ({
  remarkFootnotes: () => {},
}))

// Mock CitationMarker
mock.module('@/components/retrieval/CitationMarker', () => ({
  CitationMarker: ({ marker }: { marker: string }) => (
    <span data-testid="citation">{marker}</span>
  ),
}))

// Import after mocks are set up
const { ChatMessage } = await import('@/components/retrieval/ChatMessage')

// Helper to create test messages
function createMessage(
  overrides: Partial<MessageWithError> = {},
): MessageWithError {
  return {
    id: 'test-message-1',
    role: 'assistant',
    content: 'Test message content',
    timestamp: Date.now(),
    ...overrides,
  }
}

describe('ChatMessage Component', () => {
  describe('Basic Rendering', () => {
    test('renders assistant message', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Hello, I am an assistant.',
      })

      const { container } = render(<ChatMessage message={message} />)

      // Should render without crashing
      expect(container).toBeDefined()
    })

    test('renders user message', () => {
      const message = createMessage({
        role: 'user',
        content: 'Hello, I am a user.',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders message with displayContent', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Full content here',
        displayContent: 'Display content shown to user',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })

  describe('Error States', () => {
    test('renders error message', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Error occurred',
        isError: true,
        errorType: 'server',
      })

      const { container } = render(<ChatMessage message={message} />)

      // Should render error state
      expect(container).toBeDefined()
    })

    test('renders timeout error', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Request timed out',
        isError: true,
        errorType: 'timeout',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders auth error', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Authentication failed',
        isError: true,
        errorType: 'auth',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders network error', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Network error',
        isError: true,
        errorType: 'network',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders error with retry callback', () => {
      const onRetry = mock()
      const message = createMessage({
        role: 'assistant',
        content: 'Error occurred',
        isError: true,
      })

      const { container } = render(
        <ChatMessage message={message} onRetry={onRetry} />,
      )

      expect(container).toBeDefined()
    })
  })

  describe('Thinking State', () => {
    test('renders thinking message', () => {
      const message = createMessage({
        role: 'assistant',
        content: '',
        isThinking: true,
        thinkingContent: 'Analyzing your question...',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders completed thinking with time', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Final response',
        isThinking: false,
        thinkingContent: 'I analyzed the data...',
        thinkingTime: 2500, // 2.5 seconds
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders message without thinking content', () => {
      const message = createMessage({
        role: 'assistant',
        content: 'Simple response without thinking',
        isThinking: false,
        thinkingContent: undefined,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })

  describe('Tab Active State', () => {
    test('renders with isTabActive true', () => {
      const message = createMessage()

      const { container } = render(
        <ChatMessage message={message} isTabActive={true} />,
      )

      expect(container).toBeDefined()
    })

    test('renders with isTabActive false', () => {
      const message = createMessage()

      const { container } = render(
        <ChatMessage message={message} isTabActive={false} />,
      )

      expect(container).toBeDefined()
    })
  })

  describe('Message Content Types', () => {
    test('renders empty content', () => {
      const message = createMessage({
        content: '',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders long content', () => {
      const message = createMessage({
        content: 'Lorem ipsum '.repeat(100),
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders content with special characters', () => {
      const message = createMessage({
        content: 'Special chars: <>&"\'`~!@#$%^&*()',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders content with newlines', () => {
      const message = createMessage({
        content: 'Line 1\nLine 2\nLine 3',
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })

  describe('Mermaid Rendering State', () => {
    test('renders with mermaidRendered false', () => {
      const message = createMessage({
        content: '```mermaid\ngraph TD\nA-->B\n```',
        mermaidRendered: false,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders with mermaidRendered true', () => {
      const message = createMessage({
        content: '```mermaid\ngraph TD\nA-->B\n```',
        mermaidRendered: true,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })

  describe('LaTeX Rendering State', () => {
    test('renders with latexRendered false', () => {
      const message = createMessage({
        content: 'Formula: $E = mc^2$',
        latexRendered: false,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders with latexRendered true', () => {
      const message = createMessage({
        content: 'Formula: $E = mc^2$',
        latexRendered: true,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })

  describe('Timestamp Display', () => {
    test('renders message with timestamp', () => {
      const message = createMessage({
        timestamp: Date.now(),
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })

    test('renders message without timestamp', () => {
      const message = createMessage({
        timestamp: undefined,
      })

      const { container } = render(<ChatMessage message={message} />)

      expect(container).toBeDefined()
    })
  })
})
