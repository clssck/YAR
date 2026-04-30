import type { Message } from '@/api/yar'

export type MessageWithError = Message & {
  /** Unique identifier for stable React keys */
  id: string
  isError?: boolean
  /** Error categorization for retry/UX decisions */
  errorType?: 'timeout' | 'auth' | 'server' | 'network' | 'unknown'
  /** True while streaming the model's "thinking" prelude. */
  isThinking?: boolean
  /** Unix timestamp when the message was created */
  timestamp?: number
  /**
   * Indicates if the mermaid diagram in this message has been rendered.
   * Used to persist the rendering state across updates and prevent flickering.
   */
  mermaidRendered?: boolean
  /**
   * Indicates if the LaTeX formulas in this message are complete and ready for rendering.
   * Used to prevent red error text during streaming of incomplete LaTeX formulas.
   */
  latexRendered?: boolean
}
