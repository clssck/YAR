import { useCallback, useRef } from 'react'
import { type ExternalToast, toast } from 'sonner'

export type UndoableActionConfig = {
  /** Message to show in the toast */
  message: string
  /** Description text below the message */
  description?: string
  /** The action to perform (can return a cleanup function) */
  action: () => void | (() => void) | Promise<undefined | (() => void)>
  /** Undo handler - called if user clicks undo within the timeout */
  onUndo?: () => void | Promise<void>
  /** Duration in ms before action becomes permanent (default: 5000) */
  duration?: number
  /** Icon to show in toast */
  icon?: React.ReactNode
}

/**
 * Hook for executing actions with an undo window.
 * Shows a toast with an "Undo" button that allows reverting the action.
 *
 * @example
 * ```tsx
 * const { executeWithUndo } = useUndoableAction()
 *
 * const handleDelete = () => {
 *   executeWithUndo({
 *     message: 'Document deleted',
 *     action: async () => {
 *       await deleteDocument(id)
 *     },
 *     onUndo: async () => {
 *       await restoreDocument(id)
 *     }
 *   })
 * }
 * ```
 */
export function useUndoableAction() {
  const pendingActionRef = useRef<{
    cleanup?: () => void
    executed: boolean
  } | null>(null)

  const executeWithUndo = useCallback(
    async ({
      message,
      description,
      action,
      onUndo,
      duration = 5000,
      icon,
    }: UndoableActionConfig) => {
      // Cancel any pending action
      if (pendingActionRef.current && !pendingActionRef.current.executed) {
        pendingActionRef.current.cleanup?.()
      }

      // Track this action
      const actionState = {
        cleanup: undefined as (() => void) | undefined,
        executed: false,
      }
      pendingActionRef.current = actionState

      // Execute the action
      try {
        const cleanup = await action()
        if (typeof cleanup === 'function') {
          actionState.cleanup = cleanup
        }
        actionState.executed = true
      } catch (error) {
        toast.error('Action failed', {
          description:
            error instanceof Error ? error.message : 'An error occurred',
        })
        return
      }

      // Show toast with undo option
      toast(message, {
        description,
        icon,
        duration,
        action: onUndo
          ? {
              label: 'Undo',
              onClick: async () => {
                try {
                  await onUndo()
                  toast.success('Action undone')
                } catch (error) {
                  toast.error('Failed to undo', {
                    description:
                      error instanceof Error
                        ? error.message
                        : 'An error occurred',
                  })
                }
              },
            }
          : undefined,
      })
    },
    [],
  )

  return { executeWithUndo }
}

export type ConfirmationConfig = {
  /** Title of the confirmation dialog */
  title: string
  /** Description/question to show */
  description?: string
  /** Text for the confirm button (default: "Confirm") */
  confirmText?: string
  /** Text for the cancel button (default: "Cancel") */
  cancelText?: string
  /** Variant for the confirm button */
  variant?: 'default' | 'destructive'
}

/**
 * Hook for simple inline confirmations using toast.
 * Use this for non-critical confirmations that don't need a modal.
 *
 * @example
 * ```tsx
 * const { confirm } = useInlineConfirmation()
 *
 * const handleClear = async () => {
 *   const confirmed = await confirm({
 *     title: 'Clear all items?',
 *     description: 'This action cannot be undone.',
 *     confirmText: 'Clear All',
 *     variant: 'destructive'
 *   })
 *
 *   if (confirmed) {
 *     await clearItems()
 *   }
 * }
 * ```
 */
export function useInlineConfirmation() {
  const confirm = useCallback(
    ({
      title,
      description,
      confirmText = 'Confirm',
      cancelText = 'Cancel',
    }: ConfirmationConfig): Promise<boolean> => {
      return new Promise((resolve) => {
        toast(title, {
          description,
          duration: 10000, // Longer duration for confirmation
          action: {
            label: confirmText,
            onClick: () => resolve(true),
          },
          cancel: {
            label: cancelText,
            onClick: () => resolve(false),
          },
          onDismiss: () => resolve(false),
        })
      })
    },
    [],
  )

  return { confirm }
}

/**
 * Helper to create a delayed action that can be cancelled.
 * Useful for implementing "soft delete" patterns.
 *
 * @example
 * ```tsx
 * const deleteAction = createDelayedAction(
 *   () => permanentlyDelete(id),
 *   5000
 * )
 *
 * // Start the action
 * deleteAction.start()
 *
 * // Cancel before it executes
 * deleteAction.cancel()
 * ```
 */
export function createDelayedAction(
  action: () => void | Promise<void>,
  delayMs: number,
) {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  let executed = false

  return {
    start: () => {
      if (executed) return
      timeoutId = setTimeout(async () => {
        executed = true
        try {
          await action()
        } catch {
          // Errors are silently caught - the action is still marked as executed
        }
      }, delayMs)
    },
    cancel: () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
    },
    isExecuted: () => executed,
    isPending: () => timeoutId !== null,
  }
}

// ==================== PROGRESS TOASTS ====================

export type ProgressToastConfig = {
  /** Initial message to show */
  message: string
  /** Description text */
  description?: string
  /** Toast options */
  options?: ExternalToast
}

/**
 * Create a progress toast that can be updated and dismissed.
 *
 * @example
 * ```tsx
 * const progress = createProgressToast({
 *   message: 'Uploading files...',
 *   description: '0 of 10 files'
 * })
 *
 * for (let i = 0; i < 10; i++) {
 *   await uploadFile(files[i])
 *   progress.update({
 *     message: 'Uploading files...',
 *     description: `${i + 1} of 10 files`
 *   })
 * }
 *
 * progress.success('Upload complete!')
 * ```
 */
export function createProgressToast({
  message,
  description,
  options,
}: ProgressToastConfig) {
  const toastId = toast.loading(message, {
    description,
    ...options,
  })

  return {
    /** Update the toast message and description */
    update: ({
      message,
      description,
    }: {
      message?: string
      description?: string
    }) => {
      toast.loading(message, {
        id: toastId,
        description,
      })
    },

    /** Complete the toast with a success message */
    success: (message: string, description?: string) => {
      toast.success(message, {
        id: toastId,
        description,
      })
    },

    /** Complete the toast with an error message */
    error: (message: string, description?: string) => {
      toast.error(message, {
        id: toastId,
        description,
      })
    },

    /** Dismiss the toast */
    dismiss: () => {
      toast.dismiss(toastId)
    },

    /** Get the toast ID for external management */
    id: toastId,
  }
}

/**
 * Hook for managing progress toasts.
 *
 * @example
 * ```tsx
 * const { startProgress } = useProgressToast()
 *
 * const handleBulkDelete = async () => {
 *   const progress = startProgress({
 *     message: 'Deleting items...',
 *     description: `0 of ${items.length}`
 *   })
 *
 *   for (let i = 0; i < items.length; i++) {
 *     await deleteItem(items[i])
 *     progress.update({
 *       description: `${i + 1} of ${items.length}`
 *     })
 *   }
 *
 *   progress.success('All items deleted')
 * }
 * ```
 */
export function useProgressToast() {
  const startProgress = useCallback((config: ProgressToastConfig) => {
    return createProgressToast(config)
  }, [])

  return { startProgress }
}
