import { beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, renderHook } from '@testing-library/react'
import type {
  ConfirmationConfig,
  UndoableActionConfig,
} from '../../hooks/useConfirmation'

const toastMainMock = mock(() => 'mock-toast-id')
const toastErrorMock = mock(() => 'error-toast-id')
const toastSuccessMock = mock(() => 'success-toast-id')
const toastLoadingMock = mock(() => 'loading-toast-id')
const toastDismissMock = mock(() => {})

const toastMock = Object.assign(toastMainMock, {
  error: toastErrorMock,
  success: toastSuccessMock,
  loading: toastLoadingMock,
  dismiss: toastDismissMock,
})

mock.module('sonner', () => ({
  toast: toastMock,
}))

const {
  useUndoableAction,
  useInlineConfirmation,
  createDelayedAction,
  createProgressToast,
  useProgressToast,
} = await import('../../hooks/useConfirmation')

describe('useUndoableAction', () => {
  beforeEach(() => {
    toastMainMock.mockClear()
    toastErrorMock.mockClear()
    toastSuccessMock.mockClear()
    toastLoadingMock.mockClear()
    toastDismissMock.mockClear()
  })

  test('hook initializes with executeWithUndo function', () => {
    const { result } = renderHook(() => useUndoableAction())
    expect(result.current).toBeDefined()
    expect(typeof result.current.executeWithUndo).toBe('function')
  })

  test('executes action and shows toast', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const actionMock = mock(() => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Action completed',
        action: actionMock,
      })
    })

    expect(actionMock).toHaveBeenCalled()
    expect(actionMock).toHaveBeenCalledTimes(1)
    expect(toastMainMock).toHaveBeenCalledWith(
      'Action completed',
      expect.any(Object),
    )
  })

  test('handles async action execution', async () => {
    const { result } = renderHook(() => useUndoableAction())
    let actionExecuted = false

    const asyncAction = async (): Promise<(() => void) | undefined> => {
      await new Promise((resolve) => setTimeout(resolve, 10))
      actionExecuted = true
      return undefined
    }

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Async action',
        action: asyncAction,
      })
    })

    expect(actionExecuted).toBe(true)
    expect(toastMainMock).toHaveBeenCalled()
  })

  test('stores cleanup function from action', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const cleanupMock = mock(() => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Action with cleanup',
        action: () => cleanupMock,
      })
    })

    expect(toastMainMock).toHaveBeenCalled()
  })

  test('handles action errors gracefully', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const errorMessage = 'Action failed'

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Failing action',
        action: () => {
          throw new Error(errorMessage)
        },
      })
    })

    expect(toastErrorMock).toHaveBeenCalledWith('Action failed', {
      description: errorMessage,
    })
  })

  test('handles non-Error exceptions', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Failing action',
        action: () => {
          throw 'string error'
        },
      })
    })

    expect(toastErrorMock).toHaveBeenCalledWith('Action failed', {
      description: 'An error occurred',
    })
  })

  test('cancels previous pending action when new one starts', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const cleanup1Mock = mock(() => {})
    const cleanup2Mock = mock(() => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'First action',
        action: () => cleanup1Mock,
      })
    })

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Second action',
        action: () => cleanup2Mock,
      })
    })

    expect(toastMainMock).toHaveBeenCalledTimes(2)
  })

  test('supports custom toast config with description', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Custom toast',
        description: 'Custom description',
        action: () => {},
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Custom toast',
      expect.objectContaining({
        description: 'Custom description',
      }),
    )
  })

  test('supports custom duration for toast', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Long duration',
        duration: 10000,
        action: () => {},
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Long duration',
      expect.objectContaining({
        duration: 10000,
      }),
    )
  })

  test('uses default duration (5000ms) if not provided', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Default duration',
        action: () => {},
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Default duration',
      expect.objectContaining({
        duration: 5000,
      }),
    )
  })

  test('includes undo action if onUndo is provided', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const onUndoMock = mock(async () => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'With undo',
        action: () => {},
        onUndo: onUndoMock,
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'With undo',
      expect.objectContaining({
        action: expect.objectContaining({
          label: 'Undo',
        }),
      }),
    )
  })

  test('does not add undo action if onUndo is not provided', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'No undo',
        action: () => {},
      })
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      Record<string, unknown>,
    ]
    expect(call[1].action).toBeUndefined()
  })

  test('undo callback shows success toast on completion', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const onUndoMock = mock(async () => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'With undo',
        action: () => {},
        onUndo: onUndoMock,
      })
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      { action: { onClick: () => Promise<void> } },
    ]
    const undoAction = call[1].action
    await act(async () => {
      await undoAction.onClick()
    })

    expect(onUndoMock).toHaveBeenCalled()
    expect(toastSuccessMock).toHaveBeenCalledWith('Action undone')
  })

  test('undo callback shows error toast on failure', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const undoError = new Error('Undo failed')
    const onUndoMock = mock(async () => {
      throw undoError
    })

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'With undo',
        action: () => {},
        onUndo: onUndoMock,
      })
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      { action: { onClick: () => Promise<void> } },
    ]
    const undoAction = call[1].action
    await act(async () => {
      await undoAction.onClick()
    })

    expect(toastErrorMock).toHaveBeenCalledWith('Failed to undo', {
      description: 'Undo failed',
    })
  })

  test('marks action as executed after completion', async () => {
    const { result } = renderHook(() => useUndoableAction())
    let isExecuted = false

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Track execution',
        action: () => {
          isExecuted = true
        },
      })
    })

    expect(isExecuted).toBe(true)
  })

  test('handles async cleanup function', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const cleanupMock = mock(() => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Async cleanup',
        action: async () => {
          await new Promise((resolve) => setTimeout(resolve, 10))
          return cleanupMock
        },
      })
    })

    expect(toastMainMock).toHaveBeenCalled()
  })

  test('supports icon in toast config', async () => {
    const { result } = renderHook(() => useUndoableAction())

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'With icon',
        icon: 'test-icon',
        action: () => {},
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'With icon',
      expect.objectContaining({
        icon: 'test-icon',
      }),
    )
  })
})

describe('useInlineConfirmation', () => {
  beforeEach(() => {
    toastMainMock.mockClear()
  })

  test('hook initializes with confirm function', () => {
    const { result } = renderHook(() => useInlineConfirmation())
    expect(result.current).toBeDefined()
    expect(typeof result.current.confirm).toBe('function')
  })

  test('confirm function returns a promise', () => {
    const { result } = renderHook(() => useInlineConfirmation())
    const confirmPromise = result.current.confirm({
      title: 'Test confirmation',
    })
    expect(confirmPromise).toBeInstanceOf(Promise)
  })

  test('uses default confirm and cancel text', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    act(() => {
      result.current.confirm({
        title: 'Test confirmation',
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Test confirmation',
      expect.objectContaining({
        action: expect.objectContaining({
          label: 'Confirm',
        }),
        cancel: expect.objectContaining({
          label: 'Cancel',
        }),
      }),
    )
  })

  test('uses custom confirm and cancel text', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    act(() => {
      result.current.confirm({
        title: 'Delete item?',
        confirmText: 'Delete',
        cancelText: 'Keep',
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Delete item?',
      expect.objectContaining({
        action: expect.objectContaining({
          label: 'Delete',
        }),
        cancel: expect.objectContaining({
          label: 'Keep',
        }),
      }),
    )
  })

  test('supports custom description', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    act(() => {
      result.current.confirm({
        title: 'Confirm action',
        description: 'This action cannot be undone',
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Confirm action',
      expect.objectContaining({
        description: 'This action cannot be undone',
      }),
    )
  })

  test('resolves to true when confirm is clicked', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    const confirmPromise = result.current.confirm({
      title: 'Confirm?',
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      { action: { onClick: () => void } },
    ]
    const confirmAction = call[1].action
    act(() => {
      confirmAction.onClick()
    })

    await expect(confirmPromise).resolves.toBe(true)
  })

  test('resolves to false when cancel is clicked', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    const confirmPromise = result.current.confirm({
      title: 'Confirm?',
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      { cancel: { onClick: () => void } },
    ]
    const cancelAction = call[1].cancel
    act(() => {
      cancelAction.onClick()
    })

    await expect(confirmPromise).resolves.toBe(false)
  })

  test('resolves to false when dismissed', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    const confirmPromise = result.current.confirm({
      title: 'Confirm?',
    })

    const call = toastMainMock.mock.calls[0] as unknown as [
      string,
      { onDismiss: () => void },
    ]
    const onDismiss = call[1].onDismiss
    act(() => {
      onDismiss()
    })

    await expect(confirmPromise).resolves.toBe(false)
  })

  test('includes all configuration options', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    act(() => {
      result.current.confirm({
        title: 'Delete all?',
        description: 'Irreversible',
        confirmText: 'Delete All',
        cancelText: 'Cancel',
        variant: 'destructive',
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Delete all?',
      expect.objectContaining({
        description: 'Irreversible',
        duration: 10000,
      }),
    )
  })

  test('uses 10000ms duration for confirmation', () => {
    const { result } = renderHook(() => useInlineConfirmation())

    act(() => {
      result.current.confirm({
        title: 'Confirm',
      })
    })

    expect(toastMainMock).toHaveBeenCalledWith(
      'Confirm',
      expect.objectContaining({
        duration: 10000,
      }),
    )
  })

  test('multiple confirmations can be created independently', async () => {
    const { result } = renderHook(() => useInlineConfirmation())

    const promise1 = result.current.confirm({
      title: 'First confirmation',
    })

    const promise2 = result.current.confirm({
      title: 'Second confirmation',
    })

    expect(promise1).toBeInstanceOf(Promise)
    expect(promise2).toBeInstanceOf(Promise)
    expect(promise1).not.toBe(promise2)
    expect(toastMainMock).toHaveBeenCalledTimes(2)
  })
})

describe('createDelayedAction', () => {
  test('creates a delayed action object', () => {
    const action = createDelayedAction(() => {}, 1000)
    expect(action).toBeDefined()
    expect(typeof action.start).toBe('function')
    expect(typeof action.cancel).toBe('function')
    expect(typeof action.isExecuted).toBe('function')
    expect(typeof action.isPending).toBe('function')
  })

  test('executes action after delay', async () => {
    const actionMock = mock(() => {})
    const action = createDelayedAction(actionMock, 50)

    action.start()
    expect(action.isPending()).toBe(true)

    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(action.isExecuted()).toBe(true)
    expect(actionMock).toHaveBeenCalledTimes(1)
  })

  test('does not execute if already executed', async () => {
    const actionMock = mock(() => {})
    const action = createDelayedAction(actionMock, 10)

    action.start()
    await new Promise((resolve) => setTimeout(resolve, 50))

    action.start()
    await new Promise((resolve) => setTimeout(resolve, 10))

    expect(actionMock).toHaveBeenCalledTimes(1)
  })

  test('cancels pending action', async () => {
    const actionMock = mock(() => {})
    const action = createDelayedAction(actionMock, 100)

    action.start()
    expect(action.isPending()).toBe(true)

    action.cancel()
    expect(action.isPending()).toBe(false)

    await new Promise((resolve) => setTimeout(resolve, 150))
    expect(actionMock).not.toHaveBeenCalled()
  })

  test('isPending returns false initially', () => {
    const action = createDelayedAction(() => {}, 1000)
    expect(action.isPending()).toBe(false)
  })

  test('isExecuted returns false initially', () => {
    const action = createDelayedAction(() => {}, 1000)
    expect(action.isExecuted()).toBe(false)
  })

  test('isExecuted returns true after execution', async () => {
    const action = createDelayedAction(() => {}, 20)

    action.start()
    expect(action.isExecuted()).toBe(false)

    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(action.isExecuted()).toBe(true)
  })

  test('handles async actions', async () => {
    let asyncComplete = false
    const asyncAction = async () => {
      await new Promise((resolve) => setTimeout(resolve, 10))
      asyncComplete = true
    }

    const action = createDelayedAction(asyncAction, 20)
    action.start()

    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(asyncComplete).toBe(true)
  })

  test('cancel does nothing if no timeout is pending', () => {
    const action = createDelayedAction(() => {}, 1000)
    expect(() => action.cancel()).not.toThrow()
  })

  test('multiple cancel calls are safe', async () => {
    const action = createDelayedAction(() => {}, 100)
    action.start()
    action.cancel()
    action.cancel()
    action.cancel()

    await new Promise((resolve) => setTimeout(resolve, 150))
    expect(action.isPending()).toBe(false)
  })

  test('handles action that throws error', async () => {
    const action = createDelayedAction(() => {
      throw new Error('Action error')
    }, 20)

    action.start()
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(action.isExecuted()).toBe(true)
  })
})

describe('createProgressToast', () => {
  beforeEach(() => {
    toastMainMock.mockClear()
    toastErrorMock.mockClear()
    toastSuccessMock.mockClear()
    toastLoadingMock.mockClear()
    toastDismissMock.mockClear()
  })

  test('creates a progress toast object', () => {
    const progress = createProgressToast({
      message: 'Loading...',
    })

    expect(progress).toBeDefined()
    expect(typeof progress.update).toBe('function')
    expect(typeof progress.success).toBe('function')
    expect(typeof progress.error).toBe('function')
    expect(typeof progress.dismiss).toBe('function')
    expect(progress.id).toBe('loading-toast-id')
  })

  test('calls toast.loading with message', () => {
    createProgressToast({
      message: 'Processing...',
    })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Processing...',
      expect.any(Object),
    )
  })

  test('includes optional description', () => {
    createProgressToast({
      message: 'Uploading files...',
      description: '0 of 10 files',
    })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Uploading files...',
      expect.objectContaining({
        description: '0 of 10 files',
      }),
    )
  })

  test('accepts toast options', () => {
    createProgressToast({
      message: 'Processing...',
      options: {
        duration: 10000,
      },
    })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Processing...',
      expect.objectContaining({
        duration: 10000,
      }),
    )
  })

  test('update method calls toast.loading with id', () => {
    const progress = createProgressToast({
      message: 'Initial message',
    })

    progress.update({ message: 'Updated message' })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Updated message',
      expect.objectContaining({
        id: 'loading-toast-id',
      }),
    )
  })

  test('update method works with description', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.update({ description: '50 of 100' })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      undefined,
      expect.objectContaining({
        id: 'loading-toast-id',
        description: '50 of 100',
      }),
    )
  })

  test('update method works with both message and description', () => {
    const progress = createProgressToast({
      message: 'Starting...',
    })

    progress.update({ message: 'In progress...', description: '1 of 10' })

    expect(toastLoadingMock).toHaveBeenCalledWith(
      'In progress...',
      expect.objectContaining({
        id: 'loading-toast-id',
        description: '1 of 10',
      }),
    )
  })

  test('success method calls toast.success', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.success('Completed!')

    expect(toastSuccessMock).toHaveBeenCalledWith('Completed!', {
      id: 'loading-toast-id',
      description: undefined,
    })
  })

  test('success method with description', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.success('All done', 'All 10 items processed')

    expect(toastSuccessMock).toHaveBeenCalledWith('All done', {
      id: 'loading-toast-id',
      description: 'All 10 items processed',
    })
  })

  test('error method shows error state', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.error('Failed!')

    expect(toastErrorMock).toHaveBeenCalledWith('Failed!', {
      id: 'loading-toast-id',
      description: undefined,
    })
  })

  test('error method with description', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.error('Operation failed', 'Network timeout')

    expect(toastErrorMock).toHaveBeenCalledWith('Operation failed', {
      id: 'loading-toast-id',
      description: 'Network timeout',
    })
  })

  test('dismiss method calls toast.dismiss', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    progress.dismiss()

    expect(toastDismissMock).toHaveBeenCalledWith('loading-toast-id')
  })

  test('toast id is accessible', () => {
    const progress = createProgressToast({
      message: 'Processing...',
    })

    expect(progress.id).toBe('loading-toast-id')
  })

  test('multiple progress toasts have different ids', () => {
    toastLoadingMock
      .mockReturnValueOnce('toast-1')
      .mockReturnValueOnce('toast-2')

    const progress1 = createProgressToast({
      message: 'Process 1...',
    })

    const progress2 = createProgressToast({
      message: 'Process 2...',
    })

    expect(progress1.id).toBe('toast-1')
    expect(progress2.id).toBe('toast-2')
  })
})

describe('useProgressToast', () => {
  beforeEach(() => {
    toastMainMock.mockClear()
    toastLoadingMock.mockClear()
  })

  test('hook initializes with startProgress function', () => {
    const { result } = renderHook(() => useProgressToast())
    expect(result.current).toBeDefined()
    expect(typeof result.current.startProgress).toBe('function')
  })

  test('startProgress returns a progress toast object', () => {
    const { result } = renderHook(() => useProgressToast())

    const progress = result.current.startProgress({
      message: 'Loading...',
    })

    expect(progress).toBeDefined()
    expect(typeof progress.update).toBe('function')
    expect(typeof progress.success).toBe('function')
  })

  test('startProgress with description', () => {
    const { result } = renderHook(() => useProgressToast())

    const progress = result.current.startProgress({
      message: 'Uploading...',
      description: '0 of 10',
    })

    expect(progress.id).toBe('loading-toast-id')
    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Uploading...',
      expect.objectContaining({
        description: '0 of 10',
      }),
    )
  })

  test('startProgress with options', () => {
    const { result } = renderHook(() => useProgressToast())

    const progress = result.current.startProgress({
      message: 'Processing...',
      options: {
        duration: 5000,
      },
    })

    expect(progress).toBeDefined()
    expect(toastLoadingMock).toHaveBeenCalledWith(
      'Processing...',
      expect.objectContaining({
        duration: 5000,
      }),
    )
  })

  test('multiple startProgress calls are independent', () => {
    const { result } = renderHook(() => useProgressToast())

    toastLoadingMock
      .mockReturnValueOnce('toast-1')
      .mockReturnValueOnce('toast-2')

    const progress1 = result.current.startProgress({
      message: 'Task 1...',
    })

    const progress2 = result.current.startProgress({
      message: 'Task 2...',
    })

    expect(progress1.id).toBe('toast-1')
    expect(progress2.id).toBe('toast-2')
  })

  test('startProgress callback is stable', () => {
    const { result, rerender } = renderHook(() => useProgressToast())
    const firstCallback = result.current.startProgress

    rerender()
    const secondCallback = result.current.startProgress

    expect(firstCallback).toBe(secondCallback)
  })
})

describe('Integration scenarios', () => {
  beforeEach(() => {
    toastMainMock.mockClear()
    toastErrorMock.mockClear()
    toastSuccessMock.mockClear()
    toastLoadingMock.mockClear()
    toastDismissMock.mockClear()
  })

  test('progress toast workflow for bulk operation', () => {
    const { result } = renderHook(() => useProgressToast())

    const progress = result.current.startProgress({
      message: 'Deleting items...',
      description: '0 of 5 items',
    })

    for (let i = 1; i <= 5; i++) {
      progress.update({
        description: `${i} of 5 items deleted`,
      })
    }

    progress.success('All items deleted successfully')

    expect(toastLoadingMock).toHaveBeenCalled()
    expect(toastSuccessMock).toHaveBeenCalledWith(
      'All items deleted successfully',
      {
        id: 'loading-toast-id',
        description: undefined,
      },
    )
  })

  test('delayed action for soft delete', async () => {
    const actionMock = mock(() => {})
    const deleteAction = createDelayedAction(actionMock, 30)

    deleteAction.start()
    expect(deleteAction.isPending()).toBe(true)

    deleteAction.cancel()
    expect(deleteAction.isPending()).toBe(false)

    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(actionMock).not.toHaveBeenCalled()
  })

  test('delayed action allows timeout to execute', async () => {
    const actionMock = mock(() => {})
    const deleteAction = createDelayedAction(actionMock, 30)

    deleteAction.start()
    expect(deleteAction.isPending()).toBe(true)

    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(deleteAction.isExecuted()).toBe(true)
    expect(actionMock).toHaveBeenCalled()
  })

  test('undoable action with cleanup flow', async () => {
    const { result } = renderHook(() => useUndoableAction())
    const cleanupMock = mock(() => {})

    await act(async () => {
      await result.current.executeWithUndo({
        message: 'Item deleted',
        action: () => cleanupMock,
      })
    })

    expect(toastMock).toHaveBeenCalled()
  })
})

describe('Type safety', () => {
  test('UndoableActionConfig types are correct', () => {
    const config: UndoableActionConfig = {
      message: 'Test',
      description: 'Test description',
      action: async () => undefined,
      onUndo: async () => {},
      duration: 5000,
    }

    expect(config.message).toBe('Test')
  })

  test('ConfirmationConfig types are correct', () => {
    const config: ConfirmationConfig = {
      title: 'Confirm',
      description: 'Are you sure?',
      confirmText: 'Yes',
      cancelText: 'No',
      variant: 'destructive',
    }

    expect(config.title).toBe('Confirm')
  })
})
