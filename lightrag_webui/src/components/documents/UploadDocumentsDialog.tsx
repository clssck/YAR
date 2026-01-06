import { CheckCircle2, ChevronDown, ChevronRight, UploadIcon, XCircle } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import type { FileRejection } from 'react-dropzone'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { type ChunkingPreset, uploadDocument } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/Dialog'
import FileUploader from '@/components/ui/FileUploader'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'
import { cn, errorMessage } from '@/lib/utils'

interface UploadDocumentsDialogProps {
  onDocumentsUploaded?: () => Promise<void>
  /** Controlled open state */
  open?: boolean
  /** Callback when open state changes */
  onOpenChange?: (open: boolean) => void
}

type UploadPhase = 'idle' | 'uploading' | 'complete'

export default function UploadDocumentsDialog({
  onDocumentsUploaded,
  open: controlledOpen,
  onOpenChange,
}: UploadDocumentsDialogProps) {
  const { t } = useTranslation()
  const [internalOpen, setInternalOpen] = useState(false)

  // Support both controlled and uncontrolled modes
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const setOpen = isControlled ? (onOpenChange ?? (() => {})) : setInternalOpen
  const [isUploading, setIsUploading] = useState(false)
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>('idle')
  const [progresses, setProgresses] = useState<Record<string, number>>({})
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [chunkingPreset, setChunkingPreset] = useState<ChunkingPreset>('semantic')

  // Track upload statistics
  const [uploadStats, setUploadStats] = useState({
    total: 0,
    completed: 0,
    failed: 0,
    currentFile: '',
  })

  // Calculate overall progress percentage
  const overallProgress = useMemo(() => {
    if (uploadStats.total === 0) return 0
    return Math.round((uploadStats.completed + uploadStats.failed) / uploadStats.total * 100)
  }, [uploadStats])

  const handleRejectedFiles = useCallback(
    (rejectedFiles: FileRejection[]) => {
      // Process rejected files and add them to fileErrors
      rejectedFiles.forEach(({ file, errors }) => {
        // Get the first error message
        let errorMsg =
          errors[0]?.message ||
          t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })

        // Simplify error message for unsupported file types
        if (errorMsg.includes('file-invalid-type')) {
          errorMsg = t('documentPanel.uploadDocuments.fileUploader.unsupportedType')
        }

        // Set progress to 100% to display error message
        setProgresses((pre) => ({
          ...pre,
          [file.name]: 100,
        }))

        // Add error message to fileErrors
        setFileErrors((prev) => ({
          ...prev,
          [file.name]: errorMsg,
        }))
      })
    },
    [t]
  )

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)
      setUploadPhase('uploading')
      let hasSuccessfulUpload = false
      let completedCount = 0
      let failedCount = 0

      // Initialize stats
      setUploadStats({
        total: filesToUpload.length,
        completed: 0,
        failed: 0,
        currentFile: '',
      })

      // Only clear errors for files that are being uploaded, keep errors for rejected files
      setFileErrors((prev) => {
        const newErrors = { ...prev }
        filesToUpload.forEach((file) => {
          delete newErrors[file.name]
        })
        return newErrors
      })

      // Show uploading toast
      const toastId = toast.loading(t('documentPanel.uploadDocuments.batch.uploading'))

      try {
        // Track errors locally to ensure we have the final state
        const uploadErrors: Record<string, string> = {}

        // Create a collator that supports Chinese sorting
        const collator = new Intl.Collator(['zh-CN', 'en'], {
          sensitivity: 'accent', // consider basic characters, accents, and case
          numeric: true, // enable numeric sorting, e.g., "File 10" will be after "File 2"
        })
        const sortedFiles = [...filesToUpload].sort((a, b) => collator.compare(a.name, b.name))

        // Upload files in sequence, not parallel
        for (const file of sortedFiles) {
          // Update current file being uploaded
          setUploadStats(prev => ({
            ...prev,
            currentFile: file.name,
          }))

          try {
            // Initialize upload progress
            setProgresses((pre) => ({
              ...pre,
              [file.name]: 0,
            }))

            const result = await uploadDocument(
              file,
              (percentCompleted: number) => {
                console.debug(
                  t('documentPanel.uploadDocuments.single.uploading', {
                    name: file.name,
                    percent: percentCompleted,
                  })
                )
                setProgresses((pre) => ({
                  ...pre,
                  [file.name]: percentCompleted,
                }))
              },
              chunkingPreset
            )

            if (result.status === 'duplicated') {
              uploadErrors[file.name] = t(
                'documentPanel.uploadDocuments.fileUploader.duplicateFile'
              )
              setFileErrors((prev) => ({
                ...prev,
                [file.name]: t('documentPanel.uploadDocuments.fileUploader.duplicateFile'),
              }))
              failedCount++
            } else if (result.status !== 'success') {
              uploadErrors[file.name] = result.message
              setFileErrors((prev) => ({
                ...prev,
                [file.name]: result.message,
              }))
              failedCount++
            } else {
              // Mark that we had at least one successful upload
              hasSuccessfulUpload = true
              completedCount++
            }
          } catch (err) {
            console.error(`Upload failed for ${file.name}:`, err)

            // Handle HTTP errors, including 400 errors
            let errorMsg = errorMessage(err)

            // If it's an axios error with response data, try to extract more detailed error info
            if (err && typeof err === 'object' && 'response' in err) {
              const axiosError = err as {
                response?: { status: number; data?: { detail?: string } }
              }
              if (axiosError.response?.status === 400) {
                // Extract specific error message from backend response
                errorMsg = axiosError.response.data?.detail || errorMsg
              }

              // Set progress to 100% to display error message
              setProgresses((pre) => ({
                ...pre,
                [file.name]: 100,
              }))
            }

            // Record error message in both local tracking and state
            uploadErrors[file.name] = errorMsg
            setFileErrors((prev) => ({
              ...prev,
              [file.name]: errorMsg,
            }))
            failedCount++
          }

          // Update progress stats after each file
          setUploadStats(prev => ({
            ...prev,
            completed: completedCount,
            failed: failedCount,
          }))
        }

        // Check if any files failed to upload using our local tracking
        const hasErrors = Object.keys(uploadErrors).length > 0

        // Update toast status
        if (hasErrors) {
          toast.error(t('documentPanel.uploadDocuments.batch.error'), { id: toastId })
        } else {
          toast.success(t('documentPanel.uploadDocuments.batch.success'), { id: toastId })
        }

        // Only update if at least one file was uploaded successfully
        if (hasSuccessfulUpload) {
          // Refresh document list
          if (onDocumentsUploaded) {
            onDocumentsUploaded().catch((err) => {
              console.error('Error refreshing documents:', err)
            })
          }
        }

        // Set phase to complete so user can upload more or close
        setUploadPhase('complete')
      } catch (err) {
        console.error('Unexpected error during upload:', err)
        toast.error(t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }), {
          id: toastId,
        })
        setUploadPhase('complete')
      } finally {
        setIsUploading(false)
      }
    },
    [t, onDocumentsUploaded, chunkingPreset]
  )

  // Reset dialog state
  const resetDialog = useCallback(() => {
    setProgresses({})
    setFileErrors({})
    setUploadPhase('idle')
    setUploadStats({ total: 0, completed: 0, failed: 0, currentFile: '' })
  }, [])

  // Handle "Upload More" action
  const handleUploadMore = useCallback(() => {
    resetDialog()
  }, [resetDialog])

  // Handle close dialog
  const handleClose = useCallback(() => {
    resetDialog()
    setOpen(false)
  }, [resetDialog, setOpen])

  return (
    <Dialog
      open={open}
      onOpenChange={(newOpen) => {
        if (isUploading) {
          return
        }
        if (!newOpen) {
          resetDialog()
        }
        setOpen(newOpen)
      }}
    >
      <DialogTrigger asChild>
        <Button
          variant="default"
          side="bottom"
          tooltip={t('documentPanel.uploadDocuments.tooltip')}
          size="sm"
        >
          <UploadIcon /> {t('documentPanel.uploadDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.uploadDocuments.title')}</DialogTitle>
          <DialogDescription>{t('documentPanel.uploadDocuments.description')}</DialogDescription>
        </DialogHeader>

        {/* Overall progress bar (shown during upload) */}
        {(uploadPhase === 'uploading' || uploadPhase === 'complete') && uploadStats.total > 0 && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">
                {uploadPhase === 'uploading'
                  ? t('documentPanel.uploadDocuments.progress.uploading', {
                      current: uploadStats.completed + uploadStats.failed + 1,
                      total: uploadStats.total,
                      defaultValue: `Uploading ${uploadStats.completed + uploadStats.failed + 1} of ${uploadStats.total}...`,
                    })
                  : t('documentPanel.uploadDocuments.progress.complete', 'Upload complete')
                }
              </span>
              <span className="font-medium">{overallProgress}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-300',
                  uploadPhase === 'complete' && uploadStats.failed === 0
                    ? 'bg-emerald-500'
                    : uploadPhase === 'complete' && uploadStats.failed > 0
                      ? 'bg-amber-500'
                      : 'bg-primary'
                )}
                style={{ width: `${overallProgress}%` }}
              />
            </div>
            {/* Summary stats */}
            {uploadPhase === 'complete' && (
              <div className="flex gap-4 text-sm">
                <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  {t('documentPanel.uploadDocuments.progress.succeeded', {
                    count: uploadStats.completed,
                    defaultValue: `${uploadStats.completed} succeeded`,
                  })}
                </span>
                {uploadStats.failed > 0 && (
                  <span className="flex items-center gap-1 text-destructive">
                    <XCircle className="h-4 w-4" />
                    {t('documentPanel.uploadDocuments.progress.failed', {
                      count: uploadStats.failed,
                      defaultValue: `${uploadStats.failed} failed`,
                    })}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Advanced Options (Collapsible) - hidden during upload/complete */}
        {uploadPhase === 'idle' && (
          <div className="space-y-3">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-sm transition-colors"
              disabled={isUploading}
            >
              {showAdvanced ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              {t('documentPanel.uploadDocuments.advancedOptions')}
            </button>

            {showAdvanced && (
              <div className="bg-muted/50 space-y-3 rounded-md border p-3">
                <div className="space-y-1.5">
                  <label htmlFor="chunking-preset" className="text-sm font-medium">
                    {t('documentPanel.uploadDocuments.chunkingPreset.label')}
                  </label>
                  <Select
                    value={chunkingPreset}
                    onValueChange={(v) => setChunkingPreset(v as ChunkingPreset)}
                    disabled={isUploading}
                  >
                    <SelectTrigger id="chunking-preset" className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="semantic">
                        {t('documentPanel.uploadDocuments.chunkingPreset.semantic')}
                      </SelectItem>
                      <SelectItem value="recursive">
                        {t('documentPanel.uploadDocuments.chunkingPreset.recursive')}
                      </SelectItem>
                      <SelectItem value="">
                        {t('documentPanel.uploadDocuments.chunkingPreset.basic')}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-muted-foreground text-xs">
                    {t('documentPanel.uploadDocuments.chunkingPreset.description')}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        <FileUploader
          maxFileCount={Number.POSITIVE_INFINITY}
          maxSize={200 * 1024 * 1024}
          description={t('documentPanel.uploadDocuments.fileTypes')}
          onUpload={handleDocumentsUpload}
          onReject={handleRejectedFiles}
          progresses={progresses}
          fileErrors={fileErrors}
          disabled={isUploading}
        />

        {/* Completion actions */}
        {uploadPhase === 'complete' && (
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={handleUploadMore}>
              <UploadIcon className="h-4 w-4 mr-2" />
              {t('documentPanel.uploadDocuments.uploadMore', 'Upload More')}
            </Button>
            <Button onClick={handleClose}>
              {t('documentPanel.uploadDocuments.done', 'Done')}
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
