import { AlertTriangleIcon, FileText, Loader2, TrashIcon } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { deleteDocuments } from '@/api/lightrag'
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
import { cn, errorMessage } from '@/lib/utils'

// Simple Label component
const Label = ({
  htmlFor,
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label htmlFor={htmlFor} className={className} {...props}>
    {children}
  </label>
)

interface DeleteDocumentsDialogProps {
  selectedDocIds: string[]
  onDocumentsDeleted?: () => Promise<void>
}

const MAX_VISIBLE_DOCS = 5

export default function DeleteDocumentsDialog({
  selectedDocIds,
  onDocumentsDeleted,
}: DeleteDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [deleteFile, setDeleteFile] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteLLMCache, setDeleteLLMCache] = useState(false)

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setDeleteFile(false)
      setDeleteLLMCache(false)
      setIsDeleting(false)
    }
  }, [open])

  const handleDelete = useCallback(async () => {
    if (isDeleting || selectedDocIds.length === 0) return

    setIsDeleting(true)
    try {
      const result = await deleteDocuments(selectedDocIds, deleteFile, deleteLLMCache)

      if (result.status === 'deletion_started') {
        toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
      } else if (result.status === 'busy') {
        toast.error(t('documentPanel.deleteDocuments.busy'))
        setIsDeleting(false)
        return
      } else if (result.status === 'not_allowed') {
        toast.error(t('documentPanel.deleteDocuments.notAllowed'))
        setIsDeleting(false)
        return
      } else {
        toast.error(t('documentPanel.deleteDocuments.failed', { message: result.message }))
        setIsDeleting(false)
        return
      }

      // Refresh document list if provided
      if (onDocumentsDeleted) {
        onDocumentsDeleted().catch(console.error)
      }

      // Close dialog after successful operation
      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.deleteDocuments.error', { error: errorMessage(err) }))
    } finally {
      setIsDeleting(false)
    }
  }, [isDeleting, selectedDocIds, deleteFile, deleteLLMCache, t, onDocumentsDeleted])

  const visibleDocs = selectedDocIds.slice(0, MAX_VISIBLE_DOCS)
  const remainingCount = selectedDocIds.length - MAX_VISIBLE_DOCS

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="destructive"
          side="bottom"
          tooltip={t('documentPanel.deleteDocuments.tooltip', { count: selectedDocIds.length })}
          size="sm"
        >
          <TrashIcon /> {t('documentPanel.deleteDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-destructive font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.deleteDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deleteDocuments.description', { count: selectedDocIds.length })}
          </DialogDescription>
        </DialogHeader>

        {/* Loading state during deletion */}
        {isDeleting ? (
          <div className="py-8 flex flex-col items-center justify-center gap-4">
            <div className="relative">
              <Loader2 className="h-12 w-12 text-destructive animate-spin" />
            </div>
            <div className="text-center">
              <div className="font-medium text-lg">
                {t('documentPanel.deleteDocuments.deletingProgress', {
                  count: selectedDocIds.length,
                  defaultValue: `Deleting ${selectedDocIds.length} document${selectedDocIds.length === 1 ? '' : 's'}...`,
                })}
              </div>
              <div className="text-sm text-muted-foreground mt-1">
                {t('documentPanel.deleteDocuments.pleaseWait', 'Please wait, this may take a moment')}
              </div>
            </div>
            {/* Progress bar animation */}
            <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-destructive rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
          </div>
        ) : (
          <>
            {/* Document list preview */}
            <div className="my-2">
              <div className="text-sm font-medium text-muted-foreground mb-2">
                {t('documentPanel.deleteDocuments.documentsToDelete', 'Documents to delete:')}
              </div>
              <div className="bg-muted/50 rounded-md p-2 max-h-[150px] overflow-y-auto">
                {visibleDocs.map((docId) => (
                  <div
                    key={docId}
                    className="flex items-center gap-2 py-1 px-2 text-sm font-mono truncate"
                  >
                    <FileText className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
                    <span className="truncate">{docId}</span>
                  </div>
                ))}
                {remainingCount > 0 && (
                  <div className="py-1 px-2 text-sm text-muted-foreground italic">
                    {t('documentPanel.deleteDocuments.andMore', {
                      count: remainingCount,
                      defaultValue: `+${remainingCount} more...`,
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Warning */}
            <div className="bg-destructive/10 border border-destructive/20 rounded-md p-3 text-sm text-destructive">
              <div className="flex items-start gap-2">
                <AlertTriangleIcon className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span>{t('documentPanel.deleteDocuments.warning')}</span>
              </div>
            </div>

            {/* Options */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="delete-file"
                  checked={deleteFile}
                  onChange={(e) => setDeleteFile(e.target.checked)}
                  disabled={isDeleting}
                  className="h-4 w-4 rounded border-input focus:ring-2 focus:ring-ring"
                />
                <Label htmlFor="delete-file" className="text-sm cursor-pointer">
                  {t('documentPanel.deleteDocuments.deleteFileOption')}
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="delete-llm-cache"
                  checked={deleteLLMCache}
                  onChange={(e) => setDeleteLLMCache(e.target.checked)}
                  disabled={isDeleting}
                  className="h-4 w-4 rounded border-input focus:ring-2 focus:ring-ring"
                />
                <Label htmlFor="delete-llm-cache" className="text-sm cursor-pointer">
                  {t('documentPanel.deleteDocuments.deleteLLMCacheOption')}
                </Label>
              </div>
            </div>
          </>
        )}

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isDeleting}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={isDeleting}
            className={cn(isDeleting && 'opacity-70')}
          >
            {isDeleting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                {t('documentPanel.deleteDocuments.deleting')}
              </>
            ) : (
              <>
                <TrashIcon className="h-4 w-4 mr-2" />
                {t('documentPanel.deleteDocuments.confirmButton', {
                  count: selectedDocIds.length,
                  defaultValue: `Delete ${selectedDocIds.length} document${selectedDocIds.length === 1 ? '' : 's'}`,
                })}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
