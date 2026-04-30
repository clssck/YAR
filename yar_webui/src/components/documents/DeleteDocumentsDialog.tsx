import { AlertTriangleIcon, FileText, Loader2, TrashIcon } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { deleteDocuments } from '@/api/yar'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
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
  onDocumentsDeleted
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
        toast.success(
          t('documentPanel.deleteDocuments.success', {
            count: selectedDocIds.length
          })
        )
      } else if (result.status === 'busy') {
        toast.error(t('documentPanel.deleteDocuments.busy'))
        setIsDeleting(false)
        return
      } else if (result.status === 'not_allowed') {
        toast.error(t('documentPanel.deleteDocuments.notAllowed'))
        setIsDeleting(false)
        return
      } else {
        toast.error(
          t('documentPanel.deleteDocuments.failed', {
            message: result.message
          })
        )
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
          tooltip={t('documentPanel.deleteDocuments.tooltip', {
            count: selectedDocIds.length
          })}
          size="sm"
        >
          <TrashIcon /> {t('documentPanel.deleteDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="text-destructive flex items-center gap-2 font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.deleteDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deleteDocuments.description', {
              count: selectedDocIds.length
            })}
          </DialogDescription>
        </DialogHeader>

        {/* Loading state during deletion */}
        {isDeleting ? (
          <div className="flex flex-col items-center justify-center gap-4 py-8">
            <div className="relative">
              <Loader2 className="text-destructive h-12 w-12 animate-spin" />
            </div>
            <div className="text-center">
              <div className="text-lg font-medium">
                {t('documentPanel.deleteDocuments.deletingProgress', {
                  count: selectedDocIds.length,
                  defaultValue: `Deleting ${selectedDocIds.length} document${selectedDocIds.length === 1 ? '' : 's'}...`
                })}
              </div>
              <div className="text-muted-foreground mt-1 text-sm">
                {t(
                  'documentPanel.deleteDocuments.pleaseWait',
                  'Please wait, this may take a moment'
                )}
              </div>
            </div>
            {/* Progress bar animation */}
            <div className="bg-muted h-1.5 w-full overflow-hidden rounded-full">
              <div
                className="bg-destructive h-full animate-pulse rounded-full"
                style={{ width: '60%' }}
              />
            </div>
          </div>
        ) : (
          <>
            {/* Document list preview */}
            <div className="my-2">
              <div className="text-muted-foreground mb-2 text-sm font-medium">
                {t('documentPanel.deleteDocuments.documentsToDelete', 'Documents to delete:')}
              </div>
              <div className="bg-muted/50 max-h-[150px] overflow-y-auto rounded-md p-2">
                {visibleDocs.map((docId) => (
                  <div
                    key={docId}
                    className="flex items-center gap-2 truncate px-2 py-1 font-mono text-sm"
                  >
                    <FileText className="text-muted-foreground h-3.5 w-3.5 flex-shrink-0" />
                    <span className="truncate">{docId}</span>
                  </div>
                ))}
                {remainingCount > 0 && (
                  <div className="text-muted-foreground px-2 py-1 text-sm italic">
                    {t('documentPanel.deleteDocuments.andMore', {
                      count: remainingCount,
                      defaultValue: `+${remainingCount} more...`
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Warning */}
            <div className="bg-destructive/10 border-destructive/20 text-destructive rounded-md border p-3 text-sm">
              <div className="flex items-start gap-2">
                <AlertTriangleIcon className="mt-0.5 h-4 w-4 flex-shrink-0" />
                <span>{t('documentPanel.deleteDocuments.warning')}</span>
              </div>
            </div>

            {/* Options */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="delete-file"
                  checked={deleteFile}
                  onCheckedChange={(checked) => setDeleteFile(checked === true)}
                  disabled={isDeleting}
                />
                <Label htmlFor="delete-file" className="cursor-pointer text-sm">
                  {t('documentPanel.deleteDocuments.deleteFileOption')}
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="delete-llm-cache"
                  checked={deleteLLMCache}
                  onCheckedChange={(checked) => setDeleteLLMCache(checked === true)}
                  disabled={isDeleting}
                />
                <Label htmlFor="delete-llm-cache" className="cursor-pointer text-sm">
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
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('documentPanel.deleteDocuments.deleting')}
              </>
            ) : (
              <>
                <TrashIcon className="mr-2 h-4 w-4" />
                {t('documentPanel.deleteDocuments.confirmButton', {
                  count: selectedDocIds.length,
                  defaultValue: `Delete ${selectedDocIds.length} document${selectedDocIds.length === 1 ? '' : 's'}`
                })}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
