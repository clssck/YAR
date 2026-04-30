import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { CheckIcon, ChevronLeftIcon, ChevronRightIcon, CopyIcon, RefreshCwIcon } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { getTableData, getTableList, getTableSchema } from '@/api/yar'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import DataTable from '@/components/ui/DataTable'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { copyToClipboard } from '@/utils/clipboard'

// Generic row type for database table data
type TableRowData = Record<string, string | number | boolean | null | object>

const HIDDEN_COLUMNS = new Set(['meta'])

// Cell value type
type CellValue = string | number | boolean | null | object

// Truncate long values for display
function truncateValue(value: CellValue, maxLength = 50): string {
  if (value === null || value === undefined) return ''

  let strValue: string
  if (typeof value === 'object') {
    strValue = JSON.stringify(value)
  } else {
    strValue = String(value)
  }

  if (strValue.length <= maxLength) return strValue
  return `${strValue.slice(0, maxLength)}...`
}

// Format value for display in modal
function formatValue(value: CellValue): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'

  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }

  return String(value)
}

// Check if value is JSON-like (object or array)
function isJsonLike(value: CellValue): boolean {
  return typeof value === 'object' && value !== null
}

// Copy button component with feedback
function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    },
    []
  )

  const handleCopy = async () => {
    const result = await copyToClipboard(text)
    const success = result.success
    if (success) {
      setCopied(true)
      toast.success(label ? `${label} copied` : 'Copied to clipboard')
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error('Failed to copy')
    }
  }

  return (
    <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={handleCopy}>
      {copied ? <CheckIcon className="h-3 w-3 text-green-500" /> : <CopyIcon className="h-3 w-3" />}
    </Button>
  )
}

// Row Detail Modal
function RowDetailModal({
  row,
  open,
  onOpenChange
}: {
  row: TableRowData | null
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const entries = useMemo(() => (row ? Object.entries(row) : []), [row])
  const fullRowJson = useMemo(() => {
    try {
      return JSON.stringify(row, null, 2)
    } catch {
      return '[Unable to serialize row]'
    }
  }, [row])

  if (!row) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[80vh] max-w-3xl flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Row Details
            <CopyButton text={fullRowJson} label="Full row" />
          </DialogTitle>
          <DialogDescription>
            Click the copy icon next to any field to copy its value
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 space-y-3 overflow-auto pr-2">
          {entries.map(([key, value]) => (
            <div key={key} className="bg-muted/30 rounded-lg border p-3">
              <div className="mb-1 flex items-center justify-between">
                <span className="text-muted-foreground text-sm font-medium">{key}</span>
                <CopyButton text={formatValue(value)} label={key} />
              </div>
              <div className={`text-sm ${isJsonLike(value) ? 'font-mono' : ''}`}>
                {isJsonLike(value) ? (
                  <pre className="bg-muted max-h-[200px] overflow-auto rounded p-2 text-xs break-all whitespace-pre-wrap">
                    {formatValue(value)}
                  </pre>
                ) : (
                  <div className="break-all whitespace-pre-wrap">{formatValue(value)}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default function TableExplorer() {
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [page, setPage] = useState(1)
  const [selectedRow, setSelectedRow] = useState<TableRowData | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const pageSize = 20

  // Fetch table list
  const { data: tableList } = useQuery({
    queryKey: ['tables', 'list'],
    queryFn: getTableList
  })

  // Derive effective selection: use state if set, otherwise default to first table
  const effectiveSelectedTable = selectedTable || (tableList?.[0] ?? '')

  // Reset page when table changes
  const handleTableChange = (value: string) => {
    setSelectedTable(value)
    setPage(1)
  }

  // Fetch schema
  const { data: schema } = useQuery({
    queryKey: ['tables', effectiveSelectedTable, 'schema'],
    queryFn: () => getTableSchema(effectiveSelectedTable),
    enabled: !!effectiveSelectedTable
  })

  // Fetch data
  const {
    data: tableData,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ['tables', effectiveSelectedTable, 'data', page],
    queryFn: () => getTableData(effectiveSelectedTable, page, pageSize),
    enabled: !!effectiveSelectedTable
  })

  // Handle row click
  const handleRowClick = useCallback((row: TableRowData) => {
    setSelectedRow(row)
    setModalOpen(true)
  }, [])

  // Generate columns dynamically from data
  const columns = useMemo(() => {
    const cols: ColumnDef<TableRowData>[] = []
    if (tableData?.data && tableData.data.length > 0) {
      const allKeys = new Set<string>()
      for (const row of tableData.data) {
        for (const key of Object.keys(row)) {
          allKeys.add(key)
        }
      }

      Array.from(allKeys)
        .sort()
        .forEach((key) => {
          if (HIDDEN_COLUMNS.has(key)) return // Skip hidden columns
          cols.push({
            accessorKey: key,
            header: () => (
              <div className="max-w-[150px] truncate text-xs font-semibold" title={key}>
                {key}
              </div>
            ),
            cell: ({ row }) => {
              const value = row.getValue(key) as CellValue
              const displayValue = truncateValue(value, 50)
              const isLong =
                typeof value === 'object' || (typeof value === 'string' && value.length > 50)

              return (
                <div
                  className={`max-w-[200px] truncate text-xs ${isLong ? 'hover:text-primary cursor-pointer' : ''}`}
                  title={isLong ? 'Click row to see full value' : displayValue}
                >
                  {displayValue}
                </div>
              )
            }
          })
        })
    }
    return cols
  }, [tableData])

  const totalPages = tableData?.total_pages || 0

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden p-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-medium">Table Explorer</CardTitle>
            <div className="flex items-center gap-2">
              <Select value={effectiveSelectedTable} onValueChange={handleTableChange}>
                <SelectTrigger className="w-[250px]">
                  <SelectValue
                    placeholder={
                      tableList && tableList.length > 0 ? 'Select a table' : 'No tables available'
                    }
                  />
                </SelectTrigger>
                <SelectContent>
                  {tableList && tableList.length > 0 ? (
                    tableList.map((table) => (
                      <SelectItem key={table} value={table}>
                        {table}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="no-tables" disabled>
                      No tables found
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
              <Button variant="outline" size="icon" onClick={() => refetch()}>
                <RefreshCwIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        {schema && (
          <CardContent className="pb-2">
            <details className="text-muted-foreground cursor-pointer text-xs">
              <summary>Show Schema (DDL)</summary>
              <pre className="bg-muted mt-2 max-h-[200px] overflow-auto rounded p-2 font-mono text-xs">
                {schema.ddl}
              </pre>
            </details>
          </CardContent>
        )}
      </Card>

      <Card className="flex flex-1 flex-col overflow-hidden">
        <CardContent className="flex-1 overflow-auto p-0">
          {isLoading ? (
            <div className="flex h-full items-center justify-center">
              <RefreshCwIcon className="text-muted-foreground h-8 w-8 animate-spin" />
            </div>
          ) : isError ? (
            <div className="text-destructive flex h-full flex-col items-center justify-center gap-2">
              <p className="font-medium">Failed to load table data</p>
              <p className="text-muted-foreground text-sm">
                {error instanceof Error ? error.message : 'Unknown error'}
              </p>
              <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-2">
                Retry
              </Button>
            </div>
          ) : (
            <div className="h-full">
              <DataTable
                columns={columns}
                data={tableData?.data || []}
                onRowClick={handleRowClick}
              />
            </div>
          )}
        </CardContent>

        <div className="bg-muted/20 flex items-center justify-between border-t p-2">
          <div className="text-muted-foreground text-sm">
            {tableData?.total ? (
              <>
                Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, tableData.total)}{' '}
                of {tableData.total} rows
              </>
            ) : (
              'No results'
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || isLoading}
            >
              <ChevronLeftIcon className="mr-1 h-4 w-4" />
              Previous
            </Button>
            <span className="min-w-[3rem] text-center text-sm font-medium">
              {page} / {totalPages || 1}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || isLoading}
            >
              Next
              <ChevronRightIcon className="ml-1 h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>

      <RowDetailModal row={selectedRow} open={modalOpen} onOpenChange={setModalOpen} />
    </div>
  )
}
