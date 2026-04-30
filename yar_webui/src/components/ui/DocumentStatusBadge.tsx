import StatusBadge from './StatusBadge'

/**
 * Pre-configured status badges for the document pipeline states.
 * Split from StatusBadge to keep both files single-purpose for Fast Refresh.
 */
const DocumentStatusBadge = {
  Processed: () => <StatusBadge status="success" label="Processed" />,
  Processing: () => <StatusBadge status="processing" label="Processing" />,
  Pending: () => <StatusBadge status="pending" label="Pending" />,
  Failed: () => <StatusBadge status="error" label="Failed" />,
  Preprocessed: () => <StatusBadge status="info" label="Preprocessed" />
}

export default DocumentStatusBadge
