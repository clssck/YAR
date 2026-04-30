import { Link } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ControlButton } from '@/components/ui/Button'
import { useBackendState } from '@/stores/state'
import OrphanConnectionDialog from './OrphanConnectionDialog'

/**
 * Control button for orphan entity connection.
 * Only visible when AUTO_CONNECT_ORPHANS is disabled (manual mode).
 */
export default function OrphanConnectionControl() {
  const { t } = useTranslation()
  const [showDialog, setShowDialog] = useState(false)
  const status = useBackendState.use.status()

  // Only show when auto_connect_orphans is explicitly false (manual mode)
  if (status?.configuration?.auto_connect_orphans !== false) {
    return null
  }

  return (
    <>
      <ControlButton
        tooltip={t('graphPanel.orphanConnection.tooltip')}
        onClick={() => setShowDialog(true)}
      >
        <Link />
      </ControlButton>
      <OrphanConnectionDialog open={showDialog} onOpenChange={setShowDialog} />
    </>
  )
}
