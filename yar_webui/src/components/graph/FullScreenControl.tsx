import { useFullScreen } from '@react-sigma/core'
import { MaximizeIcon, MinimizeIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ControlButton } from '@/components/ui/Button'

/**
 * Component that toggles full screen mode.
 */
const FullScreenControl = () => {
  const { isFullScreen, toggle } = useFullScreen()
  const { t } = useTranslation()

  return (
    <ControlButton
      onClick={toggle}
      tooltip={t(
        isFullScreen
          ? 'graphPanel.sideBar.fullScreenControl.windowed'
          : 'graphPanel.sideBar.fullScreenControl.fullScreen'
      )}
    >
      {isFullScreen ? <MinimizeIcon /> : <MaximizeIcon />}
    </ControlButton>
  )
}

export default FullScreenControl
