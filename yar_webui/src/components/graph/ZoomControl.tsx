import { useCamera, useSigma } from '@react-sigma/core'
import {
  FullscreenIcon,
  RotateCcwIcon,
  RotateCwIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react'
import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { ControlButton } from '@/components/ui/Button'

/**
 * Component that provides zoom controls for the graph viewer.
 */
const ZoomControl = () => {
  const { zoomIn, zoomOut, reset } = useCamera({ duration: 200, factor: 1.5 })
  const sigma = useSigma()
  const { t } = useTranslation()

  const handleZoomIn = useCallback(() => zoomIn(), [zoomIn])
  const handleZoomOut = useCallback(() => zoomOut(), [zoomOut])
  const handleResetZoom = useCallback(() => {
    if (!sigma) return

    try {
      // First clear any custom bounding box and refresh
      sigma.setCustomBBox(null)
      sigma.refresh()

      // Get graph after refresh
      const graph = sigma.getGraph()

      // Check if graph has nodes before accessing them
      if (!graph?.order || graph.nodes().length === 0) {
        // Use reset() for empty graph case
        reset()
        return
      }

      sigma
        .getCamera()
        .animate({ x: 0.5, y: 0.5, ratio: 1.1 }, { duration: 1000 })
    } catch (error) {
      console.error('Error resetting zoom:', error)
      // Use reset() as fallback on error
      reset()
    }
  }, [sigma, reset])

  const handleRotate = useCallback(() => {
    if (!sigma) return

    const camera = sigma.getCamera()
    const currentAngle = camera.angle
    const newAngle = currentAngle + Math.PI / 8

    camera.animate({ angle: newAngle }, { duration: 200 })
  }, [sigma])

  const handleRotateCounterClockwise = useCallback(() => {
    if (!sigma) return

    const camera = sigma.getCamera()
    const currentAngle = camera.angle
    const newAngle = currentAngle - Math.PI / 8

    camera.animate({ angle: newAngle }, { duration: 200 })
  }, [sigma])

  return (
    <>
      <ControlButton
        onClick={handleRotate}
        tooltip={t('graphPanel.sideBar.zoomControl.rotateCamera')}
      >
        <RotateCwIcon />
      </ControlButton>
      <ControlButton
        onClick={handleRotateCounterClockwise}
        tooltip={t(
          'graphPanel.sideBar.zoomControl.rotateCameraCounterClockwise',
        )}
      >
        <RotateCcwIcon />
      </ControlButton>
      <ControlButton
        onClick={handleResetZoom}
        tooltip={t('graphPanel.sideBar.zoomControl.resetZoom')}
      >
        <FullscreenIcon />
      </ControlButton>
      <ControlButton
        onClick={handleZoomIn}
        tooltip={t('graphPanel.sideBar.zoomControl.zoomIn')}
      >
        <ZoomInIcon />
      </ControlButton>
      <ControlButton
        onClick={handleZoomOut}
        tooltip={t('graphPanel.sideBar.zoomControl.zoomOut')}
      >
        <ZoomOutIcon />
      </ControlButton>
    </>
  )
}

export default ZoomControl
