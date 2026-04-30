import type React from 'react'
import { useTranslation } from 'react-i18next'
import { Card } from '@/components/ui/Card'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { useGraphStore } from '@/stores/graph'

interface LegendProps {
  className?: string
}

const Legend: React.FC<LegendProps> = ({ className }) => {
  const { t } = useTranslation()
  const typeColorMap = useGraphStore.use.typeColorMap()

  if (!typeColorMap || typeColorMap.size === 0) {
    return null
  }

  return (
    <Card className={`max-w-xs p-2 ${className}`}>
      {/* Shapes section */}
      <h3 className="mb-2 text-sm font-medium">{t('graphPanel.legend.shapes')}</h3>
      <div className="mb-3 flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded-full border border-slate-400 bg-sky-400" />
          <span className="text-xs">{t('graphPanel.legend.connectedNode')}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded-full border-[3px] border-gray-700 bg-sky-400" />
          <span className="text-xs">{t('graphPanel.legend.orphanNode')}</span>
        </div>
      </div>

      {/* Colors section */}
      <h3 className="mb-2 text-sm font-medium">{t('graphPanel.legend.colors')}</h3>
      <ScrollArea className="max-h-64">
        <div className="flex flex-col gap-1">
          {Array.from(typeColorMap.entries()).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div className="h-4 w-4 rounded-full" style={{ backgroundColor: color }} />
              <span className="truncate text-xs" title={type}>
                {t(`graphPanel.nodeTypes.${type.toLowerCase().replace(/\s+/g, '')}`, type)}
              </span>
            </div>
          ))}
        </div>
      </ScrollArea>
    </Card>
  )
}

export default Legend
