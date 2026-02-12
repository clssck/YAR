/**
 * Simple Slider component using native HTML range input
 */

import { forwardRef } from 'react'
import { cn } from '@/lib/utils'

export interface SliderProps
  extends Omit<
    React.InputHTMLAttributes<HTMLInputElement>,
    'type' | 'onChange'
  > {
  /** Current value */
  value?: number
  /** Called when value changes */
  onValueChange?: (value: number) => void
  /** Minimum value */
  min?: number
  /** Maximum value */
  max?: number
  /** Step increment */
  step?: number
  /** Labels to show at min/max */
  labels?: { min?: string; max?: string }
  /** Show current value */
  showValue?: boolean
  /** Format function for displayed value */
  formatValue?: (value: number) => string
}

const Slider = forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      className,
      value = 0,
      onValueChange,
      min = 0,
      max = 100,
      step = 1,
      labels,
      showValue = true,
      formatValue = (v) => v.toString(),
      disabled,
      ...props
    },
    ref,
  ) => {
    // Calculate percentage for gradient fill
    const percentage = ((value - min) / (max - min)) * 100

    return (
      <div className={cn('w-full', className)}>
        {/* Labels row */}
        {(labels?.min || labels?.max || showValue) && (
          <div className="flex justify-between items-center mb-1 text-[10px] text-muted-foreground">
            <span>{labels?.min || ''}</span>
            {showValue && (
              <span className="font-medium text-foreground">
                {formatValue(value)}
              </span>
            )}
            <span>{labels?.max || ''}</span>
          </div>
        )}

        {/* Slider track */}
        <input
          ref={ref}
          type="range"
          value={value}
          onChange={(e) => onValueChange?.(parseFloat(e.target.value))}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          className={cn(
            'w-full h-2 rounded-full appearance-none cursor-pointer',
            'bg-muted',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            // Thumb styling
            '[&::-webkit-slider-thumb]:appearance-none',
            '[&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4',
            '[&::-webkit-slider-thumb]:rounded-full',
            '[&::-webkit-slider-thumb]:bg-primary',
            '[&::-webkit-slider-thumb]:shadow-sm',
            '[&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background',
            '[&::-webkit-slider-thumb]:transition-transform',
            '[&::-webkit-slider-thumb]:hover:scale-110',
            '[&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4',
            '[&::-moz-range-thumb]:rounded-full',
            '[&::-moz-range-thumb]:bg-primary',
            '[&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-background',
            '[&::-moz-range-thumb]:transition-transform',
            '[&::-moz-range-thumb]:hover:scale-110',
          )}
          style={{
            background: `linear-gradient(to right, hsl(var(--primary)) ${percentage}%, hsl(var(--muted)) ${percentage}%)`,
          }}
          {...props}
        />
      </div>
    )
  },
)

Slider.displayName = 'Slider'

export default Slider
