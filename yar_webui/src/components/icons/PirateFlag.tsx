import type { SVGProps } from 'react'

export function PirateFlag(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Pole */}
      <line x1="5" y1="3" x2="5" y2="21" />
      {/* Flag */}
      <path d="M5 3c4 0 6 2 10 2 2 0 3-.5 4-1v9c-1 .5-2 1-4 1-4 0-6-2-10-2" />
      {/* Skull */}
      <circle cx="12" cy="8" r="2" fill="currentColor" />
      {/* Crossbones */}
      <path d="M9 11l6-6M15 11l-6-6" strokeWidth="1.5" />
    </svg>
  )
}
