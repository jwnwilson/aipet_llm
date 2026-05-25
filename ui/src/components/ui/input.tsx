import * as React from 'react'
import { cn } from '@/lib/utils'

/**
 * Editorial input — clean ink-on-paper.
 * 1.5px border, focus collapses to ink black, no glow rings.
 */
const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full bg-white text-[0.92rem] text-[#1a1a1a]",
        "font-['Outfit'] font-normal",
        'px-3 py-2 rounded-[3px]',
        'border-[1.5px] border-[#d0d0c8]',
        'placeholder:text-[#b3b1a6]',
        'transition-colors duration-150',
        'focus-visible:outline-none focus-visible:border-[#1a1a1a]',
        'disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-[#f6f5ef]',
        "file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-[#1a1a1a] file:font-['IBM_Plex_Mono'] file:mr-3",
        className,
      )}
      ref={ref}
      {...props}
    />
  )
)
Input.displayName = 'Input'

export { Input }
