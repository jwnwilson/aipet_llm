import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

/**
 * Editorial Scientific button.
 * No saturated colors — ink black, paper white, dark academic accents.
 * Body text uses Outfit; uppercase tracking on small variants for label feel.
 */
const buttonVariants = cva(
  [
    'inline-flex items-center justify-center gap-1.5',
    'font-medium text-[0.85rem]',
    "font-['Outfit']",
    'transition-colors duration-150',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ink)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-paper)]',
    'disabled:pointer-events-none disabled:opacity-40',
    'whitespace-nowrap select-none',
    'rounded-[3px]',
  ].join(' '),
  {
    variants: {
      variant: {
        // Solid ink — primary action
        default: 'bg-[#1a1a1a] text-[#fafaf7] border border-[#1a1a1a] hover:bg-black hover:border-black',
        // Outline — ink rule, fills on hover
        outline: 'bg-transparent text-[#1a1a1a] border-[1.5px] border-[#1a1a1a] hover:bg-[#1a1a1a] hover:text-[#fafaf7]',
        // Destructive — academic dark red
        destructive: 'bg-[#7f1d1d] text-[#fafaf7] border border-[#7f1d1d] hover:bg-[#651616] hover:border-[#651616]',
        // Ghost — shows ink on hover
        ghost: 'bg-transparent text-[#3a3a36] border border-transparent hover:text-[#1a1a1a] hover:bg-[#f3f2ec]',
        // Subtle — paper alt surface
        subtle: 'bg-[#f3f2ec] text-[#1a1a1a] border border-[#d0d0c8] hover:bg-[#ebe9df]',
        // Link
        link: 'text-[#1a1a1a] underline underline-offset-4 decoration-[#b3b1a6] hover:decoration-[#1a1a1a] bg-transparent border-transparent p-0 h-auto',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: "h-8 px-3 text-[0.72rem] uppercase tracking-[0.12em] font-['Outfit'] font-semibold",
        lg: 'h-11 px-7 text-[0.95rem]',
        icon: 'h-9 w-9 p-0',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }
