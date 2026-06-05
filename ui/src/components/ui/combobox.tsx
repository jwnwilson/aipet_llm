import * as React from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ComboboxProps {
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder?: string
  disabled?: boolean
  className?: string
  id?: string
}

export function Combobox({ value, onChange, options, placeholder, disabled, className, id }: ComboboxProps) {
  const [open, setOpen] = React.useState(false)
  const [activeIndex, setActiveIndex] = React.useState(-1)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const listRef = React.useRef<HTMLUListElement>(null)

  const filtered = React.useMemo(
    () =>
      value.trim() === ''
        ? options
        : options.filter(opt => opt.toLowerCase().includes(value.toLowerCase())),
    [value, options]
  )

  React.useEffect(() => { setActiveIndex(-1) }, [filtered])

  React.useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const el = listRef.current.children[activeIndex] as HTMLElement | undefined
      el?.scrollIntoView({ block: 'nearest' })
    }
  }, [activeIndex])

  React.useEffect(() => {
    if (!open) return
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      setOpen(true)
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && activeIndex >= 0) {
      e.preventDefault()
      onChange(filtered[activeIndex])
      setOpen(false)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div ref={containerRef} className={cn('relative w-full', className)}>
      <input
        id={id}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        aria-controls="combobox-listbox"
        autoComplete="off"
        disabled={disabled}
        value={value}
        placeholder={placeholder}
        onChange={e => {
          onChange(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex h-10 w-full bg-white px-3 py-2 pr-9 text-[0.92rem] text-[#1a1a1a]",
          "font-['Outfit'] rounded-[3px] border-[1.5px] border-[#d0d0c8]",
          'transition-colors duration-150',
          'placeholder:text-[#767676]',
          'focus-visible:outline-none focus-visible:border-[#1a1a1a]',
          'disabled:cursor-not-allowed disabled:opacity-50',
        )}
      />
      <ChevronDown
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[#6b6b6b]"
        aria-hidden
      />
      {open && filtered.length > 0 && (
        <ul
          id="combobox-listbox"
          role="listbox"
          ref={listRef}
          className="absolute z-50 mt-1 max-h-60 w-full overflow-auto bg-white border border-[#d0d0c8] rounded-[3px] py-1 shadow-[0_4px_14px_rgba(0,0,0,0.10)] text-[0.9rem] text-[#1a1a1a] font-['Outfit']"
        >
          {filtered.map((opt, i) => (
            <li
              key={opt}
              role="option"
              aria-selected={opt === value}
              onMouseDown={e => {
                e.preventDefault()
                onChange(opt)
                setOpen(false)
              }}
              onMouseEnter={() => setActiveIndex(i)}
              className={cn(
                'cursor-pointer select-none px-3 py-1.5',
                i === activeIndex && 'bg-[#f3f2ec] text-[#1a1a1a]',
                opt === value && i !== activeIndex && 'font-medium'
              )}
            >
              {opt}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
