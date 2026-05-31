import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface PaginationProps {
  page: number
  pages: number
  onPageChange: (page: number) => void
}

export function Pagination({ page, pages, onPageChange }: PaginationProps) {
  if (pages <= 1) return null

  return (
    <div className="flex items-center justify-end gap-3 mt-4">
      <Button
        size="sm"
        variant="outline"
        aria-label="Previous page"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </Button>
      <span className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#3a3a36]">
        {page} / {pages}
      </span>
      <Button
        size="sm"
        variant="outline"
        aria-label="Next page"
        disabled={page >= pages}
        onClick={() => onPageChange(page + 1)}
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
