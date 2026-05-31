import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { Pagination } from '@/components/Pagination'

describe('Pagination', () => {
  it('renders page info', () => {
    render(<Pagination page={1} pages={5} onPageChange={() => {}} />)
    expect(screen.getByText('1 / 5')).toBeInTheDocument()
  })

  it('disables prev on first page', () => {
    render(<Pagination page={1} pages={5} onPageChange={() => {}} />)
    expect(screen.getByLabelText('Previous page')).toBeDisabled()
  })

  it('disables next on last page', () => {
    render(<Pagination page={5} pages={5} onPageChange={() => {}} />)
    expect(screen.getByLabelText('Next page')).toBeDisabled()
  })

  it('calls onPageChange with next page', async () => {
    const onChange = vi.fn()
    render(<Pagination page={2} pages={5} onPageChange={onChange} />)
    await userEvent.click(screen.getByLabelText('Next page'))
    expect(onChange).toHaveBeenCalledWith(3)
  })

  it('calls onPageChange with prev page', async () => {
    const onChange = vi.fn()
    render(<Pagination page={3} pages={5} onPageChange={onChange} />)
    await userEvent.click(screen.getByLabelText('Previous page'))
    expect(onChange).toHaveBeenCalledWith(2)
  })

  it('does not render when pages <= 1', () => {
    const { container } = render(<Pagination page={1} pages={1} onPageChange={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})
