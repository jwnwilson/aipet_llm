import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PipelineStages } from '@/components/PipelineStages'
import type { PipelineStage } from '@/components/PipelineStages'

const stages: PipelineStage[] = [
  { name: 'Generate', status: 'completed' },
  { name: 'Train', status: 'active' },
  { name: 'Evaluate', status: 'pending' },
  { name: 'Export', status: 'pending' },
]

function mockMobile() {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === '(max-width: 767px)',
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
}

describe('PipelineStages', () => {
  it('renders all stage names', () => {
    render(<PipelineStages stages={stages} />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
    expect(screen.getByText('Train')).toBeInTheDocument()
    expect(screen.getByText('Evaluate')).toBeInTheDocument()
    expect(screen.getByText('Export')).toBeInTheDocument()
  })

  it('pending stages have reduced opacity class', () => {
    render(<PipelineStages stages={stages} />)
    const pendingStage = screen.getByTestId('stage-evaluate')
    expect(pendingStage).toHaveClass('opacity-40')
  })

  it('active stage does not have opacity-40', () => {
    render(<PipelineStages stages={stages} />)
    const activeStage = screen.getByTestId('stage-train')
    expect(activeStage).not.toHaveClass('opacity-40')
  })

  it('completed stage does not have opacity-40', () => {
    render(<PipelineStages stages={stages} />)
    const completedStage = screen.getByTestId('stage-generate')
    expect(completedStage).not.toHaveClass('opacity-40')
  })
})

describe('PipelineStages — mobile', () => {
  it('renders mobile grid container on mobile', () => {
    mockMobile()
    render(<PipelineStages stages={stages} />)
    expect(screen.getByTestId('pipeline-mobile-grid')).toBeInTheDocument()
  })

  it('renders all stage names on mobile', () => {
    mockMobile()
    render(<PipelineStages stages={stages} />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
    expect(screen.getByText('Train')).toBeInTheDocument()
    expect(screen.getByText('Evaluate')).toBeInTheDocument()
    expect(screen.getByText('Export')).toBeInTheDocument()
  })

  it('does not render connecting lines on mobile', () => {
    mockMobile()
    render(<PipelineStages stages={stages} />)
    expect(screen.queryByRole('separator')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('separator')).not.toBeInTheDocument()
  })
})
