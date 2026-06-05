import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import type { RunStatus } from '@/types'

const cases: Array<[RunStatus, string]> = [
  ['pending', 'Pending'],
  ['generating', 'Generating'],
  ['training', 'Training'],
  ['evaluating', 'Evaluating'],
  ['exporting', 'Exporting'],
  ['running', 'Running'],
  ['completed', 'Completed'],
  ['failed', 'Failed'],
]

describe('RunStatusBadge', () => {
  it.each(cases)('renders label for status %s', (status, label) => {
    render(<RunStatusBadge status={status} />)
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('applies success styling for completed', () => {
    render(<RunStatusBadge status="completed" />)
    const badge = screen.getByTestId('run-status-badge')
    expect(badge.className).toContain('bg-[#e8efe9]')
    expect(badge.className).toContain('text-[#2d6a4f]')
  })

  it('applies danger styling for failed', () => {
    render(<RunStatusBadge status="failed" />)
    const badge = screen.getByTestId('run-status-badge')
    expect(badge.className).toContain('bg-[#f1e2e0]')
    expect(badge.className).toContain('text-[#7f1d1d]')
  })

  it('applies active styling for running', () => {
    render(<RunStatusBadge status="running" />)
    const badge = screen.getByTestId('run-status-badge')
    expect(badge.className).toContain('bg-[#1a1a1a]')
    expect(badge.className).toContain('text-[#fafaf7]')
  })
})
