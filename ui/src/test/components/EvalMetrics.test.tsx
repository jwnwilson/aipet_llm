import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EvalMetrics } from '@/components/EvalMetrics'
import { QUALITY_REPORT_FIXTURE } from '../msw/fixtures'

describe('EvalMetrics — basic (no quality report)', () => {
  it('shows eval score percentage', () => {
    render(<EvalMetrics validPct={0.97} passed={true} />)
    expect(screen.getByText(/97\.0%/)).toBeInTheDocument()
  })

  it('shows Passed when passed=true', () => {
    render(<EvalMetrics validPct={0.97} passed={true} />)
    expect(screen.getByText(/passed/i)).toBeInTheDocument()
  })

  it('shows Failed when passed=false', () => {
    render(<EvalMetrics validPct={0.80} passed={false} />)
    expect(screen.getByText(/failed/i)).toBeInTheDocument()
  })
})

describe('EvalMetrics — with quality report', () => {
  it('renders a row for each stat in per_stat_accuracy', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    expect(screen.getByText('hunger')).toBeInTheDocument()
    expect(screen.getByText('boredom')).toBeInTheDocument()
    expect(screen.getByText('social')).toBeInTheDocument()
    expect(screen.getByText('tiredness')).toBeInTheDocument()
    expect(screen.getByText('toilet')).toBeInTheDocument()
  })

  it('renders action distribution counts', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    expect(screen.getByText('EAT')).toBeInTheDocument()
    expect(screen.getByText('50')).toBeInTheDocument()
  })

  it('renders accuracy percentage for a stat', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    // hunger: accuracy=0.95 → shows "95.0%"
    expect(screen.getAllByText(/95\.0%/).length).toBeGreaterThanOrEqual(1)
  })

  it('does not render per-stat table when qualityReport is null', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={null} />)
    expect(screen.queryByText('hunger')).not.toBeInTheDocument()
  })
})
