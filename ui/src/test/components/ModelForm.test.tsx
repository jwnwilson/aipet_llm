import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ModelForm } from '@/components/ModelForm'

describe('ModelForm', () => {
  it('pre-populates default values', () => {
    render(<ModelForm onSubmit={vi.fn()} />)
    expect((screen.getByDisplayValue('HuggingFaceTB/SmolLM2-360M') as HTMLInputElement).value)
      .toBe('HuggingFaceTB/SmolLM2-360M')
  })

  it('shows validation error when name is empty', async () => {
    render(<ModelForm onSubmit={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() => expect(screen.getByText('Name is required')).toBeInTheDocument())
  })

  it('calls onSubmit with correct values when form is valid', async () => {
    const onSubmit = vi.fn()
    render(<ModelForm onSubmit={onSubmit} />)
    await userEvent.type(screen.getByPlaceholderText('my-experiment'), 'test-run')
    await userEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce())
    expect(onSubmit.mock.calls[0][0].name).toBe('test-run')
  })

  it('pre-fills values from defaultValues prop', () => {
    render(<ModelForm defaultValues={{ name: 'existing', epochs: 10 }} onSubmit={vi.fn()} />)
    expect((screen.getByDisplayValue('existing') as HTMLInputElement).value).toBe('existing')
    expect((screen.getByDisplayValue('10') as HTMLInputElement).value).toBe('10')
  })

  it('shows all base model options when the default value is selected', async () => {
    render(<ModelForm onSubmit={vi.fn()} />)
    await userEvent.click(screen.getByDisplayValue('HuggingFaceTB/SmolLM2-360M'))
    const options = await screen.findAllByRole('option')
    expect(options.length).toBeGreaterThan(1)
    expect(screen.getByRole('option', { name: 'Qwen/Qwen2.5-0.5B' })).toBeInTheDocument()
  })

  it('narrows base model options while typing a partial query', async () => {
    render(<ModelForm onSubmit={vi.fn()} />)
    const input = screen.getByDisplayValue('HuggingFaceTB/SmolLM2-360M')
    await userEvent.clear(input)
    await userEvent.type(input, 'qwen')
    const options = await screen.findAllByRole('option')
    expect(options.every(opt => /qwen/i.test(opt.textContent ?? ''))).toBe(true)
  })
})
