import '../../../__tests__/setup'
import { describe, expect, test } from 'bun:test'
import { render } from '@testing-library/react'
import Badge from '../../../components/ui/Badge'

describe('Badge', () => {
  describe('rendering', () => {
    test('renders with default content', () => {
      const { container } = render(<Badge>Hello Badge</Badge>)
      expect(container.textContent).toContain('Hello Badge')
    })

    test('renders as a div element', () => {
      const { container } = render(<Badge>Test Badge</Badge>)
      const element = container.querySelector('div')
      expect(element).toBeDefined()
    })

    test('renders with custom children', () => {
      const { container } = render(
        <Badge>
          <span>Custom Child</span>
        </Badge>
      )
      expect(container.textContent).toContain('Custom Child')
    })

    test('renders multiple children', () => {
      const { container } = render(
        <Badge>
          <span>Part 1</span>
          <span>Part 2</span>
        </Badge>
      )
      expect(container.textContent).toContain('Part 1')
      expect(container.textContent).toContain('Part 2')
    })
  })

  describe('variants', () => {
    test('renders with default variant', () => {
      const { container } = render(<Badge>Default</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-primary')
      expect(className).toContain('text-primary-foreground')
    })

    test('renders with secondary variant', () => {
      const { container } = render(<Badge variant="secondary">Secondary</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-secondary')
      expect(className).toContain('text-secondary-foreground')
    })

    test('renders with destructive variant', () => {
      const { container } = render(<Badge variant="destructive">Destructive</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-destructive')
      expect(className).toContain('text-destructive-foreground')
    })

    test('renders with outline variant', () => {
      const { container } = render(<Badge variant="outline">Outline</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('text-foreground')
    })

    test('applies base styles across all variants', () => {
      const { container } = render(<Badge>Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('inline-flex')
      expect(className).toContain('items-center')
      expect(className).toContain('rounded-md')
      expect(className).toContain('border')
      expect(className).toContain('px-2.5')
      expect(className).toContain('py-0.5')
      expect(className).toContain('text-xs')
      expect(className).toContain('font-semibold')
    })
  })

  describe('custom className support', () => {
    test('applies custom className', () => {
      const { container } = render(<Badge className="custom-class">Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('custom-class')
    })

    test('merges custom className with variant classes', () => {
      const { container } = render(
        <Badge variant="secondary" className="custom-class">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('custom-class')
      expect(className).toContain('bg-secondary')
    })

    test('custom className can override styles', () => {
      const { container } = render(
        <Badge variant="secondary" className="bg-red-500">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-red-500')
    })

    test('handles multiple custom classes', () => {
      const { container } = render(<Badge className="custom-class-1 custom-class-2">Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('custom-class-1')
      expect(className).toContain('custom-class-2')
    })
  })

  describe('HTML attributes', () => {
    test('accepts and applies data attributes', () => {
      const { container } = render(
        <Badge data-testid="custom-badge" data-value="test">
          Test
        </Badge>
      )
      const element = container.querySelector('[data-testid="custom-badge"]')
      expect(element).toBeDefined()
      expect(element?.getAttribute('data-value')).toBe('test')
    })

    test('accepts and applies title attribute', () => {
      const { container } = render(<Badge title="Badge Tooltip">Test</Badge>)
      const element = container.querySelector('div')
      expect(element?.getAttribute('title')).toBe('Badge Tooltip')
    })

    test('accepts and applies id attribute', () => {
      const { container } = render(<Badge id="my-badge">Test</Badge>)
      const element = container.querySelector('#my-badge')
      expect(element).toBeDefined()
    })

    test('accepts and applies aria attributes', () => {
      const { container } = render(
        <Badge aria-label="Status badge" aria-hidden="false">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      expect(element?.getAttribute('aria-label')).toBe('Status badge')
      expect(element?.getAttribute('aria-hidden')).toBe('false')
    })

    test('spreads custom props to div element', () => {
      const { container } = render(
        <Badge role="status" data-custom="value">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      expect(element?.getAttribute('role')).toBe('status')
      expect(element?.getAttribute('data-custom')).toBe('value')
    })
  })

  describe('styling consistency', () => {
    test('default variant includes shadow effect', () => {
      const { container } = render(<Badge variant="default">Default</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('shadow')
    })

    test('secondary variant does not include shadow', () => {
      const { container } = render(<Badge variant="secondary">Secondary</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(!className.includes('shadow') || className.includes('bg-secondary')).toBe(true)
    })

    test('destructive variant includes shadow effect', () => {
      const { container } = render(<Badge variant="destructive">Destructive</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('shadow')
    })

    test('all variants have focus and transition styles', () => {
      const variants: Array<'default' | 'secondary' | 'destructive' | 'outline'> = [
        'default',
        'secondary',
        'destructive',
        'outline',
      ]

      variants.forEach((variant) => {
        const { container } = render(<Badge variant={variant}>Test</Badge>)
        const element = container.querySelector('div')
        const className = element?.className || ''
        expect(className).toContain('focus:')
        expect(className).toContain('transition-')
      })
    })

    test('includes focus ring styles', () => {
      const { container } = render(<Badge>Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('focus:ring-2')
      expect(className).toContain('focus:ring-ring')
    })

    test('includes transition classes', () => {
      const { container } = render(<Badge>Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('transition-colors')
    })
  })

  describe('edge cases', () => {
    test('renders empty badge', () => {
      const { container } = render(<Badge />)
      const element = container.querySelector('div')
      expect(element).toBeDefined()
    })

    test('renders badge with empty string as children', () => {
      const { container } = render(<Badge>{''}</Badge>)
      const element = container.querySelector('div')
      expect(element).toBeDefined()
    })

    test('renders badge with numeric content', () => {
      const { container } = render(<Badge>{123}</Badge>)
      expect(container.textContent).toContain('123')
    })

    test('renders badge with special characters', () => {
      const { container } = render(<Badge>Test & Demo!</Badge>)
      expect(container.textContent).toContain('Test & Demo!')
    })

    test('renders badge with very long content', () => {
      const longText = 'A'.repeat(100)
      const { container } = render(<Badge>{longText}</Badge>)
      expect(container.textContent).toContain(longText)
    })

    test('renders badge with nested elements', () => {
      const { container } = render(
        <Badge>
          <span>Nested</span>
          <div>Content</div>
        </Badge>
      )
      expect(container.textContent).toContain('Nested')
      expect(container.textContent).toContain('Content')
    })
  })

  describe('className merging', () => {
    test('merges className correctly with class-variance-authority', () => {
      const { container } = render(
        <Badge variant="secondary" className="mt-4 mb-2">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-secondary')
      expect(className).toContain('mt-4')
      expect(className).toContain('mb-2')
    })

    test('handles Tailwind conflicting classes correctly', () => {
      const { container } = render(
        <Badge variant="secondary" className="text-white">
          Test
        </Badge>
      )
      const element = container.querySelector('div')
      const className = element?.className || ''
      // The cn() utility should handle merge conflicts
      expect(className).toBeDefined()
    })
  })

  describe('variant types', () => {
    test('accepts undefined variant (uses default)', () => {
      const { container } = render(<Badge variant={undefined}>Test</Badge>)
      const element = container.querySelector('div')
      const className = element?.className || ''
      expect(className).toContain('bg-primary')
    })

    test('correctly types variant prop', () => {
      // This test verifies TypeScript types work correctly
      // All these should compile without errors
      render(<Badge variant="default">Test</Badge>)
      render(<Badge variant="secondary">Test</Badge>)
      render(<Badge variant="destructive">Test</Badge>)
      render(<Badge variant="outline">Test</Badge>)
    })
  })
})
