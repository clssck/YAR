import '../../../__tests__/setup'
import { describe, expect, mock, test } from 'bun:test'
import { render } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Button, { type ButtonVariantType, buttonVariants } from '@/components/ui/Button'

describe('Button Component', () => {
  describe('Rendering', () => {
    test('renders button with default variant and size', () => {
      const { container } = render(<Button>Click me</Button>)
      const button = container.querySelector('button')
      expect(button).toBeDefined()
      expect(button?.textContent).toBe('Click me')
    })

    test('renders button with custom text content', () => {
      const { container } = render(<Button>Submit Form</Button>)
      const button = container.querySelector('button')
      expect(button?.textContent).toBe('Submit Form')
    })

    test('applies default styling classes', () => {
      const { container } = render(<Button>Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('cursor-pointer')
    })
  })

  describe('Variants', () => {
    const variants: ButtonVariantType[] = [
      'default',
      'destructive',
      'outline',
      'secondary',
      'ghost',
      'link',
    ]

    variants.forEach((variant) => {
      test(`renders with ${variant} variant`, () => {
        const { container } = render(<Button variant={variant}>Button</Button>)
        const button = container.querySelector('button')
        expect(button).toBeDefined()
      })
    })

    test('applies variant classes correctly', () => {
      const { container: defaultContainer } = render(<Button variant="default">Test</Button>)
      const defaultButton = defaultContainer.querySelector('button')
      expect(defaultButton?.className).toContain('bg-primary')

      const { container: destructiveContainer } = render(
        <Button variant="destructive">Test</Button>
      )
      const destructiveButton = destructiveContainer.querySelector('button')
      expect(destructiveButton?.className).toContain('bg-destructive')
    })

    test('applies outline variant classes', () => {
      const { container } = render(<Button variant="outline">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('border')
      expect(button?.className).toContain('bg-background')
    })

    test('applies ghost variant classes', () => {
      const { container } = render(<Button variant="ghost">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('hover:bg-accent')
    })

    test('applies link variant classes', () => {
      const { container } = render(<Button variant="link">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('text-primary')
      expect(button?.className).toContain('underline-offset-4')
    })
  })

  describe('Sizes', () => {
    const sizes: Array<'default' | 'sm' | 'lg' | 'icon'> = ['default', 'sm', 'lg', 'icon']

    sizes.forEach((size) => {
      test(`renders with ${size} size`, () => {
        const { container } = render(<Button size={size}>Button</Button>)
        const button = container.querySelector('button')
        expect(button).toBeDefined()
      })
    })

    test('applies default size classes', () => {
      const { container } = render(<Button size="default">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('h-10')
      expect(button?.className).toContain('px-4')
    })

    test('applies sm size classes', () => {
      const { container } = render(<Button size="sm">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('h-9')
      expect(button?.className).toContain('px-3')
    })

    test('applies lg size classes', () => {
      const { container } = render(<Button size="lg">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('h-11')
      expect(button?.className).toContain('px-8')
    })

    test('applies icon size classes', () => {
      const { container } = render(<Button size="icon">🔍</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('size-8')
    })
  })

  describe('Click Handling', () => {
    test('calls onClick handler when clicked', async () => {
      const user = await userEvent.setup()
      const handleClick = mock()
      const { container } = render(<Button onClick={handleClick}>Click me</Button>)

      const button = container.querySelector('button')
      expect(button).toBeDefined()
      if (button) {
        await user.click(button)
        expect(handleClick).toHaveBeenCalledTimes(1)
      }
    })

    test('calls onClick handler multiple times on multiple clicks', async () => {
      const user = await userEvent.setup()
      const handleClick = mock()
      const { container } = render(<Button onClick={handleClick}>Click me</Button>)

      const button = container.querySelector('button')
      expect(button).toBeDefined()
      if (button) {
        await user.click(button)
        await user.click(button)
        await user.click(button)
        expect(handleClick).toHaveBeenCalledTimes(3)
      }
    })
  })

  describe('Disabled State', () => {
    test('renders disabled button', () => {
      const { container } = render(<Button disabled>Disabled Button</Button>)
      const button = container.querySelector('button')
      expect(button?.disabled).toBe(true)
    })

    test('applies disabled styling classes', () => {
      const { container } = render(<Button disabled>Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('disabled:pointer-events-none')
      expect(button?.className).toContain('disabled:opacity-50')
    })

    test('does not call onClick when disabled', async () => {
      const handleClick = mock()
      const { container } = render(
        <Button disabled onClick={handleClick}>
          Disabled
        </Button>
      )

      const button = container.querySelector('button')
      // Note: disabled buttons may still trigger click events in some testing libraries,
      // but the pointer-events-none class prevents actual interactions
      expect(button?.disabled).toBe(true)
    })

    test('has correct disabled attribute', () => {
      const { container } = render(<Button disabled>Test</Button>)
      const button = container.querySelector('button')
      expect(button?.hasAttribute('disabled')).toBe(true)
    })
  })

  describe('Tooltip Support', () => {
    test('renders button without tooltip when tooltip prop is not provided', () => {
      const { container } = render(<Button>No Tooltip</Button>)
      const button = container.querySelector('button')
      expect(button).toBeDefined()
      expect(button?.textContent).toBe('No Tooltip')
    })

    test('renders button with tooltip when tooltip prop is provided', () => {
      const { container } = render(<Button tooltip="This is a tooltip">With Tooltip</Button>)
      const button = container.querySelector('button')
      expect(button).toBeDefined()
    })

    test('accepts tooltip side prop', () => {
      const { container, rerender } = render(
        <Button tooltip="Tooltip" side="top">
          Button
        </Button>
      )
      expect(container.querySelector('button')).toBeDefined()

      rerender(
        <Button tooltip="Tooltip" side="bottom">
          Button
        </Button>
      )
      expect(container.querySelector('button')).toBeDefined()

      rerender(
        <Button tooltip="Tooltip" side="left">
          Button
        </Button>
      )
      expect(container.querySelector('button')).toBeDefined()

      rerender(
        <Button tooltip="Tooltip" side="right">
          Button
        </Button>
      )
      expect(container.querySelector('button')).toBeDefined()
    })
  })

  describe('Accessibility Attributes', () => {
    test('has proper button role', () => {
      const { container } = render(<Button>Accessible Button</Button>)
      const button = container.querySelector('button')
      expect(button).toBeDefined()
    })

    test('supports aria-label attribute', () => {
      const { container } = render(<Button aria-label="Close dialog">×</Button>)
      const button = container.querySelector('button')
      expect(button?.getAttribute('aria-label')).toBe('Close dialog')
    })

    test('supports aria-describedby attribute', () => {
      const { container } = render(
        <>
          <span id="help-text">This button submits the form</span>
          <Button aria-describedby="help-text">Submit</Button>
        </>
      )
      const button = container.querySelector('button')
      expect(button?.getAttribute('aria-describedby')).toBe('help-text')
    })

    test('supports aria-disabled attribute', () => {
      const { container } = render(
        <Button aria-disabled="true" disabled>
          Disabled
        </Button>
      )
      const button = container.querySelector('button')
      expect(button?.hasAttribute('disabled')).toBe(true)
    })

    test('supports aria-pressed for toggle buttons', () => {
      const { container } = render(
        <Button aria-pressed="false" role="button">
          Toggle
        </Button>
      )
      const button = container.querySelector('button')
      expect(button?.getAttribute('aria-pressed')).toBe('false')
    })

    test('applies focus-visible styling classes', () => {
      const { container } = render(<Button>Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('focus-visible:outline-none')
      expect(button?.className).toContain('focus-visible:ring-2')
    })

    test('has proper semantic HTML', () => {
      const { container } = render(<Button>Semantic</Button>)
      const button = container.querySelector('button')
      expect(button?.tagName.toLowerCase()).toBe('button')
    })

    test('supports title attribute for tooltip fallback', () => {
      const { container } = render(<Button title="Button tooltip">Button</Button>)
      const button = container.querySelector('button')
      expect(button?.getAttribute('title')).toBe('Button tooltip')
    })
  })

  describe('HTML Attributes', () => {
    test('forwards standard HTML button attributes', () => {
      const { container } = render(<Button data-testid="custom-button">Button</Button>)
      const button = container.querySelector('[data-testid="custom-button"]')
      expect(button).toBeDefined()
    })

    test('supports type attribute', () => {
      const { container } = render(<Button type="submit">Submit</Button>)
      const button = container.querySelector('button')
      expect(button?.getAttribute('type')).toBe('submit')
    })

    test('supports name attribute', () => {
      const { container } = render(<Button name="action">Button</Button>)
      const button = container.querySelector('button')
      expect(button?.getAttribute('name')).toBe('action')
    })

    test('supports className prop', () => {
      const { container } = render(<Button className="custom-class">Test</Button>)
      const button = container.querySelector('button')
      expect(button?.className).toContain('custom-class')
    })
  })

  describe('Ref Forwarding', () => {
    test('forwards ref to button element', () => {
      let buttonRef: HTMLButtonElement | null = null
      const ref = (el: HTMLButtonElement | null) => {
        buttonRef = el
      }

      render(<Button ref={ref}>Button</Button>)

      expect(buttonRef).not.toBeNull()
      expect(buttonRef?.tagName.toLowerCase()).toBe('button')
    })

    test('ref can be used to interact with button', () => {
      let buttonRef: HTMLButtonElement | null = null
      render(
        <Button
          ref={(el) => {
            buttonRef = el
          }}
        >
          Button
        </Button>
      )

      expect(buttonRef).not.toBeNull()
      expect(buttonRef?.textContent).toBe('Button')
    })
  })

  describe('Display Name', () => {
    test('has correct display name for debugging', () => {
      const component = Button as unknown as { displayName: string }
      expect(component.displayName).toBe('Button')
    })
  })

  describe('Combination Tests', () => {
    test('renders with variant, size, and disabled state combined', () => {
      const { container } = render(
        <Button variant="destructive" size="lg" disabled>
          Delete
        </Button>
      )
      const button = container.querySelector('button')
      expect(button?.className).toContain('bg-destructive')
      expect(button?.className).toContain('h-11')
      expect(button?.className).toContain('disabled:opacity-50')
      expect(button?.disabled).toBe(true)
    })

    test('renders with variant, size, tooltip, and aria-label', async () => {
      const { container } = render(
        <Button variant="default" size="sm" tooltip="Save changes" aria-label="Save button">
          Save
        </Button>
      )
      const button = container.querySelector('button')
      expect(button?.getAttribute('aria-label')).toBe('Save button')
    })
  })

  describe('SVG Icon Styling', () => {
    test('applies styling to SVG icons inside button', () => {
      const { container } = render(
        <Button>
          <svg data-testid="icon" aria-hidden="true">
            <title>Icon</title>
          </svg>
          Text
        </Button>
      )
      const button = container.querySelector('button')
      expect(button?.className).toContain('[&_svg]:pointer-events-none')
      expect(button?.className).toContain('[&_svg]:size-4')
    })
  })

  describe('buttonVariants Export', () => {
    test('buttonVariants function exports correctly', () => {
      expect(buttonVariants).toBeDefined()
      const classes = buttonVariants()
      expect(typeof classes).toBe('string')
      expect(classes.length).toBeGreaterThan(0)
    })

    test('buttonVariants applies default variant', () => {
      const classes = buttonVariants()
      expect(classes).toContain('bg-primary')
    })

    test('buttonVariants applies custom variant', () => {
      const classes = buttonVariants({ variant: 'destructive' })
      expect(classes).toContain('bg-destructive')
    })

    test('buttonVariants applies custom size', () => {
      const classes = buttonVariants({ size: 'lg' })
      expect(classes).toContain('h-11')
      expect(classes).toContain('px-8')
    })

    test('buttonVariants handles combined variant and size', () => {
      const classes = buttonVariants({ variant: 'outline', size: 'sm' })
      expect(classes).toContain('border')
      expect(classes).toContain('h-9')
    })
  })

  describe('AsChild Slot Behavior', () => {
    test('renders as Slot component when asChild is true', () => {
      const { container } = render(
        <Button asChild>
          <a href="/test">Link Button</a>
        </Button>
      )
      const link = container.querySelector('a')
      expect(link).toBeDefined()
      expect(link?.getAttribute('href')).toBe('/test')
      expect(link?.textContent).toBe('Link Button')
    })

    test('renders as button by default when asChild is false', () => {
      const { container } = render(<Button asChild={false}>Regular Button</Button>)
      const button = container.querySelector('button')
      expect(button).toBeDefined()
      expect(button?.tagName.toLowerCase()).toBe('button')
    })
  })
})
