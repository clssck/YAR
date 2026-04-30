import { CommandGroup, CommandItem } from '@/components/ui/Command'

/**
 * Generic loading skeleton row used as a fallback for AsyncSearch /
 * AsyncSelect dropdowns while their results are in flight. Was duplicated
 * verbatim across both components before being extracted here.
 */
export default function DefaultLoadingSkeleton() {
  return (
    <CommandGroup>
      <CommandItem disabled>
        <div className="flex w-full items-center gap-2">
          <div className="bg-muted h-6 w-6 animate-pulse rounded-full" />
          <div className="flex flex-1 flex-col gap-1">
            <div className="bg-muted h-4 w-24 animate-pulse rounded" />
            <div className="bg-muted h-3 w-16 animate-pulse rounded" />
          </div>
        </div>
      </CommandItem>
    </CommandGroup>
  )
}
