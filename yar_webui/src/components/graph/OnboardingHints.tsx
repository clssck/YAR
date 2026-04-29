/**
 * Public re-export shim for the onboarding component.
 *
 * The implementation lives in `OnboardingHintsImpl` so consumer tests
 * (e.g. GraphViewer) can mock `@/components/graph/OnboardingHints`
 * without preventing the OnboardingHints test suite from importing
 * the real implementation directly.
 */
export {
  default,
  ONBOARDING_KEY,
  ONBOARDING_RESET_EVENT,
} from './OnboardingHintsImpl'
