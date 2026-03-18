import '@/lib/extensions' // Import all global extensions
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import {
  Route,
  HashRouter as Router,
  Routes,
  useNavigate,
} from 'react-router-dom'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from 'sonner'
import ThemeProvider from '@/components/ThemeProvider'
import ComponentDemo from '@/features/ComponentDemo'
import LoginPage from '@/features/LoginPage'
import { navigationService } from '@/services/navigation'
import { useAuthStore } from '@/stores/state'
import App from './App'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1 minute
      gcTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-8">
      <h1 className="text-2xl font-bold text-destructive">Something went wrong</h1>
      <pre className="max-w-lg overflow-auto rounded-md bg-muted p-4 text-sm">
        {error.message}
      </pre>
      <button
        onClick={resetErrorBoundary}
        className="rounded-md bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
      >
        Try again
      </button>
    </div>
  )
}

const AppContent = () => {
  const [initializing, setInitializing] = useState(true)
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Set navigate function for navigation service
  useEffect(() => {
    navigationService.setNavigate(navigate)
  }, [navigate])

  // Token validity check
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('YAR-API-TOKEN')

        if (token && isAuthenticated) {
          setInitializing(false)
          return
        }

        if (!token) {
          useAuthStore.getState().logout()
        }
      } catch (error) {
        console.error('Auth initialization error:', error)
        if (!isAuthenticated) {
          useAuthStore.getState().logout()
        }
      } finally {
        setInitializing(false)
      }
    }

    checkAuth()

    return () => {}
  }, [isAuthenticated])

  // Redirect effect for protected routes
  useEffect(() => {
    if (!initializing && !isAuthenticated) {
      const currentPath = window.location.hash.slice(1)
      // Don't redirect for public routes (login, demo)
      const publicRoutes = ['/login', '/demo']
      if (!publicRoutes.some((route) => currentPath.startsWith(route))) {
        navigate('/login')
      }
    }
  }, [initializing, isAuthenticated, navigate])

  // Show nothing while initializing
  if (initializing) {
    return null
  }

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        {/* Component demo page - accessible without auth for testing */}
        <Route path="/demo" element={<ComponentDemo />} />
        <Route path="/*" element={isAuthenticated ? <App /> : null} />
      </Routes>
    </ErrorBoundary>
  )
}

const AppRouter = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Router>
          <AppContent />
          <Toaster
            position="bottom-center"
            theme="system"
            closeButton
            richColors
          />
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default AppRouter
