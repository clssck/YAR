import type { ButtonVariantType } from '@/components/ui/Button'

// API calls need to go one level up from /webui/ to reach the API at /
// e.g., from /proxy/9621/webui/ -> /proxy/9621/health
export const backendBaseUrl = '..'
// Use relative base './' so assets work behind reverse proxies with path prefixes
export const webuiPrefix = './'

export const controlButtonVariant: ButtonVariantType = 'ghost'

// Dark theme graph palette tuned for contrast on charcoal backgrounds
export const labelColorDarkTheme = '#E5ECFF'
export const LabelColorHighlightedDarkTheme = '#0F172A'
export const labelColorLightTheme = '#000'

export const nodeColorDisabled = '#9CA3AF'
export const nodeBorderColor = '#CBD5E1'
export const nodeBorderColorSelected = '#F97316'
export const nodeBorderColorHiddenConnections = '#F59E0B' // Amber color for nodes with hidden connections

export const edgeColorDarkTheme = '#4B5563'
export const edgeColorSelected = '#F97316'
export const edgeColorHighlightedDarkTheme = '#F59E0B'
export const edgeColorHighlightedLightTheme = '#F57F17'

export const searchResultLimit = 50
export const labelListLimit = 100

// Search History Configuration
export const searchHistoryMaxItems = 500
export const searchHistoryVersion = '1.0'

// API Request Limits
export const popularLabelsDefaultLimit = 300
export const searchLabelsDefaultLimit = 50

// UI Display Limits
export const dropdownDisplayLimit = 300

export const minNodeSize = 4
export const maxNodeSize = 20

export const healthCheckInterval = 15 // seconds

export const defaultQueryLabel = '*'

// Supported file types - must match backend (document_routes.py DocumentManager)
// Reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/Common_types
export const supportedFileTypes = {
  // Office documents
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/msword': ['.doc'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-excel': ['.xls'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/vnd.ms-powerpoint': ['.ppt'],
  'application/vnd.oasis.opendocument.text': ['.odt'],
  'application/vnd.oasis.opendocument.spreadsheet': ['.ods'],
  'application/vnd.oasis.opendocument.presentation': ['.odp'],
  'application/rtf': ['.rtf'],
  // Ebooks
  'application/epub+zip': ['.epub'],
  'application/x-mobipocket-ebook': ['.mobi'],
  // Markup & Text
  'text/html': ['.html', '.htm'],
  'text/markdown': ['.md'],
  'text/x-rst': ['.rst'],
  'application/x-tex': ['.tex'],
  'text/asciidoc': ['.asciidoc'],
  // Data formats
  'application/json': ['.json'],
  'application/xml': ['.xml'],
  'application/x-yaml': ['.yaml', '.yml'],
  'text/csv': ['.csv'],
  'text/tab-separated-values': ['.tsv'],
  // Email
  'message/rfc822': ['.eml'],
  'application/vnd.ms-outlook': ['.msg'],
}

export const SiteInfo = {
  name: 'LightRAG',
  home: '/',
  github: 'https://github.com/HKUDS/LightRAG',
}
