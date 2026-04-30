import { useTranslation } from 'react-i18next'

/**
 * Hook returning a function that translates a graph property name via i18next,
 * falling back to the raw key if no translation exists. Centralises the
 * pattern that was previously duplicated across PropertiesView,
 * PropertyEditDialog, and PropertyRowComponents.
 */
export const useTranslatedPropertyName = () => {
  const { t } = useTranslation()
  return (name: string): string => {
    const key = `graphPanel.propertiesView.node.propertyNames.${name}`
    const translation = t(key)
    return translation === key ? name : translation
  }
}
