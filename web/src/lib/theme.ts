export type Theme = 'system' | 'light' | 'dark'

const storageKey = 'cubecheck-theme'
const changeEvent = 'cubecheck-theme-change'

function validTheme(value: string | null | undefined): Theme {
  return value === 'light' || value === 'dark' ? value : 'system'
}

export function getTheme(): Theme {
  return validTheme(document.documentElement.dataset.theme)
}

function applyTheme(theme: Theme) {
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  const root = document.documentElement
  root.dataset.theme = theme
  root.classList.toggle('dark', dark)
  root.style.colorScheme = dark ? 'dark' : 'light'
}

export function setTheme(theme: Theme) {
  applyTheme(theme)
  try { localStorage.setItem(storageKey, theme) } catch { /* The choice still works without storage. */ }
  window.dispatchEvent(new Event(changeEvent))
}

export function subscribeTheme(onChange: () => void) {
  const media = window.matchMedia('(prefers-color-scheme: dark)')
  const onSystemChange = () => { if (getTheme() === 'system') applyTheme('system') }
  const onStorage = (event: StorageEvent) => {
    if (event.storageArea === localStorage && (event.key === storageKey || event.key === null)) {
      applyTheme(validTheme(event.newValue))
      onChange()
    }
  }
  // Reconcile a system change that happened before hydration or while navigating.
  applyTheme(getTheme())
  media.addEventListener('change', onSystemChange)
  window.addEventListener('storage', onStorage)
  window.addEventListener(changeEvent, onChange)
  return () => {
    media.removeEventListener('change', onSystemChange)
    window.removeEventListener('storage', onStorage)
    window.removeEventListener(changeEvent, onChange)
  }
}

// Runs before the page paints so a stored preference never flashes the wrong theme.
export const themeInitScript = `(() => {
  let theme = 'system';
  try {
    const saved = localStorage.getItem('cubecheck-theme');
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch {}
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.classList.toggle('dark', dark);
  root.style.colorScheme = dark ? 'dark' : 'light';
})()`
