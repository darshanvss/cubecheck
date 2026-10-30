'use client'

import { useSyncExternalStore } from 'react'
import { Button } from '@/components/ui/button'
import { getTheme, setTheme, subscribeTheme, type Theme } from '@/lib/theme'

const getServerTheme = (): Theme => 'system'

export function ThemeToggle({ showLabel = false }: { showLabel?: boolean }) {
  const theme = useSyncExternalStore(subscribeTheme, getTheme, getServerTheme)
  const nextTheme: Theme = theme === 'system' ? 'light' : theme === 'light' ? 'dark' : 'system'
  const label = `Theme: ${theme}. Switch to ${nextTheme} theme`

  return <Button type="button" variant="outline" size={showLabel ? 'default' : 'icon'} className={showLabel ? 'h-11 min-w-32 justify-start gap-2 rounded-xl px-3' : 'size-11'} aria-label={label} title={label} onClick={() => setTheme(nextTheme)}>
    <svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      {theme === 'light' ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></> : theme === 'dark' ? <path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z" /> : <><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M8 21h8m-4-4v4" /></>}
    </svg>
    {showLabel && <span>{theme === 'system' ? 'System' : theme === 'light' ? 'Light' : 'Dark'}</span>}
  </Button>
}
