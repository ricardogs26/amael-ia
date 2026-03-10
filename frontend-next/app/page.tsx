'use client'
import { useEffect, useState, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Login from './components/Login'
import Chat from './components/Chat'

function AppInner() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [token, setToken]       = useState<string | null>(null)
  const [userName, setUserName] = useState('')
  const [userPic, setUserPic]   = useState('')

  useEffect(() => {
    const t    = searchParams.get('token')
    const name = searchParams.get('name')
    const pic  = searchParams.get('picture')
    const err  = searchParams.get('error')

    if (err) { alert('No tienes permiso para acceder.'); router.replace('/'); return }

    if (t) {
      setToken(t)
      if (name) setUserName(decodeURIComponent(name))
      if (pic)  setUserPic(decodeURIComponent(pic))
      localStorage.setItem('amael-token', t)
      localStorage.setItem('amael-name', name ? decodeURIComponent(name) : '')
      localStorage.setItem('amael-picture', pic ? decodeURIComponent(pic) : '')
      router.replace('/')
      return
    }

    const saved = localStorage.getItem('amael-token')
    if (saved) {
      setToken(saved)
      setUserName(localStorage.getItem('amael-name') || '')
      setUserPic(localStorage.getItem('amael-picture') || '')
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const logout = () => {
    ['amael-token','amael-name','amael-picture'].forEach(k => localStorage.removeItem(k))
    setToken(null); setUserName(''); setUserPic('')
  }

  if (!token) return <Login />
  return <Chat token={token} userName={userName} userPicture={userPic} onLogout={logout} />
}

export default function Page() {
  return (
    <Suspense fallback={<div style={{ background: 'var(--bg-base)', height: '100vh' }} />}>
      <AppInner />
    </Suspense>
  )
}
