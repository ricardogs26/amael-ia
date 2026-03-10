'use client'

const BACKEND = 'https://amael-ia.richardx.dev/api'

// Google logo SVG
function GoogleLogo() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.716v2.259h2.908C16.658 14.017 17.64 11.71 17.64 9.2z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  )
}

export default function Login() {
  return (
    <div className="login-root" style={{
      display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: '24px 16px',
      background: 'var(--bg-base)',
    }}>
      <div className="fade-up" style={{
        width: '100%', maxWidth: '400px',
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: '14px',
        padding: '36px 28px 28px',
      }}>
        {/* Logo mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'var(--primary)', display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontSize: '16px', fontWeight: 700, color: '#fff',
          }}>A</div>
          <span style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>
            Amael
          </span>
        </div>

        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '32px' }}>
          Tu asistente inteligente personal
        </p>

        {/* What's new badge */}
        <div style={{
          background: 'var(--primary-subtle)', border: '1px solid rgba(99,102,241,0.2)',
          borderRadius: '8px', padding: '10px 14px', marginBottom: '24px', fontSize: '13px',
          color: 'var(--text-secondary)',
        }}>
          <span style={{ color: 'var(--primary)', fontWeight: 600 }}>Nuevo · </span>
          Conversaciones persistentes, modo streaming y modo claro/oscuro
        </div>

        <a
          href={`${BACKEND}/auth/login`}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
            width: '100%', padding: '14px 20px', background: '#ffffff', color: '#1f2937',
            fontSize: '16px', fontWeight: 500, borderRadius: '10px', textDecoration: 'none',
            border: '1px solid #dadce0', cursor: 'pointer', transition: 'box-shadow .15s',
            minHeight: '52px',
          }}
          onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 1px 6px rgba(60,64,67,.15)')}
          onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
        >
          <GoogleLogo />
          Continuar con Google
        </a>

        <div style={{ height: '1px', background: 'var(--border)', margin: '24px 0' }} />

        <p style={{ fontSize: '12px', color: 'var(--text-disabled)', textAlign: 'center' }}>
          🔒 OAuth 2.0 · No almacenamos contraseñas
        </p>
      </div>
    </div>
  )
}
