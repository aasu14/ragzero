import { useState } from 'react'
import { adminToken } from '../api'

/* Shown when the admin API returns 401 — this server was started with an
   admin token. Stores the entered token and retries. */
export default function AdminGate({ onUnlock }) {
  const [value, setValue] = useState('')
  const submit = () => {
    if (!value.trim()) return
    adminToken.set(value.trim())
    onUnlock()
  }
  return (
    <div className="modal-overlay">
      <div className="modal" style={{ maxWidth: 420 }}>
        <div className="modal-head">
          <div className="m-ico">🔒</div>
          <div>
            <div className="m-title">Admin access</div>
            <div className="m-sub">this console is protected</div>
          </div>
        </div>
        <div className="modal-body">
          <div className="faint" style={{ fontSize: 12, marginBottom: 12 }}>
            Enter the admin token this server was started with
            (<span className="kbd">--admin-token</span>). Visitors using the public
            assistant link don’t need this.
          </div>
          <input
            type="password" value={value} autoFocus placeholder="Admin token"
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') submit() }}
          />
          <div className="row" style={{ marginTop: 14 }}>
            <button className="primary" onClick={submit} disabled={!value.trim()}>Unlock</button>
          </div>
        </div>
      </div>
    </div>
  )
}
