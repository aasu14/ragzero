import { useEffect } from 'react'

/* Glass dialog. Closes on Escape or a click on the backdrop (not on a
   drag that merely ends there). Locks body scroll while open. */
export default function Modal({ icon, title, subtitle, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  return (
    <div
      className="modal-overlay"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal-head">
          {icon && <div className="m-ico">{icon}</div>}
          <div>
            <div className="m-title">{title}</div>
            {subtitle && <div className="m-sub">{subtitle}</div>}
          </div>
          <button className="modal-close ghost" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}
