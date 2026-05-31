import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import PublicApp from './PublicApp.jsx'
import './styles.css'

// Two surfaces, one bundle: /a/<slug> is the public ask-only assistant;
// everything else is the admin console.
const isPublic = window.location.pathname.startsWith('/a/')

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {isPublic ? <PublicApp /> : <App />}
  </React.StrictMode>
)
