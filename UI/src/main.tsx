import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { apiClient } from './services/ApiClient.ts'
import App from './App.tsx'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App apiClient={apiClient} />
  </StrictMode>,
)
