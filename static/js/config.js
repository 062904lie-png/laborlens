/**
 * LaborLens – Backend URL Configuration
 * ════════════════════════════════════════════════════════════════
 *  ⚙️  THIS IS THE ONLY FILE YOU EDIT WHEN YOUR NGROK URL CHANGES.
 *
 *  How to get your ngrok URL:
 *    1. Run ngrok (via start_windows.bat or start_unix.sh)
 *    2. Look for the "Forwarding" line, e.g.:
 *         https://abc123-45-67-89.ngrok-free.app -> http://localhost:8000
 *    3. Copy the HTTPS part (without trailing slash)
 *    4. Paste it below, keeping /api at the end
 *
 *  Example:
 *    API_URL: 'https://abc123-45-67-89.ngrok-free.app/api'
 *
 *  💡 TIP: Claim a free static domain at dashboard.ngrok.com so this
 *           URL never changes and you only edit this file once!
 * ════════════════════════════════════════════════════════════════
 */

window.LABORLENS_CONFIG = {
  API_URL: 'https://plaster-multiply-tint.ngrok-free.dev/api'
};
