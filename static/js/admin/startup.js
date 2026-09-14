/* ── Startup ── */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeAdminNav();
  if (e.key==='Enter' && !document.getElementById('login-screen').classList.contains('gone')) doLogin();
});
window.addEventListener('load', () => {
  restoreAdminSession();
});
