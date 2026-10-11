/* ── Startup ── */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeAdminNav();
  if (e.key==='Enter' && !e.repeat && ['l-email','l-pass'].includes(e.target?.id) && !document.getElementById('login-screen').classList.contains('gone')) doLogin();
});
window.addEventListener('load', () => {
  restoreAdminSession();
});
