(() => {
  const button = document.querySelector('.admin-password-toggle');
  const input = document.getElementById('l-pass');
  if (!button || !input) return;
  button.addEventListener('click', () => {
    const reveal = input.type === 'password';
    input.type = reveal ? 'text' : 'password';
    button.setAttribute('aria-pressed', String(reveal));
    button.setAttribute('aria-label', reveal ? 'Hide password' : 'Show password');
    button.querySelector('.admin-eye-slash').style.display = reveal ? 'none' : '';
  });
})();
