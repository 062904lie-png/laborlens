/* Accessible password visibility controls for existing account forms. */
(() => {
  const scope = document.getElementById('auth-screen');
  if (!scope) return;
  scope.querySelectorAll('input[type="password"]').forEach(input => {
    const control = document.createElement('div');
    control.className = 'password-control';
    input.before(control);
    control.append(input);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'password-toggle';
    button.setAttribute('aria-label', 'Show password');
    button.setAttribute('aria-controls', input.id);
    button.setAttribute('aria-pressed', 'false');
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/><path class="eye-slash" d="m3 3 18 18"/></svg>';
    button.addEventListener('click', () => {
      const reveal = input.type === 'password';
      input.type = reveal ? 'text' : 'password';
      button.setAttribute('aria-label', reveal ? 'Hide password' : 'Show password');
      button.setAttribute('aria-pressed', String(reveal));
      button.querySelector('.eye-slash').style.display = reveal ? 'none' : '';
    });
    control.append(button);
  });
})();
