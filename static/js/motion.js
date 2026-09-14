/* Explicit request states: history never replays animations and deltas render immediately. */
(function () {
  function setReplyState(row, state, label = 'Preparing your response…') {
    if (!row) return;
    const bubble = row.querySelector('.msg-bubble, .bubble');
    if (!bubble) return;
    row.dataset.responseState = state;
    // Reveal once, on the first text or a complete non-streamed reply.
    if (state === 'streaming' || state === 'complete') row.dataset.replyReveal = 'true';
    row.setAttribute('aria-busy', String(state === 'thinking' || state === 'streaming'));
    if (state === 'thinking') {
      let status = bubble.querySelector('.ll-thinking');
      if (!status) {
        status = document.createElement('div');
        status.className = 'll-thinking';
        status.setAttribute('role', 'status');
        status.innerHTML = '<span class="ll-thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="ll-thinking-label"></span>';
        bubble.replaceChildren(status);
      }
      status.querySelector('.ll-thinking-label').textContent = label;
    }
  }
  function setResponding(active) {
    const app = document.getElementById('app');
    if (app) app.dataset.responding = String(Boolean(active));
  }
  window.LaborLensMotion = { setReplyState, setResponding };
})();
