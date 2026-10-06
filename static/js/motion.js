/* Explicit request states: history never replays animations and deltas render immediately. */
(function () {
  const responseTimers = new WeakMap();

  function clockLabel(milliseconds) {
    const total = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return minutes ? `${minutes}:${String(seconds).padStart(2, '0')}` : `0:${String(seconds).padStart(2, '0')}`;
  }

  function elapsedMilliseconds(timer) {
    return (timer.stoppedAt ?? performance.now()) - timer.startedAt;
  }

  function ensureTimerNode(row, bubble, timer, state) {
    const thinking = state === 'thinking';
    const parent = thinking ? bubble.querySelector('.ll-thinking') : bubble;
    if (!parent) return;
    if (!timer.node || !timer.node.isConnected || timer.node.parentElement !== parent) {
      timer.node = document.createElement('span');
      timer.node.className = 'll-response-time';
      timer.node.setAttribute('role', 'timer');
      timer.node.setAttribute('aria-live', 'off');
      parent.appendChild(timer.node);
    }
    timer.node.classList.toggle('ll-response-time-live', thinking || state === 'streaming');
    timer.node.textContent = state === 'complete'
      ? `Answered in ${clockLabel(elapsedMilliseconds(timer))}`
      : state === 'error'
        ? `Stopped after ${clockLabel(elapsedMilliseconds(timer))}`
        : clockLabel(elapsedMilliseconds(timer));
    const elapsedSeconds = Math.floor(elapsedMilliseconds(timer) / 1000);
    timer.node.setAttribute('aria-label', `Elapsed response time: ${elapsedSeconds} seconds`);
  }

  function stopTimer(row, bubble, state) {
    const timer = responseTimers.get(row);
    if (!timer) return;
    if (timer.interval) clearInterval(timer.interval);
    timer.interval = null;
    timer.stoppedAt ??= performance.now();
    ensureTimerNode(row, bubble, timer, state);
  }

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
    if (state === 'thinking' && !responseTimers.has(row)) {
      const timer = { startedAt: performance.now(), stoppedAt: null, interval: null, node: null };
      responseTimers.set(row, timer);
      ensureTimerNode(row, bubble, timer, state);
      timer.interval = setInterval(() => {
        if (!row.isConnected) {
          clearInterval(timer.interval);
          timer.interval = null;
          responseTimers.delete(row);
          return;
        }
        ensureTimerNode(row, bubble, timer, row.dataset.responseState);
      }, 1000);
    } else if ((state === 'streaming' || state === 'thinking') && responseTimers.has(row)) {
      ensureTimerNode(row, bubble, responseTimers.get(row), state);
    } else if (state === 'complete' || state === 'error') {
      stopTimer(row, bubble, state);
    }
  }
  function setResponding(active) {
    const app = document.getElementById('app');
    if (app) app.dataset.responding = String(Boolean(active));
  }
  window.LaborLensMotion = { setReplyState, setResponding };
})();
