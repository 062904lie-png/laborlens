/* Browser speech-to-text. Recording starts only on an explicit microphone click. */
(function () {
  let recognition = null, baseText = '', inputWasReadOnly = false, timer = null;
  const micIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></svg>';
  const input = () => document.getElementById('msg-input') || document.getElementById('message-input');
  const status = () => document.getElementById('voice-state');
  function notify(text) {
    if (status()) status().textContent = text;
    const panel = document.getElementById('voice-recording-tools');
    if (panel) panel.hidden = !text;
  }
  function refresh(active) {
    const button = document.getElementById('mic-btn');
    if (!button) return;
    button.classList.toggle('listening', active);
    button.setAttribute('aria-pressed', String(active));
    button.setAttribute('aria-label', active ? 'Stop voice input' : 'Start voice input');
    button.title = active ? 'Stop and keep transcript' : 'Speak your question';
    button.innerHTML = active ? '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/></svg>' : micIcon;
    const cancel = document.getElementById('voice-cancel');
    if (cancel) cancel.hidden = !active;
  }
  function changed() {
    const field = input();
    if (!field) return;
    field.dispatchEvent(new Event('input', {bubbles:true}));
    if (typeof window.autoResize === 'function') window.autoResize(field);
    if (typeof window.updateSendState === 'function') window.updateSendState();
  }
  function detach() {
    const previous = recognition;
    recognition = null; // Ignore late events after cancellation, sending, or navigation.
    clearTimeout(timer);
    if (input() && previous) input().readOnly = inputWasReadOnly;
    refresh(false);
    return previous;
  }
  function finish() {
    const previous = detach();
    if (previous) { try { previous.abort(); } catch (_) {} }
    notify('');
  }
  function cancel() {
    const wasActive = Boolean(recognition);
    finish();
    if (wasActive && input()) { input().value = baseText; changed(); }
    input()?.focus();
  }
  function fail(message) {
    finish();
    notify(message);
  }
  function toggle() {
    if (recognition) {
      notify('Finishing transcription…');
      try { recognition.stop(); } catch (_) { finish(); }
      clearTimeout(timer);
      timer = setTimeout(() => { finish(); notify('Voice input stopped. Review your question before sending.'); }, 3000);
      return;
    }
    const field = input();
    if (!field) return;
    if (document.getElementById('app')?.dataset.responding === 'true') {
      notify('Please wait for the current reply before recording.'); return;
    }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition || !window.isSecureContext) {
      notify(!window.isSecureContext ? 'Voice input needs HTTPS or localhost.' : 'Speech recognition is unavailable in this browser. Try a supported Chrome or Edge browser, or type your question.');
      return;
    }
    baseText = field.value;
    inputWasReadOnly = field.readOnly;
    const selected = document.getElementById('topbar-lang')?.value || document.getElementById('lang-selector')?.value || 'en';
    const locale = {en:'en-PH',fil:'fil-PH',hil:'hil-PH',auto:document.documentElement.lang || 'en-PH'}[selected] || 'en-PH';
    try {
      const current = new Recognition();
      recognition = current;
      current.lang = locale;
      current.continuous = false;
      current.interimResults = true;
      current.maxAlternatives = 1;
      field.readOnly = true;
      current.onstart = () => {
        if (recognition !== current) return;
        notify('Listening… Speak your question. Your browser may process audio online.');
        clearTimeout(timer);
        timer = setTimeout(() => { if (recognition === current) toggle(); }, 90000);
      };
      current.onresult = event => {
        if (recognition !== current) return;
        const transcript = Array.from(event.results || []).map(result => result[0]?.transcript || '').join(' ').trim();
        field.value = [baseText.trim(), transcript].filter(Boolean).join(' ');
        changed();
      };
      current.onerror = event => {
        if (recognition !== current) return;
        const errors = {
          'not-allowed':'Microphone access was denied. Allow microphone access in your browser and try again.',
          'service-not-allowed':'Speech recognition is blocked by this browser or device policy.',
          'audio-capture':'No microphone is available. Connect a microphone and try again.',
          'no-speech':'No speech was detected. Please try again.',
          'network':'Speech recognition could not connect. Check your internet connection.',
          'language-not-supported':'Your browser cannot recognize this language. Choose another language or type your question.',
          'aborted':'Voice input stopped.'
        };
        fail(errors[event.error] || 'Voice input failed. Please try again or type your question.');
      };
      current.onend = () => {
        if (recognition !== current) return;
        detach();
        notify(field.value.trim() !== baseText.trim() ? 'Transcript ready. Review it, then press Send.' : 'No speech was detected. Please try again.');
        changed();
        field.focus();
      };
      refresh(true);
      notify('Requesting microphone access. Your browser may process audio online…');
      timer = setTimeout(() => fail('Microphone access timed out. Please try again.'), 20000);
      current.start();
    } catch (_) {
      fail('Voice input could not start. Please check microphone access and try again.');
    }
  }
  function mount() {
    const button = document.getElementById('mic-btn');
    if (!button || button.dataset.voiceBound === '1') return;
    button.dataset.voiceBound = '1';
    button.addEventListener('click', toggle);
    document.getElementById('voice-cancel')?.addEventListener('click', cancel);
    refresh(false);
  }
  window.LaborLensVoice = {mount, toggle, cancel, finish};
  document.addEventListener('DOMContentLoaded', mount);
  window.addEventListener('pagehide', finish);
  document.addEventListener('visibilitychange', () => { if (document.hidden) finish(); });
})();
