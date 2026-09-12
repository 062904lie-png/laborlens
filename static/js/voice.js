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
    button.title = active ? 'Finish and send question' : 'Speak your question — sends automatically';
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
    stopSpeaking();
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
    stopSpeaking();
    inputWasReadOnly = field.readOnly;
    const selected = document.getElementById('topbar-lang')?.value || document.getElementById('lang-selector')?.value || 'en';
    const locale = {en:'en-US',fil:'fil-PH',hil:'hil-PH',auto:document.documentElement.lang || 'en-US'}[selected] || 'en-US';
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
        console.log('Speech recognition error:', event.error);
        console.log('Message:', event.message || '(no additional message)');
        if (event.error === 'network') {
          console.log('The browser could not connect to the speech recognition service. In Edge, this is a speech-service connection error.');
        }
        const errors = {
          'not-allowed':'Microphone access was denied. Allow microphone access in your browser and try again.',
          'service-not-allowed':'Speech recognition is blocked by this browser or device policy.',
          'audio-capture':'No microphone is available. Connect a microphone and try again.',
          'no-speech':'No speech was detected. Please try again.',
          'network':'Your browser could not connect to its speech-recognition service. This is different from microphone permission being denied. If other websites load, try restarting or updating the browser, or test voice input in another browser.',
          'language-not-supported':'Your browser cannot recognize this language. Choose another language or type your question.',
          'aborted':'Voice input stopped.'
        };
        fail(errors[event.error] || 'Voice input failed. Please try again or type your question.');
      };
      current.onend = () => {
        if (recognition !== current) return;
        detach();
        changed();
        field.focus();
        if (field.value.trim() !== baseText.trim()) {
          notify('');
          const send = window.sendMsg || window.sendMessage;
          if (typeof send === 'function') send();
        } else notify('No speech was detected. Please try again.');
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
    window.speechSynthesis?.getVoices?.(); // Start asynchronous voice discovery early.
    const button = document.getElementById('mic-btn');
    if (!button || button.dataset.voiceBound === '1') return;
    button.dataset.voiceBound = '1';
    button.addEventListener('click', toggle);
    document.getElementById('voice-cancel')?.addEventListener('click', cancel);
    refresh(false);
  }
  let utterance = null;
  let preferredVoiceURI = '';

  function availableVoices() {
    return window.speechSynthesis?.getVoices?.() || [];
  }

  function voiceId(voice) {
    return String(voice?.voiceURI || `${voice?.name || ''}|${voice?.lang || ''}`);
  }

  function setPreferredVoice(voiceURI) {
    preferredVoiceURI = String(voiceURI || '');
  }

  function cleanAnswer(text) {
    const marker = '(?:Sources?\\s*\\d+|S\\s*\\d+)(?:\\s*(?:,|;|&|and|[-–])\\s*(?:(?:Sources?|S)\\s*)?\\d+)*';
    return String(text || '')
      .replace(new RegExp('\\[' + marker + '\\]\\([^)]*\\)', 'gi'), '')
      .replace(new RegExp('[\\[(]\\s*' + marker + '\\s*[\\])]', 'gi'), '')
      .replace(new RegExp('^\\s*(?:[-*]\\s*)?' + marker + '\\s*:?\\s*$', 'gim'), '')
      .replace(/[ \t]+([.,;:!?])/g, '$1')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\n{3,}/g, '\n\n').trim();
  }
  function stopSpeaking() {
    utterance = null;
    window.speechSynthesis?.cancel();
    const button = document.getElementById('voice-stop-reading');
    if (button) button.hidden = true;
    const playback = document.getElementById('voice-playback');
    if (playback) playback.hidden = true;
  }
  function speak(text, language, anchor) {
    stopSpeaking();
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance || recognition || document.hidden) return;
    const plain = cleanAnswer(text).replace(/```[\s\S]*?```/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/https?:\/\/\S+/g, '')
      .replace(/[*#_`>|]/g, '').trim();
    if (!plain) return;
    // Keep requests short without splitting words, amounts, or decimal numbers.
    const chunks = [];
    let chunk = '';
    for (const word of plain.split(/\s+/)) {
      if (chunk && chunk.length + word.length + 1 > 220) {
        chunks.push(chunk); chunk = '';
      }
      chunk += (chunk ? ' ' : '') + word;
      if (chunk.length >= 100 && /[.!?]$/.test(word)) {
        chunks.push(chunk); chunk = '';
      }
    }
    if (chunk) chunks.push(chunk);
    const speech = new window.SpeechSynthesisUtterance(chunks[0]);
    const requestedLanguage = String(language || document.getElementById('topbar-lang')?.value || document.getElementById('lang-selector')?.value || 'auto').toLowerCase();
    const isEnglish = ['en', 'en-us', 'en-ph', 'english'].includes(requestedLanguage);
    speech.lang = isEnglish ? 'en-US' : 'fil-PH';
    const voices = availableVoices();
    const aliases = isEnglish ? ['en'] : ['fil','tl'];
    // The Web Speech API has no gender field. Prefer known female Philippine
    // voices or voices explicitly named Female, while keeping language matching.
    const candidates = voices.filter(voice => aliases.includes(voice.lang.toLowerCase().split('-')[0]));
    const voiceScore = voice => {
      const locale = voice.lang.toLowerCase();
      const female = /blessica|\brosa\b|\bfemale\b/i.test(voice.name || '');
      return (locale === speech.lang.toLowerCase() ? 20 : 0)
        + (female ? 50 : 0) + (/-ph$/.test(locale) ? 100 : 0);
    };
    const matchingVoice = candidates.sort((a,b) => voiceScore(b) - voiceScore(a))[0];
    const selectedVoice = preferredVoiceURI && voices.find(voice => voiceId(voice) === preferredVoiceURI);
    const availableVoice = selectedVoice || matchingVoice || voices.find(voice => voice.default) || voices[0];
    if (availableVoice) {
      speech.voice = availableVoice;
      speech.lang = availableVoice.lang;
    } else {
      // An empty list can mean voices have not loaded yet. Let the browser
      // choose its default immediately, preserving the mobile tap gesture.
      speech.lang = '';
    }
    notify((selectedVoice || matchingVoice) ? '' : 'Using the device’s available voice. Pronunciation may differ.');
    let button = document.getElementById('voice-stop-reading');
    if (!button) {
      const playback = document.createElement('span');
      playback.id = 'voice-playback';
      playback.innerHTML = '<span class="voice-speaking" role="status" aria-label="Reading answer aloud"><span aria-hidden="true"></span><span aria-hidden="true"></span><span aria-hidden="true"></span><span aria-hidden="true"></span><span aria-hidden="true"></span></span>';
      button = document.createElement('button');
      button.id = 'voice-stop-reading';
      button.type = 'button';
      button.setAttribute('aria-label', 'Stop reading');
      button.title = 'Stop reading';
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
      button.addEventListener('click', stopSpeaking);
      playback.appendChild(button);
      (anchor || document.getElementById('mic-btn'))?.insertAdjacentElement('afterend', playback);
    }
    const playback = document.getElementById('voice-playback');
    if (playback) {
      (anchor || document.getElementById('mic-btn'))?.insertAdjacentElement('afterend', playback);
      playback.hidden = false;
      playback.classList.remove('is-speaking');
    }
    button.hidden = false;
    function playChunk(index) {
      const current = index === 0 ? speech : new window.SpeechSynthesisUtterance(chunks[index]);
      if (availableVoice) current.voice = availableVoice;
      current.lang = speech.lang;
      utterance = current;
      current.onstart = () => { if (utterance === current) playback?.classList.add('is-speaking'); };
      current.onend = () => {
        if (utterance !== current) return;
        if (index + 1 < chunks.length) { playChunk(index + 1); return; }
        utterance = null; button.hidden = true;
        if (playback) playback.hidden = true;
      };
      current.onerror = event => {
        if (utterance !== current) return;
        stopSpeaking();
        console.log('Speech playback error:', event.error, current.lang);
        notify(event.error === 'not-allowed'
          ? 'Audio playback was blocked. Tap the answer’s speaker button to start reading.'
          : `Your device could not play this voice (${event.error || 'unknown error'}). Try another installed speech voice or browser.`);
      };
      try {
        if (window.speechSynthesis.paused) window.speechSynthesis.resume();
        window.speechSynthesis.speak(current);
      } catch (_) { stopSpeaking(); notify('Reading could not start. Tap the answer’s speaker button to try again.'); }
    }
    playChunk(0);
  }
  window.LaborLensVoice = {
    mount, toggle, cancel, finish, speak, stopSpeaking, cleanAnswer,
    availableVoices, setPreferredVoice,
  };
  document.addEventListener('DOMContentLoaded', mount);
  window.addEventListener('pagehide', finish);
  document.addEventListener('visibilitychange', () => { if (document.hidden) finish(); });
})();
