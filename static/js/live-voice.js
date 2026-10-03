/* Native Live audio: explicit consent, server-side keys, no browser speech APIs. */
(() => {
  const assetRoot = new URL('.', document.currentScript.src);
  document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('input-zone');
    if (!zone || document.getElementById('live-voice-open')) return;
    const button = document.createElement('button');
    button.id = 'live-voice-open'; button.type = 'button';
    button.textContent = 'Live voice (preview)';
    button.style.cssText = 'margin:8px 0;padding:8px 12px;border:1px solid #b8cbe1;border-radius:8px;background:#fff;color:#12365a;cursor:pointer';
    zone.append(button);
    const dialog = document.createElement('dialog');
    dialog.style.cssText = 'width:min(560px,90vw);max-height:85vh;overflow:auto;border:1px solid #b8cbe1;border-radius:16px;padding:24px;color:#12365a;background:white';
    dialog.innerHTML = '<h2>Live voice consultation</h2><p>Preview: audio is sent to Google while connected. Legal questions use the LaborLens knowledge base. This separate voice session is not saved to chat history. Use headphones to avoid echo.</p><p role="status" aria-live="polite"></p><button type="button" data-start>Start microphone</button> <button type="button" data-stop>Stop</button> <button type="button" data-close>Close</button><h3>Latest verified answer</h3><div data-evidence style="white-space:pre-wrap;overflow-wrap:anywhere">No answer yet.</div>';
    document.body.append(dialog);
    const status = dialog.querySelector('[role=status]');
    const start = dialog.querySelector('[data-start]');
    const evidence = dialog.querySelector('[data-evidence]');
    let socket, stream, context, capture, input, watchdog, setupTimer, generation = 0;
    let playbackTime = 0, activeSources = new Set();
    function clearPlayback() {
      for (const source of activeSources) { try { source.stop(); } catch (_) {} }
      activeSources.clear(); playbackTime = 0;
    }
    function stop(message = 'Microphone stopped.') {
      generation++;
      clearInterval(watchdog); watchdog = null;
      clearTimeout(setupTimer); setupTimer = null;
      if (socket) { socket.onclose = null; socket.close(); socket = null; }
      if (capture) { capture.disconnect(); capture.port.onmessage = null; capture = null; }
      if (input) { input.disconnect(); input = null; }
      if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
      clearPlayback();
      if (context) { context.close().catch(() => {}); context = null; }
      start.disabled = false; status.textContent = message;
    }
    function play(encoded, mime) {
      if (!context || !mime?.startsWith('audio/pcm')) return;
      const raw = atob(encoded);
      if (raw.length % 2 || raw.length > 512000) return;
      const data = new DataView(Uint8Array.from(raw, c => c.charCodeAt(0)).buffer);
      const rate = Number(/rate=(\d+)/.exec(mime)?.[1] || 24000);
      if (rate < 8000 || rate > 48000) return;
      const buffer = context.createBuffer(1, raw.length / 2, rate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < channel.length; i++) channel[i] = data.getInt16(i * 2, true) / 32768;
      const source = context.createBufferSource();
      source.buffer = buffer; source.connect(context.destination);
      playbackTime = Math.max(playbackTime, context.currentTime + 0.03);
      if (playbackTime - context.currentTime > 60) { stop('Audio queue exceeded its limit. Please reconnect.'); return; }
      source.start(playbackTime); playbackTime += buffer.duration;
      activeSources.add(source); source.onended = () => activeSources.delete(source);
    }
    start.onclick = async () => {
      const auth = localStorage.getItem('ll_token');
      if (!auth) { status.textContent = 'Please sign in to use live voice.'; return; }
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
        status.textContent = 'Live voice requires HTTPS or localhost and a browser with AudioWorklet support.'; return;
      }
      stop(); const run = generation; start.disabled = true;
      status.textContent = 'Connecting…';
      evidence.textContent = 'No answer yet.';
      try {
        context = new AudioContext({sampleRate: 16000});
        await context.resume();
        if (run !== generation) return;
        if (context.sampleRate !== 16000) throw new Error('Unsupported microphone sample rate');
        await context.audioWorklet.addModule(new URL('live-capture.js', assetRoot));
        if (run !== generation) return;
        const acquired = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
        if (run !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
        stream = acquired;
        const base = typeof API !== 'undefined' ? API : window.LABORLENS_CONFIG?.API_URL;
        const url = new URL(base.replace(/\/$/, '') + '/voice/live', location.href);
        url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(url);
        setupTimer = setTimeout(() => {
          if (run === generation) stop('Voice setup timed out. Please use text chat or reconnect.');
        }, 40000);
        const current = socket;
        socket.onopen = () => current.send(JSON.stringify({token: auth}));
        socket.onerror = () => { if (run === generation) stop('Unable to connect. Check the server and Live model access.'); };
        socket.onclose = () => { if (run === generation) stop('Voice session ended. You can reconnect or use text chat.'); };
        socket.onmessage = event => {
          if (run !== generation) return;
          try {
            const data = JSON.parse(event.data);
            if (data.error) { stop(data.error); return; }
            if (data.ready) {
              clearTimeout(setupTimer); setupTimer = null;
              status.textContent = 'Listening — microphone is on. Stop when finished.';
              input = context.createMediaStreamSource(stream);
              capture = new AudioWorkletNode(context, 'laborlens-capture');
              capture.port.onmessage = event => {
                if (current.readyState === WebSocket.OPEN) {
                  if (current.bufferedAmount > 96000) { stop('Connection too slow. Please reconnect.'); return; }
                  current.send(event.data);
                }
              };
              input.connect(capture); capture.connect(context.destination);
            }
            if (data.evidence) {
              evidence.replaceChildren();
              const answer = document.createElement('p'); answer.textContent = data.evidence.answer;
              evidence.append(answer);
              for (const source of data.evidence.sources || []) {
                const line = document.createElement('p');
                line.textContent = source.title || source.document_title || source.source_name || 'Retrieved source';
                evidence.append(line);
              }
            }
            const content = data.serverContent;
            if (content?.interrupted) clearPlayback();
            for (const part of content?.modelTurn?.parts || []) {
              if (part.inlineData) play(part.inlineData.data, part.inlineData.mimeType);
            }
          } catch (_) { stop('Unexpected voice response. Please use text chat.'); }
        };
        watchdog = setInterval(() => {
          if (localStorage.getItem('ll_token') !== auth) stop('Signed out. Voice session stopped.');
        }, 500);
      } catch (_) {
        if (run === generation) stop('Microphone or voice setup failed. Check browser permission and server settings.');
      }
    };
    button.onclick = () => {
      window.LaborLensVoice?.finish?.();
      dialog.showModal();
    };
    dialog.querySelector('[data-stop]').onclick = () => stop();
    dialog.querySelector('[data-close]').onclick = () => dialog.close();
    dialog.addEventListener('close', () => stop());
    dialog.addEventListener('cancel', () => stop());
    window.addEventListener('pagehide', () => stop());
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop('Voice paused while this tab is hidden.'); });
  });
})();

