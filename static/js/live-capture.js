/* PCM capture runs off the UI thread. Browser context is requested at 16 kHz. */
class LaborLensCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Int16Array(2048); this.offset = 0; }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) for (const value of input) {
      this.buffer[this.offset++] = Math.max(-1, Math.min(1, value)) * 32767;
      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
        this.buffer = new Int16Array(2048); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('laborlens-capture', LaborLensCapture);

