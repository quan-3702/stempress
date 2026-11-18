// StemPress – MP3 encoder (lamejs, LGPL). Streams: open(id) → chunk(id, L, R) … → end(id) returns the file.
// Or one call: encode(id, L, R) with Float32 channels.
importScripts('lame.min.js');
const enc = new Map();
function open(id, kbps) { enc.set(id, { e: new lamejs.Mp3Encoder(2, 44100, kbps || 256), parts: [], size: 0 }); }
function feed(id, L, R) {
  const s = enc.get(id); if (!s) return;
  const N = 1152 * 20;
  for (let i = 0; i < L.length; i += N) { const o = s.e.encodeBuffer(L.subarray(i, i + N), R.subarray(i, i + N)); if (o.length) { s.parts.push(new Uint8Array(o)); s.size += o.length; } }
}
function end(id) {
  const s = enc.get(id); enc.delete(id); if (!s) return null;
  const o = s.e.flush(); if (o.length) { s.parts.push(new Uint8Array(o)); s.size += o.length; }
  const out = new Uint8Array(s.size); let k = 0; for (const p of s.parts) { out.set(p, k); k += p.length; }
  return out;
}
const i16 = (f) => { const o = new Int16Array(f.length); for (let i = 0; i < f.length; i++) { const s = f[i] > 1 ? 1 : f[i] < -1 ? -1 : f[i]; o[i] = s < 0 ? s * 0x8000 : s * 0x7fff; } return o; };
self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'open') open(m.id, m.kbps);
    else if (m.type === 'chunk') feed(m.id, m.L, m.R);
    else if (m.type === 'end') { const mp3 = end(m.id); self.postMessage({ type: 'done', id: m.id, mp3 }, [mp3.buffer]); }
    else if (m.type === 'encode') {
      open(m.id, m.kbps); const L = i16(m.L), R = i16(m.R), N = 44100 * 10;
      for (let i = 0; i < L.length; i += N) { feed(m.id, L.subarray(i, i + N), R.subarray(i, i + N)); self.postMessage({ type: 'progress', id: m.id, p: i / L.length }); }
      const mp3 = end(m.id); self.postMessage({ type: 'done', id: m.id, mp3 }, [mp3.buffer]);
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) }); }
};
