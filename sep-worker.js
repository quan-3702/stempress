// StemPress – stem separation worker. Splits a song into vocals, drums, bass, guitar, piano and other with the
// HTDemucs 6-source AI model (Meta, MIT licence; WebGPU ONNX by kramp, MIT) – on this computer, nothing is uploaded.
// The song is cut into 7.8-second windows that overlap; finished audio is sent back while the rest is still working,
// so even long songs need little memory.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.numThreads = 1;

const MODEL_URL = 'https://huggingface.co/kramp/htdemucs-6s-webgpu-onnx/resolve/main/htdemucs_6s.onnx';
const MODEL_CACHE = 'stempress-model';
const MODEL_SIZE = 285000000;
const SEG = 343980; // samples per window at 44.1 kHz
const NSRC = 6; // drums, bass, other, vocals, guitar, piano

async function modelBytes() {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch (_) {}
  let res = cache ? await cache.match(MODEL_URL) : null;
  const fromCache = !!res;
  if (!res) {
    postMessage({ type: 'progress', phase: 'download', p: 0 });
    res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error('The AI model could not be downloaded (' + res.status + '). Check the internet connection and try again.');
  }
  if (fromCache) return new Uint8Array(await res.arrayBuffer());
  const total = Number(res.headers.get('Content-Length')) || MODEL_SIZE;
  const reader = res.body.getReader();
  const parts = []; let got = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (!fromCache && got - last > 2e6) { last = got; postMessage({ type: 'progress', phase: 'download', p: Math.min(1, got / total) }); }
  }
  if (!fromCache && got < 1e8) throw new Error('The AI model download was cut off. Please try again.');
  const buf = new Uint8Array(got); let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (!fromCache && cache) { try { await cache.put(MODEL_URL, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(got) } })); } catch (_) {} }
  return buf;
}

let session = null, device = '';
async function hasWebGPU() { try { return !!(self.navigator.gpu && (await self.navigator.gpu.requestAdapter())); } catch (_) { return false; } }
async function load(force) {
  if (session && (!force || force === device)) return;
  session = null;
  const bytes = await modelBytes();
  postMessage({ type: 'progress', phase: 'load', p: 0 });
  const tries = force ? [force] : (await hasWebGPU()) ? ['webgpu', 'wasm'] : ['wasm'];
  let err = null;
  for (const ep of tries) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'basic' }); device = ep; return; }
    catch (e) { err = e; }
  }
  throw err || new Error('The AI model could not start on this computer.');
}

let cancel = false;
// L, R: the whole song (Float32, 44.1 kHz). overlap: 0.25 (standard) or 0.5 (high quality, slower)
async function run(L0, R0, overlap) {
  const n = L0.length;
  // like Demucs: bring the song to a standard level first, and back afterwards
  let mean = 0; for (let i = 0; i < n; i++) mean += (L0[i] + R0[i]) / 2; mean /= Math.max(1, n);
  let v = 0; for (let i = 0; i < n; i++) { const d = (L0[i] + R0[i]) / 2 - mean; v += d * d; }
  const std = Math.sqrt(v / Math.max(1, n)) || 1;
  const stride = Math.floor(SEG * (1 - overlap));
  const segs = Math.max(1, Math.ceil(Math.max(0, n - SEG) / stride) + 1);
  // rolling overlap-add buffers: [stem][channel]
  const W = SEG + stride;
  const acc = Array.from({ length: NSRC }, () => [new Float32Array(W), new Float32Array(W)]), wsum = new Float32Array(W);
  let base = 0; // song position of acc[...][0]
  const fade = Math.max(1, SEG - stride); // cross-fade over the whole overlap
  const x = new Float32Array(2 * SEG);
  let k = 0, t0 = Date.now();
  const flush = (upto) => { // send samples [base, upto) – they will not change any more
    const len = upto - base; if (len <= 0) return;
    const stems = acc.map(([a, b]) => {
      const l = new Int16Array(len), r = new Int16Array(len);
      for (let i = 0; i < len; i++) {
        const w = wsum[i] || 1;
        let s = (a[i] / w) * std + mean / NSRC; l[i] = s >= 1 ? 32767 : s <= -1 ? -32768 : Math.round(s * 32767);
        s = (b[i] / w) * std + mean / NSRC; r[i] = s >= 1 ? 32767 : s <= -1 ? -32768 : Math.round(s * 32767);
      }
      return [l, r];
    });
    postMessage({ type: 'chunk', start: base, len, stems }, stems.flatMap((s) => [s[0].buffer, s[1].buffer]));
    // shift the buffers left
    for (const [a, b] of acc) { a.copyWithin(0, len); a.fill(0, W - len); b.copyWithin(0, len); b.fill(0, W - len); }
    wsum.copyWithin(0, len); wsum.fill(0, W - len);
    base = upto;
  };
  for (let start = 0; start < n; start += stride) {
    if (cancel) throw new Error('cancelled');
    const len = Math.min(SEG, n - start);
    for (let i = 0; i < SEG; i++) { const j = start + i; x[i] = j < n ? (L0[j] - mean) / std : 0; x[SEG + i] = j < n ? (R0[j] - mean) / std : 0; }
    const feeds = {}; feeds[session.inputNames[0]] = new ort.Tensor('float32', x, [1, 2, SEG]);
    const res = await session.run(feeds);
    const out = res[session.outputNames[0]], d = out.data; // [1, 6, 2, SEG]
    const first = start === 0, last = start + SEG >= n;
    for (let i = 0; i < len; i++) {
      const w = Math.max(1e-3, Math.min(1, first ? 1 : (i + 1) / fade, last ? 1 : (len - i) / fade));
      const o = start + i - base;
      for (let s = 0; s < NSRC; s++) { const p = (s * 2) * SEG + i; acc[s][0][o] += d[p] * w; acc[s][1][o] += d[p + SEG] * w; }
      wsum[o] += w;
    }
    try { out.dispose && out.dispose(); } catch (_) {}
    k++;
    const el = (Date.now() - t0) / 1000;
    postMessage({ type: 'progress', phase: 'separate', p: k / segs, eta: (el / k) * (segs - k), device });
    if (last) break;
    flush(start + stride);
  }
  flush(n);
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'cancel') { cancel = true; return; }
  if (m.type !== 'run') return;
  cancel = false;
  try {
    await load(m.force);
    postMessage({ type: 'progress', phase: 'separate', p: 0, device });
    await run(m.left, m.right, m.overlap || 0.25);
    postMessage({ type: 'done', device });
  } catch (err) {
    postMessage({ type: 'error', message: String((err && err.message) || err), device });
  }
};
