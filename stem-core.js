// StemPress – music analysis (tempo, key, chords), mixing, WAV and ZIP writers. No DOM: tested with node.
const STC = (() => {
const STEMS = [
  { id: 'vocals', name: 'Vocals', icon: '🎤', color: '#ec4899' },
  { id: 'drums', name: 'Drums', icon: '🥁', color: '#f59e0b' },
  { id: 'bass', name: 'Bass', icon: '🎸', color: '#8b5cf6' },
  { id: 'guitar', name: 'Guitar', icon: '🪕', color: '#22c55e' },
  { id: 'piano', name: 'Piano & keys', icon: '🎹', color: '#3b82f6' },
  { id: 'other', name: 'Other (strings, synths, effects)', icon: '🎻', color: '#06b6d4' },
];
// order of the htdemucs_6s model output
const MODEL_ORDER = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
const NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const AN_RATE = 11025; // analysis signals are downsampled to this rate

// ---------- FFT (in place, radix 2) ----------
const fftCache = new Map();
function fftTables(n) {
  if (fftCache.has(n)) return fftCache.get(n);
  const rev = new Uint32Array(n), bits = Math.log2(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
  for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos((2 * Math.PI * k) / n); sin[k] = -Math.sin((2 * Math.PI * k) / n); }
  const t = { rev, cos, sin }; fftCache.set(n, t); return t;
}
function fft(re, im) {
  const n = re.length, { rev, cos, sin } = fftTables(n);
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s = 0; s < n; s += size) for (let k = 0; k < half; k++) {
      const c = cos[k * step], sn = sin[k * step], a = s + k, b = a + half;
      const tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
const hann = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); return w; };
// magnitude spectra of x: frames of size n every hop samples
function spectra(x, n, hop, each) {
  const w = hann(n), re = new Float64Array(n), im = new Float64Array(n), mag = new Float64Array(n / 2);
  for (let s = 0, f = 0; s + n <= x.length || (f === 0 && x.length); s += hop, f++) {
    for (let i = 0; i < n; i++) { re[i] = (x[s + i] || 0) * w[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < n / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    each(mag, f, s);
    if (s + n > x.length) break;
  }
}

// ---------- tempo ----------
// onset strength (spectral flux of the drums), then the period that repeats best between 60 and 200 BPM
function onsetEnvelope(x, rate) {
  const n = 1024, hop = 256, env = []; let prev = null;
  spectra(x, n, hop, (mag) => {
    const lm = new Float64Array(n / 2); for (let k = 0; k < n / 2; k++) lm[k] = Math.log1p(100 * mag[k]);
    let s = 0; if (prev) for (let k = 1; k < n / 2; k++) { const d = lm[k] - prev[k]; if (d > 0) s += d; }
    env.push(s); prev = lm;
  });
  const e = Float64Array.from(env);
  // remove the slow trend so only the hits remain
  const out = new Float64Array(e.length), W = 16;
  for (let i = 0; i < e.length; i++) { let m = 0, c = 0; for (let j = Math.max(0, i - W); j <= Math.min(e.length - 1, i + W); j++) { m += e[j]; c++; } out[i] = Math.max(0, e[i] - m / c); }
  return { env: out, fps: rate / hop };
}
function tempo(x, rate) {
  rate = rate || AN_RATE;
  const { env, fps } = onsetEnvelope(x, rate);
  if (env.length < fps * 4) return { bpm: 0, beats: [], conf: 0 };
  let mean = 0; for (const v of env) mean += v; mean /= env.length;
  const e = env.map((v) => v - mean);
  const minLag = Math.floor((60 / 200) * fps), maxLag = Math.ceil((60 / 55) * fps);
  const ac = new Float64Array(maxLag + 2);
  for (let lag = minLag - 1; lag <= maxLag + 1; lag++) { let s = 0; for (let i = lag; i < e.length; i++) s += e[i] * e[i - lag]; ac[lag] = s / (e.length - lag); }
  let best = 0, bestLag = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    const bpm = (60 * fps) / lag;
    const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.9, 2));
    // a true beat period also repeats at twice the period
    const sc = (ac[lag] + 0.5 * (ac[2 * lag] || 0)) * prior;
    if (sc > best) { best = sc; bestLag = lag; }
  }
  if (!bestLag) return { bpm: 0, beats: [], conf: 0 };
  const a = ac[bestLag - 1], b = ac[bestLag], c = ac[bestLag + 1];
  const d = a - 2 * b + c, lagF = bestLag + (d < 0 ? (0.5 * (a - c)) / d : 0);
  let bpm = (60 * fps) / lagF;
  // beat positions: the phase that lines up with the most onsets
  const P = lagF; let bestPh = 0, bestS = -1;
  for (let ph = 0; ph < P; ph += 0.5) { let s = 0; for (let t = ph; t < env.length; t += P) s += env[Math.round(t)] || 0; if (s > bestS) { bestS = s; bestPh = ph; } }
  const lag0 = 1024 / rate; // an onset shows up when it enters the analysis window
  const beats = []; for (let t = bestPh; t < env.length; t += P) beats.push(+(t / fps + lag0).toFixed(3));
  const conf = ac[0] > 0 ? Math.max(0, Math.min(1, b / (ac[minLag - 1] > 0 ? Math.max(...ac.slice(minLag, maxLag + 1)) : b))) : 0;
  return { bpm: Math.round(bpm * 10) / 10, beats, conf };
}

// ---------- chroma, key, chords ----------
function chroma(x, rate) {
  rate = rate || AN_RATE;
  const n = 4096, hop = 2048, frames = [], bin = rate / n;
  const pcOf = new Int8Array(n / 2).fill(-1);
  for (let k = 1; k < n / 2; k++) { const f = k * bin; if (f < 55 || f > 2100) continue; const midi = 69 + 12 * Math.log2(f / 440); pcOf[k] = ((Math.round(midi) % 12) + 12) % 12; }
  const energy = [];
  spectra(x, n, hop, (mag) => {
    const c = new Float64Array(12); let e = 0;
    for (let k = 1; k < n / 2; k++) { const p = pcOf[k]; if (p < 0) continue; const v = mag[k]; c[p] += v * v; e += v * v; }
    for (let p = 0; p < 12; p++) c[p] = Math.sqrt(c[p]);
    frames.push(c); energy.push(e);
  });
  return { frames, energy, fps: rate / hop };
}
const KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
function corr(a, b) { const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n; let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; } return sa && sb ? s / Math.sqrt(sa * sb) : 0; }
function keyOf(frames) {
  const tot = new Float64Array(12);
  for (const c of frames) { let s = 0; for (const v of c) s += v; if (s > 0) for (let p = 0; p < 12; p++) tot[p] += c[p] / s; }
  let best = -2, key = null;
  for (let r = 0; r < 12; r++) for (const [prof, mode] of [[KS_MAJ, 'major'], [KS_MIN, 'minor']]) {
    const rot = prof.map((_, i) => prof[(i - r + 12) % 12]);
    const c = corr(tot, rot); if (c > best) { best = c; key = { root: r, mode }; }
  }
  if (!key) return null;
  return { ...key, name: NOTES[key.root] + (key.mode === 'minor' ? ' minor' : ' major'), conf: best };
}
const CHORD_NAMES = [];
const TEMPLATES = [];
for (let r = 0; r < 12; r++) for (const [q, iv] of [['', [0, 4, 7]], ['m', [0, 3, 7]]]) {
  const t = new Float64Array(12); t[r] = 1; t[(r + iv[1]) % 12] = 0.85; t[(r + iv[2]) % 12] = 0.9;
  TEMPLATES.push(t); CHORD_NAMES.push(NOTES[r] + q);
}
function chords(ch, opts) {
  opts = opts || {};
  const { frames, energy, fps } = ch, S = TEMPLATES.length + 1, T = frames.length;
  if (!T) return [];
  const maxE = Math.max(...energy) || 1;
  // score of every chord (and "no chord") for every frame
  const sc = frames.map((c, f) => {
    const norm = Math.hypot(...c) || 1, row = new Float64Array(S);
    for (let k = 0; k < TEMPLATES.length; k++) { const t = TEMPLATES[k]; let d = 0; for (let p = 0; p < 12; p++) d += c[p] * t[p]; row[k] = d / norm / 1.596; }
    row[S - 1] = energy[f] < maxE * 1e-3 ? 1 : 0.3; // "no chord" only where it is (almost) silent
    return row;
  });
  // Viterbi: changing chord costs something, so short wobbles are ignored
  const pen = opts.penalty || 0.18;
  let prev = sc[0].slice(); const back = [];
  for (let f = 1; f < T; f++) {
    let bi = 0; for (let k = 1; k < S; k++) if (prev[k] > prev[bi]) bi = k;
    const cur = new Float64Array(S), bk = new Uint8Array(S);
    for (let k = 0; k < S; k++) { const stay = prev[k], jump = prev[bi] - pen; if (stay >= jump) { cur[k] = stay + sc[f][k]; bk[k] = k; } else { cur[k] = jump + sc[f][k]; bk[k] = bi; } }
    back.push(bk); prev = cur;
  }
  let k = 0; for (let j = 1; j < S; j++) if (prev[j] > prev[k]) k = j;
  const path = new Uint8Array(T); path[T - 1] = k;
  for (let f = T - 1; f > 0; f--) { k = back[f - 1][k]; path[f - 1] = k; }
  // to segments
  const segs = [];
  for (let f = 0; f < T; f++) {
    const name = path[f] === S - 1 ? 'N' : CHORD_NAMES[path[f]], t = f / fps + (f ? 0.5 / fps : 0); // frame centre
    if (segs.length && segs[segs.length - 1].c === name) segs[segs.length - 1].e = t + 1 / fps;
    else segs.push({ c: name, s: t, e: t + 1 / fps });
  }
  // very short chords join their neighbours
  for (let changed = true; changed;) {
    changed = false;
    for (let i = 0; i < segs.length; i++) {
      if (segs.length > 1 && segs[i].e - segs[i].s < (opts.minLen || 0.55)) {
        const j = i === 0 ? 1 : i === segs.length - 1 ? i - 1 : segs[i - 1].e - segs[i - 1].s >= segs[i + 1].e - segs[i + 1].s ? i - 1 : i + 1;
        if (j < i) segs[j].e = segs[i].e; else segs[j].s = segs[i].s;
        segs.splice(i, 1); changed = true; break;
      }
    }
    for (let i = 1; i < segs.length; i++) if (segs[i].c === segs[i - 1].c) { segs[i - 1].e = segs[i].e; segs.splice(i, 1); changed = true; break; }
  }
  return segs.map((s) => ({ c: s.c, s: +s.s.toFixed(2), e: +s.e.toFixed(2) }));
}
// chord names in another key
function transposeChord(c, semis) {
  if (!c || c === 'N' || !semis) return c;
  const m = c.match(/^([A-G](?:#|b)?)(.*)$/); if (!m) return c;
  const alias = { Db: 'C#', 'D#': 'Eb', Gb: 'F#', 'G#': 'Ab', 'A#': 'Bb' };
  const i = NOTES.indexOf(alias[m[1]] || m[1]); if (i < 0) return c;
  return NOTES[(((i + semis) % 12) + 12) % 12] + m[2];
}
const keyName = (k, semis) => (k ? NOTES[(((k.root + (semis || 0)) % 12) + 12) % 12] + (k.mode === 'minor' ? ' minor' : ' major') : '');

// ---------- analysis signals: the separation worker sends 4:1 downsampled mono chunks ----------
function down4(L, R, out, at) { // averages 4 samples (a simple low-pass) – enough for tempo and chords
  const n = Math.floor(L.length / 4);
  for (let i = 0; i < n; i++) { const j = i * 4; out[at + i] = (L[j] + L[j + 1] + L[j + 2] + L[j + 3] + R[j] + R[j + 1] + R[j + 2] + R[j + 3]) / 8; }
  return n;
}

// ---------- peaks for the waveform ----------
function peaks(chans, count) {
  const n = chans[0].length, out = new Float32Array(count), per = Math.max(1, Math.floor(n / count));
  for (let p = 0; p < count; p++) { let m = 0; const a = p * per, b = Math.min(n, a + per); for (const c of chans) for (let i = a; i < b; i += 4) { const v = Math.abs(c[i]); if (v > m) m = v; } out[p] = m; }
  let mx = 0; for (const v of out) if (v > mx) mx = v; if (mx > 0) for (let i = 0; i < count; i++) out[i] /= mx;
  return out;
}

// ---------- mixing ----------
// tracks: [{ chans: [L, R], gain, pan (-1..1) }] -> [L, R] (soft limit so nothing clips)
function mix(tracks, len) {
  const L = new Float32Array(len), R = new Float32Array(len);
  for (const t of tracks) {
    if (!t.gain) continue;
    const p = Math.max(-1, Math.min(1, t.pan || 0)), gl = t.gain * Math.cos(((p + 1) * Math.PI) / 4) * Math.SQRT2, gr = t.gain * Math.sin(((p + 1) * Math.PI) / 4) * Math.SQRT2;
    const a = t.chans[0], b = t.chans[1] || t.chans[0], n = Math.min(len, a.length);
    for (let i = 0; i < n; i++) { L[i] += a[i] * gl; R[i] += b[i] * gr; }
  }
  let peak = 0; for (let i = 0; i < len; i++) { const m = Math.max(Math.abs(L[i]), Math.abs(R[i])); if (m > peak) peak = m; }
  if (peak > 0.99) { const g = 0.99 / peak; for (let i = 0; i < len; i++) { L[i] *= g; R[i] *= g; } }
  return [L, R];
}

// ---------- WAV ----------
function wav(chans, rate, bits) {
  bits = bits || 16;
  const nc = chans.length, n = chans[0].length, bps = bits / 8, size = 44 + n * nc * bps;
  const buf = new ArrayBuffer(size), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, size - 8, true); str(8, 'WAVE'); str(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, bits === 32 ? 3 : 1, true); v.setUint16(22, nc, true); v.setUint32(24, rate, true); v.setUint32(28, rate * nc * bps, true); v.setUint16(32, nc * bps, true); v.setUint16(34, bits, true);
  str(36, 'data'); v.setUint32(40, n * nc * bps, true);
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) {
    const s = Math.max(-1, Math.min(1, chans[c][i]));
    if (bits === 16) { v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2; }
    else if (bits === 24) { const x = Math.round(s < 0 ? s * 0x800000 : s * 0x7fffff); v.setUint8(o, x & 255); v.setUint8(o + 1, (x >> 8) & 255); v.setUint8(o + 2, (x >> 16) & 255); o += 3; }
    else { v.setFloat32(o, s, true); o += 4; }
  }
  return new Uint8Array(buf);
}

// ---------- ZIP (stored, no compression – audio is already compressed or does not shrink) ----------
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b, c) { c = c == null ? 0xffffffff : c; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return c; }
function zipParts(files) { // files: [[name, Uint8Array]] -> array of Uint8Array parts (no 4 GB support)
  const enc = new TextEncoder(), parts = [], central = []; let off = 0;
  for (const [name, data] of files) {
    const nb = enc.encode(name), crc = (crc32(data) ^ 0xffffffff) >>> 0;
    const h = new DataView(new ArrayBuffer(30)); h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, nb.length, true);
    parts.push(new Uint8Array(h.buffer), nb, data);
    const c = new DataView(new ArrayBuffer(46)); c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, nb.length, true); c.setUint32(42, off, true);
    central.push(new Uint8Array(c.buffer), nb); off += 30 + nb.length + data.length;
  }
  const cs = central.reduce((n, x) => n + x.length, 0), e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, cs, true); e.setUint32(16, off, true);
  return [...parts, ...central, new Uint8Array(e.buffer)];
}
const clock = (s) => { s = Math.max(0, s || 0); const m = Math.floor(s / 60), x = Math.floor(s % 60); return m + ':' + String(x).padStart(2, '0'); };
const safeName = (s) => String(s || 'Song').replace(/\.[a-z0-9]{2,4}$/i, '').replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80) || 'Song';

return { STEMS, MODEL_ORDER, NOTES, AN_RATE, fft, tempo, chroma, keyOf, chords, transposeChord, keyName, down4, peaks, mix, wav, zipParts, crc32, clock, safeName, onsetEnvelope };
})();
if (typeof module !== 'undefined') module.exports = STC;
