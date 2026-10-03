// maia-tim in the browser: the live engine, ported from Python, running the
// network with onnxruntime-web. Must build exactly the inputs the Python
// engine builds -- web_parity.py checks that against hundreds of positions.

import { Chess } from 'https://cdn.jsdelivr.net/npm/chess.js@1.0.0/+esm';

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js';
const HISTORY = 8;
const N_PLANES = 17 + 2 * HISTORY;
const N_POLICY = 4096;
const PREMOVE_S = 0.10, T_MAX = 30.0, N_FINE = 40, N_TIME = N_FINE + 1;
const LOG_EDGES = Array.from({ length: N_FINE + 1 }, (_, i) =>
  Math.log10(PREMOVE_S) + i * (Math.log10(T_MAX) - Math.log10(PREMOVE_S)) / N_FINE);
const CAP_VALUE = [0, 1, 3, 3, 5, 9, 0];
const PIECE_TYPE = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };

const sqIndex = (s) => (s.charCodeAt(0) - 97) + 8 * (s.charCodeAt(1) - 49);
const flipSq = (i) => (i < 0 ? i : (7 - (i >> 3)) * 8 + (i & 7));

// --- inputs ----------------------------------------------------------------
// r: { board[64] piece codes (0 empty, 1-6 white PNBRQK, 7-12 black),
//      meta[10]: me_white, wK, wQ, bK, bQ, ep, my_clock, opp_clock, base, inc,
//      rating, oppRating, ply, bzMe, bzOpp, isTim,
//      hfrom[8], hto[8], hcap[8], ht[8] }   (index 0 = most recent ply)
export function buildInputs(r, layout) {
  const x = new Float32Array(N_PLANES * 64);
  const nBase = layout.n_base, nCtx = nBase + 4 * HISTORY;
  const c = new Float32Array(nCtx);
  const meta = r.meta, white = meta[0] > 0.5;

  for (let sq = 0; sq < 64; sq++) {
    let v = r.board[white ? sq : flipSq(sq)];
    if (!v) continue;
    if (!white) v = v <= 6 ? v + 6 : v - 6;
    x[(v - 1) * 64 + sq] = 1;
  }
  const [wk, wq, bk, bq] = [meta[1], meta[2], meta[3], meta[4]];
  const fill = (plane, val) => { if (val) x.fill(val, plane * 64, plane * 64 + 64); };
  fill(12, white ? wk : bk); fill(13, white ? wq : bq);
  fill(14, white ? bk : wk); fill(15, white ? bq : wq);
  if (meta[5] >= 0) {
    const e = white ? meta[5] : flipSq(meta[5]);
    x[16 * 64 + e] = 1;
  }
  for (let k = 0; k < HISTORY; k++) {
    const f = white ? r.hfrom[k] : flipSq(r.hfrom[k]);
    const t = white ? r.hto[k] : flipSq(r.hto[k]);
    if (f >= 0) x[(17 + 2 * k) * 64 + f] = 1;
    if (t >= 0) x[(18 + 2 * k) * 64 + t] = 1;
  }

  const base = Math.max(meta[8], 1), myc = meta[6], oppc = meta[7];
  const rating = r.rating, opp = r.oppRating > 0 ? r.oppRating : r.rating;
  c[0] = myc / base;
  c[1] = oppc / base;
  c[2] = Math.log1p(Math.max(myc, 0)) / 5;
  c[3] = (myc - oppc) / base;
  c[4] = base / 60;
  c[5] = meta[9] / 10;
  c[6] = (rating - 2000) / 400;
  c[7] = (opp - 2000) / 400;
  c[8] = (rating - opp) / 400;
  c[9] = r.ply / 100;
  c[10] = white ? 1 : 0;
  c[11] = r.bzMe ? 1 : 0;
  c[12] = r.bzOpp ? 1 : 0;
  if (layout.is_tim) c[13] = r.isTim ?? 1;
  for (let k = 0; k < HISTORY; k++) {
    const present = r.hfrom[k] >= 0 ? 1 : 0;
    const known = r.ht[k] >= 0 && r.hfrom[k] >= 0 ? 1 : 0;
    c[nBase + k] = present;
    c[nBase + HISTORY + k] = known;
    c[nBase + 2 * HISTORY + k] = (Math.log1p(Math.max(r.ht[k], 0)) / 2) * known;
    c[nBase + 3 * HISTORY + k] = CAP_VALUE[r.hcap[k]] / 9;
  }
  return { x, c };
}

// Legal moves as model indices (mover's orientation), auto-queen.
export function legalMoves(chess) {
  const white = chess.turn() === 'w', out = [];
  for (const m of chess.moves({ verbose: true })) {
    if (m.promotion && m.promotion !== 'q') continue;
    let f = sqIndex(m.from), t = sqIndex(m.to);
    if (!white) { f = flipSq(f); t = flipSq(t); }
    out.push({ uci: m.from + m.to + (m.promotion || ''), idx: f * 64 + t });
  }
  return out;
}

function sampleSeconds(bin) {
  if (bin === 0) return 0.02 + Math.random() * 0.07;
  const lo = LOG_EDGES[bin - 1], hi = LOG_EDGES[bin];
  return 10 ** (lo + Math.random() * (hi - lo));
}

function sample(logits, temperature) {
  if (temperature <= 0) return logits.indexOf(Math.max(...logits));
  const mx = Math.max(...logits);
  const w = logits.map((l) => Math.exp((l - mx) / temperature));
  let u = Math.random() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < w.length; i++) { u -= w[i]; if (u <= 0) return i; }
  return w.length - 1;
}

// --- engine ------------------------------------------------------------------
export class Engine {
  constructor(ort, sessions, meta) {
    this.ort = ort; this.enc = sessions.encode; this.tim = sessions.time;
    this.layout = meta.layout;
    this.calPolicy = meta.temp_policy ?? 1; this.calTime = meta.temp_time ?? 1;
    this.rating = meta.default_rating; this.oppRating = null;
    this.temperature = 1; this.mimic = true;
    this.newGame();
  }

  newGame() {
    this.chess = new Chess(); this.caps = []; this.secs = new Map();
    this.base = null; this.bzMe = false; this.bzOpp = false; this.lastGo = null;
  }

  setPosition(moves) {
    this.chess = new Chess(); this.caps = [];
    for (const u of moves) {
      const m = this.chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] || 'q' });
      this.caps.push(m.captured ? PIECE_TYPE[m.captured] : 0);
    }
  }

  features(wtime, btime, winc, binc) {
    const ch = this.chess, white = ch.turn() === 'w';
    const my = (white ? wtime : btime) / 1000, opp = (white ? btime : wtime) / 1000;
    const inc = (white ? winc : binc) / 1000;
    if (this.base === null) {
      // The base is never stated: at the first turn the larger clock is it;
      // a side holding about half of it started berserked.
      this.base = Math.max(my, opp, 1);
      this.bzMe = my < 0.75 * this.base;
      this.bzOpp = opp < 0.75 * this.base;
    }
    const hist = ch.history({ verbose: true }), n = hist.length;
    if (this.lastGo && n >= 1 && !this.secs.has(n - 1) && this.lastGo[0] === n - 2) {
      this.secs.set(n - 1, Math.max(this.lastGo[2] - opp + inc, 0));
    }

    const board = new Array(64).fill(0);
    const rows = ch.board();
    for (let rr = 0; rr < 8; rr++) for (let f = 0; f < 8; f++) {
      const p = rows[rr][f];
      if (p) board[(7 - rr) * 8 + f] = PIECE_TYPE[p.type] + (p.color === 'b' ? 6 : 0);
    }
    const castle = ch.fen().split(' ')[2];
    let ep = -1;  // set after ANY double pawn push, as python-chess does
    if (n) {
      const last = hist[n - 1];
      if (last.piece === 'p' && Math.abs(sqIndex(last.to) - sqIndex(last.from)) === 16) {
        ep = (sqIndex(last.from) + sqIndex(last.to)) / 2;
      }
    }
    const r = {
      board, rating: this.rating, oppRating: this.oppRating || this.rating, ply: n,
      bzMe: this.bzMe, bzOpp: this.bzOpp, isTim: 1,
      meta: [white ? 1 : 0, +castle.includes('K'), +castle.includes('Q'),
             +castle.includes('k'), +castle.includes('q'), ep, my, opp, this.base, inc],
      hfrom: new Array(HISTORY).fill(-1), hto: new Array(HISTORY).fill(-1),
      hcap: new Array(HISTORY).fill(0), ht: new Array(HISTORY).fill(-1),
    };
    for (let k = 0; k < Math.min(HISTORY, n); k++) {
      const j = n - 1 - k;
      r.hfrom[k] = sqIndex(hist[j].from);
      r.hto[k] = sqIndex(hist[j].to);
      r.hcap[k] = this.caps[j] || 0;
      if (this.secs.has(j)) r.ht[k] = this.secs.get(j);
    }
    this.lastGo = [n, my, opp];
    return r;
  }

  async think(wtime, btime, winc = 0, binc = 0) {
    const t0 = performance.now(), ort = this.ort;
    const n = this.chess.history().length;
    // The bot does not see its OWN past think times (measured to be closer
    // to the real player in self-play); the opponent's are used.
    const hidden = [...this.secs].filter(([j]) => j % 2 === n % 2);
    for (const [j] of hidden) this.secs.delete(j);
    const r = this.features(wtime, btime, winc, binc);
    for (const [j, s] of hidden) this.secs.set(j, s);

    const { x, c } = buildInputs(r, this.layout);
    const legal = legalMoves(this.chess);
    const mask = new Uint8Array(N_POLICY);
    for (const m of legal) mask[m.idx] = 1;
    const out = await this.enc.run({
      x: new ort.Tensor('float32', x, [1, N_PLANES, 8, 8]),
      c: new ort.Tensor('float32', c, [1, c.length]),
      mask: new ort.Tensor('bool', mask, [1, N_POLICY]),
    });
    const logits = legal.map((m) => out.p.data[m.idx] / this.calPolicy);
    const k = sample(logits, this.temperature);

    if (this.mimic && (wtime > 0 || btime > 0)) {
      const t = await this.tim.run({
        h: out.h, p: out.p, a: out.a,
        move: new ort.Tensor('int64', BigInt64Array.of(BigInt(legal[k].idx)), [1]),
      });
      const tl = Array.from(t.t.data, (v) => v / this.calTime);
      const target = sampleSeconds(sample(tl, 1));
      const wait = target * 1000 - (performance.now() - t0);
      if (wait > 0) await new Promise((res) => setTimeout(res, wait));
    }
    this.secs.set(n, (performance.now() - t0) / 1000);
    return legal[k].uci;
  }
}

// --- loading + the backend the player page talks to ---------------------------
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  const total = +res.headers.get('Content-Length') || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onBytes(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const ch of chunks) { buf.set(ch, o); o += ch.length; }
  return buf;
}

export async function loadModel(onStatus = () => {}) {
  if (!window.ort) {
    await new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = ORT_URL; s.onload = res; s.onerror = rej; document.head.appendChild(s);
    });
  }
  const ort = window.ort;
  ort.env.wasm.numThreads = 1;  // static hosting has no cross-origin isolation
  const meta = await (await fetch('meta.json')).json();
  const mb = (b) => (b / 1e6).toFixed(1);
  const sessions = {};
  for (const name of ['encode', 'time']) {
    const bytes = await fetchWithProgress(`${name}.onnx`, (got, total) =>
      onStatus(`loading model: ${name} ${mb(got)}${total ? ' / ' + mb(total) : ''} MB`));
    sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  }
  return { ort, sessions, meta };
}

export async function createBackend(onStatus) {
  const { ort, sessions, meta } = await loadModel(onStatus);
  const engines = { w: new Engine(ort, sessions, meta), b: new Engine(ort, sessions, meta) };
  const configure = (e, cfg, opp) => {
    e.newGame();
    e.rating = Number(cfg.rating ?? e.rating);
    e.oppRating = Number(opp) || null;
    e.temperature = Number(cfg.temperature ?? 100) / 100;
    e.mimic = cfg.mimic ?? true;
  };
  return {
    info: async () => ({
      version: meta.version, device: 'in your browser', default_rating: meta.default_rating,
      rating_buckets: meta.eras.map((r) => ({ rating: r })), bucket: meta.step,
    }),
    newGame: async (b) => {
      for (const [s, o] of [['w', 'b'], ['b', 'w']]) {
        if (b[s] != null) configure(engines[s], b[s], (b[o] || {}).rating || b.human_rating);
      }
      return { ok: true };
    },
    move: async (b) => {
      const e = engines[b.side || 'w'];
      e.setPosition(b.moves || []);
      return { move: await e.think(b.wtime, b.btime, 0, 0) };
    },
    save: async (pgn) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([pgn], { type: 'application/x-chess-pgn' }));
      a.download = `maia-tim-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.pgn`;
      a.click();
      return { saved: 'downloaded' };
    },
  };
}
