# maia-tim-play -- GPL-3.0
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HISTORY = 8
PREMOVE_S = 0.1
T_MAX = 30.0
N_FINE = 40
LOG_EDGES = np.linspace(np.log10(PREMOVE_S), np.log10(T_MAX), N_FINE + 1)
N_TIME = N_FINE + 1
SIGMA_BINS = 0.75
TIME_EDGES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
N_OLD = len(TIME_EDGES) + 1
_centers = np.r_[0.05, 10 ** ((LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2)]
OLD_OF_BIN = np.digitize(_centers, TIME_EDGES).astype(np.int64)
N_PLANES = 17 + 2 * HISTORY
N_BASE_CTX = 14
N_PLY_CTX = 4
N_CTX = N_BASE_CTX + N_PLY_CTX * HISTORY
N_BOARD = 12

def n_planes(board_hist=0):
    return N_PLANES + N_BOARD * board_hist

def n_ctx(time_hist=HISTORY):
    return N_CTX + N_PLY_CTX * max(time_hist - HISTORY, 0)
N_POLICY = 64 * 64
MASKED = -1000000000.0
CAP_VALUE = np.array([0, 1, 3, 3, 5, 9, 0], np.float32)
FLIP = np.arange(64).reshape(8, 8)[::-1].reshape(-1)

def time_bucket(seconds):
    return np.digitize(seconds, TIME_EDGES).astype(np.int64)

def time_bin(seconds):
    s = np.asarray(seconds, np.float64)
    lt = np.log10(np.clip(s, PREMOVE_S, T_MAX))
    fine = 1 + np.clip(np.digitize(lt, LOG_EDGES[1:-1]), 0, N_FINE - 1)
    return np.where(s < PREMOVE_S, 0, fine).astype(np.int64)

def time_targets(seconds):
    edges = torch.as_tensor(LOG_EDGES, dtype=torch.float32, device=seconds.device)
    pre = seconds < PREMOVE_S
    lt = torch.log10(seconds.clamp(PREMOVE_S, T_MAX))
    sigma = SIGMA_BINS * float(LOG_EDGES[1] - LOG_EDGES[0])
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - lt[:, None]) / (sigma * math.sqrt(2))))
    mass = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0)
    mass = mass / mass.sum(1, keepdim=True).clamp_min(1e-08)
    tgt = torch.zeros(seconds.shape[0], N_TIME, device=seconds.device)
    tgt[:, 1:] = mass
    tgt[pre] = 0.0
    tgt[pre, 0] = 1.0
    return tgt

def sample_seconds(bin_idx, rng=np.random):
    if bin_idx == 0:
        return rng.uniform(0.02, 0.09)
    (lo, hi) = (LOG_EDGES[bin_idx - 1], LOG_EDGES[bin_idx])
    return float(10 ** rng.uniform(lo, hi))

def flip_sq(sq):
    sq = np.asarray(sq)
    return np.where(sq >= 0, (7 - sq // 8) * 8 + sq % 8, sq)

def _orient(board, black):
    b = board.copy()
    if black.any():
        v = b[black][:, FLIP]
        recol = v.copy()
        recol[(v >= 1) & (v <= 6)] += 6
        recol[v >= 7] -= 6
        b[black] = recol
    return b

def build_inputs(r, time_hist=HISTORY, board_hist=0):
    (board, meta) = (r['board'], r['meta'])
    B = board.shape[0]
    x = np.zeros((B, n_planes(board_hist), 8, 8), np.float32)
    me_white = meta[:, 0] > 0.5
    black = ~me_white
    b = _orient(board, black)
    (r_idx, sq) = np.nonzero(b)
    x[r_idx, b[r_idx, sq] - 1, sq // 8, sq % 8] = 1.0
    for m in range(board_hist):
        hb = _orient(r['hboard'][:, m], black)
        (r_idx, sq) = np.nonzero(hb)
        x[r_idx, N_PLANES + N_BOARD * m + hb[r_idx, sq] - 1, sq // 8, sq % 8] = 1.0
    (wk, wq, bk, bq) = (meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 4])
    x[:, 12] = np.where(me_white, wk, bk)[:, None, None]
    x[:, 13] = np.where(me_white, wq, bq)[:, None, None]
    x[:, 14] = np.where(me_white, bk, wk)[:, None, None]
    x[:, 15] = np.where(me_white, bq, wq)[:, None, None]
    ep = meta[:, 5].astype(np.int64)
    has_ep = ep >= 0
    if has_ep.any():
        e = np.where(me_white[has_ep], ep[has_ep], flip_sq(ep[has_ep]))
        x[np.nonzero(has_ep)[0], 16, e // 8, e % 8] = 1.0
    hfrom = r['hfrom'].astype(np.int64)
    hto = r['hto'].astype(np.int64)
    hfrom = np.where(black[:, None], flip_sq(hfrom), hfrom)
    hto = np.where(black[:, None], flip_sq(hto), hto)
    for k in range(HISTORY):
        for (plane, sqs) in ((17 + 2 * k, hfrom[:, k]), (18 + 2 * k, hto[:, k])):
            m = sqs >= 0
            if m.any():
                s = sqs[m]
                x[np.nonzero(m)[0], plane, s // 8, s % 8] = 1.0
    base = np.maximum(meta[:, 8], 1.0)
    (myc, oppc) = (meta[:, 6], meta[:, 7])
    rating = r['rating'].astype(np.float32)
    opp = r['opp_rating'].astype(np.float32)
    opp = np.where(opp > 0, opp, rating)
    present = (r['hfrom'] >= 0).astype(np.float32)
    known = ((r['ht'] >= 0) & (r['hfrom'] >= 0)).astype(np.float32)
    c = np.zeros((B, n_ctx(time_hist)), np.float32)
    c[:, 0] = myc / base
    c[:, 1] = oppc / base
    c[:, 2] = np.log1p(np.maximum(myc, 0)) / 5.0
    c[:, 3] = (myc - oppc) / base
    c[:, 4] = base / 60.0
    c[:, 5] = meta[:, 9] / 10.0
    c[:, 6] = (rating - 2000.0) / 400.0
    c[:, 7] = (opp - 2000.0) / 400.0
    c[:, 8] = (rating - opp) / 400.0
    c[:, 9] = r['ply'].astype(np.float32) / 100.0
    c[:, 10] = me_white.astype(np.float32)
    c[:, 11] = r['bz_me'].astype(np.float32)
    c[:, 12] = r['bz_opp'].astype(np.float32)
    c[:, 13] = np.asarray(r.get('is_tim', np.ones(B)), np.float32)
    h0 = N_BASE_CTX
    c[:, h0:h0 + HISTORY] = present
    c[:, h0 + HISTORY:h0 + 2 * HISTORY] = known
    c[:, h0 + 2 * HISTORY:h0 + 3 * HISTORY] = np.log1p(np.maximum(r['ht'], 0)) / 2.0 * known
    c[:, h0 + 3 * HISTORY:N_CTX] = CAP_VALUE[r['hcap']] / 9.0
    extra = time_hist - HISTORY
    if extra > 0:
        xp = r['xpresent'].astype(np.float32)
        xk = ((r['xht'] >= 0) & (r['xpresent'] > 0)).astype(np.float32)
        c[:, N_CTX:N_CTX + extra] = xp
        c[:, N_CTX + extra:N_CTX + 2 * extra] = xk
        c[:, N_CTX + 2 * extra:N_CTX + 3 * extra] = np.log1p(np.maximum(r['xht'], 0)) / 2.0 * xk
        c[:, N_CTX + 3 * extra:] = CAP_VALUE[r['xcap']] / 9.0
    return (x, c)

def legal_mask(rows, moves, batch, device):
    m = torch.zeros(batch, N_POLICY, dtype=torch.bool, device=device)
    m[rows, moves] = True
    return m

class Block(nn.Module):

    def __init__(self, ch, drop_path=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.drop_path = drop_path

    def forward(self, x, mod=None):
        y = self.b1(self.c1(x))
        if mod is not None:
            y = y * (1 + mod[0][:, :, None, None]) + mod[1][:, :, None, None]
        y = F.relu(y)
        y = self.b2(self.c2(y))
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
            y = y * mask / keep
        return F.relu(x + y)
FUSED_ATTENTION = False

class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def make_norm(kind, d):
    return RMSNorm(d) if kind == 'rms' else nn.LayerNorm(d)

class TBlock(nn.Module):

    def __init__(self, d, heads, mlp=4, drop_path=0.0, norm='layer', act='gelu'):
        super().__init__()
        (self.heads, self.drop_path) = (heads, drop_path)
        self.act = F.relu if act == 'relu' else F.gelu
        self.ln1 = make_norm(norm, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = make_norm(norm, d)
        self.fc1 = nn.Linear(d, mlp * d)
        self.fc2 = nn.Linear(mlp * d, d)

    def _dp(self, y):
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            y = y * (torch.rand(y.shape[0], 1, 1, device=y.device) < keep) / keep
        return y

    def forward(self, x, bias=None, mod=None):
        (B, T, d) = x.shape
        (h, dh) = (self.heads, d // self.heads)
        xn = self.ln1(x)
        if mod is not None:
            xn = xn * (1 + mod[0][:, None]) + mod[1][:, None]
        (q, k, v) = self.qkv(xn).reshape(B, T, 3, h, dh).permute(2, 0, 3, 1, 4)
        if FUSED_ATTENTION:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            att = q @ k.transpose(-2, -1) * dh ** (-0.5)
            if bias is not None:
                att = att + bias
            y = att.softmax(-1) @ v
        y = y.transpose(1, 2).reshape(B, T, d)
        x = x + self._dp(self.proj(y))
        xn = self.ln2(x)
        if mod is not None:
            xn = xn * (1 + mod[2][:, None]) + mod[3][:, None]
        return x + self._dp(self.fc2(self.act(self.fc1(xn))))

def _rel_index():
    (r, f) = np.divmod(np.arange(64), 8)
    dr = r[None, :] - r[:, None] + 7
    df = f[None, :] - f[:, None] + 7
    return torch.as_tensor(dr * 15 + df, dtype=torch.long)

class RelBias(nn.Module):

    def __init__(self, heads):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(heads, 225))
        self.register_buffer('idx', _rel_index(), persistent=False)

    def forward(self, x):
        return self.table[:, self.idx][None]

class GABCompress(nn.Module):

    def __init__(self, d, heads, d1, d2, d3, norm='layer'):
        super().__init__()
        (self.heads, self.d3) = (heads, d3)
        self.sm1 = nn.Linear(d, d1)
        self.sm2 = nn.Linear(64 * d1, d2)
        self.ln = make_norm(norm, d2)
        self.sm3 = nn.Linear(d2, heads * d3)

    def forward(self, x):
        y = self.sm1(x).flatten(1)
        y = self.ln(F.gelu(self.sm2(y)))
        return F.gelu(self.sm3(y)).view(-1, self.heads, self.d3)

def from_args(a, **kw):
    return MaiaTim(a['channels'], a['blocks'], arch=a.get('arch', 'resnet'), tf_layers=a.get('tf_layers', 0), heads=a.get('heads', 8), mlp=a.get('mlp', 4), norm=a.get('norm', 'layer'), act=a.get('act', 'gelu'), ctx_token=a.get('ctx_token', True), attn_bias=a.get('attn_bias', 'none'), gab_dims=tuple(a.get('gab_dims', (8, 64, 64))), aux_head=a.get('aux_weight', 0.0) > 0, time_hist=a.get('time_hist', HISTORY), board_hist=a.get('board_hist', 0), film=a.get('film', False), **kw)
N_AUX = 7

class MaiaTim(nn.Module):

    def __init__(self, channels=128, blocks=10, qk_dim=96, drop_path=0.0, head_dropout=0.0, arch='resnet', tf_layers=0, heads=8, mlp=4, norm='layer', act='gelu', ctx_token=True, attn_bias='none', gab_dims=(8, 64, 64), aux_head=False, time_hist=HISTORY, board_hist=0, film=False):
        super().__init__()
        (self.arch, self.ctx_token, self.attn_bias) = (arch, ctx_token, attn_bias)
        (self.time_hist, self.board_hist) = (time_hist, board_hist)
        (self.n_planes, self.n_ctx) = (n_planes(board_hist), n_ctx(time_hist))
        self.stem = nn.Conv2d(self.n_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.ctx = nn.Sequential(nn.Linear(self.n_ctx, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
        depth = blocks + (tf_layers if arch == 'hybrid' else 0)
        self.trunk = nn.Sequential(*[Block(channels, drop_path * (i + 1) / depth) for i in range(blocks)])
        self.film = film
        if film:
            self.film_conv = nn.ModuleList([nn.Linear(channels, 2 * channels) for _ in range(blocks)])
            self.film_tf = nn.ModuleList([nn.Linear(channels, 4 * channels) for _ in range(tf_layers if arch == 'hybrid' else 0)])
            for lin in [*self.film_conv, *self.film_tf]:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        if arch == 'hybrid':
            self.sq_pos = nn.Parameter(torch.randn(64, channels) * 0.02)
            if ctx_token:
                self.ctx_tok = nn.Linear(self.n_ctx, channels)
            self.tf = nn.ModuleList([TBlock(channels, heads, mlp, drop_path * (blocks + i + 1) / depth, norm, act) for i in range(tf_layers)])
            self.tf_ln = make_norm(norm, channels)
            if attn_bias == 'rel':
                self.rel = nn.ModuleList([RelBias(heads) for _ in range(tf_layers)])
            elif attn_bias == 'gab':
                (d1, d2, d3) = gab_dims
                self.gab = nn.ModuleList([GABCompress(channels, heads, d1, d2, d3, norm) for _ in range(tf_layers)])
                self.gab_w = nn.Parameter(torch.zeros(d3, 64 * 64))
            elif attn_bias != 'none':
                raise ValueError(f'unknown attn_bias {attn_bias!r}')
        self.q = nn.Conv2d(channels, qk_dim, 1)
        self.k = nn.Conv2d(channels, qk_dim, 1)
        self.pos_q = nn.Parameter(torch.zeros(64, qk_dim))
        self.pos_k = nn.Parameter(torch.zeros(64, qk_dim))
        self.qk_scale = 1.0 / math.sqrt(qk_dim)
        self.aux_conv = nn.Sequential(nn.Conv2d(channels, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        aux = 16 * 64 + self.n_ctx
        self.channels = channels
        self.t_fc = nn.Sequential(nn.Linear(aux + 2 * channels + 2, 384), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(384, 192), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(192, N_TIME))
        self.v_fc = nn.Sequential(nn.Linear(aux, 128), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(128, 3))
        if aux_head:
            self.aux_fc = nn.Sequential(nn.Linear(aux, 128), nn.ReLU(inplace=True), nn.Linear(128, N_AUX))

    def _bias(self, i, seq):
        if self.attn_bias == 'rel':
            b = self.rel[i](seq)
        elif self.attn_bias == 'gab':
            code = self.gab[i](seq[:, :64])
            b = (code @ self.gab_w).view(seq.shape[0], -1, 64, 64)
        else:
            return None
        if seq.shape[1] > 64:
            b = F.pad(b, (0, seq.shape[1] - 64, 0, seq.shape[1] - 64))
        return b

    def encode(self, x, c, mask=None):
        if not self.film:
            h = self.stem_bn(self.stem(x)) + self.ctx(c)[:, :, None, None]
            h = self.trunk(F.relu(h))
        else:
            e = self.ctx(c)
            s = F.silu(e)
            h = F.relu(self.stem_bn(self.stem(x)) + e[:, :, None, None])
            for (blk, lin) in zip(self.trunk, self.film_conv):
                h = blk(h, lin(s).chunk(2, dim=1))
        if self.arch == 'hybrid':
            (B, C) = (h.shape[0], h.shape[1])
            seq = h.flatten(2).transpose(1, 2) + self.sq_pos
            if self.ctx_token:
                seq = torch.cat([seq, self.ctx_tok(c)[:, None, :]], dim=1)
            for (i, blk) in enumerate(self.tf):
                mod = self.film_tf[i](s).chunk(4, dim=1) if self.film else None
                seq = blk(seq, self._bias(i, seq), mod)
            h = self.tf_ln(seq)[:, :64].transpose(1, 2).reshape(B, C, 8, 8)
        q = self.q(h).flatten(2).transpose(1, 2) + self.pos_q
        k = self.k(h).flatten(2).transpose(1, 2) + self.pos_k
        p = (q @ k.transpose(1, 2)).flatten(1) * self.qk_scale
        if mask is not None:
            p = p.float().masked_fill(~mask, MASKED)
        a = torch.cat([self.aux_conv(h).flatten(1), c], dim=1)
        return (h, p, a)

    def time_logits(self, h, p, a, move):
        hf = h.flatten(2)
        idx_from = (move // 64)[:, None, None].expand(-1, hf.shape[1], 1)
        idx_to = (move % 64)[:, None, None].expand(-1, hf.shape[1], 1)
        h_from = hf.gather(2, idx_from).squeeze(2)
        h_to = hf.gather(2, idx_to).squeeze(2)
        logp = F.log_softmax(p.detach().float(), dim=1)
        lp_move = logp.gather(1, move[:, None]).clamp_min(-20.0) / 10.0
        entropy = -(logp.exp() * logp).sum(1, keepdim=True) / 8.0
        return self.t_fc(torch.cat([a, h_from, h_to, lp_move, entropy], dim=1))

    def forward(self, x, c, move=None, mask=None, with_summary=False):
        (h, p, a) = self.encode(x, c, mask)
        if move is None:
            move = p.argmax(1)
        out = (p, self.time_logits(h, p, a, move), self.v_fc(a))
        return out + (a,) if with_summary else out

def aux_targets(x, move):
    B = x.shape[0]
    (frm, to) = (move // 64, move % 64)
    sq = x.flatten(2)
    rows = torch.arange(B, device=x.device)
    piece = sq[rows, :6, frm].argmax(1)
    capture = (sq[rows, 6:12, to].sum(1) > 0) | (piece == 0) & (sq[rows, 16, to] > 0)
    return (piece, capture.float())

def policy_target(label, me_white):
    (frm, to) = (label // 64, label % 64)
    flip = ~np.asarray(me_white)
    frm = np.where(flip, (7 - frm // 8) * 8 + frm % 8, frm)
    to = np.where(flip, (7 - to // 8) * 8 + to % 8, to)
    return frm * 64 + to
