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
N_BASE_CTX = 13
N_PLY_CTX = 4
N_CTX = N_BASE_CTX + N_PLY_CTX * HISTORY
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

def build_inputs(r):
    (board, meta) = (r['board'], r['meta'])
    B = board.shape[0]
    x = np.zeros((B, N_PLANES, 8, 8), np.float32)
    me_white = meta[:, 0] > 0.5
    black = ~me_white
    b = board.copy()
    if black.any():
        v = b[black][:, FLIP]
        recol = v.copy()
        recol[(v >= 1) & (v <= 6)] += 6
        recol[v >= 7] -= 6
        b[black] = recol
    (r_idx, sq) = np.nonzero(b)
    x[r_idx, b[r_idx, sq] - 1, sq // 8, sq % 8] = 1.0
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
    c = np.zeros((B, N_CTX), np.float32)
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
    h0 = N_BASE_CTX
    c[:, h0:h0 + HISTORY] = present
    c[:, h0 + HISTORY:h0 + 2 * HISTORY] = known
    c[:, h0 + 2 * HISTORY:h0 + 3 * HISTORY] = np.log1p(np.maximum(r['ht'], 0)) / 2.0 * known
    c[:, h0 + 3 * HISTORY:] = CAP_VALUE[r['hcap']] / 9.0
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

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
            y = y * mask / keep
        return F.relu(x + y)

class MaiaTim(nn.Module):

    def __init__(self, channels=128, blocks=10, qk_dim=96, drop_path=0.0, head_dropout=0.0):
        super().__init__()
        self.stem = nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.ctx = nn.Sequential(nn.Linear(N_CTX, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
        self.trunk = nn.Sequential(*[Block(channels, drop_path * (i + 1) / blocks) for i in range(blocks)])
        self.q = nn.Conv2d(channels, qk_dim, 1)
        self.k = nn.Conv2d(channels, qk_dim, 1)
        self.pos_q = nn.Parameter(torch.zeros(64, qk_dim))
        self.pos_k = nn.Parameter(torch.zeros(64, qk_dim))
        self.qk_scale = 1.0 / math.sqrt(qk_dim)
        self.aux_conv = nn.Sequential(nn.Conv2d(channels, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        aux = 16 * 64 + N_CTX
        self.channels = channels
        self.t_fc = nn.Sequential(nn.Linear(aux + 2 * channels + 2, 384), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(384, 192), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(192, N_TIME))
        self.v_fc = nn.Sequential(nn.Linear(aux, 128), nn.ReLU(inplace=True), nn.Dropout(head_dropout), nn.Linear(128, 3))

    def encode(self, x, c, mask=None):
        h = self.stem_bn(self.stem(x)) + self.ctx(c)[:, :, None, None]
        h = self.trunk(F.relu(h))
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

    def forward(self, x, c, move=None, mask=None):
        (h, p, a) = self.encode(x, c, mask)
        if move is None:
            move = p.argmax(1)
        return (p, self.time_logits(h, p, a, move), self.v_fc(a))

def policy_target(label, me_white):
    (frm, to) = (label // 64, label % 64)
    flip = ~np.asarray(me_white)
    frm = np.where(flip, (7 - frm // 8) * 8 + frm % 8, frm)
    to = np.where(flip, (7 - to // 8) * 8 + to % 8, to)
    return frm * 64 + to
