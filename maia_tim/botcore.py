# maia-tim-play -- GPL-3.0
import time
import chess
import numpy as np
import torch

def think(eng, N, wtime, btime, winc, binc, *, temperature=1.0, mimic=True, cap=None, hide_own=True):
    t0 = time.time()
    board = eng.board
    n = len(board.move_stack)
    hidden = {}
    if hide_own:
        hidden = {j: s for (j, s) in eng.secs.items() if j % 2 == n % 2}
        for j in hidden:
            del eng.secs[j]
    (x, c, my_clock) = eng.features(wtime, btime, winc, binc)
    eng.secs.update(hidden)
    white = board.turn == chess.WHITE
    legal = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
    idx = []
    for mv in legal:
        (f, to) = (mv.from_square, mv.to_square)
        if not white:
            (f, to) = (int(N.flip_sq(f)), int(N.flip_sq(to)))
        idx.append(f * 64 + to)
    mask = torch.zeros(1, N.N_POLICY, dtype=torch.bool, device=eng.dev)
    mask[0, idx] = True
    with torch.no_grad():
        (h, p, a) = eng.model.encode(x, c, mask)
        logits = p[0, idx].float().cpu().numpy() / getattr(eng, 'cal_policy', 1.0)
        if temperature <= 0:
            choice = int(np.argmax(logits))
        else:
            z = logits / temperature
            z = np.exp(z - z.max())
            choice = int(np.random.choice(len(legal), p=z / z.sum()))
        t = eng.model.time_logits(h, p, a, torch.tensor([idx[choice]], device=eng.dev))
    if mimic and (wtime > 0 or btime > 0):
        probs = torch.softmax(t[0].float() / getattr(eng, 'cal_time', 1.0), 0)
        probs = probs.cpu().numpy().astype(np.float64)
        target = N.sample_seconds(int(np.random.choice(len(probs), p=probs / probs.sum())))
        if cap:
            target = min(target, max(my_clock * cap, 0.0))
        wait = target - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    eng.secs[n] = time.time() - t0
    return legal[choice]
