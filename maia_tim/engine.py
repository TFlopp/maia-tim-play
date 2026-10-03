# maia-tim-play -- GPL-3.0
import os
import pathlib
import sys
import time
import chess
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import net as N
CKPT = pathlib.Path(__file__).resolve().parent / "weights.pt"
DEFAULT_RATING = 2450
PIECE_CODE = {(chess.PAWN, True): 1, (chess.KNIGHT, True): 2, (chess.BISHOP, True): 3, (chess.ROOK, True): 4, (chess.QUEEN, True): 5, (chess.KING, True): 6, (chess.PAWN, False): 7, (chess.KNIGHT, False): 8, (chess.BISHOP, False): 9, (chess.ROOK, False): 10, (chess.QUEEN, False): 11, (chess.KING, False): 12}

class Engine:

    def __init__(self):
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(CKPT, map_location=self.dev, weights_only=False)
        a = ck['args']
        self.model = N.MaiaTim(a['channels'], a['blocks']).to(self.dev).eval()
        self.model.load_state_dict(ck['model'])
        self.cal_policy = float(ck.get('temp_policy', 1.0))
        self.cal_time = float(ck.get('temp_time', 1.0))
        self.temperature = 1.0
        self.mimic_time = True
        self.rating = DEFAULT_RATING
        self.opp_rating = None
        self.new_game()

    def new_game(self):
        self.board = chess.Board()
        self.caps = []
        self.secs = {}
        self.base = None
        self.bz_me = self.bz_opp = False
        self.last_go = None

    def set_position(self, fen, moves):
        b = chess.Board(fen) if fen else chess.Board()
        caps = []
        for u in moves:
            mv = chess.Move.from_uci(u)
            if b.is_en_passant(mv):
                caps.append(chess.PAWN)
            else:
                victim = b.piece_at(mv.to_square)
                caps.append(victim.piece_type if victim else 0)
            b.push(mv)
        (self.board, self.caps) = (b, caps)

    def features(self, wtime, btime, winc, binc):
        b = self.board
        white = b.turn == chess.WHITE
        my = (wtime if white else btime) / 1000.0
        opp = (btime if white else wtime) / 1000.0
        inc = (winc if white else binc) / 1000.0
        if self.base is None:
            self.base = max(my, opp, 1.0)
            self.bz_me = my < 0.75 * self.base
            self.bz_opp = opp < 0.75 * self.base
        n = len(b.move_stack)
        if self.last_go is not None and n >= 1 and (n - 1 not in self.secs):
            (prev_ply, _, prev_opp) = self.last_go
            if prev_ply == n - 2:
                self.secs[n - 1] = max(prev_opp - opp + inc, 0.0)
        r = {'board': np.zeros((1, 64), np.uint8), 'meta': np.array([[1.0 if white else 0.0, float(b.has_kingside_castling_rights(chess.WHITE)), float(b.has_queenside_castling_rights(chess.WHITE)), float(b.has_kingside_castling_rights(chess.BLACK)), float(b.has_queenside_castling_rights(chess.BLACK)), float(b.ep_square if b.ep_square is not None else -1), my, opp, self.base, inc]], np.float32), 'rating': np.array([self.rating]), 'opp_rating': np.array([self.opp_rating or self.rating]), 'ply': np.array([b.ply()]), 'bz_me': np.array([int(self.bz_me)]), 'bz_opp': np.array([int(self.bz_opp)]), 'hfrom': np.full((1, N.HISTORY), -1, np.int8), 'hto': np.full((1, N.HISTORY), -1, np.int8), 'hcap': np.zeros((1, N.HISTORY), np.uint8), 'ht': np.full((1, N.HISTORY), -1.0, np.float32)}
        for (sq, pc) in b.piece_map().items():
            r['board'][0, sq] = PIECE_CODE[pc.piece_type, pc.color]
        for k in range(min(N.HISTORY, n)):
            j = n - 1 - k
            mv = b.move_stack[j]
            r['hfrom'][0, k] = mv.from_square
            r['hto'][0, k] = mv.to_square
            r['hcap'][0, k] = self.caps[j] if j < len(self.caps) else 0
            if j in self.secs:
                r['ht'][0, k] = self.secs[j]
        (x, c) = N.build_inputs(r)
        self.last_go = (n, my, opp)
        return (torch.from_numpy(x).to(self.dev), torch.from_numpy(c).to(self.dev), my)

    @torch.no_grad()
    def think(self, wtime, btime, winc, binc):
        t0 = time.time()
        (x, c, my_clock) = self.features(wtime, btime, winc, binc)
        white = self.board.turn == chess.WHITE
        legal = [m for m in self.board.legal_moves if m.promotion in (None, chess.QUEEN)]
        idx = []
        for mv in legal:
            (f, to) = (mv.from_square, mv.to_square)
            if not white:
                (f, to) = (int(N.flip_sq(f)), int(N.flip_sq(to)))
            idx.append(f * 64 + to)
        mask = torch.zeros(1, N.N_POLICY, dtype=torch.bool, device=self.dev)
        mask[0, idx] = True
        (h, p_t, a) = self.model.encode(x, c, mask)
        logits = p_t[0, idx].float().cpu().numpy() / self.cal_policy
        if self.temperature <= 0:
            choice = int(np.argmax(logits))
        else:
            z = logits / self.temperature
            z = np.exp(z - z.max())
            choice = int(np.random.choice(len(legal), p=z / z.sum()))
        mv = legal[choice]
        t = self.model.time_logits(h, p_t, a, torch.tensor([idx[choice]], device=self.dev))
        if self.mimic_time and (wtime > 0 or btime > 0):
            probs = torch.softmax(t[0].float() / self.cal_time, 0).cpu().numpy().astype(np.float64)
            b = int(np.random.choice(len(probs), p=probs / probs.sum()))
            target = min(N.sample_seconds(b), max(my_clock * 0.4, 0.0))
            wait = target - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
        self.secs[len(self.board.move_stack)] = time.time() - t0
        return mv

def main():
    eng = None
    out = sys.stdout
    for line in sys.stdin:
        tok = line.strip().split()
        if not tok:
            continue
        cmd = tok[0]
        if cmd == 'uci':
            out.write('id name maia-tim v4\nid author maia-tim-play\n')
            out.write('option name Temperature type spin default 100 min 0 max 300\n')
            out.write('option name MimicTime type check default true\n')
            out.write(f'option name Rating type spin default {DEFAULT_RATING} min 900 max 2600\n')
            out.write('option name OppRating type spin default 0 min 0 max 3500\n')
            out.write('uciok\n')
        elif cmd == 'isready':
            if eng is None:
                eng = Engine()
            out.write('readyok\n')
        elif cmd == 'setoption':
            if eng is None:
                eng = Engine()
            name = tok[tok.index('name') + 1] if 'name' in tok else ''
            val = tok[tok.index('value') + 1] if 'value' in tok else ''
            if name == 'Temperature':
                eng.temperature = int(val) / 100.0
            elif name == 'MimicTime':
                eng.mimic_time = val.lower() == 'true'
            elif name == 'Rating':
                eng.rating = int(val)
            elif name == 'OppRating':
                eng.opp_rating = int(val) or None
        elif cmd == 'ucinewgame':
            if eng is None:
                eng = Engine()
            eng.new_game()
        elif cmd == 'position':
            if eng is None:
                eng = Engine()
            if tok[1] == 'startpos':
                (fen, rest) = (None, tok[2:])
            else:
                end = tok.index('moves') if 'moves' in tok else len(tok)
                (fen, rest) = (' '.join(tok[2:end]), tok[end:])
            eng.set_position(fen, rest[1:] if rest and rest[0] == 'moves' else [])
        elif cmd == 'go':
            if eng is None:
                eng = Engine()
            kv = {'wtime': 0, 'btime': 0, 'winc': 0, 'binc': 0}
            for k in kv:
                if k in tok:
                    kv[k] = int(tok[tok.index(k) + 1])
            mv = eng.think(kv['wtime'], kv['btime'], kv['winc'], kv['binc'])
            out.write(f'bestmove {mv.uci()}\n')
        elif cmd == 'quit':
            break
        out.flush()
if __name__ == '__main__':
    main()
