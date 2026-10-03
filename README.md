# maia-tim-play

A neural network that imitates one bullet player — both the moves and the
time taken over them. Play it in your browser, or watch it play itself.

**Play in your browser, nothing to install:** https://tflopp.github.io/maia-tim-play/

Or run it locally:

## Requirements

- Python 3.10 or newer
- PyTorch, NumPy and python-chess:

```
pip install -r requirements.txt
```

A CPU is enough. For a smaller, CPU-only PyTorch download:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy chess
```

## Play

- Windows: double-click `play.bat`
- macOS / Linux: `./play.sh`

It opens http://localhost:8765 in your browser. Options: `--port N`,
`--gpu` (use a CUDA GPU), `--no-browser`.

In the page: pick a time control (30+0 or 1+0), your colour, the bot's era
(its rating at the time), your rating, and press Start. Premoves work as on
Lichess. "Watch it play itself" lets two eras play each other. Finished games
are saved as PGN in `games/`.

## Local play only

Do not use this bot, or anything derived from it, while playing on Lichess
or any other site. That is engine assistance and breaks their rules.

## License

GPL-3.0 (see `LICENSE`). It builds on python-chess and Lichess's chessground
board, both GPL-3.0.
