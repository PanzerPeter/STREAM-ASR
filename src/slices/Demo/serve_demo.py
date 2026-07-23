# src/slices/Demo/serve_demo.py — entry point for the local demo server.
# Loads the trained transducer checkpoint once, builds the Decode handler (offline for uploads, the
# same config drives live StreamingSessions), and serves the browser UI on 127.0.0.1. Heavy/GPU
# runs are the user's; this only holds one model resident and answers local requests.
#
# Run: PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo  (open http://127.0.0.1:8000)
# The LM re-ranks the acoustic n-best; it is off by default (decode.lm_weight=0 reproduces the
# plain decoder). Pass the weights tuned on dev by `evaluate.py --tune` to run the demo at the
# configuration the reported WER was measured with: --lm-weight 0.6 --ilm-weight 0.2.
import argparse

import torch
import uvicorn

from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Demo.DemoServer_Handler import build_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/checkpoints/transducer_best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe500.model")
    ap.add_argument("--host", default="127.0.0.1")  # local-only; no auth on this server
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--lm-weight",
        type=float,
        default=None,
        help="n-best rescoring weight (alpha); >0 loads the LM. Tuned value: 0.6",
    )
    ap.add_argument(
        "--ilm-weight",
        type=float,
        default=None,
        help="ILME subtraction weight (beta); only acts alongside --lm-weight. Tuned value: 0.2",
    )
    ap.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="beam width; 1 = greedy (faster, ~0.8 abs WER worse). Default: decode.beam_size",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransducerModel()
    load_checkpoint(args.checkpoint, model)
    model = model.to(device).eval()
    handler = StreamingDecoder_Handler(
        model,
        SentencePieceTokenizer(args.tokenizer),
        beam_size=args.beam_size,
        lm_weight=args.lm_weight,
        ilm_weight=args.ilm_weight,
    )
    # Echo the *resolved* decode configuration (CLI flag or decode.yaml fallback), so a demo run
    # that silently fell back to the alpha=0 regression lock is visible instead of just sounding
    # slightly worse than the reported numbers.
    lm = f"alpha={handler.lm_weight} beta={handler.ilm_weight}" if handler.lm_scorer else "off"
    print(
        f"STREAM ASR demo on http://{args.host}:{args.port}"
        f"  (device={device}, beam={handler.beam_size}, LM {lm})"
    )
    uvicorn.run(build_app(handler), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
