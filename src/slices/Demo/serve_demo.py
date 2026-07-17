# src/slices/Demo/serve_demo.py — entry point for the local demo server.
# Loads the trained transducer checkpoint once, builds the Decode handler (offline for uploads, the
# same config drives live StreamingSessions), and serves the browser UI on 127.0.0.1. Heavy/GPU
# runs are the user's; this only holds one model resident and answers local requests.
#
# Run: PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo  (open http://127.0.0.1:8000)
# LM shallow fusion is off by default (lm_weight=0 reproduces the plain decoder); pass --lm-weight
# to enable it in the live/upload paths.
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
        "--lm-weight", type=float, default=None, help="LM fusion alpha; >0 loads the LM"
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransducerModel()
    load_checkpoint(args.checkpoint, model)
    model = model.to(device).eval()
    handler = StreamingDecoder_Handler(
        model, SentencePieceTokenizer(args.tokenizer), lm_weight=args.lm_weight
    )
    print(f"STREAM ASR demo on http://{args.host}:{args.port}  (device={device})")
    uvicorn.run(build_app(handler), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
