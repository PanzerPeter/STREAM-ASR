# CLI entry: decode one FLAC with the Stage-B hybrid checkpoint. Heavy/GPU runs are the user's.
import argparse

import torch

from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--checkpoint", default="data/checkpoints/stage_b_best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe500.model")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridCtcAttention()
    load_checkpoint(args.checkpoint, model)  # maps to CPU, loads the "model" key
    model = model.to(device).eval()
    handler = StreamingDecoder_Handler(model, SentencePieceTokenizer(args.tokenizer))
    with torch.no_grad():
        resp = handler.decode(
            StreamingDecode_Command(audio_path=args.audio_path, streaming=not args.offline)
        )
    print(
        f"[{'offline' if args.offline else 'streaming'}] rtf={resp.rtf:.3f} "
        f"latency={resp.first_partial_latency_s:.3f}s"
    )
    print(resp.text)


if __name__ == "__main__":
    main()
