# CLI entry: decode one FLAC with the trained transducer checkpoint. Heavy/GPU runs are the user's.
import argparse

import torch

from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--checkpoint", default="data/checkpoints/transducer_avg.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe500.model")
    ap.add_argument("--offline", action="store_true")
    # LM rescoring is off unless --lm-weight > 0 (decode.yaml's zero is the pure-acoustic decoder).
    ap.add_argument(
        "--lm-weight", type=float, default=None, help="rescoring alpha; >0 loads the LM"
    )
    ap.add_argument("--ilm-weight", type=float, default=None, help="ILME subtraction beta")
    ap.add_argument("--length-bonus", type=float, default=None, help="per-token rescoring bonus")
    ap.add_argument("--lm-checkpoint", default=None, help="default decode.lm_checkpoint")
    ap.add_argument("--lm-tokenizer", default=None, help="default lm.tokenizer")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransducerModel()
    load_checkpoint(args.checkpoint, model)  # maps to CPU, loads the "model" key
    model = model.to(device).eval()
    handler = StreamingDecoder_Handler(
        model,
        SentencePieceTokenizer(args.tokenizer),
        lm_weight=args.lm_weight,
        ilm_weight=args.ilm_weight,
        length_bonus=args.length_bonus,
        lm_checkpoint=args.lm_checkpoint,
        lm_tokenizer=args.lm_tokenizer,
    )
    with torch.no_grad():
        resp = handler.decode(
            StreamingDecode_Command(audio_path=args.audio_path, streaming=not args.offline)
        )
    lat = resp.first_partial_latency_s
    print(
        f"[{'offline' if args.offline else 'streaming'}] rtf={resp.rtf:.3f} "
        f"audio={resp.audio_s:.2f}s decode={resp.decode_s:.2f}s finalize={resp.finalize_s:.3f}s "
        f"first_partial={'n/a (offline)' if lat is None else f'{lat:.3f}s'}"
    )
    print(resp.text)


if __name__ == "__main__":
    main()
