# Paired 2x speed-perturb the train manifest: every utterance once untouched, once at one factor
# drawn per-utterance from (0.9, 1.1). The output is exactly 2x the input, which is what keeps the
# fp16 mel cache affordable (53 GB per copy of the 961 h corpus) while still covering both
# directions across the corpus.
from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
from src.slices.BuildManifest.SpeedPerturbManifest_Handler import build_speed_perturb_manifest


def main() -> None:
    cmd = SpeedPerturbManifestCommand(
        manifest_in="data/manifests/train.jsonl",
        manifest_out="data/manifests/train_sp2.jsonl",
    )
    rows = build_speed_perturb_manifest(cmd)
    print(f"{cmd.manifest_out}: {rows} utts (2x: original + one of {cmd.speeds})")


if __name__ == "__main__":
    main()
