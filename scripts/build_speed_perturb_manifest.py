# 3-way speed-perturb the train manifest (CR-CTC recipe)
from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
from src.slices.BuildManifest.SpeedPerturbManifest_Handler import build_speed_perturb_manifest


def main() -> None:
    cmd = SpeedPerturbManifestCommand(
        manifest_in="data/manifests/train.jsonl",
        manifest_out="data/manifests/train_sp.jsonl",
    )
    rows = build_speed_perturb_manifest(cmd)
    print(f"{cmd.manifest_out}: {rows} utts ({len(cmd.speeds)}x speed-perturbed)")


if __name__ == "__main__":
    main()
