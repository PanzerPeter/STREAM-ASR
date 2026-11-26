# CMVN over the training sources exactly as the trainer reads them: train_sp2 (speed-perturbed
# copies included) and cv-train (trimmed spans), so each weighs in by its share of training frames.
from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand
from src.slices.ComputeCmvn.ComputeCmvn_Handler import compute_cmvn


def main() -> None:
    stats = compute_cmvn(
        ComputeCmvnCommand(
            manifests=("data/manifests/train_sp2.jsonl", "data/manifests/cv-train.jsonl"),
            cmvn_out="data/features/cmvn.pt",
            sample_frac=0.05,
            seed=0,
        )
    )
    print(f"CMVN over train: mean[0]={stats['mean'][0]:.3f} std[0]={stats['std'][0]:.3f}")


if __name__ == "__main__":
    main()
