# scripts/compute_cmvn.py
from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand
from src.slices.ComputeCmvn.ComputeCmvn_Handler import compute_cmvn


def main() -> None:
    stats = compute_cmvn(
        ComputeCmvnCommand(
            manifest="data/manifests/train.jsonl",
            cmvn_out="data/features/cmvn.pt",
            sample_frac=0.15,
            seed=0,
        )
    )
    print(f"CMVN over train: mean[0]={stats['mean'][0]:.3f} std[0]={stats['std'][0]:.3f}")


if __name__ == "__main__":
    main()
