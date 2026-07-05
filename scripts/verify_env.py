import torch

EXPECTED_CAPABILITY = (12, 0)  # RTX 5070 Blackwell sm_120


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — check the cu128 install.")

    cap = torch.cuda.get_device_capability()
    if cap != EXPECTED_CAPABILITY:
        raise SystemExit(
            f"Expected {EXPECTED_CAPABILITY}, got {cap} ({torch.cuda.get_device_name()})."
        )

    # Compiling a trivial module proves the toolchain works before any training run.
    compiled = torch.compile(torch.nn.Linear(8, 8).cuda())
    compiled(torch.randn(4, 8, device="cuda"))

    print(f"OK: {torch.cuda.get_device_name()} cap={cap} torch={torch.__version__}")


if __name__ == "__main__":
    main()
