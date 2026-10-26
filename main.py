"""Launch SignalLab or its dedicated MATLAB inspector."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect", action="store_true", help="Open the MATLAB dataset inspector"
    )
    args = parser.parse_args()
    try:
        if args.inspect:
            from signallab.inspector import main as launch
        else:
            from signallab.analyzer import main as launch
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "\nSignalLab could not start because a required dependency is missing.\n"
            "\nUse the assigned environment and install the project requirements first:\n"
            "  conda activate saleYar\n"
            "  python -m pip install -r requirements.txt\n"
            "\nIf you are using the managed workspace, open the project workspace for the assigned interpreter.\n"
            f"Original error: {type(exc).__name__}: {exc}\n"
        )
        raise SystemExit(2)
    launch()


if __name__ == "__main__":
    main()
