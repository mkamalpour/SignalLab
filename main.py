"""Launch SignalLab or its dedicated MATLAB inspector."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect", action="store_true", help="Open the MATLAB dataset inspector"
    )
    args = parser.parse_args()
    if args.inspect:
        from signallab.inspector import main as launch
    else:
        from signallab.analyzer import main as launch
    launch()


if __name__ == "__main__":
    main()
