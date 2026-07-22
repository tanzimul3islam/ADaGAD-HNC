import argparse
import glob
from pathlib import Path

import pandas as pd


def export_results(input_pattern: str, output: str):
    files = glob.glob(input_pattern)
    if not files:
        raise ValueError(f"No files found for pattern: {input_pattern}")
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    output = Path(output)
    combined.to_csv(output.with_suffix(".csv"), index=False)
    with open(output.with_suffix(".md"), "w") as f:
        f.write(combined.to_markdown(index=False))
    with open(output.with_suffix(".tex"), "w") as f:
        f.write(combined.to_latex(index=False))
    print(f"Exported results to {output.with_suffix('')}.{{csv,md,tex}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_results(args.input, args.output)


if __name__ == "__main__":
    main()
