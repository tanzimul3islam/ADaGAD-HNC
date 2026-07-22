import argparse

import pandas as pd


def generate_report(csv_paths: list[str], output_path: str):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_path + ".csv", index=False)
    markdown = df.to_markdown(index=False)
    with open(output_path + ".md", "w") as f:
        f.write(markdown)
    latex = df.to_latex(index=False)
    with open(output_path + ".tex", "w") as f:
        f.write(latex)
    print(f"Report saved to {output_path}.{{csv,md,tex}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/results_table")
    args = parser.parse_args()
    generate_report(args.input, args.output)


if __name__ == "__main__":
    main()
