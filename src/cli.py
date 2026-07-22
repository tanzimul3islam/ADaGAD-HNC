import argparse
import logging
import sys

LOGGER = logging.getLogger("AdaGAD-HNC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AdaGAD-HNC",
        description="Adaptive Graph Anomaly Detection with Hard-Negative Curriculum",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", help="Run training")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--resume", default=None)

    p = sub.add_parser("eval", help="Evaluate a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/default.yaml")

    p = sub.add_parser("multi", help="Multi-seed evaluation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))

    p = sub.add_parser("ablate", help="Run ablation suite")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--suite", required=True)

    return parser


def main(argv: list | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from src.main import load_config, train

        cfg = load_config(args.config)
        train(cfg)
    elif args.command == "eval":
        from src.main import evaluate, load_config

        cfg = load_config(args.config)
        evaluate(cfg, args.ckpt)
    elif args.command == "multi":
        from src.experiments.run_multiseed import run_multiseed

        run_multiseed(args.config, args.seeds)
    elif args.command == "ablate":
        import yaml

        from src.experiments.ablations import run_ablation_suite

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_ablation_suite(cfg, args.suite)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
