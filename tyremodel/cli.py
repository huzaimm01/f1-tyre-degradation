"""Command-line entry point: python -m tyremodel --year 2024 --event Monza"""
import argparse
import json
from pathlib import Path

from . import evaluate, features, models


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train lap-time models on one F1 race.")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--event", default="Monza")
    p.add_argument("--drivers", nargs="*", help="three-letter codes; default is every driver")
    p.add_argument("--target", choices=["corrected", "raw"], default="corrected")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from .data import load_laps

    raw = load_laps(args.year, args.event)
    total_laps = int(raw["LapNumber"].max())
    laps = features.clean_laps(raw)
    if args.drivers:
        laps = laps[laps["Driver"].isin(args.drivers)].reset_index(drop=True)
    laps = features.fuel_correct(laps, total_laps)

    if laps["stint_id"].nunique() < 4:
        raise SystemExit("Fewer than four stints after cleaning; use more drivers or another race.")

    train, test = features.split_by_stint(laps, seed=args.seed)
    X_train, y_train = features.make_xy(train, args.target)
    X_test, y_test = features.make_xy(test, args.target)

    fitted = {
        "Mean baseline": models.fit_mean(X_train, y_train),
        "Linear regression": models.fit_linear(X_train, y_train),
        "Neural network": models.LapTimeNet(args.seed).fit(X_train, y_train),
    }
    predictions = {name: m.predict(X_test) for name, m in fitted.items()}
    scores = {name: round(evaluate.mae(y_test, pred), 3) for name, pred in predictions.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluate.plot_predicted_vs_actual(y_test, predictions, out / "predicted_vs_actual.png")
    evaluate.plot_degradation(laps, out / "degradation_by_compound.png")

    metrics = {
        "race": f"{args.year} {args.event}",
        "target": args.target,
        "laps_train": len(train),
        "laps_test": len(test),
        "stints_train": int(train["stint_id"].nunique()),
        "stints_test": int(test["stint_id"].nunique()),
        "mae_seconds": scores,
        "degradation_seconds_per_lap": evaluate.degradation_rates(laps),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"{metrics['race']}: {len(train)} train laps, {len(test)} test laps")
    for name, score in scores.items():
        print(f"  {name:<18} MAE {score:.3f} s")
    print("  degradation (s/lap):", metrics["degradation_seconds_per_lap"])
