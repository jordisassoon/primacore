import argparse
import os

import pandas as pd

from primacore.models.rf import RF
from primacore.models.brt import BRT
from primacore.models.mat import MAT
from primacore.plots import scatter_predictions, line_predictions

parser = argparse.ArgumentParser(description="Train and predict climate variables")
parser.add_argument("filename", help="Path to test data CSV file")
parser.add_argument(
    "--model",
    choices=["MAT", "BRT", "RF"],
    default="RF",
    help="Model type: MAT, BRT, or RF",
)
args = parser.parse_args()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    climate = pd.read_csv(os.path.join(data_dir, "synthetic_climate_data.csv"))
    modern = pd.read_csv(os.path.join(data_dir, "synthetic_modern_data.csv"))
    test = pd.read_csv(args.filename)

    # Merge training data on OBSNAME
    train = modern.merge(climate, on="OBSNAME")

    feature_cols = [c for c in modern.columns if c != "OBSNAME"]
    target_cols = [c for c in climate.columns if c != "OBSNAME"]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    # Select model
    model_classes = {"MAT": MAT, "BRT": BRT, "RF": RF}
    model_cls = model_classes[args.model]

    # Collect non-feature metadata from test data (e.g. Age, Depth, OBSNAME)
    metadata_cols = [c for c in test.columns if c not in feature_cols]
    results = test[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=test.index)

    # Train a model per target and predict
    for target in target_cols:
        model = model_cls()
        model.fit(X_train, train[target])
        results[target] = model.predict(X_test)

    print(results)

    # Plot each predicted variable
    x_col = metadata_cols[0] if metadata_cols else None
    if x_col is None:
        results["sample"] = results.index
        x_col = "sample"

    for col in target_cols:
        scatter_predictions(results, x_col, col, title=f"{args.model} - {col}")
        line_predictions(results, x_col, col, title=f"{args.model} - {col}")

    return results


if __name__ == "__main__":
    main()