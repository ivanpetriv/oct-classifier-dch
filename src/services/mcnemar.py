from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import _Bunch, mcnemar  # type: ignore


def run_mcnemar(csv_model_a: Path, csv_model_b: Path):
    """Runs a McNemar test to determine a statistically significant change between two tables.

    Args:
        csv_model_a (Path): A table.
        csv_model_b (Path): A table.

    Raises:
        ValueError: _description_
    """
    df_a = pd.read_csv(csv_model_a)
    df_b = pd.read_csv(csv_model_b)

    if not (df_a["image_path"] == df_b["image_path"]).all():
        raise ValueError(
            "Image paths do not match. Ensure tests were run on the identical dataset."
        )

    # Convert to boolean arrays (True if the model predicted correctly, False if wrong)
    a_correct = df_a["y_true"] == df_a["y_pred"]
    b_correct = df_b["y_true"] == df_b["y_pred"]

    # Build the 2x2 contingency table
    table = [
        [sum(a_correct & b_correct), sum(a_correct & ~b_correct)],
        [sum(~a_correct & b_correct), sum(~a_correct & ~b_correct)],
    ]

    print("Contingency Table:")
    print(f"Both Correct: {table[0][0]} | A Correct, B Wrong: {table[0][1]}")
    print(f"A Wrong, B Correct: {table[1][0]} | Both Wrong: {table[1][1]}\n")

    result: _Bunch = mcnemar(table, exact=False, correction=True)

    print(f"Statistic: {result.statistic:.4f}")
    print(f"p-value:   {result.pvalue:.4e}")

    alpha = 0.05
    print(
        f"Value: {result.pvalue}. Statistically significant: {result.pvalue < alpha}",
    )
