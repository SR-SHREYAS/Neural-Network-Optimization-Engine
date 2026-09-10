# experiments/plot_results.py

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "structured_pruning_results.csv"
)


def load_results():
    with open(
        RESULTS_PATH,
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main():
    results = load_results()

    pruning = [
        float(result["pruning"]) * 100
        for result in results
    ]

    accuracy = [
        float(result["accuracy"])
        for result in results
    ]

    parameters = [
        int(result["parameters"])
        for result in results
    ]

    model_size = [
        float(result["model_size_mb"])
        for result in results
    ]

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Accuracy vs Pruning
    # ---------------------------------------------------------

    plt.figure()
    plt.plot(
        pruning,
        accuracy,
        marker="o",
    )
    plt.xlabel("Structured Pruning (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Structured Pruning")
    plt.grid(True)
    plt.savefig(
        results_dir / "accuracy_vs_pruning.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Parameters vs Pruning
    # ---------------------------------------------------------

    plt.figure()
    plt.plot(
        pruning,
        parameters,
        marker="o",
    )
    plt.xlabel("Structured Pruning (%)")
    plt.ylabel("Trainable Parameters")
    plt.title("Model Parameters vs Structured Pruning")
    plt.grid(True)
    plt.savefig(
        results_dir / "parameters_vs_pruning.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Model Storage vs Pruning
    # ---------------------------------------------------------

    plt.figure()
    plt.plot(
        pruning,
        model_size,
        marker="o",
    )
    plt.xlabel("Structured Pruning (%)")
    plt.ylabel("Estimated Parameter Storage (MB)")
    plt.title("Model Storage vs Structured Pruning")
    plt.grid(True)
    plt.savefig(
        results_dir / "model_size_vs_pruning.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Structured pruning plots saved to results/")


if __name__ == "__main__":
    main()