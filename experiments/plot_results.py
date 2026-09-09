import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "pruning_results.csv"


def load_results():
    with open(RESULTS_PATH, newline="") as file:
        return list(csv.DictReader(file))


def main():
    results = load_results()

    pruning = [
        float(result["pruning"]) * 100
        for result in results
    ]

    sparsity = [
        float(result["sparsity"]) * 100
        for result in results
    ]

    accuracy = [
        float(result["accuracy"])
        for result in results
    ]

    latency = [
        float(result["latency_ms"])
        for result in results
    ]

    nonzero_parameters = [
        int(result["nonzero_parameters"])
        for result in results
    ]

    results_dir = PROJECT_ROOT / "results"

    plt.figure()
    plt.plot(sparsity, accuracy, marker="o")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Sparsity")
    plt.grid(True)
    plt.savefig(
        results_dir / "accuracy_vs_sparsity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(sparsity, nonzero_parameters, marker="o")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Non-zero Weights")
    plt.title("Non-zero Weights vs Sparsity")
    plt.grid(True)
    plt.savefig(
        results_dir / "nonzero_weights_vs_sparsity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(sparsity, latency, marker="o")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Inference Latency (ms)")
    plt.title("Inference Latency vs Sparsity")
    plt.grid(True)
    plt.savefig(
        results_dir / "latency_vs_sparsity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Plots saved to results/")


if __name__ == "__main__":
    main()
    