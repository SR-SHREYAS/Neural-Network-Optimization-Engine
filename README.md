# Neural Network Optimization Engine

An MNIST experiment pipeline for reducing the size of a trained PyTorch MLP while keeping its predictive accuracy within a configurable limit.

The project compares two pruning approaches:

- **Structured pruning:** removes complete hidden neurons and rebuilds a smaller dense network.
- **Unstructured pruning:** removes individual weights while keeping the original layer shapes.

The main optimization path profiles a baseline model, evaluates several structured-pruning levels, fine-tunes each candidate, and saves the smallest candidate that satisfies the accuracy constraint.

## Results at a glance

The checked-in structured-pruning results were produced with a maximum allowed accuracy loss of **0.50 percentage points**.

| Model | Architecture | Parameters | Test accuracy | Accuracy change |
| --- | --- | ---: | ---: | ---: |
| Baseline | `784 -> 256 -> 128 -> 10` | 235,146 | 97.51% | 0.00 pp |
| Selected optimized model | `784 -> 102 -> 51 -> 10` | 85,843 | 97.48% | -0.03 pp |

The selected model uses 60% structured pruning and has approximately **63.5% fewer trainable parameters** than the baseline. The 70% candidate was rejected because its accuracy fell to 96.55%, a loss of 0.96 percentage points.

These numbers are one experiment run, not a multi-seed benchmark. Accuracy and latency can vary with the installed library versions, hardware, and random seed.

## Model and optimization method

The baseline is a fully connected MLP for 28 x 28 MNIST images:

```text
784 -> 256 -> 128 -> 10
```

ReLU activations follow the two hidden linear layers. For structured pruning, neuron importance is estimated from the L1 magnitude of outgoing weights. The least important neurons are removed from each hidden layer, the adjacent linear layers are reconstructed, and the reduced model is fine-tuned.

The structured sweep evaluates pruning levels of 10%, 20%, 30%, 40%, 50%, 60%, and 70%. Every candidate starts from the original baseline checkpoint, so pruning does not accumulate across the sweep.

Candidate selection follows this rule:

1. Reject candidates whose accuracy loss exceeds the configured limit.
2. Among valid candidates, prefer the model with fewer trainable parameters.
3. If parameter counts tie, prefer the higher-accuracy model.

## Quick start

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The training and experiment scripts download MNIST through `torchvision` into `data/` when the dataset is not already available.

### 2. Train the baseline

```bash
python src/train.py
```

Output:

```text
models/baseline_mlp.pth
```

### 3. Evaluate and profile the baseline

```bash
python src/evaluate.py
python src/profiler.py
```

The profiler reports trainable parameters, weight counts, non-zero weights, serialized checkpoint size, and single-sample inference latency on the selected device.

### 4. Run structured optimization

```bash
python experiments/structured_pruning_experiment.py
```

This runs the full sweep, fine-tunes every structured candidate, writes the CSV results, selects the best valid candidate, and saves:

```text
models/optimized_mlp.pth
results/structured_pruning_results.csv
```

### 5. Verify the optimized checkpoint

```bash
python experiments/verify_optimized_model.py
```

The verification script reads the hidden-layer sizes stored in the checkpoint, reconstructs the matching MLP, loads its weights, and evaluates it independently on the MNIST test set.

### 6. Generate plots

```bash
python experiments/plot_results.py
```

Plots are written to `results/`:

- `accuracy_vs_pruning.png`
- `parameters_vs_pruning.png`
- `model_size_vs_pruning.png`

## Results schema

`results/structured_pruning_results.csv` records the following fields for the baseline and each candidate:

| Field | Meaning |
| --- | --- |
| `pruning` | Fraction of hidden neurons removed |
| `hidden1_size`, `hidden2_size` | Resulting hidden-layer widths |
| `parameters` | Total trainable parameters, including biases |
| `weight_parameters` | Parameters belonging to linear-layer weights |
| `nonzero_parameters` | Non-zero linear-layer weights |
| `sparsity` | Weight sparsity after structured pruning |
| `model_size_mb` | Estimated dense parameter storage |
| `accuracy` | Final MNIST test accuracy after fine-tuning |
| `accuracy_change` | Accuracy change relative to the baseline, in percentage points |
| `latency_ms` | Measured single-sample inference latency |

The estimated parameter storage is different from the serialized `.pth` checkpoint size. Latency is hardware- and runtime-dependent; fewer parameters do not guarantee a proportional speedup.

## Unstructured pruning comparison

The repository also includes an unstructured pruning experiment:

```bash
python experiments/pruning_experiment.py
```

It applies global L1 unstructured pruning to linear-layer weights without changing the MLP architecture. Its outputs are stored in:

```text
results/pruning_results.csv
```

This approach can increase sparsity, but sparse weights do not automatically produce the same dense-inference gains as smaller structured layers.

## Repository layout

```text
neural-network-optimization-engine/
├── data/                         # MNIST files
├── models/                       # Baseline and optimized checkpoints
├── results/                      # CSV metrics and generated plots
├── src/
│   ├── model.py                  # MLP definition
│   ├── train.py                  # Baseline training
│   ├── evaluate.py               # Test-set evaluation
│   ├── profiler.py               # Model metrics and latency
│   ├── pruning.py                # Unstructured pruning helper
│   ├── structured_pruning.py     # Structured neuron pruning
│   ├── finetune.py               # Candidate fine-tuning
│   └── optimizer.py              # Candidate selection and checkpointing
└── experiments/
    ├── pruning_experiment.py
    ├── structured_pruning_experiment.py
    ├── verify_optimized_model.py
    └── plot_results.py
```

## Dependencies

- Python 3
- PyTorch
- Torchvision
- NumPy
- scikit-learn
- Matplotlib

Exact package versions are pinned in [requirements.txt](requirements.txt).

## Scope and limitations

This is an experiment-focused implementation rather than a general model-serving system. It currently targets the implemented MLP architecture and MNIST, uses fixed pruning levels and fine-tuning settings, and does not provide multi-seed statistical analysis, quantization, ONNX export, or edge-device benchmarking.

Natural next steps include configurable experiment parameters, additional architectures and datasets, quantization, ONNX export, and deployment-oriented benchmarks.
