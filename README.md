Neural Network Optimization Engine

An automated Deep Learning optimization engine that reduces the complexity of trained neural networks through structured pruning and fine-tuning, while maintaining an acceptable level of predictive performance.

The project focuses on the trade-off between model complexity and predictive accuracy, with the goal of producing smaller dense neural networks suitable for resource-constrained inference environments.

Problem

Deep learning models can contain more parameters than necessary for their prediction task. These redundant parameters increase model storage and computation requirements and can make deployment on resource-constrained environments more difficult.

This project builds an optimization pipeline that:

Profiles a trained neural network.
Identifies less important hidden neurons.
Removes neurons through structured pruning.
Fine-tunes the resulting smaller model.
Evaluates predictive performance.
Rejects models that exceed the allowed accuracy loss.
Selects the smallest valid model.
Saves the optimized model for later deployment.
Objective

The optimization objective is:

Minimize model parameter count while keeping accuracy loss within a configurable threshold.

The current experiment uses a maximum allowed accuracy loss of:

0.50 percentage points
Architecture

The baseline model is a fully connected Multilayer Perceptron (MLP):

784 → 256 → 128 → 10

where:

784 — MNIST image pixels
256 — first hidden layer
128 — second hidden layer
10 — output classes

The model uses ReLU activations between the linear layers.

Structured pruning reduces the hidden-layer dimensions while preserving the input and output dimensions.

Example:

Baseline:
784 → 256 → 128 → 10

Optimized:
784 → 102 → 51 → 10
Optimization Pipeline
Trained Baseline Model
        ↓
      Profile
        ↓
 Structured Pruning
        ↓
    Fine-Tuning
        ↓
     Evaluate
        ↓
Accuracy Constraint
    ↓          ↓
 Reject      Valid
               ↓
      Best Model Selection
               ↓
      Optimized Checkpoint

Each pruning level is evaluated independently from the original baseline model so that pruning does not accumulate across experiments.

Pruning Strategy

The primary optimization technique is structured neuron pruning.

For each hidden layer, neuron importance is estimated using the L1 magnitude of its outgoing weights. Less important neurons are removed and the adjacent layers are reconstructed to create a smaller dense MLP.

The project also contains an unstructured pruning experiment for comparison.

Unstructured pruning

Individual weights are removed while the original layer dimensions remain unchanged.

Dense Layer
    ↓
Sparse Layer

This can produce high weight sparsity without necessarily producing proportional dense inference speed improvements.

Structured pruning

Entire hidden neurons are removed.

784 → 256 → 128 → 10
            ↓
784 → 102 → 51 → 10

The resulting model contains fewer parameters and smaller dense layers.

Experiment

The structured pruning sweep evaluates:

10%
20%
30%
40%
50%
60%
70%

For every candidate:

Start from the original baseline model.
Apply structured pruning.
Measure accuracy before fine-tuning.
Fine-tune the smaller model.
Measure final test accuracy.
Measure model complexity and latency.
Compare accuracy against the baseline.
Apply the accuracy-loss constraint.
Keep the smallest valid candidate.
Latest Result

The latest complete experiment selected:

Pruning:       60%
Architecture:  784 → 102 → 51 → 10
Parameters:    85,843
Accuracy:      97.48%
Accuracy loss: 0.03 percentage points

Baseline:

Architecture:  784 → 256 → 128 → 10
Parameters:    235,146
Accuracy:      97.51%

The selected optimized model therefore achieves approximately:

63.5% fewer trainable parameters

and approximately:

63.5% lower dense parameter storage

while remaining within the configured:

0.50 percentage-point accuracy-loss limit

The 70% pruning candidate was rejected because its accuracy decreased to 96.55%, corresponding to a 0.96 percentage-point loss.

Accuracy values after pruning represent the result of pruning followed by fine-tuning. They should not be interpreted as evidence that pruning itself improves accuracy.

Metrics

The engine records:

Total trainable parameters
Weight parameters
Non-zero weights
Weight sparsity
Estimated dense parameter storage
Test accuracy
Accuracy change
Inference latency
Parameter Storage

The reported model storage is an estimate based on the number of parameters and their tensor data types.

It represents dense parameter storage and is distinct from the serialized checkpoint file size.

Inference Latency

Latency measurements depend on the hardware, runtime, and execution environment.

Parameter reduction does not automatically guarantee proportional inference latency reduction.

Results

Structured pruning results are stored in:

results/structured_pruning_results.csv

Generated plots:

results/accuracy_vs_pruning.png
results/parameters_vs_pruning.png
results/model_size_vs_pruning.png
Project Structure
neural-network-optimization-engine/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── models/
│   ├── baseline_mlp.pth
│   └── optimized_mlp.pth
│
├── results/
│   ├── structured_pruning_results.csv
│   ├── accuracy_vs_pruning.png
│   ├── parameters_vs_pruning.png
│   └── model_size_vs_pruning.png
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── profiler.py
│   ├── pruning.py
│   ├── structured_pruning.py
│   ├── finetune.py
│   └── optimizer.py
│
└── experiments/
    ├── pruning_experiment.py
    ├── structured_pruning_experiment.py
    ├── verify_optimized_model.py
    └── plot_results.py
Setup

Create and activate the virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

uv pip install -r requirements.txt
Usage
Train the baseline
python src/train.py

The trained model is saved to:

models/baseline_mlp.pth
Evaluate the baseline
python src/evaluate.py
Profile the baseline
python src/profiler.py
Run structured pruning optimization
python experiments/structured_pruning_experiment.py

This performs the complete structured pruning sweep, evaluates candidates, selects the best valid model, and saves:

models/optimized_mlp.pth
Verify the optimized checkpoint
python experiments/verify_optimized_model.py

This reconstructs the optimized architecture from the checkpoint metadata, loads the saved weights, and independently evaluates the model.

Generate plots
python experiments/plot_results.py
Dependencies
Python
PyTorch
Torchvision
Matplotlib
Current Scope

The current implementation focuses on:

Fully connected MLP models
MNIST
Structured hidden-neuron pruning
L1-based neuron importance
Fine-tuning after pruning
Accuracy-constrained model selection
Model complexity profiling
Experiment result persistence
Optimized model checkpointing

The project intentionally focuses on the core optimization engine rather than deployment infrastructure.

Limitations

The current implementation:

Supports the implemented MLP architecture rather than arbitrary neural networks.
Uses MNIST as the experimental dataset.
Uses a fixed set of pruning levels.
Uses fixed fine-tuning epochs and learning rate.
Uses individual experimental runs rather than multi-seed statistical evaluation.
Measures latency on the local execution environment.
Does not guarantee proportional latency improvement from parameter reduction.
Does not currently provide a general-purpose regularization framework.
Future Development

After the core optimization engine is finalized, it can be extended with:

REST API
Model upload and optimization
Configurable optimization parameters
Experiment history
Result visualization interface
Additional pruning strategies
Additional neural network architectures
ONNX export
Quantization
Edge-device benchmarking

The API and interface are intentionally separate from the current core implementation.