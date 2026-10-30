export type ModelProfile = {
  model: string;
  architecture: string;
  device: string;
  total_parameters: number;
  weight_parameters: number;
  nonzero_weights: number;
  model_size_mb: number;
  inference_latency_ms: number;
};

export type ExperimentResult = {
  pruning: number;
  hidden1_size: number;
  hidden2_size: number;
  parameters: number;
  weight_parameters: number;
  nonzero_parameters: number;
  sparsity: number;
  model_size_mb: number;
  accuracy: number;
  accuracy_change: number;
  latency_ms: number;
};

export type FineTuningEpoch = {
  epoch: number;
  total_epochs: number;
  accuracy: number;
  loss: number;
};

export type CandidateResult = Omit<ExperimentResult, "sparsity"> & {
  sparsity?: number;
  before_finetuning_accuracy?: number;
  fine_tuning_history?: FineTuningEpoch[];
};

export type OptimizationResponse = {
  device: string;
  baseline_accuracy: number;
  max_accuracy_loss: number;
  best_model: CandidateResult;
  results: CandidateResult[];
};

export type OptimizationJobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress: number;
  message: string;
  candidate: number | null;
  total_candidates: number | null;
  epoch: number | null;
  total_epochs: number | null;
  accuracy: number | null;
  loss: number | null;
  result: OptimizationResponse | null;
  error: string | null;
};

export type ActivityLog = {
  id: number;
  time: string;
  message: string;
};

export type ManualStageKey = "prune" | "finetune" | "evaluate" | "select";

export type PipelineCandidate = {
  pruning: number;
  before_finetuning_accuracy: number;
  hidden1_size: number;
  hidden2_size: number;
  parameters: number;
  weight_parameters: number;
  nonzero_parameters: number;
  model_size_mb: number;
  latency_ms: number;
  fine_tuning_history?: FineTuningEpoch[];
};

export type PipelineResult = PipelineCandidate & {
  accuracy: number;
  accuracy_change: number;
};

export type PipelineStage =
  | "idle"
  | "pruning"
  | "pruned"
  | "finetuning"
  | "finetuned"
  | "evaluation"
  | "evaluated"
  | "selection"
  | "selected";

export type PipelineState = {
  stage: PipelineStage;
  baselineAccuracy: number | null;
  candidates: PipelineCandidate[];
  results: PipelineResult[];
  bestModel: PipelineResult | null;
};

export type WeightConnection = {
  source_index: number;
  target_index: number;
  weight: number;
};

export type WeightLayer = {
  name: string;
  size: number;
  sampled_indices: number[];
  sampled_biases: number[];
  importance: number[] | null;
};

export type WeightModelSnapshot = {
  model: "baseline" | "optimized";
  architecture: number[];
  total_parameters: number;
  layers: WeightLayer[];
  connections: Array<{
    source_layer: number;
    target_layer: number;
    weights: WeightConnection[];
  }>;
  importance_method: string;
  sample_note: string;
};

export type NeuronExample = {
  index: number;
  importance: number;
  bias: number | null;
  top_weights: Array<{
    source_index: number;
    weight: number;
  }>;
};

export type PruningLayerInfo = {
  name: string;
  original_size: number;
  keep_count: number;
  removed_count: number;
  kept_indices: number[];
  removed_indices: number[];
  importance: number[];
  lowest_retained: NeuronExample | null;
  highest_removed: NeuronExample | null;
};

export type PruningResponse = {
  level: number;
  baseline_architecture: number[];
  pruned_architecture: number[];
  baseline_parameters: number;
  pruned_parameters: number;
  parameter_reduction_pct: number;
  layers: PruningLayerInfo[];
  importance_method: string;
  retention_rule: string;
  note: string;
};

export type BaselineMetrics = {
  architecture: number[];
  parameters: number;
  weight_parameters: number;
  nonzero_parameters: number;
  model_size_mb: number;
  checkpoint_size_mb: number | null;
  latency_ms: number;
  test_accuracy: number;
};

export type PrunedAnalysisMetrics = {
  architecture: number[];
  parameters: number;
  weight_parameters: number;
  nonzero_parameters: number;
  model_size_mb: number;
  latency_ms: number;
  before_finetuning_accuracy: number;
  accuracy_change_before_finetuning: number;
};

export type ModelAnalysis = {
  level: number | null;
  device: string;
  baseline_accuracy: number;
  baseline: BaselineMetrics;
  pruned: PrunedAnalysisMetrics | null;
  layers: PruningLayerInfo[];
  importance_method: string;
  retention_rule: string;
  note: string;
};

export type ActivationStats = {
  size: number;
  mean: number;
  max: number;
  fraction_zero: number;
};

export type SamplePrediction = {
  index: number;
  label: number;
  pixels: number[][];
  baseline: {
    logits: number[];
    probabilities: number[];
    prediction: number;
    confidence: number;
    activation_stats: {
      hidden1: ActivationStats;
      hidden2: ActivationStats;
    };
  };
  optimized: {
    logits: number[];
    probabilities: number[];
    prediction: number;
    confidence: number;
    activation_stats: {
      hidden1: ActivationStats;
      hidden2: ActivationStats;
    };
  } | null;
  neuron_example: {
    layer: string;
    neuron_index: number;
    layer_size: number;
    bias: number;
    pre_activation: number;
    activation: number;
    top_contributions: Array<{
      pixel_index: number;
      row: number;
      col: number;
      pixel_value: number;
      weight: number;
      contribution: number;
    }>;
    note: string;
  };
  note: string;
};
