import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

type ModelProfile = {
  model: string;
  architecture: string;
  device: string;
  total_parameters: number;
  weight_parameters: number;
  nonzero_weights: number;
  model_size_mb: number;
  inference_latency_ms: number;
};

type ExperimentResult = {
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

type OptimizationResponse = {
  device: string;
  baseline_accuracy: number;
  max_accuracy_loss: number;
  best_model: ExperimentResult & {
    before_finetuning_accuracy: number;
  };
  results: Array<
    ExperimentResult & {
      before_finetuning_accuracy: number;
    }
  >;
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatArchitecture(result: ExperimentResult) {
  return `784 → ${result.hidden1_size} → ${result.hidden2_size} → 10`;
}

function App() {
  const [profile, setProfile] = useState<ModelProfile | null>(null);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [health, setHealth] = useState(false);
  const [loading, setLoading] = useState(true);
  const [optimizing, setOptimizing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [pruningLevels, setPruningLevels] = useState(
    "10, 20, 30, 40, 50, 60, 70",
  );
  const [maxAccuracyLoss, setMaxAccuracyLoss] = useState("0.50");
  const [optimization, setOptimization] =
    useState<OptimizationResponse | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);

        const [healthResponse, profileResponse, resultsResponse] =
          await Promise.all([
            fetch(`${API_BASE_URL}/health`),
            fetch(`${API_BASE_URL}/model/profile`),
            fetch(`${API_BASE_URL}/results`),
          ]);

        if (!healthResponse.ok) {
          throw new Error("Optimization API is unavailable.");
        }

        if (!profileResponse.ok || !resultsResponse.ok) {
          throw new Error("Unable to load model information.");
        }

        setHealth(true);
        setProfile(await profileResponse.json());
        setResults(await resultsResponse.json());
      } catch (err) {
        setHealth(false);
        setError(
          err instanceof Error
            ? err.message
            : "Unable to connect to the optimization API.",
        );
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const displayedResults = optimization?.results ?? results;

  const baseline = useMemo(
    () => displayedResults.find((result) => result.pruning === 0),
    [displayedResults],
  );

  const bestModel =
    optimization?.best_model ??
    results
      .filter((result) => result.pruning > 0)
      .reduce<ExperimentResult | null>(
        (best, result) =>
          !best || result.parameters < best.parameters ? result : best,
        null,
      );

  const parameterReduction =
    profile && bestModel
      ? ((profile.total_parameters - bestModel.parameters) /
          profile.total_parameters) *
        100
      : 0;

  const sizeReduction =
    profile && bestModel
      ? ((profile.model_size_mb - bestModel.model_size_mb) /
          profile.model_size_mb) *
        100
      : 0;

  async function runOptimization() {
    try {
      setOptimizing(true);
      setError(null);

      const levels = pruningLevels
        .split(",")
        .map((value) => Number(value.trim()) / 100)
        .filter((value) => !Number.isNaN(value));

      if (
        levels.length === 0 ||
        levels.some((level) => level < 0 || level >= 1)
      ) {
        throw new Error(
          "Pruning levels must be percentages between 0 and 99.",
        );
      }

      const loss = Number(maxAccuracyLoss);

      if (Number.isNaN(loss) || loss < 0) {
        throw new Error("Maximum accuracy loss must be non-negative.");
      }

      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pruning_levels: levels,
          max_accuracy_loss: loss,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Optimization failed.");
      }

      setOptimization(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Optimization failed.",
      );
    } finally {
      setOptimizing(false);
    }
  }

  return (
    <div className="site">
      <header className="header">
        <a className="wordmark" href="/">
          NNOE
        </a>

        <div className="header-meta">
          <span>Neural Network Optimization Engine</span>
          <span className={`connection ${health ? "online" : ""}`}>
            <i />
            {health ? "Connected" : "Disconnected"}
          </span>
        </div>
      </header>

      <main>
        <section className="intro">
          <span className="label">MODEL OPTIMIZATION</span>
          <h1>
            Make the network
            <br />
            smaller.
          </h1>
          <p>
            An automated pruning engine for finding a lower-complexity neural
            network while preserving acceptable predictive performance.
          </p>
        </section>

        {error && (
          <div className="error">
            <strong>Error</strong>
            <span>{error}</span>
          </div>
        )}

        <section className="section model-section">
          <SectionHeading number="01" title="Model" />

          <div className="model-line">
            <div>
              <span className="label">ARCHITECTURE</span>
              <strong className="architecture">
                {profile?.architecture ?? "784 → 256 → 128 → 10"}
              </strong>
            </div>

            <div className="model-device">
              <span className="label">RUNTIME</span>
              <strong>{profile?.device ?? "—"}</strong>
            </div>
          </div>

          <div className="metrics">
            <Metric
              label="Test accuracy"
              value={baseline ? `${baseline.accuracy.toFixed(2)}%` : "—"}
            />
            <Metric
              label="Parameters"
              value={
                profile ? formatNumber(profile.total_parameters) : "—"
              }
            />
            <Metric
              label="Model size"
              value={
                profile ? `${profile.model_size_mb.toFixed(2)} MB` : "—"
              }
            />
            <Metric
              label="Inference"
              value={
                profile
                  ? `${profile.inference_latency_ms.toFixed(3)} ms`
                  : "—"
              }
            />
          </div>
        </section>

        <section className="section optimize-section">
          <SectionHeading number="02" title="Optimize" />

          <div className="controls">
            <div className="control">
              <label htmlFor="levels">Pruning levels</label>
              <div className="input-row">
                <input
                  id="levels"
                  value={pruningLevels}
                  onChange={(event) =>
                    setPruningLevels(event.target.value)
                  }
                />
                <span>%</span>
              </div>
              <small>Comma-separated. Each level is evaluated independently.</small>
            </div>

            <div className="control">
              <label htmlFor="loss">Maximum accuracy loss</label>
              <div className="input-row">
                <input
                  id="loss"
                  type="number"
                  min="0"
                  step="0.05"
                  value={maxAccuracyLoss}
                  onChange={(event) =>
                    setMaxAccuracyLoss(event.target.value)
                  }
                />
                <span>pp</span>
              </div>
              <small>
                Candidates exceeding this loss are rejected.
              </small>
            </div>
          </div>

          <div className="action-row">
            <button
              className="run-button"
              onClick={runOptimization}
              disabled={optimizing || loading}
            >
              {optimizing ? "Optimizing..." : "Run optimization"}
              <span>↗</span>
            </button>

            <span className="action-note">
              Structured pruning · fine-tuning · evaluation
            </span>
          </div>
        </section>

        <section className="section result-section">
          <SectionHeading number="03" title="Result" />

          {bestModel ? (
            <div className="result">
              <div className="result-main">
                <span className="label">SELECTED MODEL</span>
                <div className="result-pruning">
                  {Math.round(bestModel.pruning * 100)}%
                </div>
                <span className="result-caption">pruning</span>
              </div>

              <div className="result-architecture">
                <span className="label">ARCHITECTURE</span>
                <strong>{formatArchitecture(bestModel)}</strong>
              </div>

              <div className="result-stats">
                <Metric
                  label="Accuracy"
                  value={`${bestModel.accuracy.toFixed(2)}%`}
                />
                <Metric
                  label="Parameters"
                  value={formatNumber(bestModel.parameters)}
                />
                <Metric
                  label="Size"
                  value={`${bestModel.model_size_mb.toFixed(2)} MB`}
                />
                <Metric
                  label="Accuracy change"
                  value={`${bestModel.accuracy_change >= 0 ? "+" : ""}${bestModel.accuracy_change.toFixed(2)} pp`}
                />
              </div>

              <div className="result-reduction">
                <div>
                  <span>Parameter reduction</span>
                  <strong>−{parameterReduction.toFixed(1)}%</strong>
                </div>
                <div>
                  <span>Size reduction</span>
                  <strong>−{sizeReduction.toFixed(1)}%</strong>
                </div>
              </div>

              <a
                className="download"
                href={`${API_BASE_URL}/model/optimized`}
                download="optimized_mlp.pth"
              >
                Download optimized model
                <span>↓</span>
              </a>
            </div>
          ) : (
            <div className="no-result">
              <span>No optimized model selected.</span>
              <span>Run an optimization sweep above.</span>
            </div>
          )}
        </section>

        <section className="section experiment-section">
          <SectionHeading number="04" title="Experiment" />

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Pruning</th>
                  <th>Architecture</th>
                  <th>Parameters</th>
                  <th>Size</th>
                  <th>Accuracy</th>
                  <th>Δ Accuracy</th>
                  <th>Latency</th>
                </tr>
              </thead>

              <tbody>
                {displayedResults.map((result) => {
                  const selected =
                    bestModel &&
                    result.pruning === bestModel.pruning &&
                    result.parameters === bestModel.parameters;

                  return (
                    <tr
                      key={`${result.pruning}-${result.parameters}`}
                      className={selected ? "selected-row" : ""}
                    >
                      <td>
                        {result.pruning === 0
                          ? "Baseline"
                          : `${Math.round(result.pruning * 100)}%`}
                      </td>
                      <td className="mono">
                        {formatArchitecture(result)}
                      </td>
                      <td>{formatNumber(result.parameters)}</td>
                      <td>{result.model_size_mb.toFixed(3)} MB</td>
                      <td>{result.accuracy.toFixed(2)}%</td>
                      <td
                        className={
                          result.accuracy_change < 0
                            ? "negative"
                            : "positive"
                        }
                      >
                        {result.accuracy_change >= 0 ? "+" : ""}
                        {result.accuracy_change.toFixed(2)} pp
                      </td>
                      <td>{result.latency_ms.toFixed(3)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer>
        <span>NNOE</span>
        <span>PyTorch · FastAPI · React</span>
      </footer>
    </div>
  );
}

function SectionHeading({
  number,
  title,
}: {
  number: string;
  title: string;
}) {
  return (
    <div className="section-heading">
      <span>{number}</span>
      <h2>{title}</h2>
    </div>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;