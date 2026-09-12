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

function formatPercent(value: number) {
  return `${value.toFixed(2)}%`;
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
    async function loadDashboard() {
      try {
        setLoading(true);
        setError(null);

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
          throw new Error("Failed to load model data.");
        }

        const profileData: ModelProfile = await profileResponse.json();
        const resultsData: ExperimentResult[] =
          await resultsResponse.json();

        setHealth(true);
        setProfile(profileData);
        setResults(resultsData);
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

    loadDashboard();
  }, []);

  const baseline = useMemo(
    () => results.find((result) => result.pruning === 0),
    [results],
  );

  const displayedResults = optimization?.results ?? results;

  const bestModel =
    optimization?.best_model ??
    results
      .filter((result) => result.pruning > 0)
      .reduce<ExperimentResult | null>(
        (best, result) =>
          !best || result.parameters < best.parameters ? result : best,
        null,
      );

  const parameterReduction = profile && bestModel
    ? ((profile.total_parameters - bestModel.parameters) /
        profile.total_parameters) *
      100
    : 0;

  const sizeReduction = profile && bestModel
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
        throw new Error("Maximum accuracy loss must be a non-negative number.");
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
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <p className="eyebrow">Deep Learning Engine</p>
            <h1>Neural Network Optimization Engine</h1>
          </div>
        </div>

        <div className={`api-status ${health ? "online" : "offline"}`}>
          <span className="status-dot" />
          {health ? "API Online" : "API Offline"}
        </div>
      </header>

      <main className="dashboard">
        <section className="hero-section">
          <div>
            <p className="section-kicker">MODEL OPTIMIZATION</p>
            <h2>Reduce complexity without losing performance.</h2>
            <p className="hero-description">
              Profile a trained neural network, evaluate structured pruning
              levels, fine-tune candidates, and select the most efficient model
              within your accuracy constraint.
            </p>
          </div>

          <div className="hero-badge">
            <span>ACTIVE MODEL</span>
            <strong>{profile?.model ?? "MLP"}</strong>
            <small>{profile?.architecture ?? "784 → 256 → 128 → 10"}</small>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <strong>Connection error</strong>
            <span>{error}</span>
          </div>
        )}

        <section className="metrics-grid">
          <MetricCard
            label="Test Accuracy"
            value={
              profile && baseline
                ? formatPercent(baseline.accuracy)
                : loading
                  ? "..."
                  : "—"
            }
            detail="Baseline model"
          />
          <MetricCard
            label="Parameters"
            value={
              profile ? formatNumber(profile.total_parameters) : "—"
            }
            detail="Trainable parameters"
          />
          <MetricCard
            label="Model Size"
            value={
              profile ? `${profile.model_size_mb.toFixed(2)} MB` : "—"
            }
            detail="Baseline checkpoint"
          />
          <MetricCard
            label="Inference"
            value={
              profile
                ? `${profile.inference_latency_ms.toFixed(3)} ms`
                : "—"
            }
            detail={profile?.device ?? "Runtime device"}
          />
        </section>

        <section className="content-grid">
          <div className="panel optimization-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">OPTIMIZATION</p>
                <h3>Configure pruning sweep</h3>
              </div>
              <span className="panel-tag">STRUCTURED</span>
            </div>

            <div className="form-group">
              <label htmlFor="pruning-levels">Pruning levels (%)</label>
              <input
                id="pruning-levels"
                value={pruningLevels}
                onChange={(event) => setPruningLevels(event.target.value)}
                placeholder="10, 20, 30, 40, 50"
              />
              <span className="input-hint">
                Comma-separated percentages to evaluate independently.
              </span>
            </div>

            <div className="form-group">
              <label htmlFor="accuracy-loss">
                Maximum accuracy loss (percentage points)
              </label>
              <input
                id="accuracy-loss"
                type="number"
                min="0"
                step="0.05"
                value={maxAccuracyLoss}
                onChange={(event) =>
                  setMaxAccuracyLoss(event.target.value)
                }
              />
              <span className="input-hint">
                Candidates beyond this threshold are rejected.
              </span>
            </div>

            <button
              className="optimize-button"
              onClick={runOptimization}
              disabled={optimizing || loading}
            >
              {optimizing ? (
                <>
                  <span className="spinner" />
                  Running optimization...
                </>
              ) : (
                <>
                  Run optimization
                  <span>→</span>
                </>
              )}
            </button>

            <p className="runtime-note">
              Runs the real PyTorch pruning and fine-tuning pipeline on the
              configured backend.
            </p>
          </div>

          <div className="panel best-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">SELECTED CANDIDATE</p>
                <h3>Best optimized model</h3>
              </div>
              {bestModel && <span className="best-check">BEST</span>}
            </div>

            {bestModel ? (
              <>
                <div className="architecture">
                  <span>784</span>
                  <i />
                  <span>{bestModel.hidden1_size}</span>
                  <i />
                  <span>{bestModel.hidden2_size}</span>
                  <i />
                  <span>10</span>
                </div>

                <div className="best-stat-grid">
                  <Stat
                    label="Pruning"
                    value={formatPercent(bestModel.pruning * 100)}
                  />
                  <Stat
                    label="Accuracy"
                    value={formatPercent(bestModel.accuracy)}
                  />
                  <Stat
                    label="Parameters"
                    value={formatNumber(bestModel.parameters)}
                  />
                  <Stat
                    label="Size"
                    value={`${bestModel.model_size_mb.toFixed(3)} MB`}
                  />
                </div>

                <div className="comparison">
                  <div>
                    <span>Parameter reduction</span>
                    <strong>{parameterReduction.toFixed(1)}%</strong>
                  </div>
                  <div>
                    <span>Size reduction</span>
                    <strong>{sizeReduction.toFixed(1)}%</strong>
                  </div>
                  <div>
                    <span>Accuracy change</span>
                    <strong
                      className={
                        bestModel.accuracy_change < 0
                          ? "negative"
                          : "positive"
                      }
                    >
                      {bestModel.accuracy_change >= 0 ? "+" : ""}
                      {bestModel.accuracy_change.toFixed(2)} pp
                    </strong>
                  </div>
                </div>

                <a
                  className="download-button"
                  href={`${API_BASE_URL}/model/optimized`}
                  download="optimized_mlp.pth"
                >
                  Download optimized model
                  <span>↓</span>
                </a>
              </>
            ) : (
              <div className="empty-state">
                <span className="empty-icon">◎</span>
                <p>Run an optimization sweep to select a model.</p>
              </div>
            )}
          </div>
        </section>

        <section className="panel results-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">EXPERIMENT RESULTS</p>
              <h3>Performance vs. complexity</h3>
            </div>
            <span className="result-count">
              {displayedResults.length} candidates
            </span>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Pruning</th>
                  <th>Architecture</th>
                  <th>Parameters</th>
                  <th>Model Size</th>
                  <th>Accuracy</th>
                  <th>Δ Accuracy</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {displayedResults.map((result) => {
                  const isBest =
                    bestModel &&
                    result.pruning === bestModel.pruning &&
                    result.parameters === bestModel.parameters;

                  return (
                    <tr key={`${result.pruning}-${result.parameters}`}>
                      <td>
                        <span className={isBest ? "best-row-label" : ""}>
                          {result.pruning === 0
                            ? "Baseline"
                            : formatPercent(result.pruning * 100)}
                        </span>
                      </td>
                      <td className="architecture-cell">
                        784 → {result.hidden1_size} →{" "}
                        {result.hidden2_size} → 10
                      </td>
                      <td>{formatNumber(result.parameters)}</td>
                      <td>{result.model_size_mb.toFixed(3)} MB</td>
                      <td>{formatPercent(result.accuracy)}</td>
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

        <section className="charts-grid">
          <ChartPanel
            title="Accuracy"
            subtitle="Test accuracy across pruning levels"
            results={displayedResults}
            valueKey="accuracy"
            unit="%"
          />
          <ChartPanel
            title="Parameters"
            subtitle="Trainable parameter count"
            results={displayedResults}
            valueKey="parameters"
            unit=""
          />
        </section>
      </main>

      <footer>
        <span>Neural Network Optimization Engine</span>
        <span>PyTorch · FastAPI · React + TypeScript</span>
      </footer>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChartPanel({
  title,
  subtitle,
  results,
  valueKey,
  unit,
}: {
  title: string;
  subtitle: string;
  results: ExperimentResult[];
  valueKey: "accuracy" | "parameters";
  unit: string;
}) {
  const values = results.map((result) => result[valueKey]);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  return (
    <div className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">TREND</p>
          <h3>{title}</h3>
          <span className="chart-subtitle">{subtitle}</span>
        </div>
      </div>

      <div className="chart">
        {results.map((result) => {
          const value = result[valueKey];
          const height =
            valueKey === "accuracy"
              ? 35 + ((value - min) / range) * 65
              : 35 + ((max - value) / range) * 65;

          return (
            <div
              className="chart-column"
              key={`${valueKey}-${result.pruning}`}
            >
              <span className="chart-value">
                {valueKey === "parameters"
                  ? `${(value / 1000).toFixed(0)}k`
                  : `${value.toFixed(2)}${unit}`}
              </span>
              <div className="bar-track">
                <div
                  className="bar"
                  style={{ height: `${height}%` }}
                />
              </div>
              <span className="chart-label">
                {result.pruning === 0
                  ? "Base"
                  : `${Math.round(result.pruning * 100)}%`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;