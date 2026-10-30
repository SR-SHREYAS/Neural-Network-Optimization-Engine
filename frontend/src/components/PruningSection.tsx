import type {
  CandidateResult,
  ModelAnalysis,
  PruningResponse,
} from "../types";
import { ImportanceCurve } from "./Charts";
import { LayerWidthComparison } from "../ModelExplorer";
import { Callout, DataNote, Formula } from "./Section";
import {
  accuracyLoss,
  architectureString,
  formatDecimal,
  formatNumber,
  formatPercent,
  formatSignedPP,
  parameterReduction,
} from "../lib/format";

function ReasoningRail() {
  const steps = [
    {
      label: "concept",
      title: "A hidden layer can be wider than it needs to be.",
      body: "Structured pruning removes complete hidden neurons. Unlike weight pruning, the matrix shapes really shrink, so the saved model has fewer rows and columns.",
    },
    {
      label: "hypothesis",
      title: "A narrower network may retain most of the accuracy.",
      body: "If many neurons are redundant, removing the least useful ones and retraining the survivors might preserve the decision boundary with fewer parameters.",
    },
    {
      label: "mechanism",
      title: "Score, rank, keep the top fraction, rebuild.",
      body: "Each hidden neuron receives an importance score from the L1 magnitude of its incoming weights. The highest-scoring neurons are kept; the next linear layers are rebuilt around the surviving indices.",
    },
    {
      label: "limitation",
      title: "Importance is a heuristic, not proof.",
      body: "A low weight score does not prove a neuron contributes nothing. The score ignores activations, data distribution, and interactions between neurons.",
    },
  ];

  return (
    <div className="reasoning-rail">
      {steps.map((step) => (
        <div className="reasoning-step" key={step.label}>
          <span>{step.label}</span>
          <strong>{step.title}</strong>
          <p>{step.body}</p>
        </div>
      ))}
    </div>
  );
}

function LayerExample({
  layer,
}: {
  layer: NonNullable<PruningResponse["layers"]>[number];
}) {
  const removed = layer.highest_removed;
  const retained = layer.lowest_retained;

  return (
    <div className="pruning-layer-card">
      <div className="pruning-layer-head">
        <span className="eyebrow">{layer.name.toUpperCase()}</span>
        <strong>
          {formatNumber(layer.original_size)} → {formatNumber(layer.keep_count)}
        </strong>
        <span>{formatNumber(layer.removed_count)} removed</span>
      </div>
      <div className="pruning-cutoff-grid">
        <div>
          <span className="cutoff-label">lowest retained</span>
          <strong>neuron {retained?.index ?? "—"}</strong>
          <span>I = {retained ? formatDecimal(retained.importance, 3) : "—"}</span>
          <span>bias = {retained?.bias === null || retained?.bias === undefined ? "—" : formatDecimal(retained.bias, 3)}</span>
        </div>
        <div>
          <span className="cutoff-label">highest removed</span>
          <strong>neuron {removed?.index ?? "—"}</strong>
          <span>I = {removed ? formatDecimal(removed.importance, 3) : "—"}</span>
          <span>bias = {removed?.bias === null || removed?.bias === undefined ? "—" : formatDecimal(removed.bias, 3)}</span>
        </div>
      </div>
      <p className="pruning-card-note">
        The cutoff is a hard boundary: every retained neuron scores at least as
        high as every removed neuron. When the two boundary values are close,
        they sit next to each other in the sorted ranking. The exact indices are
        listed below.
      </p>
      <details className="index-details">
        <summary>Inspect retained / removed indices</summary>
        <div className="index-lists">
          <div>
            <span className="cutoff-label">retained ({layer.kept_indices.length})</span>
            <p>{layer.kept_indices.join(", ")}</p>
          </div>
          <div>
            <span className="cutoff-label">removed ({layer.removed_indices.length})</span>
            <p>{layer.removed_indices.join(", ")}</p>
          </div>
        </div>
      </details>
    </div>
  );
}

export function PruningSection({
  analysis,
  pruningInfo,
  loading,
  level,
  availableLevels,
  onLevelChange,
  baselineParameters,
  results,
  maxAccuracyLoss,
}: {
  analysis: ModelAnalysis | null;
  pruningInfo: PruningResponse | null;
  loading: boolean;
  level: number | null;
  availableLevels: number[];
  onLevelChange: (level: number) => void;
  baselineParameters: number | null;
  results: CandidateResult[];
  maxAccuracyLoss: number;
}) {
  const activeLevel = level ?? analysis?.level ?? null;
  const activeLayers =
    analysis && analysis.level === activeLevel && analysis.layers.length > 0
      ? analysis.layers
      : pruningInfo?.layers ?? null;
  const activeArchitecture =
    analysis && analysis.level === activeLevel && analysis.pruned
      ? analysis.pruned.architecture
      : pruningInfo?.pruned_architecture ?? null;
  const activePrunedParameters =
    analysis && analysis.level === activeLevel && analysis.pruned
      ? analysis.pruned.parameters
      : pruningInfo?.pruned_parameters ?? null;
  const activeBeforeAccuracy =
    analysis && analysis.level === activeLevel && analysis.pruned
      ? analysis.pruned.before_finetuning_accuracy
      : null;
  const storedResult = results.find((result) => result.pruning === activeLevel) ?? null;
  const activeReduction = parameterReduction(
    baselineParameters,
    activePrunedParameters,
  );
  const storedLoss = storedResult ? accuracyLoss(storedResult.accuracy_change) : null;

  return (
    <div className="pruning-section">
      <ReasoningRail />

      <div className="level-selector" role="group" aria-label="Choose a structured pruning level">
        <span className="eyebrow">PRUNING LEVEL</span>
        <div className="level-buttons">
          {availableLevels.map((candidateLevel) => (
            <button
              key={candidateLevel}
              type="button"
              className={`level-button ${candidateLevel === activeLevel ? "is-active" : ""}`}
              onClick={() => onLevelChange(candidateLevel)}
              aria-pressed={candidateLevel === activeLevel}
            >
              {Math.round(candidateLevel * 100)}%
            </button>
          ))}
        </div>
        <span className="level-selector-note">
          {loading ? "loading exact indices…" : "indices recomputed from the baseline checkpoint"}
        </span>
      </div>

      <div className="pruning-grid">
        <div className="pruning-visual">
          <div className="pruning-visual-head">
            <span className="eyebrow">THE WIDTH CHANGE</span>
            <strong>
              {activeArchitecture
                ? architectureString(activeArchitecture)
                : "Choose a level to inspect"}
            </strong>
          </div>
          {activeArchitecture ? (
            <LayerWidthComparison
              baseline={
                analysis?.baseline.architecture ??
                pruningInfo?.baseline_architecture ??
                [784, 256, 128, 10]
              }
              selected={activeArchitecture}
              level={activeLevel ?? 0}
            />
          ) : (
            <p className="chart-empty">Exact widths will appear after the API responds.</p>
          )}
          <div className="pruning-measures">
            <div>
              <span>parameters</span>
              <strong>{formatNumber(activePrunedParameters)}</strong>
            </div>
            <div>
              <span>reduction</span>
              <strong>
                {activeReduction === null ? "—" : `−${activeReduction.toFixed(1)}%`}
              </strong>
            </div>
            <div>
              <span>before fine-tuning</span>
              <strong>{formatPercent(activeBeforeAccuracy)}</strong>
            </div>
            <div>
              <span>after fine-tuning</span>
              <strong>{storedResult ? formatPercent(storedResult.accuracy) : "—"}</strong>
            </div>
          </div>
          {activeBeforeAccuracy === null ? (
            <DataNote tone="warning">
              The before-fine-tuning accuracy for this level is not in the
              stored CSV. Run the analysis for the selected level, or run a new
              optimization to capture every candidate's intermediate value.
            </DataNote>
          ) : null}
        </div>

        <div className="pruning-explainer">
          <Formula
            expression="Iⱼ = Σᵢ |Wⱼᵢ|"
            explanation="Importance of hidden neuron j: sum the absolute incoming weights from the previous layer. The implementation calls weight.abs().sum(dim=1)."
          />
          <Formula
            expression="keep = round(n × (1 − p))"
            explanation="The same fraction p is applied independently to both hidden layers. At least one neuron is retained."
          />
          <Callout title="What the score does not know" tone="rust">
            It does not observe activations, the input distribution, or
            interactions between neurons. It is a fast structural heuristic, not
            a causal attribution method.
          </Callout>
        </div>
      </div>

      {activeLayers ? (
        <>
          <div className="importance-grid">
            {activeLayers.map((layer) => (
              <ImportanceCurve
                key={layer.name}
                values={layer.importance}
                keepCount={layer.keep_count}
                title={`${layer.name} importance · sorted`}
                color={layer.name === "Hidden 1" ? "var(--teal)" : "var(--green)"}
              />
            ))}
          </div>
          <div className="pruning-layer-grid">
            {activeLayers.map((layer) => (
              <LayerExample key={layer.name} layer={layer} />
            ))}
          </div>
        </>
      ) : (
        <div className="loading-block">
          <span>Computing exact importance scores and removal indices…</span>
        </div>
      )}

      <div className="pruning-observation">
        <div>
          <span className="eyebrow">OBSERVATION AT THIS LEVEL</span>
          <p>
            {storedResult
              ? `The stored run measured ${formatPercent(
                  storedResult.accuracy,
                )} after fine-tuning, with ${formatNumber(
                  storedResult.parameters,
                )} parameters and an accuracy change of ${formatSignedPP(
                  storedResult.accuracy_change,
                )}.`
              : "No post-fine-tuning measurement is stored for this level in the current experiment table."}
          </p>
        </div>
        <div>
          <span className="eyebrow">INTERPRETATION</span>
          <p>
            {storedLoss === null
              ? "The accuracy-loss constraint can only be applied once a post-fine-tuning test accuracy exists."
              : storedLoss <= maxAccuracyLoss
                ? `The measured loss of ${storedLoss.toFixed(
                    2,
                  )} pp is within the ${maxAccuracyLoss.toFixed(
                    2,
                  )} pp limit, so this candidate is eligible.`
                : `The measured loss of ${storedLoss.toFixed(
                    2,
                  )} pp exceeds the ${maxAccuracyLoss.toFixed(
                    2,
                  )} pp limit, so this candidate is rejected even if it is smaller.`}
          </p>
        </div>
      </div>
      <DataNote>
        The importance arrays, retained indices, and removed indices above come
        from the current baseline checkpoint. They are not illustrative
        placeholders. The before-fine-tuning accuracy is a separate measured
        evaluation and is only shown when it is available.
      </DataNote>
    </div>
  );
}
