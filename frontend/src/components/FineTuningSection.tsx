import type { CandidateResult, FineTuningEpoch } from "../types";
import { LinePlot } from "./Charts";
import { Callout, DataNote, Formula } from "./Section";
import {
  formatDecimal,
  formatNumber,
  formatPercent,
  formatSignedPP,
} from "../lib/format";

function RecoveryMeter({
  baselineAccuracy,
  beforeAccuracy,
  afterAccuracy,
}: {
  baselineAccuracy: number | null;
  beforeAccuracy: number | null;
  afterAccuracy: number | null;
}) {
  const min = Math.min(
    baselineAccuracy ?? 0,
    beforeAccuracy ?? 0,
    afterAccuracy ?? 0,
  );
  const max = Math.max(
    baselineAccuracy ?? 100,
    beforeAccuracy ?? 100,
    afterAccuracy ?? 100,
  );
  const span = Math.max(max - min, 0.5);
  const position = (value: number | null) =>
    value === null ? 0 : ((value - min) / span) * 100;

  return (
    <div className="recovery-meter">
      <div className="recovery-track">
        <div
          className="recovery-range"
          style={{
            left: `${position(Math.min(beforeAccuracy ?? min, afterAccuracy ?? min))}%`,
            width: `${Math.max(
              position(Math.max(beforeAccuracy ?? min, afterAccuracy ?? min)) -
                position(Math.min(beforeAccuracy ?? min, afterAccuracy ?? min)),
              1,
            )}%`,
          }}
        />
        {baselineAccuracy !== null ? (
          <span
            className="recovery-marker marker-baseline"
            style={{ left: `${position(baselineAccuracy)}%` }}
            title={`baseline ${formatPercent(baselineAccuracy)}`}
          >
            <i />
            <small>baseline</small>
          </span>
        ) : null}
        {beforeAccuracy !== null ? (
          <span
            className="recovery-marker marker-before"
            style={{ left: `${position(beforeAccuracy)}%` }}
            title={`before fine-tuning ${formatPercent(beforeAccuracy)}`}
          >
            <i />
            <small>before</small>
          </span>
        ) : null}
        {afterAccuracy !== null ? (
          <span
            className="recovery-marker marker-after"
            style={{ left: `${position(afterAccuracy)}%` }}
            title={`after fine-tuning ${formatPercent(afterAccuracy)}`}
          >
            <i />
            <small>after</small>
          </span>
        ) : null}
      </div>
      <div className="recovery-values">
        <div>
          <span>baseline</span>
          <strong>{formatPercent(baselineAccuracy)}</strong>
        </div>
        <div>
          <span>immediately after pruning</span>
          <strong>{formatPercent(beforeAccuracy)}</strong>
        </div>
        <div>
          <span>after fine-tuning</span>
          <strong>{formatPercent(afterAccuracy)}</strong>
        </div>
      </div>
    </div>
  );
}

function TrainingHistory({
  history,
}: {
  history: FineTuningEpoch[] | undefined;
}) {
  if (!history || history.length === 0) {
    return (
      <DataNote tone="warning">
        Per-epoch fine-tuning history is captured for new optimization runs but
        is not stored in the existing CSV. Run the one-click optimization to
        record the training loss and training accuracy for each epoch.
      </DataNote>
    );
  }

  return (
    <div className="training-history">
      <LinePlot
        series={[
          {
            label: "training loss",
            color: "var(--rust)",
            points: history.map((epoch) => ({
              x: epoch.epoch,
              y: epoch.loss,
            })),
          },
        ]}
        xLabel="fine-tuning epoch"
        yLabel="cross-entropy loss"
        formatX={(value) => `epoch ${value}`}
        formatY={(value) => formatDecimal(value, 4)}
        height={220}
      />
      <LinePlot
        series={[
          {
            label: "training accuracy",
            color: "var(--green)",
            points: history.map((epoch) => ({
              x: epoch.epoch,
              y: epoch.accuracy,
            })),
          },
        ]}
        xLabel="fine-tuning epoch"
        yLabel="training accuracy (%)"
        formatX={(value) => `epoch ${value}`}
        formatY={(value) => formatPercent(value, 1)}
        height={220}
      />
    </div>
  );
}

export function FineTuningSection({
  selected,
  baselineAccuracy,
  selectedBeforeAccuracy,
  results,
}: {
  selected: CandidateResult | null;
  baselineAccuracy: number | null;
  selectedBeforeAccuracy: number | null;
  results: CandidateResult[];
}) {
  const beforePoints = results
    .filter((result) => result.before_finetuning_accuracy !== undefined)
    .map((result) => ({
      x: result.pruning,
      y: result.before_finetuning_accuracy as number,
    }));
  const afterPoints = results.map((result) => ({
    x: result.pruning,
    y: result.accuracy,
  }));
  const hasBeforeSeries = beforePoints.length > 1;
  const selectedHistory = selected?.fine_tuning_history;

  return (
    <div className="fine-tuning-section">
      <div className="fine-tuning-grid">
        <div className="fine-tuning-copy">
          <span className="eyebrow">WHY RETRAIN AFTER REMOVING NEURONS?</span>
          <h3>Pruning changes the function. Fine-tuning lets the survivors adapt.</h3>
          <p>
            Removing a neuron deletes its outgoing weights and the matching
            incoming weights in the next layer. The remaining weights still
            point in roughly the right direction, but the network has lost
            capacity and its decision boundary has shifted. Fine-tuning performs
            additional gradient steps on the training split to recover useful
            performance.
          </p>
          <p>
            The recovery is measured, not assumed. The model is evaluated on the
            held-out test set immediately after pruning and again after
            fine-tuning. Those are different measurements from the training
            accuracy reported during each epoch.
          </p>
          <Formula
            expression="L = −Σ y log(softmax(z))"
            explanation="Cross-entropy loss over the 10 MNIST classes."
          />
        </div>
        <div className="fine-tuning-config">
          <span className="eyebrow">VERIFIED CONFIGURATION</span>
          <dl>
            <div>
              <dt>epochs</dt>
              <dd>2</dd>
            </div>
            <div>
              <dt>optimizer</dt>
              <dd>Adam</dd>
            </div>
            <div>
              <dt>learning rate</dt>
              <dd>1 × 10⁻⁴</dd>
            </div>
            <div>
              <dt>batch size</dt>
              <dd>128</dd>
            </div>
            <div>
              <dt>training split</dt>
              <dd>60,000 images</dd>
            </div>
            <div>
              <dt>test split</dt>
              <dd>10,000 images</dd>
            </div>
          </dl>
          <Callout title="Training accuracy is not test accuracy" tone="amber">
            The progress callback reports accuracy on the shuffled training
            loader. The final candidate accuracy is measured on the held-out
            test split. The two numbers answer different questions.
          </Callout>
        </div>
      </div>

      <div className="recovery-panel">
        <div className="recovery-panel-head">
          <span className="eyebrow">SELECTED CANDIDATE RECOVERY</span>
          <strong>
            {selected
              ? `${Math.round(selected.pruning * 100)}% pruning · ${formatNumber(
                  selected.parameters,
                )} parameters`
              : "No selected candidate yet"}
          </strong>
        </div>
        <RecoveryMeter
          baselineAccuracy={baselineAccuracy}
          beforeAccuracy={selectedBeforeAccuracy}
          afterAccuracy={selected?.accuracy ?? null}
        />
        {selectedBeforeAccuracy === null ? (
          <DataNote tone="warning">
            The before-fine-tuning accuracy for the selected candidate is not
            available from the current data source. Run a new optimization, or
            load the analysis for the selected level, to measure it.
          </DataNote>
        ) : null}
      </div>

      <div className="recovery-chart-panel">
        <div className="recovery-chart-head">
          <span className="eyebrow">BEFORE AND AFTER FINE-TUNING</span>
          <span>
            {hasBeforeSeries
              ? "every candidate has a measured intermediate accuracy"
              : "the stored CSV contains post-fine-tuning accuracy; intermediate values require a new run"}
          </span>
        </div>
        {hasBeforeSeries ? (
          <LinePlot
            series={[
              {
                label: "immediately after pruning",
                color: "var(--rust)",
                points: beforePoints,
              },
              {
                label: "after fine-tuning",
                color: "var(--green)",
                points: afterPoints,
              },
            ]}
            xLabel="pruning level"
            yLabel="test accuracy (%)"
            formatX={(value) => `${Math.round(value * 100)}%`}
            formatY={(value) => formatPercent(value, 1)}
          />
        ) : (
          <LinePlot
            series={[
              {
                label: "after fine-tuning",
                color: "var(--green)",
                points: afterPoints,
              },
            ]}
            xLabel="pruning level"
            yLabel="test accuracy (%)"
            formatX={(value) => `${Math.round(value * 100)}%`}
            formatY={(value) => formatPercent(value, 1)}
          />
        )}
        {selected ? (
          <p className="chart-footnote">
            Selected candidate:{" "}
            {selectedBeforeAccuracy === null
              ? "intermediate accuracy unavailable"
              : `${formatPercent(selectedBeforeAccuracy)} before → ${formatPercent(
                  selected.accuracy,
                )} after`}
            {selectedBeforeAccuracy === null
              ? ""
              : ` (${formatSignedPP(selected.accuracy - selectedBeforeAccuracy)})`}
            .
          </p>
        ) : null}
      </div>

      <div className="training-history-panel">
        <div className="recovery-chart-head">
          <span className="eyebrow">TRAINING PROGRESS</span>
          <span>
            {selectedHistory && selectedHistory.length > 0
              ? "captured from the current optimization run"
              : "not present in the stored experiment"}
          </span>
        </div>
        <TrainingHistory history={selectedHistory} />
      </div>
    </div>
  );
}
