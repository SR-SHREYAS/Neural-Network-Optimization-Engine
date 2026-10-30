import type { SamplePrediction } from "../types";
import { ProbabilityBars } from "./Charts";
import {
  formatDecimal,
  formatNumber,
  formatPercent,
} from "../lib/format";

function PixelGrid({ sample }: { sample: SamplePrediction }) {
  return (
    <div className="pixel-frame">
      <div
        className="pixel-grid"
        role="img"
        aria-label={`28 by 28 pixel MNIST sample number ${sample.index}, true label ${sample.label}`}
      >
        {sample.pixels.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => (
            <span
              key={`${rowIndex}-${columnIndex}`}
              className="pixel-cell"
              style={{
                backgroundColor: `rgba(24, 24, 24, ${0.04 + (value / 255) * 0.96})`,
              }}
              title={`row ${rowIndex}, col ${columnIndex}, value ${value}`}
            />
          )),
        )}
      </div>
      <div className="pixel-caption">
        <span>28 × 28 pixels</span>
        <span>true label</span>
        <strong>{sample.label}</strong>
      </div>
    </div>
  );
}

function PredictionFlow({
  sample,
  model,
}: {
  sample: SamplePrediction;
  model: "baseline" | "optimized";
}) {
  const prediction = sample[model];
  if (!prediction) {
    return (
      <div className="prediction-flow prediction-flow-empty">
        <span>Selected checkpoint is not available yet.</span>
      </div>
    );
  }

  const stats = prediction.activation_stats;
  const stages = [
    { label: "input", value: 784, note: "flattened pixels" },
    { label: "hidden 1", value: stats.hidden1.size, note: "ReLU units" },
    { label: "hidden 2", value: stats.hidden2.size, note: "ReLU units" },
    { label: "output", value: 10, note: "class scores" },
  ];

  return (
    <div className={`prediction-flow prediction-${model}`}>
      <div className="prediction-flow-head">
        <span className="eyebrow">{model === "baseline" ? "BASELINE" : "SAVED OPTIMIZED"}</span>
        <span className="mono">
          {prediction.prediction === sample.label ? "correct" : "incorrect"}
        </span>
      </div>
      <div className="prediction-stages">
        {stages.map((stage, index) => (
          <div className="prediction-stage" key={stage.label}>
            <span>{stage.label}</span>
            <strong>{formatNumber(stage.value)}</strong>
            <small>{stage.note}</small>
            {index < stages.length - 1 ? <i aria-hidden="true">→</i> : null}
          </div>
        ))}
      </div>
      <div className="prediction-answer">
        <div>
          <span className="eyebrow">ARGMAX</span>
          <strong className="prediction-digit">{prediction.prediction}</strong>
        </div>
        <div>
          <span>confidence</span>
          <strong>{formatPercent(prediction.confidence * 100)}</strong>
        </div>
        <div>
          <span>hidden-1 active</span>
          <strong>{formatPercent((1 - stats.hidden1.fraction_zero) * 100, 1)}</strong>
        </div>
      </div>
      <ProbabilityBars
        probabilities={prediction.probabilities}
        prediction={prediction.prediction}
        label={`${model} output probabilities`}
      />
    </div>
  );
}

export function DigitSection({
  sample,
  loading,
  error,
  onPrevious,
  onNext,
  onRandom,
}: {
  sample: SamplePrediction | null;
  loading: boolean;
  error: string | null;
  onPrevious: () => void;
  onNext: () => void;
  onRandom: () => void;
}) {
  return (
    <div className="digit-section">
      <div className="digit-controls">
        <div className="sample-index">
          <span className="eyebrow">TEST SAMPLE</span>
          <strong>{sample ? `#${String(sample.index).padStart(4, "0")}` : "#----"}</strong>
        </div>
        <div className="sample-buttons">
          <button type="button" className="button button-small" onClick={onPrevious} disabled={loading}>
            ← Previous
          </button>
          <button type="button" className="button button-small" onClick={onRandom} disabled={loading}>
            Random
          </button>
          <button type="button" className="button button-small" onClick={onNext} disabled={loading}>
            Next →
          </button>
        </div>
        <span className="sample-source">
          {loading ? "loading a real MNIST test sample…" : "source: MNIST test split"}
        </span>
      </div>

      {error ? <p className="data-note data-note-warning">{error}</p> : null}

      {sample ? (
        <>
          <div className="digit-main">
            <PixelGrid sample={sample} />
            <div className="digit-story">
              <div className="digit-story-step">
                <span className="story-index">01</span>
                <div>
                  <strong>The image is a 28 × 28 grid.</strong>
                  <p>
                    Each cell is a grayscale pixel. Darker cells are larger
                    values. The network does not receive a picture; it receives
                    784 numbers.
                  </p>
                </div>
              </div>
              <div className="digit-story-step">
                <span className="story-index">02</span>
                <div>
                  <strong>Flattening preserves the values.</strong>
                  <p>
                    <code>x.view(batch, -1)</code> turns the 28 × 28 tensor into
                    a 784-dimensional vector before the first linear layer.
                  </p>
                </div>
              </div>
              <div className="digit-story-step">
                <span className="story-index">03</span>
                <div>
                  <strong>The network produces 10 scores.</strong>
                  <p>
                    The output with the largest raw logit is the prediction.
                    Softmax probabilities are shown for interpretation, but the
                    inference code uses <code>argmax</code> over logits.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="prediction-grid">
            <PredictionFlow sample={sample} model="baseline" />
            <PredictionFlow sample={sample} model="optimized" />
          </div>
          <p className="data-note">{sample.note}</p>
        </>
      ) : (
        <div className="loading-block">
          <span>Loading a real test digit and both checkpoint predictions…</span>
        </div>
      )}
    </div>
  );
}

export function NeuronSection({ sample }: { sample: SamplePrediction | null }) {
  if (!sample) {
    return (
      <div className="loading-block">
        <span>The neuron worked example appears when a test sample is loaded.</span>
      </div>
    );
  }

  const example = sample.neuron_example;
  const maxContribution = Math.max(
    0.000001,
    ...example.top_contributions.map((item) => Math.abs(item.contribution)),
  );

  return (
    <div className="neuron-section">
      <div className="neuron-equation">
        <div className="neuron-equation-head">
          <span className="eyebrow">ONE HIDDEN NEURON</span>
          <strong>
            Hidden 1 · neuron {example.neuron_index} of {formatNumber(example.layer_size)}
          </strong>
        </div>
        <div className="equation-block">
          <code>z = b + Σᵢ xᵢ · wᵢ</code>
          <code>a = max(0, z)</code>
        </div>
        <div className="neuron-values">
          <div>
            <span>bias b</span>
            <strong>{formatDecimal(example.bias, 4)}</strong>
          </div>
          <div>
            <span>pre-activation z</span>
            <strong>{formatDecimal(example.pre_activation, 4)}</strong>
          </div>
          <div>
            <span>ReLU output a</span>
            <strong>{formatDecimal(example.activation, 4)}</strong>
          </div>
        </div>
        <p>
          This is the hidden-1 neuron with the largest activation for the
          current digit. Its full weighted sum uses all 784 inputs; the table
          below shows only the twelve largest absolute contributions.
        </p>
      </div>

      <div className="contribution-table" role="table" aria-label="Largest contributions to one hidden neuron">
        <div className="contribution-head" role="row">
          <span role="columnheader">pixel</span>
          <span role="columnheader">xᵢ</span>
          <span role="columnheader">wᵢ</span>
          <span role="columnheader">xᵢ · wᵢ</span>
          <span role="columnheader">contribution</span>
        </div>
        {example.top_contributions.map((item) => (
          <div className="contribution-row" role="row" key={item.pixel_index}>
            <span className="contribution-pixel" role="cell">
              <i
                style={{
                  backgroundColor: `rgba(24, 24, 24, ${0.06 + (item.pixel_value / 1) * 0.9})`,
                }}
              />
              {`${item.row},${item.col}`}
            </span>
            <span className="mono" role="cell">{item.pixel_value.toFixed(3)}</span>
            <span className="mono" role="cell">{item.weight.toFixed(4)}</span>
            <span
              className={`mono ${item.contribution >= 0 ? "positive" : "negative"}`}
              role="cell"
            >
              {item.contribution >= 0 ? "+" : ""}
              {item.contribution.toFixed(4)}
            </span>
            <span className="contribution-bar-cell" role="cell">
              <i
                className={item.contribution >= 0 ? "bar-positive" : "bar-negative"}
                style={{
                  width: `${(Math.abs(item.contribution) / maxContribution) * 100}%`,
                }}
              />
            </span>
          </div>
        ))}
      </div>
      <p className="data-note">{example.note}</p>
    </div>
  );
}
