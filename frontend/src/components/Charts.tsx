import {
  accuracyLoss,
  clamp,
  formatNumber,
  formatPercent,
} from "../lib/format";

type ScatterCandidate = {
  pruning: number;
  parameters: number;
  accuracy: number;
  accuracy_change: number;
};

const WIDTH = 760;
const HEIGHT = 300;
const PAD = { top: 28, right: 24, bottom: 52, left: 66 };

type LinePoint = { x: number; y: number; label?: string };

export function LinePlot({
  series,
  xLabel,
  yLabel,
  formatX = (value) => `${value}`,
  formatY = (value) => `${value}`,
  height = HEIGHT,
  yPadding = 0.12,
  reference,
}: {
  series: Array<{
    label: string;
    color: string;
    points: LinePoint[];
    dashed?: boolean;
  }>;
  xLabel: string;
  yLabel: string;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  height?: number;
  yPadding?: number;
  reference?: { value: number; label: string; color?: string };
}) {
  const points = series.flatMap((item) => item.points);
  if (points.length === 0) {
    return <p className="chart-empty">No measurements available.</p>;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys, reference?.value ?? Infinity);
  const yMaxRaw = Math.max(...ys, reference?.value ?? -Infinity);
  const yRangeRaw = Math.max(yMaxRaw - yMinRaw, 0.01);
  const yMin = yMinRaw - yRangeRaw * yPadding;
  const yMax = yMaxRaw + yRangeRaw * yPadding;
  const xRange = Math.max(xMax - xMin, 0.0001);
  const yRange = Math.max(yMax - yMin, 0.0001);
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = height - PAD.top - PAD.bottom;

  const x = (value: number) =>
    PAD.left + ((value - xMin) / xRange) * plotWidth;
  const y = (value: number) =>
    PAD.top + plotHeight - ((value - yMin) / yRange) * plotHeight;

  const ticks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <div className="chart">
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
        {reference ? (
          <span>
            <i className="legend-threshold" />
            {reference.label}
          </span>
        ) : null}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${height}`}
        role="img"
        aria-label={`${yLabel} by ${xLabel}`}
        className="chart-svg"
      >
        <line
          x1={PAD.left}
          y1={PAD.top}
          x2={PAD.left}
          y2={PAD.top + plotHeight}
          className="chart-axis"
        />
        <line
          x1={PAD.left}
          y1={PAD.top + plotHeight}
          x2={PAD.left + plotWidth}
          y2={PAD.top + plotHeight}
          className="chart-axis"
        />
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              y1={y(tick)}
              x2={PAD.left + plotWidth}
              y2={y(tick)}
              className="chart-grid"
            />
            <text
              x={PAD.left - 10}
              y={y(tick) + 3}
              textAnchor="end"
              className="chart-tick"
            >
              {formatY(tick)}
            </text>
          </g>
        ))}
        {reference ? (
          <g>
            <line
              x1={PAD.left}
              y1={y(reference.value)}
              x2={PAD.left + plotWidth}
              y2={y(reference.value)}
              stroke={reference.color ?? "var(--muted)"}
              strokeDasharray="5 5"
              strokeWidth={1.5}
            />
            <text
              x={PAD.left + plotWidth}
              y={y(reference.value) - 7}
              textAnchor="end"
              className="chart-threshold-label"
            >
              {`${reference.label}: ${formatY(reference.value)}`}
            </text>
          </g>
        ) : null}
        {series.map((item) => (
          <g key={item.label}>
            <polyline
              points={item.points
                .map((point) => `${x(point.x)},${y(point.y)}`)
                .join(" ")}
              fill="none"
              stroke={item.color}
              strokeWidth={item.dashed ? 1.5 : 2}
              strokeDasharray={item.dashed ? "5 5" : undefined}
              className="chart-series"
            />
            {item.points.map((point) => (
              <g key={`${item.label}-${point.x}-${point.y}`}>
                <circle
                  cx={x(point.x)}
                  cy={y(point.y)}
                  r={4}
                  fill="var(--paper)"
                  stroke={item.color}
                  strokeWidth={2}
                >
                  <title>
                    {`${item.label}: ${formatY(point.y)} at ${formatX(point.x)}`}
                  </title>
                </circle>
              </g>
            ))}
          </g>
        ))}
        <text
          x={PAD.left + plotWidth / 2}
          y={height - 12}
          textAnchor="middle"
          className="chart-axis-label"
        >
          {xLabel}
        </text>
        <text
          x={PAD.left - 10}
          y={PAD.top - 8}
          textAnchor="start"
          className="chart-axis-label"
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

export function AccuracyScatter({
  results,
  baselineAccuracy,
  maxAccuracyLoss,
  selectedPruning,
}: {
  results: ScatterCandidate[];
  baselineAccuracy: number | null;
  maxAccuracyLoss: number;
  selectedPruning: number | null;
}) {
  if (results.length === 0) {
    return <p className="chart-empty">No candidate measurements available yet.</p>;
  }

  const threshold =
    baselineAccuracy === null ? null : baselineAccuracy - maxAccuracyLoss;
  const xs = results.map((result) => result.parameters);
  const ys = results.map((result) => result.accuracy);
  const allYs = threshold === null ? ys : [...ys, threshold];
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...allYs);
  const yMaxRaw = Math.max(...allYs);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.18, 0.08);
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const xRange = Math.max(xMax - xMin, 1);
  const yRange = Math.max(yMax - yMin, 0.01);
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const x = (value: number) =>
    PAD.left + ((value - xMin) / xRange) * plotWidth;
  const y = (value: number) =>
    PAD.top + plotHeight - ((value - yMin) / yRange) * plotHeight;

  return (
    <div className="chart">
      <div className="chart-legend">
        <span>
          <i className="legend-selected" /> selected
        </span>
        <span>
          <i className="legend-valid" /> valid
        </span>
        <span>
          <i className="legend-rejected" /> rejected
        </span>
        {threshold !== null ? (
          <span>
            <i className="legend-threshold" /> constraint
          </span>
        ) : null}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Test accuracy versus parameter count for each pruning candidate"
        className="chart-svg"
      >
        <line
          x1={PAD.left}
          y1={PAD.top}
          x2={PAD.left}
          y2={PAD.top + plotHeight}
          className="chart-axis"
        />
        <line
          x1={PAD.left}
          y1={PAD.top + plotHeight}
          x2={PAD.left + plotWidth}
          y2={PAD.top + plotHeight}
          className="chart-axis"
        />
        {threshold !== null ? (
          <g>
            <line
              x1={PAD.left}
              y1={y(threshold)}
              x2={PAD.left + plotWidth}
              y2={y(threshold)}
              className="chart-threshold"
            />
            <text
              x={PAD.left + plotWidth}
              y={y(threshold) - 8}
              textAnchor="end"
              className="chart-threshold-label"
            >
              {`minimum acceptable ${formatPercent(threshold)}`}
            </text>
          </g>
        ) : null}
        {results.map((result) => {
          const loss = accuracyLoss(result.accuracy_change);
          const selected = result.pruning === selectedPruning;
          const valid = loss <= maxAccuracyLoss;
          const className = selected
            ? "point-selected"
            : valid
              ? "point-valid"
              : "point-rejected";
          return (
            <g key={`${result.pruning}-${result.parameters}`}>
              <circle
                cx={x(result.parameters)}
                cy={y(result.accuracy)}
                r={selected ? 9 : 6}
                className={`scatter-point ${className}`}
              >
                <title>
                  {`${Math.round(result.pruning * 100)}% pruning · ${formatNumber(
                    result.parameters,
                  )} parameters · ${formatPercent(result.accuracy)} · loss ${loss.toFixed(
                    2,
                  )} pp`}
                </title>
              </circle>
              <text
                x={x(result.parameters)}
                y={y(result.accuracy) - (selected ? 16 : 13)}
                textAnchor="middle"
                className="scatter-label"
              >
                {`${Math.round(result.pruning * 100)}%`}
              </text>
            </g>
          );
        })}
        <text
          x={PAD.left + plotWidth / 2}
          y={HEIGHT - 12}
          textAnchor="middle"
          className="chart-axis-label"
        >
          Trainable parameters (smaller is better)
        </text>
        <text
          x={PAD.left - 10}
          y={PAD.top - 8}
          textAnchor="start"
          className="chart-axis-label"
        >
          Test accuracy (%)
        </text>
      </svg>
      <p className="chart-footnote">
        The dashed line is the lowest accuracy that satisfies the configured
        loss limit. The selector chooses the smallest valid candidate, not
        necessarily the most accurate one.
      </p>
    </div>
  );
}

export function ImportanceCurve({
  values,
  keepCount,
  title,
  color = "var(--teal)",
}: {
  values: number[];
  keepCount: number;
  title: string;
  color?: string;
}) {
  if (values.length === 0) {
    return <p className="chart-empty">Importance scores unavailable.</p>;
  }

  const sorted = [...values].sort((a, b) => b - a);
  const width = 460;
  const height = 220;
  const pad = { top: 20, right: 16, bottom: 36, left: 52 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const maxValue = Math.max(...sorted, 0.0001);
  const x = (index: number) =>
    pad.left + (index / Math.max(sorted.length - 1, 1)) * plotWidth;
  const y = (value: number) =>
    pad.top + plotHeight - (value / maxValue) * plotHeight;
  const cutoffX = x(clamp(keepCount - 1, 0, sorted.length - 1));

  const kept = sorted.slice(0, keepCount);
  const removed = sorted.slice(keepCount);

  return (
    <div className="importance-chart">
      <div className="importance-chart-head">
        <span>{title}</span>
        <span>
          keep {keepCount} / {sorted.length}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${title} sorted importance scores with a retention cutoff`}
      >
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={pad.top + plotHeight}
          className="chart-axis"
        />
        <line
          x1={pad.left}
          y1={pad.top + plotHeight}
          x2={pad.left + plotWidth}
          y2={pad.top + plotHeight}
          className="chart-axis"
        />
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            key={fraction}
            x1={pad.left}
            y1={pad.top + plotHeight * fraction}
            x2={pad.left + plotWidth}
            y2={pad.top + plotHeight * fraction}
            className="chart-grid"
          />
        ))}
        <polyline
          points={kept
            .map((value, index) => `${x(index)},${y(value)}`)
            .join(" ")}
          fill="none"
          stroke={color}
          strokeWidth={2}
        />
        <polyline
          points={removed
            .map((value, index) => `${x(keepCount + index)},${y(value)}`)
            .join(" ")}
          fill="none"
          stroke="var(--rust)"
          strokeWidth={2}
        />
        <line
          x1={cutoffX}
          y1={pad.top}
          x2={cutoffX}
          y2={pad.top + plotHeight}
          className="chart-threshold"
        />
        <text
          x={cutoffX + 6}
          y={pad.top + 12}
          className="chart-threshold-label"
        >
          cutoff
        </text>
        <text
          x={pad.left - 8}
          y={pad.top + 4}
          textAnchor="end"
          className="chart-tick"
        >
          {maxValue.toFixed(1)}
        </text>
        <text
          x={pad.left - 8}
          y={pad.top + plotHeight}
          textAnchor="end"
          className="chart-tick"
        >
          0
        </text>
        <text
          x={pad.left + plotWidth / 2}
          y={height - 8}
          textAnchor="middle"
          className="chart-axis-label"
        >
          neurons ranked by importance (highest first)
        </text>
      </svg>
      <div className="importance-legend">
        <span>
          <i style={{ background: color }} /> retained
        </span>
        <span>
          <i style={{ background: "var(--rust)" }} /> removed
        </span>
      </div>
    </div>
  );
}

export function ProbabilityBars({
  probabilities,
  prediction,
  label,
}: {
  probabilities: number[];
  prediction: number;
  label: string;
}) {
  return (
    <div className="probability-bars" aria-label={label}>
      {probabilities.map((probability, index) => (
        <div
          key={index}
          className={`probability-row ${
            index === prediction ? "is-predicted" : ""
          }`}
        >
          <span className="probability-class">{index}</span>
          <div className="probability-track">
            <div
              className="probability-fill"
              style={{ width: `${Math.max(probability * 100, 1)}%` }}
            />
          </div>
          <span className="probability-value">
            {(probability * 100).toFixed(2)}%
          </span>
        </div>
      ))}
    </div>
  );
}
