import { useMemo, useState } from "react";
import type {
  PruningLayerInfo,
  SamplePrediction,
  WeightModelSnapshot,
} from "./types";
import {
  architectureString,
  formatNumber,
  formatPercent,
  formatSignedPP,
  parameterReduction,
} from "./lib/format";

type Inspection = {
  kind: "node" | "edge";
  label: string;
  detail: string;
};

const layerNames = ["Input", "Hidden 1", "Hidden 2", "Output"];

function gridColumns(layerIndex: number, size: number): number {
  if (layerIndex === 0) return 28;
  if (layerIndex === 3) return 5;
  return size > 128 ? 16 : 16;
}

function DotLayer({
  layerIndex,
  layer,
  pixels,
  removedIndices,
  onInspect,
}: {
  layerIndex: number;
  layer: WeightModelSnapshot["layers"][number];
  pixels?: number[][];
  removedIndices?: Set<number>;
  onInspect?: (inspection: Inspection) => void;
}) {
  const columns = gridColumns(layerIndex, layer.size);
  const importance = layer.importance;
  const maxImportance = importance ? Math.max(...importance, 0.0001) : 1;
  const isInput = layerIndex === 0;
  const isOutput = layerIndex === 3;

  return (
    <div className={`layer-column layer-${layerIndex}`}>
      <div className="layer-column-head">
        <span className="layer-name">{layerNames[layerIndex]}</span>
        <span className="layer-size">
          {formatNumber(layer.size)} {isInput ? "features" : isOutput ? "classes" : "neurons"}
        </span>
      </div>
      <div
        className={`dot-grid ${isInput ? "dot-grid-input" : ""}`}
        style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
        aria-hidden="true"
      >
        {Array.from({ length: layer.size }, (_, index) => {
          const removed = removedIndices?.has(index) ?? false;
          const pixelValue = isInput
            ? (pixels?.[Math.floor(index / 28)]?.[index % 28] ?? 0) / 255
            : 0;
          const score = importance?.[index] ?? 0;
          const strength = importance ? Math.max(0.18, score / maxImportance) : 0.8;
          const style = isInput
            ? {
                backgroundColor: `rgba(24, 24, 24, ${0.06 + pixelValue * 0.9})`,
              }
            : isOutput
              ? { backgroundColor: "rgba(79, 122, 90, 0.85)" }
              : removed
                ? { backgroundColor: "rgba(180, 101, 63, 0.82)" }
                : {
                    backgroundColor: `rgba(47, 93, 98, ${0.22 + strength * 0.72})`,
                  };
          return (
            <span
              key={index}
              className={`dot ${removed ? "dot-removed" : ""}`}
              style={style}
              title={
                isInput
                  ? `Input feature ${index} · pixel value ${(pixelValue * 255).toFixed(0)}`
                  : `${layerNames[layerIndex]} neuron ${index} · importance ${score.toFixed(3)}${removed ? " · removed" : ""}`
              }
            />
          );
        })}
      </div>
      {onInspect && layer.sampled_indices.length > 0 ? (
        <div className="sample-node-row">
          <span className="sample-node-label">sample</span>
          {layer.sampled_indices.map((index, sampleIndex) => {
            const bias = layer.sampled_biases[sampleIndex];
            const score = layer.importance?.[index];
            return (
              <button
                type="button"
                key={index}
                className="sample-node"
                onMouseEnter={() =>
                  onInspect({
                    kind: "node",
                    label: `${layerNames[layerIndex]} · neuron ${index}`,
                    detail: [
                      bias === undefined ? null : `bias ${bias.toFixed(4)}`,
                      score === undefined ? null : `importance ${score.toFixed(3)}`,
                      `layer size ${formatNumber(layer.size)}`,
                    ]
                      .filter(Boolean)
                      .join(" · "),
                  })
                }
                onFocus={() =>
                  onInspect({
                    kind: "node",
                    label: `${layerNames[layerIndex]} · neuron ${index}`,
                    detail: [
                      bias === undefined ? null : `bias ${bias.toFixed(4)}`,
                      score === undefined ? null : `importance ${score.toFixed(3)}`,
                    ]
                      .filter(Boolean)
                      .join(" · "),
                  })
                }
              >
                {index}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ConnectionBundle({
  sourceLayer,
  connections,
  onInspect,
}: {
  sourceLayer: number;
  connections: WeightModelSnapshot["connections"];
  onInspect?: (inspection: Inspection) => void;
}) {
  const bundle = connections.find((item) => item.source_layer === sourceLayer);
  if (!bundle || bundle.weights.length === 0) {
    return <div className="connection-bundle connection-empty" aria-hidden="true" />;
  }

  const maxWeight = Math.max(
    0.000001,
    ...bundle.weights.map((item) => Math.abs(item.weight)),
  );
  const sourceCount = Math.max(
    ...bundle.weights.map((item) => item.source_index),
    0,
  ) + 1;
  const targetCount = Math.max(
    ...bundle.weights.map((item) => item.target_index),
    0,
  ) + 1;
  const yFor = (index: number, count: number) =>
    28 + ((index + 0.5) / Math.max(count, 1)) * 204;

  return (
    <div className="connection-bundle">
      <svg viewBox="0 0 86 260" role="img" aria-label={`Sampled connections from layer ${sourceLayer + 1} to layer ${sourceLayer + 2}`}>
        {bundle.weights.map((edge) => {
          const magnitude = Math.abs(edge.weight);
          const positive = edge.weight >= 0;
          return (
            <line
              key={`${edge.source_index}-${edge.target_index}`}
              x1="4"
              y1={yFor(edge.source_index, sourceCount)}
              x2="82"
              y2={yFor(edge.target_index, targetCount)}
              stroke={positive ? "var(--teal)" : "var(--rust)"}
              strokeWidth={0.5 + (magnitude / maxWeight) * 1.7}
              opacity={0.2 + (magnitude / maxWeight) * 0.55}
              tabIndex={0}
              onMouseEnter={() =>
                onInspect?.({
                  kind: "edge",
                  label: `${layerNames[sourceLayer]} ${edge.source_index} → ${layerNames[sourceLayer + 1]} ${edge.target_index}`,
                  detail: `actual sampled weight ${edge.weight.toFixed(5)}`,
                })
              }
              onFocus={() =>
                onInspect?.({
                  kind: "edge",
                  label: `${layerNames[sourceLayer]} ${edge.source_index} → ${layerNames[sourceLayer + 1]} ${edge.target_index}`,
                  detail: `actual sampled weight ${edge.weight.toFixed(5)}`,
                })
              }
            >
              <title>{`weight ${edge.weight.toFixed(5)}`}</title>
            </line>
          );
        })}
      </svg>
      <span>5×5 sample</span>
    </div>
  );
}

function NetworkSnapshot({
  title,
  snapshot,
  pixels,
  removedIndices,
  accuracy,
  parameters,
  tone,
}: {
  title: string;
  snapshot: WeightModelSnapshot | null;
  pixels?: number[][];
  removedIndices?: Record<string, Set<number>>;
  accuracy: number | null;
  parameters: number | null;
  tone: "baseline" | "selected";
}) {
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const layers = snapshot?.layers ?? [];

  return (
    <div className={`network-panel network-panel-${tone}`}>
      <div className="network-panel-head">
        <div>
          <span className={`eyebrow ${tone === "baseline" ? "eyebrow-teal" : "eyebrow-green"}`}>
            {title}
          </span>
          <strong>
            {snapshot ? architectureString(snapshot.architecture) : "Checkpoint loading…"}
          </strong>
        </div>
        <div className="network-panel-stats">
          <span>{parameters === null ? "—" : `${formatNumber(parameters)} params`}</span>
          <span>{accuracy === null ? "accuracy —" : `${formatPercent(accuracy)} test accuracy`}</span>
        </div>
      </div>
      {snapshot ? (
        <div className="network-stage">
          {layers.map((layer, index) => (
            <div className="network-stage-group" key={layer.name}>
              <DotLayer
                layerIndex={index}
                layer={layer}
                pixels={pixels}
                removedIndices={
                  index === 1
                    ? removedIndices?.["Hidden 1"]
                    : index === 2
                      ? removedIndices?.["Hidden 2"]
                      : undefined
                }
                onInspect={setInspection}
              />
              {index < layers.length - 1 ? (
                <ConnectionBundle
                  sourceLayer={index}
                  connections={snapshot.connections}
                  onInspect={setInspection}
                />
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="network-placeholder">
          <span>Waiting for the checkpoint summary.</span>
          <span>The dot matrix and sampled weights appear when the API responds.</span>
        </div>
      )}
      <div className="network-legend">
        <span>
          <i className="legend-dot-teal" /> retained / low→high importance
        </span>
        <span>
          <i className="legend-dot-rust" /> removed neuron
        </span>
        <span>
          <i className="legend-line-teal" /> positive sampled weight
        </span>
        <span>
          <i className="legend-line-rust" /> negative sampled weight
        </span>
      </div>
      <div className="network-inspector" aria-live="polite">
        <span className="eyebrow">INSPECT</span>
        {inspection ? (
          <>
            <strong>{inspection.label}</strong>
            <span>{inspection.detail}</span>
          </>
        ) : (
          <span>
            Hover or focus a sampled node or connection to inspect an actual
            checkpoint value. Each dot is one real unit; the lines are a 5×5
            sample, not the full dense matrix.
          </span>
        )}
      </div>
    </div>
  );
}

export function ModelExplorer({
  baseline,
  optimized,
  baselineParameters,
  optimizedParameters,
  baselineAccuracy,
  optimizedAccuracy,
  sample,
  pruning,
}: {
  baseline: WeightModelSnapshot | null;
  optimized: WeightModelSnapshot | null;
  baselineParameters: number | null;
  optimizedParameters: number | null;
  baselineAccuracy: number | null;
  optimizedAccuracy: number | null;
  sample: SamplePrediction | null;
  pruning: PruningLayerInfo[] | null;
}) {
  const removedIndices = useMemo(() => {
    if (!pruning) return undefined;
    const map: Record<string, Set<number>> = {};
    for (const layer of pruning) {
      map[layer.name] = new Set(layer.removed_indices);
    }
    return map;
  }, [pruning]);

  const reduction = parameterReduction(baselineParameters, optimizedParameters);
  const hidden1 = baseline?.architecture[1] ?? 256;
  const hidden2 = baseline?.architecture[2] ?? 128;
  const selected1 = optimized?.architecture[1] ?? null;
  const selected2 = optimized?.architecture[2] ?? null;

  return (
    <div className="model-explorer">
      <div className="explorer-intro">
        <div>
          <span className="eyebrow">THE ARCHITECTURE, AT CHECKPOINT SCALE</span>
          <h3>Same task. A visibly narrower network.</h3>
        </div>
        <p>
          Every dot below represents one actual input feature, hidden neuron, or
          output class. The selected model is not a redrawn approximation: its
          dot counts come from the saved checkpoint. Removed baseline neurons
          are marked in rust at the currently inspected pruning level.
        </p>
      </div>

      <div className="model-pair">
        <NetworkSnapshot
          title="01 / Baseline checkpoint"
          snapshot={baseline}
          pixels={sample?.pixels}
          removedIndices={removedIndices}
          accuracy={baselineAccuracy}
          parameters={baselineParameters}
          tone="baseline"
        />
        <div className="model-transform-arrow" aria-hidden="true">
          <span>structured pruning</span>
          <strong>→</strong>
          <span>fine-tuning</span>
        </div>
        <NetworkSnapshot
          title="02 / Selected checkpoint"
          snapshot={optimized}
          pixels={sample?.pixels}
          accuracy={optimizedAccuracy}
          parameters={optimizedParameters}
          tone="selected"
        />
      </div>

      <div className="model-change-strip">
        <div>
          <span>Hidden 1</span>
          <strong>
            {formatNumber(hidden1)} → {selected1 === null ? "—" : formatNumber(selected1)}
          </strong>
        </div>
        <div>
          <span>Hidden 2</span>
          <strong>
            {formatNumber(hidden2)} → {selected2 === null ? "—" : formatNumber(selected2)}
          </strong>
        </div>
        <div>
          <span>Parameters</span>
          <strong>
            {formatNumber(baselineParameters)} → {formatNumber(optimizedParameters)}
          </strong>
        </div>
        <div>
          <span>Reduction</span>
          <strong>{reduction === null ? "—" : `−${reduction.toFixed(1)}%`}</strong>
        </div>
        <div>
          <span>Accuracy change</span>
          <strong>
            {baselineAccuracy === null || optimizedAccuracy === null
              ? "—"
              : formatSignedPP(optimizedAccuracy - baselineAccuracy)}
          </strong>
        </div>
      </div>
      <p className="data-note">
        The diagram is an abstraction in scale but not in counts: 784 input
        features, 256 and 128 hidden neurons, and 10 output classes are rendered
        as individual dots. The connection lines are a readable 5×5 sample of
        the actual weight matrix; no connection or weight is invented.
      </p>
    </div>
  );
}

export function LayerWidthComparison({
  baseline,
  selected,
  level,
}: {
  baseline: number[];
  selected: number[];
  level: number;
}) {
  return (
    <div className="width-comparison" aria-label="Hidden layer width before and after pruning">
      {[1, 2].map((index) => {
        const before = baseline[index];
        const after = selected[index];
        const ratio = after / Math.max(before, 1);
        return (
          <div className="width-row" key={index}>
            <span className="width-label">Hidden {index}</span>
            <div className="width-bars">
              <div className="width-bar width-bar-before" style={{ width: "100%" }}>
                <span>{before}</span>
              </div>
              <div
                className="width-bar width-bar-after"
                style={{ width: `${Math.max(ratio * 100, 4)}%` }}
              >
                <span>{after}</span>
              </div>
            </div>
            <span className="width-note">{`${Math.round(level * 100)}% removed`}</span>
          </div>
        );
      })}
    </div>
  );
}
