import type { ReactNode } from "react";
import { Callout, Stat } from "./Section";
import {
  architectureString,
  formatNumber,
  formatPercent,
  formatSignedPP,
  parameterReduction,
} from "../lib/format";

const chapters = [
  { id: "problem", number: "01", label: "The task" },
  { id: "digit", number: "02", label: "A digit" },
  { id: "neuron", number: "03", label: "A neuron" },
  { id: "architecture", number: "04", label: "Architecture" },
  { id: "pruning", number: "05", label: "Pruning" },
  { id: "fine-tuning", number: "06", label: "Fine-tuning" },
  { id: "selection", number: "07", label: "Selection" },
  { id: "evidence", number: "08", label: "Evidence" },
  { id: "lab", number: "09", label: "Run it" },
  { id: "conclusion", number: "10", label: "Conclusion" },
];

export function ChapterNav({
  health,
  loading,
}: {
  health: boolean;
  loading: boolean;
}) {
  return (
    <header className="topbar">
      <a className="wordmark" href="#top" aria-label="Neural Network Optimization Engine home">
        NNOE<span className="wordmark-dot">.</span>
      </a>
      <nav className="chapter-nav" aria-label="Chapters">
        {chapters.map((chapter) => (
          <a key={chapter.id} href={`#${chapter.id}`}>
            <span>{chapter.number}</span>
            {chapter.label}
          </a>
        ))}
      </nav>
      <div className="topbar-status">
        <span className={`status-dot ${health ? "is-online" : ""}`} />
        <span>{loading ? "Loading" : health ? "API connected" : "API offline"}</span>
        <a className="topbar-cta" href="#lab">
          Run experiment <span>↗</span>
        </a>
      </div>
    </header>
  );
}

function HeroLayer({
  label,
  value,
  tone,
  ratio,
}: {
  label: string;
  value: number;
  tone: "input" | "baseline" | "selected" | "output";
  ratio: number;
}) {
  return (
    <div className={`hero-layer hero-layer-${tone}`} style={{ flex: ratio }}>
      <div className="hero-layer-bar">
        <span>{formatNumber(value)}</span>
      </div>
      <small>{label}</small>
    </div>
  );
}

export function Hero({
  baselineArchitecture,
  selectedArchitecture,
  baselineAccuracy,
  selectedAccuracy,
  baselineParameters,
  selectedParameters,
  selectedPruning,
}: {
  baselineArchitecture: number[];
  selectedArchitecture: number[];
  baselineAccuracy: number | null;
  selectedAccuracy: number | null;
  baselineParameters: number | null;
  selectedParameters: number | null;
  selectedPruning: number | null;
}) {
  const reduction = parameterReduction(baselineParameters, selectedParameters);
  const selectedHidden = selectedArchitecture.slice(1, 3).reduce((sum, value) => sum + value, 0);
  const baselineHidden = baselineArchitecture.slice(1, 3).reduce((sum, value) => sum + value, 0);

  return (
    <section className="hero" id="top">
      <div className="hero-copy">
        <span className="eyebrow">MNIST · STRUCTURED PRUNING · FINE-TUNING</span>
        <h1>
          A smaller network.
          <br />
          <em>Almost the same answer.</em>
        </h1>
        <p className="hero-lead">
          This is a live lab notebook for a college deep-learning project. It
          explains how a 784 → 256 → 128 → 10 network reads a handwritten digit,
          how we remove whole neurons, how fine-tuning tries to recover the lost
          accuracy, and what the measured experiment actually shows.
        </p>
        <div className="hero-actions">
          <a className="button button-primary" href="#lab">
            Run the optimization <span>↗</span>
          </a>
          <a className="button button-quiet" href="#problem">
            Start with the basics <span>↓</span>
          </a>
        </div>
        <p className="hero-note">
          The numbers below come from the current baseline checkpoint and the
          stored experiment. They update when you run a new optimization.
        </p>
      </div>

      <div className="hero-visual" aria-label="Architecture width comparison">
        <div className="hero-visual-head">
          <span className="eyebrow">THE TRANSFORMATION</span>
          <span className="mono">
            {selectedPruning === null
              ? "stored run"
              : `${Math.round(selectedPruning * 100)}% pruning selected`}
          </span>
        </div>
        <div className="hero-network">
          <div className="hero-network-label">
            <span>baseline</span>
            <strong>{architectureString(baselineArchitecture)}</strong>
          </div>
          <div className="hero-layer-row">
            {baselineArchitecture.map((value, index) => (
              <HeroLayer
                key={`baseline-${index}`}
                label={["input", "hidden 1", "hidden 2", "output"][index]}
                value={value}
                tone={index === 0 ? "input" : index === 3 ? "output" : "baseline"}
                ratio={index === 1 ? 1.6 : index === 2 ? 1 : 0.65}
              />
            ))}
          </div>
          <div className="hero-shrink">
            <span>remove neurons</span>
            <i />
            <span>retrain survivors</span>
          </div>
          <div className="hero-network-label hero-network-label-selected">
            <span>selected</span>
            <strong>{architectureString(selectedArchitecture)}</strong>
          </div>
          <div className="hero-layer-row">
            {selectedArchitecture.map((value, index) => (
              <HeroLayer
                key={`selected-${index}`}
                label={["input", "hidden 1", "hidden 2", "output"][index]}
                value={value}
                tone={index === 0 ? "input" : index === 3 ? "output" : "selected"}
                ratio={
                  index === 1
                    ? Math.max((value / Math.max(baselineArchitecture[1], 1)) * 1.6, 0.7)
                    : index === 2
                      ? Math.max((value / Math.max(baselineArchitecture[2], 1)) * 1, 0.55)
                      : 0.65
                }
              />
            ))}
          </div>
        </div>
        <div className="hero-hidden-summary">
          <span>
            hidden width <strong>{formatNumber(baselineHidden)} → {formatNumber(selectedHidden)}</strong>
          </span>
          <span>
            parameters <strong>{reduction === null ? "—" : `−${reduction.toFixed(1)}%`}</strong>
          </span>
        </div>
      </div>

      <div className="hero-stats">
        <Stat
          label="Baseline test accuracy"
          value={formatPercent(baselineAccuracy)}
          detail="held-out MNIST test set"
          tone="teal"
        />
        <Stat
          label="Selected test accuracy"
          value={formatPercent(selectedAccuracy)}
          detail={
            baselineAccuracy === null || selectedAccuracy === null
              ? "run or load an experiment"
              : formatSignedPP(selectedAccuracy - baselineAccuracy)
          }
          tone="green"
        />
        <Stat
          label="Parameter reduction"
          value={reduction === null ? "—" : `−${reduction.toFixed(1)}%`}
          detail={`${formatNumber(baselineParameters)} → ${formatNumber(selectedParameters)}`}
          tone="amber"
        />
        <Stat
          label="Selected pruning"
          value={selectedPruning === null ? "—" : `${Math.round(selectedPruning * 100)}%`}
          detail="same fraction applied to both hidden layers"
          tone="rust"
        />
      </div>
    </section>
  );
}

export function Conclusion({
  baselineAccuracy,
  selectedAccuracy,
  baselineParameters,
  selectedParameters,
  selectedPruning,
  maxAccuracyLoss,
}: {
  baselineAccuracy: number | null;
  selectedAccuracy: number | null;
  baselineParameters: number | null;
  selectedParameters: number | null;
  selectedPruning: number | null;
  maxAccuracyLoss: number;
}) {
  const reduction = parameterReduction(baselineParameters, selectedParameters);
  const delta =
    baselineAccuracy === null || selectedAccuracy === null
      ? null
      : selectedAccuracy - baselineAccuracy;
  const loss = delta === null ? null : Math.max(0, -delta);

  return (
    <section className="conclusion" id="conclusion" aria-labelledby="conclusion-title">
      <div className="conclusion-head">
        <span className="eyebrow">10 / WHAT THE EVIDENCE SUPPORTS</span>
        <h2 id="conclusion-title">
          The result is a measured trade-off,
          <br />
          not a universal law.
        </h2>
      </div>
      <div className="conclusion-grid">
        <div className="conclusion-copy">
          <p>
            On this MNIST run, structured pruning at{" "}
            <strong>{selectedPruning === null ? "—" : `${Math.round(selectedPruning * 100)}%`}</strong>{" "}
            reduced the parameter count from{" "}
            <strong>{formatNumber(baselineParameters)}</strong> to{" "}
            <strong>{formatNumber(selectedParameters)}</strong>
            {reduction === null ? "" : ` (${reduction.toFixed(1)}% fewer)`}. The
            selected model scored <strong>{formatPercent(selectedAccuracy)}</strong>{" "}
            on the held-out test set versus{" "}
            <strong>{formatPercent(baselineAccuracy)}</strong> for the baseline
            {delta === null ? "" : `, a change of ${formatSignedPP(delta)}`}.
          </p>
          <p>
            The configured limit allowed at most{" "}
            <strong>{maxAccuracyLoss.toFixed(2)} percentage points</strong> of
            accuracy loss. The measured loss was{" "}
            <strong>{loss === null ? "—" : `${loss.toFixed(2)} pp`}</strong>
            {loss === null
              ? "."
              : loss <= maxAccuracyLoss
                ? ", so the candidate satisfied the constraint."
                : ", which would violate the constraint."}
          </p>
          <p>
            This supports a narrow, defensible conclusion: for this architecture,
            dataset, seed, pruning rule, and fine-tuning recipe, a smaller dense
            network retained almost all of the measured accuracy. It does not prove
            that the same pruning fraction will work on another architecture or
            dataset, and it is not a multi-seed statistical claim.
          </p>
        </div>
        <div className="conclusion-limits">
          <Callout title="What this experiment establishes" tone="green">
            <ul>
              <li>The exact parameter and architecture change is measured.</li>
              <li>The accuracy change is measured on the same held-out test set.</li>
              <li>The selector applied a configured accuracy-loss constraint.</li>
              <li>The remaining weights were fine-tuned for two epochs.</li>
            </ul>
          </Callout>
          <Callout title="What it does not establish" tone="rust">
            <ul>
              <li>That low importance proves a neuron is causally unnecessary.</li>
              <li>That the same pruning fraction generalizes to other models.</li>
              <li>That fewer parameters always means proportionally lower latency.</li>
              <li>That one run is a statistically significant benchmark.</li>
            </ul>
          </Callout>
        </div>
      </div>
      <p className="conclusion-footnote">
        Importance is a heuristic based on incoming weight magnitude. Latency is
        hardware- and runtime-dependent. The serialized checkpoint size and the
        dense parameter-storage estimate are related but not identical
        measurements.
      </p>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="site-footer">
      <span>NNOE · Neural Network Optimization Engine</span>
      <span>PyTorch · FastAPI · React</span>
      <a href="#top">Back to top ↑</a>
    </footer>
  );
}

export function LoadingBlock({ children }: { children: ReactNode }) {
  return <div className="loading-block">{children}</div>;
}
