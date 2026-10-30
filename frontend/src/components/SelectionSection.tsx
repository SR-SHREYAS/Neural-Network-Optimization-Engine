import { useState } from "react";

import type { CandidateResult } from "../types";
import { AccuracyScatter } from "./Charts";
import { Callout, DataNote, Formula } from "./Section";
import {
  accuracyLoss,
  formatNumber,
  formatPercent,
  formatSignedPP,
  parameterReduction,
} from "../lib/format";

type DownloadState = "idle" | "loading" | "success" | "error";

function DownloadModelButton({
  selected,
  downloadUrl,
}: {
  selected: CandidateResult | null;
  downloadUrl: string;
}) {
  const [state, setState] = useState<DownloadState>("idle");
  const [message, setMessage] = useState("");

  async function downloadModel() {
    if (!selected) return;

    setState("loading");
    setMessage("Preparing the saved checkpoint…");

    try {
      const response = await fetch(downloadUrl, { cache: "no-store" });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(
          payload?.detail ?? "The optimized model could not be downloaded.",
        );
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = "optimized_mlp.pth";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);

      setState("success");
      setMessage("Download started · optimized_mlp.pth");
    } catch (error) {
      const detail =
        error instanceof Error && error.message !== "Failed to fetch"
          ? error.message
          : "Check the API connection and try again.";
      setState("error");
      setMessage(`Download failed. ${detail}`);
    }
  }

  const loading = state === "loading";

  return (
    <div className="download-model">
      <div className="download-model-copy">
        <span className="eyebrow">SAVED OPTIMIZED CHECKPOINT</span>
        <strong>
          {selected
            ? `${Math.round(selected.pruning * 100)}% pruning · ${formatNumber(
                selected.parameters,
              )} parameters`
            : "No optimized model selected"}
        </strong>
        <span>
          The download contains the exact saved network, ready to load with
          PyTorch.
        </span>
      </div>
      <div className="download-model-action">
        <button
          type="button"
          className={`download-model-button is-${state}`}
          onClick={downloadModel}
          disabled={!selected || loading}
          aria-describedby="download-model-status"
        >
          <span className="download-model-icon" aria-hidden="true">
            {loading ? (
              <span className="download-spinner" />
            ) : state === "success" ? (
              "✓"
            ) : (
              "↓"
            )}
          </span>
          <span className="download-model-label">
            <strong>
              {loading
                ? "Preparing download…"
                : state === "success"
                  ? "Download again"
                  : "Download new model"}
            </strong>
            <small>optimized_mlp.pth</small>
          </span>
          <span className="download-model-arrow" aria-hidden="true">
            ↗
          </span>
        </button>
        <span
          id="download-model-status"
          className={`download-model-status is-${state}`}
          role={state === "error" ? "alert" : "status"}
          aria-live={state === "error" ? "assertive" : "polite"}
        >
          {message}
        </span>
      </div>
    </div>
  );
}

export function SelectionSection({
  results,
  baselineAccuracy,
  baselineParameters,
  selected,
  maxAccuracyLoss,
  downloadUrl,
}: {
  results: CandidateResult[];
  baselineAccuracy: number | null;
  baselineParameters: number | null;
  selected: CandidateResult | null;
  maxAccuracyLoss: number;
  downloadUrl: string;
}) {
  const threshold =
    baselineAccuracy === null ? null : baselineAccuracy - maxAccuracyLoss;
  const smallest = results.reduce<CandidateResult | null>((best, result) => {
    if (!best || result.parameters < best.parameters) return result;
    return best;
  }, null);
  const selectedReduction = parameterReduction(
    baselineParameters,
    selected?.parameters ?? null,
  );
  const smallestLoss =
    smallest === null ? null : accuracyLoss(smallest.accuracy_change);
  const selectedLoss =
    selected === null ? null : accuracyLoss(selected.accuracy_change);

  return (
    <div className="selection-section">
      <div className="selection-intro">
        <div>
          <span className="eyebrow">THE SELECTION RULE, EXACTLY AS IMPLEMENTED</span>
          <h3>The smallest valid model wins, not the most accurate one.</h3>
          <p>
            A candidate is eligible when its measured accuracy loss is no larger
            than the configured limit. Among eligible candidates, the optimizer
            prefers fewer trainable parameters. If two candidates have the same
            parameter count, the higher accuracy wins.
          </p>
        </div>
        <div className="selection-threshold">
          <span className="eyebrow">CONFIGURED CONSTRAINT</span>
          <strong>{maxAccuracyLoss.toFixed(2)} pp</strong>
          <span>
            minimum acceptable accuracy
            <br />
            <b>{formatPercent(threshold)}</b>
          </span>
        </div>
      </div>

      <div className="selection-grid">
        <div className="selection-chart-panel">
          <AccuracyScatter
            results={results}
            baselineAccuracy={baselineAccuracy}
            maxAccuracyLoss={maxAccuracyLoss}
            selectedPruning={selected?.pruning ?? null}
          />
        </div>
        <div className="selection-logic">
          <div className="logic-step">
            <span>01</span>
            <div>
              <strong>Compute loss in percentage points.</strong>
              <Formula
                expression="L_A = max(0, A_baseline − A_candidate)"
                explanation="An accuracy improvement is treated as zero loss, so it is eligible."
              />
            </div>
          </div>
          <div className="logic-step">
            <span>02</span>
            <div>
              <strong>Reject candidates above the limit.</strong>
              <p>
                If <code>L_A &gt; {maxAccuracyLoss.toFixed(2)} pp</code>, the
                candidate cannot be selected even if it is smaller.
              </p>
            </div>
          </div>
          <div className="logic-step">
            <span>03</span>
            <div>
              <strong>Prefer fewer parameters.</strong>
              <p>
                Among valid candidates, compare parameter counts. If they tie,
                compare test accuracy.
              </p>
            </div>
          </div>
          <Callout title="Units matter" tone="amber">
            Accuracy change is shown in percentage points (pp). A change from
            97.51% to 97.39% is −0.12 pp, not a −0.12% relative change.
          </Callout>
        </div>
      </div>

      <div className="verdict-grid">
        <div className="verdict-card verdict-selected">
          <span className="eyebrow">SELECTED</span>
          {selected ? (
            <>
              <strong>{Math.round(selected.pruning * 100)}% pruning</strong>
              <span>
                {formatNumber(selected.parameters)} parameters ·{" "}
                {formatPercent(selected.accuracy)} accuracy ·{" "}
                {formatSignedPP(selected.accuracy_change)}
              </span>
              <span>
                loss {selectedLoss === null ? "—" : `${selectedLoss.toFixed(2)} pp`}{" "}
                {selectedLoss !== null && selectedLoss <= maxAccuracyLoss
                  ? "within the limit"
                  : "above the limit"}
              </span>
              <span>
                parameter reduction{" "}
                {selectedReduction === null
                  ? "—"
                  : `−${selectedReduction.toFixed(1)}%`}
              </span>
            </>
          ) : (
            <span>Run an optimization or load a stored result.</span>
          )}
        </div>
        <div className="verdict-card verdict-smallest">
          <span className="eyebrow">SMALLEST CANDIDATE</span>
          {smallest ? (
            <>
              <strong>{Math.round(smallest.pruning * 100)}% pruning</strong>
              <span>
                {formatNumber(smallest.parameters)} parameters ·{" "}
                {formatPercent(smallest.accuracy)} accuracy
              </span>
              <span>
                loss {smallestLoss === null ? "—" : `${smallestLoss.toFixed(2)} pp`}{" "}
                {smallestLoss !== null && smallestLoss <= maxAccuracyLoss
                  ? "eligible"
                  : "rejected by the constraint"}
              </span>
              <span>
                {smallest.pruning === selected?.pruning
                  ? "this is also the selected model"
                  : "smaller, but not necessarily valid"}
              </span>
            </>
          ) : (
            <span>No candidate measurements available.</span>
          )}
        </div>
      </div>

      <DownloadModelButton
        key={selected ? `${selected.pruning}-${selected.parameters}` : "empty"}
        selected={selected}
        downloadUrl={downloadUrl}
      />

      <div className="candidate-verdicts">
        <span className="eyebrow">CANDIDATE-BY-CANDIDATE VERDICT</span>
        <div className="candidate-verdict-list">
          {results.map((result) => {
            const loss = accuracyLoss(result.accuracy_change);
            const valid = loss <= maxAccuracyLoss;
            const isSelected = result.pruning === selected?.pruning;
            return (
              <div
                key={`${result.pruning}-${result.parameters}`}
                className={`candidate-verdict ${isSelected ? "is-selected" : valid ? "is-valid" : "is-rejected"}`}
              >
                <strong>{Math.round(result.pruning * 100)}%</strong>
                <span>{formatNumber(result.parameters)} params</span>
                <span>{formatPercent(result.accuracy)}</span>
                <span>{formatSignedPP(result.accuracy_change)}</span>
                <span>
                  {isSelected ? "selected" : valid ? "valid" : "rejected"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <DataNote>
        The constraint is applied to the measured test accuracy of each
        candidate. The threshold above is baseline accuracy minus the configured
        loss limit. If the baseline accuracy is unavailable, the threshold is
        shown as unavailable rather than guessed.
      </DataNote>
    </div>
  );
}
