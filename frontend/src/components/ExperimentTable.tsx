import type { CandidateResult } from "../types";
import {
  accuracyLoss,
  architectureString,
  formatMegabytes,
  formatMilliseconds,
  formatNumber,
  formatPercent,
  formatSignedPP,
} from "../lib/format";

export function ExperimentTable({
  results,
  selected,
  maxAccuracyLoss,
  source,
}: {
  results: CandidateResult[];
  selected: CandidateResult | null;
  maxAccuracyLoss: number;
  source: string;
}) {
  const sorted = [...results].sort((a, b) => a.pruning - b.pruning);

  return (
    <div className="experiment-table-wrap">
      <div className="experiment-table-head">
        <span className="eyebrow">ALL MEASUREMENTS</span>
        <span>{source}</span>
      </div>
      <div className="table-scroll">
        <table>
          <caption className="sr-only">
            Structured pruning candidates with architecture, parameter count,
            model size, test accuracy, accuracy change, before-fine-tuning
            accuracy, latency, and selection status.
          </caption>
          <thead>
            <tr>
              <th scope="col">Pruning</th>
              <th scope="col">Architecture</th>
              <th scope="col">Parameters</th>
              <th scope="col">Dense size</th>
              <th scope="col">Test accuracy</th>
              <th scope="col">Δ accuracy</th>
              <th scope="col">Before FT</th>
              <th scope="col">Latency</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((result) => {
              const loss = accuracyLoss(result.accuracy_change);
              const valid = result.pruning === 0 || loss <= maxAccuracyLoss;
              const isSelected =
                selected !== null &&
                result.pruning === selected.pruning &&
                result.parameters === selected.parameters;
              return (
                <tr
                  key={`${result.pruning}-${result.parameters}`}
                  className={isSelected ? "selected-row" : ""}
                >
                  <td>
                    {result.pruning === 0
                      ? "Baseline"
                      : `${Math.round(result.pruning * 100)}%`}
                  </td>
                  <td className="mono">
                    {architectureString([
                      784,
                      result.hidden1_size,
                      result.hidden2_size,
                      10,
                    ])}
                  </td>
                  <td>{formatNumber(result.parameters)}</td>
                  <td>{formatMegabytes(result.model_size_mb, 3)}</td>
                  <td>{formatPercent(result.accuracy)}</td>
                  <td
                    className={
                      result.accuracy_change < 0 ? "negative" : "positive"
                    }
                  >
                    {formatSignedPP(result.accuracy_change)}
                  </td>
                  <td>
                    {result.before_finetuning_accuracy === undefined
                      ? "—"
                      : formatPercent(result.before_finetuning_accuracy)}
                  </td>
                  <td>{formatMilliseconds(result.latency_ms)}</td>
                  <td>
                    <span
                      className={`status-pill ${
                        result.pruning === 0
                          ? "status-baseline"
                          : isSelected
                            ? "status-selected"
                            : valid
                              ? "status-valid"
                              : "status-rejected"
                      }`}
                    >
                      {result.pruning === 0
                        ? "baseline"
                        : isSelected
                          ? "selected"
                          : valid
                            ? "valid"
                            : "rejected"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="data-note">
        “Dense size” is the parameter-storage estimate returned by the
        experiment service, not the serialized checkpoint size. “Before FT” is
        blank when the stored CSV does not contain that intermediate
        measurement. Latency is a single-sample measurement and varies with
        hardware and runtime.
      </p>
    </div>
  );
}
