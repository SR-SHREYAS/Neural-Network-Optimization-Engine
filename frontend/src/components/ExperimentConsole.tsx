import { useState, type ReactNode } from "react";
import type {
  ActivityLog,
  CandidateResult,
  ManualStageKey,
  OptimizationJobStatus,
  OptimizationResponse,
  PipelineState,
  ModelProfile,
} from "../types";
import { AccuracyScatter, LinePlot } from "./Charts";
import { Callout, DataNote } from "./Section";
import {
  formatNumber,
  formatPercent,
  parameterReduction,
} from "../lib/format";

type ConsoleMode = "auto" | "manual";

function ActivityLogList({ items }: { items: ActivityLog[] }) {
  if (items.length === 0) return null;
  return (
    <div className="activity-log" aria-live="polite">
      <div className="activity-log-heading">
        <span>EXECUTION TRACE</span>
        <span>{items.length} events</span>
      </div>
      {items.slice(-6).map((item) => (
        <div className="activity-log-line" key={item.id}>
          <time>{item.time}</time>
          <span>{item.message}</span>
        </div>
      ))}
    </div>
  );
}

function ProgressBar({
  label,
  value,
  message,
  detail,
  active,
}: {
  label: string;
  value: number;
  message: string;
  detail?: string;
  active: boolean;
}) {
  const safeValue = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div className="console-progress" aria-live="polite">
      <div className="console-progress-head">
        <span className="eyebrow">{label}</span>
        <span>{safeValue}%</span>
      </div>
      <div className={`progress-track ${active ? "is-active" : ""}`}>
        <div className="progress-fill" style={{ width: `${safeValue}%` }} />
      </div>
      <div className="console-progress-meta">
        <span>{message}</span>
        {detail ? <span>{detail}</span> : null}
      </div>
    </div>
  );
}

function ParameterBars({
  candidates,
  baseline,
}: {
  candidates: PipelineState["candidates"];
  baseline: number | null;
}) {
  if (candidates.length === 0) return null;
  const max = Math.max(...candidates.map((item) => item.parameters), baseline ?? 1);
  return (
    <div className="parameter-bars">
      {candidates.map((candidate) => (
        <div className="parameter-bar-row" key={candidate.pruning}>
          <span>{Math.round(candidate.pruning * 100)}%</span>
          <div className="parameter-bar-track">
            <div
              className="parameter-bar-fill"
              style={{ width: `${(candidate.parameters / max) * 100}%` }}
            />
          </div>
          <strong>{formatNumber(candidate.parameters)}</strong>
        </div>
      ))}
      <span className="parameter-bars-note">
        remaining trainable parameters · baseline {formatNumber(baseline)}
      </span>
    </div>
  );
}

function StageCard({
  number,
  title,
  description,
  status,
  actionLabel,
  onAction,
  disabled,
  children,
}: {
  number: string;
  title: string;
  description: string;
  status: string;
  actionLabel: string;
  onAction: () => void;
  disabled: boolean;
  children?: ReactNode;
}) {
  return (
    <article className="stage-card">
      <div className="stage-card-head">
        <span className="stage-number">{number}</span>
        <div>
          <strong>{title}</strong>
          <span>{description}</span>
        </div>
        <span className="stage-status">{status}</span>
      </div>
      <div className="stage-card-action">
        <button
          type="button"
          className="button button-small"
          onClick={onAction}
          disabled={disabled}
        >
          {actionLabel}
        </button>
      </div>
      {children}
    </article>
  );
}

export function ExperimentConsole({
  profile,
  selected,
  maxAccuracyLoss,
  pruningLevels,
  setPruningLevels,
  optimizing,
  optimizationJob,
  optimization,
  optimizationLogs,
  requestStartedAt,
  elapsedTick,
  runOptimization,
  pipeline,
  pipelineBusy,
  manualProgress,
  pipelineMessage,
  manualStageLogs,
  canFineTune,
  canEvaluate,
  canSelect,
  runPipelinePruning,
  runPipelineFinetuning,
  runPipelineEvaluation,
  runPipelineSelection,
  setMaxAccuracyLoss,
}: {
  profile: ModelProfile | null;
  selected: CandidateResult | null;
  maxAccuracyLoss: string;
  pruningLevels: string;
  setPruningLevels: (value: string) => void;
  optimizing: boolean;
  optimizationJob: OptimizationJobStatus | null;
  optimization: OptimizationResponse | null;
  optimizationLogs: ActivityLog[];
  requestStartedAt: number | null;
  elapsedTick: number;
  runOptimization: () => void;
  pipeline: PipelineState;
  pipelineBusy: boolean;
  manualProgress: number;
  pipelineMessage: string;
  manualStageLogs: Record<ManualStageKey, ActivityLog[]>;
  canFineTune: boolean;
  canEvaluate: boolean;
  canSelect: boolean;
  runPipelinePruning: () => void;
  runPipelineFinetuning: () => void;
  runPipelineEvaluation: () => void;
  runPipelineSelection: () => void;
  setMaxAccuracyLoss: (value: string) => void;
}) {
  const [mode, setMode] = useState<ConsoleMode>("auto");
  const selectedReduction = parameterReduction(
    profile?.total_parameters ?? null,
    selected?.parameters ?? null,
  );
  const elapsed =
    requestStartedAt === null ? null : `${(elapsedTick / 4).toFixed(1)} s elapsed`;

  return (
    <div className="lab-console">
      <div className="console-tabs" role="tablist" aria-label="Experiment mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "auto"}
          className={mode === "auto" ? "is-active" : ""}
          onClick={() => setMode("auto")}
        >
          One-click optimization
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "manual"}
          className={mode === "manual" ? "is-active" : ""}
          onClick={() => setMode("manual")}
        >
          Manual pipeline
        </button>
      </div>

      {mode === "auto" ? (
        <div className="console-panel">
          <div className="console-controls">
            <div className="control">
              <label htmlFor="levels">Pruning levels</label>
              <div className="input-row">
                <input
                  id="levels"
                  value={pruningLevels}
                  onChange={(event) => setPruningLevels(event.target.value)}
                  spellCheck={false}
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
                  onChange={(event) => setMaxAccuracyLoss(event.target.value)}
                />
                <span>pp</span>
              </div>
              <small>Candidates above this loss are rejected.</small>
            </div>
            <div className="console-run">
              <button
                type="button"
                className="button button-primary"
                onClick={runOptimization}
                disabled={optimizing}
              >
                {optimizing ? "Optimizing…" : "Run optimization"} <span>↗</span>
              </button>
              <span>
                profile → prune → fine-tune → evaluate → select → save
              </span>
            </div>
          </div>

          <div className="console-side">
            <div className="console-side-head">
              <span className="eyebrow">PIPELINE</span>
              <span>{optimization ? "complete run available" : "ready"}</span>
            </div>
            <ol className="pipeline-steps">
              <li>Evaluate the baseline on the held-out test set.</li>
              <li>Prune each requested level from the original checkpoint.</li>
              <li>Fine-tune every candidate for two epochs.</li>
              <li>Measure each candidate on the test set.</li>
              <li>Select the smallest candidate within the loss limit.</li>
              <li>Save the selected checkpoint as <code>optimized_mlp.pth</code>.</li>
            </ol>
          </div>

          {optimizing ? (
            <div className="console-live">
              <ProgressBar
                label="OPTIMIZATION"
                value={optimizationJob?.progress ?? 0}
                message={optimizationJob?.message ?? "Preparing optimization…"}
                detail={
                  optimizationJob?.candidate && optimizationJob?.total_candidates
                    ? `candidate ${optimizationJob.candidate}/${optimizationJob.total_candidates}${
                        optimizationJob.epoch
                          ? ` · epoch ${optimizationJob.epoch}/${optimizationJob.total_epochs}`
                          : ""
                      }`
                    : elapsed ?? undefined
                }
                active
              />
              <ActivityLogList items={optimizationLogs} />
            </div>
          ) : null}

          <div className="run-trace">
            <div className="run-trace-head">
              <span className="eyebrow">RUN TRACE</span>
              <span>
                {optimization
                  ? "complete pipeline output"
                  : "appears after the complete run"}
              </span>
            </div>
            {optimization ? (
              <div className="trace-grid">
                <div>
                  <span>BASELINE</span>
                  <strong>{formatPercent(optimization.baseline_accuracy)}</strong>
                </div>
                <div>
                  <span>CANDIDATES</span>
                  <strong>{optimization.results.length}</strong>
                </div>
                <div>
                  <span>BEST AFTER FT</span>
                  <strong>
                    {formatPercent(
                      Math.max(...optimization.results.map((item) => item.accuracy)),
                    )}
                  </strong>
                </div>
                <div>
                  <span>SELECTED</span>
                  <strong>{Math.round(optimization.best_model.pruning * 100)}%</strong>
                </div>
                <div>
                  <span>PARAMETER CHANGE</span>
                  <strong>
                    {selectedReduction === null
                      ? "—"
                      : `−${selectedReduction.toFixed(1)}%`}
                  </strong>
                </div>
              </div>
            ) : (
              <p className="trace-empty">
                Run the complete pipeline to replace this placeholder with the
                measured baseline, candidate, selection, and parameter changes.
              </p>
            )}
          </div>
          {optimization ? (
            <DataNote>
              This run used the levels and loss limit shown above. The stored
              experiment table and the educational sections update from these
              measurements after the job completes.
            </DataNote>
          ) : null}
        </div>
      ) : (
        <div className="console-panel console-manual">
          <div className="manual-head">
            <div>
              <span className="eyebrow">MANUAL PIPELINE</span>
              <strong>{pipelineMessage}</strong>
            </div>
            <span className={`status-pill ${pipelineBusy ? "status-valid" : "status-baseline"}`}>
              {pipelineBusy
                ? "running"
                : pipeline.stage === "selected"
                  ? "complete"
                  : pipeline.stage === "idle"
                    ? "ready"
                    : "in progress"}
            </span>
          </div>

          <div className="manual-grid">
            <StageCard
              number="01"
              title="Prune"
              description="Generate reduced candidates from the baseline checkpoint."
              status={
                pipeline.stage === "pruning"
                  ? "running"
                  : pipeline.candidates.length > 0
                    ? `${pipeline.candidates.length} candidates`
                    : "ready"
              }
              actionLabel={pipeline.candidates.length > 0 ? "Run again" : "Run"}
              onAction={runPipelinePruning}
              disabled={pipelineBusy}
            >
              {pipeline.candidates.length > 0 ? (
                <div className="stage-output">
                  <ParameterBars
                    candidates={pipeline.candidates}
                    baseline={profile?.total_parameters ?? null}
                  />
                  <ActivityLogList items={manualStageLogs.prune} />
                </div>
              ) : null}
            </StageCard>

            <StageCard
              number="02"
              title="Fine-tune"
              description="Update the remaining weights to recover performance."
              status={
                pipeline.stage === "finetuning"
                  ? "running"
                  : ["finetuned", "evaluated", "selected"].includes(pipeline.stage)
                    ? "complete"
                    : "locked"
              }
              actionLabel={
                pipeline.stage === "finetuning"
                  ? "Running…"
                  : ["finetuned", "evaluated", "selected"].includes(pipeline.stage)
                    ? "Run again"
                    : "Run"
              }
              onAction={runPipelineFinetuning}
              disabled={pipelineBusy || !canFineTune}
            >
              {pipeline.candidates.length > 0 &&
              ["finetuned", "evaluated", "selected"].includes(pipeline.stage) ? (
                <div className="stage-output">
                  <LinePlot
                    series={[
                      {
                        label: "before fine-tuning",
                        color: "var(--rust)",
                        points: pipeline.candidates.map((candidate) => ({
                          x: candidate.pruning,
                          y: candidate.before_finetuning_accuracy,
                        })),
                      },
                    ]}
                    xLabel="pruning level"
                    yLabel="test accuracy (%)"
                    formatX={(value) => `${Math.round(value * 100)}%`}
                    formatY={(value) => formatPercent(value, 1)}
                    height={220}
                    reference={
                      pipeline.baselineAccuracy === null
                        ? undefined
                        : {
                            value: pipeline.baselineAccuracy,
                            label: "baseline",
                            color: "var(--muted)",
                          }
                    }
                  />
                  <ActivityLogList items={manualStageLogs.finetune} />
                </div>
              ) : null}
            </StageCard>

            <StageCard
              number="03"
              title="Evaluate"
              description="Measure each fine-tuned candidate on the held-out test split."
              status={
                pipeline.stage === "evaluation"
                  ? "running"
                  : pipeline.results.length > 0
                    ? "complete"
                    : "locked"
              }
              actionLabel={
                pipeline.stage === "evaluation"
                  ? "Running…"
                  : pipeline.results.length > 0
                    ? "Run again"
                    : "Run"
              }
              onAction={runPipelineEvaluation}
              disabled={pipelineBusy || !canEvaluate}
            >
              {pipeline.results.length > 0 ? (
                <div className="stage-output">
                  <LinePlot
                    series={[
                      {
                        label: "after fine-tuning",
                        color: "var(--green)",
                        points: pipeline.results.map((result) => ({
                          x: result.pruning,
                          y: result.accuracy,
                        })),
                      },
                    ]}
                    xLabel="pruning level"
                    yLabel="test accuracy (%)"
                    formatX={(value) => `${Math.round(value * 100)}%`}
                    formatY={(value) => formatPercent(value, 1)}
                    height={220}
                    reference={
                      pipeline.baselineAccuracy === null
                        ? undefined
                        : {
                            value: pipeline.baselineAccuracy,
                            label: "baseline",
                            color: "var(--muted)",
                          }
                    }
                  />
                  <ActivityLogList items={manualStageLogs.evaluate} />
                </div>
              ) : null}
            </StageCard>

            <StageCard
              number="04"
              title="Select"
              description="Apply the loss limit and keep the smallest valid model."
              status={
                pipeline.stage === "selection"
                  ? "running"
                  : pipeline.bestModel
                    ? "complete"
                    : "locked"
              }
              actionLabel={
                pipeline.stage === "selection"
                  ? "Running…"
                  : pipeline.bestModel
                    ? "Run again"
                    : "Run"
              }
              onAction={runPipelineSelection}
              disabled={pipelineBusy || !canSelect}
            >
              {pipeline.bestModel && pipeline.results.length > 0 ? (
                <div className="stage-output">
                  <AccuracyScatter
                    results={pipeline.results}
                    baselineAccuracy={pipeline.baselineAccuracy}
                    maxAccuracyLoss={Number(maxAccuracyLoss)}
                    selectedPruning={pipeline.bestModel.pruning}
                  />
                  <ActivityLogList items={manualStageLogs.select} />
                </div>
              ) : null}
            </StageCard>
          </div>

          <ProgressBar
            label="MANUAL PIPELINE"
            value={manualProgress}
            message={pipelineMessage}
            detail={elapsed ?? undefined}
            active={pipelineBusy}
          />
          <Callout title="The two flows are independent" tone="teal">
            The one-click optimization always sweeps every requested level from
            the original baseline and saves the selected checkpoint. The manual
            pipeline exposes the same stages separately so you can inspect each
            intermediate output before selecting. Running one flow does not
            replace the stored CSV unless that stage writes it.
          </Callout>
        </div>
      )}
    </div>
  );
}
