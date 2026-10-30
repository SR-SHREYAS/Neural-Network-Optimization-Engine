import { useEffect, useMemo, useRef, useState } from "react";

import "./App.css";
import { ModelExplorer } from "./ModelExplorer";
import { ChapterNav, Conclusion, Footer, Hero } from "./components/Shell";
import { Callout, DataNote, Formula, Section, Stat } from "./components/Section";
import { DigitSection, NeuronSection } from "./components/DigitSection";
import { PruningSection } from "./components/PruningSection";
import { FineTuningSection } from "./components/FineTuningSection";
import { SelectionSection } from "./components/SelectionSection";
import { ExperimentTable } from "./components/ExperimentTable";
import { ExperimentConsole } from "./components/ExperimentConsole";
import type {
  ActivityLog,
  CandidateResult,
  ExperimentResult,
  ManualStageKey,
  ModelAnalysis,
  ModelProfile,
  OptimizationJobStatus,
  OptimizationResponse,
  PipelineCandidate,
  PipelineResult,
  PipelineState,
  PruningResponse,
  SamplePrediction,
  WeightModelSnapshot,
} from "./types";
import {
  accuracyLoss,
  architectureString,
  formatMegabytes,
  formatMilliseconds,
  formatNumber,
  formatPercent,
  formatSignedPP,
  parameterReduction,
  sizeReduction,
} from "./lib/format";

const API_BASE_URL = "http://127.0.0.1:8000";

const DEFAULT_ARCHITECTURE = [784, 256, 128, 10];
const DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

function App() {
  const [profile, setProfile] = useState<ModelProfile | null>(null);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [health, setHealth] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [baselineWeights, setBaselineWeights] =
    useState<WeightModelSnapshot | null>(null);
  const [optimizedWeights, setOptimizedWeights] =
    useState<WeightModelSnapshot | null>(null);
  const [optimizedWeightsNote, setOptimizedWeightsNote] = useState<string | null>(
    null,
  );
  const [analysis, setAnalysis] = useState<ModelAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [sample, setSample] = useState<SamplePrediction | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [pruningInfo, setPruningInfo] = useState<PruningResponse | null>(null);
  const [pruningLoading, setPruningLoading] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);

  const [optimizing, setOptimizing] = useState(false);
  const [optimizationJob, setOptimizationJob] =
    useState<OptimizationJobStatus | null>(null);
  const [optimization, setOptimization] =
    useState<OptimizationResponse | null>(null);
  const [activeSource, setActiveSource] = useState<
    "optimization" | "manual" | null
  >(null);
  const pollingTimer = useRef<number | null>(null);
  const lastOptimizationEvent = useRef<string>("");
  const nextLogId = useRef(1);
  const [optimizationLogs, setOptimizationLogs] = useState<ActivityLog[]>([]);
  const [manualStageLogs, setManualStageLogs] = useState<
    Record<ManualStageKey, ActivityLog[]>
  >({
    prune: [],
    finetune: [],
    evaluate: [],
    select: [],
  });
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null);
  const [elapsedTick, setElapsedTick] = useState(0);

  const [pruningLevels, setPruningLevels] = useState(
    "10, 20, 30, 40, 50, 60, 70",
  );
  const [maxAccuracyLoss, setMaxAccuracyLoss] = useState("0.50");

  const [pipeline, setPipeline] = useState<PipelineState>({
    stage: "idle",
    baselineAccuracy: null,
    candidates: [],
    results: [],
    bestModel: null,
  });
  const [pipelineBusy, setPipelineBusy] = useState(false);
  const [manualProgress, setManualProgress] = useState(0);
  const [pipelineMessage, setPipelineMessage] = useState(
    "Ready to run an optimization stage.",
  );

  /* ------------------------------------------------------------------
   * Initial load: health, profile, stored experiment
   * ------------------------------------------------------------------ */
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);

        const [healthResponse, profileResponse, resultsResponse] =
          await Promise.all([
            fetch(`${API_BASE_URL}/health`),
            fetch(`${API_BASE_URL}/model/profile`),
            fetch(`${API_BASE_URL}/results`, { cache: "no-store" }),
          ]);

        if (!healthResponse.ok) {
          throw new Error("Optimization API is unavailable.");
        }
        if (!profileResponse.ok) {
          throw new Error("Unable to load the baseline model profile.");
        }

        setHealth(true);
        setProfile(await profileResponse.json());

        if (resultsResponse.ok) {
          setResults(await resultsResponse.json());
        } else {
          setResults([]);
        }
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

    void loadData();
  }, []);

  /* ------------------------------------------------------------------
   * Baseline checkpoint weights
   * ------------------------------------------------------------------ */
  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE_URL}/model/weights?model=baseline`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) return null;
        return (await response.json()) as WeightModelSnapshot;
      })
      .then((snapshot) => {
        if (!cancelled && snapshot) setBaselineWeights(snapshot);
      })
      .catch(() => {
        /* The structural sections remain available from the profile. */
      });

    return () => {
      cancelled = true;
    };
  }, []);

  /* ------------------------------------------------------------------
   * Selected model: manual selection > current one-click run > stored run
   * ------------------------------------------------------------------ */
  const storedFallback = useMemo<CandidateResult | null>(() => {
    const allowedLoss = Number(maxAccuracyLoss);
    if (!Number.isFinite(allowedLoss) || allowedLoss < 0) return null;

    return results
      .filter(
        (result) =>
          result.pruning > 0 &&
          accuracyLoss(result.accuracy_change) <= allowedLoss,
      )
      .reduce<CandidateResult | null>((best, result) => {
        if (!best || result.parameters < best.parameters) return result;
        if (
          result.parameters === best.parameters &&
          result.accuracy > best.accuracy
        ) {
          return result;
        }
        return best;
      }, null);
  }, [results, maxAccuracyLoss]);

  const selectedModel = useMemo<CandidateResult | null>(() => {
    if (activeSource === "manual" && pipeline.bestModel) {
      return pipeline.bestModel;
    }
    if (optimization?.best_model) return optimization.best_model;
    return storedFallback ?? pipeline.bestModel;
  }, [activeSource, pipeline.bestModel, optimization, storedFallback]);

  const baselineRow = useMemo<CandidateResult | null>(() => {
    if (results.length === 0) return null;
    return results.find((result) => result.pruning === 0) ?? null;
  }, [results]);

  const baselineArchitecture = useMemo(
    () =>
      analysis?.baseline.architecture ??
      baselineWeights?.architecture ??
      (baselineRow
        ? [784, baselineRow.hidden1_size, baselineRow.hidden2_size, 10]
        : DEFAULT_ARCHITECTURE),
    [analysis, baselineWeights, baselineRow],
  );

  const selectedArchitecture = useMemo(
    () =>
      selectedModel
        ? [
            784,
            selectedModel.hidden1_size,
            selectedModel.hidden2_size,
            10,
          ]
        : [],
    [selectedModel],
  );

  const baselineAccuracy =
    optimization?.baseline_accuracy ??
    pipeline.baselineAccuracy ??
    analysis?.baseline_accuracy ??
    baselineRow?.accuracy ??
    null;

  const baselineParameters =
    profile?.total_parameters ??
    analysis?.baseline.parameters ??
    baselineRow?.parameters ??
    null;

  const baselineCheckpointSize =
    profile?.model_size_mb ??
    analysis?.baseline.checkpoint_size_mb ??
    baselineRow?.model_size_mb ??
    null;

  const baselineLatency =
    profile?.inference_latency_ms ??
    analysis?.baseline.latency_ms ??
    baselineRow?.latency_ms ??
    null;

  const maxLossValue = Number(maxAccuracyLoss);
  const maxLossSafe =
    Number.isFinite(maxLossValue) && maxLossValue >= 0
      ? maxLossValue
      : 0.5;

  /* ------------------------------------------------------------------
   * Available pruning levels
   * ------------------------------------------------------------------ */
  const availableLevels = useMemo(() => {
    const levels = new Set<number>();
    results.forEach((result) => {
      if (result.pruning > 0) levels.add(result.pruning);
    });
    optimization?.results.forEach((result) => {
      if (result.pruning > 0) levels.add(result.pruning);
    });
    pipeline.candidates.forEach((candidate) => levels.add(candidate.pruning));

    if (levels.size === 0) {
      DEFAULT_LEVELS.forEach((level) => levels.add(level));
    }

    return Array.from(levels).sort((a, b) => a - b);
  }, [results, optimization, pipeline.candidates]);

  const activePruningLevel =
    selectedLevel ?? selectedModel?.pruning ?? availableLevels[0] ?? null;

  /* ------------------------------------------------------------------
   * Analysis for the selected level (also provides baseline accuracy)
   * ------------------------------------------------------------------ */
  useEffect(() => {
    let cancelled = false;
    const level = selectedModel?.pruning ?? null;
    const url =
      level === null
        ? `${API_BASE_URL}/model/analysis`
        : `${API_BASE_URL}/model/analysis?level=${level}`;

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAnalysisLoading(true);
    setAnalysisError(null);

    fetch(url, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          throw new Error(
            payload?.detail ?? "Unable to load the model analysis.",
          );
        }
        return (await response.json()) as ModelAnalysis;
      })
      .then((payload) => {
        if (!cancelled) setAnalysis(payload);
      })
      .catch((err) => {
        if (!cancelled) {
          setAnalysisError(
            err instanceof Error ? err.message : "Unable to load the model analysis.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setAnalysisLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    selectedModel?.pruning,
    selectedModel?.parameters,
    selectedModel?.accuracy,
    optimization?.baseline_accuracy,
    pipeline.baselineAccuracy,
  ]);

  /* ------------------------------------------------------------------
   * Exact pruning indices for the level currently being inspected
   * ------------------------------------------------------------------ */
  useEffect(() => {
    if (activePruningLevel === null) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPruningInfo(null);
      return;
    }

    let cancelled = false;
    setPruningLoading(true);

    fetch(
      `${API_BASE_URL}/model/pruning?level=${activePruningLevel}`,
      { cache: "no-store" },
    )
      .then(async (response) => {
        if (!response.ok) return null;
        return (await response.json()) as PruningResponse;
      })
      .then((payload) => {
        if (!cancelled && payload) setPruningInfo(payload);
      })
      .catch(() => {
        /* The analysis endpoint may still provide the same layer data. */
      })
      .finally(() => {
        if (!cancelled) setPruningLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activePruningLevel]);

  /* ------------------------------------------------------------------
   * Optimized checkpoint weights, only when they match the selection
   * ------------------------------------------------------------------ */
  useEffect(() => {
    if (!selectedModel) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setOptimizedWeights(null);
      setOptimizedWeightsNote(null);
      return;
    }

    let cancelled = false;

    fetch(`${API_BASE_URL}/model/weights?model=optimized`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) return null;
        return (await response.json()) as WeightModelSnapshot;
      })
      .then((snapshot) => {
        if (cancelled) return;

        if (!snapshot) {
          setOptimizedWeights(null);
          setOptimizedWeightsNote(
            "No saved optimized checkpoint is available yet. Run the optimization to create one.",
          );
          return;
        }

        const matchesSelection =
          snapshot.architecture[1] === selectedModel.hidden1_size &&
          snapshot.architecture[2] === selectedModel.hidden2_size;

        if (!matchesSelection) {
          setOptimizedWeights(null);
          setOptimizedWeightsNote(
            `The saved checkpoint is ${architectureString(
              snapshot.architecture,
            )}, which does not match the selected candidate ${architectureString(
              selectedArchitecture,
            )}. Run the optimization to save the selected architecture.`,
          );
          return;
        }

        setOptimizedWeights(snapshot);
        setOptimizedWeightsNote(null);
      })
      .catch(() => {
        if (!cancelled) {
          setOptimizedWeights(null);
          setOptimizedWeightsNote(
            "The optimized checkpoint could not be inspected.",
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedModel, selectedArchitecture]);

  /* ------------------------------------------------------------------
   * A real MNIST test sample and both checkpoint predictions
   * ------------------------------------------------------------------ */
  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSampleLoading(true);
    setSampleError(null);

    fetch(`${API_BASE_URL}/model/sample?index=${sampleIndex}`, {
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          throw new Error(payload?.detail ?? "Unable to load a test sample.");
        }
        return (await response.json()) as SamplePrediction;
      })
      .then((payload) => {
        if (!cancelled) setSample(payload);
      })
      .catch((err) => {
        if (!cancelled) {
          setSampleError(
            err instanceof Error ? err.message : "Unable to load a test sample.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setSampleLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sampleIndex, selectedModel?.parameters]);

  /* ------------------------------------------------------------------
   * Progress clock
   * ------------------------------------------------------------------ */
  useEffect(() => {
    if (!optimizing && !pipelineBusy) return;
    const timer = window.setInterval(
      () => setElapsedTick((value) => value + 1),
      250,
    );
    return () => window.clearInterval(timer);
  }, [optimizing, pipelineBusy]);

  useEffect(() => {
    return () => {
      if (pollingTimer.current !== null) {
        window.clearTimeout(pollingTimer.current);
      }
    };
  }, []);

  function formatLogTime() {
    return new Date().toLocaleTimeString([], {
      hour12: false,
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function addOptimizationLog(message: string) {
    const entry = {
      id: nextLogId.current++,
      time: formatLogTime(),
      message,
    };
    setOptimizationLogs((current) => [...current, entry].slice(-10));
  }

  function addManualLog(stage: ManualStageKey, message: string) {
    const entry = {
      id: nextLogId.current++,
      time: formatLogTime(),
      message,
    };
    setManualStageLogs((current) => ({
      ...current,
      [stage]: [...current[stage], entry].slice(-8),
    }));
  }

  function parsePruningLevels(): number[] {
    const levels = pruningLevels
      .split(",")
      .map((value) => Number(value.trim()) / 100)
      .filter((value) => !Number.isNaN(value));

    if (
      levels.length === 0 ||
      levels.some((level) => level < 0 || level >= 1)
    ) {
      throw new Error("Pruning levels must be percentages between 0 and 99.");
    }

    return levels;
  }

  function parseMaxLoss(): number {
    const loss = Number(maxAccuracyLoss);
    if (!Number.isFinite(loss) || loss < 0) {
      throw new Error("Maximum accuracy loss must be non-negative.");
    }
    return loss;
  }

  function applyOptimizationResult(data: OptimizationResponse) {
    setOptimization(data);
    setActiveSource("optimization");
    setSelectedLevel(null);
    setPipelineMessage(
      `${Math.round(data.best_model.pruning * 100)}% pruning selected in the complete run.`,
    );
  }

  async function pollOptimization(jobId: string) {
    try {
      const response = await fetch(
        `${API_BASE_URL}/optimize/status/${jobId}`,
        { cache: "no-store" },
      );
      const data: OptimizationJobStatus = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ?? "Unable to read optimization status.",
        );
      }

      setOptimizationJob(data);

      const eventKey = [
        data.status,
        data.stage,
        data.progress,
        data.candidate,
        data.epoch,
        data.accuracy,
        data.loss,
        data.message,
      ].join("|");

      if (eventKey !== lastOptimizationEvent.current) {
        lastOptimizationEvent.current = eventKey;
        const details = [
          data.candidate && data.total_candidates
            ? `candidate ${data.candidate}/${data.total_candidates}`
            : "",
          data.epoch && data.total_epochs
            ? `epoch ${data.epoch}/${data.total_epochs}`
            : "",
          data.accuracy !== null ? `accuracy ${data.accuracy.toFixed(2)}%` : "",
          data.loss !== null ? `loss ${data.loss.toFixed(5)}` : "",
          `${Math.round(data.progress)}%`,
        ]
          .filter(Boolean)
          .join(" · ");
        addOptimizationLog(
          `${String(data.stage || data.status).replaceAll("_", " ")}: ${
            data.message || data.status
          }${details ? ` — ${details}` : ""}`,
        );
      }

      if (data.status === "completed") {
        if (!data.result) {
          throw new Error("Optimization completed without a result.");
        }

        applyOptimizationResult(data.result);
        setOptimizing(false);
        setRequestStartedAt(null);

        if (pollingTimer.current !== null) {
          window.clearTimeout(pollingTimer.current);
          pollingTimer.current = null;
        }

        const resultsResponse = await fetch(`${API_BASE_URL}/results`, {
          cache: "no-store",
        });
        if (resultsResponse.ok) {
          setResults(await resultsResponse.json());
        }
        return;
      }

      if (data.status === "failed") {
        setError(data.error ?? "Optimization failed.");
        setOptimizing(false);
        setRequestStartedAt(null);

        if (pollingTimer.current !== null) {
          window.clearTimeout(pollingTimer.current);
          pollingTimer.current = null;
        }
        return;
      }

      pollingTimer.current = window.setTimeout(() => {
        void pollOptimization(jobId);
      }, 500);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to read optimization status.",
      );
      setOptimizing(false);
      setRequestStartedAt(null);

      if (pollingTimer.current !== null) {
        window.clearTimeout(pollingTimer.current);
        pollingTimer.current = null;
      }
    }
  }

  async function runOptimization() {
    try {
      setOptimizing(true);
      setError(null);
      setActiveSource("optimization");
      setOptimizationJob(null);
      setOptimizationLogs([]);
      lastOptimizationEvent.current = "";
      setRequestStartedAt(Date.now());
      setElapsedTick(0);
      addOptimizationLog("Optimization request submitted to the API.");
      setPipelineMessage("Starting optimization...");

      const levels = parsePruningLevels();
      const loss = parseMaxLoss();

      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pruning_levels: levels,
          max_accuracy_loss: loss,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Optimization failed.");
      }

      if (data.job_id) {
        await pollOptimization(data.job_id);
        return;
      }

      if (Array.isArray(data.results) && data.best_model) {
        applyOptimizationResult(data as OptimizationResponse);
        setOptimizing(false);
        setRequestStartedAt(null);

        const resultsResponse = await fetch(`${API_BASE_URL}/results`, {
          cache: "no-store",
        });
        if (resultsResponse.ok) {
          setResults(await resultsResponse.json());
        }
        return;
      }

      throw new Error("Optimization API returned an invalid response.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Optimization failed.");
      setOptimizing(false);
      setRequestStartedAt(null);
    }
  }

  async function runPipelinePruning() {
    try {
      setActiveSource("manual");
      setRequestStartedAt(Date.now());
      setElapsedTick(0);
      setManualProgress(8);
      setPipelineBusy(true);
      setError(null);
      setPipelineMessage("Generating pruned model candidates...");
      setPipeline((current) => ({ ...current, stage: "pruning" }));
      addManualLog("prune", `Request submitted with levels: ${pruningLevels}.`);
      addManualLog(
        "prune",
        `Baseline parameter count: ${
          profile?.total_parameters?.toLocaleString() ?? "loading"
        }.`,
      );

      const levels = parsePruningLevels();
      const response = await fetch(`${API_BASE_URL}/pipeline/prune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pruning_levels: levels }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Pruning failed.");
      }

      setPipeline({
        stage: "pruned",
        baselineAccuracy: data.baseline_accuracy,
        candidates: data.candidates,
        results: [],
        bestModel: null,
      });
      setManualProgress(35);
      setPipelineMessage(`${data.candidates.length} pruned candidates generated.`);
      addManualLog(
        "prune",
        `${data.candidates.length} candidates returned by the backend.`,
      );
      data.candidates.forEach((candidate: PipelineCandidate) =>
        addManualLog(
          "prune",
          `${Math.round(candidate.pruning * 100)}% pruning → ${candidate.parameters.toLocaleString()} parameters · ${candidate.model_size_mb.toFixed(2)} MB.`,
        ),
      );
    } catch (err) {
      setPipeline((current) => ({ ...current, stage: "idle" }));
      addManualLog(
        "prune",
        `Request failed: ${err instanceof Error ? err.message : "unknown error"}.`,
      );
      setError(err instanceof Error ? err.message : "Pruning failed.");
      setManualProgress(100);
      setPipelineMessage("Pruning failed.");
    } finally {
      setPipelineBusy(false);
      setRequestStartedAt(null);
    }
  }

  async function runPipelineFinetuning() {
    try {
      setActiveSource("manual");
      setRequestStartedAt(Date.now());
      setElapsedTick(0);
      setManualProgress(45);
      setPipelineBusy(true);
      setError(null);
      setPipelineMessage("Fine-tuning pruned candidates...");
      setPipeline((current) => ({ ...current, stage: "finetuning" }));
      addManualLog(
        "finetune",
        `Fine-tuning request submitted for ${pipeline.candidates.length} candidates.`,
      );
      addManualLog(
        "finetune",
        "The backend updates candidate weights; this stage returns when all candidates finish.",
      );

      const response = await fetch(`${API_BASE_URL}/pipeline/finetune`, {
        method: "POST",
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Fine-tuning failed.");
      }

      setPipeline((current) => ({ ...current, stage: "finetuned" }));
      setManualProgress(68);
      setPipelineMessage(
        `${data.candidate_count} candidates fine-tuned successfully.`,
      );
      addManualLog(
        "finetune",
        `${data.candidate_count} candidates reported fine-tuned by the backend.`,
      );
    } catch (err) {
      setPipeline((current) => ({ ...current, stage: "pruned" }));
      addManualLog(
        "finetune",
        `Request failed: ${err instanceof Error ? err.message : "unknown error"}.`,
      );
      setError(err instanceof Error ? err.message : "Fine-tuning failed.");
      setManualProgress(100);
      setPipelineMessage("Fine-tuning failed.");
    } finally {
      setPipelineBusy(false);
      setRequestStartedAt(null);
    }
  }

  async function runPipelineEvaluation() {
    try {
      setActiveSource("manual");
      setRequestStartedAt(Date.now());
      setElapsedTick(0);
      setManualProgress(75);
      setPipelineBusy(true);
      setError(null);
      setPipelineMessage("Evaluating fine-tuned candidates...");
      setPipeline((current) => ({ ...current, stage: "evaluation" }));
      addManualLog(
        "evaluate",
        `Evaluation request submitted for ${pipeline.candidates.length} candidates.`,
      );
      addManualLog(
        "evaluate",
        "Waiting for held-out evaluation metrics from the backend.",
      );

      const response = await fetch(`${API_BASE_URL}/pipeline/evaluate`, {
        method: "POST",
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Evaluation failed.");
      }

      setPipeline((current) => ({
        ...current,
        stage: "evaluated",
        baselineAccuracy: data.baseline_accuracy,
        results: data.results,
      }));

      const bestAccuracy = Math.max(
        ...data.results.map((result: PipelineResult) => result.accuracy),
      );
      setManualProgress(88);
      setPipelineMessage(
        `Evaluation complete. Best observed accuracy: ${bestAccuracy.toFixed(2)}%.`,
      );
      addManualLog(
        "evaluate",
        `Received ${data.results.length} evaluated candidates; best accuracy ${bestAccuracy.toFixed(2)}%.`,
      );
      data.results.forEach((result: PipelineResult) =>
        addManualLog(
          "evaluate",
          `${Math.round(result.pruning * 100)}% pruning → ${result.accuracy.toFixed(2)}% accuracy · ${result.parameters.toLocaleString()} parameters.`,
        ),
      );
    } catch (err) {
      setPipeline((current) => ({ ...current, stage: "finetuned" }));
      addManualLog(
        "evaluate",
        `Request failed: ${err instanceof Error ? err.message : "unknown error"}.`,
      );
      setError(err instanceof Error ? err.message : "Evaluation failed.");
      setManualProgress(100);
      setPipelineMessage("Evaluation failed.");
    } finally {
      setPipelineBusy(false);
      setRequestStartedAt(null);
    }
  }

  async function runPipelineSelection() {
    try {
      setActiveSource("manual");
      setRequestStartedAt(Date.now());
      setElapsedTick(0);
      setManualProgress(92);
      setPipelineBusy(true);
      setError(null);
      setPipelineMessage("Selecting the smallest valid model...");
      setPipeline((current) => ({ ...current, stage: "selection" }));
      addManualLog(
        "select",
        `Selection request submitted; maximum accuracy loss ${maxAccuracyLoss} percentage points.`,
      );
      addManualLog(
        "select",
        `Comparing ${pipeline.results.length} evaluated candidates against the configured constraint.`,
      );

      const loss = parseMaxLoss();
      const response = await fetch(`${API_BASE_URL}/pipeline/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_accuracy_loss: loss }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Selection failed.");
      }

      setPipeline((current) => ({
        ...current,
        stage: "selected",
        bestModel: data.best_model,
      }));
      setManualProgress(100);
      setPipelineMessage(
        `${Math.round(data.best_model.pruning * 100)}% pruning selected.`,
      );
      addManualLog(
        "select",
        `Selected ${Math.round(data.best_model.pruning * 100)}% pruning: ${data.best_model.parameters.toLocaleString()} parameters, ${data.best_model.accuracy.toFixed(2)}% accuracy.`,
      );
      addManualLog(
        "select",
        `Constraint applied: accuracy change ${data.best_model.accuracy_change.toFixed(2)} percentage points; maximum loss ${loss.toFixed(2)} pp.`,
      );
    } catch (err) {
      setPipeline((current) => ({ ...current, stage: "evaluated" }));
      addManualLog(
        "select",
        `Request failed: ${err instanceof Error ? err.message : "unknown error"}.`,
      );
      setError(err instanceof Error ? err.message : "Selection failed.");
      setManualProgress(100);
      setPipelineMessage("Selection failed.");
    } finally {
      setPipelineBusy(false);
      setRequestStartedAt(null);
    }
  }

  const canFineTune =
    pipeline.stage === "pruned" ||
    pipeline.stage === "finetuned" ||
    pipeline.stage === "evaluated" ||
    pipeline.stage === "selected";
  const canEvaluate =
    pipeline.stage === "finetuned" ||
    pipeline.stage === "evaluated" ||
    pipeline.stage === "selected";
  const canSelect =
    pipeline.stage === "evaluated" || pipeline.stage === "selected";

  /* ------------------------------------------------------------------
   * Unified evidence set
   * ------------------------------------------------------------------ */
  const activeResults = useMemo<CandidateResult[]>(() => {
    if (activeSource === "manual" && pipeline.results.length > 0) {
      return pipeline.results;
    }
    if (optimization?.results) return optimization.results;
    if (results.length > 0) return results;
    return pipeline.results;
  }, [activeSource, pipeline.results, optimization, results]);

  const candidateResults = useMemo(
    () => activeResults.filter((result) => result.pruning > 0),
    [activeResults],
  );

  const experimentResults = useMemo<CandidateResult[]>(() => {
    const baseline = baselineRow
      ? [baselineRow]
      : baselineAccuracy !== null && baselineParameters !== null
        ? [
            {
              pruning: 0,
              hidden1_size: baselineArchitecture[1] ?? 256,
              hidden2_size: baselineArchitecture[2] ?? 128,
              parameters: baselineParameters,
              weight_parameters: analysis?.baseline.weight_parameters ?? profile?.weight_parameters ?? 0,
              nonzero_parameters:
                analysis?.baseline.nonzero_parameters ??
                profile?.nonzero_weights ??
                0,
              model_size_mb: baselineCheckpointSize ?? 0,
              accuracy: baselineAccuracy,
              accuracy_change: 0,
              latency_ms: baselineLatency ?? 0,
            } satisfies CandidateResult,
          ]
        : [];

    return [...baseline, ...candidateResults];
  }, [
    baselineRow,
    baselineAccuracy,
    baselineParameters,
    baselineArchitecture,
    analysis,
    profile,
    baselineCheckpointSize,
    baselineLatency,
    candidateResults,
  ]);

  const selectedBeforeAccuracy =
    selectedModel?.before_finetuning_accuracy ??
    (analysis && selectedModel && analysis.level === selectedModel.pruning
      ? analysis.pruned?.before_finetuning_accuracy
      : undefined) ??
    null;

  const selectedHistory = selectedModel?.fine_tuning_history;
  const selectedReduction = parameterReduction(
    baselineParameters,
    selectedModel?.parameters ?? null,
  );
  const selectedSizeReduction = sizeReduction(
    baselineCheckpointSize,
    selectedModel?.model_size_mb ?? null,
  );
  const selectedDelta =
    baselineAccuracy === null || selectedModel === null
      ? null
      : selectedModel.accuracy - baselineAccuracy;
  const selectedLoss =
    selectedDelta === null ? null : Math.max(0, -selectedDelta);

  const sourceLabel =
    activeSource === "manual" && pipeline.results.length > 0
      ? "manual pipeline · current session"
      : optimization
        ? "one-click optimization · current run"
        : results.length > 0
          ? "stored structured_pruning_results.csv"
          : "no stored measurements yet";

  const parameterFormula =
    baselineParameters === null
      ? "784·h₁ + h₁ + h₁·h₂ + h₂ + h₂·10 + 10"
      : `${baselineArchitecture[0]}×${baselineArchitecture[1]} + ${
          baselineArchitecture[1]
        } + ${baselineArchitecture[1]}×${baselineArchitecture[2]} + ${
          baselineArchitecture[2]
        } + ${baselineArchitecture[2]}×${baselineArchitecture[3]} + ${
          baselineArchitecture[3]
        } = ${formatNumber(baselineParameters)}`;

  return (
    <div className="site">
      <ChapterNav health={health} loading={loading} />

      <main>
        <Hero
          baselineArchitecture={baselineArchitecture}
          selectedArchitecture={selectedArchitecture}
          baselineAccuracy={baselineAccuracy}
          selectedAccuracy={selectedModel?.accuracy ?? null}
          baselineParameters={baselineParameters}
          selectedParameters={selectedModel?.parameters ?? null}
          selectedPruning={selectedModel?.pruning ?? null}
        />

        {error ? (
          <div className="error-banner" role="alert">
            <span className="eyebrow">ERROR</span>
            <strong>{error}</strong>
            <button type="button" onClick={() => setError(null)}>
              dismiss
            </button>
          </div>
        ) : null}

        <Section
          id="problem"
          number="01"
          eyebrow="THE TASK"
          title={
            <>
              A network can be bigger
              <br />
              than the task requires.
            </>
          }
          lead="MNIST asks a simple question: which digit is in this 28 × 28 image? The baseline network answers it well, but it carries 235,146 trainable parameters. This project tests whether a smaller dense network can answer the same question nearly as well."
        >
          <div className="problem-grid">
            <div className="prose">
              <p>
                A parameter is a number the model learns during training. In a
                fully connected layer, every input is multiplied by a weight and
                added to a bias. The parameter count is therefore a structural
                property of the architecture, not a direct measure of how much
                “knowledge” the network contains.
              </p>
              <p>
                More parameters can mean more storage and more arithmetic. But
                parameter count, checkpoint size, inference latency, and test
                accuracy are four different measurements. A smaller model is not
                automatically faster, and a larger model is not automatically
                more accurate.
              </p>
              <p>
                The baseline flattens each image into{" "}
                <strong>784 input values</strong>, passes them through two ReLU
                hidden layers, and produces <strong>10 class scores</strong>. The
                question is what happens if we remove whole hidden neurons.
              </p>
            </div>
            <div className="problem-stats">
              <Stat
                label="Baseline parameters"
                value={formatNumber(baselineParameters)}
                detail="weights + biases"
                tone="teal"
              />
              <Stat
                label="Checkpoint size"
                value={formatMegabytes(baselineCheckpointSize)}
                detail="serialized baseline_mlp.pth"
                tone="neutral"
              />
              <Stat
                label="Test accuracy"
                value={formatPercent(baselineAccuracy)}
                detail="10,000 held-out images"
                tone="green"
              />
              <Stat
                label="Inference latency"
                value={formatMilliseconds(baselineLatency)}
                detail="single-sample, measured at runtime"
                tone="amber"
              />
            </div>
          </div>

          <div className="parameter-calc">
            <span className="eyebrow">WHERE THE PARAMETER COUNT COMES FROM</span>
            <Formula
              expression={parameterFormula}
              explanation="Each linear layer contributes weight matrix entries plus one bias per output unit. The input and output dimensions stay fixed; structured pruning changes h₁ and h₂."
            />
          </div>

          <div className="concept-grid">
            <div className="concept-card">
              <span className="eyebrow">PARAMETERS</span>
              <strong>A structural count.</strong>
              <p>
                The number of learned weights and biases in the architecture.
                This is what the pruning experiment directly reduces.
              </p>
            </div>
            <div className="concept-card">
              <span className="eyebrow">STORAGE</span>
              <strong>A serialized size.</strong>
              <p>
                The bytes needed to store the checkpoint. It depends on dtype and
                format, so it is not identical to the dense parameter estimate.
              </p>
            </div>
            <div className="concept-card">
              <span className="eyebrow">LATENCY</span>
              <strong>A runtime measurement.</strong>
              <p>
                The time for one forward pass on the current device. It is
                hardware- and runtime-dependent, not a guaranteed linear function
                of parameter count.
              </p>
            </div>
            <div className="concept-card">
              <span className="eyebrow">ACCURACY</span>
              <strong>A held-out measurement.</strong>
              <p>
                The fraction of test images classified correctly. It answers
                whether the model still works, not how large it is.
              </p>
            </div>
          </div>
        </Section>

        <Section
          id="digit"
          number="02"
          eyebrow="FROM PIXELS TO A PREDICTION"
          title={
            <>
              A digit becomes 784 numbers,
              <br />
              then ten scores.
            </>
          }
          lead="This is one real MNIST test sample, not a drawing. Follow the same values through the baseline and the saved optimized checkpoint. The prediction is the largest output score."
        >
          <DigitSection
            sample={sample}
            loading={sampleLoading}
            error={sampleError}
            onPrevious={() =>
              setSampleIndex((current) => (current - 1 + 10000) % 10000)
            }
            onNext={() => setSampleIndex((current) => (current + 1) % 10000)}
            onRandom={() => setSampleIndex(Math.floor(Math.random() * 10000))}
          />
          {analysisError ? (
            <DataNote tone="warning">
              The analysis endpoint could not be reached: {analysisError}. The
              sample and architecture sections use whatever checkpoint data is
              available.
            </DataNote>
          ) : null}
        </Section>

        <Section
          id="neuron"
          number="03"
          eyebrow="INSIDE ONE NEURON"
          title={
            <>
              A neuron is a weighted sum
              <br />
              followed by a ReLU.
            </>
          }
          lead="Weights decide how strongly each input matters. The bias shifts the result. ReLU keeps positive values and clamps negative values to zero. The example below uses the actual weights of the most active hidden-1 neuron for the current digit."
        >
          <NeuronSection sample={sample} />
        </Section>

        <Section
          id="architecture"
          number="04"
          eyebrow="THE ARCHITECTURE"
          title={
            <>
              Every dot is a real unit.
              <br />
              The selected network is narrower.
            </>
          }
          lead="The baseline has 256 then 128 hidden neurons. The selected checkpoint has fewer. This diagram renders each unit as a dot and marks the baseline neurons that the pruning rule removes at the inspected level."
        >
          <ModelExplorer
            baseline={baselineWeights}
            optimized={optimizedWeights}
            baselineParameters={baselineParameters}
            optimizedParameters={selectedModel?.parameters ?? null}
            baselineAccuracy={baselineAccuracy}
            optimizedAccuracy={selectedModel?.accuracy ?? null}
            sample={sample}
            pruning={pruningInfo?.layers ?? analysis?.layers ?? null}
          />
          {optimizedWeightsNote ? (
            <DataNote tone="warning">{optimizedWeightsNote}</DataNote>
          ) : null}
          <div className="architecture-foot">
            <Callout title="Read the diagram carefully" tone="teal">
              The dot counts are exact. The lines between layers are a 5×5
              sample of the actual dense weight matrix, chosen only for
              readability. Hover or focus a sampled node or line to inspect a
              real checkpoint value.
            </Callout>
            <div className="architecture-equation">
              <span className="eyebrow">PARAMETER CHANGE</span>
              <strong>
                {formatNumber(baselineParameters)} →{" "}
                {formatNumber(selectedModel?.parameters ?? null)}
              </strong>
              <span>
                {selectedReduction === null
                  ? "run an optimization to measure the change"
                  : `${selectedReduction.toFixed(1)}% fewer trainable parameters`}
              </span>
            </div>
          </div>
        </Section>

        <Section
          id="pruning"
          number="05"
          eyebrow="STRUCTURED PRUNING"
          title={
            <>
              Score the neurons.
              <br />
              Keep the top fraction. Rebuild.
            </>
          }
          lead="The implementation applies the same pruning fraction to both hidden layers. It ranks neurons by the sum of absolute incoming weights, keeps the highest-scoring fraction, and reconstructs the adjacent linear layers around the surviving indices."
        >
          <PruningSection
            analysis={analysis}
            pruningInfo={pruningInfo}
            loading={pruningLoading || analysisLoading}
            level={activePruningLevel}
            availableLevels={availableLevels}
            onLevelChange={setSelectedLevel}
            baselineParameters={baselineParameters}
            results={candidateResults}
            maxAccuracyLoss={maxLossSafe}
          />
        </Section>

        <Section
          id="fine-tuning"
          number="06"
          eyebrow="FINE-TUNING"
          title={
            <>
              Removing neurons changes the function.
              <br />
              Training tries to recover it.
            </>
          }
          lead="Fine-tuning is not a reset. It continues training the surviving weights with cross-entropy loss, Adam, and a small learning rate. The before and after test accuracies show whether recovery actually happened."
        >
          <FineTuningSection
            selected={selectedModel}
            baselineAccuracy={baselineAccuracy}
            selectedBeforeAccuracy={selectedBeforeAccuracy}
            results={candidateResults}
          />
          {selectedHistory && selectedHistory.length > 0 ? (
            <DataNote>
              Per-epoch training history is available for this run. The loss and
              training accuracy above are measured on the training split; the
              final candidate accuracy is measured on the held-out test split.
            </DataNote>
          ) : null}
        </Section>

        <Section
          id="selection"
          number="07"
          eyebrow="EVALUATION AND SELECTION"
          title={
            <>
              The smallest model
              <br />
              is not automatically the winner.
            </>
          }
          lead="A candidate must first satisfy the accuracy-loss constraint. Only then does the optimizer prefer fewer parameters. This section applies that rule to the measured candidates."
        >
          <SelectionSection
            results={candidateResults}
            baselineAccuracy={baselineAccuracy}
            baselineParameters={baselineParameters}
            selected={selectedModel}
            maxAccuracyLoss={maxLossSafe}
            downloadUrl={`${API_BASE_URL}/model/optimized`}
          />
        </Section>

        <Section
          id="evidence"
          number="08"
          eyebrow="THE EXPERIMENT TABLE"
          title={
            <>
              Every candidate,
              <br />
              in one place.
            </>
          }
          lead="This table is the experimental evidence behind the narrative. The selected row is highlighted. A candidate marked rejected exceeded the configured accuracy-loss limit."
        >
          <div className="evidence-summary">
            <Stat
              label="Baseline"
              value={formatPercent(baselineAccuracy)}
              detail={`${formatNumber(baselineParameters)} parameters`}
              tone="teal"
            />
            <Stat
              label="Selected"
              value={formatPercent(selectedModel?.accuracy ?? null)}
              detail={`${formatNumber(selectedModel?.parameters ?? null)} parameters`}
              tone="green"
            />
            <Stat
              label="Parameter reduction"
              value={
                selectedReduction === null
                  ? "—"
                  : `−${selectedReduction.toFixed(1)}%`
              }
              detail={
                selectedSizeReduction === null
                  ? "checkpoint size change unavailable"
                  : `checkpoint size ${selectedSizeReduction.toFixed(1)}% smaller`
              }
              tone="amber"
            />
            <Stat
              label="Accuracy loss"
              value={
                selectedLoss === null ? "—" : `${selectedLoss.toFixed(2)} pp`
              }
              detail={`limit ${maxLossSafe.toFixed(2)} pp`}
              tone="rust"
            />
          </div>

          <ExperimentTable
            results={experimentResults}
            selected={selectedModel}
            maxAccuracyLoss={maxLossSafe}
            source={sourceLabel}
          />

          <div className="calculation-grid">
            <div className="calculation-card">
              <span className="eyebrow">PARAMETER REDUCTION</span>
              <Formula
                expression="(P_baseline − P_optimized) / P_baseline × 100"
                explanation={
                  baselineParameters === null || selectedModel === null
                    ? "The calculation becomes available when both parameter counts are known."
                    : `(${formatNumber(baselineParameters)} − ${formatNumber(
                        selectedModel.parameters,
                      )}) / ${formatNumber(baselineParameters)} × 100 = ${
                        selectedReduction === null
                          ? "—"
                          : `${selectedReduction.toFixed(2)}%`
                      }`
                }
              />
            </div>
            <div className="calculation-card">
              <span className="eyebrow">ACCURACY CHANGE</span>
              <Formula
                expression="ΔA = A_optimized − A_baseline"
                explanation={
                  selectedDelta === null
                    ? "The calculation becomes available when both accuracies are known."
                    : `${formatPercent(selectedModel?.accuracy ?? null)} − ${formatPercent(
                        baselineAccuracy,
                      )} = ${formatSignedPP(selectedDelta)}`
                }
              />
            </div>
            <div className="calculation-card">
              <span className="eyebrow">ACCURACY LOSS</span>
              <Formula
                expression="L_A = max(0, A_baseline − A_optimized)"
                explanation={
                  selectedLoss === null
                    ? "The loss is compared with the configured limit in percentage points."
                    : `${selectedLoss.toFixed(2)} pp measured · ${maxLossSafe.toFixed(
                        2,
                      )} pp allowed`
                }
              />
            </div>
          </div>
        </Section>

        <Section
          id="lab"
          number="09"
          eyebrow="RUN THE REAL PIPELINE"
          title={
            <>
              Change the experiment.
              <br />
              Watch the evidence update.
            </>
          }
          lead="The optimization is not simulated in the browser. The one-click flow runs the real PyTorch pipeline. The manual flow exposes the same stages one at a time. Both use the same backend and checkpoints."
        >
          <ExperimentConsole
            profile={profile}
            selected={selectedModel}
            maxAccuracyLoss={maxAccuracyLoss}
            pruningLevels={pruningLevels}
            setPruningLevels={setPruningLevels}
            optimizing={optimizing}
            optimizationJob={optimizationJob}
            optimization={optimization}
            optimizationLogs={optimizationLogs}
            requestStartedAt={requestStartedAt}
            elapsedTick={elapsedTick}
            runOptimization={runOptimization}
            pipeline={pipeline}
            pipelineBusy={pipelineBusy}
            manualProgress={manualProgress}
            pipelineMessage={pipelineMessage}
            manualStageLogs={manualStageLogs}
            canFineTune={canFineTune}
            canEvaluate={canEvaluate}
            canSelect={canSelect}
            runPipelinePruning={runPipelinePruning}
            runPipelineFinetuning={runPipelineFinetuning}
            runPipelineEvaluation={runPipelineEvaluation}
            runPipelineSelection={runPipelineSelection}
            setMaxAccuracyLoss={setMaxAccuracyLoss}
          />
          {analysisLoading ? (
            <DataNote>
              Recomputing the selected level's baseline and before-fine-tuning
              measurements…
            </DataNote>
          ) : null}
        </Section>

        <Conclusion
          baselineAccuracy={baselineAccuracy}
          selectedAccuracy={selectedModel?.accuracy ?? null}
          baselineParameters={baselineParameters}
          selectedParameters={selectedModel?.parameters ?? null}
          selectedPruning={selectedModel?.pruning ?? null}
          maxAccuracyLoss={maxLossSafe}
        />
      </main>

      <Footer />
    </div>
  );
}

export default App;
