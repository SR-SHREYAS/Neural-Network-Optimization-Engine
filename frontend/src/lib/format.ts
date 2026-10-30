export const numberFormatter = new Intl.NumberFormat("en-US");

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return numberFormatter.format(value);
}

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(digits)}%`;
}

export function formatSignedPP(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} pp`;
}

export function formatSignedPercent(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

export function formatDecimal(
  value: number | null | undefined,
  digits = 4,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

export function formatMilliseconds(
  value: number | null | undefined,
  digits = 3,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(digits)} ms`;
}

export function formatMegabytes(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(digits)} MB`;
}

export function accuracyLoss(accuracyChange: number): number {
  return Math.max(0, -accuracyChange);
}

export function parameterReduction(
  baselineParameters: number | null | undefined,
  optimizedParameters: number | null | undefined,
): number | null {
  if (
    baselineParameters === null ||
    baselineParameters === undefined ||
    optimizedParameters === null ||
    optimizedParameters === undefined ||
    baselineParameters === 0
  ) {
    return null;
  }
  return ((baselineParameters - optimizedParameters) / baselineParameters) * 100;
}

export function sizeReduction(
  baselineSize: number | null | undefined,
  optimizedSize: number | null | undefined,
): number | null {
  if (
    baselineSize === null ||
    baselineSize === undefined ||
    optimizedSize === null ||
    optimizedSize === undefined ||
    baselineSize === 0
  ) {
    return null;
  }
  return ((baselineSize - optimizedSize) / baselineSize) * 100;
}

export function architectureString(
  architecture: number[] | null | undefined,
): string {
  if (!architecture || architecture.length === 0) {
    return "—";
  }
  return architecture.join(" → ");
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function formatCompact(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return `${value}`;
}

export function classLabel(index: number): string {
  return `${index}`;
}
