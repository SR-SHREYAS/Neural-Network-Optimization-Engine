import type { ReactNode } from "react";
import { useReveal } from "../hooks/useReveal";

export function Section({
  id,
  number,
  eyebrow,
  title,
  lead,
  children,
  className = "",
}: {
  id: string;
  number: string;
  eyebrow: string;
  title: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const { ref, visible } = useReveal<HTMLElement>();

  return (
    <section
      id={id}
      ref={ref}
      className={`chapter ${visible ? "is-visible" : ""} ${className}`.trim()}
      aria-labelledby={`${id}-title`}
    >
      <div className="chapter-marker" aria-hidden="true">
        <span>{number}</span>
        <i />
      </div>
      <div className="chapter-body">
        <header className="chapter-head">
          <span className="eyebrow">{eyebrow}</span>
          <h2 id={`${id}-title`}>{title}</h2>
          {lead ? <p className="chapter-lead">{lead}</p> : null}
        </header>
        {children}
      </div>
    </section>
  );
}

export function Stat({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: "neutral" | "teal" | "rust" | "green" | "amber";
}) {
  return (
    <div className={`stat stat-${tone}`}>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {detail ? <span className="stat-detail">{detail}</span> : null}
    </div>
  );
}

export function Callout({
  title,
  children,
  tone = "neutral",
}: {
  title: string;
  children: ReactNode;
  tone?: "neutral" | "teal" | "rust" | "green" | "amber";
}) {
  return (
    <aside className={`callout callout-${tone}`}>
      <span className="callout-title">{title}</span>
      <div>{children}</div>
    </aside>
  );
}

export function Formula({
  expression,
  explanation,
}: {
  expression: string;
  explanation?: ReactNode;
}) {
  return (
    <div className="formula">
      <code>{expression}</code>
      {explanation ? <span>{explanation}</span> : null}
    </div>
  );
}

export function DataNote({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "warning";
}) {
  return <p className={`data-note data-note-${tone}`}>{children}</p>;
}

export function Divider() {
  return <div className="rule" aria-hidden="true" />;
}
