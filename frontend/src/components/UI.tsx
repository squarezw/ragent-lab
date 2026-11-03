import {
  AlertCircle,
  ArrowUpRight,
  FlaskConical,
  LoaderCircle,
} from "lucide-react";
import type { ReactNode } from "react";
export function Brand() {
  return (
    <div className="brand">
      <span className="brand-icon">
        <FlaskConical size={24} />
      </span>
      <span>
        ragent<span className="brand-light"> / lab</span>
        <small>THE EXPERIMENT WORKSPACE</small>
      </span>
    </div>
  );
}
export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </header>
  );
}
export function Panel({
  title,
  meta,
  children,
  className = "",
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <h2>{title}</h2>
        {meta}
      </div>
      {children}
    </section>
  );
}
export function ErrorMessage({ message }: { message: string }) {
  return message ? (
    <div className="error-message" role="alert">
      <AlertCircle size={17} />
      <span>{message}</span>
    </div>
  ) : null;
}
export function RunButton({
  busy,
  label = "运行实验",
}: {
  busy: boolean;
  label?: string;
}) {
  return (
    <button type="submit" className="button primary run-button" disabled={busy}>
      {busy ? (
        <LoaderCircle className="spin" size={17} />
      ) : (
        <FlaskConical size={17} />
      )}{" "}
      {busy ? "实验运行中…" : label}
      <span className="button-shortcut">↵</span>
    </button>
  );
}
export function EmptyState({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="empty-state">
      <div className="empty-illustration">
        {icon}
        <span />
        <i />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <span className="empty-note">每一个好结果，都从一次实验开始</span>
    </div>
  );
}
export function ExternalLink({
  href,
  children,
}: {
  href: string;
  children: ReactNode;
}) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-link">
      {children}
      <ArrowUpRight size={14} />
    </a>
  );
}
export function Metrics({
  items,
}: {
  items: { label: string; value: string | number; unit?: string }[];
}) {
  return (
    <div className="metrics">
      {items.map((item) => (
        <div key={item.label}>
          <span>{item.label}</span>
          <strong>
            {item.value}
            <small>{item.unit}</small>
          </strong>
        </div>
      ))}
    </div>
  );
}
