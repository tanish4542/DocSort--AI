import React from "react";
import { useAppState } from "../context/AppContext";

function formatHistoryTime(timestamp) {
  if (!timestamp) {
    return "Just now";
  }

  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function HistoryPanel({ title, items, emptyText }) {
  const { domainPalette } = useAppState();

  return (
    <section className="contentPanel historyPanel">
      <div className="panelHeading panelHeading--compact">
        <span className="eyebrow">History</span>
        <h2>{title}</h2>
      </div>

      {items.length === 0 ? (
        <div className="emptyState">{emptyText}</div>
      ) : (
        <div className="historyList">
          {items.map((item) => {
            const theme = domainPalette[item.prediction] || domainPalette.Unknown;

            return (
              <article key={`${item.timestamp}-${item.filename}`} className="historyItem cardRise">
                <div>
                  <strong>{item.filename}</strong>
                  <span>{item.prediction}</span>
                  <span className="historyItem__time">{formatHistoryTime(item.timestamp)}</span>
                </div>
                <span
                  className="historyItem__tag"
                  style={{ background: theme.soft, borderColor: theme.border, color: theme.color }}
                >
                  {theme.label}
                </span>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default HistoryPanel;
