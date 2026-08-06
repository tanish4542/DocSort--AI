import React from "react";

const floatingCards = [
  { label: "Domains", value: "4" },
  { label: "Formats", value: "PDF / DOCX / TXT" },
  { label: "Sorting", value: "Local folders" },
];

const metrics = [
  { label: "Backend", value: "http://127.0.0.1:8001" },
  { label: "Prediction", value: "FastAPI / FormData" },
  { label: "Output", value: "Filename + folder path" },
];

function HeroSection({ onStart, onLearnMore }) {
  return (
    <section className="heroSection">
      <div className="heroSection__copy cardRise">
        <span className="eyebrow">Productivity tool</span>
        <h1>Multi Domain Document Classification System</h1>
        <p>
          Upload documents and automatically classify and organize them into domain-specific folders.
        </p>

        <div className="heroActions">
          <button className="button button--primary" type="button" onClick={onStart}>
            Get Started
          </button>
          <button className="button button--secondary" type="button" onClick={onLearnMore}>
            Learn More
          </button>
        </div>

        <div className="heroMetrics">
          {metrics.map((metric) => (
            <div key={metric.label} className="heroMetric">
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="heroSection__visual cardRise">
        <div className="heroVisualBackground" aria-hidden="true" />
        <div className="heroVisualGlow heroVisualGlow--one" aria-hidden="true" />
        <div className="heroVisualGlow heroVisualGlow--two" aria-hidden="true" />
        <div className="heroVisualCard heroVisualCard--main">
          <span className="heroVisualCard__label">Live document pipeline</span>
          <strong>Upload → Predict → Sort</strong>
          <span>FastAPI receives the file, classifies the domain, and returns the storage path.</span>
        </div>

        {floatingCards.map((item, index) => (
          <div key={item.label} className={`heroVisualCard heroVisualCard--float heroVisualCard--${index + 1}`}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

export default HeroSection;
