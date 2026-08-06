import React, { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAppState } from "../context/AppContext";
import HistoryPanel from "../components/HistoryPanel";
import Navbar from "../components/Navbar";
import ResultDashboard from "../components/ResultDashboard";

function Result() {
  const navigate = useNavigate();
  const { prediction, history, resetWorkflow, domainPalette } = useAppState();

  const isBulk = Array.isArray(prediction);
  const results = useMemo(
    () => (isBulk ? prediction : prediction ? [prediction] : []),
    [isBulk, prediction]
  );

  useEffect(() => {
    if (!prediction || (Array.isArray(prediction) && prediction.length === 0)) {
      navigate("/upload", { replace: true });
    }
  }, [navigate, prediction]);

  if (!prediction || results.length === 0) {
    return null;
  }

  const onUploadAnother = () => {
    resetWorkflow();
    navigate("/upload");
  };

  const onBackHome = () => navigate("/");

  return (
    <PageShell
      history={history}
      isBulk={isBulk}
      results={results}
      domainPalette={domainPalette}
      onUploadAnother={onUploadAnother}
      onBackHome={onBackHome}
    />
  );
}

function PageShell({ history, isBulk, results, domainPalette, onUploadAnother, onBackHome }) {
  return (
    <div className="pageShell resultPage">
      <Navbar />

      <main className="contentGrid contentGrid--result">
        <section className="contentPanel resultPanel resultPanel--wide">
          <div className="panelHeading">
            <span className="eyebrow">Analysis dashboard</span>
            <h1>Intelligent classification &amp; storage report</h1>
            <p>
              Explore prediction confidence, lexical drivers, data-mining operations, and the exact filesystem destination
              produced by the FastAPI service.
            </p>
          </div>

          {isBulk ? (
            <div className="bulkResultStack">
              {results.map((item, index) => {
                const theme = domainPalette[item.prediction] || domainPalette.Unknown;
                return (
                  <div key={`${item.filename}-${item.timestamp}-${index}`} className="bulkResultItem">
                    {results.length > 1 ? (
                      <p className="bulkResultItem__label">
                        File {index + 1} of {results.length}: <strong>{item.filename}</strong>
                      </p>
                    ) : null}
                    <ResultDashboard
                      result={item}
                      theme={theme}
                      onUploadAnother={onUploadAnother}
                      onBackHome={onBackHome}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            <ResultDashboard
              result={results[0]}
              theme={domainPalette[results[0].prediction] || domainPalette.Unknown}
              onUploadAnother={onUploadAnother}
              onBackHome={onBackHome}
            />
          )}
        </section>

        <aside className="sidebarStack">
          <HistoryPanel title="Recent predictions" items={history} emptyText="No predictions yet." />

          <div className="contentPanel infoPanel infoPanel--tip">
            <span className="eyebrow">Tip</span>
            <h2>Re-run uploads</h2>
            <p>
              Each prediction refreshes session storage with the latest confidence, TF-IDF keywords, and reasoning copy
              tailored to the predicted domain.
            </p>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default Result;
