import React, { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import FeatureCards from "../components/FeatureCards";
import HeroSection from "../components/HeroSection";

const howItWorks = [
  {
    step: "01",
    title: "Upload file",
    text: "Drop in a PDF, DOCX, or TXT document. The UI streams it to the FastAPI classifier with zero friction.",
  },
  {
    step: "02",
    title: "Machine learning pipeline",
    text: "Text is extracted, cleaned, tokenized, and de-noised so domain-specific vocabulary can surface reliably.",
  },
  {
    step: "03",
    title: "AI vectorization",
    text: "TF-IDF converts the document into a weighted feature space that highlights informative terms.",
  },
  {
    step: "04",
    title: "Auto sorting",
    text: "LinearSVC predicts the category and the backend moves the file into the matching desktop folder instantly.",
  },
];

const supportedDomains = [
  {
    name: "Technology & Computing",
    icon: "💻",
    tagline: "Software, algorithms, cloud, and computing architecture",
    description: "Source code documentation, system specifications, technical whitepapers, and computing infrastructure benchmarks.",
    color: "#8b5cf6",
    soft: "rgba(139, 92, 246, 0.16)",
    border: "rgba(139, 92, 246, 0.32)",
    sampleTerms: "Software, API, Cloud, Architecture, Hardware",
  },
  {
    name: "Medical Health",
    icon: "🏥",
    tagline: "Clinical diagnoses, pharmacology, and patient care",
    description: "Clinical trial summaries, medical diagnoses, treatment protocols, and pharmaceutical documentation.",
    color: "#0ea5e9",
    soft: "rgba(14, 165, 233, 0.16)",
    border: "rgba(14, 165, 233, 0.32)",
    sampleTerms: "Patient, Clinical, Diagnosis, Therapy, Healthcare",
  },
  {
    name: "Business & Finance",
    icon: "📈",
    tagline: "Corporate finance, investments, and market economics",
    description: "Quarterly balance sheets, equity research, earnings reports, regulatory filings, and macroeconomic updates.",
    color: "#10b981",
    soft: "rgba(16, 185, 129, 0.16)",
    border: "rgba(16, 185, 129, 0.32)",
    sampleTerms: "Revenue, Investment, Market, Equity, Earnings",
  },
  {
    name: "Entertainment",
    icon: "🎬",
    tagline: "Cinema, music, performing arts, and media production",
    description: "Film reviews, concert and festival announcements, screenplay drafts, album releases, and media commentary.",
    color: "#d946ef",
    soft: "rgba(217, 70, 239, 0.16)",
    border: "rgba(217, 70, 239, 0.32)",
    sampleTerms: "Film, Music, Album, Box Office, Festival",
  },
  {
    name: "Sports",
    icon: "⚽",
    tagline: "Athletics, competitive leagues, and tournament coverage",
    description: "Match recap analysis, league standings, player statistics, coaching strategies, and tournament schedules.",
    color: "#f59e0b",
    soft: "rgba(245, 158, 11, 0.16)",
    border: "rgba(245, 158, 11, 0.32)",
    sampleTerms: "Team, Tournament, Match, League, Season",
  },
  {
    name: "Science & Academics",
    icon: "🔬",
    tagline: "Empirical research, academic courseware, syllabi, and lab manuals",
    description: "Peer-reviewed scientific preprints, academic syllabi, lab manuals, course outlines, and research publications.",
    color: "#f43f5e",
    soft: "rgba(244, 63, 94, 0.16)",
    border: "rgba(244, 63, 94, 0.32)",
    sampleTerms: "Syllabus, Lab Manual, Research, Course, Experiment",
  },
];

const modelWorkflowSteps = [
  {
    icon: "📄",
    title: "Document Upload",
    text: "User uploads a PDF, DOCX, or TXT file into the processing pipeline.",
  },
  {
    icon: "📑",
    title: "Text Extraction",
    text: "Format-specific parsers extract raw textual content from the document.",
  },
  {
    icon: "🧼",
    title: "Preprocessing",
    text: "Whitespace normalization, control character removal, and lowercase tokenization prepare the text.",
  },
  {
    icon: "📊",
    title: "TF-IDF Feature Extraction",
    text: "100,000 unigram/bigram features with sublinear term frequency scaling construct a sparse numerical vector.",
  },
  {
    icon: "🤖",
    title: "Three ML Classifiers",
    text: "All three research models (MultinomialNB, Logistic Regression, LinearSVC) evaluate the document vector.",
  },
  {
    icon: "⚖️",
    title: "Model Comparison",
    text: "Dynamic predictions and confidence/decision scores are compared alongside held-out test benchmarks.",
  },
  {
    icon: "🏆",
    title: "LinearSVC Selection",
    text: "LinearSVC is selected as the final classifier because it achieved the highest test Macro F1 (94.49%).",
  },
  {
    icon: "🎯",
    title: "Final Prediction",
    text: "The winning category, decision margin, uncertainty tier, and top lexical features are finalized.",
  },
  {
    icon: "📁",
    title: "Document Sorting",
    text: "The document is automatically moved into its designated domain folder on disk (or staged if ambiguous).",
  },
];

function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const featuresRef = useRef(null);

  useEffect(() => {
    if (location.hash === "#about") {
      const el = document.getElementById("about");
      if (el) {
        requestAnimationFrame(() => el.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    }
  }, [location.hash, location.pathname]);

  const handleLearnMore = () => {
    featuresRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="pageShell homePage">
      <Navbar />

      <main>
        <HeroSection onStart={() => navigate("/upload")} onLearnMore={handleLearnMore} />

        {/* SIX RESEARCH DOMAINS SECTION */}
        <section className="sectionBlock sectionBlock--domains" id="domains">
          <div className="sectionHeading">
            <span className="eyebrow">Supported Domains</span>
            <h2>Six Research Categories</h2>
            <p>
              DocSort AI is powered by a high-dimensional LinearSVC classifier trained and evaluated across
              all six research domains with 94.50% test accuracy.
            </p>
          </div>

          <div className="domainGrid">
            {supportedDomains.map((domain) => (
              <article
                key={domain.name}
                className="domainCard cardRise"
                style={{
                  borderColor: domain.border,
                  background: `linear-gradient(145deg, var(--surface) 0%, ${domain.soft} 100%)`,
                }}
              >
                <div className="domainCard__header">
                  <span
                    className="domainCard__icon"
                    aria-hidden="true"
                    style={{ background: domain.soft, borderColor: domain.border }}
                  >
                    {domain.icon}
                  </span>
                  <span
                    className="domainCard__badge"
                    style={{ color: domain.color, borderColor: domain.border, background: domain.soft }}
                  >
                    {domain.name}
                  </span>
                </div>
                <h3>{domain.name}</h3>
                <p className="domainCard__tagline">{domain.tagline}</p>
                <p className="domainCard__desc">{domain.description}</p>
                <div className="domainCard__terms">
                  <small>Key Signals: {domain.sampleTerms}</small>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section ref={featuresRef} className="sectionBlock sectionBlock--features" id="features">
          <div className="sectionHeading">
            <span className="eyebrow">Features</span>
            <h2>Designed to be a multidomain productivity tool</h2>
            <p>
              Intelligent automation, a transparent ML workflow, and presentation-ready visuals help you explain the
              cross-domain document machine learning pipeline in seconds.
            </p>
          </div>

          <FeatureCards />
        </section>

        <section className="sectionBlock sectionBlock--process">
          <div className="sectionHeading">
            <span className="eyebrow">How it works</span>
            <h2>From upload to sorted document — built for demos</h2>
            <p>
              Each stage mirrors real machine learning practice: ingest, preprocess, vectorize, classify, and operationalize
              the outcome on disk.
            </p>
          </div>

          <div className="processGrid">
            {howItWorks.map((item) => (
              <article key={item.step} className="processCard cardRise">
                <span className="processStep">{item.step}</span>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="sectionBlock sectionBlock--mlWorkflow">
          <div className="sectionHeading">
            <span className="eyebrow">Machine learning</span>
            <h2>How the Model Works</h2>
            <p>
              Follow the animated pipeline from raw bytes to automated folder placement — ideal for vivas, portfolios,
              and stakeholder walkthroughs.
            </p>
          </div>

          <div className="mlWorkflow">
            <div className="mlWorkflow__rail" aria-hidden="true" />
            <div className="mlWorkflow__steps">
              {modelWorkflowSteps.map((step, index) => (
                <article key={step.title} className="mlWorkflowStep cardRise" style={{ animationDelay: `${index * 0.05}s` }}>
                  <div className="mlWorkflowStep__iconWrap">
                    <span className="mlWorkflowStep__icon" aria-hidden="true">
                      {step.icon}
                    </span>
                    <span className="mlWorkflowStep__index">{String(index + 1).padStart(2, "0")}</span>
                  </div>
                  <div className="mlWorkflowStep__body">
                    <h3>{step.title}</h3>
                    <p>{step.text}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="sectionBlock sectionBlock--modelDetails">
          <div className="modelDetailsCard cardRise">
            <div className="modelDetailsCard__header">
              <span className="eyebrow">Model details</span>
              <h2>What happens under the hood</h2>
            </div>
            <ul className="modelDetailsCard__list">
              <li>
                The system uses <strong>TF-IDF vectorization</strong> with 100,000 unigram and bigram features to convert
                textual content into numerical feature vectors that machine learning models consume.
              </li>
              <li>
                <strong>Three individual classifiers</strong> were evaluated on the held-out research test set:
                Multinomial Naive Bayes (91.17% Macro F1), Logistic Regression (94.05% Macro F1), and LinearSVC (94.49% Macro F1).
              </li>
              <li>
                <strong>Ensemble experiments</strong> (Soft Voting at 94.15% Macro F1 and Stacking at 94.47% Macro F1)
                did not outperform the standalone LinearSVC baseline on the held-out test set.
              </li>
              <li>
                A <strong>LinearSVC classifier (C=0.5)</strong> evaluates maximum-margin decision hyperplanes learned
                during training across all six domains and was selected as the final production inference model.
              </li>
              <li>
                The classifier predicts the <strong>most decisive domain</strong> by comparing hyperplane distances,
                producing an exact label plus an operational decision margin for uncertainty estimation.
              </li>
              <li>
                After prediction, the <strong>FastAPI backend automatically sorts</strong> the uploaded file into the
                matching local desktop folder so the UI can display a concrete storage path.
              </li>
            </ul>
          </div>
        </section>

        <section className="sectionBlock sectionBlock--about" id="about">
          <div className="aboutPanel cardRise">
            <div className="sectionHeading">
              <span className="eyebrow">About</span>
              <h2>Real-world usefulness &amp; intelligent automation</h2>
            </div>
            <div className="aboutPanel__grid">
              <div>
                <h3>Why it matters</h3>
                <p>
                  Knowledge workers routinely drown in mixed-format inboxes. This project demonstrates how lightweight
                  NLP plus disciplined folder hygiene can scale personal organization without manual tagging.
                </p>
              </div>
              <div>
                <h3>Machine learning lens</h3>
                <p>
                  Every upload exercises a full machine learning stack: selection, cleaning, transformation, modeling, and
                  evaluation through margin and lexical diagnostics surfaced on the results dashboard.
                </p>
              </div>
              <div>
                <h3>Presentation ready</h3>
                <p>
                  The interface keeps the pastel DocuSense palette, soft glass cards, and smooth motion so judges or
                  interviewers immediately read it as a modern AI SaaS experience—not a homework script.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="footerBar">
        <div>
          AI-powered cross-domain document classification and auto-sorting system for demos, vivas, and portfolio
          storytelling.
        </div>
        <div>FastAPI backend · TF-IDF + LinearSVC · Local folder organization · Smooth React UI</div>
      </footer>
    </div>
  );
}

export default Home;
