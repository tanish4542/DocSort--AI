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
    title: "Data mining pipeline",
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
    text: "Naive Bayes predicts the domain and the backend moves the file into the matching desktop folder instantly.",
  },
];

const modelWorkflowSteps = [
  {
    icon: "⬆️",
    title: "Document Upload",
    text: "Secure multipart upload delivers the raw file to the inference service.",
  },
  {
    icon: "📑",
    title: "Text Extraction",
    text: "Format-aware parsers pull readable text from PDF, DOCX, or TXT sources.",
  },
  {
    icon: "🧼",
    title: "Text Cleaning",
    text: "Lowercasing and symbol stripping normalize tokens for consistent modeling.",
  },
  {
    icon: "✂️",
    title: "Stopword Removal",
    text: "Frequent function words are removed so signal concentrates on content words.",
  },
  {
    icon: "📊",
    title: "TF-IDF Vectorization",
    text: "Each term receives a score combining local frequency with global rarity.",
  },
  {
    icon: "⚖️",
    title: "Feature Weight Calculation",
    text: "Important stems receive higher weights inside the sparse document vector.",
  },
  {
    icon: "🧠",
    title: "Naive Bayes Classification",
    text: "Class-conditional probabilities rank each domain and select the argmax label.",
  },
  {
    icon: "🎯",
    title: "Domain Prediction",
    text: "The winning category is serialized back to the React dashboard with storage metadata.",
  },
  {
    icon: "📁",
    title: "Automatic Folder Sorting",
    text: "The file is moved into the predicted domain folder under ~/Desktop/SortedDocuments.",
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

        <section ref={featuresRef} className="sectionBlock sectionBlock--features" id="features">
          <div className="sectionHeading">
            <span className="eyebrow">Features</span>
            <h2>Designed to be a multidomain productivity tool</h2>
            <p>
              Intelligent automation, a transparent ML workflow, and presentation-ready visuals help you explain the
              cross-domain document mining pipeline in seconds.
            </p>
          </div>

          <FeatureCards />
        </section>

        <section className="sectionBlock sectionBlock--process">
          <div className="sectionHeading">
            <span className="eyebrow">How it works</span>
            <h2>From upload to sorted document — built for demos</h2>
            <p>
              Each stage mirrors real data mining practice: ingest, preprocess, vectorize, classify, and operationalize
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
                The system uses <strong>TF-IDF vectorization</strong> to convert textual content into numerical feature
                vectors that machine learning models can consume.
              </li>
              <li>
                <strong>TF-IDF assigns higher importance</strong> to meaningful domain-specific words that are frequent
                in the document yet comparatively rare across the broader corpus.
              </li>
              <li>
                A <strong>Naive Bayes classifier</strong> estimates probability distributions of words conditioned on each
                document category learned during training.
              </li>
              <li>
                The classifier predicts the <strong>most probable domain</strong> by combining those likelihoods with
                priors, producing an interpretable label plus a confidence signal from class probabilities.
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
                <h3>Data mining lens</h3>
                <p>
                  Every upload exercises a full mining stack: selection, cleaning, transformation, modeling, and
                  evaluation through confidence and lexical diagnostics surfaced on the results dashboard.
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
        <div>FastAPI backend · TF-IDF + Naive Bayes · Local folder organization · Smooth React UI</div>
      </footer>
    </div>
  );
}

export default Home;
