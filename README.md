# DocSort-AI

DocSort-AI is a document classification and auto-sorting app. It accepts PDF, DOCX, and TXT files, predicts a document domain with a TF-IDF + Naive Bayes pipeline, and returns confidence, keywords, and storage metadata in a React dashboard.

## What It Does

- Upload one or more documents from the browser.
- Classify each document into a domain.
- Show the prediction, confidence, TF-IDF terms, and manual keyword evidence.
- Sort files into local folders on disk.
- Keep recent predictions in browser storage.

## Repository Layout

- `src/` - React frontend used for the main UI.
- `public/` - Static frontend assets.
- `backend/multi_domain_doc_classifier/` - Core classifier package, training code, data artifacts, and a Streamlit demo.
- `backend/main.py` - Legacy FastAPI sorter API used by the current React workflow.
- `sorted_documents/` - Example output folders.
- `build/` - Production frontend build output.

## Frontend

The React app includes:

- A landing page with feature highlights and an explanation of the ML pipeline.
- An upload page with drag-and-drop support for PDF, DOCX, and TXT files.
- A results page that displays prediction confidence, extracted keywords, and stored file metadata.
- Local history for recent uploads and predictions.

The frontend uses React Router and stores recent state in `sessionStorage` and `localStorage`.

## Backend

There are two FastAPI entry points in the repository:

- `backend/multi_domain_doc_classifier/api/main.py` exposes the current classifier API:
	- `GET /health`
	- `POST /predict_text`
	- `POST /predict_file`
- `backend/main.py` exposes the legacy auto-sorting API used by the current React workflow:
	- `POST /predict`
	- `POST /predict-bulk`
	- `POST /confirm-sort`

The legacy sorter writes files under `~/Desktop/SortedDocuments` and supports manual confirmation when the confidence is not exact.

## Model Pipeline

The classifier package uses:

- format-aware text extraction for PDF, DOCX, and TXT
- text cleaning and stopword removal
- TF-IDF vectorization
- curated keyword matching per domain
- Multinomial Naive Bayes classification

The training script saves the model and vectorizer artifacts to `backend/multi_domain_doc_classifier/models/`.

## Supported Domains

- Finance
- Medical
- Sports
- Technology
- Education is present in the legacy sorter API and frontend UI, but the current classifier package is configured for the four-domain model above.

## Requirements

- Node.js 18+ and npm
- Python 3.10+ recommended
- A working internet connection the first time NLTK downloads its stopword corpus

## Setup

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Install backend dependencies

```bash
cd backend/multi_domain_doc_classifier
pip install -r requirements.txt
```

## Running the App

### Frontend

From the repository root:

```bash
npm start
```

This runs the React app on `http://localhost:3000`.

### Legacy sorter backend used by the upload flow

From the `backend/` directory:

```bash
uvicorn main:app --reload --port 8001
```

This is the backend the current upload page expects at `http://127.0.0.1:8001`.

### Current classifier API

From `backend/multi_domain_doc_classifier/api`:

```bash
uvicorn main:app --reload --app-dir ..
```

This exposes `/health`, `/predict_text`, and `/predict_file`.

### Streamlit demo

From `backend/multi_domain_doc_classifier`:

```bash
streamlit run streamlit_app.py
```

## Training the Model

The training script lives at `backend/multi_domain_doc_classifier/scripts/train.py`.

```bash
cd backend/multi_domain_doc_classifier/scripts
python train.py
```

Use `--quick` for a smaller development run.

## Notes

- The frontend currently posts uploads to `POST /predict-bulk` and confirmation data to `POST /confirm-sort`.
- The newer FastAPI service in `backend/multi_domain_doc_classifier/api/main.py` is better for direct text or single-file classification, but it does not implement the legacy folder-sorting endpoints.
- NLTK downloads the English stopword corpus at runtime in the legacy backend.

## Troubleshooting

- If the frontend cannot reach the backend, make sure the FastAPI server is running on port `8001`.
- If you see missing NLTK data errors, rerun the backend once with internet access so the stopwords corpus can download.
- If a file is rejected, verify that it is a `.pdf`, `.docx`, or `.txt` document.

## License

No explicit license file is present in the repository.
