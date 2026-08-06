import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppState } from "../context/AppContext";
import HistoryPanel from "../components/HistoryPanel";
import Loader from "../components/Loader";
import Navbar from "../components/Navbar";
import UploadBox from "../components/UploadBox";

function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const {
    selectedFiles,
    setSelectedFiles,
    loading,
    error,
    setError,
    history,
    validateFile,
    submitDocument,
    formatFileSize,
    setPrediction,
  } = useAppState();

  const clearSelection = () => {
    setSelectedFiles([]);
    setPrediction(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFile = (files) => {
    if (!files || files.length === 0) {
      clearSelection();
      return;
    }

    const fileArray = Array.from(files);

    for (const file of fileArray) {
      const validationMessage = validateFile(file);
      if (validationMessage) {
        setSelectedFiles([]);
        setError(validationMessage);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }
    }

    setError("");
    setSelectedFiles(fileArray);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    handleFile(event.dataTransfer?.files || []);
  };

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError("Please upload a document before starting classification.");
      return;
    }

    try {
      await submitDocument(selectedFiles);
      navigate("/result");
    } catch (uploadError) {
      setError(uploadError?.message || "Unable to upload the file.");
    }
  };

  const fileDetails =
    selectedFiles && selectedFiles.length > 0
      ? {
          name:
            selectedFiles.length === 1
              ? selectedFiles[0].name
              : `${selectedFiles.length} files selected`,
          size: formatFileSize(selectedFiles.reduce((sum, file) => sum + file.size, 0)),
          type:
            selectedFiles.length === 1
              ? selectedFiles[0].name.includes(".")
                ? selectedFiles[0].name.split(".").pop().toUpperCase()
                : selectedFiles[0].type || "Unknown"
              : "Multiple",
        }
      : null;

  return (
    <div className="pageShell uploadPage">
      <Navbar />

      <main className="contentGrid contentGrid--upload">
        <section className="contentPanel uploadPanel">
          <div className="panelHeading">
            <span className="eyebrow">Upload</span>
            <h1>Classify and sort a document in one click</h1>
            <p>
              Drop a file into the workspace, send it to the FastAPI backend, and get a prediction
              plus a stored folder path back.
            </p>
          </div>

          <UploadBox
            fileInputRef={fileInputRef}
            fileDetails={fileDetails}
            isDragging={isDragging}
            loading={loading}
            error={error}
            onBrowse={() => fileInputRef.current?.click()}
            onClear={clearSelection}
            onDrop={handleDrop}
            onDragStateChange={setIsDragging}
            onFileChange={(event) => handleFile(event.target.files)}
            onUpload={handleUpload}
          />

          {loading && <Loader text="Analyzing and Sorting Document..." />}
        </section>

        <aside className="sidebarStack">
          <HistoryPanel title="Recent uploads" items={history} emptyText="No uploads yet." />

          <div className="contentPanel infoPanel">
            <span className="eyebrow">Backend</span>
            <h2>Connected to FastAPI</h2>
            <p>
  The frontend sends a <strong>POST</strong> request to
  <span className="mono">http://127.0.0.1:8001/predict-bulk</span>
  with the file key set to <span className="mono">files</span>.
</p>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default Upload;
