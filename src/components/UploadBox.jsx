import React from "react";

function UploadBox({
  fileInputRef,
  fileDetails,
  isDragging,
  loading,
  error,
  onBrowse,
  onClear,
  onDrop,
  onDragStateChange,
  onFileChange,
  onUpload,
}) {
  return (
    <div className="uploadBoxWrap">
      <div
        className={`uploadBox ${isDragging ? "uploadBox--active" : ""}`}
        role="button"
        tabIndex={0}
        onClick={onBrowse}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            onBrowse();
          }
        }}
        onDragEnter={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onDragStateChange(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onDragStateChange(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onDragStateChange(false);
        }}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file" multiple
          accept=".pdf,.docx,.txt,application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          className="uploadBox__input"
          onChange={onFileChange}
        />

        <div className="uploadBox__badge">Drag and drop</div>
        <div className="uploadBox__icon" aria-hidden="true">
          <span />
          <span />
        </div>
        <h2>{fileDetails ? "Document ready" : "Drop your document here"}</h2>
        <p>
          Click to browse, or drag in a PDF, DOCX, or TXT file to start the classification workflow.
        </p>

        <div className="formatRow">
          {["PDF", "DOCX", "TXT"].map((type) => (
            <span key={type} className="formatChip">
              {type}
            </span>
          ))}
        </div>
      </div>

      <div className="uploadMetaArea">
        {fileDetails ? (
          <div className="uploadMetaCard cardRise">
            <div className="uploadMetaCard__header">
              <h3>Selected file</h3>
              <button className="textButton" type="button" onClick={onClear} disabled={loading}>
                Clear
              </button>
            </div>

            <dl className="metaGrid">
              <div>
                <dt>Name</dt>
                <dd>{fileDetails.name}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{fileDetails.size}</dd>
              </div>
              <div>
                <dt>Type</dt>
                <dd>{fileDetails.type}</dd>
              </div>
            </dl>
          </div>
        ) : null}

          <div className="supportCard cardRise">
            <span className="supportCard__label">Supported types</span>
            <div className="formatRow formatRow--compact">
              {[
                { label: "PDF", note: "Portable Document Format" },
                { label: "DOCX", note: "Word document" },
                { label: "TXT", note: "Plain text file" },
              ].map((item) => (
                <span key={item.label} className="formatChip formatChip--stacked">
                  <strong>{item.label}</strong>
                  <small>{item.note}</small>
                </span>
              ))}
            </div>
          </div>

        {error ? (
          <div className="errorCard cardRise" role="alert">
            <strong>Upload issue</strong>
            <p>{error}</p>
          </div>
        ) : null}

        <div className="actionRow">
          <button className="button button--primary actionButton" type="button" onClick={onUpload} disabled={loading}>
            {loading ? "Uploading..." : "Upload & Predict"}
          </button>
          <button className="button button--ghost actionButton" type="button" onClick={onClear} disabled={loading}>
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

export default UploadBox;
