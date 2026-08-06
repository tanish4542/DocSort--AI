import React from "react";

function Loader({ text }) {
  return (
    <div className="loaderCard cardRise" role="status" aria-live="polite">
      <span className="loaderSpinner" aria-hidden="true" />
      <span className="loaderCard__copy">
        <strong>{text}</strong>
        <small>Sending the file to the FastAPI backend and waiting for the prediction.</small>
      </span>
    </div>
  );
}

export default Loader;
