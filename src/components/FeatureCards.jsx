import React from "react";

const features = [
  {
    title: " Classification",
    text: "Uses backend predictions to map each document to the best-fit domain category.",
  },
  {
    title: "Automatic File Sorting",
    text: "Stores the uploaded file in the matching local folder without extra manual steps.",
  },
  {
    title: "Multi-Domain Support",
    text: "Handles Medical, Technology, Finance, Sports, and Education with clear labels.",
  },
  {
    title: "PDF / DOCX / TXT Support",
    text: "Accepts the common document formats used in academic and business workflows.",
  },
  {
    title: "Local Folder Organization",
    text: "Displays the saved path returned by the backend for easy verification and demos.",
  },
];

function FeatureCards() {
  return (
    <div className="featureGrid">
      {features.map((feature, index) => (
        <article key={feature.title} className="featureCard cardRise">
          <div className="featureCard__index" aria-hidden="true">
            {String(index + 1).padStart(2, "0")}
          </div>
          <h3>{feature.title}</h3>
          <p>{feature.text}</p>
        </article>
      ))}
    </div>
  );
}

export default FeatureCards;
