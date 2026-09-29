import React from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAppState } from "../context/AppContext";

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const aboutActive = location.pathname === "/" && location.hash === "#about";
  const { localDirName, isFileSystemAccessSupported, selectLocalDirectory } = useAppState();

  return (
    <header className="topBar">
      <button className="brandMark" type="button" onClick={() => navigate("/")} aria-label="Go home">
        <span className="brandMark__ring" />
        <span className="brandMark__dot" />
      </button>

      <div className="brandCopy">
        <span className="brandCopy__eyebrow">Cross-domain workflow</span>
        <button className="brandCopy__title" type="button" onClick={() => navigate("/")}>
          Document Sorter
        </button>
        <span className="brandCopy__subtext">FastAPI-powered classification and local folder organization</span>
      </div>

      <nav className="navLinks" aria-label="Primary navigation">
        {isFileSystemAccessSupported && (
          <button
            className="folderBadge"
            type="button"
            onClick={selectLocalDirectory}
            title={localDirName ? `Active local folder: ${localDirName}` : "Choose local sorting folder"}
          >
            <span className="folderBadge__icon">📁</span>
            <span className="folderBadge__text">{localDirName || "Choose Folder"}</span>
          </button>
        )}
        <NavLink to="/" end className={({ isActive }) => `navLinks__item ${isActive ? "navLinks__item--active" : ""}`}>
          Home
        </NavLink>
        <NavLink
          to="/upload"
          className={({ isActive }) => `navLinks__item ${isActive ? "navLinks__item--active" : ""}`}
        >
          Upload
        </NavLink>
        <Link
          to={{ pathname: "/", hash: "about" }}
          className={`navLinks__item ${aboutActive ? "navLinks__item--active" : ""}`}
        >
          About
        </Link>
      </nav>
    </header>
  );
}

export default Navbar;
