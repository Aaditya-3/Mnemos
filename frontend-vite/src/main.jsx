import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const initialTheme = localStorage.getItem("mnemos_theme");
if (initialTheme === "dark" || initialTheme === "light") {
  document.documentElement.setAttribute("data-theme", initialTheme);
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
