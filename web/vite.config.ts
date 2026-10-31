import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps asset paths relative so the site works at https://USER.github.io/REPO/.
// Files in public/ (including model.json, written by python/export_model.py) are copied to the site as-is.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
