import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/chat": "http://localhost:8000",
      "/chats": "http://localhost:8000",
      "/auth": "http://localhost:8000",
      "/tools": "http://localhost:8000",
      "/memories": "http://localhost:8000",
      "/metrics": "http://localhost:8000"
    }
  }
});

