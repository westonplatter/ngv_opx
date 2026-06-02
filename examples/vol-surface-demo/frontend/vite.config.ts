import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  optimizeDeps: {
    // Prevent esbuild from pre-bundling the WASM package — it can't inline
    // .wasm binaries and would break the wasm-pack init() URL resolution.
    exclude: ["@ngv/opx"],
  },
});
