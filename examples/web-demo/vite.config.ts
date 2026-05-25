import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173, open: true },
  // wasm-pack web output is async-init; Vite handles it via ?init or
  // top-level await fine in dev. No special config needed for our flow.
});
