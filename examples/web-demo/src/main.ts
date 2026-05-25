import { init, impliedVol, impliedVolBatch, version } from "@ngv/opx";
// Vite resolves this to a hashed asset URL at build time and to a
// dev-server URL at dev time, which is what wasm-pack's init() needs.
// @ts-expect-error — Vite ?url import has no declared type.
import wasmUrl from "@ngv/opx/pkg-web/ngv_opx_wasm_bg.wasm?url";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main() {
  await init({ wasmUrl });
  console.log("[ngv-opx] wasm loaded, version =", version());

  // ---- live IV solver ----
  const inputs = {
    f: $("iv-f") as HTMLInputElement,
    k: $("iv-k") as HTMLInputElement,
    r: $("iv-r") as HTMLInputElement,
    d: $("iv-d") as HTMLInputElement,
    p: $("iv-p") as HTMLInputElement,
    cp: $("iv-cp") as HTMLSelectElement,
  };
  const out = $("iv-out") as HTMLDivElement;

  const recompute = () => {
    const f = parseFloat(inputs.f.value);
    const k = parseFloat(inputs.k.value);
    const r = parseFloat(inputs.r.value);
    const t = parseFloat(inputs.d.value) / 365;
    const price = parseFloat(inputs.p.value);
    const isCall = inputs.cp.value === "1";
    const iv = impliedVol(f, k, r, t, price, isCall);
    if (iv === -1.0) {
      out.className = "result sentinel";
      out.textContent = "IV undefined (price outside [intrinsic, upper bound])";
    } else {
      out.className = "result";
      out.textContent = `${(iv * 100).toFixed(3)}%`;
    }
  };
  for (const el of Object.values(inputs)) el.addEventListener("input", recompute);
  recompute();

  // ---- batch benchmark ----
  const benchBtn = $("bench") as HTMLButtonElement;
  const benchOut = $("bench-out").querySelector(".val") as HTMLSpanElement;

  benchBtn.addEventListener("click", () => {
    const N = 10_000;
    const f = new Float64Array(N);
    const k = new Float64Array(N);
    const r = new Float64Array(N).fill(0.045);
    const t = new Float64Array(N);
    const mp = new Float64Array(N);
    const cp = new Uint8Array(N);

    for (let i = 0; i < N; i++) {
      f[i] = 75 + (Math.random() - 0.5) * 10;
      k[i] = f[i] + (Math.random() - 0.5) * 20;
      t[i] = (1 + Math.random() * 365) / 365;
      const intrinsic = Math.max(f[i] - k[i], 0);
      mp[i] = intrinsic + 0.5 + Math.random() * 3;
      cp[i] = 1;
    }

    const tStart = performance.now();
    const ivs = impliedVolBatch(f, k, r, t, mp, cp);
    const tEnd = performance.now();

    let valid = 0;
    for (let i = 0; i < ivs.length; i++) if (ivs[i] !== -1.0) valid++;
    const elapsed = (tEnd - tStart).toFixed(1);
    benchOut.textContent = `${elapsed} ms — ${valid}/${N} solved (${((valid / N) * 100).toFixed(1)}%)`;
  });
}

main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<pre style="color:#c44">Failed to init wasm: ${err.message}</pre>`,
  );
});
