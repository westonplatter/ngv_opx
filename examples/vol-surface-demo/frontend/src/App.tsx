import { useEffect, useRef, useState, useCallback } from "react";
import { init, impliedVolBatch } from "@ngv/opx";
// Vite resolves this ?url import to the hashed asset path at build time.
// @ts-expect-error — no declared type for ?url imports
import wasmUrl from "@ngv/opx/pkg-web/ngv_opx_wasm_bg.wasm?url";
import Plotly from "plotly.js-dist-min";

// ---------- constants ----------

const STRIKES = Array.from({ length: 26 }, (_, i) => 35 + i); // 35..60
const EXPIRIES = ["1wk", "2wk", "3wk", "4wk", "5wk"];
const EXPIRY_WEEKS = [1, 2, 3, 4, 5];
const RATE = 0.05;
const WS_URL = "ws://localhost:8000/ws";

// Plotly chart config
const CHART_CONFIG: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["toImage"],
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor: "#0d1117",
  font: { color: "#e6edf3", family: "ui-monospace, monospace", size: 12 },
  margin: { l: 0, r: 0, b: 0, t: 52 },
  scene: {
    xaxis: {
      title: { text: "Strike ($)" },
      gridcolor: "#30363d",
      zerolinecolor: "#30363d",
    },
    yaxis: {
      title: { text: "Expiry" },
      tickvals: EXPIRY_WEEKS,
      ticktext: EXPIRIES,
      gridcolor: "#30363d",
    },
    zaxis: {
      title: { text: "Implied Vol (%)" },
      gridcolor: "#30363d",
    },
    bgcolor: "#0d1117",
    camera: { eye: { x: 1.8, y: -1.6, z: 0.9 } },
  },
};

// ---------- types ----------

interface Quote {
  expiry: string;
  t_years: number;
  strike: number;
  call_bid: number;
  call_ask: number;
  put_bid: number;
  put_ask: number;
}

interface Snapshot {
  spot: number;
  timestamp: number;
  quotes: Quote[];
}

// ---------- IV surface computation ----------

function computeIVMatrix(snap: Snapshot): number[][] {
  const z: number[][] = [];

  for (const expLabel of EXPIRIES) {
    const expQuotes = snap.quotes.filter((q) => q.expiry === expLabel);
    const quoteMap = new Map(expQuotes.map((q) => [q.strike, q]));

    const n = STRIKES.length;
    const forwards = new Float64Array(n).fill(snap.spot);
    const strikesArr = new Float64Array(STRIKES);
    const rates = new Float64Array(n).fill(RATE);
    const t_years = expQuotes[0]?.t_years ?? 0.02;
    const times = new Float64Array(n).fill(t_years);
    const prices = new Float64Array(n);
    const isCalls = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const K = STRIKES[i];
      const q = quoteMap.get(K);
      if (!q) {
        prices[i] = 0.01;
        isCalls[i] = 1;
        continue;
      }
      // Use OTM option for each strike: call when K >= spot, put when K < spot
      if (K >= snap.spot) {
        prices[i] = (q.call_bid + q.call_ask) / 2;
        isCalls[i] = 1;
      } else {
        prices[i] = (q.put_bid + q.put_ask) / 2;
        isCalls[i] = 0;
      }
    }

    const ivs = impliedVolBatch(forwards, strikesArr, rates, times, prices, isCalls);

    const row: number[] = [];
    for (let i = 0; i < n; i++) {
      // sentinel -1 means IV undefined; NaN renders as a gap in Plotly surface
      row.push(ivs[i] < 0 ? NaN : ivs[i] * 100);
    }
    z.push(row);
  }

  return z;
}

// ---------- component ----------

export default function App() {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInitialized = useRef(false);
  const [status, setStatus] = useState("Initializing WASM…");
  const [wasmReady, setWasmReady] = useState(false);
  const [stats, setStats] = useState<{ spot: number; fps: number } | null>(null);
  const fpsCountRef = useRef({ frames: 0, last: performance.now() });

  // Initialize WASM once on mount
  useEffect(() => {
    init({ wasmUrl })
      .then(() => {
        setStatus("WASM ready — connecting…");
        setWasmReady(true);
      })
      .catch((err: Error) => {
        setStatus(`WASM init failed: ${err.message}`);
      });
  }, []);

  const renderSurface = useCallback((snap: Snapshot) => {
    if (!chartRef.current) return;

    const z = computeIVMatrix(snap);

    const trace: Plotly.Data = {
      type: "surface",
      x: STRIKES,
      y: EXPIRY_WEEKS,
      z,
      colorscale: "Viridis",
      colorbar: {
        title: { text: "IV (%)", side: "right" },
        thickness: 14,
        len: 0.75,
      },
      contours: {
        z: { show: true, usecolormap: true, project: { z: true } },
      },
      hovertemplate:
        "Strike: $%{x}<br>Expiry: %{y}wk<br>IV: %{z:.1f}%<extra></extra>",
    } as Plotly.Data;

    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      title: {
        text: `Vol Surface — Spot $${snap.spot.toFixed(2)}`,
        font: { size: 15, color: "#e6edf3" },
        x: 0.5,
      },
    };

    if (!chartInitialized.current) {
      Plotly.newPlot(chartRef.current, [trace], layout, CHART_CONFIG);
      chartInitialized.current = true;
    } else {
      // react() updates data + layout without re-creating the DOM element,
      // preserving camera angle the user may have rotated to.
      Plotly.react(chartRef.current, [trace], layout, CHART_CONFIG);
    }

    // FPS counter
    const fc = fpsCountRef.current;
    fc.frames++;
    const now = performance.now();
    if (now - fc.last >= 1000) {
      setStats({ spot: snap.spot, fps: fc.frames });
      fc.frames = 0;
      fc.last = now;
    }
  }, []);

  // WebSocket — connect once WASM is ready
  useEffect(() => {
    if (!wasmReady) return;

    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => setStatus("Live — receiving quotes");

      ws.onmessage = (evt) => {
        try {
          const snap: Snapshot = JSON.parse(evt.data as string);
          renderSurface(snap);
        } catch {
          // malformed frame — skip
        }
      };

      ws.onclose = () => {
        setStatus("Disconnected — retrying in 2s…");
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        setStatus("WebSocket error — check that the backend is running");
        ws.close();
      };
    };

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [wasmReady, renderSurface]);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Status bar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "6px 14px",
          background: "#161b22",
          borderBottom: "1px solid #30363d",
          fontSize: 12,
          color: "#8b949e",
          flexShrink: 0,
        }}
      >
        <span>{status}</span>
        {stats && (
          <span>
            Spot: <b style={{ color: "#e6edf3" }}>${stats.spot.toFixed(2)}</b>
            &nbsp;&nbsp;{stats.fps} fps
          </span>
        )}
      </div>

      {/* Expiry legend */}
      <div
        style={{
          display: "flex",
          gap: 16,
          padding: "5px 14px",
          background: "#0d1117",
          borderBottom: "1px solid #21262d",
          fontSize: 11,
          color: "#8b949e",
          flexShrink: 0,
        }}
      >
        {EXPIRIES.map((e) => (
          <span key={e}>
            <b style={{ color: e === "2wk" ? "#f78166" : "#79c0ff" }}>{e}</b>
            {e === "2wk" ? " ★ earnings" : ""}
          </span>
        ))}
      </div>

      {/* Plotly chart */}
      <div ref={chartRef} style={{ flex: 1, minHeight: 0 }} />
    </div>
  );
}
