import { useEffect, useRef, useState, useCallback } from "react";
import { init, impliedVol, impliedVolBatch } from "@ngv/opx";
// Vite resolves this ?url import to the hashed asset path at build time.
// @ts-expect-error — no declared type for ?url imports
import wasmUrl from "@ngv/opx/pkg-web/ngv_opx_wasm_bg.wasm?url";
import Plotly from "plotly.js-dist-min";

// ---------- constants ----------

const STRIKE_MIN = 35;
const STRIKES = Array.from({ length: 26 }, (_, i) => STRIKE_MIN + i); // 35..60
const EXPIRIES = ["1wk", "2wk", "3wk", "4wk", "5wk"];
const RATE = 0.05;
const WS_URL = "ws://localhost:8000/ws";

// One color per expiry (2wk is the earnings event). RGB so we can vary opacity
// per marker to fade points that haven't traded recently.
const EXPIRY_RGB: Record<string, string> = {
  "1wk": "88,166,255",
  "2wk": "247,129,102",
  "3wk": "86,211,100",
  "4wk": "188,140,255",
  "5wk": "227,179,65",
};

// A traded marker is full-bright, then fades to MIN_OPACITY over FADE_MS.
const FADE_MS = 6000;
const MIN_OPACITY = 0.12;

const CHART_CONFIG: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["toImage"],
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor: "#0d1117",
  font: { color: "#e6edf3", family: "ui-monospace, monospace", size: 12 },
  margin: { l: 64, r: 24, b: 52, t: 52 },
  xaxis: {
    title: { text: "Strike ($)" },
    gridcolor: "#21262d",
    zerolinecolor: "#30363d",
  },
  yaxis: {
    title: { text: "Implied Vol (%)" },
    gridcolor: "#21262d",
    zerolinecolor: "#30363d",
    rangemode: "tozero",
    ticksuffix: "%",
  },
  hovermode: "x unified",
  legend: { orientation: "h", y: 1.04, x: 0, font: { size: 11 } },
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
  type: "snapshot";
  spot: number;
  timestamp: number;
  quotes: Quote[];
}

interface Trade {
  expiry: string;
  t_years: number;
  strike: number;
  is_call: boolean;
  price: number;
  size: number;
}

interface TradesMsg {
  type: "trades";
  spot: number;
  timestamp: number;
  trades: Trade[];
}

// The surface we keep in memory and patch as trades arrive.
// iv / lastTrade are indexed [expiryIndex][strikeIndex].
interface Surface {
  spot: number;
  iv: number[][]; // implied vol in %, NaN where undefined
  lastTrade: number[][]; // performance.now() ms of last update per point
  tYears: number[]; // time-to-expiry per expiry
}

const strikeIndex = (strike: number) => strike - STRIKE_MIN;

// ---------- snapshot → full IV matrix (one batch solve per expiry) ----------

function solveSnapshot(snap: Snapshot): { iv: number[][]; tYears: number[] } {
  const iv: number[][] = [];
  const tYears: number[] = [];

  for (const expLabel of EXPIRIES) {
    const expQuotes = snap.quotes.filter((q) => q.expiry === expLabel);
    const quoteMap = new Map(expQuotes.map((q) => [q.strike, q]));

    const n = STRIKES.length;
    const forwards = new Float64Array(n).fill(snap.spot);
    const strikesArr = new Float64Array(STRIKES);
    const rates = new Float64Array(n).fill(RATE);
    const t = expQuotes[0]?.t_years ?? 0.02;
    const times = new Float64Array(n).fill(t);
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
      // OTM option per strike: call when K >= spot, put when K < spot.
      if (K >= snap.spot) {
        prices[i] = (q.call_bid + q.call_ask) / 2;
        isCalls[i] = 1;
      } else {
        prices[i] = (q.put_bid + q.put_ask) / 2;
        isCalls[i] = 0;
      }
    }

    const solved = impliedVolBatch(forwards, strikesArr, rates, times, prices, isCalls);
    const row: number[] = [];
    for (let i = 0; i < n; i++) {
      // sentinel -1 means IV undefined; NaN renders as a gap in the line.
      row.push(solved[i] < 0 ? NaN : solved[i] * 100);
    }
    iv.push(row);
    tYears.push(t);
  }

  return { iv, tYears };
}

// ---------- component ----------

export default function App() {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInitialized = useRef(false);
  const surfaceRef = useRef<Surface | null>(null);
  const tradeCountRef = useRef({ n: 0, last: performance.now() });
  const [status, setStatus] = useState("Initializing WASM…");
  const [wasmReady, setWasmReady] = useState(false);
  const [stats, setStats] = useState<{ spot: number; tps: number } | null>(null);

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

  // Draw the current surface. Markers fade with age; the line is a faint guide.
  const renderChart = useCallback(() => {
    const surface = surfaceRef.current;
    if (!chartRef.current || !surface) return;
    const now = performance.now();

    const traces: Plotly.Data[] = EXPIRIES.map((label, e) => {
      const colors = STRIKES.map((_, i) => {
        const age = now - (surface.lastTrade[e]?.[i] ?? 0);
        const opacity = Math.max(MIN_OPACITY, 1 - age / FADE_MS);
        return `rgba(${EXPIRY_RGB[label]},${opacity.toFixed(3)})`;
      });
      return {
        type: "scatter",
        mode: "lines+markers",
        name: label,
        x: STRIKES,
        y: surface.iv[e],
        connectgaps: false,
        line: { color: `rgba(${EXPIRY_RGB[label]},0.3)`, width: 1.5 },
        marker: { size: 7, color: colors },
        hovertemplate: `${label} — $%{x}: %{y:.1f}%<extra></extra>`,
      } as Plotly.Data;
    });

    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      title: {
        text: `Vol Smile by Expiry — Spot $${surface.spot.toFixed(2)}`,
        font: { size: 15, color: "#e6edf3" },
        x: 0.5,
      },
      shapes: [
        {
          type: "line",
          x0: surface.spot,
          x1: surface.spot,
          yref: "paper",
          y0: 0,
          y1: 1,
          line: { color: "#6e7681", width: 1, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: surface.spot,
          yref: "paper",
          y: 1,
          text: "spot",
          showarrow: false,
          font: { size: 10, color: "#8b949e" },
          xanchor: "left",
          yanchor: "bottom",
        },
      ],
    };

    if (!chartInitialized.current) {
      Plotly.newPlot(chartRef.current, traces, layout, CHART_CONFIG);
      chartInitialized.current = true;
    } else {
      // react() updates in place, preserving any zoom/pan the user applied.
      Plotly.react(chartRef.current, traces, layout, CHART_CONFIG);
    }
  }, []);

  // Full snapshot → reset the in-memory surface.
  const ingestSnapshot = useCallback((snap: Snapshot) => {
    const now = performance.now();
    const { iv, tYears } = solveSnapshot(snap);
    surfaceRef.current = {
      spot: snap.spot,
      iv,
      tYears,
      lastTrade: iv.map((row) => row.map(() => now)), // start bright, then fade
    };
  }, []);

  // Trade burst → solve IV for just the traded contracts and patch those points.
  const ingestTrades = useCallback((msg: TradesMsg) => {
    const surface = surfaceRef.current;
    if (!surface) return; // ignore trades until we have a snapshot
    const now = performance.now();
    surface.spot = msg.spot;

    for (const t of msg.trades) {
      const e = EXPIRIES.indexOf(t.expiry);
      const i = strikeIndex(t.strike);
      if (e < 0 || i < 0 || i >= STRIKES.length) continue;

      const T = t.t_years ?? surface.tYears[e];
      const solved = impliedVol(surface.spot, t.strike, RATE, T, t.price, t.is_call);
      if (solved >= 0) {
        surface.iv[e][i] = solved * 100;
        surface.lastTrade[e][i] = now;
      }
    }
    tradeCountRef.current.n += msg.trades.length;
  }, []);

  // Animation tick: re-render so markers fade smoothly, and report trades/sec.
  useEffect(() => {
    const id = setInterval(() => {
      renderChart();
      const tc = tradeCountRef.current;
      const now = performance.now();
      const elapsed = now - tc.last;
      if (elapsed >= 1000 && surfaceRef.current) {
        setStats({ spot: surfaceRef.current.spot, tps: Math.round((tc.n / elapsed) * 1000) });
        tc.n = 0;
        tc.last = now;
      }
    }, 200);
    return () => clearInterval(id);
  }, [renderChart]);

  // WebSocket — connect once WASM is ready
  useEffect(() => {
    if (!wasmReady) return;

    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => setStatus("Live — receiving trades");

      ws.onmessage = (evt) => {
        try {
          const msg: Snapshot | TradesMsg = JSON.parse(evt.data as string);
          if (msg.type === "trades") {
            ingestTrades(msg);
          } else {
            ingestSnapshot(msg);
          }
          renderChart();
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
  }, [wasmReady, ingestSnapshot, ingestTrades, renderChart]);

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
            &nbsp;&nbsp;{stats.tps} trades/s
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
            <b style={{ color: `rgb(${EXPIRY_RGB[e]})` }}>{e}</b>
            {e === "2wk" ? " ★ earnings" : ""}
          </span>
        ))}
        <span style={{ marginLeft: "auto", fontStyle: "italic" }}>
          bright = just traded · faded = stale
        </span>
      </div>

      {/* Plotly chart */}
      <div ref={chartRef} style={{ flex: 1, minHeight: 0 }} />
    </div>
  );
}
