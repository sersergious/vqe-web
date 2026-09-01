"use client";

import Plot from "./Plot";
import type { Histogram, Landscape, Scenario, VqeResult } from "@/lib/types";

const COLORS = {
  exact: "#4c8dd8",
  ideal: "#3fa16b",
  noisy: "#e0603f",
  fake: "#9b7fd4",
  target: "#e05252",
};

const GRID = { gridcolor: "#2c333d", zerolinecolor: "#2c333d" };

/** Every noisy series names the readout error it was drawn at, so a saved plot
 *  is self-describing. */
const readoutLabel = (readoutError: number) =>
  `readout ${Math.round(readoutError * 100)}%`;

export function LandscapeChart({
  data,
  scenario,
  sweepLabel,
}: {
  data: Landscape;
  scenario: Scenario;
  sweepLabel: string;
}) {
  const line = (y: number[], name: string, color: string, dash?: boolean) => ({
    x: data.x,
    y,
    name,
    type: "scatter",
    mode: dash ? "lines" : "lines+markers",
    line: { color, width: dash ? 2.5 : 1.5 },
    marker: { size: 5 },
  });

  return (
    <Plot
      data={[
        line(data.exact, "Exact (statevector)", COLORS.exact, true),
        line(data.ideal_sampled, "Sampled — ideal", COLORS.ideal),
        line(
          data.noisy_sampled,
          `Sampled — noisy (${readoutLabel(data.readout_error)})`,
          COLORS.noisy,
        ),
        line(data.fake_sampled, "Sampled — fake backend", COLORS.fake),
      ]}
      layout={{
        title: { text: `Energy landscape — ${scenario.hamiltonian}` },
        xaxis: { title: { text: `${sweepLabel} [rad]` }, ...GRID },
        yaxis: { title: { text: "⟨H⟩" }, ...GRID },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: scenario.true_ground_energy,
            y1: scenario.true_ground_energy,
            line: { color: COLORS.target, width: 1.5, dash: "dash" },
          },
        ],
      }}
    />
  );
}

export function ConvergenceChart({ result }: { result: VqeResult }) {
  return (
    <Plot
      data={[
        {
          y: result.history.map((point) => point.energy),
          name: "⟨H⟩",
          type: "scatter",
          mode: "lines+markers",
          line: { color: COLORS.exact, width: 1.5 },
          marker: { size: 4 },
        },
      ]}
      layout={{
        title: {
          text: `Convergence — ${result.iterations} iterations, final ⟨H⟩ = ${result.optimal_energy.toFixed(6)}`,
        },
        xaxis: { title: { text: "Optimizer iteration" }, ...GRID },
        yaxis: { title: { text: "⟨H⟩" }, ...GRID },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: result.true_ground_energy,
            y1: result.true_ground_energy,
            line: { color: COLORS.target, width: 1.5, dash: "dash" },
          },
        ],
        annotations: [
          {
            xref: "paper",
            x: 1,
            y: result.true_ground_energy,
            xanchor: "right",
            yanchor: "bottom",
            text: `true ground state ${result.true_ground_energy.toFixed(4)}`,
            showarrow: false,
            font: { color: COLORS.target, size: 11 },
          },
        ],
      }}
    />
  );
}

export function HistogramChart({ data }: { data: Histogram }) {
  const labels = data.bitstrings.map((bits) => `|${bits}⟩`);
  const bar = (
    y: number[],
    name: string,
    color: string,
    std?: number[],
  ) => ({
    x: labels,
    y,
    name,
    type: "bar",
    marker: { color },
    error_y: std
      ? { type: "data", array: std, visible: true, color: "#8b949e", thickness: 1.5 }
      : undefined,
  });

  return (
    <Plot
      data={[
        bar(data.exact, "Exact (statevector)", COLORS.exact),
        bar(data.ideal.mean, "Ideal", COLORS.ideal, data.ideal.std),
        bar(
          data.noisy.mean,
          `Noisy — ${readoutLabel(data.readout_error)}`,
          COLORS.noisy,
          data.noisy.std,
        ),
        bar(data.fake.mean, "Fake backend", COLORS.fake, data.fake.std),
      ]}
      layout={{
        title: {
          text: `Measurement outcomes — mean ± std over ${data.n_reps} × ${data.n_shots} shots`,
        },
        barmode: "group",
        xaxis: { ...GRID },
        yaxis: { title: { text: "Probability" }, range: [0, 1.05], ...GRID },
      }}
    />
  );
}
