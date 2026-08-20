"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { ParamSliders, ScenarioSelect, Slider } from "@/components/Controls";
import { ConvergenceChart, HistogramChart, LandscapeChart } from "@/components/charts";
import * as api from "@/lib/api";
import type { Energies, Histogram, Landscape, Scenario, VqeResult } from "@/lib/types";

export default function Page() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [params, setParams] = useState<number[]>([]);
  const [coefficient, setCoefficient] = useState<number | null>(null);
  const [sweepIndex, setSweepIndex] = useState(0);
  const [nPoints, setNPoints] = useState(24);

  const [energies, setEnergies] = useState<Energies | null>(null);
  const [landscape, setLandscape] = useState<Landscape | null>(null);
  const [vqe, setVqe] = useState<VqeResult | null>(null);
  const [histogram, setHistogram] = useState<Histogram | null>(null);

  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scenario = useMemo(
    () => scenarios.find((entry) => entry.id === scenarioId) ?? null,
    [scenarios, scenarioId],
  );

  useEffect(() => {
    api
      .getScenarios()
      .then((list) => {
        setScenarios(list);
        if (list.length) selectScenario(list[0]);
      })
      .catch((cause) => setError(String(cause.message ?? cause)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function selectScenario(next: Scenario) {
    setScenarioId(next.id);
    setParams(next.params.map((spec) => spec.default));
    setCoefficient(next.coefficient ? next.coefficient.default : null);
    setSweepIndex(0);
    setEnergies(null);
    setLandscape(null);
    setVqe(null);
    setHistogram(null);
    setError(null);
  }

  const run = useCallback(
    async (name: string, action: () => Promise<void>) => {
      setBusy(name);
      setError(null);
      try {
        await action();
      } catch (cause) {
        setError(String((cause as Error).message ?? cause));
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  if (!scenario) {
    return (
      <main className="shell">
        <p className="muted">{error ? `Error: ${error}` : "Loading scenarios…"}</p>
      </main>
    );
  }

  const body = { coefficient };
  const disabled = busy !== null;

  return (
    <main className="shell">
      <header className="header">
        <h1>Variational Quantum Eigensolver</h1>
        <p className="muted">
          {scenario.hamiltonian} · {scenario.n_qubits} qubit
          {scenario.n_qubits > 1 ? "s" : ""} · true ground state{" "}
          {scenario.true_ground_energy.toFixed(4)}
        </p>
      </header>

      <div className="layout">
        <aside className="panel">
          <ScenarioSelect
            scenarios={scenarios}
            value={scenarioId}
            onChange={(id) => {
              const next = scenarios.find((entry) => entry.id === id);
              if (next) selectScenario(next);
            }}
          />

          {scenario.coefficient && coefficient !== null && (
            <Slider
              spec={scenario.coefficient}
              value={coefficient}
              onChange={setCoefficient}
            />
          )}

          <h2>Ansatz parameters</h2>
          <ParamSliders specs={scenario.params} values={params} onChange={setParams} />

          <h2>Sweep</h2>
          <label className="field">
            <span className="field-label">Parameter</span>
            <select
              value={sweepIndex}
              onChange={(event) => setSweepIndex(Number(event.target.value))}
            >
              {scenario.params.map((spec, index) => (
                <option key={spec.name} value={index}>
                  {spec.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">
              Points<output>{nPoints}</output>
            </span>
            <input
              type="range"
              min={4}
              max={120}
              step={4}
              value={nPoints}
              onChange={(event) => setNPoints(Number(event.target.value))}
            />
          </label>

          <div className="actions">
            <button
              disabled={disabled}
              onClick={() =>
                run("evaluate", async () =>
                  setEnergies(await api.evaluate(scenario.id, params, body)),
                )
              }
            >
              {busy === "evaluate" ? "Evaluating…" : "Evaluate energy"}
            </button>
            <button
              disabled={disabled}
              onClick={() =>
                run("vqe", async () =>
                  setVqe(await api.runVqe(scenario.id, params, body)),
                )
              }
            >
              {busy === "vqe" ? "Optimizing…" : "Run VQE"}
            </button>
            <button
              disabled={disabled}
              onClick={() =>
                run("landscape", async () =>
                  setLandscape(
                    await api.sweepLandscape(
                      scenario.id,
                      params,
                      sweepIndex,
                      nPoints,
                      body,
                    ),
                  ),
                )
              }
            >
              {busy === "landscape" ? "Sweeping…" : "Sweep landscape"}
            </button>
            <button
              disabled={disabled}
              onClick={() =>
                run("histogram", async () =>
                  setHistogram(await api.getHistogram(scenario.id, params, body)),
                )
              }
            >
              {busy === "histogram" ? "Sampling…" : "Show histogram"}
            </button>
          </div>

          {error && <p className="error">{error}</p>}
        </aside>

        <section className="results">
          {energies && (
            <div className="card metrics">
              {(
                [
                  ["Exact", energies.exact],
                  ["Ideal", energies.ideal_sampled],
                  ["Noisy", energies.noisy_sampled],
                  ["Fake backend", energies.fake_sampled],
                ] as const
              ).map(([label, value]) => (
                <div key={label} className="metric">
                  <span className="muted">{label}</span>
                  <strong>{value.toFixed(6)}</strong>
                </div>
              ))}
            </div>
          )}

          {vqe && (
            <div className="card">
              <ConvergenceChart result={vqe} />
              <div className="row">
                <span className="muted">
                  optimal:{" "}
                  {vqe.optimal_params.map((value) => value.toFixed(4)).join(", ")}
                </span>
                <button onClick={() => setParams(vqe.optimal_params)}>
                  Load into sliders
                </button>
              </div>
            </div>
          )}

          {landscape && (
            <div className="card">
              <LandscapeChart
                data={landscape}
                scenario={scenario}
                sweepLabel={scenario.params[sweepIndex]?.label ?? "θ"}
              />
            </div>
          )}

          {histogram && (
            <div className="card">
              <HistogramChart data={histogram} />
            </div>
          )}

          {!energies && !vqe && !landscape && !histogram && (
            <p className="muted">
              Pick a scenario, set the parameters, then run one of the actions.
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
