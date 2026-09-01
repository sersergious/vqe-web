"use client";

import type { ParamSpec, Scenario } from "@/lib/types";

export function ScenarioSelect({
  scenarios,
  value,
  onChange,
}: {
  scenarios: Scenario[];
  value: string;
  onChange: (id: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">Scenario</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {scenarios.map((scenario) => (
          <option key={scenario.id} value={scenario.id}>
            {scenario.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Slider({
  spec,
  value,
  onChange,
  step = "any",
  format = (value: number) => value.toFixed(3),
}: {
  spec: ParamSpec;
  value: number;
  onChange: (value: number) => void;
  /** Discrete steps for knobs swept on a fixed grid, e.g. readout error. */
  step?: number | "any";
  format?: (value: number) => string;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {spec.label}
        <output>{format(value)}</output>
      </span>
      <input
        type="range"
        min={spec.min}
        max={spec.max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function ParamSliders({
  specs,
  values,
  onChange,
}: {
  specs: ParamSpec[];
  values: number[];
  onChange: (values: number[]) => void;
}) {
  return (
    <>
      {specs.map((spec, index) => (
        <Slider
          key={spec.name}
          spec={spec}
          value={values[index] ?? spec.default}
          onChange={(next) =>
            onChange(values.map((value, i) => (i === index ? next : value)))
          }
        />
      ))}
    </>
  );
}
