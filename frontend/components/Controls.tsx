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
}: {
  spec: ParamSpec;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {spec.label}
        <output>{value.toFixed(3)}</output>
      </span>
      <input
        type="range"
        min={spec.min}
        max={spec.max}
        step="any"
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
