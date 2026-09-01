export type ParamSpec = {
  name: string;
  label: string;
  default: number;
  min: number;
  max: number;
};

export type Scenario = {
  id: string;
  label: string;
  hamiltonian: string;
  n_qubits: number;
  params: ParamSpec[];
  coefficient: ParamSpec | null;
  true_ground_energy: number;
};

/** The shot-noise settings a result was produced with, echoed by the API so a
 *  figure can be regenerated exactly. */
export type Sampling = {
  readout_error: number;
  seed: number;
};

export type Energies = Sampling & {
  exact: number;
  ideal_sampled: number;
  noisy_sampled: number;
  fake_sampled: number;
};

type EnergySeries = "exact" | "ideal_sampled" | "noisy_sampled" | "fake_sampled";

export type Landscape = Sampling & { x: number[] } & Record<EnergySeries, number[]>;

export type VqeResult = {
  optimal_params: number[];
  optimal_energy: number;
  true_ground_energy: number;
  iterations: number;
  history: { params: number[]; energy: number }[];
};

export type SampledProbs = { mean: number[]; std: number[] };

export type Histogram = Sampling & {
  bitstrings: string[];
  exact: number[];
  ideal: SampledProbs;
  noisy: SampledProbs;
  fake: SampledProbs;
  n_shots: number;
  n_reps: number;
};
