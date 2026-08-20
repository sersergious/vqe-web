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

export type Energies = {
  exact: number;
  ideal_sampled: number;
  noisy_sampled: number;
  fake_sampled: number;
};

export type Landscape = { x: number[] } & Record<keyof Energies, number[]>;

export type VqeResult = {
  optimal_params: number[];
  optimal_energy: number;
  true_ground_energy: number;
  iterations: number;
  history: { params: number[]; energy: number }[];
};

export type SampledProbs = { mean: number[]; std: number[] };

export type Histogram = {
  bitstrings: string[];
  exact: number[];
  ideal: SampledProbs;
  noisy: SampledProbs;
  fake: SampledProbs;
  n_shots: number;
  n_reps: number;
};
