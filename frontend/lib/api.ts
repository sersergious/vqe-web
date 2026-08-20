import type { Energies, Histogram, Landscape, Scenario, VqeResult } from "./types";

async function call<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.detail ? JSON.stringify(payload.detail) : response.statusText);
  }
  return payload as T;
}

type Body = { coefficient: number | null };

export const getScenarios = () => call<Scenario[]>("scenarios");

export const evaluate = (id: string, params: number[], body: Body) =>
  call<Energies>(`scenarios/${id}/evaluate`, { params, ...body });

export const sweepLandscape = (
  id: string,
  fixed_params: number[],
  sweep_param_index: number,
  n_points: number,
  body: Body,
) =>
  call<Landscape>(`scenarios/${id}/landscape`, {
    fixed_params,
    sweep_param_index,
    n_points,
    ...body,
  });

export const runVqe = (id: string, x0: number[], body: Body) =>
  call<VqeResult>(`scenarios/${id}/vqe`, { x0, ...body });

export const getHistogram = (id: string, params: number[], body: Body) =>
  call<Histogram>(`scenarios/${id}/histogram`, { params, ...body });
