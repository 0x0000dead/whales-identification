// src/api.ts
const BASE = import.meta.env.VITE_BACKEND ?? 'http://localhost:8000';

export interface Detection {
  image_ind: string;
  bbox: [number, number, number, number];
  class_animal: string;
  id_animal: string;
  probability: number;
  mask?: string;
}

export async function predictSingle(file: File): Promise<Detection> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/predict-single`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictBatch(zipFile: File): Promise<Detection[]> {
  const form = new FormData();
  form.append('archive', zipFile);
  const res = await fetch(`${BASE}/predict-batch`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
