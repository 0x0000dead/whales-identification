// src/api.ts
const BASE = import.meta.env.VITE_BACKEND ?? 'http://localhost:8000';

if (typeof window !== 'undefined' && BASE === 'http://localhost:8000') {
  const host = window.location.hostname;
  if (host !== 'localhost' && host !== '127.0.0.1' && host !== '') {
    // eslint-disable-next-line no-console
    console.warn(
      `[EcoMarineAI] VITE_BACKEND is hardcoded to http://localhost:8000 but the page is served from "${host}". ` +
        'Set VITE_BACKEND at build time to your backend URL (e.g. http://192.168.1.42:8000).'
    );
  }
}

export type RejectionReason = 'not_a_marine_mammal' | 'low_confidence' | 'corrupted_image' | null;

export interface Detection {
  image_ind: string;
  bbox: [number, number, number, number];
  class_animal: string;
  id_animal: string;
  probability: number;
  mask?: string | null;
  is_cetacean?: boolean;
  cetacean_score?: number;
  rejected?: boolean;
  rejection_reason?: RejectionReason;
  model_version?: string;
}

export async function predictSingle(file: File): Promise<Detection> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/v1/predict-single`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictBatch(zipFile: File): Promise<Detection[]> {
  const form = new FormData();
  form.append('archive', zipFile);
  const res = await fetch(`${BASE}/v1/predict-batch`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
