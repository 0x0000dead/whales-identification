// src/api.ts
//
// Backend URL resolution order:
//   1. Build-time override via `VITE_BACKEND=...` env var (recommended for production).
//      Vite's `define` in vite.config.ts replaces __VITE_BACKEND__ at compile time —
//      no import.meta at runtime, so Jest can import this file without SyntaxError.
//   2. Runtime derivation from `window.location.hostname`: if the page is
//      served from any host, point the API at the same host on port 8000.
//      This lets the same Docker image work when opened from 127.0.0.1,
//      192.168.x.y, a LAN hostname, or behind a reverse proxy.
//   3. Only if neither are available (SSR, tests) fall back to localhost.
//
// Why this matters: Экспертиза 2.0 §1.2.4 raised a repeated issue where a
// hardcoded http://localhost:8000 default broke the UI for any viewer
// browsing from a non-localhost address.
function resolveBase(): string {
  // __VITE_BACKEND__ is replaced at Vite build time (vite.config.ts define).
  // In Jest it is set via jest.config.cjs globals. No import.meta required.
  const override =
    typeof __VITE_BACKEND__ !== 'undefined' ? __VITE_BACKEND__ : '';
  if (override && override !== 'undefined') {
    return override;
  }
  if (typeof window !== 'undefined' && window.location?.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return 'http://localhost:8000';
}

const BASE = resolveBase();

export type RejectionReason = 'not_a_marine_mammal' | 'low_confidence' | 'corrupted_image' | null;

export interface Candidate {
  class_animal: string;
  id_animal: string;
  probability: number;
}

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
  candidates?: Candidate[];
}

function networkError(e: unknown): Error {
  if (e instanceof TypeError && String(e.message).includes('fetch')) {
    return new Error('Сервер анализа недоступен. Убедитесь, что backend запущен и откройте страницу заново.');
  }
  return e instanceof Error ? e : new Error(String(e));
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function predictSingle(file: File): Promise<Detection> {
  const form = new FormData();
  form.append('file', file);
  let res: Response;
  try {
    res = await fetch(`${BASE}/v1/predict-single`, { method: 'POST', body: form });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) throw new Error(`Ошибка сервера (${res.status}): ${await res.text()}`);
  return res.json();
}

export async function predictBatch(zipFile: File): Promise<Detection[]> {
  const form = new FormData();
  form.append('archive', zipFile);
  let res: Response;
  try {
    res = await fetch(`${BASE}/v1/predict-batch`, { method: 'POST', body: form });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) throw new Error(`Ошибка сервера (${res.status}): ${await res.text()}`);
  return res.json();
}
