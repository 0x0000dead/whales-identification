const BASE = import.meta.env.VITE_BACKEND ?? 'http://localhost:8000';

export async function predictSingle(file: File) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/predict-single`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json(); // ожидаем { id, whale_id, confidence, ... }
}

export async function predictBatch(zipFile: File) {
  const form = new FormData();
  form.append('archive', zipFile);

  const res = await fetch(`${BASE}/predict-batch`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());

  // CSV придёт как blob → скачиваем
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), {
    href: url,
    download: 'predictions.csv'
  });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
