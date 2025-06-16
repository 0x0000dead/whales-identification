import { useState } from 'react';
import { predictSingle, predictBatch } from './api';

export default function App() {
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [batchFile,  setBatchFile]  = useState<File | null>(null);
  const [result, setResult]         = useState<unknown>(null);
  const [busy, setBusy]             = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const handleSingle = async () => {
    if (!singleFile) return;
    setBusy(true); setError(null);
    try {
      const data = await predictSingle(singleFile);
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally { setBusy(false); }
  };

  const handleBatch = async () => {
    if (!batchFile) return;
    setBusy(true); setError(null);
    try { await predictBatch(batchFile); }
    catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  };

  return (
    <main className="mx-auto max-w-xl p-6 font-sans">
      <h1 className="text-2xl font-bold mb-6">Whale ID frontend</h1>

      {/* ---- Single image ---- */}
      <section className="mb-8">
        <h2 className="font-semibold mb-2">1️⃣ Предикт по одному фото</h2>
        <input
          type="file"
          accept="image/*"
          onChange={e => setSingleFile(e.target.files?.[0] ?? null)}
          className="mb-2"
        />
        <button
          onClick={handleSingle}
          disabled={!singleFile || busy}
          className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
        >
          {busy ? 'Ждём...' : 'Отправить'}
        </button>

        {result && (
          <pre className="mt-4 p-3 bg-gray-100 rounded overflow-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </section>

      {/* ---- Batch ZIP ---- */}
      <section>
        <h2 className="font-semibold mb-2">2️⃣ ZIP → CSV</h2>
        <input
          type="file"
          accept=".zip"
          onChange={e => setBatchFile(e.target.files?.[0] ?? null)}
          className="mb-2"
        />
        <button
          onClick={handleBatch}
          disabled={!batchFile || busy}
          className="px-4 py-2 rounded bg-green-600 text-white disabled:opacity-50"
        >
          {busy ? 'Ждём...' : 'Скачать CSV'}
        </button>
      </section>

      {error && <p className="mt-4 text-red-600">{error}</p>}
    </main>
  );
}
