// src/App.tsx
import { useState } from 'react';
import { predictSingle, predictBatch, Detection } from './api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts';

export default function App() {
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [batchFile,  setBatchFile]  = useState<File | null>(null);
  const [result, setResult]         = useState<Detection | null>(null);
  const [batchResults, setBatchResults] = useState<Detection[] | null>(null);
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
    try {
      const data = await predictBatch(batchFile);
      setBatchResults(data);
    } catch (e: any) {
      setError(e.message);
    } finally { setBusy(false); }
  };

  // сгруппировать по id_animal
  const chartData = batchResults
    ? Object.entries(
        batchResults.reduce<Record<string,number>>((acc, d) => {
          acc[d.id_animal] = (acc[d.id_animal]||0) + 1;
          return acc;
        }, {})
      ).map(([name, count]) => ({ name, count }))
    : [];

  return (
    <main className="mx-auto max-w-xl p-6 font-sans">
      <h1 className="text-2xl font-bold mb-6">Whale ID Frontend</h1>

      {/* Single */}
      <section className="mb-8">
        <h2 className="font-semibold mb-2">1️⃣ Single</h2>
        <input type="file" accept="image/*" onChange={e=>setSingleFile(e.target.files?.[0]||null)} />
        <button onClick={handleSingle} disabled={!singleFile||busy}>
          {busy?'…':'Send'}
        </button>
        {result && (
          <pre className="mt-4 p-3 bg-gray-100 rounded">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </section>

      {/* Batch + Dashboard */}
      <section>
        <h2 className="font-semibold mb-2">2️⃣ Batch + Dashboard</h2>
        <input type="file" accept=".zip" onChange={e=>setBatchFile(e.target.files?.[0]||null)} />
        <button onClick={handleBatch} disabled={!batchFile||busy}>
          {busy?'…':'Run Batch'}
        </button>

        {batchResults && (
          <div className="mt-6">
            <h3 className="font-medium mb-2">Whale Types Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {error && <p className="mt-4 text-red-600">{error}</p>}
    </main>
  );
}
