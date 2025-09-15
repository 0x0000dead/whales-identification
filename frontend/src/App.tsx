import {useState, useEffect} from 'react';
import {predictSingle, predictBatch, Detection} from './api';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';

export default function App() {
    const [singleFile, setSingleFile] = useState<File | null>(null);
    const [batchFile, setBatchFile] = useState<File | null>(null);
    const [result, setResult] = useState<Detection | null>(null);
    const [batchResults, setBatchResults] = useState<Detection[] | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        if (singleFile) {
            const url = URL.createObjectURL(singleFile);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setPreviewUrl(null);
        }
    }, [singleFile]);

    const handleSingle = async () => {
        if (!singleFile) return;
        setBusy(true);
        setError(null);
        try {
            const data = await predictSingle(singleFile);
            setResult(data);
        } catch (e: any) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    };

    const handleBatch = async () => {
        if (!batchFile) return;
        setBusy(true);
        setError(null);
        try {
            const data = await predictBatch(batchFile);
            setBatchResults(data);
        } catch (e: any) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    };

    const chartData = batchResults
        ? Object.entries(
            batchResults.reduce<Record<string, number>>((acc, d) => {
                acc[d.id_animal] = (acc[d.id_animal] || 0) + 1;
                return acc;
            }, {})
        ).map(([name, count]) => ({name, count}))
        : [];
    const closeModal = () => {
        setError(null);
    };
    return (
        <main className="mx-auto max-w-4xl p-4 sm:p-6 font-sans">
            <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-center">Идентификация морских млекопитающих</h1>

            <section className="mb-8">
                <h2 className="font-semibold mb-2 text-lg sm:text-xl">1️⃣ Одиночная обработка</h2>
                <div className="flex flex-col sm:flex-row items-center gap-4">
                    <input
                        type="file"
                        accept="image/*"
                        className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-500 file:text-white hover:file:bg-blue-600 w-full sm:w-auto"
                        onChange={(e) => setSingleFile(e.target.files?.[0] || null)}
                    />
                    <button
                        onClick={handleSingle}
                        disabled={!singleFile || busy}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
                    >
                        {busy ? (
                            <span className="flex items-center gap-2 justify-center">
                <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                  <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                  ></circle>
                  <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8h8a8 8 0 01-8 8 8 8 0 01-8-8z"
                  ></path>
                </svg>
                Обработка...
              </span>
                        ) : (
                            'Отправить'
                        )}
                    </button>
                </div>
                {previewUrl && (
                    <div className="mt-4">
                        <h3 className="font-medium mb-2">Предпросмотр:</h3>
                        <img
                            src={previewUrl}
                            alt="Uploaded whale"
                            className="max-w-full max-h-80 h-auto rounded-lg shadow-md sm:max-w-md"
                        />
                    </div>
                )}
                {result && (
                    <div className="mt-4 p-4 bg-gray-100 rounded-lg shadow">
                        <h3 className="font-medium mb-2">Результат:</h3>
                        <p className="text-sm overflow-hidden max-w-full">На картинке: {result.id_animal}</p>
                        <p className="text-sm overflow-hidden max-w-full">Вид животного: {result.class_animal}</p>
                        <p className="text-sm overflow-hidden max-w-full">Вероятность: {result.probability}</p>
                    </div>
                )}
            </section>

            <section>
            <h2 className="font-semibold mb-2 text-lg sm:text-xl">2️⃣ Пакетная обработка</h2>
                <div className="flex flex-col sm:flex-row items-center gap-4">
                    <input
                        type="file"
                        accept=".zip"
                        className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-500 file:text-white hover:file:bg-blue-600 w-full sm:w-auto"
                        onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                    />
                    <button
                        onClick={handleBatch}
                        disabled={!batchFile || busy}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
                    >
                        {busy ? (
                            <span className="flex items-center gap-2 justify-center">
                <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                  <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                  ></circle>
                  <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8h8a8 8 0 01-8 8 8 8 0 01-8-8z"
                  ></path>
                </svg>
                Обработка...
              </span>
                        ) : (
                            'Отправить пакет'
                        )}
                    </button>
                </div>

                {batchResults && (
                    <div className="mt-6">
                        <h3 className="font-medium mb-2 text-center">Распределение типов сущностей</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData}>
                                <XAxis dataKey="name"/>
                                <YAxis/>
                                <Tooltip/>
                                <Bar dataKey="count" fill="#3B82F6"/>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </section>

            {error && (
                <div
                    className="fixed inset-0 bg-transparent backdrop-blur-md flex items-center justify-center z-50"
                    onClick={closeModal}
                >
                    <div
                        className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full mx-4 sm:mx-auto border border-gray-200 animate-fade-in"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-semibold text-red-600 flex items-center gap-2">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Ошибка
                            </h3>
                        </div>
                        <p className="text-gray-700 mb-6">{error}</p>
                        <div className="flex justify-end">
                            <button
                                onClick={closeModal}
                                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                Закрыть
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
}