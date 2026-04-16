import {useState, useEffect, useRef, useCallback} from 'react';
import {predictSingle, predictBatch, checkHealth, Detection} from './api';
import {RejectionCard} from './components/RejectionCard';
import {ConfidenceGauge} from './components/ConfidenceGauge';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer, Cell,
} from 'recharts';

/* ─── Tooltip helper ───────────────────────────────────────────── */
function Tip({text, children}: {text: string; children: React.ReactNode}) {
    return (
        <span className="relative group inline-block cursor-help">
            {children}
            <span className="pointer-events-none absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-1
                             w-52 rounded bg-gray-800 px-2 py-1 text-xs text-white leading-snug
                             opacity-0 group-hover:opacity-100 transition-opacity whitespace-normal text-center">
                {text}
            </span>
        </span>
    );
}

/* ─── Spinner ─────────────────────────────────────────────────── */
function Spinner() {
    return (
        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8h8a8 8 0 01-8 8 8 8 0 01-8-8z"/>
        </svg>
    );
}

/* ─── DropZone ────────────────────────────────────────────────── */
interface DropZoneProps {
    accept: string;
    label: string;
    hint: string;
    file: File | null;
    onFile: (f: File) => void;
    disabled?: boolean;
}
function DropZone({accept, label, hint, file, onFile, disabled}: DropZoneProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragging, setDragging] = useState(false);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
    }, [onFile]);

    return (
        <div
            onClick={() => !disabled && inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={[
                'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 cursor-pointer transition-colors',
                disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-50',
                dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300',
            ].join(' ')}
        >
            <input
                ref={inputRef}
                type="file"
                accept={accept}
                className="hidden"
                disabled={disabled}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
            />
            <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
            <p className="text-sm font-medium text-gray-700">{label}</p>
            <p className="text-xs text-gray-500">{file ? file.name : hint}</p>
        </div>
    );
}

/* ─── Main App ────────────────────────────────────────────────── */
export default function App() {
    const [singleFile, setSingleFile] = useState<File | null>(null);
    const [batchFile, setBatchFile]   = useState<File | null>(null);
    const [result, setResult]         = useState<Detection | null>(null);
    const [batchResults, setBatchResults] = useState<Detection[] | null>(null);
    const [busy, setBusy]             = useState(false);
    const [error, setError]           = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [backendOk, setBackendOk]   = useState<boolean | null>(null); // null = checking

    /* Health-check on mount and every 30 s */
    useEffect(() => {
        let cancelled = false;
        const poll = async () => {
            const ok = await checkHealth();
            if (!cancelled) setBackendOk(ok);
        };
        poll();
        const id = setInterval(poll, 30_000);
        return () => { cancelled = true; clearInterval(id); };
    }, []);

    /* Image preview */
    useEffect(() => {
        if (!singleFile) { setPreviewUrl(null); return; }
        const url = URL.createObjectURL(singleFile);
        setPreviewUrl(url);
        return () => URL.revokeObjectURL(url);
    }, [singleFile]);

    const handleSingle = async () => {
        if (!singleFile) return;
        setBusy(true); setError(null); setResult(null);
        try {
            setResult(await predictSingle(singleFile));
        } catch (e: any) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    };

    const handleBatch = async () => {
        if (!batchFile) return;
        setBusy(true); setError(null); setBatchResults(null);
        try {
            setBatchResults(await predictBatch(batchFile));
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

    const CHART_COLOURS = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899'];

    return (
        <div className="min-h-screen bg-gradient-to-br from-sky-50 to-blue-100">
            {/* Header */}
            <header className="bg-white shadow-sm sticky top-0 z-20">
                <div className="mx-auto max-w-4xl px-4 py-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                        <span className="text-2xl select-none">🐋</span>
                        <h1 className="text-lg sm:text-xl font-bold text-gray-900 leading-tight">
                            EcoMarineAI
                        </h1>
                    </div>
                    {/* Backend status pill */}
                    <Tip text={
                        backendOk === null ? 'Проверяем доступность сервера анализа...' :
                        backendOk ? 'Сервер анализа работает и готов к запросам' :
                        'Сервер анализа недоступен. Убедитесь, что backend запущен'
                    }>
                        <span className={[
                            'flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full',
                            backendOk === null ? 'bg-gray-100 text-gray-500' :
                            backendOk ? 'bg-emerald-100 text-emerald-700' :
                            'bg-red-100 text-red-700',
                        ].join(' ')}>
                            <span className={[
                                'w-2 h-2 rounded-full',
                                backendOk === null ? 'bg-gray-400 animate-pulse' :
                                backendOk ? 'bg-emerald-500' : 'bg-red-500 animate-pulse',
                            ].join(' ')}/>
                            {backendOk === null ? 'Проверка...' : backendOk ? 'Сервер готов' : 'Сервер недоступен'}
                        </span>
                    </Tip>
                </div>
            </header>

            <main className="mx-auto max-w-4xl px-4 py-6 space-y-6">

                {/* Intro card — shown until first result */}
                {!result && !batchResults && (
                    <div className="bg-white rounded-2xl shadow-sm border border-blue-100 p-5">
                        <h2 className="font-semibold text-gray-800 mb-2">Как использовать</h2>
                        <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                            <li>Выберите или перетащите <strong>фото одного кита</strong> (JPG/PNG) в секцию ниже</li>
                            <li>Нажмите <strong>«Определить»</strong> — модель назовёт вид и уникальный ID особи</li>
                            <li>Для нескольких снимков загрузите <strong>ZIP-архив</strong> в секцию пакетной обработки</li>
                        </ol>
                        <p className="mt-3 text-xs text-gray-400">
                            Поддерживаются аэрофотоснимки китов и дельфинов · Рекомендуемое разрешение 1920×1080+
                        </p>
                    </div>
                )}

                {/* ── Section 1: Single ─────────────────────────── */}
                <section className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="px-5 pt-5 pb-4 border-b border-gray-50">
                        <h2 className="font-semibold text-gray-900 text-base">
                            Одиночная обработка
                        </h2>
                        <p className="text-xs text-gray-500 mt-0.5">
                            Загрузите снимок одного животного для идентификации вида и особи
                        </p>
                    </div>
                    <div className="p-5 space-y-4">
                        <DropZone
                            accept="image/*"
                            label="Перетащите фото или нажмите для выбора"
                            hint="JPG, PNG, WEBP · любое разрешение"
                            file={singleFile}
                            onFile={(f) => { setSingleFile(f); setResult(null); }}
                            disabled={busy}
                        />

                        {previewUrl && (
                            <div className="rounded-xl overflow-hidden border border-gray-200">
                                <img src={previewUrl} alt="Предпросмотр" className="w-full max-h-72 object-contain bg-gray-50"/>
                            </div>
                        )}

                        <button
                            onClick={handleSingle}
                            disabled={!singleFile || busy}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            {busy ? <><Spinner/>Анализ...</> : 'Определить'}
                        </button>

                        {result && result.rejected && <RejectionCard result={result}/>}

                        {result && !result.rejected && (
                            <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4 space-y-3">
                                <div className="flex items-center gap-2">
                                    <svg className="w-5 h-5 text-emerald-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7"/>
                                    </svg>
                                    <span className="font-semibold text-emerald-900">Млекопитающее идентифицировано</span>
                                </div>
                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div className="bg-white rounded-lg p-3 border border-emerald-100">
                                        <Tip text="Биологический вид животного, определённый моделью по 30 видам китообразных">
                                            <p className="text-xs text-gray-500 mb-1 underline decoration-dotted">Вид</p>
                                        </Tip>
                                        <p className="font-semibold text-gray-900 capitalize">{result.id_animal.replace(/_/g,' ')}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-3 border border-emerald-100">
                                        <Tip text="Уникальный идентификатор конкретной особи по 13 837 известным китам. Совпадение по хвостовому плавнику.">
                                            <p className="text-xs text-gray-500 mb-1 underline decoration-dotted">ID особи</p>
                                        </Tip>
                                        <p className="font-mono text-xs text-gray-700 break-all">{result.class_animal || '—'}</p>
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <Tip text="Уверенность модели в правильности идентификации конкретной особи (ArcFace метрическое обучение)">
                                        <div className="underline decoration-dotted inline-block mb-1 text-xs text-gray-600">
                                            Уверенность идентификации
                                        </div>
                                    </Tip>
                                    <ConfidenceGauge
                                        label=""
                                        value={result.probability}
                                        variant={result.probability >= 0.6 ? 'success' : result.probability >= 0.3 ? 'warning' : 'danger'}
                                    />
                                    {typeof result.cetacean_score === 'number' && (
                                        <>
                                            <Tip text="Антифрод-фильтр: вероятность того, что на фото именно кит или дельфин, а не посторонний объект (порог ≥ 50%)">
                                                <div className="underline decoration-dotted inline-block mb-1 text-xs text-gray-600">
                                                    Антифрод-скор (кит/дельфин)
                                                </div>
                                            </Tip>
                                            <ConfidenceGauge
                                                label=""
                                                value={result.cetacean_score}
                                                variant="success"
                                            />
                                        </>
                                    )}
                                </div>
                                {result.model_version && (
                                    <p className="text-xs text-gray-400 text-right">
                                        Модель: {result.model_version}
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                </section>

                {/* ── Section 2: Batch ──────────────────────────── */}
                <section className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="px-5 pt-5 pb-4 border-b border-gray-50">
                        <h2 className="font-semibold text-gray-900 text-base">
                            Пакетная обработка
                        </h2>
                        <p className="text-xs text-gray-500 mt-0.5">
                            Загрузите ZIP-архив с несколькими снимками — получите сводку по всем
                        </p>
                    </div>
                    <div className="p-5 space-y-4">
                        <DropZone
                            accept=".zip"
                            label="Перетащите ZIP-архив или нажмите для выбора"
                            hint="Архив с фото китов · .zip"
                            file={batchFile}
                            onFile={(f) => { setBatchFile(f); setBatchResults(null); }}
                            disabled={busy}
                        />

                        <button
                            onClick={handleBatch}
                            disabled={!batchFile || busy}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            {busy ? <><Spinner/>Обработка архива...</> : 'Обработать архив'}
                        </button>

                        {batchResults && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="font-medium text-gray-800">
                                        Результаты: {batchResults.length} снимков
                                    </h3>
                                    <span className="text-xs text-gray-500">
                                        {batchResults.filter(d => !d.rejected).length} опознано,{' '}
                                        {batchResults.filter(d => d.rejected).length} отклонено
                                    </span>
                                </div>

                                {chartData.length > 0 && (
                                    <div>
                                        <p className="text-xs text-gray-500 mb-2 text-center">
                                            Распределение по видам
                                        </p>
                                        <ResponsiveContainer width="100%" height={250}>
                                            <BarChart data={chartData} margin={{top:5, right:10, left:-20, bottom:40}}>
                                                <XAxis dataKey="name" angle={-35} textAnchor="end" tick={{fontSize:10}}/>
                                                <YAxis tick={{fontSize:11}} allowDecimals={false}/>
                                                <ChartTooltip/>
                                                <Bar dataKey="count" radius={[4,4,0,0]}>
                                                    {chartData.map((_,i) => (
                                                        <Cell key={i} fill={CHART_COLOURS[i % CHART_COLOURS.length]}/>
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}

                                {/* Top results table */}
                                <div className="overflow-x-auto rounded-lg border border-gray-200">
                                    <table className="w-full text-xs">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="px-3 py-2 text-left text-gray-600 font-medium">Файл</th>
                                                <th className="px-3 py-2 text-left text-gray-600 font-medium">Вид</th>
                                                <th className="px-3 py-2 text-right text-gray-600 font-medium">Уверенность</th>
                                                <th className="px-3 py-2 text-center text-gray-600 font-medium">Статус</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {batchResults.slice(0,20).map((d, i) => (
                                                <tr key={i} className="hover:bg-gray-50">
                                                    <td className="px-3 py-2 text-gray-700 max-w-[140px] truncate">{d.image_ind}</td>
                                                    <td className="px-3 py-2 text-gray-700 capitalize">{d.id_animal.replace(/_/g,' ')}</td>
                                                    <td className="px-3 py-2 text-right text-gray-700">{(d.probability*100).toFixed(1)}%</td>
                                                    <td className="px-3 py-2 text-center">
                                                        {d.rejected
                                                            ? <span className="text-red-500">✗</span>
                                                            : <span className="text-emerald-500">✓</span>}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                    {batchResults.length > 20 && (
                                        <p className="text-xs text-gray-400 text-center py-2">
                                            Показано 20 из {batchResults.length} результатов
                                        </p>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </section>

            </main>

            {/* Error Modal */}
            {error && (
                <div
                    className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4"
                    onClick={() => setError(null)}
                >
                    <div
                        className="bg-white rounded-2xl shadow-2xl max-w-sm w-full p-6"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center gap-3 mb-3">
                            <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">
                                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                </svg>
                            </div>
                            <h3 className="font-semibold text-gray-900">Ошибка</h3>
                        </div>
                        <p className="text-sm text-gray-600 mb-5">{error}</p>
                        <button
                            onClick={() => setError(null)}
                            className="w-full px-4 py-2 bg-gray-900 text-white rounded-xl hover:bg-gray-700 transition-colors text-sm font-medium"
                        >
                            Закрыть
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
