import {useState, useEffect, useRef, useCallback, DragEvent} from 'react';
import {predictSingle, predictBatch, checkHealth, Detection} from './api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip,
  ResponsiveContainer, Cell, CartesianGrid,
} from 'recharts';

// ─── Species display names ──────────────────────────────────────────────────
const SPECIES_RU: Record<string, string> = {
  killer_whale: 'Косатка', humpback_whale: 'Горбатый кит',
  bottlenose_dolphin: 'Афалина', fin_whale: 'Финвал',
  blue_whale: 'Синий кит', sperm_whale: 'Кашалот',
  gray_whale: 'Серый кит', minke_whale: 'Малый полосатик',
  beluga_whale: 'Белуха', narwhal: 'Нарвал',
  right_whale: 'Гладкий кит', bowhead_whale: 'Гренландский кит',
  sei_whale: 'Сейвал', bryde_whale: 'Кит Брайда',
  pilot_whale: 'Гринда', common_dolphin: 'Обыкновенный дельфин',
  spinner_dolphin: 'Вертячка', striped_dolphin: 'Полосатый дельфин',
  pantropical_spotted_dolphin: 'Пятнистый дельфин',
  risso_dolphin: "Дельфин Риссо", orca: 'Косатка',
  false_killer_whale: 'Малая косатка', pygmy_killer_whale: 'Карликовая косатка',
  melon_headed_whale: 'Дыневидная косатка',
};
function speciesName(key: string): string {
  return SPECIES_RU[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Tooltip ────────────────────────────────────────────────────────────────
function Tip({text, children, above = false}: {text: string; children: React.ReactNode; above?: boolean}) {
  return (
    <span className="relative group/tip inline-flex items-center gap-1">
      {children}
      <span className="pointer-events-none absolute z-50 left-0 w-56 rounded-lg bg-gray-900 px-3 py-2
                       text-xs text-white leading-relaxed shadow-xl
                       opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150
                       whitespace-normal"
        style={{[above ? 'bottom' : 'top']: '100%', marginTop: above ? 0 : 6, marginBottom: above ? 6 : 0}}>
        {text}
      </span>
    </span>
  );
}

function InfoIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" strokeWidth="2"/>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 16v-4m0-4h.01"/>
    </svg>
  );
}

// ─── Spinner ────────────────────────────────────────────────────────────────
function Spinner({size = 5}: {size?: number}) {
  return (
    <svg className={`animate-spin w-${size} h-${size}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8v8h8a8 8 0 01-8 8 8 8 0 01-8-8z"/>
    </svg>
  );
}

// ─── Confidence bar ─────────────────────────────────────────────────────────
function ConfBar({value, label, tooltip}: {value: number; label: string; tooltip?: string}) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = pct >= 70 ? '#06b6d4' : pct >= 40 ? '#f59e0b' : '#ef4444';
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center text-xs">
        <span className="flex items-center gap-1 text-slate-500">
          {tooltip
            ? <Tip text={tooltip}><span className="underline decoration-dotted cursor-help">{label}</span><InfoIcon/></Tip>
            : label}
        </span>
        <span className="font-semibold text-slate-700 tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-slate-200">
        <div className="conf-bar h-1.5 rounded-full" style={{width: `${pct}%`, background: color}}/>
      </div>
    </div>
  );
}

// ─── Drop zone ───────────────────────────────────────────────────────────────
interface DropZoneProps {
  accept: string; hint: string; icon: React.ReactNode;
  file: File | null; onFile: (f: File) => void; disabled?: boolean;
}
function DropZone({accept, hint, icon, file, onFile, disabled}: DropZoneProps) {
  const ref = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault(); setOver(false);
    const f = e.dataTransfer.files[0];
    if (f && !disabled) onFile(f);
  }, [onFile, disabled]);

  return (
    <div
      onClick={() => !disabled && ref.current?.click()}
      onDragOver={e => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
      className={[
        'group relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed',
        'py-10 px-6 cursor-pointer transition-all duration-200 select-none',
        over || file ? 'border-cyan-400 bg-cyan-50' : 'border-slate-200 bg-white hover:border-cyan-300 hover:bg-slate-50',
        disabled ? 'opacity-50 cursor-not-allowed' : '',
      ].filter(Boolean).join(' ')}
    >
      <input ref={ref} type="file" accept={accept} className="hidden" disabled={disabled}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }}/>

      <div className={['w-12 h-12 rounded-full flex items-center justify-center transition-colors',
        file ? 'bg-cyan-100 text-cyan-600' : 'bg-slate-100 text-slate-400 group-hover:bg-cyan-100 group-hover:text-cyan-500',
      ].join(' ')}>
        {icon}
      </div>

      {file ? (
        <div className="text-center">
          <p className="text-sm font-medium text-cyan-700 truncate max-w-[260px]">{file.name}</p>
          <p className="text-xs text-slate-400 mt-0.5">{(file.size / 1024).toFixed(0)} KB · нажмите чтобы изменить</p>
        </div>
      ) : (
        <div className="text-center">
          <p className="text-sm font-medium text-slate-600">Перетащите файл или нажмите для выбора</p>
          <p className="text-xs text-slate-400 mt-1">{hint}</p>
        </div>
      )}
    </div>
  );
}

// ─── Detection result card ───────────────────────────────────────────────────
function DetectionCard({result, imageUrl}: {result: Detection; imageUrl?: string | null}) {
  if (result.rejected) {
    const reasonText: Record<string, string> = {
      not_a_marine_mammal: 'На фото не обнаружено китообразное. Загрузите аэрофотоснимок кита или дельфина.',
      low_confidence: 'Модель обнаружила возможное китообразное, но уверенность слишком мала для идентификации. Попробуйте более чёткий или крупный снимок.',
      corrupted_image: 'Файл изображения повреждён или не может быть декодирован.',
    };
    return (
      <div className="slide-up mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-5 flex gap-4">
        <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
          <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
          </svg>
        </div>
        <div>
          <p className="font-semibold text-amber-900 text-sm">Снимок отклонён антифрод-фильтром</p>
          <p className="text-sm text-amber-800 mt-1">{reasonText[result.rejection_reason ?? ''] ?? 'Неизвестная причина отклонения.'}</p>
          {typeof result.cetacean_score === 'number' && (
            <div className="mt-3 w-48">
              <ConfBar value={result.cetacean_score} label="Уверенность «это кит»"
                tooltip="Антифрод-фильтр на базе CLIP. Порог срабатывания: 50%"/>
            </div>
          )}
        </div>
      </div>
    );
  }

  const conf = result.probability;
  const tier = conf >= 0.7 ? {label: 'Высокая', color: 'text-emerald-700', bg: 'bg-emerald-50', border: 'border-emerald-200'}
             : conf >= 0.4 ? {label: 'Средняя',  color: 'text-amber-700',   bg: 'bg-amber-50',   border: 'border-amber-200'}
             :               {label: 'Низкая',    color: 'text-red-700',     bg: 'bg-red-50',     border: 'border-red-200'};

  return (
    <div className={`slide-up mt-5 rounded-2xl border ${tier.border} ${tier.bg} overflow-hidden`}>
      {/* Header stripe */}
      <div className="bg-ocean-800 px-5 py-3 flex items-center justify-between" style={{background:'#0f2744'}}>
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z" opacity=".3"/>
            <path d="M12 6a1 1 0 110 2 1 1 0 010-2zm0 4a1 1 0 011 1v5a1 1 0 11-2 0v-5a1 1 0 011-1z"/>
          </svg>
          <span className="text-xs font-medium text-cyan-200 tracking-wide uppercase">Результат идентификации</span>
        </div>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${tier.color} ${tier.bg}`}>
          {tier.label} уверенность
        </span>
      </div>

      <div className="p-5 grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Left: image */}
        {imageUrl && (
          <div className="sm:row-span-2">
            <p className="text-xs font-medium text-slate-500 mb-2 uppercase tracking-wide">Загруженный снимок</p>
            <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
              <img src={imageUrl} alt="Whale" className="w-full object-contain max-h-56"/>
            </div>
          </div>
        )}

        {/* Right: species + ID */}
        <div className="space-y-4">
          <div>
            <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wide flex items-center gap-1">
              Вид
              <Tip text="Биологический вид животного. Определяется по морфологии плавника из 30 видов китообразных." above>
                <InfoIcon/>
              </Tip>
            </p>
            <p className="text-xl font-bold text-slate-900">{speciesName(result.id_animal)}</p>
            <p className="text-xs text-slate-400 italic mt-0.5">{result.id_animal}</p>
          </div>

          <div>
            <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wide flex items-center gap-1">
              Идентификатор особи
              <Tip text="Уникальный ID конкретного животного из базы 13 837 особей. Метрическое обучение ArcFace по хвостовому / спинному плавнику." above>
                <InfoIcon/>
              </Tip>
            </p>
            <code className="font-mono text-sm bg-slate-100 text-slate-700 px-2 py-1 rounded-lg block break-all">
              {result.class_animal || '—'}
            </code>
          </div>
        </div>

        {/* Scores */}
        <div className="space-y-3">
          <ConfBar value={result.probability}
            label="Уверенность идентификации"
            tooltip="Вероятность совпадения с конкретной особью в базе. Значения ≥70% — надёжная идентификация; 40–69% — предварительная; <40% — требует проверки."/>
          {typeof result.cetacean_score === 'number' && (
            <ConfBar value={result.cetacean_score}
              label="Антифрод-скор (CLIP)"
              tooltip="Вероятность того, что снимок содержит китообразное, по мнению CLIP-фильтра. Порог: 50%. Защита от случайных загрузок."/>
          )}
        </div>
      </div>

      {/* Footer meta */}
      {result.model_version && (
        <div className="px-5 py-2.5 border-t border-slate-200 bg-white/50 flex items-center justify-between">
          <span className="text-xs text-slate-400">Модель: <code className="font-mono">{result.model_version}</code></span>
          <span className="text-xs text-slate-400">bbox: {result.bbox.join(', ')}</span>
        </div>
      )}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [batchFile, setBatchFile]   = useState<File | null>(null);
  const [result, setResult]         = useState<Detection | null>(null);
  const [batchResults, setBatchResults] = useState<Detection[] | null>(null);
  const [busy, setBusy]             = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [backendOk, setBackendOk]   = useState<boolean | null>(null);
  const [tab, setTab]               = useState<'single' | 'batch'>('single');

  // Backend health poll
  useEffect(() => {
    let alive = true;
    const check = async () => { const ok = await checkHealth(); if (alive) setBackendOk(ok); };
    check();
    const id = setInterval(check, 20_000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Image preview
  useEffect(() => {
    if (!singleFile) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(singleFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [singleFile]);

  const handleSingle = async () => {
    if (!singleFile) return;
    setBusy(true); setError(null); setResult(null);
    try { setResult(await predictSingle(singleFile)); }
    catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  };

  const handleBatch = async () => {
    if (!batchFile) return;
    setBusy(true); setError(null); setBatchResults(null);
    try { setBatchResults(await predictBatch(batchFile)); }
    catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  };

  const chartData = batchResults
    ? Object.entries(batchResults.reduce<Record<string,number>>((acc, d) => {
        acc[speciesName(d.id_animal)] = (acc[speciesName(d.id_animal)] || 0) + 1;
        return acc;
      }, {})).map(([name, count]) => ({name, count}))
    : [];

  const CHART_COLORS = ['#0096c7','#06b6d4','#22d3ee','#67e8f9','#a5f3fc','#0077b6'];

  return (
    <div className="min-h-screen" style={{background:'#f0f4f8'}}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <header style={{background:'#0f2744'}} className="sticky top-0 z-30 shadow-lg">
        <div className="mx-auto max-w-5xl px-4 h-14 flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{background:'#0077b6'}}>
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064"/>
              </svg>
            </div>
            <div>
              <p className="text-white font-semibold text-sm leading-none">EcoMarineAI</p>
              <p className="text-cyan-300 text-[10px] leading-none mt-0.5">Идентификация китообразных</p>
            </div>
          </div>

          {/* Backend status */}
          <div className={[
            'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors',
            backendOk === null ? 'bg-slate-700 text-slate-300' :
            backendOk ? 'bg-emerald-900/60 text-emerald-300' :
            'bg-red-900/60 text-red-300',
          ].join(' ')}>
            <span className={[
              'relative w-2 h-2 rounded-full flex-shrink-0',
              backendOk === null ? 'bg-slate-400' :
              backendOk ? 'bg-emerald-400' : 'bg-red-400',
              backendOk === null || !backendOk ? 'animate-pulse' : '',
            ].join(' ')}/>
            {backendOk === null ? 'Проверка...' : backendOk ? 'Сервер готов' : 'Сервер недоступен'}
          </div>
        </div>

        {/* Tabs */}
        <div className="mx-auto max-w-5xl px-4 flex gap-1 pb-0">
          {(['single','batch'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={[
                'px-5 py-2.5 text-sm font-medium rounded-t-lg transition-colors',
                tab === t
                  ? 'bg-[#f0f4f8] text-[#0077b6]'
                  : 'text-cyan-300 hover:text-white hover:bg-white/10',
              ].join(' ')}>
              {t === 'single' ? 'Одиночный анализ' : 'Пакетная обработка'}
            </button>
          ))}
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────────── */}
      <main className="mx-auto max-w-5xl px-4 py-6">

        {/* ── Tab: Single ─────────────────────────────────────── */}
        {tab === 'single' && (
          <div className="space-y-5">
            {/* Upload + action panel */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
              <div>
                <h2 className="font-semibold text-slate-800">Загрузка снимка</h2>
                <p className="text-sm text-slate-500 mt-0.5">
                  Аэрофотоснимок одного животного · JPG, PNG, WEBP · рекомендуется 1920×1080+
                </p>
              </div>

              <DropZone
                accept="image/*"
                hint="JPG, PNG, WEBP — любое разрешение"
                file={singleFile}
                onFile={f => { setSingleFile(f); setResult(null); }}
                disabled={busy}
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z"/>
                  </svg>
                }
              />

              {/* Preview */}
              {previewUrl && (
                <div className="rounded-xl overflow-hidden border border-slate-200">
                  <img src={previewUrl} alt="Preview" className="w-full object-contain max-h-64 bg-slate-50"/>
                </div>
              )}

              <button onClick={handleSingle} disabled={!singleFile || busy}
                className="w-full h-11 rounded-xl flex items-center justify-center gap-2 text-sm font-semibold
                           text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                style={{background: singleFile && !busy ? '#0077b6' : undefined,
                        backgroundColor: (!singleFile || busy) ? '#94a3b8' : undefined}}>
                {busy ? <><Spinner/> Анализ снимка...</> : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"/>
                    </svg>
                    Определить вид и особь
                  </>
                )}
              </button>
            </div>

            {/* Result */}
            {result && <DetectionCard result={result} imageUrl={previewUrl}/>}

            {/* Empty state */}
            {!result && !busy && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                  Как работает система
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {[
                    {step:'1', title:'Загрузите снимок', desc:'Аэрофото кита или дельфина. Оптимально — чёткое изображение плавника на фоне воды.'},
                    {step:'2', title:'ML-анализ', desc:'EfficientNet-B4 + ArcFace определяют вид из 30 классов и особь из 13 837 китов в базе.'},
                    {step:'3', title:'Результат', desc:'Вид, уникальный ID особи, уверенность модели. Данные пригодны для научных отчётов.'},
                  ].map(({step,title,desc}) => (
                    <div key={step} className="flex gap-3">
                      <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold text-white"
                        style={{background:'#0077b6'}}>
                        {step}
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-700">{title}</p>
                        <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{desc}</p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-5 pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    {v:'30', u:'видов', sub:'китообразных'},
                    {v:'13 837', u:'особей', sub:'в базе данных'},
                    {v:'93.6%', u:'точность', sub:'Precision@1'},
                    {v:'<550 мс', u:'задержка', sub:'p95 latency'},
                  ].map(({v,u,sub}) => (
                    <div key={u} className="text-center p-3 rounded-xl bg-slate-50 border border-slate-100">
                      <p className="text-lg font-bold" style={{color:'#0077b6'}}>{v}</p>
                      <p className="text-xs font-semibold text-slate-600">{u}</p>
                      <p className="text-[10px] text-slate-400">{sub}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Tab: Batch ──────────────────────────────────────── */}
        {tab === 'batch' && (
          <div className="space-y-5">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
              <div>
                <h2 className="font-semibold text-slate-800">Пакетная обработка</h2>
                <p className="text-sm text-slate-500 mt-0.5">
                  ZIP-архив с несколькими снимками — система обработает все и вернёт сводку
                </p>
              </div>

              <DropZone
                accept=".zip"
                hint=".zip архив со снимками китообразных"
                file={batchFile}
                onFile={f => { setBatchFile(f); setBatchResults(null); }}
                disabled={busy}
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"/>
                  </svg>
                }
              />

              <button onClick={handleBatch} disabled={!batchFile || busy}
                className="w-full h-11 rounded-xl flex items-center justify-center gap-2 text-sm font-semibold
                           text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                style={{background: batchFile && !busy ? '#0077b6' : undefined,
                        backgroundColor: (!batchFile || busy) ? '#94a3b8' : undefined}}>
                {busy ? <><Spinner/> Обработка архива...</> : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/>
                    </svg>
                    Обработать архив
                  </>
                )}
              </button>
            </div>

            {/* Batch results */}
            {batchResults && (
              <div className="space-y-4 slide-up">
                {/* Summary row */}
                <div className="grid grid-cols-3 gap-3">
                  {[
                    {v: batchResults.length, label: 'Всего снимков'},
                    {v: batchResults.filter(d=>!d.rejected).length, label: 'Опознано'},
                    {v: batchResults.filter(d=>d.rejected).length, label: 'Отклонено'},
                  ].map(({v,label}) => (
                    <div key={label} className="bg-white rounded-xl border border-slate-200 p-4 text-center">
                      <p className="text-2xl font-bold" style={{color:'#0077b6'}}>{v}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{label}</p>
                    </div>
                  ))}
                </div>

                {/* Chart */}
                {chartData.length > 0 && (
                  <div className="bg-white rounded-2xl border border-slate-200 p-5">
                    <p className="text-sm font-semibold text-slate-700 mb-4">Распределение по видам</p>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={chartData} margin={{top:0, right:10, left:-15, bottom:50}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                        <XAxis dataKey="name" angle={-40} textAnchor="end" tick={{fontSize:11, fill:'#64748b'}} interval={0}/>
                        <YAxis tick={{fontSize:11, fill:'#64748b'}} allowDecimals={false}/>
                        <ChartTooltip contentStyle={{borderRadius:8, border:'1px solid #e2e8f0', fontSize:12}}/>
                        <Bar dataKey="count" radius={[4,4,0,0]}>
                          {chartData.map((_,i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Results table */}
                <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
                  <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between">
                    <p className="text-sm font-semibold text-slate-700">Детали ({batchResults.length} записей)</p>
                    {batchResults.length > 20 && (
                      <p className="text-xs text-slate-400">Показано первые 20</p>
                    )}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
                          <th className="px-4 py-3 text-left font-medium">Файл</th>
                          <th className="px-4 py-3 text-left font-medium">Вид</th>
                          <th className="px-4 py-3 text-left font-medium font-mono">ID особи</th>
                          <th className="px-4 py-3 text-right font-medium">Уверенность</th>
                          <th className="px-4 py-3 text-center font-medium">Статус</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {batchResults.slice(0,20).map((d,i) => (
                          <tr key={i} className="hover:bg-slate-50 transition-colors">
                            <td className="px-4 py-3 text-slate-600 max-w-[130px] truncate text-xs">{d.image_ind}</td>
                            <td className="px-4 py-3 text-slate-800 font-medium text-xs">{speciesName(d.id_animal)}</td>
                            <td className="px-4 py-3">
                              <code className="font-mono text-xs text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                                {d.class_animal?.slice(0,12) || '—'}
                              </code>
                            </td>
                            <td className="px-4 py-3 text-right">
                              <span className={`text-xs font-semibold tabular-nums ${
                                d.probability >= 0.7 ? 'text-emerald-600' :
                                d.probability >= 0.4 ? 'text-amber-600' : 'text-red-500'}`}>
                                {(d.probability*100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="px-4 py-3 text-center">
                              {d.rejected
                                ? <span className="inline-flex items-center gap-1 text-xs text-red-600 bg-red-50 px-2 py-0.5 rounded-full">✗ Отклонён</span>
                                : <span className="inline-flex items-center gap-1 text-xs text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">✓ Принят</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Batch empty state */}
            {!batchResults && !busy && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Формат архива</p>
                <div className="bg-slate-50 rounded-xl p-4 font-mono text-xs text-slate-600 space-y-0.5">
                  <p>📦 archive.zip</p>
                  <p className="pl-4">├── whale_001.jpg</p>
                  <p className="pl-4">├── whale_002.png</p>
                  <p className="pl-4">└── whale_003.jpg</p>
                </div>
                <p className="text-xs text-slate-500 mt-3">
                  Принимаются JPG/PNG/WEBP внутри ZIP. Вложенные папки не поддерживаются.
                  Каждый снимок обрабатывается независимо.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Footer ────────────────────────────────────────────── */}
      <footer style={{background:'#0f2744'}} className="mt-8 py-4 text-center">
        <p className="text-[11px] text-cyan-900" style={{color:'#4a7fa5'}}>
          EcoMarineAI · EfficientNet-B4 ArcFace · 13 837 особей · 30 видов ·
          ФСИ Грант 2024–2025
        </p>
      </footer>

      {/* ── Error modal ──────────────────────────────────────── */}
      {error && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{background:'rgba(2,6,23,0.6)', backdropFilter:'blur(6px)'}}
          onClick={() => setError(null)}>
          <div className="bg-white rounded-2xl shadow-2xl max-w-sm w-full p-6 slide-up"
            onClick={e => e.stopPropagation()}>
            <div className="flex items-start gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <div>
                <p className="font-semibold text-slate-900">Ошибка</p>
                <p className="text-sm text-slate-600 mt-1">{error}</p>
              </div>
            </div>
            <button onClick={() => setError(null)}
              className="w-full h-10 rounded-xl text-sm font-semibold text-white transition-colors"
              style={{background:'#0f2744'}}>
              Закрыть
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
