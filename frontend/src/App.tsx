import {useState, useEffect, useRef, useCallback, DragEvent} from 'react';
import {predictSingle, predictBatch, checkHealth, Detection, Candidate} from './api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip,
  ResponsiveContainer, Cell, CartesianGrid,
} from 'recharts';

// ─── Species display names ──────────────────────────────────────────────────
const SPECIES_RU: Record<string, string> = {
  killer_whale: 'Косатка', orca: 'Косатка',
  humpback_whale: 'Горбатый кит',
  bottlenose_dolphin: 'Афалина',
  fin_whale: 'Финвал',
  blue_whale: 'Синий кит',
  sperm_whale: 'Кашалот',
  gray_whale: 'Серый кит',
  minke_whale: 'Малый полосатик',
  beluga_whale: 'Белуха',
  narwhal: 'Нарвал',
  right_whale: 'Гладкий кит',
  bowhead_whale: 'Гренландский кит',
  sei_whale: 'Сейвал',
  bryde_whale: 'Кит Брайда',
  pilot_whale: 'Гринда',
  common_dolphin: 'Обыкновенный дельфин',
  spinner_dolphin: 'Вертячка',
  striped_dolphin: 'Полосатый дельфин',
  pantropical_spotted_dolphin: 'Пятнистый дельфин',
  risso_dolphin: "Дельфин Риссо",
  false_killer_whale: 'Малая косатка',
  pygmy_killer_whale: 'Карликовая косатка',
  melon_headed_whale: 'Дыневидная косатка',
  cetacean_unidentified: 'Морское млекопитающее (не определено)',
};

function speciesName(key: string): string {
  return SPECIES_RU[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Tooltip ────────────────────────────────────────────────────────────────
function Tip({text, children}: {text: string; children: React.ReactNode}) {
  return (
    <span className="relative group/tip inline-flex items-center gap-0.5 cursor-help">
      {children}
      <span className="pointer-events-none absolute z-50 left-0 top-full mt-1.5 w-56 rounded-lg
                       bg-gray-900 px-3 py-2 text-xs text-white leading-relaxed shadow-xl
                       opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150 whitespace-normal">
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
    <svg className={`animate-spin w-${size} h-${size} flex-shrink-0`}
      xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-80" fill="currentColor"
        d="M4 12a8 8 0 018-8v8h8a8 8 0 01-8 8 8 8 0 01-8-8z"/>
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
        <span className="text-slate-500 flex items-center gap-1">
          {tooltip
            ? <Tip text={tooltip}><span className="underline decoration-dotted">{label}</span><InfoIcon/></Tip>
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
        'flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed',
        'py-8 px-6 cursor-pointer transition-all duration-200 select-none',
        over || file ? 'border-cyan-400 bg-cyan-50' : 'border-slate-200 bg-white hover:border-cyan-300 hover:bg-slate-50',
        disabled ? 'opacity-60 cursor-not-allowed pointer-events-none' : '',
      ].filter(Boolean).join(' ')}
    >
      <input ref={ref} type="file" accept={accept} className="hidden" disabled={disabled}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ''; }}/>

      <div className={['w-12 h-12 rounded-full flex items-center justify-center transition-colors',
        file ? 'bg-cyan-100 text-cyan-600' : 'bg-slate-100 text-slate-400 group-hover:bg-cyan-100',
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

// ─── Rejection card ──────────────────────────────────────────────────────────
function RejectionCard({result}: {result: Detection}) {
  const isNotMammal = result.rejection_reason === 'not_a_marine_mammal';
  const isLowConf  = result.rejection_reason === 'low_confidence';

  const subject = encodeURIComponent('EcoMarineAI — новая особь для базы данных');
  const body = encodeURIComponent(
    `Здравствуйте!\n\nЯ загрузил(а) снимок морского млекопитающего, которое система не смогла опознать.\n\nПричина отклонения: ${result.rejection_reason ?? 'неизвестна'}\nАнтифрод-скор: ${Math.round((result.cetacean_score ?? 0) * 100)}%\n\nПожалуйста, рассмотрите добавление этой особи в базу данных. Прикрепляю снимок.`
  );
  const mailto = `mailto:vandanov2010@gmail.com?subject=${subject}&body=${body}`;

  return (
    <div className="slide-up mt-5 rounded-2xl overflow-hidden border border-amber-200">
      {/* Header */}
      <div className="bg-amber-50 px-5 py-4 border-b border-amber-100">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
            </svg>
          </div>
          <div>
            <p className="font-semibold text-amber-900 text-sm">
              {isNotMammal
                ? 'Не является морским млекопитающим'
                : isLowConf
                  ? 'Особь не найдена в базе данных'
                  : 'Снимок отклонён'}
            </p>
            <p className="text-xs text-amber-700 mt-0.5">
              {isNotMammal
                ? 'Система не обнаружила признаков морского млекопитающего на этом снимке.'
                : isLowConf
                  ? 'Животное опознано как морское млекопитающее, но конкретная особь отсутствует в базе или снимок недостаточно чёткий.'
                  : 'Изображение не прошло проверку.'}
            </p>
          </div>
        </div>
      </div>

      {/* Scores */}
      <div className="bg-white px-5 py-4 space-y-3">
        {typeof result.cetacean_score === 'number' && (
          <ConfBar value={result.cetacean_score}
            label="Уверенность «это морское млекопитающее»"
            tooltip="Антифрод-фильтр на базе CLIP. Порог срабатывания: 50%"/>
        )}

        {isLowConf && typeof result.probability === 'number' && result.probability > 0 && (
          <ConfBar value={result.probability}
            label="Уверенность идентификации особи"
            tooltip="Вероятность совпадения с ближайшей особью из базы. Слишком мала для надёжного определения."/>
        )}

        {/* Что делать */}
        <div className="pt-2 space-y-2">
          <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Что делать</p>
          {isNotMammal && (
            <p className="text-sm text-slate-600">
              Загрузите фотографию морского млекопитающего — кита, дельфина, морского котика и других.
              Рекомендуется аэрофотоснимок с хорошо видимым плавником.
            </p>
          )}
          {isLowConf && (
            <p className="text-sm text-slate-600">
              Попробуйте загрузить более чёткий снимок с хорошо видимым спинным или хвостовым плавником.
              Если вы уверены, что это новая или редкая особь — отправьте снимок нам для добавления в базу.
            </p>
          )}
        </div>

        {/* Кнопка отправки */}
        <a href={mailto}
          className="mt-2 flex items-center justify-center gap-2 w-full rounded-xl px-4 py-2.5
                     text-sm font-semibold transition-colors border"
          style={{color:'#0077b6', borderColor:'#bae6fd', background:'#f0f9ff'}}>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
          Отправить снимок для добавления в базу
        </a>
        <p className="text-xs text-slate-400 text-center">
          vandanov2010@gmail.com — не прикладывайте архивы, только JPEG/PNG
        </p>
      </div>
    </div>
  );
}

// ─── Detection result card ───────────────────────────────────────────────────
function DetectionCard({result, imageUrl}: {result: Detection; imageUrl?: string | null}) {
  if (result.rejected) return <RejectionCard result={result}/>;

  const conf = result.probability;
  const tier = conf >= 0.7
    ? {label:'Высокая', color:'text-emerald-700', bg:'bg-emerald-100', border:'border-emerald-200', strip:'bg-emerald-700'}
    : conf >= 0.4
    ? {label:'Средняя', color:'text-amber-700', bg:'bg-amber-100', border:'border-amber-200', strip:'bg-amber-700'}
    : {label:'Низкая',  color:'text-red-700',   bg:'bg-red-100',   border:'border-red-200',   strip:'bg-red-700'};

  const candidates: Candidate[] = result.candidates ?? [];

  return (
    <div className={`slide-up mt-5 rounded-2xl border ${tier.border} overflow-hidden`}>
      {/* Header strip */}
      <div className="px-5 py-3 flex items-center justify-between" style={{background:'#0f2744'}}>
        <span className="text-xs font-semibold text-cyan-200 uppercase tracking-widest">
          Результат идентификации
        </span>
        <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${tier.color} ${tier.bg}`}>
          {tier.label} уверенность
        </span>
      </div>

      {/* Body */}
      <div className="bg-white p-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {/* Image column */}
          {imageUrl && (
            <div>
              <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Загруженный снимок
              </p>
              <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
                <img src={imageUrl} alt="Снимок" className="w-full object-contain max-h-52"/>
              </div>
            </div>
          )}

          {/* Info column */}
          <div className="space-y-4">
            {/* Species */}
            <div>
              <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                Вид
                <Tip text="Биологический вид, определённый моделью из 30 видов морских млекопитающих.">
                  <InfoIcon/>
                </Tip>
              </p>
              <p className="text-2xl font-bold text-slate-900 leading-tight">{speciesName(result.id_animal)}</p>
              <p className="text-xs text-slate-400 italic mt-0.5">{result.id_animal}</p>
            </div>

            {/* Individual ID */}
            <div>
              <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                Идентификатор особи
                <Tip text="Уникальный ID конкретного животного из базы 13 837 особей. Метрическое обучение ArcFace по плавнику.">
                  <InfoIcon/>
                </Tip>
              </p>
              <code className="font-mono text-sm bg-slate-100 text-slate-700 px-2.5 py-1.5 rounded-lg block break-all">
                {result.class_animal || '—'}
              </code>
            </div>

            {/* Confidence bars */}
            <div className="space-y-2.5">
              <ConfBar value={result.probability}
                label="Уверенность идентификации"
                tooltip="Вероятность совпадения с конкретной особью. ≥70% — надёжно; 40–69% — предварительно; <40% — требует проверки."/>
              {typeof result.cetacean_score === 'number' && (
                <ConfBar value={result.cetacean_score}
                  label="Антифрод-скор (CLIP)"
                  tooltip="Вероятность того, что на снимке морское млекопитающее. Порог: 50%."/>
              )}
            </div>
          </div>
        </div>

        {/* Alternative candidates */}
        {candidates.length > 0 && (
          <div className="mt-5 pt-4 border-t border-slate-100">
            <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Альтернативные варианты
            </p>
            <div className="space-y-2">
              {candidates.map((c, i) => {
                const pct = Math.round(c.probability * 100);
                return (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px]
                                     font-bold text-slate-500 bg-slate-100 flex-shrink-0">
                      {i + 2}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-slate-700 font-medium truncate">{speciesName(c.id_animal)}</span>
                        <span className="text-xs text-slate-400 tabular-nums flex-shrink-0">{pct}%</span>
                      </div>
                      <code className="font-mono text-[10px] text-slate-400">{c.class_animal}</code>
                    </div>
                    <div className="w-20 h-1 rounded-full bg-slate-100 flex-shrink-0">
                      <div className="h-1 rounded-full bg-slate-300"
                        style={{width: `${Math.max(4, pct)}%`}}/>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Footer meta */}
      {result.model_version && (
        <div className="px-5 py-2 border-t border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-xs text-slate-400">
            Модель: <code className="font-mono">{result.model_version}</code>
          </span>
          <span className="text-xs text-slate-400">
            {result.bbox ? `bbox: ${result.bbox.join(', ')}` : ''}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Loading overlay on image ────────────────────────────────────────────────
function ImageLoadingOverlay() {
  return (
    <div className="absolute inset-0 rounded-xl bg-white/75 backdrop-blur-sm flex flex-col
                    items-center justify-center gap-2 z-10">
      <Spinner size={8}/>
      <p className="text-sm font-medium text-slate-600">Анализ снимка...</p>
      <p className="text-xs text-slate-400">Обычно занимает 5–15 секунд</p>
    </div>
  );
}

// ─── Main ────────────────────────────────────────────────────────────────────
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

  useEffect(() => {
    let alive = true;
    const check = async () => { const ok = await checkHealth(); if (alive) setBackendOk(ok); };
    check();
    const id = setInterval(check, 20_000);
    return () => { alive = false; clearInterval(id); };
  }, []);

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
    ? Object.entries(batchResults.reduce<Record<string,number>>((acc,d) => {
        const k = speciesName(d.id_animal);
        acc[k] = (acc[k]||0)+1; return acc;
      }, {})).map(([name,count]) => ({name,count}))
    : [];

  const CHART_COLORS = ['#0096c7','#06b6d4','#22d3ee','#0077b6','#67e8f9','#a5f3fc'];

  return (
    <div className="min-h-screen flex flex-col" style={{background:'#f0f4f8'}}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <header style={{background:'#0f2744'}} className="sticky top-0 z-30 shadow-lg">
        <div className="mx-auto max-w-5xl px-4 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{background:'#0077b6'}}>
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor"
                strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064"/>
              </svg>
            </div>
            <div>
              <p className="text-white font-semibold text-sm leading-none">EcoMarineAI</p>
              <p className="text-cyan-300 text-[10px] leading-none mt-0.5">
                Идентификация морских млекопитающих
              </p>
            </div>
          </div>

          <div className={[
            'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium',
            backendOk === null ? 'bg-slate-700 text-slate-300' :
            backendOk ? 'bg-emerald-900/60 text-emerald-300' :
            'bg-red-900/60 text-red-300',
          ].join(' ')}>
            <span className={[
              'w-2 h-2 rounded-full flex-shrink-0',
              backendOk === null ? 'bg-slate-400 animate-pulse' :
              backendOk ? 'bg-emerald-400' : 'bg-red-400 animate-pulse',
            ].join(' ')}/>
            {backendOk === null ? 'Проверка...' : backendOk ? 'Сервер готов' : 'Сервер недоступен'}
          </div>
        </div>

        <div className="mx-auto max-w-5xl px-4 flex gap-1">
          {(['single','batch'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={[
                'px-5 py-2.5 text-sm font-medium rounded-t-lg transition-colors',
                tab===t ? 'bg-[#f0f4f8] text-[#0077b6]' : 'text-cyan-300 hover:text-white hover:bg-white/10',
              ].join(' ')}>
              {t==='single' ? 'Одиночный анализ' : 'Пакетная обработка'}
            </button>
          ))}
        </div>
      </header>

      {/* ── Content ──────────────────────────────────────────────── */}
      <main className="mx-auto max-w-5xl w-full px-4 py-6 flex-1">

        {/* ── Single tab ───────────────────────────────────────── */}
        {tab==='single' && (
          <div className="space-y-5">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
              <div>
                <h2 className="font-semibold text-slate-800">Загрузка снимка</h2>
                <p className="text-sm text-slate-500 mt-0.5">
                  Фотография одного животного · JPG, PNG, WEBP
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

              {/* Preview with loading overlay */}
              {previewUrl && (
                <div className="relative rounded-xl overflow-hidden border border-slate-200">
                  <img src={previewUrl} alt="Preview"
                    className={`w-full object-contain max-h-64 bg-slate-50 transition-opacity ${busy ? 'opacity-40' : ''}`}/>
                  {busy && <ImageLoadingOverlay/>}
                </div>
              )}

              {/* Analyse button */}
              <button onClick={handleSingle} disabled={!singleFile || busy}
                className="w-full h-12 rounded-xl flex items-center justify-center gap-2
                           text-sm font-semibold text-white transition-all select-none
                           disabled:opacity-50 disabled:cursor-not-allowed"
                style={{background: (!singleFile || busy) ? '#94a3b8' : '#0077b6'}}>
                {busy ? (
                  <>
                    <Spinner size={5}/>
                    <span>Анализ снимка...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"/>
                    </svg>
                    <span>Определить вид и особь</span>
                  </>
                )}
              </button>
            </div>

            {result && <DetectionCard result={result} imageUrl={previewUrl}/>}

            {/* Empty state */}
            {!result && !busy && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
                  {[
                    {step:'1', title:'Загрузите снимок',
                     desc:'Фотография морского млекопитающего — кит, дельфин, морской котик и другие. Оптимально — чёткий плавник.'},
                    {step:'2', title:'Анализ',
                     desc:'EfficientNet-B4 с метрическим обучением определяет вид и конкретную особь из базы 13 837 животных.'},
                    {step:'3', title:'Результат',
                     desc:'Вид, уникальный ID особи, 5 альтернативных вариантов идентификации и уверенность модели.'},
                  ].map(({step,title,desc}) => (
                    <div key={step} className="flex gap-3">
                      <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center
                                      text-xs font-bold text-white" style={{background:'#0077b6'}}>
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
                    {v:'30',      u:'видов',    sub:'морских млекопитающих'},
                    {v:'13 837',  u:'особей',   sub:'в базе данных'},
                    {v:'93.6%',   u:'точность', sub:'Precision@1'},
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

        {/* ── Batch tab ─────────────────────────────────────────── */}
        {tab==='batch' && (
          <div className="space-y-5">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
              <div>
                <h2 className="font-semibold text-slate-800">Пакетная обработка</h2>
                <p className="text-sm text-slate-500 mt-0.5">
                  ZIP-архив с несколькими снимками — система обработает все
                </p>
              </div>

              <DropZone
                accept=".zip"
                hint=".zip архив со снимками"
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

              {/* Batch loading indicator */}
              {busy && (
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-blue-50 border border-blue-100">
                  <Spinner size={5}/>
                  <div>
                    <p className="text-sm font-medium text-blue-800">Обрабатываем архив...</p>
                    <p className="text-xs text-blue-600">Каждый снимок обрабатывается независимо, это может занять несколько минут</p>
                  </div>
                </div>
              )}

              <button onClick={handleBatch} disabled={!batchFile || busy}
                className="w-full h-12 rounded-xl flex items-center justify-center gap-2
                           text-sm font-semibold text-white transition-all select-none
                           disabled:opacity-50 disabled:cursor-not-allowed"
                style={{background: (!batchFile || busy) ? '#94a3b8' : '#0077b6'}}>
                {busy ? <><Spinner size={5}/><span>Обработка...</span></> : (
                  <><svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/>
                  </svg><span>Обработать архив</span></>
                )}
              </button>
            </div>

            {batchResults && (
              <div className="space-y-4 slide-up">
                <div className="grid grid-cols-3 gap-3">
                  {[
                    {v:batchResults.length, label:'Всего снимков'},
                    {v:batchResults.filter(d=>!d.rejected).length, label:'Опознано'},
                    {v:batchResults.filter(d=>d.rejected).length, label:'Не найдено'},
                  ].map(({v,label}) => (
                    <div key={label} className="bg-white rounded-xl border border-slate-200 p-4 text-center">
                      <p className="text-2xl font-bold" style={{color:'#0077b6'}}>{v}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{label}</p>
                    </div>
                  ))}
                </div>

                {chartData.length>0 && (
                  <div className="bg-white rounded-2xl border border-slate-200 p-5">
                    <p className="text-sm font-semibold text-slate-700 mb-4">Распределение по видам</p>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={chartData} margin={{top:0,right:10,left:-15,bottom:55}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                        <XAxis dataKey="name" angle={-40} textAnchor="end" tick={{fontSize:11,fill:'#64748b'}} interval={0}/>
                        <YAxis tick={{fontSize:11,fill:'#64748b'}} allowDecimals={false}/>
                        <ChartTooltip contentStyle={{borderRadius:8,border:'1px solid #e2e8f0',fontSize:12}}/>
                        <Bar dataKey="count" radius={[4,4,0,0]}>
                          {chartData.map((_,i) => <Cell key={i} fill={CHART_COLORS[i%CHART_COLORS.length]}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
                  <div className="px-5 py-3 border-b border-slate-100">
                    <p className="text-sm font-semibold text-slate-700">
                      Детали ({batchResults.length} записей)
                    </p>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
                          <th className="px-4 py-3 text-left font-medium">Файл</th>
                          <th className="px-4 py-3 text-left font-medium">Вид</th>
                          <th className="px-4 py-3 text-left font-medium font-mono">ID</th>
                          <th className="px-4 py-3 text-right font-medium">Уверенность</th>
                          <th className="px-4 py-3 text-center font-medium">Статус</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {batchResults.slice(0,20).map((d,i) => (
                          <tr key={i} className="hover:bg-slate-50 transition-colors">
                            <td className="px-4 py-3 text-slate-600 max-w-[120px] truncate text-xs">{d.image_ind}</td>
                            <td className="px-4 py-3 text-slate-800 font-medium text-xs">{speciesName(d.id_animal)}</td>
                            <td className="px-4 py-3">
                              <code className="font-mono text-[11px] text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                                {d.class_animal?.slice(0,10)||'—'}
                              </code>
                            </td>
                            <td className="px-4 py-3 text-right">
                              <span className={`text-xs font-semibold tabular-nums
                                ${d.probability>=0.7?'text-emerald-600':d.probability>=0.4?'text-amber-600':'text-red-500'}`}>
                                {(d.probability*100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="px-4 py-3 text-center">
                              {d.rejected
                                ? <span className="text-xs text-red-600 bg-red-50 px-2 py-0.5 rounded-full">✗ Не найдено</span>
                                : <span className="text-xs text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">✓ Опознано</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {batchResults.length>20 && (
                      <p className="text-xs text-slate-400 text-center py-2">
                        Показано 20 из {batchResults.length}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {!batchResults && !busy && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Формат архива
                </p>
                <div className="bg-slate-50 rounded-xl p-4 font-mono text-xs text-slate-600 space-y-0.5">
                  <p>📦 archive.zip</p>
                  <p className="pl-4">├── animal_001.jpg</p>
                  <p className="pl-4">├── animal_002.png</p>
                  <p className="pl-4">└── animal_003.jpg</p>
                </div>
                <p className="text-xs text-slate-500 mt-3">
                  Принимаются JPG/PNG/WEBP. Вложенные папки не поддерживаются.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer style={{background:'#0f2744'}} className="py-4 text-center mt-auto">
        <p className="text-[11px]" style={{color:'#4a7fa5'}}>
          EcoMarineAI · EfficientNet-B4 ArcFace · 13 837 особей · 30 видов ·
          При поддержке Фонда содействия инновациям · 2026
        </p>
      </footer>

      {/* ── Error modal ──────────────────────────────────────────── */}
      {error && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{background:'rgba(2,6,23,0.6)',backdropFilter:'blur(6px)'}}
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
              className="w-full h-10 rounded-xl text-sm font-semibold text-white"
              style={{background:'#0f2744'}}>
              Закрыть
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
