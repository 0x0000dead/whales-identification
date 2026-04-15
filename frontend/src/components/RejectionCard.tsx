import { Detection } from '../api';

interface Props {
  result: Detection;
}

export function RejectionCard({ result }: Props) {
  const score = result.cetacean_score ?? 0;
  const scorePct = Math.round(score * 100);

  if (result.rejection_reason === 'not_a_marine_mammal') {
    return (
      <div
        role="alert"
        className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow"
      >
        <div className="flex items-start gap-3">
          <svg
            className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <div className="flex-1">
            <h3 className="font-semibold text-red-900">
              Это не похоже на морское млекопитающее
            </h3>
            <p className="text-sm text-red-800 mt-1">
              Антифрод-фильтр отклонил изображение: на нём не обнаружены киты или дельфины.
              Загрузите фотографию морского млекопитающего, например аэроснимок.
            </p>
            <div className="mt-3">
              <div className="flex justify-between text-xs text-red-800 mb-1">
                <span>Уверенность «это кит/дельфин»</span>
                <span>{scorePct}%</span>
              </div>
              <div className="w-full bg-red-200 rounded-full h-2">
                <div
                  className="bg-red-500 h-2 rounded-full transition-all"
                  style={{ width: `${scorePct}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (result.rejection_reason === 'low_confidence') {
    return (
      <div
        role="alert"
        className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded-lg shadow"
      >
        <h3 className="font-semibold text-yellow-900">Слабая уверенность модели</h3>
        <p className="text-sm text-yellow-800 mt-1">
          Изображение распознано как морское млекопитающее, но модель не смогла надёжно
          определить особь. Попробуйте загрузить более чёткий или более крупный кадр.
        </p>
        <p className="text-xs text-yellow-700 mt-2">
          Кетацеан-скор: {scorePct}% · Уверенность ID: {(result.probability * 100).toFixed(1)}%
        </p>
      </div>
    );
  }

  return null;
}
