interface Props {
  label: string;
  value: number;
  variant?: 'success' | 'warning' | 'danger';
}

export function ConfidenceGauge({ label, value, variant = 'success' }: Props) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const colour =
    variant === 'success'
      ? 'bg-emerald-500'
      : variant === 'warning'
        ? 'bg-yellow-500'
        : 'bg-red-500';
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-700 mb-1">
        <span>{label}</span>
        <span className="font-medium">{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`${colour} h-2 rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
