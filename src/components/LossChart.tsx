import React from 'react';

interface LossChartProps {
  lossHistory: number[];
  lossSteps: number[];
  currentLoss: number;
  epoch: number;
  totalSteps: number;
  learningRate: number;
}

export const LossChart: React.FC<LossChartProps> = ({
  lossHistory,
  lossSteps,
  currentLoss,
  epoch,
  totalSteps,
  learningRate,
}) => {
  // The measured series, always ending at the exact current loss shown in the header.
  const series = lossHistory.map((loss, i) => ({ step: lossSteps[i] ?? i, loss }));
  const last = series[series.length - 1];
  if (!last || last.step < totalSteps) {
    series.push({ step: totalSteps, loss: currentLoss });
  } else if (last.step === totalSteps) {
    last.loss = currentLoss;
  }

  const losses = series.map((p) => p.loss);
  const minVal = Math.min(...losses);
  const maxVal = Math.max(...losses);
  const firstStep = series[0].step;
  const lastStep = series[series.length - 1].step;
  const useLog = minVal > 0;

  const width = 320;
  const height = 110;
  const padLeft = 48;
  const padRight = 12;
  const padTop = 14;
  const padBottom = 22;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const toNorm = (v: number) => {
    if (maxVal === minVal) return 0.5;
    return useLog
      ? (Math.log(v) - Math.log(minVal)) / (Math.log(maxVal) - Math.log(minVal))
      : (v - minVal) / (maxVal - minVal);
  };
  const midVal = useLog ? Math.sqrt(minVal * maxVal) : (minVal + maxVal) / 2;

  const getX = (step: number) =>
    lastStep === firstStep ? padLeft + chartW / 2 : padLeft + ((step - firstStep) / (lastStep - firstStep)) * chartW;
  const getY = (val: number) => padTop + (1 - toNorm(val)) * chartH;

  const pathD = series
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(p.step).toFixed(1)} ${getY(p.loss).toFixed(1)}`)
    .join(' ');
  const baseY = (padTop + chartH).toFixed(1);
  const areaD = `${pathD} L ${getX(lastStep).toFixed(1)} ${baseY} L ${getX(firstStep).toFixed(1)} ${baseY} Z`;

  const lastX = getX(lastStep);
  const lastY = getY(series[series.length - 1].loss);

  const firstLoss = series[0].loss;
  const delta = series[series.length - 1].loss - firstLoss;
  const deltaPct = firstLoss !== 0 ? (delta / firstLoss) * 100 : 0;
  const signed = (v: number, digits: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`;

  return (
    <div className="rounded border border-[#DCD6CA] bg-white p-3 space-y-2">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-semibold text-[#1C1917]">L1 Loss Convergence</span>
          <span className="text-[10px] text-[#78716C]">·</span>
          <span className="font-mono-tabular text-[10px] text-[#78716C]">
            η={learningRate} · Adam (β1=0.9, β2=0.999)
          </span>
        </div>
        <span className="font-mono-tabular text-[11px] font-semibold text-[#1C1917]">
          {currentLoss.toFixed(5)}
        </span>
      </div>

      <div className="relative overflow-hidden rounded bg-[#FAF8F5] border border-[#EAE5DC]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-[110px] block"
          aria-label="Loss convergence curve"
        >
          <defs>
            <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1C1917" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#1C1917" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          <line x1={padLeft} y1={padTop} x2={padLeft + chartW} y2={padTop} stroke="#EAE5DC" strokeDasharray="2,2" strokeWidth="1" />
          <line x1={padLeft} y1={padTop + chartH * 0.5} x2={padLeft + chartW} y2={padTop + chartH * 0.5} stroke="#EAE5DC" strokeDasharray="2,2" strokeWidth="1" />
          <line x1={padLeft} y1={padTop + chartH} x2={padLeft + chartW} y2={padTop + chartH} stroke="#DCD6CA" strokeWidth="1" />

          {/* Y-axis: exact maximum, mid-point and minimum of the plotted series */}
          <text x={padLeft - 4} y={padTop + 4} textAnchor="end" fontSize="8" fill="#78716C" className="font-mono-tabular">
            {maxVal.toFixed(5)}
          </text>
          <text x={padLeft - 4} y={padTop + chartH * 0.5 + 3} textAnchor="end" fontSize="8" fill="#78716C" className="font-mono-tabular">
            {midVal.toFixed(5)}
          </text>
          <text x={padLeft - 4} y={padTop + chartH} textAnchor="end" fontSize="8" fill="#78716C" className="font-mono-tabular">
            {minVal.toFixed(5)}
          </text>

          <path d={areaD} fill="url(#lossGradient)" />
          <path d={pathD} fill="none" stroke="#1C1917" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />

          <circle cx={lastX} cy={lastY} r="3" fill="#1C1917" />
          <circle cx={lastX} cy={lastY} r="6" fill="#1C1917" fillOpacity="0.2" />

          <text x={padLeft} y={height - 5} fontSize="8" fill="#78716C" className="font-mono-tabular">
            step {firstStep.toLocaleString()}
          </text>
          <text x={padLeft + chartW / 2} y={height - 5} textAnchor="middle" fontSize="8" fill="#A8A29E" className="font-mono-tabular">
            {useLog ? 'log scale' : 'linear scale'}
          </text>
          <text x={padLeft + chartW} y={height - 5} textAnchor="end" fontSize="8" fill="#78716C" className="font-mono-tabular">
            Epoch {epoch} · step {lastStep.toLocaleString()}
          </text>
        </svg>
      </div>

      <div className="flex items-center justify-between text-[10px] text-[#655E57] font-mono-tabular">
        <span>Min: {minVal.toFixed(5)}</span>
        <span>Max: {maxVal.toFixed(5)}</span>
        <span>
          Δ: {signed(delta, 5)} ({signed(deltaPct, 1)}%)
        </span>
      </div>
    </div>
  );
};
