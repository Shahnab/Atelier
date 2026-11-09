import React from 'react';
import { NeuralTelemetry, rgbToHex } from '../engine/neuralPainter';

const formatNorm = (v: number) => (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(3) : v.toFixed(5));

interface TrainingConsoleProps {
  telemetry: NeuralTelemetry;
  targetRgb: [number, number, number];
  predictedRgb: [number, number, number];
  onStepBurst: (steps: number) => void;
  onResetWeights: () => void;
}

export const TrainingConsole: React.FC<TrainingConsoleProps> = ({
  telemetry,
  targetRgb,
  predictedRgb,
  onStepBurst,
  onResetWeights,
}) => {
  const targetHex = rgbToHex(targetRgb[0], targetRgb[1], targetRgb[2]);
  const predHex = rgbToHex(predictedRgb[0], predictedRgb[1], predictedRgb[2]);

  return (
    <div className="rounded border border-[#DCD6CA] bg-white p-3 space-y-2.5">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-semibold text-[#1C1917]">Online Adam &amp; Chromatic Net</span>
          <span className="text-[10px] text-[#78716C]">·</span>
          <span className="font-mono-tabular text-[10px] text-[#78716C]">
            t = {telemetry.totalSteps.toLocaleString()}
          </span>
        </div>
        <button
          type="button"
          onClick={onResetWeights}
          data-testid="sidebar-reset-weights-button"
          className="text-[10px] font-mono-tabular text-[#57534E] hover:text-[#1C1917] px-2 py-0.5 rounded border border-[#D4CFC4] bg-[#FAF8F5] hover:bg-[#EFECE6] transition-colors cursor-pointer"
        >
          Re-Init ~ N(0, σ²)
        </button>
      </div>

      {/* Live Neural Color Learning Readout (Target Patch RGB vs Neural Predicted RGB) */}
      <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 space-y-1.5 font-mono-tabular">
        <div className="flex items-center justify-between text-[9px] text-[#78716C]">
          <span>Chromatic Sub-Net (4→8→1 per RGB channel)</span>
          <span className="font-semibold text-[#1C1917]">
            L<sub>color</sub> = {telemetry.colorLoss.toFixed(5)}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1.5 text-[9px]">
          <div className="flex items-center gap-1.5 rounded bg-white border border-[#EAE5DC] px-2 py-1">
            <span
              className="w-3 h-3 rounded-[2px] border border-[#ABA396] shrink-0"
              style={{ backgroundColor: targetHex }}
            />
            <div className="min-w-0">
              <div className="text-[8px] text-[#78716C]">Target RGB (receptive-field mean)</div>
              <div className="font-semibold text-[#1C1917] truncate">
                {targetRgb[0]}, {targetRgb[1]}, {targetRgb[2]}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5 rounded bg-white border border-[#EAE5DC] px-2 py-1">
            <span
              className="w-3 h-3 rounded-[2px] border border-[#ABA396] shrink-0"
              style={{ backgroundColor: predHex }}
            />
            <div className="min-w-0">
              <div className="text-[8px] text-[#78716C]">Painted RGB (network output)</div>
              <div className="font-semibold text-[#1C1917] truncate">
                {predictedRgb[0]}, {predictedRgb[1]}, {predictedRgb[2]}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Adam Moment Norms & Optimizer State */}
      <div className="grid grid-cols-3 gap-1.5 font-mono-tabular text-[9px]">
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5">
          <div className="text-[#78716C]">1st Moment ‖m‖₂</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">
            {formatNorm(telemetry.adamFirstMomentNorm)}
          </div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5">
          <div className="text-[#78716C]">2nd Moment ‖v‖₂</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">
            {formatNorm(telemetry.adamSecondMomentNorm)}
          </div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5">
          <div className="text-[#78716C]">Weight Init</div>
          <div className="font-semibold text-emerald-800 mt-0.5">Box-Muller N(0,σ²)</div>
        </div>
      </div>

      {/* Online Dual-Tensor Burst Training Controls */}
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onStepBurst(250)}
          data-testid="train-weights-run-button"
          className="px-2.5 py-1.5 rounded border border-[#1C1917] bg-white text-xs font-semibold text-[#1C1917] hover:bg-[#F5F2EB] transition-colors cursor-pointer"
        >
          Step +250 Backprops
        </button>
        <button
          type="button"
          onClick={() => onStepBurst(1000)}
          className="px-2.5 py-1.5 rounded bg-[#1C1917] text-xs font-semibold text-white hover:bg-[#332E2A] transition-colors cursor-pointer"
        >
          Step +1,000 Backprops
        </button>
      </div>
    </div>
  );
};
