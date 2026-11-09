import React from 'react';
import { BRUSH_SIZES, BrushSize, PaintingPassName, getPassNameForSize } from '../engine/neuralPainter';

interface PassScheduleViewProps {
  activeSize: BrushSize;
  activePassName: PaintingPassName;
  activeHex: string;
  progressPct: number;
  completedStrokes: number;
  totalStrokes: number;
  passStrokeCounts: number[];
  onSelectSize: (size: BrushSize) => void;
}

const HIERARCHY_STAGES: { stage: PaintingPassName; shortLabel: string }[] = [
  { stage: 'Coarse Receptive Field', shortLabel: '125–69px' },
  { stage: 'Mid-Scale Tensor', shortLabel: '38–21px' },
  { stage: 'Fine Residual Field', shortLabel: '12–7px' },
  { stage: 'Micro-Pixel Convergence', shortLabel: '4–1px' },
];

export const PassScheduleView: React.FC<PassScheduleViewProps> = ({
  activeSize,
  activePassName,
  activeHex,
  progressPct,
  completedStrokes,
  totalStrokes,
  passStrokeCounts,
  onSelectSize,
}) => {
  return (
    <div className="flex-1 rounded border border-[#DCD6CA] bg-white p-3 flex flex-col justify-between gap-2.5">
      <div>
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-[#1C1917]">
            Multi-Scale Neural Hierarchy
          </span>
          <span className="font-mono-tabular text-[11px] text-[#78716C]">
            {progressPct}% complete
          </span>
        </div>

        {/* Active Stage & Stroke Progress */}
        <div className="mt-1.5 flex items-center justify-between rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2.5 py-1.5">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0 transition-colors duration-75"
              style={{ backgroundColor: activeHex }}
              aria-hidden="true"
            />
            <span
              className="text-xs font-semibold text-[#1C1917] truncate"
              data-testid="active-pass-name"
            >
              {activePassName}
            </span>
          </div>
          <span className="font-mono-tabular text-[10px] text-[#655E57] shrink-0">
            {completedStrokes.toLocaleString()} / {totalStrokes.toLocaleString()}
          </span>
        </div>

        {/* 4-Stage Hierarchy Bar */}
        <div className="mt-2 grid grid-cols-4 gap-1">
          {HIERARCHY_STAGES.map(({ stage, shortLabel }) => {
            const isCurrent = activePassName === stage;
            return (
              <div key={stage} className="space-y-0.5">
                <div
                  className={`h-1.5 rounded-full transition-colors ${
                    isCurrent ? 'bg-[#1C1917]' : 'bg-[#E5E0D5]'
                  }`}
                />
                <div
                  className={`text-[9px] font-mono-tabular truncate ${
                    isCurrent ? 'font-semibold text-[#1C1917]' : 'text-[#8C847A]'
                  }`}
                >
                  {shortLabel}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 10-Brush-Size Hierarchy Breakdown (125px down to 1px) */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between text-[10px] text-[#78716C] mb-1">
          <span>Receptive Field Scale (125px → 1px)</span>
          <span className="font-mono-tabular">Click scale to jump</span>
        </div>
        <div className="grid grid-cols-2 gap-1">
          {BRUSH_SIZES.map((sz, idx) => {
            const isCurrent = activeSize === sz;
            const count = passStrokeCounts[idx] ?? 0;
            const maxCount = Math.max(1, ...passStrokeCounts);
            const barPct = Math.max(6, Math.round((count / maxCount) * 100));
            const passStage = getPassNameForSize(sz);
            const shortName =
              passStage === 'Coarse Receptive Field'
                ? 'Coarse'
                : passStage === 'Mid-Scale Tensor'
                ? 'Mid'
                : passStage === 'Fine Residual Field'
                ? 'Fine'
                : 'Micro';

            return (
              <button
                key={sz}
                type="button"
                onClick={() => onSelectSize(sz)}
                className={`flex flex-col justify-between px-2 py-1 rounded border text-left font-mono-tabular transition-colors cursor-pointer ${
                  isCurrent
                    ? 'bg-[#1C1917] text-white border-[#1C1917]'
                    : 'bg-[#FAF8F5] text-[#44403C] border-[#EAE5DC] hover:bg-[#EFECE6]'
                }`}
              >
                <div className="flex items-center justify-between w-full text-[9px]">
                  <span className="font-semibold">
                    {sz}px · {shortName}
                  </span>
                  <span className={isCurrent ? 'text-[#E7E5E4]' : 'text-[#78716C]'}>
                    {count.toLocaleString()}
                  </span>
                </div>
                <div
                  className={`mt-1 h-0.5 w-full rounded-full overflow-hidden ${
                    isCurrent ? 'bg-[#44403C]' : 'bg-[#E5E0D5]'
                  }`}
                >
                  <div
                    className={`h-full rounded-full ${
                      isCurrent ? 'bg-white' : 'bg-[#78716C]'
                    }`}
                    style={{ width: `${barPct}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
