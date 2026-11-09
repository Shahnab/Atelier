import React from 'react';

interface PolicyActionHeadProps {
  cx: number;
  cy: number;
  dxOffsetPx: number;
  dyOffsetPx: number;
  angleDeg: number;
  lengthScale: number;
  lengthPx: number;
  pressureCurve: [number, number, number];
  curvature: [number, number];
  rgb: [number, number, number];
  hexColor: string;
  reasoning: string;
  brushSize: number;
}

export const PolicyActionHead: React.FC<PolicyActionHeadProps> = ({
  cx,
  cy,
  dxOffsetPx,
  dyOffsetPx,
  angleDeg,
  lengthScale,
  lengthPx,
  pressureCurve,
  curvature,
  rgb,
  hexColor,
  reasoning,
  brushSize,
}) => {
  const rad = (angleDeg * Math.PI) / 180;
  const arrowX = Math.cos(rad) * 11;
  const arrowY = Math.sin(rad) * 11;

  return (
    <div className="rounded border border-[#DCD6CA] bg-white p-3 space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-[#1C1917]">
          Policy Action Vector π<sub>θ</sub>(s<sub>t</sub>)
        </span>
        <span className="font-mono-tabular text-[10px] text-[#78716C]">
          13-D continuous neural output
        </span>
      </div>

      {/* Top Row: Spatial Position (x, y, Δx, Δy), Angle θ, and Neural RGB Pigment */}
      <div className="grid grid-cols-3 gap-1.5">
        {/* Neural Position (x, y) + Sub-Patch Offset */}
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 flex flex-col justify-between">
          <div className="text-[9px] text-[#78716C]">Position (x, y)</div>
          <div className="font-mono-tabular text-[10px] font-semibold text-[#1C1917]">
            ({cx}, {cy})
          </div>
          <div className="font-mono-tabular text-[8px] text-[#78716C]">
            Δ({dxOffsetPx >= 0 ? `+${dxOffsetPx}` : dxOffsetPx},{' '}
            {dyOffsetPx >= 0 ? `+${dyOffsetPx}` : dyOffsetPx}) px
          </div>
        </div>

        {/* Angle θ with Mini Compass */}
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 flex items-center gap-1.5">
          <svg width="26" height="26" viewBox="-14 -14 28 28" className="shrink-0">
            <circle cx="0" cy="0" r="12" fill="#EFECE6" stroke="#DCD6CA" strokeWidth="1" />
            <line
              x1="0"
              y1="0"
              x2={arrowX.toFixed(1)}
              y2={arrowY.toFixed(1)}
              stroke="#1C1917"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <circle cx="0" cy="0" r="2" fill="#1C1917" />
          </svg>
          <div className="min-w-0">
            <div className="text-[9px] text-[#78716C]">Angle θ</div>
            <div className="font-mono-tabular text-[11px] font-semibold text-[#1C1917]">
              {angleDeg}°
            </div>
          </div>
        </div>

        {/* Neural Predicted RGB */}
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-[#78716C]">Neural RGB</span>
            <span
              className="w-2.5 h-2.5 rounded-[2px] border border-[#ABA396]"
              style={{ backgroundColor: hexColor }}
            />
          </div>
          <div className="font-mono-tabular text-[10px] font-semibold text-[#1C1917]">
            {rgb[0]},{rgb[1]},{rgb[2]}
          </div>
          <div className="font-mono-tabular text-[8px] text-[#78716C]">clamped to [0, 255]</div>
        </div>
      </div>

      {/* Second Row: Length Scale & Bézier Curvature */}
      <div className="grid grid-cols-2 gap-1.5">
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5 flex items-center justify-between">
          <div>
            <div className="text-[9px] text-[#78716C]">Length Scale</div>
            <div className="font-mono-tabular text-[10px] font-semibold text-[#1C1917]">
              {lengthScale.toFixed(2)}× ({lengthPx.toFixed(1)}px)
            </div>
          </div>
          <div className="w-12 h-1.5 rounded-full bg-[#E5E0D5] overflow-hidden">
            <div
              className="h-full bg-[#1C1917]"
              style={{ width: `${Math.min(100, (lengthScale / 1.7) * 100)}%` }}
            />
          </div>
        </div>

        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5 flex items-center justify-between">
          <div>
            <div className="text-[9px] text-[#78716C]">Curvature (c1, c2)</div>
            <div className="font-mono-tabular text-[10px] font-semibold text-[#1C1917]">
              {curvature[0].toFixed(2)}, {curvature[1].toFixed(2)}
            </div>
          </div>
          <span className="text-[8px] font-mono-tabular text-[#78716C]">Bézier</span>
        </div>
      </div>

      {/* Pressure Profile (P0, Pmid, P1) */}
      <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 space-y-1">
        <div className="flex items-center justify-between text-[9px] text-[#78716C]">
          <span>Stylus Pressure Neurons (Attack P0 · Peak PMid · Release P1)</span>
          <span className="font-mono-tabular text-[#1C1917]">
            Peak {Math.round(pressureCurve[1] * 100)}%
          </span>
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          {[
            { label: 'P0 Attack', val: pressureCurve[0] },
            { label: 'PMid Peak', val: pressureCurve[1] },
            { label: 'P1 Release', val: pressureCurve[2] },
          ].map((pt) => (
            <div key={pt.label} className="space-y-0.5">
              <div className="flex items-center justify-between text-[8px] font-mono-tabular text-[#57534E]">
                <span>{pt.label}</span>
                <span className="font-semibold text-[#1C1917]">{pt.val.toFixed(2)}</span>
              </div>
              <div className="h-1 w-full rounded-full bg-[#E5E0D5] overflow-hidden">
                <div
                  className="h-full bg-[#1C1917] transition-all duration-100"
                  style={{ width: `${Math.min(100, Math.round(pt.val * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reasoning and active color */}
      <div className="flex items-center gap-2 rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2.5 py-1.5">
        <span
          className="w-3.5 h-3.5 rounded-[2px] border border-[#C8C2B8] shrink-0"
          style={{ backgroundColor: hexColor }}
          aria-hidden="true"
        />
        <span
          className="font-mono-tabular text-[10px] text-[#1C1917] truncate flex-1"
          data-testid="ai-reasoning-readout"
        >
          {reasoning}
        </span>
      </div>
    </div>
  );
};
