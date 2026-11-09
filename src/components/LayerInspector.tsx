import React, { useState } from 'react';
import { LayerSummary } from '../engine/neuralPainter';

interface LayerInspectorProps {
  layers: LayerSummary[];
  gradientNorm: number;
  totalParameters: number;
  weightSnapshot?: number[];
}

export const LayerInspector: React.FC<LayerInspectorProps> = ({
  layers,
  gradientNorm,
  totalParameters,
  weightSnapshot = [],
}) => {
  const [viewMode, setViewMode] = useState<'bars' | 'matrix'>('bars');

  return (
    <div className="rounded border border-[#DCD6CA] bg-white p-3 space-y-2.5">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-semibold text-[#1C1917]">12-Layer Synaptic Highway</span>
          <span className="text-[10px] text-[#78716C]">·</span>
          <span className="font-mono-tabular text-[10px] text-[#78716C]">
            {totalParameters.toLocaleString()} weights
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono-tabular text-[10px] text-[#655E57]">
            ‖∇W‖ = {gradientNorm.toFixed(5)}
          </span>
          <div className="flex items-center rounded border border-[#EAE5DC] bg-[#FAF8F5] p-0.5 text-[9px] font-mono-tabular">
            <button
              type="button"
              onClick={() => setViewMode('bars')}
              className={`px-1.5 py-0.5 rounded cursor-pointer ${
                viewMode === 'bars' ? 'bg-white font-semibold text-[#1C1917] shadow-2xs' : 'text-[#78716C]'
              }`}
            >
              Bars
            </button>
            <button
              type="button"
              onClick={() => setViewMode('matrix')}
              className={`px-1.5 py-0.5 rounded cursor-pointer ${
                viewMode === 'matrix' ? 'bg-white font-semibold text-[#1C1917] shadow-2xs' : 'text-[#78716C]'
              }`}
            >
              Weights W<sub>ij</sub>
            </button>
          </div>
        </div>
      </div>

      {viewMode === 'bars' ? (
        /* Per-layer gradient norm, as a percentage of the largest layer's ‖∇W‖ */
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[9px] text-[#78716C] font-mono-tabular">
            <span>Bar height = layer ‖∇W‖ ÷ max layer ‖∇W‖</span>
            <span>hover for exact values</span>
          </div>
        <div
          className="grid grid-cols-12 gap-1 rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2"
          data-testid="mlp-12-layer-strip"
        >
          {layers.map((layer) => (
            <div
              key={layer.name}
              title={`${layer.name} (${layer.inDim}→${layer.outDim})${
                layer.residual ? ' [Residual Skip]' : ''
              }\n|W̄| = ${layer.meanAbsWeight.toFixed(5)}\n‖∇W‖ = ${layer.gradNorm.toFixed(5)} (${layer.relativeGradPct.toFixed(1)}% of max layer)`}
              className="flex flex-col items-center group cursor-pointer"
            >
              <div className="h-12 w-full flex items-end justify-center bg-[#EFECE6] rounded-[2px] overflow-hidden relative">
                {layer.residual && (
                  <div
                    className="absolute top-0.5 inset-x-0 mx-auto w-1 h-1 rounded-full bg-[#1C1917] opacity-60"
                    title="Residual Highway Connection"
                  />
                )}
                <div
                  className="w-full bg-[#1C1917] transition-all duration-150 group-hover:bg-[#44403C]"
                  style={{ height: `${layer.relativeGradPct}%` }}
                />
              </div>
              <span className="mt-1 font-mono-tabular text-[8px] font-semibold text-[#44403C]">
                {layer.name}
              </span>
            </div>
          ))}
        </div>
        </div>
      ) : (
        /* Live Synaptic Weight Matrix Heatmap (Micrograd/Karpathy Style) */
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-2 space-y-1.5">
          <div className="flex items-center justify-between text-[9px] text-[#78716C] font-mono-tabular">
            <span>First weight and middle weight of each layer (L1–L12)</span>
            <span>live values</span>
          </div>
          <div className="grid grid-cols-6 gap-1 font-mono-tabular text-[8px]">
            {weightSnapshot.slice(0, 24).map((w, idx) => {
              const absVal = Math.min(1, Math.abs(w));
              const isPositive = w >= 0;
              return (
                <div
                  key={idx}
                  title={`L${Math.floor(idx / 2) + 1} ${idx % 2 === 0 ? 'first' : 'middle'} weight: ${w.toFixed(6)}`}
                  className={`p-1 rounded text-center truncate border ${
                    isPositive
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                      : 'bg-rose-50 border-rose-200 text-rose-900'
                  }`}
                  style={{ opacity: 0.4 + absVal * 0.6 }}
                >
                  {w.toFixed(3)}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Layer Architecture Specs */}
      <div className="grid grid-cols-4 gap-1.5 text-[9px] font-mono-tabular">
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-1.5">
          <div className="text-[#78716C]">L1 Input</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">10 → 32</div>
          <div className="text-[8px] text-[#78716C]">Dual-Tensor</div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-1.5">
          <div className="text-[#78716C]">L2-L9 Res</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">32 → 32</div>
          <div className="text-[8px] text-[#78716C]">Skip-Conn</div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-1.5">
          <div className="text-[#78716C]">L10-L11 Bottleneck</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">32 → 24 → 16</div>
          <div className="text-[8px] text-[#78716C]">Leaky ReLU</div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] p-1.5">
          <div className="text-[#78716C]">L12 Policy Head</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">16 → 8</div>
          <div className="text-[8px] text-[#78716C]">Tanh Action</div>
        </div>
      </div>
    </div>
  );
};
