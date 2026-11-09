import React, { useEffect, useRef, useState } from 'react';

interface DualTensorSensoryViewProps {
  referenceUrl: string;
  studyTitle: string;
  canvasDimensions: { w: number; h: number };
  analysisDimensions: { w: number; h: number };
  residualL1Error: number;
  targetRgba?: Uint8ClampedArray | null;
  workingRgba?: Uint8ClampedArray | null;
  stepCount?: number;
  onUploadClick: () => void;
  onFileDrop: (file: File) => void;
}

export const DualTensorSensoryView: React.FC<DualTensorSensoryViewProps> = ({
  referenceUrl,
  studyTitle,
  canvasDimensions,
  analysisDimensions,
  residualL1Error,
  targetRgba,
  workingRgba,
  stepCount = 0,
  onUploadClick,
  onFileDrop,
}) => {
  const [activeSensoryMode, setActiveSensoryMode] = useState<'target' | 'canvas' | 'residual'>(
    'target'
  );
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [imgHasError, setImgHasError] = useState(false);
  const tensorCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    setImgHasError(false);
  }, [referenceUrl]);

  useEffect(() => {
    if (activeSensoryMode === 'target' && !imgHasError) return;
    const canvas = tensorCanvasRef.current;
    if (!canvas || !targetRgba) return;

    const fw = analysisDimensions.w;
    const fh = analysisDimensions.h;
    if (targetRgba.length !== fw * fh * 4) return;
    if (activeSensoryMode !== 'target' && (!workingRgba || workingRgba.length !== fw * fh * 4)) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(fw, fh);
    const out = imgData.data;

    if (activeSensoryMode === 'target') {
      for (let i = 0; i < out.length; i += 4) {
        out[i] = targetRgba[i];
        out[i + 1] = targetRgba[i + 1];
        out[i + 2] = targetRgba[i + 2];
        out[i + 3] = 255;
      }
    } else if (activeSensoryMode === 'canvas') {
      for (let i = 0; i < out.length; i += 4) {
        out[i] = workingRgba![i];
        out[i + 1] = workingRgba![i + 1];
        out[i + 2] = workingRgba![i + 2];
        out[i + 3] = 255;
      }
    } else {
      // True Dual-Tensor Residual ΔI = |I_target(x,y) - I_canvas(x,y)|
      for (let i = 0; i < out.length; i += 4) {
        const dr = Math.abs(targetRgba[i] - workingRgba![i]);
        const dg = Math.abs(targetRgba[i + 1] - workingRgba![i + 1]);
        const db = Math.abs(targetRgba[i + 2] - workingRgba![i + 2]);
        // Amplify residual slightly (2.2x) so fine perceptual errors remain clearly visible
        out[i] = Math.min(255, Math.round(dr * 2.2));
        out[i + 1] = Math.min(255, Math.round(dg * 2.2));
        out[i + 2] = Math.min(255, Math.round(db * 2.2));
        out[i + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [activeSensoryMode, imgHasError, targetRgba, workingRgba, analysisDimensions.w, analysisDimensions.h, stepCount, residualL1Error]);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDraggingOver(true);
      }}
      onDragLeave={() => setIsDraggingOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDraggingOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFileDrop(file);
      }}
      className={`rounded border bg-white p-3 space-y-2.5 transition-colors ${
        isDraggingOver ? 'border-[#1C1917] bg-[#FAF8F5]' : 'border-[#DCD6CA]'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-xs font-semibold text-[#1C1917] shrink-0">
            Dual-Tensor Sensory Field
          </span>
          <span className="text-[10px] text-[#78716C]">·</span>
          <span className="text-[11px] text-[#655E57] truncate">{studyTitle}</span>
        </div>
        <button
          type="button"
          onClick={onUploadClick}
          className="px-2 py-0.5 text-[10px] font-medium text-[#1C1917] bg-[#F5F2EB] hover:bg-[#EAE5DC] rounded border border-[#D4CFC4] transition-colors whitespace-nowrap cursor-pointer"
        >
          Upload
        </button>
      </div>

      {/* Sensory Mode Tabs */}
      <div className="flex items-center gap-1 p-0.5 bg-[#FAF8F5] rounded border border-[#EAE5DC] text-[10px]">
        <button
          type="button"
          onClick={() => setActiveSensoryMode('target')}
          className={`flex-1 py-1 rounded text-center font-medium transition-colors cursor-pointer ${
            activeSensoryMode === 'target'
              ? 'bg-white text-[#1C1917] shadow-2xs font-semibold'
              : 'text-[#655E57] hover:text-[#1C1917]'
          }`}
        >
          Target I<sub>target</sub>
        </button>
        <button
          type="button"
          onClick={() => setActiveSensoryMode('canvas')}
          className={`flex-1 py-1 rounded text-center font-medium transition-colors cursor-pointer ${
            activeSensoryMode === 'canvas'
              ? 'bg-white text-[#1C1917] shadow-2xs font-semibold'
              : 'text-[#655E57] hover:text-[#1C1917]'
          }`}
        >
          Canvas I<sub>canvas</sub>
        </button>
        <button
          type="button"
          onClick={() => setActiveSensoryMode('residual')}
          className={`flex-1 py-1 rounded text-center font-medium transition-colors cursor-pointer ${
            activeSensoryMode === 'residual'
              ? 'bg-white text-[#1C1917] shadow-2xs font-semibold'
              : 'text-[#655E57] hover:text-[#1C1917]'
          }`}
        >
          Residual ΔI
        </button>
      </div>

      {/* Visual Tensor Viewport */}
      <div className="relative overflow-hidden rounded bg-[#FAF8F5] border border-[#EAE5DC] flex items-center justify-center min-h-[170px] max-h-[190px]">
        {activeSensoryMode === 'target' && !imgHasError && (
          <img
            src={referenceUrl}
            alt={`Ground-truth reference for ${studyTitle}`}
            referrerPolicy="no-referrer"
            onError={() => setImgHasError(true)}
            className="w-full max-h-[190px] object-contain block"
          />
        )}
        {(activeSensoryMode !== 'target' || imgHasError) && (
          <canvas
            ref={tensorCanvasRef}
            width={analysisDimensions.w}
            height={analysisDimensions.h}
            className="w-full max-h-[190px] object-contain block bg-[#1C1917]"
          />
        )}
        {activeSensoryMode === 'residual' && (
          <div className="absolute inset-x-0 bottom-0 pointer-events-none flex items-center justify-between p-1.5">
            <span className="font-mono-tabular text-[9px] text-white bg-black/75 px-1.5 py-0.5 rounded">
              ΔI = |I_target − I_canvas| · L1 {residualL1Error.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Sensory Field Dimensions & Metric Badges */}
      <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono-tabular">
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5">
          <div className="text-[#78716C]">Ground Truth I<sub>target</sub></div>
          <div className="font-semibold text-[#1C1917] mt-0.5">
            {canvasDimensions.w} × {canvasDimensions.h} px
          </div>
        </div>
        <div className="rounded bg-[#FAF8F5] border border-[#EAE5DC] px-2 py-1.5">
          <div className="text-[#78716C]">Sensory Grid</div>
          <div className="font-semibold text-[#1C1917] mt-0.5">
            {analysisDimensions.w} × {analysisDimensions.h} cells
          </div>
        </div>
      </div>
    </div>
  );
};
