/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  BRUSH_SIZES,
  BrushSize,
  computeCanvasL1Loss,
  executeOnlineStrokeStep,
  formatSignedValue,
  getPassNameForSize,
  NeuralStrokePolicy,
  NeuralTelemetry,
  PaintingPassName,
} from './engine/neuralPainter';
import { renderFullStroke } from './engine/brushRenderer';
import { LossChart } from './components/LossChart';
import { LayerInspector } from './components/LayerInspector';
import { PolicyActionHead } from './components/PolicyActionHead';
import { DualTensorSensoryView } from './components/DualTensorSensoryView';
import { PassScheduleView } from './components/PassScheduleView';
import { TrainingConsole } from './components/TrainingConsole';

const CANVAS_WIDTH = 825;
const CANVAS_HEIGHT = 875;
const ANALYSIS_WIDTH = 412;
const ANALYSIS_HEIGHT = 438;

const TOTAL_PLANNED_STROKES = 50000;

const PASS_START_THRESHOLDS: Record<BrushSize, number> = {
  125: 0,
  69: 400,
  38: 1200,
  21: 2800,
  12: 6000,
  7: 11500,
  4: 19500,
  3: 29000,
  2: 38000,
  1: 45000,
};

interface CursorReticule {
  x: number;
  y: number;
  radius: number;
  visible: boolean;
}

const DEFAULT_STEVE_JOBS_URL = `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/assets/steve-jobs.jpg`;

export default function App() {
  const autoStartRequested =
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('autostart') === '1';

  // Sensory Ground-Truth State (Default: Steve Jobs portrait)
  const [studyTitle, setStudyTitle] = useState<string>('Steve Jobs');
  const [referenceUrl, setReferenceUrl] = useState<string>(DEFAULT_STEVE_JOBS_URL);
  const [canvasDimensions, setCanvasDimensions] = useState<{ w: number; h: number }>({
    w: CANVAS_WIDTH,
    h: CANVAS_HEIGHT,
  });

  // Online Neural Learning State (Built and trained from scratch)
  const [activeSize, setActiveSize] = useState<BrushSize>(125);
  const [activeHex, setActiveHex] = useState<string>('#42372d');
  const [activePassName, setActivePassName] = useState<PaintingPassName>('Coarse Receptive Field');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [hasStarted, setHasStarted] = useState<boolean>(autoStartRequested);
  const [isPlaying, setIsPlaying] = useState<boolean>(autoStartRequested);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);
  const [showFlowOverlay, setShowFlowOverlay] = useState<boolean>(true);
  const [liveRewardDelta, setLiveRewardDelta] = useState<number>(0);

  // Live 13-D Neural Action Head Output Vector State
  const [latestReasoning, setLatestReasoning] = useState<string>(
    'Weights Initialized from Scratch ~ N(0, σ²) · Epoch 0 · Click "Start Training"'
  );
  const [currentCoords, setCurrentCoords] = useState<[number, number]>([412, 438]);
  const [currentOffsetPx, setCurrentOffsetPx] = useState<[number, number]>([0, 0]);
  const [currentAngleDeg, setCurrentAngleDeg] = useState<number>(45);
  const [currentLengthScale, setCurrentLengthScale] = useState<number>(1.0);
  const [currentLengthPx, setCurrentLengthPx] = useState<number>(0);
  const [currentPressureCurve, setCurrentPressureCurve] = useState<[number, number, number]>([
    0.52, 0.88, 0.44,
  ]);
  const [currentCurvature, setCurrentCurvature] = useState<[number, number]>([0.0, 0.0]);
  const [currentRgb, setCurrentRgb] = useState<[number, number, number]>([66, 55, 45]);
  const [currentTargetRgb, setCurrentTargetRgb] = useState<[number, number, number]>([66, 55, 45]);

  const [strokeStats, setStrokeStats] = useState<{ completed: number; total: number }>({
    completed: 0,
    total: TOTAL_PLANNED_STROKES,
  });
  const [passStrokeCounts, setPassStrokeCounts] = useState<number[]>(() =>
    BRUSH_SIZES.map(() => 0)
  );
  const passCountsRef = useRef<number[]>(BRUSH_SIZES.map(() => 0));
  const lastBrushSizeRef = useRef<BrushSize>(125);

  // Pure TypeScript 12-Layer Residual Neural Policy initialized from scratch at Epoch 0
  const [initialPolicy] = useState<NeuralStrokePolicy>(() => new NeuralStrokePolicy());
  const policyRef = useRef<NeuralStrokePolicy>(initialPolicy);
  const [telemetry, setTelemetry] = useState<NeuralTelemetry>(() =>
    policyRef.current.getTelemetry()
  );

  // Canvas & Buffer Refs
  const paintCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const targetRgbaRef = useRef<Uint8ClampedArray | null>(null);
  const fullResTargetRgbaRef = useRef<Uint8ClampedArray | null>(null);
  const workingRgbaRef = useRef<Uint8ClampedArray | null>(null);
  const fullResWorkingRgbaRef = useRef<Uint8ClampedArray | null>(null);

  const currentStrokeIdRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(autoStartRequested);
  const speedMultiplierRef = useRef<number>(1);
  const showFlowOverlayRef = useRef<boolean>(true);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    speedMultiplierRef.current = speedMultiplier;
  }, [speedMultiplier]);

  /**
   * Draws the live computer-use brush reticule (`⊙`) and 12-layer neural policy vector field
   */
  const renderOverlay = useCallback((reticule: CursorReticule) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Live 12-Layer Neural Policy Vector Field Overlay (evaluated directly from policy.forward)
    if (
      showFlowOverlayRef.current &&
      targetRgbaRef.current &&
      workingRgbaRef.current
    ) {
      const tBuf = targetRgbaRef.current;
      const wBuf = workingRgbaRef.current;
      const fw = ANALYSIS_WIDTH;
      const fh = ANALYSIS_HEIGHT;
      const sx = overlay.width / fw;
      const sy = overlay.height / fh;
      ctx.save();
      ctx.strokeStyle = 'rgba(28, 25, 23, 0.28)';
      ctx.lineWidth = 1;
      const stride = 18;
      for (let fy = 9; fy < fh - 2; fy += stride) {
        for (let fx = 9; fx < fw - 2; fx += stride) {
          const iC = (fy * fw + fx) * 4;
          const iL = (fy * fw + (fx - 2)) * 4;
          const iR = (fy * fw + (fx + 2)) * 4;
          const iU = ((fy - 2) * fw + fx) * 4;
          const iD = ((fy + 2) * fw + fx) * 4;

          const tLumL = (0.299 * tBuf[iL] + 0.587 * tBuf[iL + 1] + 0.114 * tBuf[iL + 2]) / 255;
          const tLumR = (0.299 * tBuf[iR] + 0.587 * tBuf[iR + 1] + 0.114 * tBuf[iR + 2]) / 255;
          const tLumU = (0.299 * tBuf[iU] + 0.587 * tBuf[iU + 1] + 0.114 * tBuf[iU + 2]) / 255;
          const tLumD = (0.299 * tBuf[iD] + 0.587 * tBuf[iD + 1] + 0.114 * tBuf[iD + 2]) / 255;

          const dX = tLumR - tLumL;
          const dY = tLumD - tLumU;
          const contrast = Math.hypot(dX, dY);
          if (contrast < 0.035) continue;

          const diffR = (tBuf[iC] - wBuf[iC]) / 255;
          const diffG = (tBuf[iC + 1] - wBuf[iC + 1]) / 255;
          const diffB = (tBuf[iC + 2] - wBuf[iC + 2]) / 255;
          const locErr = Math.sqrt(0.3 * diffR * diffR + 0.59 * diffG * diffG + 0.11 * diffB * diffB);

          const nnVec = policyRef.current.forward([
            Math.max(-1, Math.min(1, dX * 3.0)),
            Math.max(-1, Math.min(1, dY * 3.0)),
            Math.max(-1, Math.min(1, dX * 2.5)),
            Math.max(-1, Math.min(1, dY * 2.5)),
            (fx / fw) * 2 - 1,
            diffR * 2,
            diffG * 2,
            diffB * 2,
            locErr * 2 - 1,
            0.5,
          ]);
          const angle = Math.atan2(nnVec[1], nnVec[0]);
          const len = 4 + Math.min(1, contrast * 3) * 8;
          const cx = fx * sx;
          const cy = fy * sy;
          const dx = Math.cos(angle) * len;
          const dy = Math.sin(angle) * len;
          ctx.beginPath();
          ctx.moveTo(cx - dx, cy - dy);
          ctx.lineTo(cx + dx, cy + dy);
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // Agent Action Trajectory Reticule (`⊙`)
    if (reticule.visible) {
      ctx.save();
      const r = Math.max(4, reticule.radius);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(reticule.x, reticule.y, r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(28, 25, 23, 0.8)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(reticule.x, reticule.y, r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.beginPath();
      ctx.arc(reticule.x, reticule.y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(28, 25, 23, 0.9)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    }
  }, []);

  useEffect(() => {
    showFlowOverlayRef.current = showFlowOverlay;
    if (!isPlayingRef.current) {
      renderOverlay({ x: 0, y: 0, radius: 0, visible: false });
    }
  }, [showFlowOverlay, renderOverlay]);

  /**
   * Clears the canvas buffer and re-initializes neural state from scratch
   */
  const initBlankCanvasBuffers = useCallback((cWidth: number, cHeight: number) => {
    const working = new Uint8ClampedArray(ANALYSIS_WIDTH * ANALYSIS_HEIGHT * 4);
    for (let i = 0; i < working.length; i += 4) {
      working[i] = 255;
      working[i + 1] = 255;
      working[i + 2] = 255;
      working[i + 3] = 254; // 254 marks unpainted
    }
    workingRgbaRef.current = working;

    const fullWorking = new Uint8ClampedArray(cWidth * cHeight * 4);
    for (let i = 0; i < fullWorking.length; i += 4) {
      fullWorking[i] = 255;
      fullWorking[i + 1] = 255;
      fullWorking[i + 2] = 255;
      fullWorking[i + 3] = 254;
    }
    fullResWorkingRgbaRef.current = fullWorking;

    const canvas = paintCanvasRef.current;
    if (canvas) {
      if (canvas.width !== cWidth) canvas.width = cWidth;
      if (canvas.height !== cHeight) canvas.height = cHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, cWidth, cHeight);
      }
    }
    const overlay = overlayCanvasRef.current;
    if (overlay) {
      if (overlay.width !== cWidth) overlay.width = cWidth;
      if (overlay.height !== cHeight) overlay.height = cHeight;
    }
    renderOverlay({ x: 0, y: 0, radius: 0, visible: false });
    currentStrokeIdRef.current = 0;
    passCountsRef.current = BRUSH_SIZES.map(() => 0);
    lastBrushSizeRef.current = 125;
    setStrokeStats({ completed: 0, total: TOTAL_PLANNED_STROKES });
    setProgressPct(0);
    setPassStrokeCounts(BRUSH_SIZES.map(() => 0));
    setActiveSize(125);
    setActivePassName('Coarse Receptive Field');
  }, [renderOverlay]);

  /**
   * Initializes sensory observations from an image source
   */
  const initializeFromImageSource = useCallback(
    (srcUrl: string, titleLabel?: string, shouldAutoStart = false) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
        let cWidth = CANVAS_WIDTH;
        let cHeight = CANVAS_HEIGHT;
        if (Math.abs(aspect - CANVAS_WIDTH / CANVAS_HEIGHT) > 0.08) {
          if (aspect >= 1) {
            cWidth = 860;
            cHeight = Math.max(520, Math.round(860 / aspect));
          } else {
            cHeight = 875;
            cWidth = Math.max(520, Math.round(875 * aspect));
          }
        }
        setCanvasDimensions({ w: cWidth, h: cHeight });

        const fullCanvas = document.createElement('canvas');
        fullCanvas.width = cWidth;
        fullCanvas.height = cHeight;
        const fCtx = fullCanvas.getContext('2d');
        if (fCtx) {
          fCtx.fillStyle = '#FFFFFF';
          fCtx.fillRect(0, 0, cWidth, cHeight);
          fCtx.drawImage(img, 0, 0, cWidth, cHeight);
          fullResTargetRgbaRef.current = fCtx.getImageData(0, 0, cWidth, cHeight).data;
        }

        const analysisCanvas = document.createElement('canvas');
        analysisCanvas.width = ANALYSIS_WIDTH;
        analysisCanvas.height = ANALYSIS_HEIGHT;
        const aCtx = analysisCanvas.getContext('2d');
        if (!aCtx) return;

        aCtx.fillStyle = '#FFFFFF';
        aCtx.fillRect(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
        aCtx.drawImage(img, 0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);

        const imgData = aCtx.getImageData(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
        targetRgbaRef.current = imgData.data;

        if (titleLabel) {
          setStudyTitle(titleLabel);
        }
        setReferenceUrl(srcUrl);

        initBlankCanvasBuffers(cWidth, cHeight);

        // Reset weights to scratch initialization at Epoch 0 and record the exact starting loss
        const policy = policyRef.current;
        policy.resetToScratchWeights();
        const initialLoss =
          fullResTargetRgbaRef.current && fullResWorkingRgbaRef.current
            ? computeCanvasL1Loss(fullResTargetRgbaRef.current, fullResWorkingRgbaRef.current)
            : computeCanvasL1Loss(imgData.data, workingRgbaRef.current ?? imgData.data);
        policy.recordLoss(initialLoss, 0);
        setTelemetry(policy.getTelemetry());

        setHasStarted(shouldAutoStart);
        setIsPlaying(shouldAutoStart);
        if (!shouldAutoStart) {
          setLatestReasoning(
            'Weights Initialized from Scratch ~ N(0, σ²) · Epoch 0 · Click "Start Training"'
          );
        }
      };
      img.src = srcUrl;
    },
    [initBlankCanvasBuffers]
  );

  // Initialize with Steve Jobs portrait
  useEffect(() => {
    initializeFromImageSource(DEFAULT_STEVE_JOBS_URL, 'Steve Jobs', autoStartRequested);
  }, [initializeFromImageSource, autoStartRequested]);

  /**
   * Executes a single online neural step (observe, sample the Gaussian policy, score by L1 reward, paint, backprop) and renders it live.
   */
  const executeSingleOnlineStep = useCallback(
    (commitUiState = true) => {
      const targetRgba = targetRgbaRef.current;
      const workingRgba = workingRgbaRef.current;
      const canvas = paintCanvasRef.current;
      if (!targetRgba || !workingRgba || !canvas) return;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      const completed = currentStrokeIdRef.current;
      // 10-Scale Resolution Hierarchy Progression (50,000 actions from 125px -> 1px)
      let currentSize: BrushSize = 125;
      if (completed < 400) currentSize = 125;
      else if (completed < 1200) currentSize = 69;
      else if (completed < 2800) currentSize = 38;
      else if (completed < 6000) currentSize = 21;
      else if (completed < 11500) currentSize = 12;
      else if (completed < 19500) currentSize = 7;
      else if (completed < 29000) currentSize = 4;
      else if (completed < 38000) currentSize = 3;
      else if (completed < 45000) currentSize = 2;
      else currentSize = 1;
      lastBrushSizeRef.current = currentSize;

      const res = executeOnlineStrokeStep({
        policy: policyRef.current,
        targetRgba,
        workingRgba,
        canvasWidth: canvasDimensions.w,
        canvasHeight: canvasDimensions.h,
        analysisWidth: ANALYSIS_WIDTH,
        analysisHeight: ANALYSIS_HEIGHT,
        activeBrushSize: currentSize,
        strokeId: completed,
        fullResTargetRgba: fullResTargetRgbaRef.current ?? undefined,
        fullResWorkingRgba: fullResWorkingRgbaRef.current ?? undefined,
        skipGlobalLossScan: true,
      });

      renderFullStroke(ctx, res.stroke);

      currentStrokeIdRef.current += 1;
      const nextCompleted = currentStrokeIdRef.current;

      const passIdx = BRUSH_SIZES.indexOf(currentSize);
      if (passIdx !== -1) {
        passCountsRef.current[passIdx] = (passCountsRef.current[passIdx] || 0) + 1;
      }

      if (!commitUiState && nextCompleted < TOTAL_PLANNED_STROKES) {
        return;
      }

      // Exact loss of what is actually on screen: every pixel of the live canvas against the
      // target. The same readback re-synchronises the simulation buffer with the real canvas.
      const policy = policyRef.current;
      const fullTarget = fullResTargetRgbaRef.current;
      if (fullTarget && canvas.width > 0 && canvas.height > 0) {
        const live = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        if (live.length === fullTarget.length) {
          policy.recordLoss(computeCanvasL1Loss(fullTarget, live), policy.totalSteps);
          const sim = fullResWorkingRgbaRef.current;
          if (sim && sim.length === live.length) sim.set(live);
        }
      }

      const lastPt = res.stroke.points[res.stroke.points.length - 1] ?? {
        x: res.candidateCoords.x,
        y: res.candidateCoords.y,
        pressure: 0.7,
      };

      renderOverlay({
        x: lastPt.x,
        y: lastPt.y,
        radius: Math.max(3, res.stroke.brushSize * 0.5 * lastPt.pressure),
        visible: nextCompleted < TOTAL_PLANNED_STROKES,
      });

      const totalTarget = TOTAL_PLANNED_STROKES;
      const pct = Math.min(100, Math.round((nextCompleted / totalTarget) * 100));

      setActiveSize(currentSize);
      setActiveHex(res.stroke.hexColor);
      setActivePassName(getPassNameForSize(currentSize));
      setProgressPct(pct);
      setStrokeStats({ completed: nextCompleted, total: totalTarget });
      setLatestReasoning(res.stroke.reasoning);
      setLiveRewardDelta(res.rewardDelta);
      setPassStrokeCounts([...passCountsRef.current]);

      const nav = res.neuralActionVector;
      setCurrentCoords([nav.cx, nav.cy]);
      setCurrentOffsetPx([nav.dxOffsetPx, nav.dyOffsetPx]);
      setCurrentAngleDeg(nav.angleDeg);
      setCurrentLengthScale(nav.lengthScale);
      setCurrentLengthPx(nav.lengthPx);
      setCurrentPressureCurve(nav.pressureCurve);
      setCurrentCurvature(nav.curvature);
      setCurrentRgb(nav.rgb);
      setCurrentTargetRgb(nav.targetRgb);

      setTelemetry(policy.getTelemetry());

      if (nextCompleted >= totalTarget) {
        setIsPlaying(false);
        renderOverlay({ x: 0, y: 0, radius: 0, visible: false });
      }
    },
    [canvasDimensions, renderOverlay]
  );

  // Main Real-Time Online Training & Painting Loop (Parallel Patch Micro-Batched at 60 FPS)
  useEffect(() => {
    if (!isPlaying) return;

    let rafId = 0;

    const tick = () => {
      if (!isPlayingRef.current) return;

      const subSteps =
        speedMultiplierRef.current === 1
          ? 45
          : speedMultiplierRef.current === 2
          ? 110
          : 250;

      for (let s = 0; s < subSteps; s++) {
        const isLast =
          s === subSteps - 1 || currentStrokeIdRef.current + 1 >= TOTAL_PLANNED_STROKES;
        executeSingleOnlineStep(isLast);
        if (currentStrokeIdRef.current >= TOTAL_PLANNED_STROKES) {
          setIsPlaying(false);
          return;
        }
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, executeSingleOnlineStep]);

  const handleTogglePlay = () => {
    if (!hasStarted) {
      setHasStarted(true);
      setIsPlaying(true);
      return;
    }
    if (currentStrokeIdRef.current >= TOTAL_PLANNED_STROKES) {
      handleResetFromScratch();
      setIsPlaying(true);
      return;
    }
    setIsPlaying((p) => !p);
  };

  const handleResetFromScratch = () => {
    setIsPlaying(false);
    setHasStarted(false);
    initBlankCanvasBuffers(canvasDimensions.w, canvasDimensions.h);

    const policy = policyRef.current;
    policy.resetToScratchWeights();
    if (fullResTargetRgbaRef.current && fullResWorkingRgbaRef.current) {
      policy.recordLoss(
        computeCanvasL1Loss(fullResTargetRgbaRef.current, fullResWorkingRgbaRef.current),
        0
      );
    }
    setTelemetry(policy.getTelemetry());
    setLatestReasoning(
      'Weights Re-initialized from Scratch ~ N(0, σ²) · Ready to train online'
    );
  };

  const handleStepSingleStroke = () => {
    setHasStarted(true);
    setIsPlaying(false);
    executeSingleOnlineStep(true);
  };

  const handleStepBurst = (steps: number) => {
    setHasStarted(true);
    for (let i = 0; i < steps; i++) {
      const isLast = i === steps - 1 || currentStrokeIdRef.current + 1 >= TOTAL_PLANNED_STROKES;
      executeSingleOnlineStep(isLast);
      if (currentStrokeIdRef.current >= TOTAL_PLANNED_STROKES) break;
    }
  };

  const handleSelectBrushSize = (targetSize: BrushSize) => {
    const jumpStep = PASS_START_THRESHOLDS[targetSize] ?? 1;
    currentStrokeIdRef.current = jumpStep;
    lastBrushSizeRef.current = targetSize;
    setActiveSize(targetSize);
    setActivePassName(getPassNameForSize(targetSize));
    const pct = Math.min(100, Math.round((jumpStep / TOTAL_PLANNED_STROKES) * 100));
    setProgressPct(pct);
    setStrokeStats({ completed: jumpStep, total: TOTAL_PLANNED_STROKES });
  };

  const handleFileUpload = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      if (dataUrl) {
        const cleanName = file.name.replace(/\.[^/.]+$/, '') || 'Custom Observation';
        setReferenceUrl(dataUrl);
        initializeFromImageSource(dataUrl, cleanName, false);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div
      className="min-h-screen w-full flex flex-col bg-[#F2EFE9] text-[#1C1917]"
      data-state={isPlaying ? 'painting' : hasStarted ? 'paused' : 'ready'}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        aria-label="Upload ground truth observation"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileUpload(file);
          e.target.value = '';
        }}
        className="hidden"
      />

      {/* Top Bar Contract */}
      <header className="w-full flex items-center justify-between gap-8 px-5 py-2.5 bg-[#FAF8F5] border-b border-[#E2DDD3]">
        <div className="flex items-center gap-3 shrink-0">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              handleResetFromScratch();
            }}
            className="text-base font-bold tracking-tight text-[#1C1917] whitespace-nowrap shrink-0"
          >
            Atelier
          </a>
          <span className="hidden lg:inline-block text-xs font-mono-tabular text-[#78716C]">
            12-Layer Deep Residual MLP · {telemetry.totalParameters.toLocaleString()} Float32 Weights ~ N(0, σ²) · Online Adam Backprop
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleTogglePlay}
            data-testid="start-painting-button"
            className="px-4 py-1.5 text-xs font-semibold text-white bg-[#1C1917] rounded hover:bg-[#332E2A] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {!hasStarted
              ? 'Start Training & Painting'
              : isPlaying
              ? 'Pause Training'
              : progressPct >= 100
              ? 'Restart from Scratch'
              : 'Resume Training'}
          </button>
        </div>
      </header>

      {/* Top ML Telemetry & Engine Execution Ribbon */}
      <section
        aria-label="Machine learning telemetry ribbon"
        className="w-full flex flex-wrap items-center justify-between gap-3 px-5 py-2 bg-[#FAF8F5] border-b border-[#E2DDD3] text-xs font-mono-tabular"
      >
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-[#78716C]">Canvas L1 Loss</span>
            <span className="font-semibold text-[#1C1917]">
              {telemetry.loss.toFixed(5)}
            </span>
          </div>

          <span className="h-3.5 w-[1px] bg-[#E2DDD3]" aria-hidden="true" />

          <div className="flex items-center gap-2">
            <span className="text-[#78716C]">Grad Norm ‖∇W‖</span>
            <span className="font-semibold text-[#1C1917]">
              {telemetry.gradientNorm.toFixed(5)}
            </span>
          </div>

          <span className="h-3.5 w-[1px] bg-[#E2DDD3]" aria-hidden="true" />

          <div className="flex items-center gap-2" title="Change in canvas L1 caused by the last stroke (negative = improvement)">
            <span className="text-[#78716C]">Last Stroke ΔL1</span>
            <span
              className={`font-semibold ${
                liveRewardDelta >= 0 ? 'text-emerald-700' : 'text-rose-700'
              }`}
            >
              {formatSignedValue(-liveRewardDelta)}
            </span>
          </div>

          <span className="h-3.5 w-[1px] bg-[#E2DDD3]" aria-hidden="true" />

          <div className="flex items-center gap-2">
            <span className="text-[#78716C]">Backprop Steps</span>
            <span className="font-semibold text-[#1C1917]">
              {telemetry.totalSteps.toLocaleString()} / {strokeStats.total.toLocaleString()}
            </span>
          </div>

          <span className="h-3.5 w-[1px] bg-[#E2DDD3]" aria-hidden="true" />

          <div className="flex items-center gap-2">
            <span className="text-[#78716C]">Architecture</span>
            <span className="font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
              100% Scratch Neural Policy (Epoch {telemetry.epoch})
            </span>
          </div>
        </div>

        {/* Right Execution & Inspection Controls */}
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            onClick={handleStepSingleStroke}
            className="px-2.5 py-1 rounded border border-[#D4CFC4] bg-white text-[#57534E] hover:text-[#1C1917] hover:bg-[#EFECE6] transition-colors whitespace-nowrap cursor-pointer"
          >
            Step Stroke
          </button>
          <button
            type="button"
            onClick={handleResetFromScratch}
            className="px-2.5 py-1 rounded border border-[#D4CFC4] bg-white text-[#57534E] hover:text-[#1C1917] hover:bg-[#EFECE6] transition-colors whitespace-nowrap cursor-pointer"
          >
            Reset from Scratch
          </button>
          <button
            type="button"
            onClick={() => setShowFlowOverlay((v) => !v)}
            aria-pressed={showFlowOverlay}
            className={`px-2.5 py-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
              showFlowOverlay
                ? 'border-[#1C1917] bg-[#1C1917] text-white'
                : 'border-[#D4CFC4] bg-white text-[#57534E] hover:text-[#1C1917]'
            }`}
          >
            Neural Vector Field
          </button>

          {/* Speed Selector */}
          <div
            className="flex items-center rounded border border-[#D4CFC4] bg-white p-0.5"
            role="group"
            aria-label="Agent execution speed"
          >
            {[1, 2, 4].map((spd) => (
              <button
                key={spd}
                type="button"
                onClick={() => setSpeedMultiplier(spd)}
                className={`px-2 py-0.5 rounded-[3px] text-[11px] transition-colors whitespace-nowrap cursor-pointer ${
                  speedMultiplier === spd
                    ? 'bg-[#1C1917] text-white font-medium'
                    : 'text-[#655E57] hover:text-[#1C1917]'
                }`}
              >
                {spd}×
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* 3-Column Research Observatory Workbench (1440px Desktop Baseline) */}
      <main className="flex-1 w-full px-5 py-3.5 grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
        {/* Left Column (3 cols): Dual-Tensor Sensory Field & Multi-Scale Hierarchy */}
        <section
          aria-label="Sensory inputs and multi-scale hierarchy"
          className="lg:col-span-3 flex flex-col gap-3"
        >
          <DualTensorSensoryView
            referenceUrl={referenceUrl}
            studyTitle={studyTitle}
            canvasDimensions={canvasDimensions}
            analysisDimensions={{ w: ANALYSIS_WIDTH, h: ANALYSIS_HEIGHT }}
            residualL1Error={telemetry.loss}
            targetRgba={targetRgbaRef.current}
            workingRgba={workingRgbaRef.current}
            stepCount={strokeStats.completed}
            onUploadClick={() => fileInputRef.current?.click()}
            onFileDrop={handleFileUpload}
          />

          <PassScheduleView
            activeSize={activeSize}
            activePassName={activePassName}
            activeHex={activeHex}
            progressPct={progressPct}
            completedStrokes={strokeStats.completed}
            totalStrokes={strokeStats.total}
            passStrokeCounts={passStrokeCounts}
            onSelectSize={handleSelectBrushSize}
          />
        </section>

        {/* Center Column (6 cols): Canvas as the Agent's Environment */}
        <section
          aria-label="Agent painting environment"
          className="lg:col-span-6 flex flex-col"
        >
          <div className="flex-1 rounded border border-[#DCD6CA] bg-[#FAF8F5] p-3 lg:p-4 flex flex-col justify-between">
            {/* Canvas Header Strip */}
            <div className="flex items-center justify-between text-xs text-[#57534E] pb-2 border-b border-[#EAE5DC] font-mono-tabular">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-[#1C1917]">Environment Canvas</span>
                <span className="text-[#A8A093]">·</span>
                <span>
                  Scale {activeSize}px ({activePassName})
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-[2px] border border-[#ABA396]"
                  style={{ backgroundColor: activeHex }}
                  aria-hidden="true"
                />
                <span data-testid="active-hex-readout">{activeHex}</span>
              </div>
            </div>

            {/* Framed Canvas Stage */}
            <div className="flex-1 flex items-center justify-center py-2 min-h-[460px]">
              <div
                className="relative w-full max-h-[calc(100vh-210px)] flex items-center justify-center"
                style={{
                  aspectRatio: `${canvasDimensions.w} / ${canvasDimensions.h}`,
                  maxWidth:
                    canvasDimensions.w >= canvasDimensions.h
                      ? '100%'
                      : `calc((100vh - 210px) * ${canvasDimensions.w / canvasDimensions.h})`,
                }}
              >
                <div className="relative w-full h-full bg-white border border-[#CFC9BC] shadow-xs overflow-hidden">
                  <canvas
                    ref={paintCanvasRef}
                    width={canvasDimensions.w}
                    height={canvasDimensions.h}
                    data-testid="painting-canvas"
                    className="w-full h-full block bg-white"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    width={canvasDimensions.w}
                    height={canvasDimensions.h}
                    className="pointer-events-none absolute inset-0 w-full h-full block"
                  />

                  {/* Ready Overlay */}
                  {!hasStarted && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/85 p-6 text-center backdrop-blur-[1px]">
                      <div className="max-w-sm rounded border border-[#DFDBD2] bg-[#FAF8F5] p-5 shadow-xs">
                        <div className="font-bold text-sm text-[#1C1917]">
                          {studyTitle}
                        </div>
                        <p className="mt-1.5 text-xs text-[#655E57] leading-relaxed">
                          12-layer deep residual MLP ({telemetry.totalParameters.toLocaleString()} Float32 parameters) initialized
                          with random Box-Muller Gaussian weights ~ N(0, σ²) at Epoch 0.
                          Observes Dual-Tensor feedback [I_target, I_canvas, ΔI] and outputs
                          13-D continuous stroke actions, learning online from the L1 reward of its own strokes via Adam backpropagation.
                        </p>
                        <div className="mt-4 flex items-center justify-center gap-2.5">
                          <button
                            type="button"
                            onClick={handleTogglePlay}
                            className="rounded bg-[#1C1917] px-4 py-2 text-xs font-medium text-white hover:bg-[#332E2A] transition-colors cursor-pointer"
                          >
                            Start Training &amp; Painting
                          </button>
                          <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="rounded border border-[#D4CFC4] bg-white px-3.5 py-2 text-xs font-medium text-[#1C1917] hover:bg-[#EFECE6] transition-colors cursor-pointer"
                          >
                            Upload Observation
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Canvas Footer Status */}
            <div className="pt-2 border-t border-[#EAE5DC] flex items-center justify-between text-xs text-[#57534E] font-mono-tabular">
              <span>
                {canvasDimensions.w} × {canvasDimensions.h} px
              </span>
              <span>
                {strokeStats.completed.toLocaleString()} / {strokeStats.total.toLocaleString()}{' '}
                actions
              </span>
              <span data-testid="painting-progress" className="font-semibold text-[#1C1917]">
                {progressPct}%
              </span>
            </div>
          </div>
        </section>

        {/* Right Column (3 cols): Deep Learning Observatory */}
        <aside
          aria-label="Deep learning telemetry and weight observatory"
          className="lg:col-span-3 flex flex-col gap-3"
        >
          {/* 1. Real-Time Canvas L1 Loss Convergence Graph */}
          <LossChart
            lossHistory={telemetry.lossHistory}
            lossSteps={telemetry.lossSteps}
            currentLoss={telemetry.loss}
            epoch={telemetry.epoch}
            totalSteps={telemetry.totalSteps}
            learningRate={telemetry.learningRate}
          />

          {/* 2. 12-Layer Synaptic Highway & Live Weight Matrix Inspector */}
          <LayerInspector
            layers={telemetry.layerSummaries}
            gradientNorm={telemetry.gradientNorm}
            totalParameters={telemetry.totalParameters}
            weightSnapshot={telemetry.weightSnapshot}
          />

          {/* 3. Continuous 13-D Policy Action Head Vector */}
          <PolicyActionHead
            cx={currentCoords[0]}
            cy={currentCoords[1]}
            dxOffsetPx={currentOffsetPx[0]}
            dyOffsetPx={currentOffsetPx[1]}
            angleDeg={currentAngleDeg}
            lengthScale={currentLengthScale}
            lengthPx={currentLengthPx}
            pressureCurve={currentPressureCurve}
            curvature={currentCurvature}
            rgb={currentRgb}
            hexColor={activeHex}
            reasoning={latestReasoning}
            brushSize={activeSize}
          />

          {/* 4. Online Adam Backprop & Chromatic Learning Console */}
          <TrainingConsole
            telemetry={telemetry}
            targetRgb={currentTargetRgb}
            predictedRgb={currentRgb}
            onStepBurst={handleStepBurst}
            onResetWeights={handleResetFromScratch}
          />
        </aside>
      </main>
    </div>
  );
}
