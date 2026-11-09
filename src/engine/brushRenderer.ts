import { PaintTool, PlannedStroke, StrokePoint } from './neuralPainter';

/**
 * Exact physical radius calibrated across all 10 multi-scale neural passes (125px down to 1px)
 * Shared between HTML5 Canvas 2D rendering and the neural Dual-Tensor working buffer simulation.
 */
export function getPhysicalBrushRadius(
  brushSize: number,
  avgPressure: number,
  tool: PaintTool = 'brush'
): number {
  const pScale = 0.82 + 0.18 * avgPressure;
  if (tool === 'detail') {
    return Math.max(0.45, brushSize * 0.34 * pScale);
  }
  if (tool === 'sketch') {
    return Math.max(0.55, brushSize * 0.38 * pScale);
  }
  if (tool === 'glaze') {
    return Math.max(0.75, brushSize * 0.44 * pScale);
  }

  if (brushSize <= 1) return 0.55 * pScale;
  if (brushSize === 2) return 0.82 * pScale;
  if (brushSize === 3) return 1.15 * pScale;
  if (brushSize === 4) return 1.52 * pScale;
  if (brushSize === 7) return 2.45 * pScale;
  if (brushSize === 12) return 3.8 * pScale;
  if (brushSize === 21) return 6.0 * pScale;
  if (brushSize === 38) return 10.5 * pScale;
  if (brushSize === 69) return 18.5 * (0.88 + 0.12 * avgPressure);
  return 32.0 * (0.88 + 0.12 * avgPressure);
}

/**
 * Renders a single sub-segment of a PlannedStroke onto a 2D CanvasRenderingContext2D.
 */
export function renderStrokeSegment(
  ctx: CanvasRenderingContext2D,
  stroke: PlannedStroke,
  fromPt: StrokePoint,
  toPt: StrokePoint,
  segmentIndex: number
): void {
  const { brushSize, tool, r, g, b, tangentAngle } = stroke;
  const avgPressure = 0.5 * (fromPt.pressure + toPt.pressure);

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const baseRadius = getPhysicalBrushRadius(brushSize, avgPressure, tool);

  const cosT = Math.cos(tangentAngle);
  const sinT = Math.sin(tangentAngle);
  const normX = -sinT;
  const normY = cosT;

  let x0 = fromPt.x;
  let y0 = fromPt.y;
  let x1 = toPt.x;
  let y1 = toPt.y;
  const segDist = Math.hypot(x1 - x0, y1 - y0);
  if (segDist < 0.45) {
    const halfMin = 0.25;
    x0 = fromPt.x - cosT * halfMin;
    y0 = fromPt.y - sinT * halfMin;
    x1 = fromPt.x + cosT * halfMin;
    y1 = fromPt.y + sinT * halfMin;
  }

  const colorStr = `rgb(${r}, ${g}, ${b})`;

  ctx.strokeStyle = colorStr;
  ctx.lineWidth = Math.max(0.95, baseRadius * 1.94);
  ctx.globalAlpha = brushSize >= 69 ? 1.0 : brushSize >= 12 ? 0.94 : 0.98;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  if (brushSize >= 21 && segmentIndex % 2 === 0) {
    const ridgeOffset = baseRadius * 0.28;
    const liftR = Math.min(255, r + 3);
    const liftG = Math.min(255, g + 3);
    const liftB = Math.min(255, b + 3);
    ctx.strokeStyle = `rgb(${liftR}, ${liftG}, ${liftB})`;
    ctx.lineWidth = Math.max(0.7, baseRadius * 0.16);
    ctx.globalAlpha = 0.08;
    ctx.beginPath();
    ctx.moveTo(x0 + normX * ridgeOffset, y0 + normY * ridgeOffset);
    ctx.lineTo(x1 + normX * ridgeOffset, y1 + normY * ridgeOffset);
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Renders an entire PlannedStroke in a single batched Canvas 2D path.
 */
export function renderFullStroke(
  ctx: CanvasRenderingContext2D,
  stroke: PlannedStroke
): void {
  const pts = stroke.points;
  if (pts.length === 0) return;
  if (pts.length === 1) {
    renderStrokeSegment(ctx, stroke, pts[0], pts[0], 0);
    return;
  }

  const { brushSize, r, g, b, tangentAngle } = stroke;
  const midPt = pts[Math.floor(pts.length / 2)];
  const avgPressure = midPt.pressure;

  const baseRadius = getPhysicalBrushRadius(brushSize, avgPressure, 'brush');

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
  ctx.lineWidth = Math.max(0.9, baseRadius * 1.94);
  // Subtle glazing blend (0.94) on mid-scales smooths out transitions on skin before 0.98 micro-detailing
  ctx.globalAlpha = brushSize >= 69 ? 1.0 : brushSize >= 12 ? 0.94 : 0.98;

  const first = pts[0];
  const last = pts[pts.length - 1];
  const totalDist = Math.hypot(last.x - first.x, last.y - first.y);

  ctx.beginPath();
  if (totalDist < 0.45) {
    const cosT = Math.cos(tangentAngle);
    const sinT = Math.sin(tangentAngle);
    ctx.moveTo(first.x - cosT * 0.25, first.y - sinT * 0.25);
    ctx.lineTo(first.x + cosT * 0.25, first.y + sinT * 0.25);
  } else {
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
  }
  ctx.stroke();
}
