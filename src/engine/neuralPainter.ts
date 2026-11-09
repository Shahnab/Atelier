/**
 * Atelier: a neural painter built and trained from scratch in pure TypeScript.
 * No ML libraries, no pre-trained weights, no hand-written stroke planner.
 *
 * Network: a 12-layer residual MLP (10 -> 32x9 -> 24 -> 16 -> 8) plus a per-channel chromatic
 * sub-network and a spatial-offset head: 10,401 Float32 parameters, Gaussian-initialised at Epoch 0.
 *
 * Every stroke:
 * 1. A reward-driven fixation memory picks where to look (it never reads target pixels directly).
 * 2. The network observes the dual-tensor patch (target, canvas, residual) and outputs the full
 *    13-D action [cos θ, sin θ, length, pAttack, pPeak, pRelease, c1, c2, R, G, B, Δx, Δy].
 * 3. The policy is Gaussian: K antithetic samples around the network's output are painted
 *    hypothetically and scored by the real L1 improvement they would cause on the canvas.
 * 4. The best sample is painted. Nothing clips, snaps or overrides the decoded parameters.
 * 5. The network is updated by backpropagation + Adam:
 *    - geometry/pressure/curvature/offset: evolution-strategies policy gradient from the rewards,
 *    - colour: the exact analytic gradient of the L1 loss with respect to the stroke colour.
 */

export const BRUSH_SIZES = [125, 69, 38, 21, 12, 7, 4, 3, 2, 1] as const;
export type BrushSize = (typeof BRUSH_SIZES)[number];

export type PaintTool = 'brush' | 'sketch' | 'glaze' | 'detail';

export type PaintingPassName =
  | 'Coarse Receptive Field'
  | 'Mid-Scale Tensor'
  | 'Fine Residual Field'
  | 'Micro-Pixel Convergence';

export interface StrokePoint {
  x: number;
  y: number;
  pressure: number;
}

export interface PlannedStroke {
  id: number;
  passIndex: number;
  passName: PaintingPassName;
  brushSize: BrushSize;
  tool: PaintTool;
  r: number;
  g: number;
  b: number;
  hexColor: string;
  opacity: number;
  points: StrokePoint[];
  tangentAngle: number;
  coherence: number;
  importance: number;
  reasoning: string;
}

export interface LayerSummary {
  index: number;
  name: string;
  inDim: number;
  outDim: number;
  residual: boolean;
  meanAbsWeight: number;
  gradNorm: number;
  /** This layer's ‖∇W‖ as a percentage of the largest layer ‖∇W‖ (exact, 0..100). */
  relativeGradPct: number;
}

export interface NeuralTelemetry {
  numLayers: number;
  totalParameters: number;
  architectureLabel: string;
  epoch: number;
  totalSteps: number;
  /** Exact mean absolute RGB error of the canvas against the target (0..1). */
  loss: number;
  /** Mean |painted colour - observed target colour| / 255 for the latest stroke. */
  colorLoss: number;
  /** L2 norm of the latest backpropagated gradient (before clipping). */
  gradientNorm: number;
  /** L2 norms of Adam's first/second moment vectors over all parameters. */
  adamFirstMomentNorm: number;
  adamSecondMomentNorm: number;
  learningRate: number;
  layerSummaries: LayerSummary[];
  weightSnapshot: number[];
  /** Full-precision loss samples and the training step each was measured at. */
  lossHistory: number[];
  lossSteps: number[];
}

export function getPassNameForSize(size: BrushSize): PaintingPassName {
  if (size >= 69) return 'Coarse Receptive Field';
  if (size >= 21) return 'Mid-Scale Tensor';
  if (size >= 7) return 'Fine Residual Field';
  return 'Micro-Pixel Convergence';
}

export function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const toHex = (v: number) => clamp(v).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/** Signed value with 3 significant digits, switching to scientific notation for tiny numbers. */
export function formatSignedValue(v: number): string {
  if (v === 0) return '0';
  const sign = v > 0 ? '+' : '−';
  const a = Math.abs(v);
  return `${sign}${a >= 0.001 ? a.toFixed(5) : a.toExponential(2)}`;
}

/**
 * Physical radius calibrated across all 10 multi-scale passes (125px down to 1px).
 * Shared between Canvas 2D rendering and the neural working-buffer simulation.
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

export interface DualTensorObservation {
  targetR: number;
  targetG: number;
  targetB: number;
  canvasR: number;
  canvasG: number;
  canvasB: number;
  diffR: number;
  diffG: number;
  diffB: number;
  normX: number;
  normY: number;
  passScale: number;
  localError: number;
}

/**
 * Extracts the dual-tensor observation (target, canvas, residual) from raw buffers at (fx, fy).
 */
export function extractDualTensorObservation(
  targetRgba: Uint8ClampedArray | Uint8Array,
  canvasRgba: Uint8ClampedArray | Uint8Array,
  fw: number,
  fh: number,
  fx: number,
  fy: number,
  passScale: number
): DualTensorObservation {
  const idx = (fy * fw + fx) * 4;

  const targetR = targetRgba[idx] / 255;
  const targetG = targetRgba[idx + 1] / 255;
  const targetB = targetRgba[idx + 2] / 255;

  const canvasR = canvasRgba[idx] / 255;
  const canvasG = canvasRgba[idx + 1] / 255;
  const canvasB = canvasRgba[idx + 2] / 255;

  const diffR = targetR - canvasR;
  const diffG = targetG - canvasG;
  const diffB = targetB - canvasB;

  const normX = (fx / Math.max(1, fw - 1)) * 2 - 1;
  const normY = (fy / Math.max(1, fh - 1)) * 2 - 1;
  const localError = Math.sqrt(0.3 * diffR * diffR + 0.59 * diffG * diffG + 0.11 * diffB * diffB);

  return {
    targetR,
    targetG,
    targetB,
    canvasR,
    canvasG,
    canvasB,
    diffR,
    diffG,
    diffB,
    normX,
    normY,
    passScale,
    localError,
  };
}

const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPS = 1e-8;
const WEIGHT_DECAY = 1e-4;
const MAX_GRAD_NORM = 5;

/** Number of Gaussian-policy samples (antithetic pairs) scored per stroke. */
export const EXPLORATION_SAMPLES = 8;

function createRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fresh entropy for a "pure random" network initialisation. */
export function randomSeed(): number {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.getRandomValues === 'function') {
    const a = new Uint32Array(1);
    c.getRandomValues(a);
    return a[0] || 1;
  }
  return ((Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0) || 1;
}

function adamUpdate(
  p: Float32Array,
  m: Float32Array,
  v: Float32Array,
  i: number,
  g: number,
  lr: number,
  bc1: number,
  bc2: number,
  decay = 0
): void {
  m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * g;
  v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * g * g;
  p[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + ADAM_EPS) + decay * p[i]);
}

export interface DeepMlpLayer {
  index: number;
  inDim: number;
  outDim: number;
  residual: boolean;
  activation: 'leaky_relu' | 'tanh';
  w: Float32Array;
  b: Float32Array;
  mW: Float32Array;
  vW: Float32Array;
  mB: Float32Array;
  vB: Float32Array;
  lastGradNorm: number;
}

/**
 * Reward-driven fixation memory. It is non-parametric and never reads target pixels:
 * it only remembers how much L1 improvement strokes painted from each canvas cell produced
 * (with an optimism bonus for rarely visited cells) and samples the next fixation from that.
 */
export class FixationAttention {
  private static readonly GRID = 24;
  private static readonly TOURNAMENT = 8;
  private value = new Float32Array(FixationAttention.GRID * FixationAttention.GRID).fill(1);
  private visits = new Uint32Array(FixationAttention.GRID * FixationAttention.GRID);
  private pass = -1;
  private rewardScale = 0;

  public reset(pass = -1): void {
    this.value.fill(1);
    this.visits.fill(0);
    this.rewardScale = 0;
    this.pass = pass;
  }

  public propose(
    rand: () => number,
    canvasWidth: number,
    canvasHeight: number,
    pass: number
  ): { x: number; y: number; cell: number } {
    if (pass !== this.pass) this.reset(pass);
    const g = FixationAttention.GRID;
    let bestCell = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < FixationAttention.TOURNAMENT; k++) {
      const cell = Math.floor(rand() * g * g);
      const score = this.value[cell] + 0.6 / Math.sqrt(this.visits[cell] + 1);
      if (score > bestScore) {
        bestScore = score;
        bestCell = cell;
      }
    }
    const cx = bestCell % g;
    const cy = Math.floor(bestCell / g);
    return {
      x: Math.min(canvasWidth - 1, Math.floor(((cx + rand()) / g) * canvasWidth)),
      y: Math.min(canvasHeight - 1, Math.floor(((cy + rand()) / g) * canvasHeight)),
      cell: bestCell,
    };
  }

  public record(cell: number, reward: number): void {
    this.rewardScale =
      this.rewardScale === 0 ? Math.abs(reward) : 0.98 * this.rewardScale + 0.02 * Math.abs(reward);
    const norm = Math.max(-1, Math.min(4, reward / Math.max(1e-9, this.rewardScale)));
    this.value[cell] = 0.6 * this.value[cell] + 0.4 * norm;
    this.visits[cell] += 1;
  }
}

export interface PolicyStepResult {
  gradNorm: number;
}

/**
 * The neural stroke policy (see file header). All weights are drawn from N(0, σ²) at construction.
 */
export class NeuralStrokePolicy {
  public readonly layerDims = [10, 32, 32, 32, 32, 32, 32, 32, 32, 32, 24, 16, 8] as const;
  public readonly layers: DeepMlpLayer[] = [];

  // Learned residual projection highway for kinematics (10 -> 8)
  public wSkipKinematics = new Float32Array(10 * 8);
  public mWSkipKinematics = new Float32Array(10 * 8);
  public vWSkipKinematics = new Float32Array(10 * 8);

  // Per-channel 2-layer chromatic sub-network: 3 x ([4 -> 8 -> 1] + skip [4 -> 1])
  public wChroma1 = new Float32Array(3 * 4 * 8);
  public mWChroma1 = new Float32Array(3 * 4 * 8);
  public vWChroma1 = new Float32Array(3 * 4 * 8);
  public bChroma1 = new Float32Array(3 * 8);
  public mBChroma1 = new Float32Array(3 * 8);
  public vBChroma1 = new Float32Array(3 * 8);

  public wChroma2 = new Float32Array(3 * 8);
  public mWChroma2 = new Float32Array(3 * 8);
  public vWChroma2 = new Float32Array(3 * 8);

  public wChromaSkip = new Float32Array(3 * 4);
  public mWChromaSkip = new Float32Array(3 * 4);
  public vWChromaSkip = new Float32Array(3 * 4);
  public bChroma2 = new Float32Array(3);
  public mBChroma2 = new Float32Array(3);
  public vBChroma2 = new Float32Array(3);

  // Sub-patch spatial offset head (16 -> 2), reading the last hidden layer
  public wSpatialOffset = new Float32Array(16 * 2);
  public mWSpatialOffset = new Float32Array(16 * 2);
  public vWSpatialOffset = new Float32Array(16 * 2);
  public bSpatialOffset = new Float32Array(2);
  public mBSpatialOffset = new Float32Array(2);
  public vBSpatialOffset = new Float32Array(2);

  public readonly attention = new FixationAttention();

  public epoch = 0;
  public totalSteps = 0;
  public lastLoss = 0;
  public lastColorLoss = 0;
  public lastGradNorm = 0;
  public learningRate = 0.003;
  public chromaLearningRate = 0.02;
  /** L2 penalty on the output logits so the tanh heads never saturate and stop learning. */
  public logitRegularization = 0.1;
  public lossHistory: number[] = [];
  public lossSteps: number[] = [];
  private lossStride = 1;
  private static readonly MAX_LOSS_POINTS = 600;

  private adamT = 0;
  private rng: () => number = createRng(1);

  constructor(seed: number = randomSeed()) {
    this.resetToScratchWeights(seed);
  }

  public nextRandom(): number {
    return this.rng();
  }

  public nextGaussian(): number {
    const u1 = Math.max(1e-7, this.rng());
    const u2 = this.rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /** Exploration std of the Gaussian policy, annealed as the network trains. */
  public explorationSigma(): number {
    return 0.1 + 0.22 * Math.exp(-this.totalSteps / 12000);
  }

  /**
   * Records an exactly measured canvas loss at a training step. The history keeps the full
   * run (not just a recent window): when it grows past MAX_LOSS_POINTS it is thinned to every
   * other sample and the sampling interval doubles, so stored values are never altered.
   */
  public recordLoss(loss: number, step: number = this.totalSteps): void {
    this.lastLoss = loss;
    const n = this.lossSteps.length;
    if (n > 0 && step === this.lossSteps[n - 1]) {
      this.lossHistory[n - 1] = loss;
      return;
    }
    if (n > 0 && step - this.lossSteps[n - 1] < this.lossStride) return;

    this.lossHistory.push(loss);
    this.lossSteps.push(step);
    if (this.lossHistory.length > NeuralStrokePolicy.MAX_LOSS_POINTS) {
      this.lossHistory = this.lossHistory.filter((_, i) => i % 2 === 0);
      this.lossSteps = this.lossSteps.filter((_, i) => i % 2 === 0);
      this.lossStride *= 2;
    }
  }

  /**
   * Re-draws every weight from N(0, σ²) (Glorot/He-scaled) with zero biases, clears all
   * optimiser state and starts again at epoch 0. No pre-training of any kind.
   */
  public resetToScratchWeights(seed: number = randomSeed()): NeuralTelemetry {
    this.rng = createRng(seed || 1);

    this.layers.length = 0;
    this.epoch = 0;
    this.totalSteps = 0;
    this.adamT = 0;
    this.lastLoss = 0;
    this.lastColorLoss = 0;
    this.lastGradNorm = 0;
    this.lossHistory = [];
    this.lossSteps = [];
    this.lossStride = 1;
    this.attention.reset();

    const numLayers = this.layerDims.length - 1;
    for (let l = 0; l < numLayers; l++) {
      const inDim = this.layerDims[l];
      const outDim = this.layerDims[l + 1];
      const isLast = l === numLayers - 1;
      const residual = !isLast && inDim === outDim;
      const scale = Math.sqrt(2.0 / (inDim + outDim)) * (residual ? 0.35 : 0.65);

      const w = new Float32Array(inDim * outDim);
      for (let i = 0; i < w.length; i++) w[i] = this.nextGaussian() * scale;

      this.layers.push({
        index: l + 1,
        inDim,
        outDim,
        residual,
        activation: isLast ? 'tanh' : 'leaky_relu',
        w,
        b: new Float32Array(outDim),
        mW: new Float32Array(w.length),
        vW: new Float32Array(w.length),
        mB: new Float32Array(outDim),
        vB: new Float32Array(outDim),
        lastGradNorm: 0,
      });
    }

    this.initAuxiliaryHeads();
    return this.getTelemetry();
  }

  private initAuxiliaryHeads(): void {
    const fill = (arr: Float32Array, scale: number) => {
      for (let i = 0; i < arr.length; i++) arr[i] = this.nextGaussian() * scale;
    };
    const zero = (...arrs: Float32Array[]) => arrs.forEach((a) => a.fill(0));

    fill(this.wSkipKinematics, Math.sqrt(2.0 / (10 + 8)) * 0.35);
    fill(this.wChroma1, Math.sqrt(2.0 / (4 + 8)) * 0.5);
    fill(this.wChroma2, Math.sqrt(2.0 / (8 + 1)) * 0.5);
    fill(this.wChromaSkip, Math.sqrt(2.0 / (4 + 1)) * 0.5);
    fill(this.wSpatialOffset, Math.sqrt(2.0 / (16 + 2)) * 0.35);

    zero(
      this.mWSkipKinematics,
      this.vWSkipKinematics,
      this.mWChroma1,
      this.vWChroma1,
      this.bChroma1,
      this.mBChroma1,
      this.vBChroma1,
      this.mWChroma2,
      this.vWChroma2,
      this.mWChromaSkip,
      this.vWChromaSkip,
      this.bChroma2,
      this.mBChroma2,
      this.vBChroma2,
      this.mWSpatialOffset,
      this.vWSpatialOffset,
      this.bSpatialOffset,
      this.mBSpatialOffset,
      this.vBSpatialOffset
    );
  }

  private leakyRelu(x: number): number {
    return x >= 0 ? x : 0.12 * x;
  }

  private dLeakyRelu(preAct: number): number {
    return preAct >= 0 ? 1.0 : 0.12;
  }

  public forwardWithCache(input: Float32Array | number[]): {
    inputs: Float32Array[];
    preActivations: Float32Array[];
    out: Float32Array;
  } {
    const inputs: Float32Array[] = [];
    const preActivations: Float32Array[] = [];

    const input0 = Float32Array.from(input);
    let current = input0;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      inputs.push(current);

      const z = new Float32Array(layer.outDim);
      const nextAct = new Float32Array(layer.outDim);
      const isLast = l === this.layers.length - 1;

      for (let o = 0; o < layer.outDim; o++) {
        let sum = layer.b[o];
        const rowOffset = o * layer.inDim;
        for (let i = 0; i < layer.inDim; i++) {
          sum += layer.w[rowOffset + i] * current[i];
        }
        if (isLast && input0.length === 10) {
          const skipOffset = o * 10;
          for (let i = 0; i < 10; i++) {
            sum += this.wSkipKinematics[skipOffset + i] * input0[i];
          }
        }
        z[o] = sum;
        if (layer.activation === 'tanh') {
          nextAct[o] = Math.tanh(sum);
        } else {
          const act = this.leakyRelu(sum);
          nextAct[o] = layer.residual ? 0.78 * current[o] + 0.35 * act : act;
        }
      }

      preActivations.push(z);
      current = nextAct;
    }

    return { inputs, preActivations, out: current };
  }

  public forward(input: Float32Array | number[]): Float32Array {
    return this.forwardWithCache(input).out;
  }

  /**
   * Forward pass predicting ALL 13 stroke parameters:
   * - kinematics (8): [cos θ, sin θ, lengthScale, pAttack, pPeak, pRelease, c1, c2]
   * - colorPos (5): [R, G, B, dxOffset, dyOffset]
   */
  public forwardFullAction(
    input: Float32Array | number[],
    pigmentInput: Float32Array | number[]
  ): {
    kinematics: Float32Array;
    colorPos: Float32Array;
    h11: Float32Array;
    chromaHidden: Float32Array;
  } {
    const { inputs, out } = this.forwardWithCache(input);
    const h11 = inputs[inputs.length - 1] ?? new Float32Array(16);
    const colorPos = new Float32Array(5);
    const chromaHidden = new Float32Array(3 * 8);

    for (let c = 0; c < 3; c++) {
      const chIn = this.chromaInput(pigmentInput, c);
      let outSum = this.bChroma2[c];
      for (let i = 0; i < 4; i++) outSum += this.wChromaSkip[c * 4 + i] * chIn[i];

      for (let h = 0; h < 8; h++) {
        let z1 = this.bChroma1[c * 8 + h];
        const rowOff = c * 32 + h * 4;
        for (let i = 0; i < 4; i++) z1 += this.wChroma1[rowOff + i] * chIn[i];
        const a1 = this.leakyRelu(z1);
        chromaHidden[c * 8 + h] = a1;
        outSum += this.wChroma2[c * 8 + h] * a1;
      }
      colorPos[c] = Math.max(-1, Math.min(1, outSum));
    }

    for (let o = 0; o < 2; o++) {
      let sum = this.bSpatialOffset[o];
      for (let h = 0; h < 16; h++) sum += this.wSpatialOffset[o * 16 + h] * (h11[h] ?? 0);
      colorPos[3 + o] = Math.tanh(sum);
    }

    return { kinematics: out, colorPos, h11, chromaHidden };
  }

  private chromaInput(pigmentInput: ArrayLike<number>, c: number): number[] {
    return [
      pigmentInput[c] ?? 0, // target_c in [-1, 1]
      pigmentInput[c + 3] ?? 0, // canvas_c in [-1, 1]
      pigmentInput[c + 6] ?? 0, // residual delta_c
      pigmentInput[9] ?? 0.5, // pass scale
    ];
  }

  /**
   * Exact backpropagation through the 12 layers (plus the kinematic skip highway) followed by
   * global-norm clipping and AdamW. `dOut` is dLoss/dOutput; `extraDLastIn` is an optional extra
   * gradient flowing into the last hidden layer (from the spatial-offset head).
   */
  private backwardTrunk(
    inputs: Float32Array[],
    preActivations: Float32Array[],
    dOut: ArrayLike<number>,
    extraDLastIn: Float32Array | null,
    lr: number,
    bc1: number,
    bc2: number
  ): number {
    const numLayers = this.layers.length;
    const gW: Float32Array[] = this.layers.map((l) => new Float32Array(l.w.length));
    const gB: Float32Array[] = this.layers.map((l) => new Float32Array(l.b.length));
    const gSkip = new Float32Array(this.wSkipKinematics.length);
    const input0 = inputs[0];
    const hasSkip = input0.length === 10;
    const logitReg = extraDLastIn ? this.logitRegularization : 0;

    let dAct = Float32Array.from(dOut as ArrayLike<number>);

    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const layerIn = inputs[l];
      const z = preActivations[l];
      const isLast = l === numLayers - 1;

      const dz = new Float32Array(layer.outDim);
      for (let o = 0; o < layer.outDim; o++) {
        if (layer.activation === 'tanh') {
          const t = Math.tanh(z[o]);
          dz[o] = dAct[o] * (1 - t * t) + logitReg * z[o];
        } else {
          dz[o] = dAct[o] * (layer.residual ? 0.35 : 1.0) * this.dLeakyRelu(z[o]);
        }
      }

      if (isLast && hasSkip) {
        for (let o = 0; o < layer.outDim; o++) {
          for (let i = 0; i < 10; i++) gSkip[o * 10 + i] = dz[o] * input0[i];
        }
      }

      const dPrev = new Float32Array(layer.inDim);
      if (layer.residual) {
        for (let i = 0; i < layer.inDim; i++) dPrev[i] = 0.78 * dAct[i];
      }
      for (let o = 0; o < layer.outDim; o++) {
        gB[l][o] = dz[o];
        const rowOffset = o * layer.inDim;
        for (let i = 0; i < layer.inDim; i++) {
          gW[l][rowOffset + i] = dz[o] * layerIn[i];
          dPrev[i] += layer.w[rowOffset + i] * dz[o];
        }
      }
      if (isLast && extraDLastIn) {
        for (let i = 0; i < layer.inDim; i++) dPrev[i] += extraDLastIn[i];
      }
      dAct = dPrev;
    }

    let totalSq = 0;
    for (let l = 0; l < numLayers; l++) {
      let layerSq = 0;
      for (let i = 0; i < gW[l].length; i++) layerSq += gW[l][i] * gW[l][i];
      this.layers[l].lastGradNorm = Math.sqrt(layerSq);
      totalSq += layerSq;
      for (let i = 0; i < gB[l].length; i++) totalSq += gB[l][i] * gB[l][i];
    }
    for (let i = 0; i < gSkip.length; i++) totalSq += gSkip[i] * gSkip[i];

    const gradNorm = Math.sqrt(totalSq);
    const clip = gradNorm > MAX_GRAD_NORM ? MAX_GRAD_NORM / gradNorm : 1;

    for (let l = 0; l < numLayers; l++) {
      const layer = this.layers[l];
      for (let i = 0; i < layer.w.length; i++) {
        adamUpdate(layer.w, layer.mW, layer.vW, i, gW[l][i] * clip, lr, bc1, bc2, WEIGHT_DECAY);
      }
      for (let i = 0; i < layer.b.length; i++) {
        adamUpdate(layer.b, layer.mB, layer.vB, i, gB[l][i] * clip, lr, bc1, bc2);
      }
    }
    if (hasSkip) {
      for (let i = 0; i < gSkip.length; i++) {
        adamUpdate(
          this.wSkipKinematics,
          this.mWSkipKinematics,
          this.vWSkipKinematics,
          i,
          gSkip[i] * clip,
          lr,
          bc1,
          bc2,
          WEIGHT_DECAY
        );
      }
    }

    return gradNorm;
  }

  private beginAdamStep(): { bc1: number; bc2: number } {
    this.adamT += 1;
    this.totalSteps += 1;
    this.epoch = Math.floor(this.totalSteps / 250);
    return {
      bc1: 1 - Math.pow(ADAM_BETA1, this.adamT),
      bc2: 1 - Math.pow(ADAM_BETA2, this.adamT),
    };
  }

  /**
   * Supervised MSE step on the 8 kinematic outputs. Used for gradient/optimiser verification;
   * the painter itself never supplies targets.
   */
  public trainStep(
    input: Float32Array | number[],
    target: Float32Array | number[],
    lr = this.learningRate
  ): { loss: number; gradNorm: number } {
    const { inputs, preActivations, out } = this.forwardWithCache(input);
    const { bc1, bc2 } = this.beginAdamStep();

    const dOut = new Float32Array(out.length);
    let loss = 0;
    for (let m = 0; m < out.length; m++) {
      const diff = out[m] - target[m];
      loss += 0.5 * diff * diff;
      dOut[m] = diff / out.length;
    }
    loss /= out.length;

    const gradNorm = this.backwardTrunk(inputs, preActivations, dOut, null, lr, bc1, bc2);
    return { loss, gradNorm };
  }

  /**
   * One online policy-gradient update.
   * @param dKin   dLoss/d(kinematic outputs) (8)
   * @param dOff   dLoss/d(dx, dy offset outputs) (2)
   * @param dColor dLoss/d(normalised R, G, B outputs) (3), straight-through the output clamp
   */
  public trainPolicyStep(
    input: Float32Array | number[],
    pigmentInput: Float32Array | number[],
    dKin: ArrayLike<number>,
    dOff: ArrayLike<number>,
    dColor: ArrayLike<number>,
    lr = this.learningRate
  ): PolicyStepResult {
    const { inputs, preActivations } = this.forwardWithCache(input);
    const { bc1, bc2 } = this.beginAdamStep();
    const h11 = inputs[inputs.length - 1];

    // Spatial offset head (16 -> 2); its gradient also flows back into the trunk.
    const dH = new Float32Array(16);
    const gWOff = new Float32Array(32);
    const gBOff = new Float32Array(2);
    for (let o = 0; o < 2; o++) {
      let z = this.bSpatialOffset[o];
      for (let h = 0; h < 16; h++) z += this.wSpatialOffset[o * 16 + h] * h11[h];
      const t = Math.tanh(z);
      const dz = (dOff[o] ?? 0) * (1 - t * t);
      gBOff[o] = dz;
      for (let h = 0; h < 16; h++) {
        gWOff[o * 16 + h] = dz * h11[h];
        dH[h] += this.wSpatialOffset[o * 16 + h] * dz;
      }
    }
    for (let i = 0; i < 32; i++) {
      adamUpdate(
        this.wSpatialOffset,
        this.mWSpatialOffset,
        this.vWSpatialOffset,
        i,
        gWOff[i],
        lr,
        bc1,
        bc2,
        WEIGHT_DECAY
      );
    }
    for (let o = 0; o < 2; o++) {
      adamUpdate(this.bSpatialOffset, this.mBSpatialOffset, this.vBSpatialOffset, o, gBOff[o], lr, bc1, bc2);
    }

    this.trainChroma(pigmentInput, dColor, bc1, bc2);

    const gradNorm = this.backwardTrunk(inputs, preActivations, dKin, dH, lr, bc1, bc2);
    this.lastGradNorm = gradNorm;
    return { gradNorm };
  }

  private trainChroma(
    pigmentInput: ArrayLike<number>,
    dColor: ArrayLike<number>,
    bc1: number,
    bc2: number
  ): void {
    const lr = this.chromaLearningRate;
    for (let c = 0; c < 3; c++) {
      const chIn = this.chromaInput(pigmentInput, c);
      const dOut = dColor[c] ?? 0;

      const z1 = new Float32Array(8);
      for (let h = 0; h < 8; h++) {
        let z = this.bChroma1[c * 8 + h];
        for (let i = 0; i < 4; i++) z += this.wChroma1[c * 32 + h * 4 + i] * chIn[i];
        z1[h] = z;
      }

      adamUpdate(this.bChroma2, this.mBChroma2, this.vBChroma2, c, dOut, lr, bc1, bc2);
      for (let i = 0; i < 4; i++) {
        adamUpdate(
          this.wChromaSkip,
          this.mWChromaSkip,
          this.vWChromaSkip,
          c * 4 + i,
          dOut * chIn[i],
          lr,
          bc1,
          bc2,
          WEIGHT_DECAY
        );
      }
      for (let h = 0; h < 8; h++) {
        const idx = c * 8 + h;
        const a1 = this.leakyRelu(z1[h]);
        const dz1 = dOut * this.wChroma2[idx] * this.dLeakyRelu(z1[h]);
        adamUpdate(this.wChroma2, this.mWChroma2, this.vWChroma2, idx, dOut * a1, lr, bc1, bc2, WEIGHT_DECAY);
        adamUpdate(this.bChroma1, this.mBChroma1, this.vBChroma1, idx, dz1, lr, bc1, bc2);
        for (let i = 0; i < 4; i++) {
          adamUpdate(
            this.wChroma1,
            this.mWChroma1,
            this.vWChroma1,
            c * 32 + h * 4 + i,
            dz1 * chIn[i],
            lr,
            bc1,
            bc2,
            WEIGHT_DECAY
          );
        }
      }
    }
  }

  public getTelemetry(): NeuralTelemetry {
    const meanAbs = (arr: Float32Array) => {
      let s = 0;
      for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
      return s / Math.max(1, arr.length);
    };

    const maxLayerGrad = Math.max(0, ...this.layers.map((l) => l.lastGradNorm));

    const layerSummaries: LayerSummary[] = this.layers.map((layer) => ({
      index: layer.index,
      name: `L${layer.index}`,
      inDim: layer.inDim,
      outDim: layer.outDim,
      residual: layer.residual,
      meanAbsWeight: meanAbs(layer.w),
      gradNorm: layer.lastGradNorm,
      relativeGradPct: maxLayerGrad > 0 ? (layer.lastGradNorm / maxLayerGrad) * 100 : 0,
    }));

    const snapshot: number[] = [];
    for (let l = 0; l < this.layers.length; l++) {
      const w = this.layers[l].w;
      snapshot.push(w[0] || 0);
      snapshot.push(w[Math.floor(w.length / 2)] || 0);
    }

    let totalParameters =
      this.wSkipKinematics.length +
      this.wChroma1.length +
      this.bChroma1.length +
      this.wChroma2.length +
      this.wChromaSkip.length +
      this.bChroma2.length +
      this.wSpatialOffset.length +
      this.bSpatialOffset.length;
    for (const layer of this.layers) totalParameters += layer.w.length + layer.b.length;

    // Exact L2 norms of Adam's moment vectors over every trainable parameter.
    const firstMoments: Float32Array[] = [
      this.mWSkipKinematics,
      this.mWChroma1,
      this.mBChroma1,
      this.mWChroma2,
      this.mWChromaSkip,
      this.mBChroma2,
      this.mWSpatialOffset,
      this.mBSpatialOffset,
    ];
    const secondMoments: Float32Array[] = [
      this.vWSkipKinematics,
      this.vWChroma1,
      this.vBChroma1,
      this.vWChroma2,
      this.vWChromaSkip,
      this.vBChroma2,
      this.vWSpatialOffset,
      this.vBSpatialOffset,
    ];
    for (const layer of this.layers) {
      firstMoments.push(layer.mW, layer.mB);
      secondMoments.push(layer.vW, layer.vB);
    }
    const l2 = (arrs: Float32Array[]) => {
      let s = 0;
      for (const a of arrs) for (let i = 0; i < a.length; i++) s += a[i] * a[i];
      return Math.sqrt(s);
    };

    return {
      numLayers: this.layers.length,
      totalParameters,
      architectureLabel: '12-Layer Deep Residual MLP + Per-Channel Chromatic Net',
      epoch: this.epoch,
      totalSteps: this.totalSteps,
      loss: this.lastLoss,
      colorLoss: this.lastColorLoss,
      gradientNorm: this.lastGradNorm,
      adamFirstMomentNorm: l2(firstMoments),
      adamSecondMomentNorm: l2(secondMoments),
      learningRate: this.learningRate,
      layerSummaries,
      weightSnapshot: snapshot,
      lossHistory: [...this.lossHistory],
      lossSteps: [...this.lossSteps],
    };
  }
}

/**
 * Exact mean absolute RGB error between target and canvas, in 0..1: the mean of
 * |target - canvas| / 255 over the R, G and B channels of every sampled pixel. This is the same
 * objective the policy is rewarded on. `sampleStride` is a pixel stride; 1 (default) is exact.
 */
export function computeCanvasL1Loss(
  targetRgba: Uint8ClampedArray | Uint8Array,
  canvasRgba: Uint8ClampedArray | Uint8Array,
  sampleStride = 1
): number {
  let totalDiff = 0;
  let count = 0;
  const len = Math.min(targetRgba.length, canvasRgba.length);
  const step = Math.max(1, Math.round(sampleStride)) * 4;
  for (let i = 0; i + 2 < len; i += step) {
    totalDiff +=
      (Math.abs(targetRgba[i] - canvasRgba[i]) +
        Math.abs(targetRgba[i + 1] - canvasRgba[i + 1]) +
        Math.abs(targetRgba[i + 2] - canvasRgba[i + 2])) /
      (3 * 255);
    count++;
  }
  return count > 0 ? totalDiff / count : 0;
}

export interface OnlineStepExecutionParams {
  policy: NeuralStrokePolicy;
  targetRgba: Uint8ClampedArray | Uint8Array;
  workingRgba: Uint8ClampedArray | Uint8Array;
  canvasWidth: number;
  canvasHeight: number;
  analysisWidth: number;
  analysisHeight: number;
  activeBrushSize: BrushSize;
  strokeId: number;
  fullResTargetRgba?: Uint8ClampedArray | Uint8Array;
  fullResWorkingRgba?: Uint8ClampedArray | Uint8Array;
  skipGlobalLossScan?: boolean;
}

export interface OnlineStepExecutionResult {
  stroke: PlannedStroke;
  /** Exact canvas L1 after this stroke (only re-measured when `skipGlobalLossScan` is false). */
  globalLoss: number;
  /** Mean-L1 error removed from the canvas by this stroke (negative if it made things worse). */
  rewardDelta: number;
  gradNorm: number;
  candidateCoords: { x: number; y: number };
  neuralActionVector: {
    cx: number;
    cy: number;
    dxOffsetPx: number;
    dyOffsetPx: number;
    angleDeg: number;
    lengthScale: number;
    /** Actual painted stroke length in canvas pixels. */
    lengthPx: number;
    pressureCurve: [number, number, number];
    curvature: [number, number];
    rgb: [number, number, number];
    targetRgb: [number, number, number];
  };
}

interface PaintBuffer {
  target: Uint8ClampedArray | Uint8Array;
  working: Uint8ClampedArray | Uint8Array;
  w: number;
  h: number;
  /** Buffer pixels per canvas pixel. */
  scale: number;
}

interface StrokeGeometry {
  points: StrokePoint[];
  angle: number;
  lengthMult: number;
  pressure: [number, number, number];
  curvature: [number, number];
  cx: number;
  cy: number;
  dxOffsetPx: number;
  dyOffsetPx: number;
}

const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function strokeAlphaForPass(passIndex: number): number {
  return passIndex <= 1 ? 1.0 : passIndex <= 4 ? 0.94 : 0.98;
}

/**
 * Builds the 10-D spatial tensor and 10-D chromatic tensor the network observes at a fixation.
 * It only reads pixels (target, canvas, their difference) inside the local receptive field.
 */
function observe(
  buf: PaintBuffer,
  fx: number,
  fy: number,
  brushSize: number,
  passScale: number
) {
  const phys = getPhysicalBrushRadius(brushSize, 0.8) * buf.scale;
  const bx = clampInt(fx * buf.scale, 0, buf.w - 1);
  const by = clampInt(fy * buf.scale, 0, buf.h - 1);
  const rr = Math.max(0.5, phys * 0.6);
  const step = Math.max(1, Math.round(phys * 0.9));

  const at = (data: Uint8ClampedArray | Uint8Array, x: number, y: number, ch: number) =>
    data[(clampInt(y, 0, buf.h - 1) * buf.w + clampInt(x, 0, buf.w - 1)) * 4 + ch];
  const lum = (data: Uint8ClampedArray | Uint8Array, x: number, y: number) =>
    (0.299 * at(data, x, y, 0) + 0.587 * at(data, x, y, 1) + 0.114 * at(data, x, y, 2)) / 255;

  const tAvg = [0, 0, 0];
  const cAvg = [0, 0, 0];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const sx = bx + dx * rr;
      const sy = by + dy * rr;
      for (let ch = 0; ch < 3; ch++) {
        tAvg[ch] += at(buf.target, sx, sy, ch) / 9 / 255;
        cAvg[ch] += at(buf.working, sx, sy, ch) / 9 / 255;
      }
    }
  }

  const diff = [tAvg[0] - cAvg[0], tAvg[1] - cAvg[1], tAvg[2] - cAvg[2]];
  const localError = Math.sqrt(0.3 * diff[0] ** 2 + 0.59 * diff[1] ** 2 + 0.11 * diff[2] ** 2);

  const tL = lum(buf.target, bx - step, by);
  const tR = lum(buf.target, bx + step, by);
  const tU = lum(buf.target, bx, by - step);
  const tD = lum(buf.target, bx, by + step);
  const cL = lum(buf.working, bx - step, by);
  const cR = lum(buf.working, bx + step, by);
  const cU = lum(buf.working, bx, by - step);
  const cD = lum(buf.working, bx, by + step);

  const nnInput = [
    clamp((tR - tL) * 3.0, -1, 1),
    clamp((tD - tU) * 3.0, -1, 1),
    clamp((tR - cR - (tL - cL)) * 2.5, -1, 1),
    clamp((tD - cD - (tU - cU)) * 2.5, -1, 1),
    (bx / Math.max(1, buf.w - 1)) * 2 - 1,
    clamp(diff[0] * 2, -1, 1),
    clamp(diff[1] * 2, -1, 1),
    clamp(diff[2] * 2, -1, 1),
    localError * 2 - 1,
    passScale,
  ];

  const pigmentInput = [
    tAvg[0] * 2 - 1,
    tAvg[1] * 2 - 1,
    tAvg[2] * 2 - 1,
    cAvg[0] * 2 - 1,
    cAvg[1] * 2 - 1,
    cAvg[2] * 2 - 1,
    clamp(diff[0], -1, 1),
    clamp(diff[1], -1, 1),
    clamp(diff[2], -1, 1),
    passScale,
  ];

  return { nnInput, pigmentInput, targetRgb: tAvg.map((v) => v * 255) as [number, number, number], localError };
}

interface StrokeContext {
  passIndex: number;
  brushSize: number;
  canvasWidth: number;
  canvasHeight: number;
  searchRad: number;
}

/**
 * Pure decoding of network outputs into a stroke. There is no clipping to image content,
 * no rollout and no override: the stroke is exactly what the action vector says.
 */
function decodeStroke(
  kin: ArrayLike<number>,
  offX: number,
  offY: number,
  anchorX: number,
  anchorY: number,
  ctx: StrokeContext
): StrokeGeometry {
  const { passIndex, brushSize, canvasWidth, canvasHeight, searchRad } = ctx;

  const angle = Math.atan2(kin[1], kin[0]);
  const dirX = Math.cos(angle);
  const dirY = Math.sin(angle);

  const lengthMult = 0.1 + 0.8 * (kin[2] + 1);
  const pAttack = clamp(0.52 + (kin[3] + 1) * 0.24, 0.28, 1.0);
  const pPeak = clamp(0.78 + (kin[4] + 1) * 0.14, 0.45, 1.0);
  const pRelease = clamp(0.46 + (kin[5] + 1) * 0.26, 0.25, 1.0);

  const dxOffsetPx = Math.round(offX * searchRad);
  const dyOffsetPx = Math.round(offY * searchRad);
  const cx = clamp(anchorX + dxOffsetPx, 0, canvasWidth - 1);
  const cy = clamp(anchorY + dyOffsetPx, 0, canvasHeight - 1);

  const halfLen = Math.max(0.5, brushSize * lengthMult * 0.45);
  const curveScale = passIndex >= 4 ? 0.025 : 0.055;
  const normX = -dirY;
  const normY = dirX;
  const deflect1 = kin[6] * halfLen * 2 * curveScale;
  const deflect2 = kin[7] * halfLen * 2 * curveScale;

  const p0x = cx - dirX * halfLen;
  const p0y = cy - dirY * halfLen;
  const p3x = cx + dirX * halfLen;
  const p3y = cy + dirY * halfLen;
  const c1x = cx - dirX * (halfLen * 0.33) + normX * deflect1;
  const c1y = cy - dirY * (halfLen * 0.33) + normY * deflect1;
  const c2x = cx + dirX * (halfLen * 0.33) + normX * deflect2;
  const c2y = cy + dirY * (halfLen * 0.33) + normY * deflect2;

  const numPoints = passIndex <= 1 ? 6 : passIndex <= 4 ? 5 : 4;
  const points: StrokePoint[] = [];
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);
    const mt = 1 - t;
    const bx = mt * mt * mt * p0x + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t * t * t * p3x;
    const by = mt * mt * mt * p0y + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t * t * t * p3y;
    const pressure =
      t < 0.5 ? pAttack + (pPeak - pAttack) * (t * 2) : pPeak + (pRelease - pPeak) * ((t - 0.5) * 2);

    points.push({
      x: clamp(bx, 0, canvasWidth - 1),
      y: clamp(by, 0, canvasHeight - 1),
      pressure: clamp(pressure, 0.28, 1.0),
    });
  }

  return {
    points,
    angle,
    lengthMult,
    pressure: [pAttack, pPeak, pRelease],
    curvature: [kin[6], kin[7]],
    cx,
    cy,
    dxOffsetPx,
    dyOffsetPx,
  };
}

interface StrokeScore {
  reward: number;
  /** Mean sign(target - painted pixel) per channel: the exact L1 gradient direction for colour. */
  signMean: [number, number, number];
}

/**
 * Reward of a hypothetical stroke: the L1 error it would remove from the canvas,
 * estimated on samples spread over the stroke footprint. Nothing is painted.
 */
function scoreStroke(
  buf: PaintBuffer,
  geo: StrokeGeometry,
  rgb: [number, number, number],
  alpha: number,
  brushSize: number
): StrokeScore {
  const pts = geo.points;
  const cum = [0];
  let radSum = 0;
  for (let i = 0; i < pts.length; i++) {
    if (i > 0) cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
    radSum += getPhysicalBrushRadius(brushSize, pts[i].pressure) * 0.96;
  }
  const pathLen = cum[cum.length - 1];
  const avgRad = radSum / pts.length;
  const nS = clampInt(Math.ceil(pathLen / Math.max(1, avgRad * 0.8)) + 1, 2, 16);

  let count = 0;
  let total = 0;
  let gainSum = 0;
  const sgn = [0, 0, 0];

  let seg = 0;
  for (let s = 0; s < nS; s++) {
    const dist = (s / (nS - 1)) * pathLen;
    while (seg < pts.length - 2 && dist > cum[seg + 1]) seg++;
    const segLen = cum[seg + 1] - cum[seg];
    const f = segLen > 1e-9 ? clamp((dist - cum[seg]) / segLen, 0, 1) : 0;
    const x = pts[seg].x + (pts[seg + 1].x - pts[seg].x) * f;
    const y = pts[seg].y + (pts[seg + 1].y - pts[seg].y) * f;
    const pressure = pts[seg].pressure + (pts[seg + 1].pressure - pts[seg].pressure) * f;
    const rad = getPhysicalBrushRadius(brushSize, pressure) * 0.96;

    const nq = rad * buf.scale >= 1.2 ? 7 : 1;
    for (let q = 0; q < nq; q++) {
      total++;
      const ang = (q - 1) * (Math.PI / 3) + s * 0.5;
      const ox = q === 0 ? 0 : Math.cos(ang) * rad * 0.65;
      const oy = q === 0 ? 0 : Math.sin(ang) * rad * 0.65;
      const px = Math.round((x + ox) * buf.scale);
      const py = Math.round((y + oy) * buf.scale);
      if (px < 0 || px >= buf.w || py < 0 || py >= buf.h) continue;

      const idx = (py * buf.w + px) * 4;
      let before = 0;
      let after = 0;
      for (let ch = 0; ch < 3; ch++) {
        const t = buf.target[idx + ch];
        const w = buf.working[idx + ch];
        const painted = w + alpha * (rgb[ch] - w);
        before += Math.abs(t - w);
        after += Math.abs(t - painted);
        sgn[ch] += Math.sign(t - painted);
      }
      gainSum += (before - after) / 3;
      count++;
    }
  }

  if (count === 0) return { reward: -1, signMean: [0, 0, 0] };

  const area = 2 * avgRad * (pathLen + 2 * avgRad) * (count / total);
  return {
    reward: ((gainSum / count) / 255) * (area / 100),
    signMean: [sgn[0] / count, sgn[1] / count, sgn[2] / count],
  };
}

/**
 * Paints a stroke into a working buffer (capsule rasterisation). When `target` is provided,
 * returns the exact mean-L1 error removed, in units of "fully corrected pixels".
 */
function paintStroke(
  working: Uint8ClampedArray | Uint8Array,
  target: Uint8ClampedArray | Uint8Array | null,
  bw: number,
  bh: number,
  scale: number,
  points: StrokePoint[],
  brushSize: number,
  rgb: [number, number, number],
  alpha: number
): number {
  let gain = 0;
  const [r, g, b] = rgb;

  for (let i = 0; i < points.length; i++) {
    const pA = points[i === 0 ? 0 : i - 1];
    const pB = points[i];
    const physRad = getPhysicalBrushRadius(brushSize, 0.5 * (pA.pressure + pB.pressure), 'brush');
    const segDist = Math.hypot(pB.x - pA.x, pB.y - pA.y);
    const subSteps = Math.max(1, Math.ceil(segDist / Math.max(0.75, physRad * 0.5)));
    const rad = Math.max(0, Math.round(physRad * 0.96 * scale));
    const radSq = rad * rad;

    for (let s = 0; s <= subSteps; s++) {
      if (i > 0 && s === 0) continue;
      const frac = s / subSteps;
      const sx = Math.round((pA.x + (pB.x - pA.x) * frac) * scale);
      const sy = Math.round((pA.y + (pB.y - pA.y) * frac) * scale);

      for (let dy = -rad; dy <= rad; dy++) {
        const ny = sy + dy;
        if (ny < 0 || ny >= bh) continue;
        for (let dx = -rad; dx <= rad; dx++) {
          if (dx * dx + dy * dy > radSq) continue;
          const nx = sx + dx;
          if (nx < 0 || nx >= bw) continue;
          const idx = (ny * bw + nx) * 4;

          const or = working[idx];
          const og = working[idx + 1];
          const ob = working[idx + 2];
          const nr = Math.round(or * (1 - alpha) + r * alpha);
          const ng = Math.round(og * (1 - alpha) + g * alpha);
          const nb = Math.round(ob * (1 - alpha) + b * alpha);

          if (target) {
            const tr = target[idx];
            const tg = target[idx + 1];
            const tb = target[idx + 2];
            gain +=
              (Math.abs(tr - or) + Math.abs(tg - og) + Math.abs(tb - ob) -
                (Math.abs(tr - nr) + Math.abs(tg - ng) + Math.abs(tb - nb))) /
              (3 * 255);
          }

          working[idx] = nr;
          working[idx + 1] = ng;
          working[idx + 2] = nb;
          working[idx + 3] = 255;
        }
      }
    }
  }

  return gain;
}

/**
 * Executes one online step of the painter (see file header) and paints the chosen stroke
 * into the working buffers.
 */
export function executeOnlineStrokeStep(
  params: OnlineStepExecutionParams
): OnlineStepExecutionResult {
  const {
    policy,
    targetRgba,
    workingRgba,
    canvasWidth,
    canvasHeight,
    analysisWidth: fw,
    analysisHeight: fh,
    activeBrushSize: brushSize,
    strokeId,
    fullResTargetRgba,
    fullResWorkingRgba,
    skipGlobalLossScan = false,
  } = params;

  const passIndex = Math.max(0, BRUSH_SIZES.indexOf(brushSize));
  const passName = getPassNameForSize(brushSize);
  const passScale = 1 - passIndex / (BRUSH_SIZES.length - 1);
  const alpha = strokeAlphaForPass(passIndex);

  const hasFullRes =
    Boolean(fullResTargetRgba && fullResTargetRgba.length === canvasWidth * canvasHeight * 4) &&
    Boolean(fullResWorkingRgba && fullResWorkingRgba.length === canvasWidth * canvasHeight * 4);

  const primary: PaintBuffer =
    hasFullRes && fullResTargetRgba && fullResWorkingRgba
      ? { target: fullResTargetRgba, working: fullResWorkingRgba, w: canvasWidth, h: canvasHeight, scale: 1 }
      : { target: targetRgba, working: workingRgba, w: fw, h: fh, scale: fw / canvasWidth };

  // 1. Where to look: learned from reward, blind to target pixels.
  const fixation = policy.attention.propose(
    () => policy.nextRandom(),
    canvasWidth,
    canvasHeight,
    passIndex
  );

  // 2. Observe the dual-tensor patch and run the network: the full 13-D action.
  const obs = observe(primary, fixation.x, fixation.y, brushSize, passScale);
  const { kinematics: kin, colorPos } = policy.forwardFullAction(obs.nnInput, obs.pigmentInput);

  const searchRad = clamp(Math.round(brushSize * 0.5), 1, 40);
  const ctx: StrokeContext = { passIndex, brushSize, canvasWidth, canvasHeight, searchRad };
  const rgb: [number, number, number] = [
    clampInt(((colorPos[0] + 1) * 0.5) * 255, 0, 255),
    clampInt(((colorPos[1] + 1) * 0.5) * 255, 0, 255),
    clampInt(((colorPos[2] + 1) * 0.5) * 255, 0, 255),
  ];

  // 3. Gaussian policy: antithetic samples around the network output, scored by real L1 reward.
  const sigma = policy.explorationSigma();
  const K = EXPLORATION_SAMPLES;
  const noiseDims = 10;
  const mean = [...Array.from(kin), colorPos[3], colorPos[4]];
  const noises: number[][] = [];
  for (let k = 0; k < K / 2; k++) {
    const eps = Array.from({ length: noiseDims }, () => policy.nextGaussian());
    noises.push(eps, eps.map((v) => -v));
  }

  const actions: number[][] = [];
  const effNoise: number[][] = [];
  const geos: StrokeGeometry[] = [];
  const scores: StrokeScore[] = [];
  for (let k = 0; k < K; k++) {
    const a = mean.map((m, d) => clamp(m + sigma * noises[k][d], -1, 1));
    actions.push(a);
    effNoise.push(a.map((v, d) => (v - mean[d]) / sigma));
    const geo = decodeStroke(a.slice(0, 8), a[8], a[9], fixation.x, fixation.y, ctx);
    geos.push(geo);
    scores.push(scoreStroke(primary, geo, rgb, alpha, brushSize));
  }

  let best = 0;
  for (let k = 1; k < K; k++) if (scores[k].reward > scores[best].reward) best = k;

  // 4. Policy-gradient signal: reward advantages weight the exploration noise.
  const meanR = scores.reduce((s, x) => s + x.reward, 0) / K;
  const varR = scores.reduce((s, x) => s + (x.reward - meanR) ** 2, 0) / K;
  const stdR = Math.sqrt(varR);
  const grad = new Array<number>(noiseDims).fill(0);
  if (stdR > 1e-9) {
    for (let k = 0; k < K; k++) {
      const adv = (scores[k].reward - meanR) / (stdR + 1e-8);
      for (let d = 0; d < noiseDims; d++) grad[d] -= (adv * effNoise[k][d]) / (K * sigma);
    }
  }

  // Exact L1 gradient for colour: dL/dc = -mean sign(target - painted pixel), straight-through.
  const dColor = best >= 0 ? scores[best].signMean.map((s) => -s * alpha) : [0, 0, 0];

  const step = policy.trainPolicyStep(
    obs.nnInput,
    obs.pigmentInput,
    grad.slice(0, 8),
    grad.slice(8, 10),
    dColor
  );

  // 5. Paint the chosen stroke.
  const geo = geos[best];
  const gain = paintStroke(
    primary.working,
    primary.target,
    primary.w,
    primary.h,
    primary.scale,
    geo.points,
    brushSize,
    rgb,
    alpha
  );
  if (primary.working !== workingRgba) {
    paintStroke(workingRgba, null, fw, fh, fw / canvasWidth, geo.points, brushSize, rgb, alpha);
  }

  const rewardDelta = gain / (primary.w * primary.h);
  policy.attention.record(fixation.cell, gain);

  policy.lastColorLoss =
    (Math.abs(rgb[0] - obs.targetRgb[0]) +
      Math.abs(rgb[1] - obs.targetRgb[1]) +
      Math.abs(rgb[2] - obs.targetRgb[2])) /
    (3 * 255);

  let lengthPx = 0;
  for (let i = 1; i < geo.points.length; i++) {
    lengthPx += Math.hypot(
      geo.points[i].x - geo.points[i - 1].x,
      geo.points[i].y - geo.points[i - 1].y
    );
  }

  const hexColor = rgbToHex(rgb[0], rgb[1], rgb[2]);
  const deg = Math.round(((geo.angle * 180) / Math.PI + 360) % 360);
  const reasoning = `πθ(s_t) → (x=${geo.cx}, y=${geo.cy}) · θ=${deg}° · RGB(${rgb[0]},${rgb[1]},${rgb[2]}) · ΔL1=${formatSignedValue(
    -rewardDelta
  )}`;

  const stroke: PlannedStroke = {
    id: strokeId,
    passIndex,
    passName,
    brushSize,
    tool: 'brush',
    r: rgb[0],
    g: rgb[1],
    b: rgb[2],
    hexColor,
    opacity: alpha,
    points: geo.points,
    tangentAngle: geo.angle,
    coherence: Math.min(1, obs.localError * 2),
    importance: obs.localError,
    reasoning,
  };

  if (!skipGlobalLossScan) {
    policy.recordLoss(
      hasFullRes && fullResTargetRgba && fullResWorkingRgba
        ? computeCanvasL1Loss(fullResTargetRgba, fullResWorkingRgba)
        : computeCanvasL1Loss(targetRgba, workingRgba),
      policy.totalSteps
    );
  }

  return {
    stroke,
    globalLoss: policy.lastLoss,
    rewardDelta,
    gradNorm: step.gradNorm,
    candidateCoords: { x: geo.cx, y: geo.cy },
    neuralActionVector: {
      cx: geo.cx,
      cy: geo.cy,
      dxOffsetPx: geo.dxOffsetPx,
      dyOffsetPx: geo.dyOffsetPx,
      angleDeg: deg,
      lengthScale: geo.lengthMult,
      lengthPx,
      pressureCurve: [geo.pressure[0], geo.pressure[1], geo.pressure[2]],
      curvature: [geo.curvature[0], geo.curvature[1]],
      rgb,
      targetRgb: obs.targetRgb.map((v) => Math.round(v)) as [number, number, number],
    },
  };
}
