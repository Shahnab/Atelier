import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BRUSH_SIZES,
  FixationAttention,
  NeuralStrokePolicy,
  computeCanvasL1Loss,
  executeOnlineStrokeStep,
  extractDualTensorObservation,
  formatSignedValue,
  getPassNameForSize,
  rgbToHex,
} from './neuralPainter';

function meanAbsChannelL1(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    s += (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / (3 * 255);
    n++;
  }
  return s / n;
}

function makeScene(cw: number, ch: number, colorAt: (x: number, y: number) => [number, number, number]) {
  const target = new Uint8ClampedArray(cw * ch * 4);
  const working = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = (y * cw + x) * 4;
      const [r, g, b] = colorAt(x, y);
      target.set([r, g, b, 255], i);
      working.set([255, 255, 255, 254], i);
    }
  }
  return { target, working };
}

test('getPassNameForSize maps all 10 multi-scale brush sizes (125px down to 1px) to neural hierarchy stages', () => {
  assert.deepEqual(BRUSH_SIZES, [125, 69, 38, 21, 12, 7, 4, 3, 2, 1]);
  assert.equal(getPassNameForSize(125), 'Coarse Receptive Field');
  assert.equal(getPassNameForSize(38), 'Mid-Scale Tensor');
  assert.equal(getPassNameForSize(12), 'Fine Residual Field');
  assert.equal(getPassNameForSize(1), 'Micro-Pixel Convergence');
  assert.equal(rgbToHex(66, 55, 45), '#42372d');
});

test('extractDualTensorObservation extracts target, canvas, and residual difference tensors accurately', () => {
  const fw = 8;
  const fh = 8;
  const targetRgba = new Uint8ClampedArray(fw * fh * 4);
  const canvasRgba = new Uint8ClampedArray(fw * fh * 4);
  const idx = (4 * fw + 4) * 4;
  targetRgba.set([200, 100, 50, 255], idx);
  canvasRgba.set([50, 100, 200, 255], idx);

  const obs = extractDualTensorObservation(targetRgba, canvasRgba, fw, fh, 4, 4, 0.5);
  assert.ok(Math.abs(obs.targetR - 200 / 255) < 1e-4);
  assert.ok(Math.abs(obs.canvasB - 200 / 255) < 1e-4);
  assert.ok(obs.diffR > 0.5);
  assert.ok(obs.diffB < -0.5);
  assert.ok(obs.localError > 0.3);
});

test('network is a 12-layer residual MLP with 10,401 parameters, drawn from N(0, σ²) at Epoch 0 with a random seed', () => {
  const policy = new NeuralStrokePolicy(42);
  assert.equal(policy.layers.length, 12);

  const tel = policy.getTelemetry();
  assert.equal(tel.epoch, 0);
  assert.equal(tel.totalSteps, 0);
  assert.equal(tel.totalParameters, 10401);
  assert.equal(tel.adamFirstMomentNorm, 0);
  assert.equal(tel.adamSecondMomentNorm, 0);

  // Weights are Gaussian: zero mean, std matching the Glorot/He scale, with real tails.
  for (const layer of policy.layers) {
    assert.ok(layer.b.every((v) => v === 0), 'biases start at zero');
    const n = layer.w.length;
    const mean = layer.w.reduce((s, v) => s + v, 0) / n;
    const std = Math.sqrt(layer.w.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
    const expected = Math.sqrt(2 / (layer.inDim + layer.outDim)) * (layer.residual ? 0.35 : 0.65);
    assert.ok(Math.abs(mean) < 0.2 * expected, `layer ${layer.index} mean ${mean}`);
    assert.ok(Math.abs(std / expected - 1) < 0.15, `layer ${layer.index} std ${std} vs ${expected}`);
    assert.ok(layer.w.some((v) => Math.abs(v) > 2 * expected), 'Gaussian tails present');
  }

  const a = new NeuralStrokePolicy(1).layers[0].w[0];
  const b = new NeuralStrokePolicy(2).layers[0].w[0];
  assert.notEqual(a, b, 'different seeds give different weights');

  const r1 = new NeuralStrokePolicy().layers[0].w[0];
  const r2 = new NeuralStrokePolicy().layers[0].w[0];
  assert.notEqual(r1, r2, 'default construction draws a fresh random seed');
});

test('backpropagation through all 12 layers matches finite-difference gradients', () => {
  const input = [0.2, -0.3, 0.4, 0.1, -0.2, 0.8, 0.75, 0.0, 1.0, 0.5];
  const target = [0.4, -0.3, 0.6, -0.2, 0.8, -0.4, 0.1, 0.7];
  const loss = (p: NeuralStrokePolicy) => {
    const out = p.forward(input);
    let s = 0;
    for (let m = 0; m < out.length; m++) s += 0.5 * (out[m] - target[m]) ** 2;
    return s / out.length;
  };

  // The first AdamW step is lr * g / (|g| + eps), so its sign is the sign of the analytic gradient.
  const reference = new NeuralStrokePolicy(5);
  const trained = new NeuralStrokePolicy(5);
  trained.trainStep(input, target, 0.01);

  const h = 0.02;
  let checked = 0;
  const checkedLayers = new Set<number>();
  for (let l = 0; l < reference.layers.length; l++) {
    const w = reference.layers[l].w;
    for (let i = 0; i < w.length; i += 3) {
      const orig = w[i];
      w[i] = orig + h;
      const up = loss(reference);
      w[i] = orig - h;
      const down = loss(reference);
      w[i] = orig;
      const numeric = (up - down) / (2 * h);
      if (Math.abs(numeric) < 5e-5) continue;
      const delta = trained.layers[l].w[i] - orig;
      assert.equal(Math.sign(delta), -Math.sign(numeric), `layer ${l + 1} weight ${i}`);
      checked++;
      checkedLayers.add(l);
    }
  }
  assert.ok(checked >= 40, `checked ${checked} weights`);
  assert.ok(checkedLayers.has(0), 'first layer receives gradient through all 12 layers');
  assert.ok(checkedLayers.has(11), 'last layer receives gradient');
});

test('Adam training reduces loss and populates optimiser moments', () => {
  const policy = new NeuralStrokePolicy(42);
  const input = [0.2, -0.3, 0.4, 0.1, -0.2, 0.8, 0.75, 0.0, 1.0, 0.5];
  const target = [0.4, -0.3, 0.6, -0.2, 0.8, -0.4, 0.1, 0.7];

  const first = policy.trainStep(input, target, 0.015);
  let last = first;
  for (let i = 0; i < 60; i++) last = policy.trainStep(input, target, 0.015);
  assert.ok(last.loss < first.loss * 0.25, `${first.loss} -> ${last.loss}`);

  assert.equal(policy.totalSteps, 61);
  assert.ok(policy.layers[0].lastGradNorm > 0);
  assert.ok(policy.layers.every((l) => l.mW.some((v) => v !== 0) && l.vW.some((v) => v !== 0)));
  assert.ok(policy.getTelemetry().layerSummaries.length === 12);
});

test('policy step learns colour from an L1-style gradient and offsets through the trunk', () => {
  const policy = new NeuralStrokePolicy(11);
  const input = [0.1, -0.2, 0.3, 0.0, 0.2, 0.4, -0.3, 0.1, 0.5, 0.7];
  const pigment = [-0.5, 0.2, 0.7, 1.0, 1.0, 1.0, -0.5, -0.8, -0.3, 0.7];
  const targetColor = [-0.5, 0.2, 0.7];
  const targetOff = [0.6, -0.4];

  const err0 = Array.from(policy.forwardFullAction(input, pigment).colorPos.slice(0, 3)).reduce(
    (s, v, c) => s + Math.abs(v - targetColor[c]),
    0
  );

  for (let i = 0; i < 400; i++) {
    const out = policy.forwardFullAction(input, pigment).colorPos;
    const dColor = [0, 1, 2].map((c) => Math.sign(out[c] - targetColor[c]));
    const dOff = [out[3] - targetOff[0], out[4] - targetOff[1]];
    policy.trainPolicyStep(input, pigment, new Float32Array(8), dOff, dColor);
  }

  const out = policy.forwardFullAction(input, pigment).colorPos;
  const err1 = [0, 1, 2].reduce((s, c) => s + Math.abs(out[c] - targetColor[c]), 0);
  assert.ok(err1 < 0.1 && err1 < err0 * 0.5, `colour error ${err0} -> ${err1}`);
  assert.ok(Math.abs(out[3] - targetOff[0]) < 0.2 && Math.abs(out[4] - targetOff[1]) < 0.2);
});

test('fixation memory is reward-driven and resets on a new pass', () => {
  const attention = new FixationAttention();
  let s = 12345;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  attention.propose(rand, 240, 240, 0);

  // Half of the canvas cells pay off, half do not; the memory must learn that from reward alone.
  for (let round = 0; round < 3; round++) {
    for (let c = 0; c < 576; c++) attention.record(c, c % 2 === 0 ? 1 : -1);
  }

  let good = 0;
  for (let i = 0; i < 400; i++) {
    if (attention.propose(rand, 240, 240, 0).cell % 2 === 0) good++;
  }
  assert.ok(good / 400 > 0.95, `rewarded cells chosen ${good}/400 times (uniform would be ~50%)`);

  // A new pass forgets the old reward map.
  let afterReset = 0;
  for (let i = 0; i < 400; i++) {
    if (attention.propose(rand, 240, 240, 1).cell % 2 === 0) afterReset++;
  }
  assert.ok(afterReset / 400 > 0.35 && afterReset / 400 < 0.65, `after reset ${afterReset}/400`);
});

test('online painting learns from scratch: network output drives strokes, reward is the measured L1 gain', () => {
  const cw = 160;
  const ch = 160;
  const aw = 80;
  const ah = 80;
  const color = (x: number): [number, number, number] =>
    x < cw / 2 ? [180, 60, 40] : [30, 70, 160];

  const run = (seed: number, steps: number) => {
    const full = makeScene(cw, ch, color);
    const analysis = makeScene(aw, ah, (x) => color(x * 2));
    const policy = new NeuralStrokePolicy(seed);
    const strokes: string[] = [];
    let sumReward = 0;
    const beforeAll = meanAbsChannelL1(full.target, full.working);

    for (let i = 1; i <= steps; i++) {
      const before = meanAbsChannelL1(full.target, full.working);
      const res = executeOnlineStrokeStep({
        policy,
        targetRgba: analysis.target,
        workingRgba: analysis.working,
        canvasWidth: cw,
        canvasHeight: ch,
        analysisWidth: aw,
        analysisHeight: ah,
        activeBrushSize: 38,
        strokeId: i,
        fullResTargetRgba: full.target,
        fullResWorkingRgba: full.working,
      });
      const after = meanAbsChannelL1(full.target, full.working);
      // Reported reward must equal the true change in mean L1 error.
      assert.ok(Math.abs(res.rewardDelta - (before - after)) < 1e-9, `reward ${res.rewardDelta} vs ${before - after}`);
      assert.ok(res.gradNorm >= 0 && Number.isFinite(res.gradNorm));
      assert.equal(res.neuralActionVector.rgb.length, 3);
      assert.ok(res.stroke.points.length >= 4);
      strokes.push(res.stroke.points.map((p) => `${p.x.toFixed(3)},${p.y.toFixed(3)}`).join('|') + res.stroke.hexColor);
      sumReward += res.rewardDelta;
    }
    return { policy, full, strokes, beforeAll, sumReward };
  };

  const a = run(77, 250);
  const finalL1 = meanAbsChannelL1(a.full.target, a.full.working);
  assert.ok(finalL1 < a.beforeAll * 0.2, `L1 ${a.beforeAll} -> ${finalL1}`);
  assert.ok(Math.abs(a.sumReward - (a.beforeAll - finalL1)) < 1e-9);

  const tel = a.policy.getTelemetry();
  assert.equal(tel.totalSteps, 250);
  assert.equal(tel.epoch, 1);

  // Painted colours were produced by the network and match the two halves of the target.
  const left = (160 * 80 + 20) * 4;
  assert.ok(Math.abs(a.full.working[left] - 180) < 40 && Math.abs(a.full.working[left + 2] - 40) < 40);
  const right = (160 * 80 + 140) * 4;
  assert.ok(Math.abs(a.full.working[right + 2] - 160) < 40 && Math.abs(a.full.working[right] - 30) < 40);

  // Everything derives from the seed: no hidden randomness or teacher.
  const b = run(77, 250);
  assert.deepEqual(a.strokes, b.strokes);
  const c = run(78, 30);
  assert.notDeepEqual(a.strokes.slice(0, 30), c.strokes);
});

test('learning is what produces the painting: a frozen random network does far worse', () => {
  const cw = 120;
  const ch = 120;
  const target: [number, number, number] = [200, 90, 60];
  const paint = (freeze: boolean) => {
    const full = makeScene(cw, ch, () => target);
    const analysis = makeScene(60, 60, () => target);
    const policy = new NeuralStrokePolicy(9);
    if (freeze) {
      policy.learningRate = 0;
      policy.chromaLearningRate = 0;
    }
    for (let i = 1; i <= 120; i++) {
      executeOnlineStrokeStep({
        policy,
        targetRgba: analysis.target,
        workingRgba: analysis.working,
        canvasWidth: cw,
        canvasHeight: ch,
        analysisWidth: 60,
        analysisHeight: 60,
        activeBrushSize: 38,
        strokeId: i,
        fullResTargetRgba: full.target,
        fullResWorkingRgba: full.working,
      });
    }
    return meanAbsChannelL1(full.target, full.working);
  };

  const learned = paint(false);
  const frozen = paint(true);
  assert.ok(learned < frozen * 0.5, `learned ${learned} vs frozen ${frozen}`);
});

test('computeCanvasL1Loss is the exact mean absolute RGB error', () => {
  // Pixel 1 differs by 255 in R only (1/3 of full scale); pixel 2 is identical.
  const target = new Uint8ClampedArray([255, 0, 0, 255, 10, 20, 30, 255]);
  const canvas = new Uint8ClampedArray([0, 0, 0, 255, 10, 20, 30, 255]);
  assert.equal(computeCanvasL1Loss(target, canvas), (1 / 3 + 0) / 2);

  // Unequal channels: |30-0| + |0-60| + |90-90| = 90 -> 90 / (3*255) for the single pixel.
  const t2 = new Uint8ClampedArray([30, 0, 90, 255]);
  const c2 = new Uint8ClampedArray([0, 60, 90, 255]);
  assert.ok(Math.abs(computeCanvasL1Loss(t2, c2) - 90 / (3 * 255)) < 1e-15);

  assert.equal(computeCanvasL1Loss(target, target), 0);
  assert.equal(computeCanvasL1Loss(new Uint8ClampedArray(0), new Uint8ClampedArray(0)), 0);
});

test('loss history keeps the whole run at full precision, with the exact first and latest values', () => {
  const policy = new NeuralStrokePolicy(1);
  const recorded = new Map<number, number>();
  const loss = (step: number) => 0.75 * Math.exp(-step / 9000) + 0.0123456789;

  for (let step = 0; step <= 50000; step += 25) {
    const v = loss(step);
    policy.recordLoss(v, step);
    recorded.set(step, v);
  }

  const tel = policy.getTelemetry();
  assert.equal(tel.lossHistory.length, tel.lossSteps.length);
  assert.ok(tel.lossHistory.length <= 600 && tel.lossHistory.length > 100);
  assert.equal(tel.lossSteps[0], 0, 'history starts at step 0');
  assert.equal(tel.lossHistory[0], loss(0), 'first value is exact, not rounded');
  for (let i = 0; i < tel.lossSteps.length; i++) {
    assert.equal(tel.lossHistory[i], recorded.get(tel.lossSteps[i]), `value at step ${tel.lossSteps[i]} is unaltered`);
    if (i > 0) assert.ok(tel.lossSteps[i] > tel.lossSteps[i - 1]);
  }
  assert.ok(tel.lossSteps[tel.lossSteps.length - 1] > 49000, 'history reaches the end of the run');
  assert.equal(tel.loss, loss(50000), 'latest loss is exact');

  policy.resetToScratchWeights(2);
  assert.deepEqual(policy.getTelemetry().lossHistory, []);
});

test('telemetry figures are exact: Adam L2 norms, gradient shares and stroke length', () => {
  const cw = 100;
  const ch = 100;
  const full = makeScene(cw, ch, () => [120, 140, 60]);
  const analysis = makeScene(50, 50, () => [120, 140, 60]);
  const policy = new NeuralStrokePolicy(21);

  let last;
  for (let i = 1; i <= 20; i++) {
    last = executeOnlineStrokeStep({
      policy,
      targetRgba: analysis.target,
      workingRgba: analysis.working,
      canvasWidth: cw,
      canvasHeight: ch,
      analysisWidth: 50,
      analysisHeight: 50,
      activeBrushSize: 21,
      strokeId: i,
      fullResTargetRgba: full.target,
      fullResWorkingRgba: full.working,
    });
  }

  const tel = policy.getTelemetry();
  const l2 = (arrs: Float32Array[]) => Math.sqrt(arrs.reduce((s, a) => s + a.reduce((q, v) => q + v * v, 0), 0));
  const m = [
    policy.mWSkipKinematics, policy.mWChroma1, policy.mBChroma1, policy.mWChroma2, policy.mWChromaSkip,
    policy.mBChroma2, policy.mWSpatialOffset, policy.mBSpatialOffset,
    ...policy.layers.flatMap((l) => [l.mW, l.mB]),
  ];
  const v = [
    policy.vWSkipKinematics, policy.vWChroma1, policy.vBChroma1, policy.vWChroma2, policy.vWChromaSkip,
    policy.vBChroma2, policy.vWSpatialOffset, policy.vBSpatialOffset,
    ...policy.layers.flatMap((l) => [l.vW, l.vB]),
  ];
  assert.ok(Math.abs(tel.adamFirstMomentNorm - l2(m)) < 1e-9, `adamFirstMomentNorm ${tel.adamFirstMomentNorm} close to ${l2(m)}`);
  assert.ok(Math.abs(tel.adamSecondMomentNorm - l2(v)) < 1e-9, `adamSecondMomentNorm ${tel.adamSecondMomentNorm} close to ${l2(v)}`);
  assert.ok(tel.adamFirstMomentNorm > 0 && tel.adamSecondMomentNorm > 0);

  const maxGrad = Math.max(...tel.layerSummaries.map((l) => l.gradNorm));
  assert.ok(maxGrad > 0);
  for (const layer of tel.layerSummaries) {
    assert.ok(Math.abs(layer.relativeGradPct - (layer.gradNorm / maxGrad) * 100) < 1e-9);
  }
  assert.equal(Math.max(...tel.layerSummaries.map((l) => l.relativeGradPct)), 100);

  const pts = last!.stroke.points;
  let len = 0;
  for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  assert.equal(last!.neuralActionVector.lengthPx, len);
  assert.equal(tel.gradientNorm, last!.gradNorm);
  assert.equal(tel.totalSteps, 20);
});

test('formatSignedValue prints sign and tiny values without rounding them to zero', () => {
  assert.equal(formatSignedValue(0), '0');
  assert.equal(formatSignedValue(0.0123456), '+0.01235');
  assert.equal(formatSignedValue(-0.5), '−0.50000');
  assert.equal(formatSignedValue(-1.234e-7), '−1.23e-7');
});

test('analysis-grid-only stepping works and keeps global L1 telemetry', () => {
  const fw = 28;
  const fh = 28;
  const { target, working } = makeScene(fw, fh, () => [180, 60, 40]);
  const policy = new NeuralStrokePolicy(3);
  const initial = computeCanvasL1Loss(target, working, 1);

  for (let s = 1; s <= 60; s++) {
    executeOnlineStrokeStep({
      policy,
      targetRgba: target,
      workingRgba: working,
      canvasWidth: 825,
      canvasHeight: 875,
      analysisWidth: fw,
      analysisHeight: fh,
      activeBrushSize: 125,
      strokeId: s,
    });
  }
  assert.ok(policy.lastLoss < initial, `${initial} -> ${policy.lastLoss}`);
});
