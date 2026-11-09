# Atelier — a neural network that learns to paint in your browser

A 12-layer deep residual MLP (10,401 Float32 parameters) that starts at Epoch 0 with random Gaussian
weights and learns to paint a portrait stroke by stroke, trained 100% from scratch in pure TypeScript.
No pre-trained weights, no ML libraries, no server: everything runs in the browser.

## Run locally

**Prerequisites:** Node.js 20.19+ (or 22.12+).

```
npm install
npm run dev
```

Open http://localhost:3000. No API key is needed. Other scripts: `npm test`, `npm run lint`, `npm run build`.

## How it works

Over a fixed 50,000-stroke curriculum the brush shrinks through ten sizes (125, 69, 38, 21, 12, 7, 4, 3, 2, 1 px).
For every stroke:

1. **Fixation.** A reward-driven memory picks where to look. It is non-parametric and never reads target
   pixels; it only remembers where strokes recently improved the painting.
2. **Observation.** The network sees the dual-tensor patch around the fixation: target, canvas and residual.
3. **Action.** The network outputs the full 13-D stroke:
   `[cos θ, sin θ, length, pAttack, pPeak, pRelease, curvature₁, curvature₂, R, G, B, Δx, Δy]`.
   Nothing clips, snaps or overrides these values.
4. **Exploration and reward.** The policy is Gaussian. Eight antithetic samples around the network output are
   scored by the real L1 error each would remove from the canvas, and the best one is painted.
5. **Learning.** Backpropagation + AdamW update the network:
   - geometry, pressure, curvature and offset use an evolution-strategies policy gradient from those rewards;
   - colour uses the exact analytic gradient of the L1 loss with respect to the stroke colour.

The UI shows the live L1 loss curve, per-layer weights and gradient norms, and the canvas as it is painted.

### Architecture

`10 → 32 ×9 (8 residual) → 24 → 16 → 8`, plus a per-channel chromatic sub-network and a spatial-offset head.
Weights are drawn from N(0, σ²) with Glorot/He scaling and zero biases, using a fresh random seed on every
start and on "Reset from Scratch". The source is in [src/engine/neuralPainter.ts](src/engine/neuralPainter.ts).

### Notes

- The canvas starts blank white; the Gaussian noise is in the network weights.
- The brush-size schedule is a fixed curriculum, not learned.
- Tests verify the parameter count, Gaussian initialisation, backpropagation against finite differences,
  that reported reward equals the measured L1 change, that a frozen network paints far worse than a learning
  one, and that all behaviour is reproducible from the seed alone.
