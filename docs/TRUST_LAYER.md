# GPU Fabric Trust Layer: Architecture, Threat Model, and Evaluation

GPU Fabric operates across heterogeneous, non-dedicated workstations and lab PCs equipped with consumer NVIDIA GPUs (e.g., RTX 4060). Unlike data-center grade GPUs (A100, H100), consumer cards lack **ECC (Error-Correcting Code) memory** and operate in uncontrolled physical environments subject to thermal throttling, overclocking instability, and unauthorized local access.

The **Trust Layer** provides a three-pillar defense-in-depth architecture:
1. **C1: Verifiable Results** — Probabilistic $O(k \cdot n^2)$ Freivalds matrix verification and sampled kernel auditing.
2. **C2: Deception Mechanisms** — Lightweight honeypot decoy nodes, cryptographic canary tokens, and unpredictable known-answer probes.
3. **C3: Node & User Reputation** — A Bayesian Beta-reputation model with single-failure transient tolerance, hysteresis, and automated Slurm scheduler drainage.

---

## 1. Mathematical Foundation (C1)

### Freivalds' Algorithm for Matrix Multiplication
Given input matrices $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$ and the worker's reported result $C \in \mathbb{R}^{m \times p}$, full CPU recomputation requires $O(m \cdot n \cdot p)$ (or $O(n^3)$ for square matrices), which quickly bottlenecks client CPU and memory.

Freivalds' algorithm verifies $C = A \cdot B$ probabilistically in $O(k \cdot n^2)$ time:
1. In each round $i \in \{1, \dots, k\}$, sample an independent Rademacher random vector $r \in \{-1, +1\}^p$.
2. Compute on the client CPU in `float64`:
   $$\text{lhs} = C \cdot r$$
   $$\text{rhs} = A \cdot (B \cdot r)$$
3. Compute the maximum residual vector difference:
   $$\Delta = \|\text{lhs} - \text{rhs}\|_\infty$$

#### Theoretical Error Bound & Floating-Point Calibration
In exact arithmetic, if $C \neq A \cdot B$, then $\Pr[\Delta = 0] \le 2^{-k}$.

In IEEE floating-point arithmetic (Higham 2002, Theorem 3.5), forward matrix multiplication error satisfies:
$$\|fl(A \cdot B) - A \cdot B\|_\infty \le n \cdot \epsilon \cdot \|A\|_\infty \cdot \|B\|_\infty$$
When projected onto $r \in \{-1, 1\}^p$, the Central Limit Theorem introduces an orthogonal $\sqrt{p}$ concentration term. The base physical scale is defined as:
$$\text{scale} = \epsilon \cdot n \cdot \|A\|_\infty \cdot \|B\|_\infty \cdot \sqrt{p}$$
where $\epsilon = \text{eps}(A\text{.dtype})$ ($1.19 \times 10^{-7}$ for `float32`).

The verification threshold is set to:
$$\text{threshold} = \text{threshold\_factor} \times \text{scale}$$
A calibrated `threshold_factor = 10.0` provides a conservative ~10x margin above honest non-associative accumulation differences while catching structural corruption, zeros, row tampering, and random bitflips.

---

## 2. Deception Architecture (C2)

### Decoy Worker Nodes (`--decoy`)
- Decoy nodes emulate real GPU Fabric workers (e.g., presenting an RTX 4060 8GB profile with realistic jittered telemetry) without requiring physical GPUs or CuPy.
- Read-only RPCs (`GetHealth`, `GetGPUInfo`, `GetGPUStatus`, `ListGPUs`) respond with authentic specs.
- Execution RPCs (`Compute`, `Execute`, `RunBenchmark`) wait a realistic scheduling delay and abort with `grpc.StatusCode.UNAVAILABLE ("GPU busy")`.
- Every incoming RPC logs an immutable security `Alert` capturing peer IP and authenticated TLS certificate Common Name (`cert_cn`).

### Cryptographic Canary Tokens
- `CanaryRegistry` generates opaque cryptographically random tokens (`secrets.token_urlsafe(24)`).
- Stored in SQLite tripwires to detect job scraping, unauthorized result fetching, and directory traversal.
- Triggered canaries emit a high-priority `canary_hit` alert.

### Known-Answer Probes
- Client-side `probe_worker` submits lightweight, precomputed matrix multiplication problems at unpredictable intervals.
- Any mismatch emits a `probe_fail` alert and penalizes the node's reputation.

---

## 3. Node & User Reputation Model (C3)

Reputation is tracked using a **Beta-reputation system** with parameters $(\alpha, \beta)$:
$$\text{Trust Score} = \frac{\alpha}{\alpha + \beta}$$

### Event Weights
- `VERIFY_PASS`: $\alpha \mathrel{+}= 1.0$
- `VERIFY_FAIL`: $\beta \mathrel{+}= 3.0$
- `PROBE_FAIL`: $\beta \mathrel{+}= 5.0$
- `DECOY_CONTACT` / `CANARY_HIT`: $\beta \mathrel{+}= 8.0$ (penalizes the **USER identity**, not the worker node)

### Exponential Forgetting
To allow temporary issues to heal and adapt to long-term trends, past evidence decays towards the uninformative prior $(\alpha_0=2.0, \beta_0=1.0)$ with decay factor $\lambda = 0.98$:
$$\alpha \leftarrow \alpha_0 + (\alpha - \alpha_0) \cdot \lambda$$
$$\beta \leftarrow \beta_0 + (\beta - \beta_0) \cdot \lambda$$

### Hysteresis & Single-Failure Tolerance
- **Transient Fault Tolerance**: On consumer GPUs without ECC, a cosmic ray or isolated bitflip can cause an honest single failure. **A single failure marks a node as `SUSPECT` and schedules a re-run on another node, but NEVER drains the node.**
- **Automated Scheduler Drainage**: A node enters `DRAIN` state only when:
  1. $\text{score} < 0.35$
  2. $\text{fail\_count} \ge 2$
  3. $\text{total\_events} \ge 3$
- **Hysteresis Recovery**: A drained node requires sustained consecutive passes reaching $\text{score} \ge 0.70$ before transitioning back to `HEALTHY`.
- **Slurm Integration**: `SlurmActions` invokes:
  ```bash
  scontrol update NodeName=<node> State=DRAIN Reason="<reason>"
  ```
  using safe argument vectors with strict alphanumeric regex validation (never `shell=True`).

---

## 4. Honest Limitations & Threat Model Boundaries

1. **Algorithmic Scope of Freivalds**:
   Freivalds' algorithm provides an asymptotic reduction only for matrix multiplication ($O(k \cdot n^2)$ vs $O(n^3)$). For vector operations (`vector_add`, `vector_mul`, `vector_dot`, `triad`), the computation is already $O(n)$ in work and memory. For these operations, fast verification relies on random element sampling ($O(\text{sample})$) or full CPU verification.

2. **Sub-Tolerance Perturbations**:
   Corruptions with relative magnitudes smaller than the numerical tolerance bound ($\sim 10^{-5}$ for `float32`) cannot be distinguished from non-associative floating-point summation reordering without bit-exact software emulation.

3. **Decoy Visibility with mTLS**:
   Because GPU Fabric enforces mutual TLS by default, unauthenticated network port scanners are rejected during the TLS handshake before reaching application RPCs. Decoy workers therefore detect **authenticated cluster users probing nodes or resources they were not allocated**, rather than unauthenticated external scans.

4. **Out-of-Scope Adversarial Models**:
   - Colluding workers returning identical forged answers across redundant executions.
   - Malicious clients deliberately submitting poisoned input tensors.
   - Side-channel attacks (power analysis, acoustic, or GPU microarchitectural cache timing).

---

## 5. Evaluation & Reproduction Commands

### Calibrate Floating-Point Tolerance
```bash
python scripts/eval/calibrate_tolerance.py --trials 50 --sizes "128,256,512,1024"
```

### Sweep Detection Rates Across Fault Injections
```bash
python scripts/eval/eval_detection.py --trials 30 --sizes "256,512" --rounds "1,2,3,4,5,6"
```

### Measure Verification Overhead vs Dimension ($n$)
```bash
python scripts/eval/eval_overhead.py --sizes "256,512,1024,2048,3072" --repeats 5
```

### Simulate Node Reputation Timeline and Drain Trigger
```bash
python scripts/eval/eval_trust_timeline.py --steps 50
```

### Generate All Evaluation Figures
```bash
python scripts/eval/plot_figures.py
python scripts/eval/plot_trust_timeline.py
```
Generated figures are saved in both `.svg` and 300-dpi `.png` under `scripts/eval/figures/`.
