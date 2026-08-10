---
layout: blog
title: "A Graph-Native CUDA Solver for MoE Load Balancing"
author: "Bochuan Lyu"
date: 2026-08-08
last_updated: 2026-08-09
description: "Replacing dense interior-point iterations with capacity-constrained edge balancing in DeepSeek LPLB"
permalink: /lplb-edge-balance/
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

[DeepSeek LPLB](https://github.com/deepseek-ai/LPLB) formulates dynamic expert-parallel load balancing in mixture-of-experts models as a small linear program. Its current experimental solver applies five interior-point iterations using dense linear algebra on one streaming multiprocessor. I implemented an alternative CUDA solver that works directly on the sparse replica graph.

The new method repeatedly balances the workloads at the endpoints of each replica edge. A clipped edge update preserves feasibility, edge coloring exposes safe parallelism, and the entire problem for one expert-parallel group fits in one warp. On an NVIDIA RTX 4060, the implementation takes **13.5–19.2 microseconds** for the graph sizes tested and reaches a maximum objective gap below **0.0017%** against SciPy HiGHS on a 960-instance corpus.

This page describes the optimization model, the algorithmic argument, the CUDA implementation, and the current experimental results.

<aside class="article-note research-status"><strong>Research status.</strong> The edge-balance implementation and this technical article were developed by Bochuan Lyu. Bochuan Lyu and Zedong Peng are extending this work through end-to-end evaluations in inference systems. The broader algorithmic analysis and experimental results are intended for a future research paper.</aside>

## The load-balancing LP

Consider one LPLB group with GPU ranks $V$ and replica edges $E$. An edge $e=(u,v)$ means that workload associated with an expert on rank $u$ may instead be processed by its replica on rank $v$.

Let:

- $b_i$ be the initial workload at rank $i$;
- $w_e$ be the redirectable workload, or capacity, of edge $e$;
- $f_e$ be the workload transferred along edge $e$;
- $T$ be the maximum final workload across ranks.

With node-edge incidence matrix $A$, the load vector after redirection is

$$
\ell = b + Af.
$$

The continuous balancing problem is

$$
\begin{aligned}
\min_{f,T}\quad & T \\
\text{s.t.}\quad & b+Af \le T\mathbf{1},\\
& 0\le f\le w.
\end{aligned}
$$

The graph is sparse. With $G$ ranks and $D$ replica slots per rank, there are only

$$
|E|=GD
$$

edges. This suggests using the graph structure directly instead of constructing the dense standard-form matrices required by a general LP method.

## Pairwise edge balancing

Maintain a feasible edge flow $f$ and the corresponding rank loads $\ell=b+Af$. For an edge $e=(u,v)$, changing its flow by $\delta$ produces

$$
f_e \leftarrow f_e+\delta,
\qquad
\ell_u \leftarrow \ell_u-\delta,
\qquad
\ell_v \leftarrow \ell_v+\delta.
$$

Ignoring capacity bounds, the movement that equalizes the two endpoint loads is

$$
\delta_{\mathrm{equal}}=\frac{\ell_u-\ell_v}{2}.
$$

The current flow must remain in $[0,w_e]$, so the applied movement is

$$
\boxed{
\delta_e=
\operatorname{clip}\left(
\frac{\ell_u-\ell_v}{2},
-f_e,
w_e-f_e
\right).
}
$$

The negative lower bound is important: a later sweep may reverse part of an earlier decision. A one-pass, forward-only implementation would be an order-dependent greedy heuristic.

Every update preserves

$$
0\le f_e\le w_e
$$

and total workload. Consequently, every iterate represents a feasible routing decision, regardless of whether the iteration limit has been reached.

## Why the update is globally meaningful

The edge update is the exact coordinate minimizer of the convex quadratic potential

$$
Q(f)=\frac{1}{2}\lVert b+Af\rVert_2^2,
\qquad 0\le f\le w.
$$

For edge $e=(u,v)$, its one-dimensional subproblem is

$$
\min_{-f_e\le\delta\le w_e-f_e}
\frac{1}{2}\left[(\ell_u-\delta)^2+(\ell_v+\delta)^2\right],
$$

whose solution is exactly the clipped equalization step above. Repeated Gauss–Seidel sweeps are therefore cyclic coordinate descent on a convex box-constrained problem.

For this graph-redistribution model, a minimizer of $Q$ is also optimal for the original min–max LP. To see the connection, let

$$
M=\max_i\ell_i
$$

at a quadratic optimum, and let $S$ be the set of ranks whose load equals $M$. If an edge leaves $S$ for a lower-load rank, it must be saturated; otherwise a small transfer would reduce $Q$. Similarly, every edge entering $S$ from a lower-load rank must carry zero flow. Therefore,

$$
|S|M=b(S)-w(\delta^+(S)),
$$

and hence

$$
M=
\frac{b(S)-w(\delta^+(S))}{|S|}.
$$

The right-hand side is a valid cut lower bound on every feasible min–max solution. The quadratic solution attains this bound, establishing min–max optimality at convergence.

The implemented solver stops at a finite tolerance, so the CUDA output is an approximation to this limiting solution rather than an exact arithmetic certificate. I validate finite-sweep solutions independently against HiGHS.

## Parallelizing edge updates safely

Edges sharing a rank cannot update concurrently without a race. I color the underlying undirected multigraph so that each color class is a matching: no two edges of the same color share an endpoint.

The preprocessing step produces two arrays:

```text
colored_edges[E]
color_offsets[number_of_colors + 1]
```

Edges of color `c` occupy

```text
colored_edges[color_offsets[c] : color_offsets[c + 1]]
```

One sweep processes colors sequentially, while all edges of a color update in parallel. A block barrier between colors gives Gauss–Seidel semantics without atomics.

The current benchmark uses a greedy coloring computed once on the CPU and reused across workloads. Minimum edge coloring is difficult on a general graph, but optimal coloring is unnecessary for correctness. It only changes the number of synchronization rounds. Standard LPLB topologies can eventually use precomputed colorings.

### Coloring overhead

Coloring is a topology-setup cost rather than a per-batch solver cost. The current implementation performs greedy coloring in Python on the CPU, creates `colored_edges` and `color_offsets`, copies both tensors to the GPU, and synchronizes before measurement. The resulting end-to-end setup cost is:

<div class="data-table-wrap">
  <table class="data-table numeric-table">
    <thead>
      <tr>
        <th scope="col">Topology</th>
        <th scope="col">Edges</th>
        <th scope="col">Colors</th>
        <th scope="col">Median setup</th>
        <th scope="col">p99 setup</th>
      </tr>
    </thead>
    <tbody>
      <tr><th scope="row"><code>4×1</code></th><td>4</td><td>2</td><td>118 µs</td><td>358 µs</td></tr>
      <tr><th scope="row"><code>8×1</code></th><td>8</td><td>2</td><td>162 µs</td><td>470 µs</td></tr>
      <tr><th scope="row"><code>8×2</code></th><td>16</td><td>4</td><td>371 µs</td><td>739 µs</td></tr>
      <tr><th scope="row"><code>16×2</code></th><td>32</td><td>4</td><td>932 µs</td><td>1,406 µs</td></tr>
    </tbody>
  </table>
</div>

This setup is more expensive than one Edge Balance solve, which takes approximately 13.5--19.2 us for these graph sizes. However, the graph topology is fixed while workloads change from batch to batch, so coloring is computed once when the planner is initialized and reused for every solve. For example, amortizing the 932 us `16x2` setup over 10,000 batches adds only about 0.093 us per batch.

Most of the measured setup time is Python traversal, small tensor allocation and transfer, and explicit synchronization. It is not GPU solver time. If topology initialization becomes important, the setup can be reduced by caching colorings by topology, using a list- or array-based host implementation, issuing one asynchronous GPU copy, or hard-coding known colorings for standard ring, cube, hypercube, and torus graphs.

### Feasibility compared with the IPM solver

Edge Balance starts from a feasible half-capacity flow and clips every update to the interval permitted by the edge bounds. As a result, every iterate is feasible, even when the sweep limit is reached before the convergence tolerance. All 960 Edge Balance outputs in the benchmark were independently feasible.

The existing five-iteration IPM does not provide the same guarantee. Independent reconstruction of each LP and validation against HiGHS found the following infeasible counts:

<div class="data-table-wrap">
  <table class="data-table feasibility-table">
    <thead>
      <tr>
        <th scope="col">Topology</th>
        <th scope="col">IPM infeasible</th>
        <th scope="col">IPM feasible rate</th>
        <th scope="col">Edge Balance infeasible</th>
      </tr>
    </thead>
    <tbody>
      <tr><th scope="row"><code>4×1</code></th><td>1 / 192</td><td><span class="rate-bar"><span style="width: 99.48%;"></span></span><span class="rate-value good">99.48%</span></td><td class="good">0 / 192</td></tr>
      <tr><th scope="row"><code>4×2</code></th><td>0 / 192</td><td><span class="rate-bar"><span style="width: 100%;"></span></span><span class="rate-value good">100%</span></td><td class="good">0 / 192</td></tr>
      <tr><th scope="row"><code>8×1</code></th><td>0 / 192</td><td><span class="rate-bar"><span style="width: 100%;"></span></span><span class="rate-value good">100%</span></td><td class="good">0 / 192</td></tr>
      <tr><th scope="row"><code>8×2</code></th><td class="warn">45 / 192</td><td><span class="rate-bar"><span class="warn-bar" style="width: 76.56%;"></span></span><span class="rate-value warn">76.56%</span></td><td class="good">0 / 192</td></tr>
      <tr><th scope="row"><code>16×2</code></th><td class="bad">112 / 192</td><td><span class="rate-bar"><span class="bad-bar" style="width: 41.67%;"></span></span><span class="rate-value bad">41.67%</span></td><td class="good">0 / 192</td></tr>
      <tr class="summary-row"><th scope="row">Overall</th><td>158 / 960</td><td><span class="rate-bar"><span class="warn-bar" style="width: 83.54%;"></span></span><span class="rate-value warn">83.54%</span></td><td class="good">0 / 960</td></tr>
    </tbody>
  </table>
</div>

The single `4x1` failure exceeded the strict bound tolerance by only about $1.7\times10^{-6}$ and is best interpreted as floating-point noise. The larger `8x2` and `16x2` failures are substantive: some IPM routing probabilities lie outside $[0,1]$, implying negative flow on the paired replica variable. The IPM's internal success flag also accepted some of these independently infeasible outputs. Edge Balance avoids this failure mode because feasibility is an invariant of the update rule rather than a post-hoc convergence check.

## CUDA organization

The implementation launches

```text
grid size  = number of independent LPLB groups
block size = 32 threads
```

One warp solves one graph group. The full state resides in dynamic shared memory:

```cpp
struct edge_balance_smem {
    float load[GROUP_SIZE];
    float flow[GROUP_SIZE * DUP_PER_RANK];
    float capacity[GROUP_SIZE * DUP_PER_RANK];
    float scale;
    float max_change;
    int completed_sweeps;
    int converged;
};
```

Storage is $O(G+E)$ and no dense LP matrices are formed.

The kernel has five main phases:

1. **Scale the workload.** A warp reduction calculates `max(max(workload), 1)`. Loads and capacities are normalized by this value to improve numerical consistency.
2. **Initialize feasible flows.** The current cold start uses half of each edge capacity, matching LPLB's half-flow fallback.
3. **Construct node loads.** One lane owns each node and scans incident edges, avoiding atomics.
4. **Perform colored coordinate sweeps.** Matching edges update concurrently; colors are separated by block barriers.
5. **Write routing fractions and diagnostics.** The output includes final loads, maximum load, sweep count, and convergence status.

The central kernel logic is:

```cpp
for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    float thread_max_change = 0.0f;

    for (int color = 0; color < num_colors; ++color) {
        for (int p = color_offsets[color] + lane;
             p < color_offsets[color + 1];
             p += 32) {

            int e = colored_edges[p];
            int u = source[e];
            int v = destination[e];

            float delta = 0.5f * (load[u] - load[v]);
            delta = clamp(delta, -flow[e], capacity[e] - flow[e]);

            flow[e] += delta;
            load[u] -= delta;
            load[v] += delta;

            thread_max_change = max(thread_max_change, abs(delta));
        }

        __syncthreads();
    }

    float max_change = warp_reduce_max(thread_max_change);
    if (max_change <= tolerance)
        break;
}
```

The production call remains asynchronous on PyTorch's current CUDA stream.

## Integration with LPLB

The implementation currently adds a separate experimental API; it does **not** replace the solver inside `Planner.run()`. The existing production path still counts workload, calls the dense IPM through `solve_probs()`, and passes its weights to `weighted_select_target()` to obtain physical expert indices.

The benchmark path calls `edge_balance_probs()` directly. It supplies saved workload tensors and precomputed edge colors, then receives routing fractions and diagnostic tensors for timing and HiGHS validation. Those results are benchmark outputs: they are not fed back to the workload input, and the experimental method is not yet connected to `weighted_select_target()`.

<figure style="margin: 2rem 0;">
  <img src="/assets/lplb-integration.svg" alt="Flow diagram showing the LPLB planner dispatching to either the existing dense IPM kernel or the graph-native edge-balance CUDA kernel." style="width: 100%; height: auto;">
  <figcaption style="margin-top: .65rem; color: #64748b; font-size: .9em;">The blue lane is the unchanged production path. The green lane is the standalone benchmark path used for the results in this post; it is not yet wired into <code>Planner.run()</code>.</figcaption>
</figure>

Moving the experiment into the production path would require three additional steps:

1. **Prepare and cache the topology.** Convert the replica map into graph endpoints and edge colors during planner initialization.
2. **Select the backend in `Planner.run()`.** Dispatch either `solve_probs()` or `edge_balance_probs()` on the current CUDA stream.
3. **Connect the output contract.** Pass the edge-balance routing fractions into the existing target-selection stage and verify equivalent end-to-end behavior.

The implementation is split across the following files:

<div style="overflow-x: auto; margin: 1.25rem 0 1.5rem; border: 1px solid #e2e8f0; border-radius: 10px;">
  <table style="width: 100%; min-width: 580px; margin: 0; border-collapse: collapse;">
    <thead>
      <tr style="background: #f1f5f9; color: #334155;">
        <th scope="col" style="padding: .75rem; text-align: left;">Layer</th>
        <th scope="col" style="padding: .75rem; text-align: left;">Location</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="padding: .7rem .75rem;">CUDA kernel</td><td style="padding: .7rem .75rem;"><code>lplb/resources/csrc-tmpl/minilp.cu</code></td></tr>
      <tr style="background: #f8fafc;"><td style="padding: .7rem .75rem;">CUDA launch and PyBind binding</td><td style="padding: .7rem .75rem;"><code>csrc/plugin.cpp</code></td></tr>
      <tr><td style="padding: .7rem .75rem;">Python planner interface</td><td style="padding: .7rem .75rem;"><code>lplb/planner.py</code></td></tr>
      <tr style="background: #f8fafc;"><td style="padding: .7rem .75rem;">Coloring and benchmark dispatch</td><td style="padding: .7rem .75rem;"><code>scripts/benchmark_solver.py</code></td></tr>
      <tr><td style="padding: .7rem .75rem;">HiGHS comparison</td><td style="padding: .7rem .75rem;"><code>scripts/compare_solver_results.py</code></td></tr>
    </tbody>
  </table>
</div>

The experimental Python entry point is

```python
Planner.edge_balance_probs(...)
```

For a graph with `G` ranks and `D` replica slots, edge `destination * D + replica_slot` has endpoints

```text
source      = r2o[destination, replica_slot]
destination = edge / D
```

The solver returns the fraction retained at the source endpoint:

```text
retained_fraction = 1 - transferred_flow / edge_capacity
```

For a zero-capacity edge, it returns `0.5`; either fraction corresponds to zero workload.

## Experimental results

I evaluated the solver on a saved corpus of 960 instances. Every result was reconstructed and compared offline with the same LP solved by SciPy HiGHS. HiGHS verification, corpus loading, and serialization were outside the timed region. Solver latency was measured with CUDA events over repeated calls.

Hardware: **NVIDIA GeForce RTX 4060**. The results below should be viewed as single-GPU development measurements rather than claims about H100/H800 production performance.

<figure style="margin: 2rem 0;">
  <img src="/assets/lplb-results.svg" alt="Horizontal bar chart comparing current LPLB and edge-balance latency across five graph shapes. Edge balance remains near 14 to 19 microseconds while current LPLB grows to 104 microseconds." style="width: 100%; height: auto;">
  <figcaption style="margin-top: .65rem; color: #64748b; font-size: .9em;">Edge-balance latency remains in a narrow band as the tested graphs grow, while the dense solver cost rises sharply. Labels on the green bars include latency and speedup.</figcaption>
</figure>

### Detailed measurements

<div style="overflow-x: auto; margin: 1.25rem 0 1.5rem; border: 1px solid #e2e8f0; border-radius: 10px;">
  <table style="width: 100%; min-width: 680px; margin: 0; border-collapse: collapse; font-variant-numeric: tabular-nums;">
    <thead>
      <tr style="background: #f1f5f9; color: #334155;">
        <th scope="col" style="padding: .75rem; text-align: left; white-space: nowrap;">Shape</th>
        <th scope="col" style="padding: .75rem; text-align: right; white-space: nowrap;">Current LPLB</th>
        <th scope="col" style="padding: .75rem; text-align: right; white-space: nowrap;">Edge balance</th>
        <th scope="col" style="padding: .75rem; text-align: right; white-space: nowrap;">Speedup</th>
        <th scope="col" style="padding: .75rem; text-align: right; white-space: nowrap;">Median sweeps</th>
        <th scope="col" style="padding: .75rem; text-align: right; white-space: nowrap;">Max gap vs. HiGHS</th>
      </tr>
    </thead>
    <tbody>
      <tr><th scope="row" style="padding: .7rem .75rem; text-align: left;"><code>4×1</code></th><td style="padding: .7rem .75rem; text-align: right;">18.21 µs</td><td style="padding: .7rem .75rem; text-align: right; color: #0f766e; font-weight: 600;">14.43 µs</td><td style="padding: .7rem .75rem; text-align: right;">1.26×</td><td style="padding: .7rem .75rem; text-align: right;">2</td><td style="padding: .7rem .75rem; text-align: right;">0.00012%</td></tr>
      <tr style="background: #f8fafc;"><th scope="row" style="padding: .7rem .75rem; text-align: left;"><code>4×2</code></th><td style="padding: .7rem .75rem; text-align: right;">22.36 µs</td><td style="padding: .7rem .75rem; text-align: right; color: #0f766e; font-weight: 600;">13.70 µs</td><td style="padding: .7rem .75rem; text-align: right;">1.63×</td><td style="padding: .7rem .75rem; text-align: right;">2</td><td style="padding: .7rem .75rem; text-align: right;">0.00021%</td></tr>
      <tr><th scope="row" style="padding: .7rem .75rem; text-align: left;"><code>8×1</code></th><td style="padding: .7rem .75rem; text-align: right;">25.59 µs</td><td style="padding: .7rem .75rem; text-align: right; color: #0f766e; font-weight: 600;">13.53 µs</td><td style="padding: .7rem .75rem; text-align: right;">1.89×</td><td style="padding: .7rem .75rem; text-align: right;">3</td><td style="padding: .7rem .75rem; text-align: right;">0.00027%</td></tr>
      <tr style="background: #f8fafc;"><th scope="row" style="padding: .7rem .75rem; text-align: left;"><code>8×2</code></th><td style="padding: .7rem .75rem; text-align: right;">39.58 µs</td><td style="padding: .7rem .75rem; text-align: right; color: #0f766e; font-weight: 600;">15.13 µs</td><td style="padding: .7rem .75rem; text-align: right;">2.62×</td><td style="padding: .7rem .75rem; text-align: right;">4</td><td style="padding: .7rem .75rem; text-align: right;">0.00034%</td></tr>
      <tr><th scope="row" style="padding: .7rem .75rem; text-align: left;"><code>16×2</code></th><td style="padding: .7rem .75rem; text-align: right;">104.05 µs</td><td style="padding: .7rem .75rem; text-align: right; color: #0f766e; font-weight: 600;">19.24 µs</td><td style="padding: .7rem .75rem; text-align: right; font-weight: 700; color: #0f766e;">5.41×</td><td style="padding: .7rem .75rem; text-align: right;">9</td><td style="padding: .7rem .75rem; text-align: right;">0.00164%</td></tr>
    </tbody>
  </table>
</div>

All 960 solutions independently satisfied the capacity and load-conservation checks. The largest measured objective gap was 0.00164%.

The speedup increases with graph size. This is consistent with the difference in structure: edge balance performs <code>O(E)</code> local work per sweep, while the interior-point implementation builds and processes increasingly large dense systems.

## Reproducing the benchmark

The experimental benchmark entry point is selected with `--algorithm edge-balance`:

```bash
cd /path/to/your/workspace
source .venv/bin/activate

python LPLB/scripts/benchmark_solver.py \
  --algorithm edge-balance \
  --solver-id edge-balance-v1 \
  --corpus benchmark-results/baseline-highs/corpus.pt \
  --repeats 1000 \
  --max-sweeps 64 \
  --tolerance 1e-5 \
  --output benchmark-results/edge-balance-v1
```

Current defaults are

```text
max_sweeps = 64
tolerance  = 1e-5
```

## Scope and limitations

The current implementation assumes:

- at most 32 ranks per group, because one warp solves one group;
- the same number of replica slots at every rank;
- two endpoints per replica edge;
- a source mapping that is effectively a permutation for each replica slot;
- a valid matching within every supplied edge color.

Disconnected graphs are supported, but workload cannot move between components. The current convergence test monitors coordinate movement; it is not itself a formal global-optimality certificate. The mathematical argument applies to the continuous token-count LP and does not automatically extend to richer models with nonlinear GEMM time, communication congestion, or coupled multi-replica decisions.

## Next steps

The most useful extensions are:

- integrate coloring and caching into `Planner` initialization;
- warm-start from the previous batch's routing fractions;
- add an inexpensive primal-dual stopping certificate;
- precompute optimal colorings for standard topologies;
- separate diagnostic and production kernels;
- reuse output buffers where PyTorch stream semantics permit;
- support multi-warp groups larger than 32 ranks;
- validate on datacenter GPUs and end-to-end MoE workloads.

The broader lesson is that an LP solver for an online inference or training path should exploit the model's combinatorial structure. For this sparse load-redistribution problem, a graph-native coordinate method provides feasible iterates, simple parallelism, and much lower latency on the tested hardware.

## Code and citation

If you build on the current implementation, please reference the upstream LPLB project, the Edge Balance implementation branch, and this technical article:

- Upstream project: [deepseek-ai/LPLB](https://github.com/deepseek-ai/LPLB)
- Edge Balance implementation: [BochuanBob/LPLB — `feature/edge-balance-solver`](https://github.com/BochuanBob/LPLB/tree/feature/edge-balance-solver)
- Technical article: [A Graph-Native CUDA Solver for MoE Load Balancing](https://bochuanbob.github.io/lplb-edge-balance/)

---

*This is an independent experimental contribution to the open-source LPLB project. It is not an official DeepSeek result.*
