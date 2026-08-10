---
layout: blog
title: "Generalized LPLB with Hyperedge Water-Filling on CUDA"
author: "Bochuan Lyu and Zedong Peng"
date: 2026-08-09
description: "A CUDA load-balancing algorithm for experts replicated across multiple ranks"
permalink: /generalized-lplb-hyperedge-waterfill/
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']]
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

This experiment studies a generalized form of LPLB in which one expert may be available on more than two GPU ranks. The eligible ranks of an expert form a hyperedge, and the optimizer distributes that expert's demand across all ranks in the hyperedge.

<div class="result-strip" aria-label="Experiment summary">
  <div class="result-card"><strong>384 / 384</strong><span>feasible at tolerance 10<sup>−4</sup></span></div>
  <div class="result-card"><strong>57.55 µs</strong><span>median at tolerance 10<sup>−4</sup></span></div>
  <div class="result-card"><strong>0.0188%</strong><span>maximum gap at tolerance 10<sup>−4</sup></span></div>
</div>

<aside class="article-note research-status"><strong>Research status.</strong> This implementation is part of ongoing research by Bochuan Lyu and Zedong Peng on GPU-native optimization algorithms for MoE load balancing. The current experiments are development results; the broader algorithmic analysis and end-to-end evaluation are intended for a future research paper.</aside>

## Formulation

Let $\mathcal I$ be the set of GPU ranks and $\mathcal N$ the set of experts. Expert $n\in\mathcal N$ has demand $d_n$ and may execute on the subset $R_n\subseteq\mathcal I$. Let $b_i$ denote fixed load already assigned to rank $i$, and let $f_{ni}$ be the portion of expert $n$'s demand assigned to eligible rank $i$.

The generalized load-balancing problem is

<div class="math-display">
$$
\begin{aligned}
\underset{H,\,f}{\operatorname{minimize}}\quad
    & H \\
\text{subject to}\quad
    & b_i+\sum_{\substack{n\in\mathcal N\\i\in R_n}}f_{ni}\le H,
    && i\in\mathcal I,\\
    & \sum_{i\in R_n}f_{ni}=d_n,
    && n\in\mathcal N,\\
    & f_{ni}\ge 0,
    && n\in\mathcal N,\ i\in R_n.
\end{aligned}
$$
</div>

The equality constraint requires all expert demand to be processed. Using an inequality here would permit the trivial solution that assigns no demand.

## Algorithm overview

The algorithm combines water-filling with the coordinate-update structure of Edge Balance. It maintains a feasible assignment and the resulting rank loads $\ell_i$. To update one expert $n$, first remove its current contribution from each eligible rank:

<div class="math-display">
$$
a_i=\ell_i-f_{ni},
\qquad i\in R_n.
$$
</div>

Here, $a_i$ is the load on rank $i$ from every expert except $n$. The one-hyperedge subproblem is

<div class="math-display">
$$
\begin{aligned}
\underset{h,\,g}{\operatorname{minimize}}\quad
    & h\\
\text{subject to}\quad
    & a_i+g_i\le h,
    && i\in R_n,\\
    & \sum_{i\in R_n}g_i=d_n,\\
    & g_i\ge0,
    && i\in R_n.
\end{aligned}
$$
</div>

Its solution is a water-fill over the eligible ranks. Choose the waterline $\lambda_n$ satisfying

<div class="math-display">
$$
\sum_{i\in R_n}(\lambda_n-a_i)_+=d_n,
$$
</div>

then set

<div class="math-display">
$$
f_{ni}\leftarrow (\lambda_n-a_i)_+,
\qquad
\ell_i\leftarrow a_i+f_{ni},
\qquad i\in R_n.
$$
</div>

This update solves the one-hyperedge subproblem exactly and preserves both nonnegativity and complete demand assignment. The resulting subproblem objective is $h=\max_{i\in R_n}\ell_i$; it can exceed $\lambda_n$ when a rank's fixed load $a_i$ already lies above the waterline.

The CUDA implementation uses one warp and one block for one optimization problem. Hyperedges are greedily colored before the solve. Hyperedges in the same color have disjoint rank sets, so different warp lanes can update them concurrently without atomics. Colors and sweeps are processed sequentially, while rank loads and expert flows remain in shared memory. The initial demand of every expert is split equally among its eligible ranks.

The solver stops when the maximum flow change in a sweep is no greater than a specified tolerance.

## Experiment setup

The benchmark ran on an NVIDIA GeForce RTX 4060 under WSL2 with PyTorch 2.13.0 and CUDA 13.0. It used six generalized-LPLB shapes:

<div class="data-table-wrap compact-table-wrap">
  <table class="data-table numeric-table compact-table">
    <thead><tr><th scope="col">Ranks</th><th scope="col">Experts</th><th scope="col">Eligible locations per expert</th></tr></thead>
    <tbody>
      <tr><td>8</td><td>32</td><td>2</td></tr>
      <tr><td>8</td><td>32</td><td>3</td></tr>
      <tr><td>8</td><td>32</td><td>4</td></tr>
      <tr><td>16</td><td>64</td><td>2</td></tr>
      <tr><td>16</td><td>64</td><td>4</td></tr>
      <tr><td>16</td><td>64</td><td>8</td></tr>
    </tbody>
  </table>
</div>

For every shape, we generated 16 instances from each of four demand distributions: balanced, lognormal, Zipf, and hotspot. This produced 64 instances per shape and 384 instances per tolerance.

The problems were solved serially. Every CUDA launch contained exactly one optimization instance and one CUDA block. After 20 warmup launches for each compiled shape, every instance was measured over 100 consecutive kernel launches. Hyperedge coloring, tensor creation, data transfer, and verification were outside the timed region.

Every solution was checked independently for nonnegative flows, complete demand assignment, and consistent reconstructed rank loads. We then solved the same LP with SciPy HiGHS outside the timed region and measured the relative objective gap.

## Tolerance results

We evaluated flow-change tolerances from $10^{-2}$ through $10^{-5}$. Each row summarizes the same 384 LP instances.

<div class="data-table-wrap">
  <table class="data-table benchmark-table tolerance-table">
    <thead><tr><th scope="col">Tolerance</th><th scope="col">Median</th><th scope="col">p90</th><th scope="col">p99</th><th scope="col">Mean sweeps</th><th scope="col">Feasible</th><th scope="col">Mean gap</th><th scope="col">Max gap</th></tr></thead>
    <tbody>
      <tr><th scope="row">10<sup>−2</sup></th><td>36.45 µs</td><td>122.58 µs</td><td>125.85 µs</td><td>3.07</td><td class="good">384 / 384</td><td>0.0780%</td><td>2.1958%</td></tr>
      <tr><th scope="row">10<sup>−3</sup></th><td>44.17 µs</td><td>122.31 µs</td><td>138.73 µs</td><td>4.35</td><td class="good">384 / 384</td><td>0.0080%</td><td>0.2231%</td></tr>
      <tr class="selected-row"><th scope="row">10<sup>−4</sup> <span class="table-badge">selected</span></th><td>57.55 µs</td><td>123.05 µs</td><td>176.78 µs</td><td>5.77</td><td class="good">384 / 384</td><td>0.00077%</td><td>0.0188%</td></tr>
      <tr><th scope="row">10<sup>−5</sup></th><td>71.90 µs</td><td>125.30 µs</td><td>228.10 µs</td><td>7.18</td><td class="good">384 / 384</td><td>0.000083%</td><td>0.00217%</td></tr>
    </tbody>
  </table>
</div>

All 384 instances were feasible and converged at every tolerance. Across the four tolerance settings, this corresponds to 1,536 feasible CUDA solves; HiGHS also solved every verification LP. Tightening the tolerance increases the number of sweeps and median latency while steadily reducing the objective gap. We use $10^{-4}$ as the current balanced setting: its median latency is 57.55 microseconds, its mean objective gap is 0.00077%, and its maximum observed gap is 0.0188%.

## Results at the selected tolerance

The following table breaks down the $10^{-4}$ results by problem shape.

<div class="data-table-wrap">
  <table class="data-table benchmark-table shape-table">
    <thead><tr><th scope="col">Ranks</th><th scope="col">Experts</th><th scope="col">Locations</th><th scope="col">Median</th><th scope="col">p99</th><th scope="col">Mean sweeps</th><th scope="col">Feasible</th><th scope="col">Mean gap</th><th scope="col">Max gap</th></tr></thead>
    <tbody>
      <tr><td>8</td><td>32</td><td>2</td><td>40.61 µs</td><td>94.89 µs</td><td>7.61</td><td class="good">64 / 64</td><td>0.00133%</td><td>0.0108%</td></tr>
      <tr><td>8</td><td>32</td><td>3</td><td>43.46 µs</td><td>117.89 µs</td><td>5.25</td><td class="good">64 / 64</td><td>0.00063%</td><td>0.0077%</td></tr>
      <tr><td>8</td><td>32</td><td>4</td><td>55.58 µs</td><td>110.12 µs</td><td>3.16</td><td class="good">64 / 64</td><td>0.00018%</td><td>0.0023%</td></tr>
      <tr><td>16</td><td>64</td><td>2</td><td>49.52 µs</td><td>122.44 µs</td><td>11.11</td><td class="good">64 / 64</td><td>0.00183%</td><td>0.0176%</td></tr>
      <tr><td>16</td><td>64</td><td>4</td><td>63.12 µs</td><td>203.55 µs</td><td>5.39</td><td class="good">64 / 64</td><td>0.00062%</td><td>0.0188%</td></tr>
      <tr><td>16</td><td>64</td><td>8</td><td>122.47 µs</td><td>176.33 µs</td><td>2.08</td><td class="good">64 / 64</td><td>0.00002%</td><td>0.0001%</td></tr>
    </tbody>
  </table>
</div>

The results show that hyperedge water-filling preserves feasibility by construction and produces objectives close to the HiGHS optimum. Larger hyperedges generally need fewer sweeps, but each update performs more work and dense hypergraphs expose fewer nonconflicting hyperedges per color. The tolerance therefore provides a direct and useful latency-quality control for the CUDA solver.

## Code and citation

If you build on the current implementation, please reference the upstream LPLB project, the Generalized LPLB implementation branch, and this technical article:

- Upstream project: [deepseek-ai/LPLB](https://github.com/deepseek-ai/LPLB)
- Generalized LPLB implementation: [BochuanBob/LPLB — `feature/generalized-lplb`](https://github.com/BochuanBob/LPLB/tree/feature/generalized-lplb)
- Technical article: [Generalized LPLB with Hyperedge Water-Filling on CUDA](https://bochuanbob.github.io/generalized-lplb-hyperedge-waterfill/)
