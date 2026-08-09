---
layout: blog
title: "Waterfill and LPLB: An Operations Research View of MoE Load Balancing"
date: 2026-08-09
description: "An operations research view of Waterfill and LPLB for load balancing in mixture-of-experts systems"
permalink: /waterfill-lplb-or-view/
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

# Waterfill and LPLB: An Operations Research View of MoE Load Balancing

*Bochuan Lyu and Zedong Peng — August 2026*

Mixture-of-Experts (MoE) inference is a load-balancing problem in disguise. Tokens create jobs, GPU ranks are machines, and the latency of an MoE layer is determined largely by the busiest rank. From an operations research (OR) viewpoint, the natural objective is therefore **min–max load balancing**: minimize the maximum workload assigned to any rank.

[Waterfill and LPLB](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) address two different sources of imbalance in SGLang. Waterfill assigns the dense shared-expert computation to relatively light ranks. LPLB distributes the tokens of replicated routed experts among their physical copies. Both change only the physical execution location, not the logical experts selected by the model.

## 1. Waterfill: balancing the shared expert

Let $\mathcal R$ be the set of GPU ranks, and let $L_r$ be the routed-expert load already assigned to rank $r\in\mathcal R$. There are $N$ shared-expert jobs—one for every participating token. If any job can be assigned to any rank, let $y_r$ denote the number of shared-expert jobs placed on rank $r$, and let $H$ denote the resulting peak load. The idealized OR model is

<div class="math-display">
$$
\begin{aligned}
\underset{H,\,y}{\operatorname{minimize}}\quad
    & H \\
\text{subject to}\quad
    & L_r+y_r \le H,
    && r\in\mathcal R, \\
    & \sum_{r\in\mathcal R} y_r = N, \\
    & y_r\in\mathbb Z_{\ge 0},
    && r\in\mathcal R.
\end{aligned}
$$
</div>

Thus, the shared-expert work should fill the low-load “valleys” until the rank loads are as level as possible.

In practice, communication matters. Let $\mathcal T=\lbrace 1,\ldots,N\rbrace$ be the token set and $C_t\subseteq\mathcal R$ the candidate ranks for token $t$. This set usually contains ranks that the token already visits for routed experts, plus its source rank as a fallback. Define $z_{tr}=1$ if token $t$ sends its shared-expert job to rank $r$, and $z_{tr}=0$ otherwise. A more precise formulation is

<div class="math-display">
$$
\begin{aligned}
\underset{H,\,z}{\operatorname{minimize}}\quad
    & H \\
\text{subject to}\quad
    & L_r+\sum_{\substack{t\in\mathcal T\\ r\in C_t}} z_{tr}\le H,
    && r\in\mathcal R, \\
    & \sum_{r\in C_t}z_{tr}=1,
    && t\in\mathcal T, \\
    & z_{tr}\in\{0,1\},
    && t\in\mathcal T,\ r\in C_t.
\end{aligned}
$$
</div>

Solving this integer program for every layer would be too expensive. Waterfill uses a lightweight randomized approximation.

### Waterfill algorithm

1. Count the routed-expert load $L_r$ on every rank.
2. Compute the target waterline

   <div class="math-display">
   $$
   H=\left\lceil
   \frac{\displaystyle\sum_{r\in\mathcal R}L_r+N}
        {|\mathcal R|}
   \right\rceil.
   $$
   </div>

3. Compute the slack below the waterline:

   <div class="math-display">
   $$
   S_r=(H-L_r)_+
   \equiv \max\{H-L_r,\,0\}.
   $$
   </div>

4. For each token $t$, restrict attention to its candidate set $C_t$. Select a target rank with probability approximately proportional to its slack $S_r$, while giving a small preference to the local rank to reduce communication.
5. If all candidate ranks have zero slack, choose a clearly lighter candidate, again using locality as a tie-breaker.
6. Dispatch the shared-expert slot together with the routed-expert work.

This is a classic OR tradeoff: Waterfill sacrifices exact optimality to obtain a decision rule with near-zero runtime overhead and communication-aware assignments.

## 2. LPLB: balancing replicated routed experts

In the LPLB setting considered here, each duplicated logical expert has exactly two physical locations: its original location and one replica. Let $\mathcal E$ be the set of duplicated experts. For each $e\in\mathcal E$, let $u_e$ be its original rank, $v_e$ its replica rank, and $d_e$ its observed demand.

Start with all demand assigned to the original locations, and let $b_r$ be the resulting load on rank $r$. The decision variable $f_e$ is the amount of expert-$e$ demand redirected from $u_e$ to $v_e$. Thus,

<div class="math-display">
$$
0\le f_e\le d_e,
\qquad e\in\mathcal E.
$$
</div>

The final load on rank $r$ is

<div class="math-display">
$$
\ell_r(f)
=b_r
-\sum_{\substack{e\in\mathcal E\\u_e=r}} f_e
+\sum_{\substack{e\in\mathcal E\\v_e=r}} f_e.
$$
</div>

The LPLB linear program is

<div class="math-display">
$$
\begin{aligned}
\underset{H,\,f}{\operatorname{minimize}}\quad
    & H \\
\text{subject to}\quad
    & \ell_r(f)\le H,
    && r\in\mathcal R, \\
    & 0\le f_e\le d_e,
    && e\in\mathcal E.
\end{aligned}
$$
</div>

The box constraint conserves demand automatically: $d_e-f_e$ remains at the original rank and $f_e$ goes to the replica. For $d_e>0$, the corresponding routing probabilities are

<div class="math-display">
$$
p_e^{\mathrm{original}}=1-\frac{f_e}{d_e},
\qquad
p_e^{\mathrm{replica}}=\frac{f_e}{d_e}.
$$
</div>

LPLB samples one of the two physical locations for each token using these probabilities. The continuous flow therefore translates naturally into randomized dispatch.

### What the interior-point method does

The interior-point method (IPM) solves this LP by replacing the nonnegativity boundary with a barrier and repeatedly taking primal–dual Newton steps. Each iteration reduces primal infeasibility, dual infeasibility, and the complementarity gap. The main computational work is forming and solving a structured linear system.

In SGLang, every rank first obtains the same global expert counts through an all-reduce, then independently solves the same LP. The GPU implementation fuses the IPM operations and uses preallocated buffers; the placement-dependent matrix structure is prepared offline, while only the right-hand side changes with each batch. This gives a high-quality min–max solution, but it still requires an all-reduce, several Newton iterations, and dense or structured linear algebra on the critical path.

## 3. Edge balance: a matrix-free alternative

With one replica per duplicated expert, the graph is immediate: expert $e$ defines one capacitated edge $e=(u_e,v_e)$ between its original and replica ranks, with capacity $d_e$ and current flow $f_e$.

Suppose the current endpoint loads are $\ell_{u_e}$ and $\ell_{v_e}$. Ignoring the bounds for a moment, moving

<div class="math-display">
$$
\frac{\ell_{u_e}-\ell_{v_e}}{2}
$$
</div>

units from $u_e$ to $v_e$ would make the two endpoint loads equal. The flow must remain in $[0,d_e]$, so the actual update is

<div class="math-display">
$$
\delta_e=
\operatorname{clip}\left(
    \frac{\ell_{u_e}-\ell_{v_e}}{2},
    -f_e,
    d_e-f_e
\right).
$$
</div>

Then update the edge flow and its two endpoint loads together:

<div class="math-display">
$$
\begin{aligned}
f_e &\leftarrow f_e+\delta_e,\\
\ell_{u_e} &\leftarrow \ell_{u_e}-\delta_e,\\
\ell_{v_e} &\leftarrow \ell_{v_e}+\delta_e.
\end{aligned}
$$
</div>

The lower bound $-f_e$ allows a later sweep to reverse an earlier transfer; the upper bound $d_e-f_e$ prevents sending more demand than the expert has. Consequently, every update preserves

<div class="math-display">
$$
0\le f_e\le d_e
$$
</div>

and therefore produces a feasible routing decision. Repeated sweeps over the replica edges progressively balance the rank loads. Edges that do not share a rank can be processed in parallel, which is exactly what the edge coloring exposes to the CUDA kernel.

For a CUDA implementation and experiments comparing this method with the existing LPLB solver, see [“A Graph-Native CUDA Solver for MoE Load Balancing”](https://bochuanbob.github.io/lplb-edge-balance/).

## Edge balance versus IPM

The main advantage of edge balance is **lower solution overhead**. It uses only local comparisons and flow transfers on the expert–rank graph. It needs no barrier parameter, KKT matrix, factorization, or linear-system solve. Its memory and work scale roughly with the number of replica edges, it maintains a feasible dispatch after every update, and it can stop early when the cost of further optimization exceeds the expected load-balancing gain. These properties are especially attractive for small per-layer problems and latency-sensitive GPU inference.

IPM, however, provides a principled solution of the full LP and a numerical optimality certificate. Edge balance wins when its much cheaper iterations reach a sufficiently good solution in a few sweeps; IPM remains preferable when exact solution quality, robustness on difficult replica graphs, or a reliable optimality gap matters more than microsecond-scale overhead.

In short, Waterfill is a communication-aware heuristic for dense shared-expert work, LPLB is a min–max LP for sparse replicated-expert work, and edge balance is a promising graph-specialized way to approximate the LPLB solution faster. This is precisely where OR meets systems design: the best optimizer is not always the one with the strongest asymptotic guarantee, but the one whose decision quality justifies its cost on the inference critical path.

## Reference

- NVIDIA Team, [“Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB,”](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) LMSYS, June 26, 2026.
- Bochuan Lyu, [“A Graph-Native CUDA Solver for MoE Load Balancing,”](https://bochuanbob.github.io/lplb-edge-balance/) August 2026.
