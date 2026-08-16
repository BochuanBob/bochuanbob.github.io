---
layout: blog
title: "Expert Placement as Optimization: An OR View of DeepSeek EPLB"
author: "Bochuan Lyu"
date: 2026-08-16
description: "An operations research interpretation of expert replication and placement in DeepSeek EPLB, SGLang, and vLLM"
permalink: /eplb-or-view/
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

Mixture-of-experts models activate only a small subset of their experts for each token. This saves computation, but it creates a systems problem: expert popularity is neither uniform nor stationary. If popular experts happen to reside on the same GPU, that GPU becomes a straggler while other GPUs wait.

Expert Parallelism Load Balancing (EPLB) changes the physical deployment of experts in response to measured load. It can create redundant copies of popular logical experts and place the resulting physical experts across GPUs. From an operations research perspective, this is a capacitated replication-and-placement problem with a makespan objective.

This article studies the algorithm that DeepSeek released and the current implementations derived from it in SGLang and vLLM. It focuses on the optimization structure and existing software behavior—not alternative algorithms.

<aside class="article-note"><strong>Snapshot.</strong> The implementation discussion refers to DeepSeek EPLB commit <code>d52c72d</code>, SGLang commit <code>22cf5e1</code>, and vLLM commit <code>83ad767</code>, inspected on August 16, 2026.</aside>

<figure>
  <img src="/assets/eplb-three-stage.svg" alt="Diagram of the three-stage DeepSeek EPLB pipeline: pack expert groups onto nodes, allocate replicas, pack physical copies onto GPUs, then install the placement plan in the serving runtime.">
  <figcaption>EPLB separates topology, replication, and GPU scheduling. The resulting maps are consumed by the serving runtime, where expert weights are moved and tokens are routed.</figcaption>
</figure>

## The planning problem

Consider one MoE layer. Let:

- $\mathcal E$ be the logical experts;
- $\mathcal G$ be the GPUs;
- $\lambda_e$ be the measured load of expert $e$;
- $P$ be the number of physical expert slots;
- $C=P/|\mathcal G|$ be the number of slots on each GPU, assuming $P$ is divisible by $|\mathcal G|$.

Let $r_e\ge 1$ be the number of physical replicas of expert $e$, and let $x_{eg}$ be the number of those replicas placed on GPU $g$. Under EPLB's load model, the traffic of an expert is divided equally among its replicas. One copy of expert $e$ therefore contributes $\lambda_e/r_e$ load to its GPU.

A natural integrated formulation is

<div class="math-display">
$$
\begin{aligned}
\underset{T,r,x}{\operatorname{minimize}}\quad & T\\
\text{subject to}\quad
& \sum_{e\in\mathcal E}x_{eg}\frac{\lambda_e}{r_e}\le T,
&&g\in\mathcal G,\\
& \sum_{g\in\mathcal G}x_{eg}=r_e,
&&e\in\mathcal E,\\
& \sum_{e\in\mathcal E}x_{eg}=C,
&&g\in\mathcal G,\\
& \sum_{e\in\mathcal E}r_e=P,\\
& r_e\in\mathbb Z_{\ge1},\quad x_{eg}\in\mathbb Z_{\ge0}.
\end{aligned}
$$
</div>

The objective minimizes the maximum estimated GPU load. The formulation couples two decisions:

1. how many replicas each expert receives;
2. where those replicas are placed.

The term $x_{eg}\lambda_e/r_e$ makes the integrated model nonlinear. Even after fixing $r$, placement is a cardinality-constrained makespan scheduling problem. A serving system also needs an answer quickly, so the released implementation uses a structured greedy decomposition rather than solving the integrated model exactly.

## The three layers of DeepSeek EPLB

DeepSeek uses hierarchical balancing when expert groups can be divided evenly among nodes. Otherwise, it falls back to global balancing. The hierarchical policy has three stages.

### 1. Pack expert groups onto nodes

DeepSeek's group-limited routing makes groups the natural unit of inter-node placement. If $\mathcal Q$ is the set of expert groups, the measured group load is

<div class="math-display">
$$
w_q=\sum_{e\in q}\lambda_e.
$$
</div>

Every node must receive the same number of groups. The corresponding optimization subproblem is a cardinality-constrained load-balancing problem:

<div class="math-display">
$$
\begin{aligned}
\underset{H,y}{\operatorname{minimize}}\quad & H\\
\text{subject to}\quad
& \sum_{q\in\mathcal Q}w_qy_{qn}\le H,
&&n\in\mathcal N,\\
& \sum_{n\in\mathcal N}y_{qn}=1,
&&q\in\mathcal Q,\\
& \sum_{q\in\mathcal Q}y_{qn}=|\mathcal Q|/|\mathcal N|,
&&n\in\mathcal N,\\
& y_{qn}\in\{0,1\}.
\end{aligned}
$$
</div>

The implementation sorts groups by decreasing load and assigns each group to the lightest node that still has capacity. This is a capacity-constrained version of longest-processing-time-first scheduling.

This stage does more than balance computation. Keeping an expert group inside one node respects the topology induced by group-limited routing and reduces unnecessary inter-node traffic.

### 2. Choose redundant experts within each node

After groups are assigned, each node owns a fixed set of logical experts and a fixed number of physical slots. Ignoring placement temporarily, replication chooses integer counts satisfying

<div class="math-display">
$$
r_e\ge1,
\qquad
\sum_e r_e=P_{\text{node}}.
$$
</div>

The proxy objective is the largest per-replica expert load:

<div class="math-display">
$$
\min_r\ \max_e\frac{\lambda_e}{r_e}.
$$
</div>

The DeepSeek rule starts every expert with one copy. For every redundant slot, it selects

<div class="math-display">
$$
e^*\in\operatorname*{arg\,max}_e\frac{\lambda_e}{r_e}
$$
</div>

and increments $r_{e^*}$. In words: repeatedly split the currently heaviest expert piece. In hierarchical mode this operation is performed independently inside each node, so replicas do not cross the node assignment selected in stage 1.

This rule is fast and directly attacks the largest individual physical-expert load. It does not yet consider which GPU will hold the new copy; that interaction is deferred to stage 3.

### 3. Pack physical experts onto GPUs

Once replica counts are fixed, every physical copy of expert $e$ receives weight

<div class="math-display">
$$
p_e=\frac{\lambda_e}{r_e}.
$$
</div>

The remaining problem assigns exactly $C$ physical experts to each GPU while minimizing the largest GPU load. DeepSeek again uses decreasing-weight greedy packing: sort physical experts from heaviest to lightest, then place each one on the lightest GPU with an available slot.

The output is not merely a vector of replica counts. It includes:

- a physical-to-logical map, identifying the logical expert stored in every physical slot;
- a logical-to-physical map, listing the slots that contain each logical expert;
- the number of replicas of every logical expert.

The global policy is the same construction with the hierarchy removed: treat the whole deployment as one node, replicate globally, and then pack across all GPUs.

## What the decomposition means

The three stages solve related but distinct approximations:

<div class="data-table-wrap">
  <table class="data-table stage-table">
    <thead><tr><th scope="col">Stage</th><th scope="col">Decision</th><th scope="col">Optimization view</th><th scope="col">Greedy rule</th></tr></thead>
    <tbody>
      <tr><th scope="row">Node packing</th><td>Group → node</td><td>Capacitated makespan scheduling</td><td>Heaviest group to lightest nonfull node</td></tr>
      <tr><th scope="row">Replication</th><td>Copies per expert</td><td>Discrete minimax resource allocation</td><td>Split the largest $\lambda_e/r_e$</td></tr>
      <tr><th scope="row">GPU packing</th><td>Physical copy → GPU</td><td>Cardinality-constrained makespan scheduling</td><td>Heaviest copy to lightest nonfull GPU</td></tr>
    </tbody>
  </table>
</div>

The decomposition is operationally attractive: each stage is simple, deterministic apart from ties, and inexpensive relative to moving expert weights. Its main modeling assumption is equally divided load among replicas. EPLB decides where copies exist; the runtime routing mechanism must then distribute traffic among those copies for the estimated GPU loads to materialize.

This distinction also separates EPLB from adjacent MoE components. EPLB plans replication and placement over a statistics window. Token routing chooses a replica while serving individual tokens, and communication libraries move token payloads between ranks. They operate at different timescales and solve different problems.

## DeepSeek's reference implementation

The [DeepSeek EPLB repository](https://github.com/deepseek-ai/EPLB) contains a compact PyTorch implementation in one main file. Its public entry point accepts a load tensor with shape `[layers, logical_experts]` and returns the placement maps and replica counts.

Although it uses PyTorch tensors, the planner is CPU-oriented. The entry point converts the load tensor with `weight.float().cpu()`. Balanced packing sorts with PyTorch, transfers the sorted indices to CPU, and uses Python loops and lists to maintain pack loads and cardinalities. Replication is more vectorized across rows but still advances through redundant slots sequentially.

Policy selection is structural:

- if the number of nodes divides the number of expert groups, use hierarchical placement;
- otherwise, invoke the same hierarchical routine with one synthetic group and one node, producing global placement.

The reference repository intentionally leaves load prediction outside its scope. A serving framework must collect and aggregate the expert statistics supplied to the planner.

## SGLang's implementation

SGLang vendors the DeepSeek algorithm and connects it to online load collection, placement updates, and expert-weight movement. Its algorithm dispatcher currently exposes several choices, including `deepseek`, `deepseek_hierarchical`, `deepseek_vec`, and `deepseek_vec_hierarchical`.

The automatic selection path chooses the ordinary DeepSeek implementation: hierarchical when the group/node divisibility condition holds, global otherwise. The ordinary path sums the recorded statistics over the observation dimension and calls a close adaptation of DeepSeek's CPU planner.

SGLang also contains `deepseek_vec.py`, a more tensorized implementation. Its interface retains a load-history dimension:

<div class="math-display">
$$
[\text{steps},\ \text{MoE layers},\ \text{logical experts}].
$$
</div>

The global/decode path performs replica decisions for layers and history together using PyTorch operations. The hierarchical/prefill path first converts statistics to CPU and uses Python-based group packing before invoking the chunkwise tensorized replication and placement routine. Thus `deepseek_vec` is substantially more batched than the default implementation, but its hierarchical path is still explicitly CPU-oriented.

The important systems fact is that planning is only one part of rebalancing. SGLang accumulates expert loads, computes a new mapping at a configured interval, and then coordinates changes to the expert weights and maps used by execution. A fast planning result does not eliminate the cost or synchronization requirements of applying that result.

## vLLM's implementation

vLLM's `DefaultEplbPolicy` is also adapted from DeepSeek EPLB, but its solver is written in NumPy. The entry point explicitly performs

```python
weight_np = weight.float().cpu().numpy()
```

and returns the final NumPy mapping as a CPU PyTorch tensor.

The policy accepts all MoE layers in one array. Replica selection uses NumPy operations across the layer dimension, while balanced packing loops over layers and then over sorted items. Therefore, the implementation is partially batched: its data model is layer-batched, but the greedy packing control flow is not.

vLLM adds an important migration-aware postprocessing step. If the previous physical-to-logical map is available and the topology is unchanged, `preserve_intragpu_slots` reorders the new plan so experts remaining on the same GPU retain their old slot positions when possible. This does not change which experts belong to a GPU; it reduces unnecessary local weight copying caused only by slot permutations.

The runtime surrounding the policy is deliberately asynchronous by default. The inspected configuration uses a load window of 1,000 steps, a rearrangement interval of 3,000 steps, and `use_async=True`. The asynchronous worker copies the global load window to CPU, computes the new mapping there, and migrates one MoE layer per model forward pass. This design makes planner latency less directly visible to requests and avoids using inference GPU resources for the optimization routine.

## Side-by-side implementation view

<div class="data-table-wrap">
  <table class="data-table system-table">
    <thead><tr><th scope="col">System</th><th scope="col">Planner representation</th><th scope="col">Layer batching</th><th scope="col">Normal planning device</th><th scope="col">Runtime integration</th></tr></thead>
    <tbody>
      <tr><th scope="row">DeepSeek EPLB</th><td>PyTorch plus Python greedy loops</td><td>Replication partly batched; packing loops by layer</td><td>CPU</td><td>Reference planner only</td></tr>
      <tr><th scope="row">SGLang <code>deepseek</code></th><td>Close DeepSeek adaptation</td><td>Same basic structure</td><td>CPU</td><td>Online statistics and coordinated remapping</td></tr>
      <tr><th scope="row">SGLang <code>deepseek_vec</code></th><td>Tensorized PyTorch</td><td>Yes for core chunkwise operations</td><td>Device-generic globally; CPU hierarchical preprocessing</td><td>Selectable alternative, not the automatic default</td></tr>
      <tr><th scope="row">vLLM default</th><td>NumPy greedy planner</td><td>Partial</td><td>CPU</td><td>Async by default; migration-aware slot preservation</td></tr>
    </tbody>
  </table>
</div>

All three systems implement the same central idea: estimate expert demand, give more copies to heavy experts, and greedily place equally weighted copies into fixed-capacity GPU bins. Their main differences are not the objective, but execution details—how statistics are represented, how much work is batched across layers, where the planner runs, and how the resulting placement is installed safely.

## The OR takeaway

EPLB is best understood as a fast decomposition of a joint replication-and-placement problem. Stage 1 captures network topology through group-to-node assignment. Stage 2 allocates a limited replication budget. Stage 3 schedules the resulting physical experts onto equal-capacity GPUs. The maximum predicted GPU load links all three stages, but the implementation separates them to obtain a plan with simple greedy operations.

That decomposition explains both the strength and the limits of the current approach. It is lightweight enough to run periodically in a serving system, respects hard slot capacities, and maps cleanly to runtime metadata. At the same time, replica counts are selected before the final packing consequences are known, and the load model assumes that traffic divides evenly among copies. Those are modeling choices, not implementation accidents, and they are the natural starting point for evaluating EPLB from an operations research perspective.

## Source map

- [DeepSeek EPLB reference implementation](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913d/eplb.py)
- [SGLang DeepSeek EPLB algorithm dispatcher](https://github.com/sgl-project/sglang/blob/22cf5e1795956e05948a65da06aee96702810bae/python/sglang/srt/eplb/eplb_algorithms/__init__.py)
- [SGLang vectorized DeepSeek implementation](https://github.com/sgl-project/sglang/blob/22cf5e1795956e05948a65da06aee96702810bae/python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py)
- [vLLM default EPLB policy](https://github.com/vllm-project/vllm/blob/83ad767eed3be3ee7f2df63be693bfaca5c7c922/vllm/distributed/eplb/policy/default.py)
- [vLLM EPLB state and asynchronous integration](https://github.com/vllm-project/vllm/blob/83ad767eed3be3ee7f2df63be693bfaca5c7c922/vllm/distributed/eplb/eplb_state.py)
