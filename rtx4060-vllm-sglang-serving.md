---
layout: blog
title: "vLLM and SGLang on an RTX 4060: A Reproducible Serving Pilot"
date: 2026-08-09
description: "Identical request traces reveal how concurrency, prefill, and decode shape single-GPU LLM serving performance"
permalink: /rtx4060-vllm-sglang-serving/
---

# vLLM and SGLang on an RTX 4060: A Reproducible Serving Pilot

*Bochuan Lyu — August 2026*

How well can modern LLM serving systems use a consumer GPU? I built a small reproducible benchmark to compare [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) on a laptop-class NVIDIA RTX 4060 under WSL2.

This experiment is a pilot, not a definitive leaderboard. Its purpose is to validate a fair test harness, establish useful workload shapes, and identify the measurements that matter before running a larger study. Both backends received the same prompts and output limits in every matching test. Across 32 benchmark points, all 256 requests completed successfully.

The main result is that there is no single winner across all workloads. At concurrency eight, SGLang was slightly faster on short chat and balanced traffic, while vLLM led on decode-heavy and prefill-heavy traffic. More importantly, both systems gained substantial throughput from continuous batching as concurrency increased.

<div class="result-strip" aria-label="Experiment summary">
  <div class="result-card"><strong>32</strong><span>benchmark points</span></div>
  <div class="result-card"><strong>256 / 256</strong><span>requests completed</span></div>
  <div class="result-card"><strong>1–8</strong><span>concurrency range</span></div>
</div>

## Experimental setup

The experiment used the following environment:

<div class="data-table-wrap">
  <table class="data-table config-table">
    <thead><tr><th scope="col">Component</th><th scope="col">Configuration</th></tr></thead>
    <tbody>
      <tr><th scope="row">GPU</th><td>NVIDIA GeForce RTX 4060, 8 GB</td></tr>
      <tr><th scope="row">Platform</th><td>WSL2, NVIDIA driver 610.88</td></tr>
      <tr><th scope="row">Model</th><td>Qwen2.5-1.5B-Instruct</td></tr>
      <tr><th scope="row">Precision</th><td>FP16</td></tr>
      <tr><th scope="row">Maximum context</th><td>4,608 tokens</td></tr>
      <tr><th scope="row">API</th><td>OpenAI-compatible <code>/v1/completions</code>, streamed</td></tr>
      <tr><th scope="row">Sampling</th><td>Temperature 0</td></tr>
      <tr><th scope="row">Requests per point</th><td>8</td></tr>
      <tr><th scope="row">Repetitions</th><td>1</td></tr>
      <tr><th scope="row">Concurrency</th><td>1, 2, 4, and 8</td></tr>
    </tbody>
  </table>
</div>

I tested four synthetic workload shapes. They isolate different phases of serving instead of reducing the experiment to one average prompt.

<div class="data-table-wrap">
  <table class="data-table workload-table">
    <thead><tr><th scope="col">Workload</th><th scope="col">Input tokens</th><th scope="col">Max output</th><th scope="col">Primary stress</th></tr></thead>
    <tbody>
      <tr><th scope="row">Short chat</th><td>128</td><td>128</td><td>Interactive generation</td></tr>
      <tr><th scope="row">Decode-heavy</th><td>32</td><td>512</td><td>Autoregressive decoding</td></tr>
      <tr><th scope="row">Balanced</th><td>1,024</td><td>256</td><td>Mixed prefill and decode</td></tr>
      <tr><th scope="row">Prefill-heavy</th><td>4,096</td><td>32</td><td>Long-prompt ingestion</td></tr>
    </tbody>
  </table>
</div>

For every matching backend, workload, and concurrency point, the client replayed an identical JSONL trace. Hashing the trace files verifies that prompt text, ordering, and generation limits were the same. This matters because small prompt or output-length differences can easily overwhelm the framework-level differences being measured.

The client measured time to first token (TTFT), end-to-end request latency, inter-token latency, request throughput, and output-token throughput. Server startup and model loading were excluded from request latency.

<aside class="article-note"><strong>How to read the results.</strong> Throughput measures total system capacity and benefits from batching. TTFT measures how quickly an individual request begins receiving output. Improving one does not necessarily improve the other.</aside>

## Results at concurrency eight

Concurrency eight was the highest load in this pilot and shows the batching behavior most clearly.

<figure>
  <img src="/assets/rtx4060-serving-results.svg" alt="Two bar charts comparing vLLM and SGLang output throughput and median time to first token across four workloads at concurrency eight.">
  <figcaption>Throughput and median TTFT at concurrency eight. Higher throughput is better; lower TTFT is better. The small sample makes close results directional rather than definitive.</figcaption>
</figure>

<div class="data-table-wrap">
  <table class="data-table benchmark-table">
    <thead>
      <tr><th scope="col">Workload</th><th scope="col">vLLM output</th><th scope="col">SGLang output</th><th scope="col">vLLM p50 TTFT</th><th scope="col">SGLang p50 TTFT</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">Short chat</th><td>528.6 tok/s</td><td class="best">547.4 tok/s</td><td>46.7 ms</td><td class="best">45.8 ms</td></tr>
      <tr><th scope="row">Decode-heavy</th><td class="best">589.2 tok/s</td><td>546.2 tok/s</td><td>45.1 ms</td><td class="best">42.3 ms</td></tr>
      <tr><th scope="row">Balanced</th><td>499.2 tok/s</td><td class="best">504.4 tok/s</td><td>50.6 ms</td><td class="best">45.7 ms</td></tr>
      <tr><th scope="row">Prefill-heavy</th><td class="best">370.3 tok/s</td><td>341.1 tok/s</td><td class="best">86.0 ms</td><td>111.9 ms</td></tr>
    </tbody>
  </table>
</div>

The differences are workload-dependent. SGLang's output throughput was 3.6% higher for short chat and 1.0% higher for the balanced workload. vLLM was 7.9% higher for decode-heavy traffic and 8.6% higher for prefill-heavy traffic. These are useful observations for this configuration, but one repetition is not enough to treat small differences as stable rankings.

## Concurrency matters more than the narrow framework gap

Both systems converted concurrency into much higher aggregate throughput. From concurrency one to eight, vLLM throughput increased by approximately 7.9× for short chat, 8.1× for decode-heavy traffic, 6.9× for balanced traffic, and 6.9× for prefill-heavy traffic. SGLang increased by approximately 7.9×, 7.1×, 6.8×, and 9.3×, respectively.

This does not mean an individual request became eight times faster. Continuous batching lets the GPU process tokens from several requests together, improving utilization and aggregate throughput. Long-output request latency remained roughly stable or increased modestly while many more output tokens were produced per second. That distinction between per-request latency and system throughput is central to interpreting serving benchmarks.

## Prefill and decode behave differently

The workload shape changes both latency and the framework ranking. At concurrency eight, median TTFT stayed around 42–51 ms for the short-chat, decode-heavy, and balanced tests. It rose to 86.0 ms for vLLM and 111.9 ms for SGLang on the 4,096-token prefill-heavy workload.

This is expected: TTFT includes processing the complete prompt before the first generated token can be returned. A short prompt followed by 512 output tokens stresses iterative decoding, while a 4,096-token prompt followed by only 32 output tokens stresses prefill. Reporting only total tokens per second would hide this distinction.

The prefill-heavy concurrency-one measurements also show much larger TTFT—382.5 ms for vLLM and 420.2 ms for SGLang—than later concurrent runs. Those points likely include cold-kernel or compilation effects. Similarly, SGLang short chat at concurrency two had a p95 TTFT of 760.4 ms despite a 32.9 ms median. These outliers are reasons to add warmups and repetitions, not reasons to select a winner from a single run.

## Compatibility choices on WSL2

The RTX 4060 was fully visible to both frameworks, but this environment required practical compatibility choices. vLLM used its V1 model runner because the V2 path required unified virtual addressing that was unavailable through this WSL setup. It also used the native sampler because the packaged FlashInfer compiler and headers did not match.

SGLang used Triton attention and PyTorch sampling for the same FlashInfer packaging issue. Consequently, this experiment compares two working end-to-end serving configurations, not identical low-level kernels or a scheduler-only change. The distinction is important: an end-to-end user cares whether a stack runs well, but an architectural study must control kernel selection more tightly.

## What this pilot establishes

The experiment supports four conclusions:

1. Both vLLM and SGLang can serve Qwen2.5-1.5B-Instruct reliably on an 8 GB RTX 4060.
2. Continuous batching substantially increases aggregate output throughput as concurrency rises.
3. Relative performance depends on whether the workload is dominated by prefill, decode, or a mixture of both.
4. Identical request traces and separate TTFT, latency, and throughput measurements are necessary for a meaningful comparison.

It does not establish that one framework is universally faster. The sample contains only one model, one GPU, one repetition, and a fixed execution order. The two frameworks also used different compatible kernel paths.

## Next steps

A stronger follow-up should add warmup requests, repeat every point several times, and alternate or randomize backend order. Reporting confidence intervals would separate persistent effects from run-to-run noise. The study should also expand to larger models that fit through quantization, vary request arrival patterns rather than using only closed-loop concurrency, and monitor GPU utilization, memory use, power, and server-side scheduling metrics.

Once the FlashInfer toolchain is aligned for both installations, a second experiment can compare matching attention and sampling paths. That would help distinguish framework scheduling behavior from kernel and packaging differences.

For now, the practical lesson is simple: on a consumer GPU, workload shape and concurrency matter at least as much as the serving framework name. A useful benchmark therefore needs to reproduce the traffic one actually expects to serve—not merely report one tokens-per-second number.
