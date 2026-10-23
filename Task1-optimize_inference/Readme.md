# Optimize a deployment for a real use case
Take a real workload and tune a deployment until it hits the targets. This is the most common shape of the work, so it is a great first rep.
Pick a recent sizing request from #perf-agent. A typical use case might be an AI support agent answering customer questions in a help chat, where each turn includes the conversation history, retrieved documents, and account context. They come through regularly, and each one carries the three things you need: the use case, the traffic and token distribution, and the latency budget. A typical request looks like this:
Workload
Input tokens p50 10k / p90 13k / p95 24k · output tokens p50 40 / p90 70 / p95 90 · prompt cache hit rate p50 60% / p90 75%
Traffic
Median 5 RPS, peak 40 RPS
Latency budget
TTFT p50 500 ms, p95 900 ms, hard ceiling 5 s — TTFT and throughput are the priority metrics
Starting config
TP4, FP4 weights


Stand up a deployment in pyroworks, load it at that traffic shape, and see how close you get. Then change one thing at a time — tensor parallelism, replica count, precision, speculative decoding, prompt caching — and note what moved TTFT versus throughput. A short write-up of what you tried is a great first thing to share with the team.
Starting points: the perfagent skill in the fireworks repo (#perf-agent channel) · on-demand deployments guide · firectl

---

## Status: DONE

Sized and tuned **DeepSeek V4.1 Flash** (`deepseek-v4p1-flash`, 552B FP4) for an agentic-coding
request from `#perf-agent`: 55k prompt / 90% cached / 1k output, 3 QPS; targets decode >= 400 tok/s,
TTFT p50 <= 600 ms, e2e <= 3.1 s (reconciled from an impossible 1.5 s). 4 deployment experiments
+ 1 on-demand sweep in `pyroworks`.

**Recommendation:** Config A — 2xB300 FP4, inline tp2, 4 draft tokens, image 6.176.1,
4 replicas = **8 GPUs** (PASS: 408 tok/s, TTFT 204 ms mean / 292 ms p95, e2e 2.66 s).
Alternative: tp4-inline 2x4 GPUs (same 8 GPUs, -15 ms TTFT).

Key levers: prompt caching dominates (90% -> 50% = 8 -> 14 GPUs); TP width trades latency
not cost; disagg lost; draft depth/precision/image version not levers. 8 GPUs is the floor.

Done in cloud agent run `bc-9d4c477f-6e6c-45d2-8b3d-b3721bd037a7`; artifacts pulled from
`fw-ai/fireworks` branch `cursor/dsv41-flash-perf-writeup-37a7`.

## Files

| file | what |
|---|---|
| `artifacts/dsv41-flash-perf-20261022/WRITEUP-dsv41-flash-optimization.md` | The write-up to share |
| `artifacts/dsv41-flash-perf-20261022/ALL-EXPERIMENTS-SUMMARY.txt` | Results table, estimator accuracy, lever outcomes |
| `artifacts/.../validation-*` | Per-experiment metric series + summaries (`bmag9uba`, 2a=`xkpw07es`, 2b=`dicchhaw`, tp4-inline=`xpok1fmt`) |
| `artifacts/.../sweep-tp4-inline-comparison.txt` | Width vs image vs disagg estimate comparison |
| `artifacts/.../estimate-post-sweep.csv` | 35 ranked estimator rows incl. tp4-inline |
| `artifacts/.../estimate-tight-targets-limits.csv` | Limits block: 190 ms p95 TTFT below measured floor |
| `CHAT_NOTES.md` | Perf Agent skill study + Q&A from local chat |

## Experiments (`pyroworks`)

| exp | config | cache | GPUs | verdict |
|---|---|---|---|---|
| `bmag9uba` | tp2 2xB300 inline 6.176.1 | 90% | 8 | PASS |
| `xkpw07es` | tp2 2xB300 inline 6.176.1 | 50% | 14 | PASS |
| `dicchhaw` | tp2 2+2 disagg 6.176.1 | 90% | 12 | FAIL (decode 398.7) |
| `xpok1fmt` | tp4-inline 4xB300 7.21.0 | 90% | 8 | PASS |

Cleanup: branch `cursor/dsv41-flash-perf-writeup-37a7` deleted from `fw-ai/fireworks` (Oct 22, 2026); local copy in `artifacts/` is the only copy.
