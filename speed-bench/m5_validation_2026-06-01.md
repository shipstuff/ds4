# M5 SpecPrefill Validation - 2026-06-01

Early external validation at about 30k context showed the live-drafter
SpecPrefill path preserving decode speed while materially reducing turn
wall time.

| Mode | Decode | Effective prompt tok/s (TTFT throughput) | Turn wall time |
| --- | ---: | ---: | ---: |
| Baseline | ~29 tok/s | ~400 tok/s | ~80 s |
| Live-drafter SpecPrefill | ~31 tok/s | ~1100 tok/s | ~32 s |

Summary: at about 30k context, live-drafter SpecPrefill reduced turn wall
time by roughly 2.5x while keeping decode speed effectively neutral.

