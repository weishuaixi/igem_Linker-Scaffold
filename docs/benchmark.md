# Benchmark results and protocol

[Back to README](../README.md)

Commands below run from the repository root.

## Run the benchmark

With the canonical data supplied, run from this directory:

```shell
bash scripts/benchmark_offline.sh outputs/benchmark_v5
bash scripts/validate_expanded.sh outputs/validation_v5_expanded
bash scripts/diagnose_conditioning.sh outputs/conditioning_v5
```

The formal benchmark requests 48 test-partition motifs and 256 candidates per motif. It compares V5 with Uniform, Markov-1 and Markov-2 under matched length geometry. The two diagnostics use validation contexts to examine generation distributions and motif-conditioned prediction.

Use a new output directory for each run. The formal benchmark accepts `--resume` after the output-directory argument and verifies saved per-motif records. Inspect `benchmark_report.md`, `model_summary.csv`, candidate-level records and the run manifest. A smoke test validates the pipeline, not scientific performance.

## Benchmark results and interpretation

The following exploratory results are transcribed from development-stage server reports. See [model identity](provenance.md) for checkpoint hashes and evidence status.

### V4: passing format checks was insufficient

The historical formal benchmark used 48 motifs and 256 candidates per motif: 12,288 candidates per method, seed 42.

| Method | Reported valid rate | Unique rate | Edit diversity | Runtime, seconds |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 1.0000 | 1.0000 | 0.4694 | 80.0 |
| Markov-1 | 1.0000 | 1.0000 | 0.4725 | 80.9 |
| Markov-2 | 1.0000 | 1.0000 | 0.4713 | 80.9 |
| Transformer V4 | 1.0000 | 1.0000 | 0.3754 | 7,633.8 |

V4 linker composition entropy was 1.2636 bits, mean maximum linker run was 8.719, and the largest linker run was 30. Thus, 100% validity under the configured validator did not mean good composition, acceptable homopolymers, correct structure or functional linkers. Empty RNAfold fields were not successful structural evaluations. The report's sample-size grade was not a model-quality certification.

### V5: raw generation comparison

Expanded validation used 32 validation-cluster contexts and 64 draws per context: 2,048 candidates per method. Each method received the same context geometry. These were raw draws without hard GC/run filtering or deduplication. The natural-reference row contains only 32 real linker examples and is not a generated method.

| Method | Mean GC, % | Composition entropy, bits | Mean maximum run | Runs >6, % | Edit diversity | Elapsed, seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural reference | 50.86 | 1.9091 | 3.469 | 0.00 | — | — |
| Confidence, 16 steps | 82.90 | 1.2940 | 6.327 | 35.11 | 0.2640 | 1,164.2 |
| Random, 16 steps | 55.02 | 1.9036 | 3.342 | 0.59 | 0.4915 | 1,172.3 |
| Random, 4 steps | 54.64 | 1.9071 | 3.329 | 1.12 | 0.4909 | 295.6 |
| Single pass | 53.75 | 1.9189 | 3.246 | 0.49 | 0.4887 | 76.7 |
| Uniform | 50.14 | 1.9327 | 3.228 | 0.63 | 0.4953 | 4.8 |
| Markov-1 | 54.67 | 1.9102 | 3.485 | 1.03 | 0.4974 | 5.0 |
| Markov-2 | 54.56 | 1.9127 | 3.411 | 0.78 | 0.4965 | 5.0 |

Within-context uniqueness was 0.9932 for confidence decoding and 1.0 for the other variants. This is not a claim of global uniqueness across all contexts. Timings include the diagnostic work in that run and are not universal hardware throughput estimates.

Interpretation:

- Random decoding substantially improved on the defective confidence-selection variant.
- Random four-step decoding was approximately four times faster than random sixteen-step decoding, with similar aggregate composition and diversity. However, it produced more runs exceeding six: 23/2,048 versus 12/2,048. It is a speed/quality trade-off, not an across-the-board winner.
- Uniform and Markov baselines remained competitive on these descriptive sequence metrics and were much faster. Higher diversity alone is not evidence of better conditional design: random sequences can score highly.
- These results do not establish that V5 generates better functional linkers than the baselines. A weaker baseline must not be selected merely to make the model look better.

### V5: does the model use the motif?

A separate diagnostic used one context from each of 64 validation clusters, with all unknown linker bases masked, zero self-conditioning and only the first forward pass. It measured unsmoothed marginal NLL in nats per token, averaging clusters equally. This is **not joint sequence likelihood** and does not teacher-force unknown linker bases.

| Predictor / context | NLL, lower is better | Token accuracy |
| --- | ---: | ---: |
| Model, correct motif | 1.364003 | 30.68% |
| Model, shuffled motif | 1.371371 | 30.28% |
| Model, swapped motif | 1.382807 | 30.18% |
| Markov-1, correct motif | 1.386180 | 29.41% |
| Training-frequency predictor | 1.384931 | 29.13% |
| Uniform predictor | 1.386294 | Not used for interpretation |

Shuffling preserves motif composition; swapping does not. The Markov conditioning baseline uses conditional marginals, not access to the hidden true linker. Uniform argmax accuracy depends on tie-breaking, so the theoretical 25% random-draw accuracy should not be substituted for its recorded argmax result.

Paired differences below are `alternative NLL − model/correct NLL`; positive values favor the correctly conditioned model. Intervals are the reported unadjusted 95% cluster-bootstrap intervals, with 10,000 samples and seed 42.

| Comparison | Mean difference | Interval |
| --- | ---: | --- |
| Markov-1 minus model/correct | 0.022177 | [0.008129, 0.037980] |
| Model/shuffled minus model/correct | 0.007368 | [0.000778, 0.016470] |
| Model/swapped minus model/correct | 0.018804 | [0.003973, 0.034801] |
| Training-frequency minus model/correct | 0.020928 | [0.006683, 0.037159] |
| Uniform minus model/correct | 0.022291 | [0.009267, 0.036518] |

This supports a modest conditioning effect in the evaluated validation sample. Validation was reused for model selection and exploratory analysis; there was one ablation draw per context and several comparisons. The result is not an independent held-out confirmation and does not isolate long-range structural learning. Generation diversity and masked-token NLL answer different questions; neither table supersedes the other.
