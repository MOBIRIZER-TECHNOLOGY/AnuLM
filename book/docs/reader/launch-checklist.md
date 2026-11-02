# Launch checklist

For the revised multilingual/multimodal package, read [Current companion workflow](current-workflow.md), [Language coverage](language-coverage.md) and [Course route](course-route.md). This 2 November 2026 update does not replace historical measurements or certify a production release.

This is an acceptance worksheet, not a claim that the current server passes. Choose one concrete use case and record evidence for each applicable item. A local demo and an operational service have different finish lines.

## Define the service before measuring it

Write down the intended users and task, languages, maximum prompt and answer lengths, expected traffic, target hardware, acceptable response time, and a task-quality threshold. State what the system should do when it cannot answer. Set numerical targets before a release candidate is evaluated.

## Book release

| Gate | Evidence required | Current state |
|---|---|---|
| Complete editions | 41 chapters, six appendices, reader guide, working contents and figures in HTML/PDF | Rebuild and inspect each edition before release. |
| Public reproducibility | An immutable public revision containing every required file; tested fresh-clone commands | Historical pin and optional course labs remain unpublished. |
| Technical review | Reviewed records for every chapter, plus resolved findings | Nine chapters remain drafted in the supplied review records. |
| Reader usability | New readers complete setup, one example, and one exercise using only the instructions | Needs a human reader trial. |
| Publication details | Author, edition, text permissions, artifact version and known limitations | Original book prose: CC BY 4.0, selected by the author on 2 November 2026. Preserve third-party notices; other publication gates remain open. |

## Model and data

- Keep a versioned record of data sources, cleaning, tokenizer, split rules, exclusions, and artifact checksums.
- Deduplicate or group equivalent examples before splitting. Rebuild the contaminated chat split and rerun affected evaluations; do not silently reuse the old scores.
- Fix a task evaluation set and success threshold. Report the chosen checkpoint's starting-task measurements, failure cases, sample sizes, and uncertainty where appropriate.
- Test the chosen precision and exported artifact against the native model on relevant tasks. Record observed differences, rather than assuming matching configuration means matching output.
- Document supported languages, context length, intended use, and observed limitations.

## Request correctness

- Demonstrate that each accepted generation option changes or controls the intended behaviour. Reject unsupported options explicitly.
- Test the supported conversation format. The historical API ignored earlier turns; current source rejects multi-message/history requests explicitly.
- Require both streamed and returned output to agree for EOS, token limits, UTF-8 boundaries, and stop strings split across tokens. Include cancellation and client disconnects.
- Verify prompt truncation, maximum output length, malformed input, and error messages.

Chapter 38 retains its historical streaming findings. The current companion fixes EOS/UTF-8/stop handling, deterministic greedy decoding and option validation, with unit and real-checkpoint API smoke checks. This closes those specific code findings only. Concurrent load, cancellation/disconnect behavior and recovery still require deployment tests.

## Operations

| Check | Result to record |
|---|---|
| Authentication | An unauthenticated request is rejected when the deployment requires authentication; configuration is checked at startup. |
| Resource limits | Prompt/output limits, bounded queue, request timeout, concurrency policy, and rate limits appropriate to the workload. |
| Load behaviour | Time to first token, per-token latency, total latency, throughput, errors, and peak memory under representative concurrent traffic. Report tail latency as well as averages. |
| Overload | What happens when the queue or device is full, and how the client can retry. |
| Monitoring | Request counts, latency, failures, queue depth, device memory, and model/artifact version. Record how sensitive content is handled in logs. |
| Recovery | Restart during active work, failed model load, disconnect, and out-of-memory recovery. |
| Rollback | Restore a known model/tokenizer/configuration bundle and demonstrate that the previous version serves correctly. |
| Task-specific misuse | Relevant adversarial and out-of-scope cases, including retrieved instructions where retrieval is enabled. Record limitations; do not assume a prompt provides a security boundary. |

Quantisation, continuous batching, paging, and multiple GPUs are optional engineering choices. Adopt them when measurements justify them. Calculating how many KV caches fit beside the weights does not establish a safe concurrency limit.

## Decision record

For each gate, record **pass**, **fail**, **not run**, or **not applicable**, the evidence location, the responsible person, and the tested version. Give a reason for every not-applicable entry. Missing artifacts and skipped tests remain visible.

Launch only for the declared workload after its applicable gates pass. Update the model card and book errata with remaining limitations. Keep human editorial review and reader trials separate from automated checks.
