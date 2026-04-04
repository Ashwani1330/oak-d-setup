# Camera-Edge Benchmark Report

Runs analyzed: 8

## Key Takeaways

- Highest receiver throughput in the main matrix: `wired_lz4` at - FPS.
- `none` wired-vs-wireless receiver FPS delta: - (positive means wired is faster).
- `lz4` wired-vs-wireless receiver FPS delta: - (positive means wired is faster).
- `zstd` wired-vs-wireless receiver FPS delta: - (positive means wired is faster).

## Main Matrix

| run_id | net | comp | sender fps | receiver fps | fps efficiency | depth latency ms | rgb latency ms | usable fused latency ms | payload mbps | packet ct | comp ms | send ms | decomp ms | source skew ms | bottleneck |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260404_wired_lz4 | wired | lz4 | 14.088 | - | - | - | - | - | 17.758 | 134.860 | 2.897 | 1.733 | - | - | balanced |
| 20260404_wired_none | wired | none | 14.084 | - | - | - | - | - | 57.687 | 437.000 | 0.006 | 31.940 | - | - | packetization-or-network-bound |
| 20260404_wired_zstd | wired | zstd | 14.238 | - | - | - | - | - | 12.916 | 97.182 | 4.731 | 1.361 | - | - | balanced |
| 20260404_wireless_lz4 | wireless | lz4 | 14.722 | - | - | - | - | - | 18.532 | 134.640 | 2.879 | 4.168 | - | - | balanced |
| 20260404_wireless_none | wireless | none | 12.879 | - | - | - | - | - | 52.751 | 437.000 | 0.007 | 26.152 | - | - | packetization-or-network-bound |
| 20260404_wireless_zstd | wireless | zstd | 14.118 | - | - | - | - | - | 12.818 | 97.245 | 4.740 | 3.110 | - | - | balanced |

## Secondary Runs

| run_id | profile | receiver fps | latency ms | payload mbps | packet ct | comp ms | send ms | bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 20260404_local_baseline | local_baseline | - | - | 17.305 | 134.914 | 2.930 | 2.589 | balanced |
| 20260404_packet_0900 | packet_overhead | - | - | 17.337 | 181.351 | 2.895 | 2.150 | balanced |

## Recommendations

- Best balanced default: `wireless_zstd` (receiver FPS -, usable fused latency - ms, payload 12.818 Mbps, bottleneck `balanced`).
- Best latency-first option: `wireless_zstd` at - ms usable fused latency.
- Best bandwidth-efficient option: `wireless_zstd` at 12.818 Mbps estimated depth payload.
- Highest throughput option: `wired_lz4` at - FPS.
- `wired_none` looks packet/network-heavy: packet count 437.000, send time 31.940 ms.
- `wireless_none` looks packet/network-heavy: packet count 437.000, send time 26.152 ms.

## Caveats

- `latency_ms_*` is only trustworthy when sender clock sync succeeded; the current runs appear synced, but future unsynced runs should be treated as invalid.
- `usable_fused_latency_ms_mean` is the new headline metric for fresh runs. It measures when the matched RGB+depth pair is actually ready on the receiver, using sender-side RGB metadata plus the depth packet timestamp.
- `fused_source_skew_ms_mean` measures how far apart the sender timestamps of the matched RGB and depth samples are. High skew means the pair is available but temporally weak.
- `practical_latency_ms_est` now prefers direct fused-ready latency. For older runs without RGB metadata it falls back to the earlier heuristic so historical comparisons still render.
- `depth_payload_mbps_est` is computed from sender depth payload only; it does not include RTSP RGB overhead, so it is best used for compression comparisons rather than total link budgeting.
- Some rows still rely on fallback latency logic because they were recorded before RGB metadata was added; rerun those scenarios if you want apples-to-apples fused-latency comparisons.
