# Camera-Edge Benchmark Report

Runs analyzed: 8

## Summary Table

| run_id | profile | net | comp | sender depth fps | receiver depth fps | latency mean ms | comp ms | decomp ms | tx mbps | rx mbps | stale | gaps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260404_local_baseline | local_baseline | local | lz4 | 14.086 | - | - | 2.895 | - | - | - | - | - |
| 20260404_packet_0900 | packet_overhead | wired | lz4 | 14.051 | - | - | 2.872 | - | - | - | - | - |
| 20260404_wired_lz4 | wired_lz4 | wired | lz4 | 13.724 | - | - | 2.884 | - | - | - | - | - |
| 20260404_wired_none | wired_none | wired | none | 14.088 | - | - | 0.007 | - | - | - | - | - |
| 20260404_wired_zstd | wired_zstd | wired | zstd | 14.275 | - | - | 4.758 | - | - | - | - | - |
| 20260404_wireless_lz4 | wireless_lz4 | wireless | lz4 | 14.623 | - | - | 2.858 | - | - | - | - | - |
| 20260404_wireless_none | wireless_none | wireless | none | 13.659 | - | - | 0.007 | - | - | - | - | - |
| 20260404_wireless_zstd | wireless_zstd | wireless | zstd | 14.856 | - | - | 4.712 | - | - | - | - | - |

## Wired vs Wireless Deltas

- `none`: wired-vs-wireless receiver FPS delta = -, wireless extra latency = - ms.
- `lz4`: wired-vs-wireless receiver FPS delta = -, wireless extra latency = - ms.
- `zstd`: wired-vs-wireless receiver FPS delta = -, wireless extra latency = - ms.
