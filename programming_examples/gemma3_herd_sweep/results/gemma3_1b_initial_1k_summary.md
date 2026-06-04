| Target | Metric | Local | Paper | Delta | Classification | Note |
| --- | --- | --- | --- | --- | --- | --- |
| prefill_ttft_seconds_gemma3_1b_cpu_1024 | prefill_ttft_seconds | 1.43077 | 4.06 | 64.76% | EXPLAINED_DEVIATION |  |
| decode_tps_gemma3_1b_cpu_1024 | decode_tps | 12.4003 | 41.9 | 70.40% | EXPLAINED_DEVIATION |  |
| prefill_ttft_seconds_gemma3_1b_igpu_1024 | prefill_ttft_seconds | 0.527178 | 0.51 | 3.37% | PAPER_MATCH |  |
| decode_tps_gemma3_1b_igpu_1024 | decode_tps | 13.738 | 38 | 63.85% | EXPLAINED_DEVIATION |  |
| prefill_ttft_seconds_gemma3_1b_npu_1024 | prefill_ttft_seconds | n/a | 0.95 | n/a | LOCAL_FAIL | BLOCKED_EXECUTION_NOT_IMPLEMENTED |
| decode_tps_gemma3_1b_npu_1024 | decode_tps | n/a | 41.1 | n/a | LOCAL_FAIL | BLOCKED_EXECUTION_NOT_IMPLEMENTED |
