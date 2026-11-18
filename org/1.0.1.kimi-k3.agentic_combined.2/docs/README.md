# Kimi K3 GB300 NVL72 submission

This line uses the NVIDIA Kimi K3 TensorRT-LLM and Dynamo implementation on a 12-node, 48-GPU GB300 NVL72 system. Individual points engage either 8 or 12 nodes as declared in each `point.yaml`.

The submitted implementation source is a clean snapshot of Git revision `eb72b6ae69aa63f2271f8dadcf71aee5466ad109`. The client runs on premises and connects to the colocated Dynamo OpenAI-compatible endpoint.

[Software disclosure](./software_disclosure.md) documents the serving stack,
images, client-to-frontend topology and supplemental measured loopback baseline.

## Provisioned power estimate

The [power descriptor](../results/Azure_ND_GB300-v6/system_power.json) uses a
component sum of **101.23 kW** for the 12 provisioned nodes (48 GPUs), replacing
the previous 90 kW calculation based on a non-Azure 135 kW rack figure.
It is **MLC Estimated Power**, not a measured draw or a Microsoft-published
total node/rack rating.

| Input | Value | Basis |
|---|---|---|
| CPUs | 24 x 300 W = 7,200 W | [Appendix D.2](https://github.com/mlcommons/endpoints_policies/blob/fcfb1e34468cf50c9d8fe4c286bf77c2b86aab6a/endpoints_rules.md#d2-processors), ARM CPUs with at most 128 cores; an MLC default, not an Azure CPU TDP |
| GPUs | 48 x 1,400 W = 67,200 W | [Microsoft Azure HPC article, Table 2](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/breaking-the-million-token-barrier-the-technical-achievement-of-azure-nd-gb300-v/4466080), ND GB300 v6 GPU power limit |
| NVLink scale-up fabric | 130 TB/s x 12/18 x 8 bit/byte x 5 pJ/bit = 3,466.666... W | [Azure rack bandwidth and node counts](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb300-v6-series), proportional allocation, and [Appendix D.1](https://github.com/mlcommons/endpoints_policies/blob/fcfb1e34468cf50c9d8fe4c286bf77c2b86aab6a/endpoints_rules.md#d1-scale-up-network) |
| Other components | 30% of major components = 23,360 W | Liquid-cooled overhead under Section 4.5.2 |
| Separate scale-out fabric | 0 W | Submitted inter-node serving path uses NVLink only, already included above |

Total before final rounding: **101,226.666... W**. No intermediate component
sum is rounded. The checker costs each node's components and then multiplies by
12, so the descriptor records **130/18 TB/s per node**, not the 12-node aggregate.
The same estimate is mirrored in the source submission template.

**Allocation assumption:** shared NVLink bandwidth/power is assigned
proportionally to 12 of 18 identical nodes. This is disclosed component-estimate
proration, not a claim that Section 4.5.2.1 expressly authorizes it: that section
explicitly addresses published rack ratings. Reviewer acceptance of this
allocation remains outstanding. The defaults and rules linked above are the
revision reviewed on 2026-11-03; their applicability to the submission's bound
ruleset must be confirmed during review. No reduced CPU/GPU power cap is assumed.
