> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Shared SFlow helpers

`tools/` holds helper scripts that the recipes' workflows run on the cluster. Each
model's runbook (`src/<model>/README.md`) says how to launch its points; none of
these is launched on its own.

| Script | Purpose |
| --- | --- |
| `run_with_env.py` | Prints shell `export` statements from flat YAML environment files; the workflows `eval` its output. |
| `generate_master_yaml.py` | Writes the `trtllm-serve disaggregated` frontend configuration from the allocated prefill and decode servers. |
| `generate_endpoint_yaml.py` | Writes the client's endpoint configuration with the frontend URLs of the run. |
| `setup_harmony_cache.sh` | Prepares the `openai_harmony` vocabulary cache for compute nodes without access to its download host. |

This release ships only the helpers its models use. When you submit a line whose recipe
calls these helpers, copy `src/sflow/` into the submission's `src/` next to
`src/<model>/`, so the submitted source can reproduce a run.
