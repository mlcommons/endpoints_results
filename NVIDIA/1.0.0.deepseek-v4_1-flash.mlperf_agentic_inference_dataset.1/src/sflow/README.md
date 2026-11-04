# Shared SFlow helpers

`tools/` holds helper scripts that the recipes' workflows run on the cluster. Each
model's runbook (`src/<model>/README.md`) says how to launch its points; none of
these is launched on its own.

| Script | Purpose |
| --- | --- |
| `run_with_env.py` | Prints shell `export` statements from flat YAML environment files; the workflows `eval` its output. |
| `generate_endpoint_yaml.py` | Writes the client's endpoint configuration with the frontend URLs of the run. |
