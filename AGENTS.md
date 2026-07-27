# FlowDAgger Repository Guide

- Keep the primary implementation under `flowdagger_pi05/` and preserve its pinned
  `openpi` submodule unless a code change explicitly requires an update.
- Use `flowdagger_pi05/train_flowdagger.py` as the native training entry point.
- Preserve upstream defaults when changing only environment, storage, logging, or
  orchestration behavior.
- Keep repository-native commands and implementation notes in this repository.
  Workspace experiment YAML and cross-run conclusions belong in the parent
  `vla-post-train` workspace.
- Treat MetaWorld Assembly smoke results as engineering evidence only. Read the
  parent method runbook before launching a workspace experiment.
