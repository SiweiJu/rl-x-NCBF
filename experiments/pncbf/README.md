# experiments/pncbf

All PNCBF (2026) experiment code and records live here (project plan §7).
Rules:

1. **One script per experiment**, committed here (sbatch / launch scripts, sweep
   drivers, eval drivers). No private local-only scripts.
2. **Every run records** the git commit hash, the full config, and wandb tags
   (`phase/P0`, `method/...`, `robot/...`, `task/...`). New runs go to the wandb
   project `pncbf-2026` (entity `catherineju-rwth-aachen-university`).
3. **Results register:** `RESULTS.md` maps every paper table/figure to its runs
   (wandb ids), the script, and the commit.
4. **Evaluation output** is always `eval_summary.json`.
5. Push regularly to branch `pncbf-v2`.

Layout:

| Path | Content |
|---|---|
| `reports/` | Task reports (`<TASK>.md`, e.g. `T01_T02.md`) |
| `paper_configs/` | Configs recovered from wandb `202601_paper` (T03) |
| `scripts/` | Launch / sbatch / sweep scripts |
| `RESULTS.md` | Results register |
