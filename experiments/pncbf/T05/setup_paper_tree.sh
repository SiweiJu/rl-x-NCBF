#!/bin/bash
# Create the fixed, read-only code tree for T05 (run once on the IAS login node).
# Paper Fig. 2 runs were trained at cacffe7 (ours, H=1) and 614c6b9 (PPO); the two commits differ only in
# commented-out logging lines, so all three methods use cacffe7 (== ncbf-q for rl_x/algorithms + rl_x/runner).
set -euo pipefail
COMMIT=${COMMIT:-cacffe7a2f7aeac37047a76c661298b230cb8f5d}
TREE=/home/ju/repo/pncbf/paper-${COMMIT:0:7}
if [[ ! -d $TREE ]]; then
  git clone -q git@github.com:SiweiJu/rl-x-NCBF.git "$TREE"
  git -C "$TREE" -c advice.detachedHead=false checkout -q "$COMMIT"
fi
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
chmod -R a-w "$TREE/rl_x"   # code is read-only; experiments/runs and wandb/ (gitignored) stay writable
echo "ok: $TREE @ $COMMIT"
