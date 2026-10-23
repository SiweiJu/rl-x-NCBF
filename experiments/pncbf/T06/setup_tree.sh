#!/bin/bash
# Create the fixed, read-only code tree for T06 (run once on the IAS login node, per commit):
#   COMMIT=<full sha on pncbf-v2> experiments/pncbf/T06/setup_tree.sh
# The tree is /home/ju/repo/pncbf/<TREE_PREFIX>-<commit7> (v2 for T06b, paper-fixed for T06a); t06.sbatch refuses to run from anything else.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the full sha to train}
TREE=/home/ju/repo/pncbf/${TREE_PREFIX:-v2}-${COMMIT:0:7}
if [[ ! -d $TREE ]]; then
  git clone -q git@github.com:SiweiJu/rl-x-NCBF.git "$TREE"
  git -C "$TREE" -c advice.detachedHead=false checkout -q "$COMMIT"
fi
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
chmod -R a-w "$TREE/rl_x"   # code is read-only; experiments/runs and wandb/ (gitignored) stay writable
mkdir -p /home/ju/repo/pncbf/T06-logs /home/ju/.cache/jax_pncbf
echo "ok: $TREE @ $COMMIT"
