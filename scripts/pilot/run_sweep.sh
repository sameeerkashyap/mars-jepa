#!/bin/sh
# Run one fixed experiment on local scratch; save data and results on its volume.
set -eu

: "${MARSWORLD_RUN_DIR:?Set a new persistent directory for this experiment}"
workers="${MARSWORLD_WORKERS:-8}"
data_dir="$MARSWORLD_RUN_DIR/data"
output_dir="$MARSWORLD_RUN_DIR/output"
mkdir -p "$MARSWORLD_RUN_DIR/provenance" "$output_dir"
source_commit="${MARSWORLD_REVISION:-local}"
if [ -f "$MARSWORLD_RUN_DIR/provenance/source_commit.txt" ]; then
    test "$(cat "$MARSWORLD_RUN_DIR/provenance/source_commit.txt")" = "$source_commit"
fi
cp data/specification/pilot/* "$MARSWORLD_RUN_DIR/provenance/"
python -m pip freeze > "$MARSWORLD_RUN_DIR/provenance/packages.txt"
printf '%s\n' "$source_commit" > "$MARSWORLD_RUN_DIR/provenance/source_commit.txt"

# Keep scientific settings fixed: 3-year records, Tier 2 at 8-10 W/m².
python -u -m scripts.pilot.gen_data --out "$data_dir" --workers "$workers" --tier2-mag 8 10
python - "$data_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert len(manifest["trajectories"]) == 210
assert manifest["splits"]["main/test_tier2"]["mag_range"] == [8.0, 10.0]
assert all(t["n_time"] == 2006 for t in manifest["trajectories"])
with (root / "SHA256SUMS").open("w") as output:
    for trajectory in manifest["trajectories"]:
        path = root / trajectory["file"]
        output.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {trajectory['file']}\n")
print("Dataset ready: 210 trajectories; hashes recorded.", flush=True)
PY

python -u -m scripts.pilot.sweep --data-dir "$data_dir" --output-dir "$output_dir" \
    --device cpu --workers "$workers"
python -u -m scripts.pilot.analyze --results "$output_dir/results.csv" \
    --run-root "$output_dir/runs" --output-dir "$output_dir"
touch "$MARSWORLD_RUN_DIR/SUCCESS"
