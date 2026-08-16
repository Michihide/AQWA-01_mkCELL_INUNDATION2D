#!/usr/bin/env bash
# 第1段階メッシュ後処理: 土地利用別標高 → pit修正 → cell.bin 生成
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

YAML="${1:-yaml/Kuma_Hitoyoshi.yaml}"
DRY="${2:-}"

echo "=== Phase 1 mesh postprocess: $YAML ==="

if [[ "$DRY" == "--dry-run" ]]; then
  python3 04_refine_elevation_by_landuse.py "$YAML" --dry-run
  python3 03_fix_mesh_pits.py "$YAML" --dry-run
  echo "(dry-run: 02_mkCELL.py はスキップ)"
  exit 0
fi

python3 04_refine_elevation_by_landuse.py "$YAML"
python3 03_fix_mesh_pits.py "$YAML"
python3 02_mkCELL.py "$YAML"

BIN_NAME=$(python3 -c "
import yaml, os, sys
with open('$YAML') as f:
    c = yaml.safe_load(f)
pn = c['project']['name']
c = yaml.safe_load(open('$YAML'))
def repl(o,n):
    if isinstance(o,dict): return {k:repl(v,n) for k,v in o.items()}
    if isinstance(o,list): return [repl(v,n) for v in o]
    if isinstance(o,str): return o.replace('{project_name}',n)
    return o
c = repl(c, pn)
print(os.path.join(c['project']['base_dir'], c['output'].get('cell_txt_dir', c['output']['dir']), c['output'].get('cell_txt', f'{pn}.bin').replace('{project_name}', pn)))
")

echo ""
echo "生成: $BIN_NAME"
echo "配置例: cp \"$BIN_NAME\" ../02_Solver/inun2dh/01_Input_cell/"
