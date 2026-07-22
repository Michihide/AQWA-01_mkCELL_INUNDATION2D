#!/usr/bin/env bash
# AQWA-INUNDATION2D メッシュ生成スクリプト一括実行

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}エラー: YAMLファイルを指定してください${NC}"
    echo "使用例: bash run_all.sh yaml/Test_inun_riv_coupling.yaml"
    exit 1
fi

YAML_FILE=$1
MODE="${2:-auto}"

if [ ! -f "$YAML_FILE" ]; then
    echo -e "${RED}エラー: ファイルが見つかりません: ${YAML_FILE}${NC}"
    exit 1
fi

PYTHON_CMD=python3
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo -e "${RED}エラー: python3 が見つかりません。リポジトリルートで nix develop を実行してください${NC}"
    exit 1
fi
echo -e "${YELLOW}使用 Python: $(command -v "$PYTHON_CMD")${NC}"

if [ "${MKCELL_USE_NATIVE:-0}" = "1" ] && [ -d native ]; then
    make -C native -s 2>/dev/null || echo -e "${YELLOW}Warning: native build skipped${NC}"
fi

USE_PARALLEL=0
if [ "$MODE" = "blocks" ]; then
    USE_PARALLEL=1
elif [ "$MODE" = "auto" ]; then
    if grep -q "enabled: true" "$YAML_FILE" 2>/dev/null && grep -q "^parallel:" "$YAML_FILE" 2>/dev/null; then
        USE_PARALLEL=1
    fi
fi

echo ""
echo "============================================================"
echo -e "${GREEN}AQWA-INUNDATION2D メッシュ生成開始${NC}"
echo "設定ファイル: ${YAML_FILE}"
echo "モード: $([ "$USE_PARALLEL" -eq 1 ] && echo blocks+merge || echo single)"
echo "============================================================"
echo ""

if [ "$USE_PARALLEL" -eq 1 ]; then
    echo -e "${YELLOW}[1/3] run_blocks.py を実行中...${NC}"
    "$PYTHON_CMD" run_blocks.py "$YAML_FILE" || exit 1
    echo -e "${YELLOW}[2/3] merge_blocks.py を実行中...${NC}"
    "$PYTHON_CMD" merge_blocks.py "$YAML_FILE" || exit 1
    STEP_CELL="[3/3]"
else
    echo -e "${YELLOW}[1/2] 01_mkINPUT_Face_Edge.py を実行中...${NC}"
    "$PYTHON_CMD" 01_mkINPUT_Face_Edge.py "$YAML_FILE" || exit 1
    STEP_CELL="[2/2]"
fi

echo ""
echo -e "${YELLOW}${STEP_CELL} 02_mkCELL.py を実行中...${NC}"
"$PYTHON_CMD" 02_mkCELL.py "$YAML_FILE" || exit 1

echo ""
echo "============================================================"
echo -e "${GREEN}🎉 すべての処理が正常に完了しました！${NC}"
echo "============================================================"
echo ""

PROJECT_NAME=$(grep "name:" "$YAML_FILE" | head -1 | awk -F'"' '{print $2}')
if [ -n "$PROJECT_NAME" ]; then
    OUTPUT_DIR="Cell/${PROJECT_NAME}/output_gpkg_csv"
    TXT_FILE="Cell/${PROJECT_NAME}/${PROJECT_NAME}.bin"

    echo "出力ファイル:"
    echo "  - ${OUTPUT_DIR}/face.csv"
    echo "  - ${OUTPUT_DIR}/face.gpkg"
    echo "  - ${OUTPUT_DIR}/edge.csv"
    echo "  - ${OUTPUT_DIR}/edge.gpkg"
    echo "  - ${TXT_FILE}"
    echo ""

    if [ -f "$TXT_FILE" ]; then
        echo "生成されたメッシュファイル:"
        ls -lh "$TXT_FILE" "$OUTPUT_DIR"/*.{csv,gpkg} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
fi

exit 0
