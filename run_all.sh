#!/usr/bin/env bash
# AQWA-INUNDATION2D メッシュ生成スクリプト一括実行

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 引数チェック
if [ $# -eq 0 ]; then
    echo -e "${RED}エラー: YAMLファイルを指定してください${NC}"
    echo "使用例: bash run_all.sh yaml/Test_inun_riv_coupling.yaml"
    exit 1
fi

YAML_FILE=$1

# YAMLファイルの存在確認
if [ ! -f "$YAML_FILE" ]; then
    echo -e "${RED}エラー: ファイルが見つかりません: ${YAML_FILE}${NC}"
    exit 1
fi

# Conda環境のアクティベート（利用可能時のみ）
echo -e "${YELLOW}Conda環境を設定中...${NC}"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate AQWA-RISK 2>/dev/null || echo -e "${YELLOW}Warning: AQWA-RISK 環境が見つかりません。現在のPythonを使用します${NC}"
    else
        echo -e "${YELLOW}Warning: conda初期化スクリプトが見つかりません。現在のPythonを使用します${NC}"
    fi
else
    echo -e "${YELLOW}Warning: conda が見つかりません。現在のPythonを使用します${NC}"
fi

# conda activate 後は python が環境のインタプリタを指す（python3 は /usr/bin のままのことがある）
PYTHON_CMD=python
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    PYTHON_CMD=python3
fi
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo -e "${RED}エラー: python/python3 が見つかりません${NC}"
    exit 1
fi
echo -e "${YELLOW}使用 Python: $(command -v "$PYTHON_CMD")${NC}"

echo ""
echo "============================================================"
echo -e "${GREEN}AQWA-INUNDATION2D メッシュ生成開始${NC}"
echo "設定ファイル: ${YAML_FILE}"
echo "============================================================"
echo ""

# 01_mkINPUT_Face_Edge.py 実行
echo -e "${YELLOW}[1/2] 01_mkINPUT_Face_Edge.py を実行中...${NC}"
"$PYTHON_CMD" 01_mkINPUT_Face_Edge.py "$YAML_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 01_mkINPUT_Face_Edge.py でエラーが発生しました${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ 01_mkINPUT_Face_Edge.py が正常に完了しました${NC}"
echo ""

# 02_mkCELL.py 実行
echo -e "${YELLOW}[2/2] 02_mkCELL.py を実行中...${NC}"
"$PYTHON_CMD" 02_mkCELL.py "$YAML_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 02_mkCELL.py でエラーが発生しました${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ 02_mkCELL.py が正常に完了しました${NC}"
echo ""

# 完了メッセージ
echo "============================================================"
echo -e "${GREEN}🎉 すべての処理が正常に完了しました！${NC}"
echo "============================================================"
echo ""

# 出力ファイルの確認
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
    
    # ファイルサイズの表示
    if [ -f "$TXT_FILE" ]; then
        echo "生成されたメッシュファイル:"
        ls -lh "$TXT_FILE" "$OUTPUT_DIR"/*.{csv,gpkg} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
fi

exit 0
