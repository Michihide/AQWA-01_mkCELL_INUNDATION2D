#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQWA-INUNDATION2D メッシュファイル(cell.bin)作成スクリプト

face.csvとedge.csvからcell.bin（バイナリ形式）を生成します。
座標の丸め誤差を避けるためnode_idベースのアルゴリズムを使用しています。

出力フォーマット（Fortran stream unformatted 互換バイナリ、リトルエンディアン）:
  ヘッダー: face_gpkg_path(1000B), edge_gpkg_path(1000B),
            total_face(i4), total_edge(i4), total_face_index(i4)
  セル × total_face:
    face_number, total_edge_in_face (i4×2)
    x, y, A, bl, bld_ratio, bld_peri (r8×6)
    A_lu10..A_lu255 (r8×13, 固定コード順)
    A_soil1..A_soil17 (r8×17, 固定コード順)
    dummy_face (i4)
    辺 × total_edge_in_face:
      en, bc_flag, fn_self, fn_adj (i4×4)
      A_CV_edge, dl, dxdl, dydl, weight_self, weight_adj, weight_face (r8×7)
      vertex_number_st, vertex_number_en (i4×2)
      x_vertex_st, y_vertex_st, bl_vertex_st,
        x_vertex_en, y_vertex_en, bl_vertex_en,
        x_edge_cnt, y_edge_cnt, bl_edge_cnt, edge_length (r8×10)

主な特徴:
  - 土地利用面積を固定13コードで格納（Fortranの変数宣言順に一致）
  - 土壌面積を固定17コードで格納（soil_1_area..soil_17_area）
  - ヘッダーに total_edge を保持 → Fortran 側の2パス読み込みが1パスに簡略化
  - 辺データ全フィールドを格納（read_face_data でそのまま読み込み可能）

使用例:
  python 02_mkCELL.py yaml/Kinu_Joso.yaml
"""

import argparse
import struct
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Fortran の変数宣言順（A_lu10_face .. A_lu255_face）に合わせた固定土地利用コードリスト
LANDUSE_CODES = ['10', '20', '50', '60', '70', '91', '92', '100', '110', '140', '150', '160', '255']
# 土壌コードの固定リスト（soil_1_area .. soil_17_area）
SOIL_CODES = [str(i) for i in range(1, 18)]
import os


def replace_project_name(obj, project_name):
    """再帰的に辞書・リスト内の{project_name}を置き換える"""
    if isinstance(obj, dict):
        return {k: replace_project_name(v, project_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_project_name(item, project_name) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("{project_name}", project_name)
    else:
        return obj


def read_config(config_file):
    """YAMLファイルから設定を読み込む"""
    print(f"\n{'='*60}")
    print(f"設定ファイルを読み込み中: {config_file}")
    print(f"{'='*60}")
    
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # プロジェクト名を取得して、config内の{project_name}を置き換え
    project_name = config["project"]["name"]
    config = replace_project_name(config, project_name)
    
    base_dir = config["project"]["base_dir"]
    output_dir = os.path.join(base_dir, config["output"]["dir"])
    
    # cell.txtの出力ディレクトリとファイル名を取得
    cell_txt_dir = config["output"].get("cell_txt_dir", output_dir)
    cell_txt_filename = config["output"].get("cell_txt", "cell.bin")
    cell_txt_dir_full = os.path.join(base_dir, cell_txt_dir) if not os.path.isabs(cell_txt_dir) else cell_txt_dir
    
    print(f"プロジェクト名: {project_name}")
    print(f"CSV出力ディレクトリ: {output_dir}")
    print(f"cell.bin出力ディレクトリ: {cell_txt_dir_full}")
    print(f"cell.binファイル名: {cell_txt_filename}")
    
    return {
        'face_csv': os.path.join(output_dir, 'face.csv'),
        'edge_csv': os.path.join(output_dir, 'edge.csv'),
        'face_gpkg': os.path.abspath(os.path.join(output_dir, 'face.gpkg')),
        'edge_gpkg': os.path.abspath(os.path.join(output_dir, 'edge.gpkg')),
        'cell_txt': os.path.join(cell_txt_dir_full, cell_txt_filename),
        'output_dir': output_dir,
        'cell_txt_dir': cell_txt_dir_full
    }


def read_face_csv(face_csv_path):
    """face.csvを読み込む"""
    print(f"\n[1] face.csvを読み込み中...")
    df_face = pd.read_csv(face_csv_path)
    print(f"   セル数: {len(df_face)}")
    print(f"   カラム: {list(df_face.columns[:10])}...")
    return df_face


def read_edge_csv(edge_csv_path):
    """edge.csvを読み込む"""
    print(f"\n[2] edge.csvを読み込み中...")
    df_edge = pd.read_csv(edge_csv_path)
    print(f"   辺数: {len(df_edge)}")
    print(f"   カラム: {list(df_edge.columns)}")
    return df_edge


def create_lines_from_sides(df_side, df_face):
    """
    辺データからラインデータを作成
    
    各セルの辺を同じLNでグループ化し、ペアにしてラインを作成します。
    node_idを使用して座標の丸め誤差を回避します。
    """
    print(f"\n[3] 辺をラインに変換中...")
    
    lines = []
    
    # CNでグループ化して各セルを処理
    for cn, cell_sides in df_side.groupby('CN'):
        # セル情報を取得
        cell_info = df_face[df_face['CN'] == cn].iloc[0]
        
        # 各セル内でLNごとにグループ化
        for ln, ln_sides in cell_sides.groupby('LN'):
            if len(ln_sides) != 2:
                print(f"   警告: CN={cn}, LN={ln} の辺数が2ではありません（{len(ln_sides)}個）")
                continue
            
            # StartNodeとEndNodeの情報を取得（どちらの行も同じ値を持つはず）
            start_node_id = int(ln_sides.iloc[0]['StartNode'])
            end_node_id = int(ln_sides.iloc[0]['EndNode'])
            peri = int(ln_sides.iloc[0]['ID'])
            z_cnt = ln_sides.iloc[0]['merge2_mn']
            
            # node_idがStartNodeの行を始点、EndNodeの行を終点として取得
            start_side = ln_sides[ln_sides['node_id'] == start_node_id]
            end_side = ln_sides[ln_sides['node_id'] == end_node_id]
            
            if len(start_side) != 1 or len(end_side) != 1:
                print(f"   警告: CN={cn}, LN={ln} の始点/終点が正しく取得できません")
                continue
            
            start_side = start_side.iloc[0]
            end_side = end_side.iloc[0]
            
            # ライン情報を作成
            line = {
                'CN': cn,
                'LN': ln,
                'peri': peri,
                'CN_end': 0,  # 後で隣接セルを探索して設定
                'X_bgn': start_side['xcoord'],
                'Y_bgn': start_side['ycoord'],
                'X_end': end_side['xcoord'],
                'Y_end': end_side['ycoord'],
                'Z_bgn': start_side['merge1_mn'],
                'Z_end': end_side['merge1_mn'],
                'Z_cnt': z_cnt,  # 辺の中心の標高（merge2_mn）
                'node_bgn': start_node_id,
                'node_end': end_node_id
            }
            
            lines.append(line)
    
    df_lines = pd.DataFrame(lines)
    print(f"   作成されたライン数: {len(df_lines)}")
    
    return df_lines


def sort_sides_circular(cell_sides):
    """
    セル内の辺を循環順に並び替え
    
    node_idを使用して、終点と次の始点が一致するように並び替えます。
    """
    sides = cell_sides.copy().reset_index(drop=True)
    
    if len(sides) == 0:
        return sides
    
    # 並び替え済みリスト
    sorted_list = [sides.iloc[0]]
    remaining = sides.iloc[1:].copy()
    
    # 循環的に次の辺を探す
    while len(remaining) > 0:
        current_node = sorted_list[-1]['node_id']
        
        # 現在のnode_idと同じnode_idを持つ次の辺を探す
        # （同じnode_idの辺が複数ある場合は、まだ追加されていないものを選ぶ）
        next_idx = None
        for idx, row in remaining.iterrows():
            if row['node_id'] == current_node:
                next_idx = idx
                break
        
        if next_idx is not None:
            sorted_list.append(remaining.loc[next_idx])
            remaining = remaining.drop(next_idx)
        else:
            # 循環が途切れた場合、残りの最初の辺を追加
            sorted_list.append(remaining.iloc[0])
            remaining = remaining.iloc[1:]
    
    return pd.DataFrame(sorted_list).reset_index(drop=True)


def find_adjacent_cells(df_lines):
    """
    隣接セルを探索してCN_endを設定
    
    同じnode_idペアを持つラインを探します（順序を問わず）。
    （座標の丸め誤差を避けるため）
    """
    print(f"\n[4] 隣接セルを探索中...")
    
    df_lines = df_lines.copy()
    df_lines['CN_end'] = 0
    
    # 内部の辺のみを対象（peri=0）
    internal_lines = df_lines[df_lines['peri'] == 0].index
    
    # node_idペアでインデックスを作成（正規化されたペアを使用）
    # 正規化: 小さい方のnode_idを先にする
    node_pair_dict = {}
    for idx, row in df_lines.iterrows():
        if row['peri'] == 0:
            n1, n2 = int(row['node_bgn']), int(row['node_end'])
            node_pair = (min(n1, n2), max(n1, n2))
            if node_pair not in node_pair_dict:
                node_pair_dict[node_pair] = []
            node_pair_dict[node_pair].append(idx)
    
    # 各内部ラインについて、同じnode_idペアを持つ別のセルのラインを探す
    matched_count = 0
    for idx in internal_lines:
        row = df_lines.loc[idx]
        n1, n2 = int(row['node_bgn']), int(row['node_end'])
        node_pair = (min(n1, n2), max(n1, n2))
        
        if node_pair in node_pair_dict:
            for match_idx in node_pair_dict[node_pair]:
                match_row = df_lines.loc[match_idx]
                
                # 自分自身ではなく、異なるセルの場合
                if match_row['CN'] != row['CN']:
                    df_lines.at[idx, 'CN_end'] = match_row['CN']
                    matched_count += 1
                    break
    
    print(f"   マッチしたライン数: {matched_count}")
    
    return df_lines


def sort_lines_counterclockwise(cell_lines):
    """
    セルの辺を反時計回り（CCW）順序に並べ替える
    
    平面直角座標系（右手座標系）の標準に従い、辺を反時計回りに配置します。
    辺の終点が次の辺の始点になるように並べ替えます。
    """
    if len(cell_lines) == 0:
        return cell_lines
    
    # 最初の辺を選択（どれでもOK）
    sorted_lines = [cell_lines.iloc[0]]
    remaining = cell_lines.iloc[1:].copy()
    
    # 残りの辺を順番に繋げていく
    while len(remaining) > 0:
        current_line = sorted_lines[-1]
        current_end = current_line['node_end']
        
        # 現在の辺の終点が始点になっている辺を探す
        next_line = remaining[remaining['node_bgn'] == current_end]
        
        if len(next_line) == 0:
            # 連続する辺が見つからない場合（エラー）
            print(f"   警告: CN={current_line['CN']} で連続する辺が見つかりません")
            # 残りの辺をそのまま追加
            sorted_lines.extend([remaining.iloc[i] for i in range(len(remaining))])
            break
        
        # 見つかった辺を追加
        next_line_idx = next_line.index[0]
        sorted_lines.append(remaining.loc[next_line_idx])
        remaining = remaining.drop(next_line_idx)
    
    # DataFrameに変換
    return pd.DataFrame(sorted_lines).reset_index(drop=True)


def write_cell_bin(df_face, df_lines, output_path, face_gpkg_path, edge_gpkg_path):
    """
    バイナリ形式でセルデータを出力（Fortran stream unformatted 互換）

    フォーマット（リトルエンディアン）:
      ヘッダー:
        face_gpkg_path      CHARACTER(1000) = 1000 bytes（末尾スペースパディング）
        edge_gpkg_path      CHARACTER(1000) = 1000 bytes
        total_face          INTEGER(4)      =    4 bytes
        total_edge          INTEGER(4)      =    4 bytes
        total_face_index    INTEGER(4)      =    4 bytes
      セルレコード × total_face（各 300 + 160×辺数 bytes）:
        face_number, total_edge_in_face     INTEGER(4)×2  =  8 bytes
        x, y, A, bl, bld_ratio, bld_peri   REAL(8)×6     = 48 bytes
        A_lu10 .. A_lu255                  REAL(8)×13    =104 bytes
        A_soil1 .. A_soil17                REAL(8)×17    =136 bytes
        dummy_face（CalMesh）               INTEGER(4)    =  4 bytes
        辺レコード × total_edge_in_face（各 160 bytes）:
          en, bc_flag, fn_self, fn_adj      INTEGER(4)×4  = 16 bytes
          A_CV_edge, dl, dxdl, dydl,
            weight_edge_self, weight_edge_adj, weight_face
                                            REAL(8)×7     = 56 bytes
          vertex_number_st, vertex_number_en INTEGER(4)×2 =  8 bytes
          x_vertex_st, y_vertex_st, bl_vertex_st,
            x_vertex_en, y_vertex_en, bl_vertex_en,
            x_edge_cnt, y_edge_cnt, bl_edge_cnt, edge_length
                                            REAL(8)×10    = 80 bytes
    """
    print(f"\n[5] cell.bin（バイナリ）を出力中...")

    face_dict   = df_face.set_index('CN').to_dict('index')
    sorted_cns  = sorted(df_face['CN'].unique())

    # CCW 順ラインデータをキャッシュ（face ループを2回回さないため）
    cell_lines_cache = {}
    for cn in sorted_cns:
        cell_lines_cache[cn] = sort_lines_counterclockwise(
            df_lines[df_lines['CN'] == cn]
        )

    # ヘッダー用集計値を事前計算
    total_face       = len(sorted_cns)
    total_edge       = sum(len(cell_lines_cache[cn]) for cn in sorted_cns)
    total_face_index = sum(1 for cn in sorted_cns if int(face_dict[cn]['CalMesh']) != 1)

    def pad_path(path_str, length=1000):
        """パス文字列を固定長バイト列に変換（末尾スペースパディング）"""
        b = path_str.encode('utf-8', errors='replace')
        return b[:length].ljust(length, b' ')

    with open(output_path, 'wb') as f:
        # ---- ヘッダー ----
        f.write(pad_path(face_gpkg_path))                             # 1000 bytes
        f.write(pad_path(edge_gpkg_path))                             # 1000 bytes
        f.write(struct.pack('<iii', total_face, total_edge, total_face_index))  # 12 bytes

        # ---- セルレコード ----
        for cn in sorted_cns:
            cell_info  = face_dict[cn]
            cell_lines = cell_lines_cache[cn]
            ln_count   = len(cell_lines)
            calc_mesh  = int(cell_info['CalMesh'])

            # 整数 2 値: face_number, total_edge_in_face
            f.write(struct.pack('<ii', cn, ln_count))

            # 実数 6 値: x, y, A, bl(標高), bld_ratio, bld_peri
            f.write(struct.pack('<6d',
                float(cell_info['xcoord']),
                float(cell_info['ycoord']),
                float(cell_info['area']),
                float(cell_info['_median']),
                float(cell_info['ratio']),
                float(cell_info.get('bill_perimeter', 0.0))
            ))

            # 実数 13 値: 土地利用面積（Fortran 変数宣言順の固定コード）
            f.write(struct.pack('<13d',
                *[float(cell_info.get(lc, 0.0)) for lc in LANDUSE_CODES]
            ))

            # 実数 17 値: 土壌面積（soil_1_area .. soil_17_area）
            f.write(struct.pack('<17d',
                *[float(cell_info.get(f"soil_{sc}_area", 0.0)) for sc in SOIL_CODES]
            ))

            # 整数 1 値: dummy_face (CalMesh)
            f.write(struct.pack('<i', calc_mesh))

            # ---- 辺レコード（このセルの辺を連続して格納）----
            for _, line in cell_lines.iterrows():
                ln     = int(line['LN'])
                peri   = int(line['peri'])
                cn_bgn = int(line['CN'])
                cn_end = int(line['CN_end'])

                x_bgn = float(line['X_bgn']); y_bgn = float(line['Y_bgn']); z_bgn = float(line['Z_bgn'])
                x_end = float(line['X_end']); y_end = float(line['Y_end']); z_end = float(line['Z_end'])
                x_cnt = (x_bgn + x_end) / 2.0
                y_cnt = (y_bgn + y_end) / 2.0
                z_cnt = float(line['Z_cnt'])

                node_bgn    = int(line['node_bgn'])
                node_end    = int(line['node_end'])
                edge_length = float(np.sqrt((x_end - x_bgn)**2 + (y_end - y_bgn)**2))

                cell_bgn_info    = face_dict[cn_bgn]
                x_cell_bgn       = float(cell_bgn_info['xcoord'])
                y_cell_bgn       = float(cell_bgn_info['ycoord'])
                area_bgn         = float(cell_bgn_info['area'])
                dist_bgn_to_cnt  = float(np.sqrt((x_cnt - x_cell_bgn)**2 + (y_cnt - y_cell_bgn)**2))

                if cn_end > 0 and cn_end in face_dict:
                    cell_end_info = face_dict[cn_end]
                    x_cell_end    = float(cell_end_info['xcoord'])
                    y_cell_end    = float(cell_end_info['ycoord'])
                    total_area    = area_bgn + float(cell_end_info['area'])
                    dx_cell       = x_cell_end - x_cell_bgn
                    dy_cell       = y_cell_end - y_cell_bgn
                    DL            = float(np.sqrt(dx_cell**2 + dy_cell**2))
                    cos_x, cos_y  = (dx_cell / DL, dy_cell / DL) if DL > 1e-10 else (0.0, 0.0)

                    dist_end_to_cnt = float(np.sqrt((x_cnt - x_cell_end)**2 + (y_cnt - y_cell_end)**2))
                    if dist_bgn_to_cnt > 1e-10 and dist_end_to_cnt > 1e-10:
                        inv_b = 1.0 / dist_bgn_to_cnt; inv_e = 1.0 / dist_end_to_cnt
                        s = inv_b + inv_e
                        weight_bgn, weight_end = inv_b / s, inv_e / s
                    elif dist_bgn_to_cnt <= 1e-10:
                        weight_bgn, weight_end = 1.0, 0.0
                    else:
                        weight_bgn, weight_end = 0.0, 1.0
                else:
                    total_area = area_bgn
                    DL         = dist_bgn_to_cnt
                    dx_cell    = x_cnt - x_cell_bgn; dy_cell = y_cnt - y_cell_bgn
                    cos_x, cos_y = (dx_cell / DL, dy_cell / DL) if DL > 1e-10 else (0.0, 0.0)
                    weight_bgn, weight_end = 1.0, 0.0

                weight_cnt = (1.0 / dist_bgn_to_cnt) / (1.0 + 1.0 / dist_bgn_to_cnt) \
                             if dist_bgn_to_cnt > 1e-10 else 1.0

                # 整数 4 値: en, bc_flag, fn_self, fn_adj
                f.write(struct.pack('<iiii', ln, peri, cn_bgn, cn_end))
                # 実数 7 値: A_CV_edge, dl, dxdl, dydl, weight_self, weight_adj, weight_face
                f.write(struct.pack('<7d',
                    total_area, DL, cos_x, cos_y,
                    weight_bgn, weight_end, weight_cnt
                ))
                # 整数 2 値: vertex_number_st, vertex_number_en
                f.write(struct.pack('<ii', node_bgn, node_end))
                # 実数 10 値: x_vertex_st..edge_length
                f.write(struct.pack('<10d',
                    x_bgn, y_bgn, z_bgn,
                    x_end, y_end, z_end,
                    x_cnt, y_cnt, z_cnt,
                    edge_length
                ))

    print(f"   ✅ cell.bin（バイナリ）を出力しました: {output_path}")
    print(f"   総セル数: {total_face}")
    print(f"   総辺数: {total_edge}")
    print(f"   計算セル数: {total_face_index}")
    print(f"   土地利用コード: {LANDUSE_CODES}")
    print(f"   土壌コード: {SOIL_CODES}")



def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='AQWA-INUNDATION2D メッシュファイル(cell.bin)作成スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python 02_mkCELL.py yaml/Kinu_Joso.yaml
        """
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='設定ファイル（YAMLファイル）のパス（例: yaml/Kinu_Joso.yaml）'
    )
    
    args = parser.parse_args()
    
    # 設定ファイルの存在確認
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"エラー: 設定ファイルが見つかりません: {args.config_file}")
        sys.exit(1)
    
    # 設定読み込み
    paths = read_config(args.config_file)
    
    # 入力ファイルの存在確認
    if not Path(paths['face_csv']).exists():
        print(f"エラー: face.csvが見つかりません: {paths['face_csv']}")
        sys.exit(1)
    
    if not Path(paths['edge_csv']).exists():
        print(f"エラー: edge.csvが見つかりません: {paths['edge_csv']}")
        sys.exit(1)
    
    # cell.bin出力ディレクトリの作成（存在しない場合）
    os.makedirs(paths['cell_txt_dir'], exist_ok=True)
    
    # データ読み込み
    df_face = read_face_csv(paths['face_csv'])
    df_edge = read_edge_csv(paths['edge_csv'])
    
    # ラインデータ作成
    df_lines = create_lines_from_sides(df_edge, df_face)
    
    # 隣接セル探索
    df_lines = find_adjacent_cells(df_lines)
    
    # cell.bin出力
    write_cell_bin(df_face, df_lines, paths['cell_txt'], paths['face_gpkg'], paths['edge_gpkg'])
    
    print(f"\n{'='*60}")
    print(f"✅ 処理が完了しました")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

