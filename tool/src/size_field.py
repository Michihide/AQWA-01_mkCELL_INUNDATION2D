"""背景サイズ場。

作業座標系上の規則格子で目標要素サイズを保持し、gmsh へは Post-processing View
として渡す（Field[PostView]）。反復細分化では、この格子に対して局所的に小さい値を
書き込むだけでよく、格子構造が変わらないため反復間の比較が容易になる。

面積下限 625 m^2 に対応する辺長を下回る値は書き込まない。指示書の細分化は
「基準を満たすまで細かく」だが、本ツールでは面積下限が優先される。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger


@dataclass
class SizeField:
    """規則格子上の目標要素サイズ [m]。

    floor は格子ごとに持つ。外周四角形帯の内側境界は節点間隔が固定されているため、
    その近傍だけ下限を高くしておかないと、細分化するたびに帯と内側要素の間に
    面積下限を割る細長い三角形が生まれてしまう。

    attempts は各格子点を細分化した回数。地形基準の中には、要素を小さくしても
    解消しない（むしろ悪化する）ものがあるため、既定回数を超えたら細分化を諦める。
    """

    values: np.ndarray  # (ny, nx) float64
    x_min: float
    y_min: float
    spacing: float
    floor_grid: np.ndarray  # (ny, nx) 各格子点でのサイズ下限
    ceiling: float
    attempts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.attempts is None:
            self.attempts = np.zeros(self.values.shape, dtype=np.int32)

    @property
    def floor(self) -> float:
        return float(self.floor_grid.min())

    @classmethod
    def uniform(
        cls,
        bounds: tuple[float, float, float, float],
        spacing: float,
        value: float,
        floor: float,
        ceiling: float,
        pad: float = 0.0,
    ) -> SizeField:
        min_x, min_y, max_x, max_y = bounds
        min_x -= pad
        min_y -= pad
        max_x += pad
        max_y += pad
        nx = max(2, int(np.ceil((max_x - min_x) / spacing)) + 1)
        ny = max(2, int(np.ceil((max_y - min_y) / spacing)) + 1)
        values = np.full((ny, nx), float(np.clip(value, floor, ceiling)))
        floor_grid = np.full((ny, nx), float(floor))
        return cls(values, min_x, min_y, spacing, floor_grid, ceiling)

    def raise_floor(self, x: np.ndarray, y: np.ndarray, value: float, radius: float = 0.0) -> None:
        """指定位置（と半径内）のサイズ下限を引き上げる。"""
        iy, ix = self._indices(x, y)
        r = int(np.ceil(radius / self.spacing))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                jy = np.clip(iy + dy, 0, self.ny - 1)
                jx = np.clip(ix + dx, 0, self.nx - 1)
                np.maximum.at(self.floor_grid, (jy, jx), value)
        np.maximum(self.values, self.floor_grid, out=self.values)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def nx(self) -> int:
        return self.values.shape[1]

    @property
    def ny(self) -> int:
        return self.values.shape[0]

    def copy(self) -> SizeField:
        return SizeField(
            self.values.copy(), self.x_min, self.y_min, self.spacing,
            self.floor_grid.copy(), self.ceiling, self.attempts.copy(),
        )

    # ------------------------------------------------------------------
    # 座標変換
    # ------------------------------------------------------------------

    def grid_coords(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + np.arange(self.nx) * self.spacing
        y = self.y_min + np.arange(self.ny) * self.spacing
        return x, y

    def _indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip(np.round((np.asarray(x) - self.x_min) / self.spacing).astype(np.int64),
                     0, self.nx - 1)
        iy = np.clip(np.round((np.asarray(y) - self.y_min) / self.spacing).astype(np.int64),
                     0, self.ny - 1)
        return iy, ix

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        iy, ix = self._indices(x, y)
        return self.values[iy, ix]

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------

    def refinable(self, x: np.ndarray, y: np.ndarray, max_attempts: int) -> np.ndarray:
        """まだ細分化を試してよい位置か。"""
        iy, ix = self._indices(x, y)
        return (self.attempts[iy, ix] < max_attempts) & (
            self.values[iy, ix] > self.floor_grid[iy, ix] + 1e-9
        )

    def at_floor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """目標サイズがすでに下限に張り付いている位置か。"""
        iy, ix = self._indices(x, y)
        return self.values[iy, ix] <= self.floor_grid[iy, ix] + 1e-9

    def lower_floor(
        self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray, hard_floor: float
    ) -> int:
        """指定位置のサイズ下限を引き下げる。書き換えた格子点数を返す。

        raise_floor で引き上げた下限（帯の内側境界や拘束線の付近）が、他の基準を
        満たす妨げになることがある。その場所だけ緩めるための操作で、面積下限に
        直結する hard_floor は下回らない。
        """
        iy, ix = self._indices(x, y)
        want = np.maximum(np.asarray(sizes, dtype=float), hard_floor)
        before = self.floor_grid[iy, ix]
        np.minimum.at(self.floor_grid, (iy, ix), want)
        return int((self.floor_grid[iy, ix] < before - 1e-9).sum())

    def apply_min(
        self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray, *, count_attempts: bool = True,
    ) -> int:
        """指定位置のサイズを（より小さければ）上書きする。書き換えた格子点数を返す。"""
        iy, ix = self._indices(x, y)
        flat = iy * self.nx + ix
        target = np.minimum(
            np.maximum(np.asarray(sizes, dtype=float), self.floor_grid.ravel()[flat]),
            self.ceiling,
        )
        current = self.values.ravel()
        # 同一格子点に複数の要求が来た場合は最小値を採る
        order = np.argsort(target)
        uniq, first = np.unique(flat[order], return_index=True)
        best = target[order][first]
        changed = best < current[uniq] - 1e-9
        current[uniq[changed]] = best[changed]
        if count_attempts:
            self.attempts.ravel()[uniq[changed]] += 1
        return int(changed.sum())

    def apply_max(self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray) -> int:
        """指定位置のサイズを（より大きければ）上書きする。粗大化に使う。"""
        iy, ix = self._indices(x, y)
        flat = iy * self.nx + ix
        target = np.minimum(
            np.maximum(np.asarray(sizes, dtype=float), self.floor_grid.ravel()[flat]),
            self.ceiling,
        )
        current = self.values.ravel()
        order = np.argsort(-target)
        uniq, first = np.unique(flat[order], return_index=True)
        best = target[order][first]
        changed = best > current[uniq] + 1e-9
        current[uniq[changed]] = best[changed]
        return int(changed.sum())

    def clamp(self) -> None:
        np.clip(self.values, self.floor_grid, self.ceiling, out=self.values)

    def smooth(self, max_ratio: float, max_sweeps: int = 200) -> int:
        """隣接格子間のサイズ比が max_ratio 以下になるまで縮める。

        h(x) <= h(y) + L*|x - y|、L = max_ratio - 1 のリプシッツ条件を、
        4 近傍の掃引で満たす。要素サイズの急変を抑え、品質の悪い遷移要素を防ぐ。
        """
        slope = (max_ratio - 1.0) * self.spacing
        v = self.values
        for sweep in range(max_sweeps):
            prev = v.copy()
            v[:, 1:] = np.minimum(v[:, 1:], v[:, :-1] + slope)
            v[:, :-1] = np.minimum(v[:, :-1], v[:, 1:] + slope)
            v[1:, :] = np.minimum(v[1:, :], v[:-1, :] + slope)
            v[:-1, :] = np.minimum(v[:-1, :], v[1:, :] + slope)
            # 平滑化で下限を割らないようにする（帯近傍の下限が高いため必要）
            np.maximum(v, self.floor_grid, out=v)
            if np.allclose(prev, v, atol=1e-6):
                return sweep + 1
        get_logger().warning("サイズ場の平滑化が %d 掃引で収束しませんでした", max_sweeps)
        return max_sweeps

    def stats(self) -> dict[str, float]:
        v = self.values
        return {
            "min": float(v.min()),
            "median": float(np.median(v)),
            "max": float(v.max()),
            "at_floor_ratio": float((v <= self.floor_grid + 1e-9).mean()),
        }

    # ------------------------------------------------------------------
    # gmsh 連携
    # ------------------------------------------------------------------

    def to_view_data(self) -> np.ndarray:
        """Field[PostView] 用のスカラー四角形リストデータを構築する。

        gmsh の Python API は numpy 配列をそのまま C 配列として渡せるため、
        Python リストに変換せずに返す（全域では数百万要素になる）。
        """
        x, y = self.grid_coords()
        xx, yy = np.meshgrid(x, y)
        v = self.values

        # セル (j, i) の 4 隅
        x0 = xx[:-1, :-1]
        x1 = xx[:-1, 1:]
        y0 = yy[:-1, :-1]
        y1 = yy[1:, :-1]
        v00 = v[:-1, :-1]
        v10 = v[:-1, 1:]
        v11 = v[1:, 1:]
        v01 = v[1:, :-1]

        n = x0.size
        data = np.empty((n, 16), dtype=np.float64)
        data[:, 0] = x0.ravel()
        data[:, 1] = x1.ravel()
        data[:, 2] = x1.ravel()
        data[:, 3] = x0.ravel()
        data[:, 4] = y0.ravel()
        data[:, 5] = y0.ravel()
        data[:, 6] = y1.ravel()
        data[:, 7] = y1.ravel()
        data[:, 8:12] = 0.0
        data[:, 12] = v00.ravel()
        data[:, 13] = v10.ravel()
        data[:, 14] = v11.ravel()
        data[:, 15] = v01.ravel()
        return data.ravel()
