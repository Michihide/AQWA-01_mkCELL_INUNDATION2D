"""作業 bin/csv の read/write と mesh.bin への pack。"""

from __future__ import annotations

import csv
import difflib
import hashlib
import struct
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh_tables import EdgeTable, FaceTable, NodeTable, NormalizedMesh

MAGIC = b"AQWAMESH"
VERSION = 1
N_SECTIONS = 9
HASH_LEN = 16
HEADER_SIZE = 32 + N_SECTIONS * HASH_LEN + N_SECTIONS * 16

SECTION_NAMES = (
    "node", "edge", "face", "landuse", "soil", "building", "veg", "strc", "couple",
)
WORK_FILES = (
    "node.bin", "edge.bin", "face.bin",
    "landuse.bin", "soil.bin", "building.bin", "veg.bin",
    "special_edges.csv", "couple.csv",
)

KIND_NONE = 0
KIND_ROAD = 1
KIND_CULVERT = 2
KIND_Q = 4
KIND_W = 5
KIND_WALL = 6

KIND_BY_NAME = {
    "NONE": KIND_NONE,
    "ROAD": KIND_ROAD,
    "CULVERT": KIND_CULVERT,
    "Q": KIND_Q,
    "W": KIND_W,
    "WALL": KIND_WALL,
}
KIND_NAME = {v: k for k, v in KIND_BY_NAME.items()}

_NODE_REC = struct.Struct("<iddd")
_EDGE_REC = struct.Struct("<iiiiid")
_FACE_HEAD = struct.Struct("<iiddddi")
_SPARSE3 = struct.Struct("<iid")       # face_id, code, area
_BUILDING = struct.Struct("<idd")      # face_id, chi, peri
_VEG = struct.Struct("<idddd")         # face_id, chi_veg, a, H_v, C_D
_VEG_LEGACY = struct.Struct("<id")     # 旧: face_id, chi_veg
_STRC_LEGACY = struct.Struct("<iidddi")  # edge, kind, zc, H, z_road, qgroup
_STRC = struct.Struct("<iidddid")      # + B [m]。0 なら辺長
_COUPLE = struct.Struct("<iid i".replace(" ", ""))  # edge, river_link, kp, bank
_U32 = struct.Struct("<I")


def sha256_16(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()[:HASH_LEN]


def hash_file(path: Path | None) -> bytes:
    if path is None or not path.is_file():
        return b"\x00" * HASH_LEN
    data = path.read_bytes()
    if data == b"":
        return b"\x00" * HASH_LEN
    return sha256_16(data)


@dataclass
class SparseTriple:
    face_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    code: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    area: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.face_id))


@dataclass
class BuildingTable:
    face_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    chi_bld: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    bld_peri: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.face_id))


@dataclass
class VegTable:
    face_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    chi_veg: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    a: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    H_v: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    C_D: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.face_id))


@dataclass
class SpecialEdgeTable:
    edge_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    kind: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    zc: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    H: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    z_road: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    qgroup: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    B: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.edge_id))


@dataclass
class CoupleTable:
    edge_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    river_link: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    kp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    bank: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    def __len__(self) -> int:
        return int(len(self.edge_id))


@dataclass
class MeshAttributes:
    landuse: SparseTriple = field(default_factory=SparseTriple)
    soil: SparseTriple = field(default_factory=SparseTriple)
    building: BuildingTable = field(default_factory=BuildingTable)
    veg: VegTable = field(default_factory=VegTable)
    special_edges: SpecialEdgeTable = field(default_factory=SpecialEdgeTable)
    couple: CoupleTable = field(default_factory=CoupleTable)


@dataclass
class PackedMesh:
    version: int
    epsg: int
    mesh: NormalizedMesh
    attrs: MeshAttributes
    hashes: list[bytes] = field(default_factory=list)


def encode_nodes(nodes: NodeTable) -> bytes:
    buf = bytearray()
    for i in range(len(nodes)):
        buf.extend(_NODE_REC.pack(
            int(nodes.id[i]), float(nodes.x[i]), float(nodes.y[i]), float(nodes.z[i]),
        ))
    return bytes(buf)


def encode_edges(edges: EdgeTable) -> bytes:
    buf = bytearray()
    for i in range(len(edges)):
        buf.extend(_EDGE_REC.pack(
            int(edges.id[i]), int(edges.v1[i]), int(edges.v2[i]),
            int(edges.face_left[i]), int(edges.face_right[i]), float(edges.z_crest[i]),
        ))
    return bytes(buf)


def encode_faces(faces: FaceTable) -> bytes:
    buf = bytearray()
    for i in range(len(faces)):
        n = int(faces.n_sides[i])
        buf.extend(_FACE_HEAD.pack(
            int(faces.id[i]), int(faces.dummy[i]),
            float(faces.x[i]), float(faces.y[i]), float(faces.A[i]), float(faces.z_bed[i]),
            n,
        ))
        ids = faces.edge_ids[i]
        if len(ids) != n:
            raise ValueError(f"face {faces.id[i]}: n_sides={n} と辺列長 {len(ids)} が不一致")
        buf.extend(struct.pack("<" + "i" * n, *[int(e) for e in ids]))
    return bytes(buf)


def encode_sparse(table: SparseTriple) -> bytes:
    buf = bytearray(_U32.pack(len(table)))
    for i in range(len(table)):
        buf.extend(_SPARSE3.pack(int(table.face_id[i]), int(table.code[i]), float(table.area[i])))
    return bytes(buf)


def encode_building(table: BuildingTable) -> bytes:
    buf = bytearray(_U32.pack(len(table)))
    for i in range(len(table)):
        buf.extend(_BUILDING.pack(
            int(table.face_id[i]), float(table.chi_bld[i]), float(table.bld_peri[i]),
        ))
    return bytes(buf)


def encode_veg(table: VegTable) -> bytes:
    buf = bytearray(_U32.pack(len(table)))
    a = table.a if len(table.a) else np.zeros(len(table))
    hv = table.H_v if len(table.H_v) else np.zeros(len(table))
    cd = table.C_D if len(table.C_D) else np.zeros(len(table))
    for i in range(len(table)):
        buf.extend(_VEG.pack(
            int(table.face_id[i]), float(table.chi_veg[i]),
            float(a[i]), float(hv[i]), float(cd[i]),
        ))
    return bytes(buf)


def encode_strc(table: SpecialEdgeTable) -> bytes:
    buf = bytearray(_U32.pack(len(table)))
    has_b = len(table.B) == len(table)
    for i in range(len(table)):
        buf.extend(_STRC.pack(
            int(table.edge_id[i]), int(table.kind[i]),
            float(table.zc[i]), float(table.H[i]), float(table.z_road[i]),
            int(table.qgroup[i]),
            float(table.B[i]) if has_b else 0.0,
        ))
    return bytes(buf)


def encode_couple(table: CoupleTable) -> bytes:
    buf = bytearray(_U32.pack(len(table)))
    for i in range(len(table)):
        buf.extend(_COUPLE.pack(
            int(table.edge_id[i]), int(table.river_link[i]),
            float(table.kp[i]), int(table.bank[i]),
        ))
    return bytes(buf)


def _read_exact(data: bytes, offset: int, n: int) -> bytes:
    chunk = data[offset:offset + n]
    if len(chunk) != n:
        raise ValueError(f"mesh.bin: {n} B 必要ですが {len(chunk)} B しかありません (off={offset})")
    return chunk


def decode_nodes(payload: bytes) -> NodeTable:
    n = len(payload) // _NODE_REC.size
    if n * _NODE_REC.size != len(payload):
        raise ValueError("node セクション長がレコードサイズの倍数ではありません")
    ids = np.empty(n, dtype=np.int32)
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    for i in range(n):
        rec = _NODE_REC.unpack_from(payload, i * _NODE_REC.size)
        ids[i], x[i], y[i], z[i] = rec
    return NodeTable(ids, x, y, z)


def decode_edges(payload: bytes) -> EdgeTable:
    n = len(payload) // _EDGE_REC.size
    if n * _EDGE_REC.size != len(payload):
        raise ValueError("edge セクション長がレコードサイズの倍数ではありません")
    ids = np.empty(n, dtype=np.int32)
    v1 = np.empty(n, dtype=np.int32)
    v2 = np.empty(n, dtype=np.int32)
    left = np.empty(n, dtype=np.int32)
    right = np.empty(n, dtype=np.int32)
    zc = np.empty(n)
    for i in range(n):
        rec = _EDGE_REC.unpack_from(payload, i * _EDGE_REC.size)
        ids[i], v1[i], v2[i], left[i], right[i], zc[i] = rec
    return EdgeTable(ids, v1, v2, left, right, zc)


def decode_faces(payload: bytes, nface: int) -> FaceTable:
    ids = np.empty(nface, dtype=np.int32)
    dummy = np.empty(nface, dtype=np.int32)
    x = np.empty(nface)
    y = np.empty(nface)
    A = np.empty(nface)
    z_bed = np.empty(nface)
    n_sides = np.empty(nface, dtype=np.int32)
    edge_ids: list[np.ndarray] = []
    off = 0
    for i in range(nface):
        rec = _FACE_HEAD.unpack_from(payload, off)
        off += _FACE_HEAD.size
        ids[i], dummy[i], x[i], y[i], A[i], z_bed[i], n_sides[i] = rec
        n = int(n_sides[i])
        raw = _read_exact(payload, off, 4 * n)
        off += 4 * n
        edge_ids.append(np.asarray(struct.unpack("<" + "i" * n, raw), dtype=np.int32))
    if off != len(payload):
        raise ValueError(f"face セクションに余りがあります ({len(payload) - off} B)")
    return FaceTable(ids, dummy, x, y, A, z_bed, n_sides, edge_ids)


def decode_sparse(payload: bytes) -> SparseTriple:
    if not payload:
        return SparseTriple()
    n = _U32.unpack_from(payload, 0)[0]
    face_id = np.empty(n, dtype=np.int32)
    code = np.empty(n, dtype=np.int32)
    area = np.empty(n)
    off = 4
    for i in range(n):
        face_id[i], code[i], area[i] = _SPARSE3.unpack_from(payload, off)
        off += _SPARSE3.size
    return SparseTriple(face_id, code, area)


def decode_building(payload: bytes) -> BuildingTable:
    if not payload:
        return BuildingTable()
    n = _U32.unpack_from(payload, 0)[0]
    face_id = np.empty(n, dtype=np.int32)
    chi = np.empty(n)
    peri = np.empty(n)
    off = 4
    for i in range(n):
        face_id[i], chi[i], peri[i] = _BUILDING.unpack_from(payload, off)
        off += _BUILDING.size
    return BuildingTable(face_id, chi, peri)


def decode_veg(payload: bytes) -> VegTable:
    if not payload:
        return VegTable()
    n = _U32.unpack_from(payload, 0)[0]
    if n == 0:
        return VegTable()
    face_id = np.empty(n, dtype=np.int32)
    chi = np.empty(n)
    a = np.zeros(n)
    hv = np.zeros(n)
    cd = np.zeros(n)
    off = 4
    rest = len(payload) - 4
    if rest == n * _VEG.size:
        for i in range(n):
            face_id[i], chi[i], a[i], hv[i], cd[i] = _VEG.unpack_from(payload, off)
            off += _VEG.size
    elif rest == n * _VEG_LEGACY.size:
        for i in range(n):
            face_id[i], chi[i] = _VEG_LEGACY.unpack_from(payload, off)
            off += _VEG_LEGACY.size
    else:
        raise ValueError(
            f"veg.bin の長さが合いません: nentry={n}, payload={len(payload)} B"
        )
    return VegTable(face_id, chi, a, hv, cd)


def decode_strc(payload: bytes) -> SpecialEdgeTable:
    if not payload:
        return SpecialEdgeTable()
    n = _U32.unpack_from(payload, 0)[0]
    edge_id = np.empty(n, dtype=np.int32)
    kind = np.empty(n, dtype=np.int32)
    zc = np.empty(n)
    H = np.empty(n)
    z_road = np.empty(n)
    qgroup = np.empty(n, dtype=np.int32)
    B = np.zeros(n)
    off = 4
    rest = len(payload) - 4
    if n == 0:
        return SpecialEdgeTable()
    if rest == n * _STRC.size:
        for i in range(n):
            edge_id[i], kind[i], zc[i], H[i], z_road[i], qgroup[i], B[i] = (
                _STRC.unpack_from(payload, off)
            )
            off += _STRC.size
    elif rest == n * _STRC_LEGACY.size:
        for i in range(n):
            edge_id[i], kind[i], zc[i], H[i], z_road[i], qgroup[i] = (
                _STRC_LEGACY.unpack_from(payload, off)
            )
            off += _STRC_LEGACY.size
    else:
        raise ValueError(
            f"strc の長さが合いません: nentry={n}, payload={len(payload)} B"
        )
    return SpecialEdgeTable(edge_id, kind, zc, H, z_road, qgroup, B)


def decode_couple(payload: bytes) -> CoupleTable:
    if not payload:
        return CoupleTable()
    n = _U32.unpack_from(payload, 0)[0]
    edge_id = np.empty(n, dtype=np.int32)
    river_link = np.empty(n, dtype=np.int32)
    kp = np.empty(n)
    bank = np.empty(n, dtype=np.int32)
    off = 4
    for i in range(n):
        edge_id[i], river_link[i], kp[i], bank[i] = _COUPLE.unpack_from(payload, off)
        off += _COUPLE.size
    return CoupleTable(edge_id, river_link, kp, bank)


_KIND_SYNONYMS = {
    "ROAD": KIND_ROAD,
    "WEIR": KIND_ROAD,
    "道路": KIND_ROAD,
    "道路堰": KIND_ROAD,
    "堰": KIND_ROAD,
    "CULVERT": KIND_CULVERT,
    "CULV": KIND_CULVERT,
    "カルバート": KIND_CULVERT,
    "暗渠": KIND_CULVERT,
    "函渠": KIND_CULVERT,
    "Q": KIND_Q,
    "FLOW": KIND_Q,
    "DISCHARGE": KIND_Q,
    "流量": KIND_Q,
    "W": KIND_W,
    "WL": KIND_W,
    "WATERLEVEL": KIND_W,
    "WATER_LEVEL": KIND_W,
    "水位": KIND_W,
    "WALL": KIND_WALL,
    "壁": KIND_WALL,
    "NONE": KIND_NONE,
}

_Z_ABS = {"abs", "absolute", "絶対", "絶対標高", "elevation", "elev"}
_Z_REL = {"rel", "relative", "相対", "相対高さ", "height", "dz", "delta"}


def resolve_kind_name(value: str) -> str:
    """kind を正規名にする。別名と近い綴りを許す。"""
    code = parse_kind(value)
    name = KIND_NAME.get(code, "NONE")
    if name == "NONE" and code != KIND_NONE:
        raise ValueError(f"不明な special_edges kind: {value!r}")
    return name


def parse_kind(value: str) -> int:
    raw = unicodedata.normalize("NFKC", str(value).strip())
    if not raw:
        raise ValueError("special_edges kind が空です")
    key = raw.upper()
    if key in ("BOTH", "3") or raw in ("両方",):
        raise ValueError("special_edges kind BOTH はありません")
    if key.isdigit():
        n = int(key)
        if n not in KIND_NAME:
            raise ValueError(f"不明な special_edges kind: {value!r}")
        return n
    if key in KIND_BY_NAME:
        return KIND_BY_NAME[key]
    if key in _KIND_SYNONYMS:
        return _KIND_SYNONYMS[key]
    if raw in _KIND_SYNONYMS:
        return _KIND_SYNONYMS[raw]
    pool = [*(KIND_BY_NAME.keys()), *(_KIND_SYNONYMS.keys())]
    pool_u = {p.upper(): p for p in pool}
    hits = difflib.get_close_matches(key, list(pool_u), n=1, cutoff=0.72)
    if hits:
        chosen = pool_u[hits[0]]
        if chosen.upper() in KIND_BY_NAME:
            return KIND_BY_NAME[chosen.upper()]
        return _KIND_SYNONYMS[chosen]
    raise ValueError(f"不明な special_edges kind: {value!r}")


def parse_z_mode(value: object, *, default: str = "absolute") -> str:
    """zc / z_road が絶対標高か相対高さか。"""
    if value is None:
        return default
    raw = unicodedata.normalize("NFKC", str(value).strip())
    if not raw:
        return default
    key = raw.lower()
    if key in _Z_ABS or raw in _Z_ABS:
        return "absolute"
    if key in _Z_REL or raw in _Z_REL:
        return "relative"
    hits = difflib.get_close_matches(key, list(_Z_ABS | _Z_REL), n=1, cutoff=0.72)
    if hits:
        return "absolute" if hits[0] in _Z_ABS else "relative"
    raise ValueError(f"不明な z_mode: {value!r}（absolute / relative）")


def parse_bank(value: str) -> int:
    key = value.strip().upper()
    if key in ("1", "L", "LEFT"):
        return 1
    if key in ("2", "R", "RIGHT"):
        return 2
    raise ValueError(f"不明な couple bank: {value!r}")


def write_special_edges_csv(path: Path, table: SpecialEdgeTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "kind", "zc", "H", "z_road", "qgroup", "B"])
        has_b = len(table.B) == len(table)
        for i in range(len(table)):
            w.writerow([
                int(table.edge_id[i]),
                KIND_NAME.get(int(table.kind[i]), int(table.kind[i])),
                float(table.zc[i]), float(table.H[i]), float(table.z_road[i]),
                int(table.qgroup[i]),
                float(table.B[i]) if has_b else 0.0,
            ])


def write_couple_csv(path: Path, table: CoupleTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "river_link", "kp", "bank"])
        for i in range(len(table)):
            w.writerow([
                int(table.edge_id[i]), int(table.river_link[i]),
                float(table.kp[i]), int(table.bank[i]),
            ])


def read_special_edges_csv(path: Path) -> SpecialEdgeTable:
    if not path.is_file() or path.stat().st_size == 0:
        return SpecialEdgeTable()
    edge_id: list[int] = []
    kind: list[int] = []
    zc: list[float] = []
    H: list[float] = []
    z_road: list[float] = []
    qgroup: list[int] = []
    B: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        if rows.fieldnames is None:
            return SpecialEdgeTable()
        for row in rows:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            edge_id.append(int(row["edge_id"]))
            kind.append(parse_kind(row.get("kind", "NONE")))
            zc.append(float(row.get("zc") or 0.0))
            H.append(float(row.get("H") or 0.0))
            z_road.append(float(row.get("z_road") or 0.0))
            qgroup.append(int(float(row.get("qgroup") or 0)))
            B.append(float(row.get("B") or 0.0))
    if not edge_id:
        return SpecialEdgeTable()
    return SpecialEdgeTable(
        np.asarray(edge_id, dtype=np.int32),
        np.asarray(kind, dtype=np.int32),
        np.asarray(zc, dtype=np.float64),
        np.asarray(H, dtype=np.float64),
        np.asarray(z_road, dtype=np.float64),
        np.asarray(qgroup, dtype=np.int32),
        np.asarray(B, dtype=np.float64),
    )


def read_couple_csv(path: Path) -> CoupleTable:
    if not path.is_file() or path.stat().st_size == 0:
        return CoupleTable()
    edge_id: list[int] = []
    river_link: list[int] = []
    kp: list[float] = []
    bank: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        if rows.fieldnames is None:
            return CoupleTable()
        for row in rows:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            edge_id.append(int(row["edge_id"]))
            river_link.append(int(row["river_link"]))
            kp.append(float(row.get("kp") or 0.0))
            bank.append(parse_bank(str(row.get("bank") or "1")))
    if not edge_id:
        return CoupleTable()
    return CoupleTable(
        np.asarray(edge_id, dtype=np.int32),
        np.asarray(river_link, dtype=np.int32),
        np.asarray(kp, dtype=np.float64),
        np.asarray(bank, dtype=np.int32),
    )


def dense_areas_to_sparse(areas: np.ndarray, codes: list[int], atol: float = 0.0) -> SparseTriple:
    """面 × コードの密配列からゼロでない (face, code, area) を拾う。"""
    if areas.size == 0:
        return SparseTriple()
    n_face, n_code = areas.shape
    if n_code != len(codes):
        raise ValueError("areas の列数が codes と一致しません")
    face_id: list[int] = []
    code: list[int] = []
    area: list[float] = []
    for i in range(n_face):
        for j in range(n_code):
            a = float(areas[i, j])
            if a > atol:
                face_id.append(i + 1)
                code.append(int(codes[j]))
                area.append(a)
    if not face_id:
        return SparseTriple()
    return SparseTriple(
        np.asarray(face_id, dtype=np.int32),
        np.asarray(code, dtype=np.int32),
        np.asarray(area, dtype=np.float64),
    )


def veg_from_dense(
    chi: np.ndarray,
    a: np.ndarray,
    hv: np.ndarray,
    cd: np.ndarray,
    atol: float = 0.0,
) -> VegTable:
    chi = np.asarray(chi, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    hv = np.asarray(hv, dtype=np.float64)
    cd = np.asarray(cd, dtype=np.float64)
    mask = (chi > atol) | (a > atol)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return VegTable()
    return VegTable(
        (idx + 1).astype(np.int32),
        chi[idx],
        a[idx],
        hv[idx],
        cd[idx],
    )


def building_from_dense(chi: np.ndarray, peri: np.ndarray, atol: float = 0.0) -> BuildingTable:
    n = len(chi)
    mask = (np.asarray(chi) > atol) | (np.asarray(peri) > atol)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return BuildingTable()
    return BuildingTable(
        (idx + 1).astype(np.int32),
        np.asarray(chi, dtype=np.float64)[idx],
        np.asarray(peri, dtype=np.float64)[idx],
    )


def pack_mesh(
    mesh: NormalizedMesh,
    attrs: MeshAttributes | None = None,
    *,
    epsg: int = 0,
    hashes: list[bytes] | None = None,
) -> bytes:
    """正規化テーブルを mesh.bin バイト列にする。"""
    attrs = attrs or MeshAttributes()
    payloads = [
        encode_nodes(mesh.nodes),
        encode_edges(mesh.edges),
        encode_faces(mesh.faces),
        encode_sparse(attrs.landuse),
        encode_sparse(attrs.soil),
        encode_building(attrs.building),
        encode_veg(attrs.veg),
        encode_strc(attrs.special_edges),
        encode_couple(attrs.couple),
    ]
    if hashes is None:
        hashes = [sha256_16(p) if p else b"\x00" * HASH_LEN for p in payloads]
    if len(hashes) != N_SECTIONS:
        raise ValueError("hashes は 9 個必要です")

    offsets: list[tuple[int, int]] = []
    off = HEADER_SIZE
    for p in payloads:
        offsets.append((off, len(p)))
        off += len(p)

    header = struct.pack(
        "<8sHHiIIII",
        MAGIC, VERSION, 0, int(epsg),
        len(mesh.nodes), len(mesh.edges), len(mesh.faces), 0,
    )
    hash_block = b"".join(h.ljust(HASH_LEN, b"\x00")[:HASH_LEN] for h in hashes)
    table = b"".join(struct.pack("<QQ", o, n) for o, n in offsets)
    return header + hash_block + table + b"".join(payloads)


def unpack_mesh(data: bytes) -> PackedMesh:
    if len(data) < HEADER_SIZE:
        raise ValueError("mesh.bin が短すぎます")
    magic, version, _flags, epsg, nnode, nedge, nface, _res = struct.unpack_from(
        "<8sHHiIIII", data, 0,
    )
    if magic != MAGIC:
        raise ValueError(f"魔法数が違います: {magic!r}")
    if version != VERSION:
        raise ValueError(f"未対応の mesh.bin 版: {version}")
    hashes = [data[32 + i * HASH_LEN:32 + (i + 1) * HASH_LEN] for i in range(N_SECTIONS)]
    table_off = 32 + N_SECTIONS * HASH_LEN
    sections: list[bytes] = []
    for i in range(N_SECTIONS):
        offset, length = struct.unpack_from("<QQ", data, table_off + i * 16)
        if length == 0:
            sections.append(b"")
        else:
            sections.append(_read_exact(data, int(offset), int(length)))

    nodes = decode_nodes(sections[0])
    edges = decode_edges(sections[1])
    faces = decode_faces(sections[2], int(nface))
    if len(nodes) != nnode or len(edges) != nedge or len(faces) != nface:
        raise ValueError("ヘッダの個数とセクション内容が一致しません")
    attrs = MeshAttributes(
        landuse=decode_sparse(sections[3]),
        soil=decode_sparse(sections[4]),
        building=decode_building(sections[5]),
        veg=decode_veg(sections[6]),
        special_edges=decode_strc(sections[7]),
        couple=decode_couple(sections[8]),
    )
    return PackedMesh(
        version=version,
        epsg=int(epsg),
        mesh=NormalizedMesh(nodes, edges, faces),
        attrs=attrs,
        hashes=hashes,
    )


def write_work_files(
    directory: Path,
    mesh: NormalizedMesh,
    attrs: MeshAttributes | None = None,
) -> dict[str, Path]:
    """作業 bin/csv を directory に書く。"""
    attrs = attrs or MeshAttributes()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / name for name in WORK_FILES}
    paths["node.bin"].write_bytes(encode_nodes(mesh.nodes))
    paths["edge.bin"].write_bytes(encode_edges(mesh.edges))
    paths["face.bin"].write_bytes(encode_faces(mesh.faces))
    paths["landuse.bin"].write_bytes(encode_sparse(attrs.landuse))
    paths["soil.bin"].write_bytes(encode_sparse(attrs.soil))
    paths["building.bin"].write_bytes(encode_building(attrs.building))
    paths["veg.bin"].write_bytes(encode_veg(attrs.veg))
    write_special_edges_csv(paths["special_edges.csv"], attrs.special_edges)
    write_couple_csv(paths["couple.csv"], attrs.couple)
    return paths


def read_work_files(directory: Path) -> tuple[NormalizedMesh, MeshAttributes, list[bytes]]:
    """作業ファイルを読み、各ファイルのハッシュも返す。"""
    paths = [directory / name for name in WORK_FILES]
    hashes = [hash_file(p) for p in paths]
    nodes = decode_nodes(paths[0].read_bytes())
    edges = decode_edges(paths[1].read_bytes())
    face_payload = paths[2].read_bytes()
    # nface はレコードを走査して数える
    nface = 0
    off = 0
    while off < len(face_payload):
        _id, _d, _x, _y, _a, _z, n = _FACE_HEAD.unpack_from(face_payload, off)
        off += _FACE_HEAD.size + 4 * int(n)
        nface += 1
    faces = decode_faces(face_payload, nface)
    attrs = MeshAttributes(
        landuse=decode_sparse(paths[3].read_bytes() if paths[3].is_file() else b""),
        soil=decode_sparse(paths[4].read_bytes() if paths[4].is_file() else b""),
        building=decode_building(paths[5].read_bytes() if paths[5].is_file() else b""),
        veg=decode_veg(paths[6].read_bytes() if paths[6].is_file() else b""),
        special_edges=read_special_edges_csv(paths[7]),
        couple=read_couple_csv(paths[8]),
    )
    return NormalizedMesh(nodes, edges, faces), attrs, hashes


def pack_work_directory(directory: Path, output_path: Path, *, epsg: int = 0) -> Path:
    mesh, attrs, hashes = read_work_files(directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pack_mesh(mesh, attrs, epsg=epsg, hashes=hashes))
    return output_path


def write_mesh_bin(
    path: Path,
    mesh: NormalizedMesh,
    attrs: MeshAttributes | None = None,
    *,
    epsg: int = 0,
    hashes: list[bytes] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack_mesh(mesh, attrs, epsg=epsg, hashes=hashes))
    return path
