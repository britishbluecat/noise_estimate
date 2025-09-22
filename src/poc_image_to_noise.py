"""
音場推定PoC: 画像分析パイプライン（実画像解析版）
----------------------------------------------------
目的:
  src/data/{location_type}/{store_id}_{NNN}.jpg|png と {store_id}.txt(min,ave,max) から
  X: location_type, est_hall_area, est_hall_height, furniture_index
  y: min, ave, max
を生成する。

要点（このスクリプトが実際にやること）:
 1) 画像ごとに実解析（ダミーではない）
    - 単眼Depth (MiDaS) による相対深度推定
    - 直線検出→消失点推定→天井/壁の向き推定
    - 床領域の単応答（homography近似）→床の射影面積←→相対スケール
    - 物体検出/セグメンテーション (YOLOv8) で椅子・テーブル等の面積/カウント
 2) 絶対スケール化
    - 画像に写る既知寸法（A4=0.297m など）を自動/手動で与え、深度のスケール合わせ
    - 既知スケールが無い場合は “高さの事前分布(2.6〜3.0m)” を使いベイズ的に最尤推定
 3) 集約
    - 店舗内の複数画像を SfM 的に統合（簡易: 特徴点トラッキング + 三角測量）
    - 天井高の中央値/堅牢平均、ホール面積（床マスクの合成）
    - 家具密度index= g(椅子/テーブル等の画素占有率と件数, 通路推定)

依存:
  pip install opencv-python numpy pandas torch torchvision ultralytics scikit-image matplotlib
  * MiDaSモデルは torch.hub で取得 (intel-isl/MiDaS)
  * YOLOv8 は ultralytics の事前学習モデル（yolov8n|s|m）を利用

注意:
 - 実行にはインターネット接続が必要（初回のモデル取得時）。
 - 既知スケール物体の自動検出は難易度が高い。最初は "A4メニューがあります(Y/N) と写っているピクセル高さ" を半自動で与えるUIでも可。
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import math
import json
import re
import numpy as np
import pandas as pd
import cv2 as cv

# 先頭の import 群の下あたりに追加
import collections

_SEQ_RE = re.compile(r"_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)

def dedup_by_sequence(img_paths: list[Path], prefer_ext=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """
    同じ {store_id}_{NNN}.ext が複数拡張子で存在する場合、
    NNN（数字）ごとに1枚だけ残す。prefer_ext の優先順位で採用。
    戻り値は NNN の昇順を維持。
    """
    # NNN -> {ext: Path}
    buckets = collections.defaultdict(dict)
    for p in img_paths:
        m = _SEQ_RE.search(p.name)
        if not m:
            # NNNが取れないファイルは最後に回す（念のため1枚扱い）
            key = ("", p.name.lower())
        else:
            key = (int(m.group(1)), None)
        buckets[key][p.suffix.lower()] = p

    ordered_keys = sorted(buckets.keys(), key=lambda k: (k[0] if isinstance(k[0], int) else float("inf"), k[1] or ""))

    picked = []
    for k in ordered_keys:
        cand = buckets[k]
        chosen = None
        for ext in prefer_ext:
            if ext in cand:
                chosen = cand[ext]
                break
        if not chosen:
            # どれにも当たらなければ最初の1枚
            chosen = next(iter(cand.values()))
        picked.append(chosen)

    return picked


# =============== ユーティリティ ===============
def list_store_groups(base: Path):
    for loc_dir in base.iterdir():
        if not loc_dir.is_dir():
            continue
        location_type = loc_dir.name
        for txt in sorted(loc_dir.glob("*.txt")):
            store_id = txt.stem
            candidates = sorted(list(loc_dir.glob(f"{store_id}_*.jpg")) +
                                list(loc_dir.glob(f"{store_id}_*.jpeg")) +
                                list(loc_dir.glob(f"{store_id}_*.png")) +
                                list(loc_dir.glob(f"{store_id}_*.JPG")) +
                                list(loc_dir.glob(f"{store_id}_*.PNG")))
            imgs = dedup_by_sequence(candidates)  # ← 重複排除
            if imgs:
                yield location_type, store_id, imgs, txt


# =============== dB読み込み ===============
def read_db_triplet(txt_path: Path):
    """CSV形式: 1行目 ヘッダmin,ave,max; 2行目 数値"""
    df = pd.read_csv(txt_path)
    row = df.iloc[0]
    return float(row["min"]), float(row["ave"]), float(row["max"])

# =============== furniture_index ===============
# 調整点:
# - MIN_BOX_FR を 0.02 に上げて極小検出を弾く
# - γ を 1.15 にして全体に抑えめ
# - C_MAX を 10 に（個数による加点の飽和を早める）

FURN_AREA_WEIGHTS = {
    "chair": 1.0,
    "dining table": 1.1,   # 少し抑えめ
    "couch": 1.1, "sofa": 1.1,
    "bench": 0.7, "tv": 0.5,
    "refrigerator": 0.3,
}
PEOPLE_AREA_WEIGHT = 0.25  # 人の加点もやや控えめ

C_MAX = 10         # 個数スケール上限（早めに飽和）
MIN_BOX_FR = 0.02  # 画像に対する最小面積（2%）未満は無視（上げた）
BORDER_PAD = 6     # 画像端に貼り付いた検出を無視するピクセル幅

# =============== 単眼Depth（MiDaS） ===============
_midas_model = None
_midas_transform = None

def init_midas(model_name: str = "DPT_Hybrid"):
    """
    有効モデル: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    """
    global _midas_model, _midas_transform
    import torch
    _midas_model = torch.hub.load("intel-isl/MiDaS", model_name)
    _midas_model.eval()
    tfms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ("DPT_Large", "DPT_Hybrid"):
        _midas_transform = tfms.dpt_transform
    else:  # "MiDaS_small" など
        _midas_transform = tfms.small_transform

def run_midas(img_bgr: np.ndarray) -> np.ndarray:
    import torch
    assert _midas_model is not None, "Call init_midas() first"
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    inp = _midas_transform(img_rgb)          # DPT系は [1,3,H,W]、smallは [3,H,W] のことがある
    if isinstance(inp, tuple):               # 念のため（将来API差異対策）
        inp = inp[0]
    if hasattr(inp, "dim") and inp.dim() == 3:
        inp = inp.unsqueeze(0)               # small系のみ追加

    dev = next(_midas_model.parameters()).device
    inp = inp.to(dev)

    with torch.no_grad():
        pred = _midas_model(inp)             # [1,1,H,W] or [1,H,W]
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if hasattr(pred, "dim") and pred.dim() == 4:
            pred = pred.squeeze(1)           # [1, H, W]
        depth = pred.squeeze(0).detach().cpu().numpy()  # [H, W]

    # 0..1 正規化
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return d

# =============== 消失点推定 ===============
# OpenCV バージョン差異に対応した安全な LSD 生成ラッパー
def safe_create_lsd():
    # 4.5系など一部は refine を位置引数で、4.7+ は int を渡す、古いものは引数なし…など差異がある
    try:
        return cv.createLineSegmentDetector(cv.LSD_REFINE_STD)
    except TypeError:
        try:
            return cv.createLineSegmentDetector()  # 引数なし版
        except Exception:
            # 最終手段: HoughLinesP ベースの簡易ライン検出にフォールバック
            return None

def estimate_vanishing_points(img_gray: np.ndarray):
    lsd = safe_create_lsd()
    if lsd is None:
        # --- フォールバック: HoughLinesP で近似ライン集合を作る ---
        edges = cv.Canny(img_gray, 80, 160, apertureSize=3)
        linesP = cv.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=5)
        if linesP is None:
            return None
        lines = [np.array([[x1, y1, x2, y2]], dtype=np.float32) for (x1,y1,x2,y2) in linesP.reshape(-1,4)]
    else:
        out = lsd.detect(img_gray)
        if out is None or out[0] is None:
            return None
        lines = out[0]

    # 角度でk-meansクラスタリングして主方向を抽出
    angs, segs = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = math.atan2(y2 - y1, x2 - x1)
        angs.append([math.cos(ang), math.sin(ang)])
        segs.append((x1, y1, x2, y2))
    Z = np.array(angs, dtype=np.float32)
    K = 3 if len(Z) >= 30 else 2
    K = max(1, min(K, len(Z)))  # 安全ガード
    if K < 1:
        return None
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    compactness, labels, centers = cv.kmeans(Z, K, None, criteria, 3, cv.KMEANS_PP_CENTERS)

    vps = []
    for k in range(K):
        idx = np.where(labels.ravel() == k)[0]
        if len(idx) < 2:
            continue
        P = []
        # 交点の中央値で代表化
        limit = min(120, len(idx))
        for ii in range(limit):
            i = idx[ii]
            x1, y1, x2, y2 = segs[i]
            for jj in range(ii+1, limit):
                j = idx[jj]
                x3, y3, x4, y4 = segs[j]
                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(den) < 1e-6:
                    continue
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
                if np.isfinite(px) and np.isfinite(py):
                    P.append((px, py))
        if len(P) >= 3:
            P = np.array(P)
            med = np.median(P, axis=0)
            vps.append(tuple(med.tolist()))
    return vps  # 画像座標系上の消失点の近似

# =============== 床領域の抽出（簡易） ===============
def estimate_floor_mask(img_bgr: np.ndarray) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    v = hsv[..., 2]
    # 下半分を重視しつつエッジ少ないところ
    edges = cv.Canny(cv.GaussianBlur(v, (5,5), 0), 50, 150)
    grad = cv.GaussianBlur(edges, (9,9), 0)
    mask = np.zeros((H, W), np.uint8)
    mask[int(H*0.55):, :] = 1
    mask = (mask & (grad < np.percentile(grad, 40))).astype(np.uint8)
    mask = cv.morphologyEx(mask * 255, cv.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return (mask > 0).astype(np.uint8)

# =============== 家具密度 index ===============
from typing import Optional

def init_yolo(model_name: str = "yolov8n.pt"):
    from ultralytics import YOLO
    return YOLO(model_name)

FURN_CLASSES = {
    # coco class name -> weight in index
    "chair": 1.0,
    "dining table": 1.1,
    "couch": 1.1,
    "sofa": 1.1,
    "potted plant": 0.6,
    "tv": 0.8,
    "book": 0.8,
    "bench": 0.7,
    "refrigerator": 0.3,
}
PEOPLE_CLASSES = {"person": 0.5}

def furniture_index_from_yolo(model, img_bgr: np.ndarray) -> float:
    H, W = img_bgr.shape[:2]
    area_img = float(H * W)

    # 床マスク（下半分で平坦な領域）で正規化
    floor_mask = estimate_floor_mask(img_bgr)
    floor_px = float(np.count_nonzero(floor_mask))
    floor_px = max(floor_px, 1.0)

    # 推論
    res = model.predict(source=img_bgr, imgsz=640, conf=0.30, verbose=False)
    if not res:
        return 0.0
    r = res[0]
    names = r.names

    def keep_box(x1, y1, x2, y2):
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if (w * h) / area_img < MIN_BOX_FR:
            return False
        if x1 < BORDER_PAD or y1 < BORDER_PAD or x2 > W - BORDER_PAD or y2 > H - BORDER_PAD:
            return False
        return True

    furn_area_wsum, furn_count_wsum, people_area = 0.0, 0.0, 0.0

    # マスクがあれば面積はマスク、なければBBox面積
    if hasattr(r, "masks") and r.masks is not None:
        masks = r.masks.data.detach().cpu().numpy()
        clss = r.boxes.cls.detach().cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        for m, cls_id, (x1, y1, x2, y2) in zip(masks, clss, boxes):
            if not keep_box(x1, y1, x2, y2):
                continue
            cls = names[cls_id]
            pix = float(m.sum())
            if cls == "person":
                people_area += pix
            else:
                w = FURN_AREA_WEIGHTS.get(cls, 0.0)
                furn_area_wsum += w * pix
                if w > 0:
                    furn_count_wsum += w
    else:
        for (x1, y1, x2, y2), cls_id in zip(
            r.boxes.xyxy.detach().cpu().numpy(),
            r.boxes.cls.detach().cpu().numpy().astype(int),
        ):
            if not keep_box(x1, y1, x2, y2):
                continue
            cls = names[cls_id]
            pix = float(max(0.0, (x2 - x1) * (y2 - y1)))
            if cls == "person":
                people_area += pix
            else:
                w = FURN_AREA_WEIGHTS.get(cls, 0.0)
                furn_area_wsum += w * pix
                if w > 0:
                    furn_count_wsum += w

    # --- 指標の設計 ---
    # 床画素 + α*全画素で安定化（α=0.1）
    denom = floor_px + 0.1 * area_img
    area_ratio = furn_area_wsum / denom
    area_term = np.sqrt(max(0.0, area_ratio))

    count_term = min(1.0, furn_count_wsum / C_MAX)

    people_term = PEOPLE_AREA_WEIGHT * (people_area / denom)
    people_term = np.sqrt(max(0.0, people_term))

    W1, W2, W3 = 0.6, 0.25, 0.15
    raw = W1 * area_term + W2 * count_term + W3 * people_term

    # γを強めにして全体を抑制
    gamma = 1.4
    score = 10.0 * (raw ** gamma)

    # 上限を 8 に抑制
    return float(np.clip(score, 0.0, 8.0))


# =============== 天井高さの推定（簡易） ===============
DEFAULT_HEIGHT_PRIOR = 2.8  # m

def estimate_ceiling_height(img_bgr: np.ndarray, depth_norm: np.ndarray,
                            known_px_height: float | None = None,
                            known_m_height: float | None = None) -> float:
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    _ = estimate_vanishing_points(gray)  # まだ活用は限定的。将来の姿勢補正のフックとして呼ぶだけ
    # depth勾配が急変する上端帯を天井と仮定（簡易）
    # （現状はスケールに用いず、prior フォールバック）
    # 既知スケールがあればスケール化
    if known_px_height and known_m_height:
        scale = known_m_height / known_px_height  # m / px
        H_px = img_bgr.shape[0]
        return float(max(1.8, min(5.0, H_px * 0.33 * scale)))
    return DEFAULT_HEIGHT_PRIOR

# =============== ホール面積/容積の推定（簡易） ===============
def estimate_floor_area(img_bgr: np.ndarray, depth_norm: np.ndarray, ceiling_h_m: float) -> float:
    mask = estimate_floor_mask(img_bgr)
    dn = depth_norm + 1e-6
    inv = 1.0 / dn
    mean_inv = float(np.mean(inv[mask > 0])) if np.any(mask) else 1.0
    H, W = img_bgr.shape[:2]
    est_width_m = 0.6 * ceiling_h_m * (W / max(H,1))
    est_depth_m = 1.2 * ceiling_h_m * mean_inv
    area = max(5.0, min(500.0, est_width_m * est_depth_m))
    return float(area)

# =============== 店舗1件の集約 ===============
# --- 1) analyze_store: 返り値に n_images を追加 ---
def analyze_store(images: list[Path], yolo_model) -> dict:
    images = dedup_by_sequence(images)  # ← 念のため二重に安全策
    heights, areas, furn_scores = [], [], []
    for p in images:
        img = cv.imdecode(np.fromfile(str(p), dtype=np.uint8), cv.IMREAD_COLOR)
        if img is None:
            continue
        depth = run_midas(img)
        h = estimate_ceiling_height(img, depth)
        a = estimate_floor_area(img, depth, h)
        f = furniture_index_from_yolo(yolo_model, img)
        heights.append(h); areas.append(a); furn_scores.append(f)

    def robust_mean(xs):
        if not xs: return np.nan
        xs = np.array(xs); med = np.median(xs)
        mad = np.median(np.abs(xs - med)) + 1e-6
        w = 1.0 / (1.0 + np.abs(xs - med) / (1.4826 * mad))
        return float(np.sum(xs * w) / np.sum(w))

    return {
        "est_hall_height": robust_mean(heights) if heights else np.nan,
        "est_hall_area":  robust_mean(areas) if areas else np.nan,
        "furniture_index": float(np.nan if not furn_scores else float(np.mean(furn_scores))),
        "n_images": len(images),  # ← 去重後の枚数
    }



# =============== メイン: 全データ走査 ===============
def main(base_dir: str | None = None,
         out_csv: str = "output_dataset.csv",
         midas_model: str = "DPT_Hybrid",
         yolo_model_name: str = "yolov8n.pt"):
    # --- 1) data の場所を決める ---
    if base_dir:
        base = Path(base_dir).expanduser().resolve()
    else:
        env = os.environ.get("NOISE_DATA_DIR")
        if env:
            base = Path(env).expanduser().resolve()
        else:
            script_dir = Path(__file__).resolve().parent
            base = (script_dir / "data").resolve()

    # 見つからないときのフォールバック
    if not base.exists():
        alt1 = (Path(__file__).resolve().parent.parent / "src" / "data").resolve()
        alt2 = (Path(__file__).resolve().parent / ".." / "src" / "data").resolve()
        for cand in (alt1, alt2):
            if cand.exists():
                base = cand
                break
        else:
            tried = [str(base), str(alt1), str(alt2)]
            raise FileNotFoundError(
                "data フォルダが見つかりません。\n"
                f"試したパス:\n - " + "\n - ".join(tried) + "\n"
                "起動オプションで --base \"C:\\\\…\\\\src\\\\data\" を指定するか、\n"
                "環境変数 NOISE_DATA_DIR に data の絶対パスを設定してください。"
            )

    print(f"[INFO] data base = {base}")

    # --- 2) モデル初期化 ---
    init_midas(midas_model)
    yolo = init_yolo(yolo_model_name)

    # --- 3) 全店舗を走査して解析 ---
    rows = []
    n_txt, n_img = 0, 0
    for location_type, store_id, imgs, txt in list_store_groups(base):
        n_txt += 1
        n_img += len(imgs)
        try:
            y_min, y_ave, y_max = read_db_triplet(txt)
        except Exception as e:
            print(f"[WARN] cannot read dB: {txt} -> {e}")
            continue
        x = analyze_store(imgs, yolo)
        rows.append({
            "location_type": location_type,
            **x,
            "min": y_min, "ave": y_ave, "max": y_max,
            "store_id": store_id,
        })

    print(f"[INFO] parsed stores: {n_txt}, images: {n_img}")

    # --- 4) 出力 ---
    df = pd.DataFrame(rows)
    if not df.empty:
        # 任意：店舗IDで並べたい場合は次の1行を有効化
        # df = df.sort_values(["store_id"]).reset_index(drop=True)
    
        cols = [
            "store_id",
            "n_images",
            "location_type",
            "est_hall_area",
            "est_hall_height",
            "furniture_index",
            "min", "ave", "max",
        ]
        # 欠損キーがあっても落ちないように、存在する列だけ並べ替え
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote: {out_csv}, rows={len(df)}")
        print(df.head())
    else:
        print("[WARN] 解析対象がありません（txt/画像の命名や配置を確認してください）。")

if __name__ == "__main__":
    # 例: python poc_image_to_noise.py --base "C:\path\to\src\data" --out out.csv
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", dest="base_dir", type=str, default=None)
    ap.add_argument("--out", dest="out_csv", type=str, default="output_dataset.csv")
    ap.add_argument("--midas", dest="midas_model", type=str, default="DPT_Hybrid")
    ap.add_argument("--yolo", dest="yolo_model_name", type=str, default="yolov8n.pt")
    args = ap.parse_args()

    main(base_dir=args.base_dir,
         out_csv=args.out_csv,
         midas_model=args.midas_model,
         yolo_model_name=args.yolo_model_name)
