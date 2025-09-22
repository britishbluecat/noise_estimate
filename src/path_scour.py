import os
import pandas as pd
from pathlib import Path

base = Path("data")

rows = []

for loc_dir in base.iterdir():
    if not loc_dir.is_dir():
        continue
    location_type = loc_dir.name

    for txt_file in loc_dir.glob("*.txt"):
        store_id = txt_file.stem
        # dBファイル読み込み
        df_txt = pd.read_csv(txt_file)
        min_db, ave_db, max_db = df_txt.iloc[0]

        # 画像ファイル取得 (jpg, png 両方対象)
        img_files = list(loc_dir.glob(f"{store_id}_*.jpg")) + \
                    list(loc_dir.glob(f"{store_id}_*.png"))
        n_imgs = len(img_files)

        # --- 簡易推定 (PoC) ---
        est_hall_height = 2.7 + 0.1 * (n_imgs > 3)  # 例: 2.7m 基準
        est_hall_area = 20 + 5 * n_imgs             # 例: 画像枚数からざっくり
        furniture_index = min(10, n_imgs * 2)       # 例: 画像数比例で障害物多いと仮定

        rows.append({
            "location_type": location_type,
            "est_hall_area": est_hall_area,
            "est_hall_height": est_hall_height,
            "furniture_index": furniture_index,
            "min": min_db,
            "ave": ave_db,
            "max": max_db,
        })

df = pd.DataFrame(rows)
df.to_csv("output_dataset.csv", index=False, encoding="utf-8-sig")
print(df.head())
