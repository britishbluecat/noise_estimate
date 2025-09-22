# Noise Estimation PoC

本リポジトリは、**店舗内の音響環境（騒音レベル）を画像と短時間録音から推定する PoC** 実装です。  
カフェ・レストラン・ラーメン店など、`src/data/{location_type}/` 以下にデータを配置し、音場特性をデータセット化します。

---

## 📂 データ構造

```plaintext
src/data/
├── cafe/
│   ├── aaa.txt        # dB値 (CSV形式: min,ave,max)
│   ├── aaa_001.jpg    # 店舗写真
│   ├── aaa_002.png
│   └── ...
├── ramen/
│   ├── bbb.txt
│   ├── bbb_001.jpg
│   └── ...
└── other/
    └── ...

- `*.txt`: 騒音測定値。例:
  ```csv
  min,ave,max
  56,64,79
*_NNN.jpg|png: 店舗の複数枚画像

pip install opencv-python numpy pandas torch torchvision ultralytics scikit-image matplotlib timm

## 🚀 実行方法

cd src
python poc_image_to_noise.py --base "./data" --out output_dataset.csv

## 📊 出力フォーマット

実行結果は CSV として保存されます。

| store_id | n_images | location_type | est_hall_area | est_hall_height | furniture_index | min | ave | max |
|----------|----------|---------------|---------------|-----------------|-----------------|-----|-----|-----|
| aaa      | 5        | cafe          | 10.0          | 2.8             | 5.7             | 56  | 64  | 79  |


store_id: 店舗ID（txtファイル名）

n_images: 使用画像枚数（拡張子重複は去重済み）

location_type: フォルダ名 (cafe, ramen など)

est_hall_area: ホール面積推定値 (㎡, 簡易)

est_hall_height: 天井高さ推定値 (m, 簡易)

furniture_index: 家具密度指標 (0〜8, 椅子/机/人混み等から算出)

min, ave, max: 騒音測定値 (dB)

## 🧩 仕組み

画像解析

MiDaS による単眼深度推定

消失点近似による室内構造推定

YOLOv8 による椅子/机/人物検出 → 家具密度 index

床マスク推定で相対面積補正

スケール合わせ

既知スケール物体（例: A4メニュー）の手動指定が可能

無い場合は高さの事前分布 (2.6〜3.0m) を使用

集約

## 📌 今後の改良ポイント

SfM (Structure from Motion) を用いた複数画像の3D統合

部屋全体の容積推定

録音波形特徴量（RT60, 周波数帯ノイズ）との統合

Web UI による簡易アップロード&解析

複数画像の堅牢平均（outlier耐性）

家具密度の平均化

