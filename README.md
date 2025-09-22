# Noise Estimation PoC

本リポジトリは、**店舗内の音響環境（騒音レベル）を画像と短時間録音から推定する PoC** 実装です。  
カフェ・レストラン・ラーメン店など、`src/data/{location_type}/` 以下にデータを配置し、音場特性をデータセット化します。

---

## 📂 データ構造

src/data/
├── cafe/
│ ├── aaa.txt # dB値 (CSV形式: min,ave,max)
│ ├── aaa_001.jpg # 店舗写真
│ ├── aaa_002.png
│ └── ...
├── ramen/
│ ├── bbb.txt
│ ├── bbb_001.jpg
│ └── ...
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
