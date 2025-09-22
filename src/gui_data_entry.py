import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import os, shutil, csv
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
UNDO_STACK = []
selected_images = []
OUTPUT_CSV = Path(__file__).resolve().parent / "output_dataset.csv"

def refresh_categories(combo):
    combo['values'] = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

def save_data():
    category = category_var.get().strip()
    store_id = store_var.get().strip()
    n_images = int(n_images_var.get())
    db_min, db_ave, db_max = min_var.get(), ave_var.get(), max_var.get()

    if not category or not store_id:
        messagebox.showerror("エラー", "カテゴリと店舗IDを入力してください")
        return
    if not selected_images:
        messagebox.showerror("エラー", "画像を選択してください")
        return

    target_dir = DATA_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)

    txt_path = target_dir / f"{store_id}.txt"
    with open(txt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["min", "ave", "max"])
        writer.writerow([db_min, db_ave, db_max])

    copied_imgs = []
    for i, img in enumerate(selected_images[:n_images], start=1):
        ext = os.path.splitext(img)[1].lower()
        dest = target_dir / f"{store_id}_{i:03d}{ext}"
        shutil.copy(img, dest)
        copied_imgs.append(dest)

    UNDO_STACK.append((txt_path, copied_imgs))
    messagebox.showinfo("保存完了", f"{store_id} を保存しました。")

def reset_fields():
    store_var.set("")
    min_var.set("")
    ave_var.set("")
    max_var.set("")
    global selected_images
    selected_images = []
    messagebox.showinfo("リセット", "入力内容をリセットしました。")

def undo_last():
    if not UNDO_STACK:
        messagebox.showwarning("Undo", "取り消す操作がありません。")
        return
    txt_path, imgs = UNDO_STACK.pop()
    if txt_path.exists():
        txt_path.unlink()
    for img in imgs:
        if img.exists():
            img.unlink()
    messagebox.showinfo("Undo", "直前の保存を取り消しました。")

def select_images():
    global selected_images
    files = filedialog.askopenfilenames(
        title="画像を選択",
        filetypes=[("画像ファイル", "*.jpg *.jpeg *.png")]
    )
    if files:
        selected_images = list(files)
        messagebox.showinfo("画像選択", f"{len(selected_images)} 枚の画像を選択しました。")

# ================== GUI ==================
root = tk.Tk()
root.title("Noise Data Entry")
root.geometry("700x500")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# ========== メインタブ ==========
frame_main = tk.Frame(notebook)
frame_main.pack(fill="both", expand=True)
notebook.add(frame_main, text="メイン")

def add_separator(frame, row):
    sep = ttk.Separator(frame, orient="horizontal")
    sep.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)

# カテゴリ選択
tk.Label(frame_main, text="カテゴリ:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
category_var = tk.StringVar()
category_combo = ttk.Combobox(frame_main, textvariable=category_var, state="readonly", width=20)
refresh_categories(category_combo)
category_combo.grid(row=0, column=1, padx=5, pady=5)
tk.Button(frame_main, text="新規カテゴリ", 
          command=lambda: (new := simpledialog.askstring("新規カテゴリ", "カテゴリ名:")) 
                          and (DATA_DIR / new).mkdir(exist_ok=True) 
                          or refresh_categories(category_combo)).grid(row=0, column=2, padx=5, pady=5)

add_separator(frame_main, 1)

# 店舗ID
tk.Label(frame_main, text="店舗ID:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
store_var = tk.StringVar()
tk.Entry(frame_main, textvariable=store_var, width=22).grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")

add_separator(frame_main, 3)

# 画像数選択
tk.Label(frame_main, text="最大画像枚数:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
n_images_var = tk.StringVar(value="5")
ttk.Combobox(frame_main, textvariable=n_images_var, values=["5", "10"], state="readonly", width=5).grid(row=4, column=1, padx=5, pady=5, sticky="w")
tk.Button(frame_main, text="画像選択", command=select_images).grid(row=4, column=2, padx=5, pady=5)

add_separator(frame_main, 5)

# dB値
tk.Label(frame_main, text="min/ave/max (dB):").grid(row=6, column=0, sticky="e", padx=5, pady=5)
min_var, ave_var, max_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
tk.Entry(frame_main, textvariable=min_var, width=6).grid(row=6, column=1, sticky="w", padx=2)
tk.Entry(frame_main, textvariable=ave_var, width=6).grid(row=6, column=1, padx=2)
tk.Entry(frame_main, textvariable=max_var, width=6).grid(row=6, column=1, sticky="e", padx=2)

add_separator(frame_main, 7)

# ボタン
tk.Button(frame_main, text="保存", command=save_data, width=10).grid(row=8, column=0, pady=10)
tk.Button(frame_main, text="リセット", command=reset_fields, width=10).grid(row=8, column=1, pady=10)
tk.Button(frame_main, text="Undo", command=undo_last, width=10).grid(row=8, column=2, pady=10)

# ========== データタブ ==========
frame_data = tk.Frame(notebook)
frame_data.pack(fill="both", expand=True)
notebook.add(frame_data, text="データ")

# TreeviewでCSVプレビュー
tree = ttk.Treeview(frame_data, show="headings")
tree.pack(fill="both", expand=True, padx=10, pady=10)

def load_csv_preview():
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    preview = df.head(5)
    # カラム設定
    tree["columns"] = list(preview.columns)
    for col in preview.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")
    # 行データ挿入
    for i, row in preview.iterrows():
        tree.insert("", "end", values=list(row))

load_csv_preview()

root.mainloop()
