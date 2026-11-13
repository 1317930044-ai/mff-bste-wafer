import argparse
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageGrab, ImageTk

from wafer.common import SYNTHETIC, labels
from wafer.data import PALETTE, read_map
from wafer.inference import available_models, available_samples, load_model, predict

BG, INK, TEAL = "#F4F8FA", "#17222F", "#127C80"
TASKS = {"8 类缺陷": "defect8", "9 类（含正常）": "all9"}


class Desktop:
    def __init__(self, root):
        self.root = root
        root.title("晶圆缺陷识别 · MFF-BSTE")
        root.geometry("1020x670")
        root.minsize(840, 580)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=BG, bordercolor="#CDDDE5", padding=12)
        style.configure("TLabelframe.Label", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 9))
        style.configure("Heading.TLabel", font=("Microsoft YaHei UI", 24, "bold"))
        style.configure("Prediction.TLabel", foreground=TEAL, font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("Accent.TButton", background=TEAL, foreground="white", padding=(15, 7))
        style.map("Accent.TButton", background=[("active", "#0C6064")])
        style.configure("Horizontal.TProgressbar", troughcolor="#DDE8EF", background=TEAL)
        self.path = None
        self.model_list, self.samples = [], []
        self.last_result = None
        self.photos = []

        main = ttk.Frame(root, padding=(24, 16))
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="晶圆缺陷识别", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(main, text="多视角特征融合 · 决策树集成 · 本地 CPU 推理").pack(anchor="w", pady=(0, 12))
        form = ttk.LabelFrame(main, text="模型与输入")
        form.pack(fill="x")
        form.columnconfigure(3, weight=1)
        ttk.Label(form, text="分类任务").grid(row=0, column=0, padx=(0, 7), sticky="w")
        self.task = ttk.Combobox(form, values=tuple(TASKS), width=16, state="readonly")
        self.task.current(0)
        self.task.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self.task.bind("<<ComboboxSelected>>", self.refresh_models)
        ttk.Label(form, text="模型").grid(row=0, column=2, padx=(0, 7), sticky="w")
        self.model = ttk.Combobox(form, state="readonly")
        self.model.grid(row=0, column=3, sticky="ew")
        self.model.bind("<<ComboboxSelected>>", self.show_model_source)
        ttk.Label(form, text="项目样本").grid(row=1, column=0, pady=8, sticky="w")
        self.sample = ttk.Combobox(form, state="readonly")
        self.sample.grid(row=1, column=1, columnspan=3, sticky="ew")
        self.sample.bind("<<ComboboxSelected>>", self.select_sample)
        ttk.Button(form, text="打开文件…", command=self.open_file).grid(row=2, column=0, columnspan=2, sticky="w")
        self.file_label = ttk.Label(form, text="支持 NPY、NPZ 和项目 PNG")
        self.file_label.grid(row=2, column=2, columnspan=2, sticky="w")
        self.source = ttk.Label(form, text="")
        self.source.grid(row=3, column=0, columnspan=3, sticky="w", pady=(7, 0))
        ttk.Button(form, text="识别晶圆", style="Accent.TButton", command=self.identify).grid(row=3, column=3, sticky="e", pady=(8, 0))

        body = ttk.Frame(main)
        body.pack(fill="both", expand=True, pady=(13, 0))
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)
        images = ttk.LabelFrame(body, text="晶圆与掩码")
        images.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        images.columnconfigure((0, 1), weight=1)
        ttk.Label(images, text="原始状态").grid(row=0, column=0, pady=6)
        ttk.Label(images, text="缺陷掩码").grid(row=0, column=1, pady=6)
        self.raw_view = ttk.Label(images, text="请选择晶圆")
        self.mask_view = ttk.Label(images, text="等待预览")
        self.raw_view.grid(row=1, column=0, padx=4)
        self.mask_view.grid(row=1, column=1, padx=4)
        self.die_count = ttk.Label(images, text="")
        self.die_count.grid(row=2, column=0, columnspan=2, pady=10)

        results = ttk.LabelFrame(body, text="识别结果")
        results.grid(row=0, column=1, sticky="nsew")
        self.prediction = ttk.Label(results, text="等待识别", style="Prediction.TLabel")
        self.prediction.pack(anchor="w")
        self.timing = ttk.Label(results, text="")
        self.timing.pack(anchor="w", pady=(0, 8))
        self.probabilities = []
        for _ in range(3):
            label = ttk.Label(results, text="—")
            label.pack(anchor="w")
            bar = ttk.Progressbar(results, maximum=100)
            bar.pack(fill="x", pady=(0, 6))
            self.probabilities.append((label, bar))
        self.explanation = ttk.Label(results, text="", wraplength=340, justify="left")
        self.explanation.pack(anchor="w", pady=(5, 0))
        self.refresh_models()

    def refresh_models(self, _event=None):
        task = TASKS[self.task.get()]
        self.model_list = available_models(task)
        self.model["values"] = [f"{path.stem} · {path.parent.parent.name}" for path, _ in self.model_list]
        self.model.set("")
        if self.model_list:
            self.model.current(0)
        self.samples = [row for row in available_samples() if row["label"] in labels(task)]
        self.sample["values"] = [f"{r['label']} · {r['id']} · {r['source']}" for r in self.samples]
        self.sample.set("")
        self.show_model_source()

    def clear_prediction(self):
        self.last_result = None
        self.prediction.configure(text="等待识别")
        self.timing.configure(text="")
        self.explanation.configure(text="")
        for label, bar in self.probabilities:
            label.configure(text="—")
            bar["value"] = 0

    def show_model_source(self, _event=None):
        i = self.model.current()
        self.source.configure(text=f"训练来源 · {self.model_list[i][1]}" if i >= 0 else "请先训练。该任务暂无已完成模型。")
        self.clear_prediction()

    def select_sample(self, _event=None):
        if self.sample.current() >= 0:
            self.set_file(self.samples[self.sample.current()]["path"])

    def open_file(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("规范晶圆", "*.npy *.npz *.png"), ("所有文件", "*.*")])
        if path:
            self.sample.set("")
            self.set_file(Path(path))

    def set_file(self, path, strict=False):
        self.clear_prediction()
        try:
            a = read_map(path)
        except (ValueError, OSError, TypeError, EOFError) as exc:
            if strict:
                raise
            messagebox.showerror("文件读取失败", str(exc), parent=self.root)
            self.path = None
            self.file_label.configure(text="输入无效，请选择规范晶圆文件")
            self.raw_view.configure(image="", text="请选择晶圆")
            self.mask_view.configure(image="", text="等待预览")
            self.die_count.configure(text="")
            return
        self.path = path
        self.file_label.configure(text=path.name)
        self.show_maps(a)

    def show_maps(self, a):
        masks = (PALETTE[a], (a == 2).astype(np.uint8) * 255)
        photos = []
        for pixels in masks:
            image = Image.fromarray(pixels)
            factor = 205 / max(image.size)
            image = image.resize(tuple(max(1, round(side*factor)) for side in image.size), Image.Resampling.NEAREST)
            photos.append(ImageTk.PhotoImage(image, master=self.root))
        self.photos = photos
        self.raw_view.configure(image=photos[0], text="")
        self.mask_view.configure(image=photos[1], text="")
        self.die_count.configure(text=f"{a.shape[0]} × {a.shape[1]} 晶粒 · 有效 {(a != 0).sum():,} · 缺陷 {(a == 2).sum():,}")

    def identify(self, model_path=None, strict=False):
        try:
            if self.path is None or (model_path is None and self.model.current() < 0):
                raise ValueError("请先选择晶圆文件和已训练模型。")
            bundle = load_model(model_path or self.model_list[self.model.current()][0])
            w, result = predict(bundle, path=self.path)
        except (ValueError, OSError, ImportError, TypeError, EOFError) as exc:
            if strict:
                raise
            messagebox.showerror("无法识别", str(exc), parent=self.root)
            return
        self.show_maps(w.a)
        self.last_result = result
        self.source.configure(text=f"训练来源 · {SYNTHETIC if result['source'] == 'synthetic' else 'WM-811K'}")
        self.prediction.configure(text=result["prediction"])
        self.timing.configure(text=f"端到端耗时  {result['elapsed_ms']:.2f} ms")
        for (name, p), (label, bar) in zip(result["top3"], self.probabilities):
            label.configure(text=f"{name}  ·  {p:.1%}")
            bar["value"] = 100*p
        self.explanation.configure(text="\n".join(f"{row['description']}  {row['value']:.3f}" for row in result["explanations"][:4]))


def main():
    parser = argparse.ArgumentParser(description="Tkinter 桌面晶圆识别")
    parser.add_argument("--smoke", action="store_true", help="打开真实窗口、识别指定样本后退出")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--screenshot", type=Path, help="在交互式 Windows 桌面上保存当前窗口画面")
    args = parser.parse_args()
    if args.smoke and not (args.model_path and args.input):
        parser.error("--smoke 需要 --model-path 和 --input")
    root = tk.Tk()
    app = Desktop(root)
    if not args.smoke:
        root.mainloop()
        return
    try:
        root.update()
        for i, (path, _) in enumerate(app.model_list):
            if path.resolve() == args.model_path.resolve():
                app.model.current(i)
                app.show_model_source()
                break
        app.set_file(args.input, strict=True)
        app.identify(args.model_path, strict=True)
        root.update()
        if app.prediction.cget("text") != app.last_result["prediction"]:
            raise RuntimeError("桌面结果组件未更新。")
        if args.screenshot:
            root.lift()
            root.attributes("-topmost", True)
            root.update()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                from ctypes import windll, wintypes

                get_ancestor = windll.user32.GetAncestor
                get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
                get_ancestor.restype = wintypes.HWND
                hwnd = get_ancestor(root.winfo_id(), 2)
                ImageGrab.grab(window=hwnd).save(args.screenshot)
            else:
                x, y = root.winfo_rootx(), root.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x+root.winfo_width(), y+root.winfo_height())).save(args.screenshot)
        print(json.dumps({k: app.last_result[k] for k in ("prediction", "top3", "model", "run_id")}, ensure_ascii=True))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
