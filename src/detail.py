"""Cửa sổ chi tiết một bình luận: dự đoán, tiền xử lý và heatmap self-attention."""

import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import numpy as np

from .widgets import (BG, BORDER, BRAND, EMPTY_MSG, MUTED, SURFACE, LABEL_VI, LABELS, TEXT, ProbBars, ScrollFrame,
                     brand_shade, dec, display, label_badge, pct, setup_theme)

AVG = "Trung bình các head"


class AttentionView(tk.Frame):
    """Chọn layer/head, tô màu token theo attention nhận được và vẽ heatmap Query × Key."""

    def __init__(self, master, tokens, is_unk, attn, **kw):
        super().__init__(master, bg=BG, **kw)
        self.tokens, self.is_unk, self.attn = tokens, is_unk, attn  # attn: (layer, head, T, T)
        n_layers, n_heads = attn.shape[:2]
        self.font = tkfont.nametofont("TkDefaultFont")
        self.small = tkfont.Font(size=max(self.font.cget("size") - 2, 9))

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x")
        tk.Label(bar, text="Self-attention", bg=BG, font=("TkDefaultFont", 13, "bold")).pack(side="left")
        self.head_var = tk.StringVar(value=AVG)
        self.layer_var = tk.StringVar(value=f"Layer {n_layers}")
        for var, values, width in (
            (self.head_var, [AVG] + [f"Head {i + 1}" for i in range(n_heads)], 20),
            (self.layer_var, [f"Layer {i + 1}" for i in range(n_layers)], 9),
        ):
            box = ttk.Combobox(bar, textvariable=var, values=values, state="readonly", width=width)
            box.pack(side="right", padx=(6, 0))
            box.bind("<<ComboboxSelected>>", lambda _: self.redraw())

        self.chips = tk.Canvas(self, bg=BG, highlightthickness=0, height=40)
        self.chips.pack(fill="x", pady=(10, 2))
        self.chips.bind("<Configure>", lambda _: self.draw_chips())
        tk.Label(self, bg=BG, fg=MUTED, font=self.small, justify="left", anchor="w",
                 text="Nền càng đậm thì token càng được các token khác chú ý nhiều. "
                      "Viền nét đứt là từ không có trong vocab (mã hoá thành <unk>).").pack(fill="x")

        self.heat = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.heat.pack(anchor="w", pady=(10, 4))
        self.heat.bind("<Motion>", self.on_hover)
        self.heat.bind("<Leave>", lambda _: self.set_hover(None))
        self.hover_text = tk.Label(self, bg=BG, anchor="w", text=" ")
        self.hover_text.pack(fill="x")
        self.redraw()

    def matrix(self) -> np.ndarray:
        layer = self.attn[int(self.layer_var.get().split()[1]) - 1]
        head = self.head_var.get()
        return layer.mean(axis=0) if head == AVG else layer[int(head.split()[1]) - 1]

    def redraw(self):
        self.m = self.matrix()
        # Mức attention trung bình mỗi token nhận được từ các token khác (trung bình theo cột).
        received = self.m.mean(axis=0)
        self.received = received / received.max()
        self.draw_chips()
        self.draw_heatmap()

    def draw_chips(self):
        c = self.chips
        c.delete("all")
        width = max(c.winfo_width(), 200)
        x, y, h, gap = 0, 2, self.font.metrics("linespace") + 8, 6
        for t, unk, r in zip(self.tokens, self.is_unk, self.received):
            w = self.font.measure(display(t)) + 16
            if x + w > width and x > 0:
                x, y = 0, y + h + gap
            c.create_rectangle(x, y, x + w, y + h, fill=brand_shade(0.06 + 0.7 * r),
                               outline=MUTED if unk else brand_shade(0.25), dash=(4, 3) if unk else None,
                               width=2 if unk else 1)
            c.create_text(x + w / 2, y + h / 2, text=display(t), fill="white" if r > 0.6 else TEXT)
            x += w + gap
        c.configure(height=y + h + 4)

    def draw_heatmap(self):
        c, n = self.heat, len(self.tokens)
        c.delete("all")
        self.cell = 14 if n > 30 else 20 if n > 16 else 30
        labels = [display(t) for t in self.tokens]
        longest = max(self.small.measure(t) for t in labels)
        self.left = longest + 10
        # Nhãn cột xoay 60° để không làm cột rộng ra theo độ dài chữ.
        self.top = int(longest * math.sin(math.radians(60))) + 12
        mx = self.m.max()
        for k, t in enumerate(labels):
            c.create_text(self.left + k * self.cell + self.cell / 2, self.top - 4, text=t, angle=60,
                          anchor="w", font=self.small, fill=MUTED, tags=f"col{k}")
        for q, row in enumerate(self.m):
            c.create_text(self.left - 6, self.top + q * self.cell + self.cell / 2, text=labels[q],
                          anchor="e", font=self.small, fill=MUTED, tags=f"row{q}")
            for k, v in enumerate(row):
                x0, y0 = self.left + k * self.cell, self.top + q * self.cell
                c.create_rectangle(x0, y0, x0 + self.cell - 1, y0 + self.cell - 1,
                                   fill=brand_shade(0.04 + 0.96 * v / mx), outline="")
        width = self.left + n * self.cell + longest // 2 + 10
        c.configure(width=width, height=self.top + n * self.cell + 2)
        self.set_hover(None)

    def on_hover(self, event):
        q = (event.y - self.top) // self.cell
        k = (event.x - self.left) // self.cell
        n = len(self.tokens)
        self.set_hover((q, k) if 0 <= q < n and 0 <= k < n else None)

    def set_hover(self, cell):
        c = self.heat
        c.delete("hover")
        for i in range(len(self.tokens)):
            c.itemconfigure(f"row{i}", fill=MUTED)
            c.itemconfigure(f"col{i}", fill=MUTED)
        if cell is None:
            self.hover_text.configure(text="Rê chuột lên heatmap để xem trọng số (hàng = Query, cột = Key).", fg=MUTED)
            return
        q, k = cell
        x0, y0 = self.left + k * self.cell, self.top + q * self.cell
        c.create_rectangle(x0, y0, x0 + self.cell - 1, y0 + self.cell - 1, outline="#f59e0b", width=2, tags="hover")
        c.itemconfigure(f"row{q}", fill=BRAND)
        c.itemconfigure(f"col{k}", fill=BRAND)
        self.hover_text.configure(
            fg=TEXT,
            text=f"Query “{display(self.tokens[q])}” chú ý vào Key “{display(self.tokens[k])}”: "
                 f"trọng số {dec(float(self.m[q, k]), 3)}")


class DetailWindow(tk.Toplevel):
    """Hiện ngay phần bình luận gốc; phần phân tích được điền khi `show()` nhận kết quả."""

    def __init__(self, master, text: str, time_str: str):
        super().__init__(master, bg=BG)
        setup_theme(self)
        self.title("Chi tiết bình luận")
        self.geometry("860x780")
        self.minsize(600, 400)
        self.bind("<Escape>", lambda _: self.destroy())

        scroll = ScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        self.body = tk.Frame(scroll.inner, bg=BG, padx=20, pady=16)
        self.body.pack(fill="both", expand=True)

        tk.Label(self.body, text=f"Bình luận gốc · {time_str}", bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        tk.Label(self.body, text=text, bg=SURFACE, anchor="w", justify="left", wraplength=780,
                 padx=14, pady=10).pack(fill="x", pady=(4, 12))
        self.status = tk.Label(self.body, text="Đang phân tích...", bg=BG, fg=MUTED, anchor="w")
        self.status.pack(fill="x")

    def section(self, title=None) -> tk.Frame:
        box = tk.Frame(self.body, bg=BG, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
        box.pack(fill="x", pady=6)
        if title:
            tk.Label(box, text=title, bg=BG, font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", pady=(0, 6))
        return box

    def show(self, result, error=None):
        if not self.winfo_exists():
            return
        if error:
            self.status.configure(text=f"Lỗi: {error}", fg="#dc2626")
            return
        if result is None:
            self.status.configure(text=EMPTY_MSG, fg="#b45309")
            return
        self.status.destroy()
        probs = result["probs"]
        pred = LABELS[int(np.argmax(probs))]

        box = self.section()
        head = tk.Frame(box, bg=BG)
        head.pack(fill="x")
        label_badge(head, pred, size=14).pack(side="left")
        tk.Label(head, text=f"{LABEL_VI[pred]} · độ tin cậy {pct(float(probs.max()))}", bg=BG).pack(side="left", padx=10)
        ProbBars(box, probs.tolist()).pack(anchor="w", pady=(10, 0))

        box = self.section("Tiền xử lý")
        grid = tk.Frame(box, bg=BG)
        grid.pack(fill="x")
        n_tok, max_len = result["n_tokens"], result["max_len"]
        rows = [
            ("Sau làm sạch", result["cleaned"]),
            ("Sau tách từ", result["segmented"]),
            ("Số token", f"{n_tok}" + (f" (vượt MAX_LEN, model chỉ đọc {max_len} token đầu)" if n_tok > max_len else "")),
            ("Token <unk>", f"{sum(result['is_unk'])}/{len(result['tokens'])} (từ không có trong vocab)"),
        ]
        for i, (k, v) in enumerate(rows):
            tk.Label(grid, text=k, bg=BG, fg=MUTED, anchor="nw").grid(row=i, column=0, sticky="nw", pady=2, padx=(0, 16))
            tk.Label(grid, text=v, bg=BG, anchor="w", justify="left", wraplength=620,
                     font="TkFixedFont" if k == "Sau tách từ" else None).grid(row=i, column=1, sticky="w", pady=2)

        box = self.section()
        AttentionView(box, result["tokens"], result["is_unk"], result["attn"]).pack(fill="x")
