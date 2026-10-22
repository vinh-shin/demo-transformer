"""Sidebar thông tin trường, đề tài và nhóm (cùng nội dung với sidebar bản web)."""

import tkinter as tk
from pathlib import Path

from .widgets import BG, BRAND, MUTED, SURFACE, brand_shade

LOGO = Path(__file__).resolve().parent / "assets" / "hcmute_logo.png"

SCHOOL = "TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM"
FACULTY = "Khoa Công nghệ Thông tin"
COURSE = "Deep Learning"
TOPIC = "Ứng dụng kỹ thuật Transformer trong bài toán phát hiện bình luận độc hại"
TEAM = {
    "name": "Nhóm 5",
    "supervisor": "PGS.TS. Hoàng Văn Dũng",
    "members": [
        ("2611318", "Vòng Vĩnh Shìn"),
        ("2611302", "Trần Nhật Duật"),
        ("2611320", "Hồ Đức Thắng"),
    ],
}

WIDTH = 260


class Sidebar(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, width=WIDTH, padx=16, pady=20, **kw)
        self.pack_propagate(False)
        wrap = WIDTH - 32

        if LOGO.exists():
            # Ảnh gốc 203 × 257, thu nhỏ một nửa cho gần cỡ bản web (96 px).
            self.logo = tk.PhotoImage(file=str(LOGO)).subsample(2)
            tk.Label(self, image=self.logo, bg=BG).pack()
        tk.Label(self, text=SCHOOL, bg=BG, font=("TkDefaultFont", 10, "bold"), wraplength=wrap,
                 justify="center").pack(pady=(8, 0))
        tk.Label(self, text=FACULTY, bg=BG, font=("TkDefaultFont", 10)).pack()

        course = tk.Frame(self, bg=SURFACE, padx=12, pady=10)
        course.pack(fill="x", pady=(16, 0))
        for title, value in (("Môn học", COURSE), ("Đề tài", TOPIC)):
            tk.Label(course, text=title, bg=SURFACE, fg=MUTED, anchor="w").pack(fill="x")
            tk.Label(course, text=value, bg=SURFACE, font=("TkDefaultFont", 11, "bold"), anchor="w",
                     justify="left", wraplength=wrap - 24).pack(fill="x", pady=(0, 6))

        tint = brand_shade(0.08)
        team = tk.Frame(self, bg=tint)
        team.pack(fill="x", pady=(12, 0))
        tk.Frame(team, width=4, bg=BRAND).pack(side="left", fill="y")
        box = tk.Frame(team, bg=tint, padx=12, pady=10)
        box.pack(side="left", fill="both", expand=True)
        tk.Label(box, text=TEAM["name"], bg=tint, font=("TkDefaultFont", 14, "bold"), anchor="w").pack(fill="x", pady=(0, 4))
        for sid, name in TEAM["members"]:
            row = tk.Frame(box, bg=tint)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, bg=tint, anchor="w").pack(side="left")
            tk.Label(row, text=sid, bg=tint, fg=MUTED, font="TkFixedFont").pack(side="right")
        tk.Frame(box, height=1, bg=brand_shade(0.25)).pack(fill="x", pady=(8, 6))
        line = tk.Frame(box, bg=tint)
        line.pack(fill="x")
        tk.Label(line, text="GVHD:", bg=tint, font=("TkDefaultFont", 11, "bold")).pack(side="left")
        tk.Label(line, text=TEAM["supervisor"], bg=tint).pack(side="left", padx=(4, 0))
