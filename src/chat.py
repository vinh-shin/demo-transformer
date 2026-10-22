"""Cửa sổ chính (Tkinter) mô phỏng khung bình luận, gắn nhãn bằng Transformer tự cài đặt.

Nạp checkpoint trực tiếp từ `models/` qua `src/inference.py`, không gọi API.
"""

import json
import queue
import threading
import time
import tkinter as tk
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from .detail import DetailWindow
from .sidebar import Sidebar
from .widgets import (BG, BORDER, BRAND, LABEL_COLOR, LABEL_VI, LABELS, MUTED, SURFACE,
                     SURFACE_HOVER, TEXT, Avatar, RoundButton, ScrollFrame, Toggle, bind_tree,
                     install_mousewheel, label_badge, pct, setup_theme)

HISTORY_FILE = Path.home() / ".vihsd-demo-chat.json"
MAX_CHARS = 2000
QUICK = [
    "Video này hay quá, cảm ơn bạn đã chia sẻ nhé",
    "Thằng ngu này nói chuyện chán vãi",
    "Bọn mày là lũ súc vật không hơn không kém",
    "Đúng là đồ óc chó, nói mãi không hiểu",
    "Mai đi cà phê không mọi người ơi",
]


class Worker(threading.Thread):
    """Nạp model rồi lần lượt phân tích các câu trong hàng đợi, để giao diện không bị đứng.

    Tk không cho cập nhật widget từ luồng khác, nên kết quả được đẩy vào `results`
    và luồng giao diện tự lấy ra định kỳ.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.ready = False
        self.error: str | None = None

    def submit(self, text: str, callback):
        self.jobs.put((text, callback))

    def run(self):
        try:
            from . import inference  # nạp torch/underthesea mất vài giây
            inference.check_artifacts()
            inference.get_model()
            self.ready = True
        except Exception as e:  # noqa: BLE001 - lỗi được hiện trên giao diện
            self.error = str(e)
        self.results.put(("status", None, None))

        while True:
            text, callback = self.jobs.get()
            if self.error:
                self.results.put((callback, None, self.error))
                continue
            try:
                self.results.put((callback, inference.analyze(text), None))
            except Exception as e:  # noqa: BLE001
                self.results.put((callback, None, str(e)))


def format_time(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M %d/%m")


def load_history() -> tuple[list[dict], bool]:
    try:
        saved = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], True
    comments = saved.get("comments", [])
    # Bình luận đang chờ lúc đóng app sẽ không bao giờ có kết quả, chuyển sang lỗi để thử lại.
    for c in comments:
        if c["status"] == "pending":
            c.update(status="error", error="Bị gián đoạn khi đóng app.")
    return comments, saved.get("hide_toxic", True)


def is_toxic(c: dict) -> bool:
    return c.get("prediction") in ("OFFENSIVE", "HATE")


class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        setup_theme(self)
        self.title("Phát hiện bình luận độc hại tiếng Việt")
        self.geometry("1280x780")
        self.minsize(1000, 560)
        self.configure(bg=BG)
        install_mousewheel(self)

        self.comments, hide = load_history()
        self.hide_toxic = tk.BooleanVar(value=hide)
        self.revealed: set[str] = set()
        self.analyses: dict[str, dict] = {}  # kết quả đầy đủ (kèm attention), chỉ giữ trong phiên
        self.rows: dict[str, tk.Frame] = {}
        self.text_labels: dict[str, tk.Label] = {}  # chữ cần ngắt dòng theo độ rộng cửa sổ
        self.confirm_clear = None  # id của lịch hẹn huỷ trạng thái "bấm lần nữa để xoá"

        self.worker = Worker()
        self.worker.start()

        Sidebar(self).pack(side="left", fill="y")
        tk.Frame(self, width=1, bg=BORDER).pack(side="left", fill="y")
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)

        self.build_header()
        self.build_footer()
        self.chat_list = ScrollFrame(self.content)
        self.chat_list.pack(fill="both", expand=True)
        self.chat_list.canvas.bind("<Configure>", self.on_resize, add="+")
        self.chat_list.inner.columnconfigure(0, weight=1)
        self.empty_hint = tk.Label(self.chat_list.inner, bg=BG, fg=MUTED, pady=120,
                                   text="Chưa có bình luận nào. Hãy nhập bình luận bên dưới hoặc chọn một câu thử nhanh.")

        self.render_all()
        self.after(50, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.input.focus_set()

    # ---------------------------------------------------------------- layout

    def build_header(self):
        top = tk.Frame(self.content, bg=BG, padx=24, pady=16)
        top.pack(fill="x")
        tk.Label(top, text="Phát hiện bình luận độc hại tiếng Việt", bg=BG, fg=TEXT,
                 font=("TkDefaultFont", 22, "bold"), anchor="w").pack(fill="x")
        tk.Label(top, bg=BG, fg=MUTED, anchor="w",
                 text="Ứng dụng kỹ thuật Transformer · Transformer Encoder tự cài đặt, "
                      "huấn luyện từ đầu trên dataset ViHSD").pack(fill="x", pady=(2, 0))
        self.status = tk.Label(top, text="Đang nạp model...", bg=BG, fg="#b45309", anchor="w")
        self.status.pack(fill="x", pady=(6, 0))

        bar = tk.Frame(self.content, bg=BG, padx=24, pady=10)
        bar.pack(fill="x")
        tk.Frame(self.content, height=1, bg=BORDER).pack(fill="x")
        tk.Label(bar, text="Mô phỏng khung bình luận", bg=BG, fg=TEXT, font=("TkDefaultFont", 15, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(bar, text="Mỗi bình luận được model gắn nhãn ngay khi gửi. Bấm vào bình luận để xem chi tiết.",
                 bg=BG, fg=MUTED, anchor="w").pack(fill="x")

        controls = tk.Frame(bar, bg=BG)
        controls.pack(fill="x", pady=(10, 0))
        pill = dict(font=("TkDefaultFont", 11, "bold"), padx=10, pady=2, radius=99)
        self.count_labels = {"total": RoundButton(controls, "", fill=SURFACE, fg=TEXT, **pill)}
        self.count_labels["total"].pack(side="left", padx=(0, 6))
        for label in LABELS:
            self.count_labels[label] = RoundButton(controls, "", fill=LABEL_COLOR[label], fg="white", **pill)
            self.count_labels[label].pack(side="left", padx=(0, 6))
        self.clear_btn = RoundButton(controls, "Xoá lịch sử", command=self.clear)
        self.clear_btn.pack(side="right")
        Toggle(controls, "Ẩn bình luận độc hại", self.hide_toxic, command=self.on_toggle_hide) \
            .pack(side="right", padx=16)

    def build_footer(self):
        bottom = tk.Frame(self.content, bg=BG, padx=24, pady=12)
        bottom.pack(side="bottom", fill="x")
        tk.Frame(self.content, height=1, bg=BORDER).pack(side="bottom", fill="x")

        quick = tk.Frame(bottom, bg=BG)
        quick.pack(fill="x")
        tk.Label(quick, text="Thử nhanh:", bg=BG, fg=MUTED).grid(row=0, column=0, sticky="w", padx=(0, 4))
        for i, q in enumerate(QUICK):
            RoundButton(quick, q, command=lambda q=q: self.send(q), kind="chip", radius=99, padx=10, pady=3) \
                .grid(row=i // 3, column=1 + i % 3, sticky="w", padx=3, pady=3)

        row = tk.Frame(bottom, bg=BG)
        row.pack(fill="x", pady=(8, 0))
        self.input = tk.Text(row, height=3, wrap="word", font="TkDefaultFont", padx=10, pady=8,
                             bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
                             highlightthickness=1, highlightbackground=BORDER, highlightcolor=BRAND)
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self.on_enter)
        self.input.bind("<Shift-Return>", lambda _: None)  # để Tk xuống dòng như mặc định
        self.input.bind("<<Modified>>", self.on_input_change)
        side = tk.Frame(row, bg=BG)
        side.pack(side="left", padx=(10, 0))
        self.send_btn = RoundButton(side, "Gửi", command=self.send, kind="primary", padx=26, pady=8,
                                    font=("TkDefaultFont", 13, "bold"))
        self.send_btn.set_enabled(False)
        self.send_btn.pack()
        self.char_count = tk.Label(side, text=f"0/{MAX_CHARS}", bg=BG, fg=MUTED)
        self.char_count.pack(pady=(4, 0))
        tk.Label(bottom, text="Enter để gửi, Shift + Enter để xuống dòng.", bg=BG, fg=MUTED,
                 anchor="w").pack(fill="x", pady=(4, 0))

    # ---------------------------------------------------------------- comments

    def draft(self) -> str:
        return self.input.get("1.0", "end-1c")

    def on_input_change(self, _=None):
        self.input.edit_modified(False)
        n = len(self.draft())
        self.char_count.configure(text=f"{n}/{MAX_CHARS}", fg="#dc2626" if n > MAX_CHARS else MUTED)
        self.send_btn.set_enabled(bool(self.draft().strip()) and n <= MAX_CHARS)

    def on_enter(self, _):
        self.send()
        return "break"

    def send(self, text: str | None = None):
        from_input = text is None
        text = (self.draft() if from_input else text).strip()
        if not text or len(text) > MAX_CHARS:
            return
        c = {"id": uuid.uuid4().hex, "text": text, "time": time.time(), "status": "pending"}
        self.comments.append(c)
        if from_input:
            self.input.delete("1.0", "end")
        self.render_comment(c)
        self.update_counts()
        self.chat_list.scroll_to_bottom()
        self.classify(c)  # không chờ: người dùng có thể gửi tiếp ngay

    def classify(self, c: dict):
        c.update(status="pending", error=None)
        self.render_comment(c)
        self.worker.submit(c["text"], lambda result, error, cid=c["id"]: self.on_result(cid, result, error))

    def on_result(self, cid: str, result, error):
        c = next((c for c in self.comments if c["id"] == cid), None)
        if c is None:  # đã xoá lịch sử trong lúc chờ
            return
        if error:
            c.update(status="error", error=error)
        elif result is None:
            c.update(status="empty")
        else:
            probs = result["probs"]
            c.update(status="done", prediction=LABELS[int(np.argmax(probs))], probs=[round(float(p), 4) for p in probs])
            self.analyses[cid] = result
        self.render_comment(c)
        self.update_counts()
        self.save()

    def poll(self):
        while True:
            try:
                callback, result, error = self.worker.results.get_nowait()
            except queue.Empty:
                break
            if callback == "status":
                self.update_status()
            else:
                callback(result, error)
        self.after(50, self.poll)

    def update_status(self):
        if self.worker.error:
            self.status.configure(text=f"Lỗi: {self.worker.error}", fg="#dc2626", wraplength=700, justify="left")
        else:
            self.status.configure(text="Model đã sẵn sàng.", fg="#16a34a")

    # ---------------------------------------------------------------- render

    def render_all(self):
        for w in self.chat_list.inner.winfo_children():
            if w is not self.empty_hint:
                w.destroy()
        self.rows.clear()
        self.text_labels.clear()
        for c in self.comments:
            self.render_comment(c)
        self.update_counts()

    def render_comment(self, c: dict):
        old = self.rows.get(c["id"])
        idx = self.comments.index(c)
        if old is not None:
            old.destroy()

        row = tk.Frame(self.chat_list.inner, bg=BG, padx=24, pady=8)
        row.grid(row=idx, column=0, sticky="ew")
        self.rows[c["id"]] = row

        Avatar(row).pack(side="left", anchor="n", padx=(0, 12))
        col = tk.Frame(row, bg=BG)
        col.pack(side="left", fill="x", expand=True)

        bubble = tk.Frame(col, bg=SURFACE)
        bubble.pack(anchor="w")
        tk.Frame(bubble, width=4, bg=LABEL_COLOR.get(c.get("prediction"), SURFACE)).pack(side="left", fill="y")
        inner = tk.Frame(bubble, bg=SURFACE, padx=14, pady=8)
        inner.pack(side="left")
        tk.Label(inner, text="Bạn", bg=SURFACE, fg=TEXT, font=("TkDefaultFont", 11, "bold"), anchor="w").pack(fill="x")

        hidden = self.hide_toxic.get() and is_toxic(c) and c["id"] not in self.revealed
        wrap = self.wrap_width()
        reveal = None
        if hidden:
            # Tk không làm mờ được chữ, nên thay nội dung bằng thông báo.
            line = tk.Frame(inner, bg=SURFACE)
            line.pack(anchor="w")
            msg = tk.Label(line, bg=SURFACE, fg="#475569", justify="left", wraplength=wrap,
                           text=f"⚠ Đã ẩn vì nội dung {LABEL_VI[c['prediction']].lower()}.")
            msg.pack(side="left")
            self.text_labels[c["id"]] = msg
            reveal = tk.Label(line, text="Xem", bg=SURFACE, fg=BRAND, cursor="hand2",
                              font=("TkDefaultFont", 11, "bold underline"))
            reveal.pack(side="left", padx=6)
        else:
            body = tk.Label(inner, text=c["text"], bg=SURFACE, fg=TEXT, justify="left", anchor="w", wraplength=wrap)
            body.pack(fill="x")
            self.text_labels[c["id"]] = body

        if c["status"] in ("done", "empty"):
            # Đổi nền cả bong bóng khi rê chuột để báo có thể bấm xem chi tiết.
            parts = [inner, *inner.winfo_children()] + ([line, msg, reveal] if hidden else [])
            bind_tree(inner, "<Enter>", lambda _: [w.configure(bg=SURFACE_HOVER) for w in parts])
            bind_tree(inner, "<Leave>", lambda _: [w.configure(bg=SURFACE) for w in parts])
            for w in parts:
                w.configure(cursor="hand2")
            bind_tree(inner, "<Button-1>", lambda _: self.open_detail(c))
        if reveal is not None:  # nút "Xem" chỉ hiện lại nội dung, không mở cửa sổ chi tiết
            reveal.unbind("<Button-1>")
            reveal.bind("<Button-1>", lambda _: self.reveal(c))

        meta = tk.Frame(col, bg=BG)
        meta.pack(anchor="w", pady=(4, 0))
        tk.Label(meta, text=format_time(c["time"]), bg=BG, fg=MUTED).pack(side="left")
        if c["status"] == "pending":
            tk.Label(meta, text="Đang phân tích...", bg=BG, fg=MUTED).pack(side="left", padx=8)
        elif c["status"] == "done":
            label_badge(meta, c["prediction"]).pack(side="left", padx=8)
            tk.Label(meta, text=pct(max(c["probs"]), 0), bg=BG, fg=TEXT).pack(side="left")
        elif c["status"] == "empty":
            RoundButton(meta, "TRỐNG", fill="#94a3b8", fg="white", font=("TkDefaultFont", 9, "bold"),
                        padx=8, pady=1, radius=99).pack(side="left", padx=8)
        else:
            tk.Label(meta, text=f"Lỗi: {c.get('error')}", bg=BG, fg="#dc2626", wraplength=wrap,
                     justify="left").pack(side="left", padx=8)
            retry = tk.Label(meta, text="Thử lại", bg=BG, fg="#dc2626", cursor="hand2",
                             font=("TkDefaultFont", 11, "bold underline"))
            retry.pack(side="left")
            retry.bind("<Button-1>", lambda _: self.classify(c))

    def reveal(self, c: dict):
        self.revealed.add(c["id"])
        self.render_comment(c)
        return "break"

    def on_toggle_hide(self):
        self.render_all()
        self.save()

    def update_counts(self):
        self.count_labels["total"].set_text(f"Tổng {len(self.comments)}")
        for label in LABELS:
            n = sum(c.get("prediction") == label for c in self.comments)
            self.count_labels[label].set_text(f"{label} {n}")
        self.clear_btn.set_enabled(bool(self.comments))
        if self.comments:
            self.empty_hint.grid_forget()
        else:
            self.empty_hint.grid(row=0, column=0, sticky="ew")

    def wrap_width(self) -> int:
        return max(self.chat_list.canvas.winfo_width() - 160, 300)

    def on_resize(self, _):
        wrap = self.wrap_width()
        for w in self.text_labels.values():
            if w.winfo_exists():
                w.configure(wraplength=wrap)

    # ---------------------------------------------------------------- actions

    def open_detail(self, c: dict):
        win = DetailWindow(self, c["text"], format_time(c["time"]))
        if c["id"] in self.analyses:
            win.show(self.analyses[c["id"]])
        else:  # bình luận từ phiên trước: phân tích lại để lấy attention
            def done(result, error, cid=c["id"]):
                if result is not None:
                    self.analyses[cid] = result
                win.show(result, error)
            self.worker.submit(c["text"], done)

    def clear(self):
        # Giống bản web: bấm lần đầu để xác nhận, bấm lần nữa trong 3 giây mới xoá.
        if self.confirm_clear is None:
            self.clear_btn.set_text("Bấm lần nữa để xoá")
            self.confirm_clear = self.after(3000, self.reset_clear)
            return
        self.reset_clear()
        self.comments.clear()
        self.revealed.clear()
        self.analyses.clear()
        self.render_all()
        self.save()

    def reset_clear(self):
        if self.confirm_clear is not None:
            self.after_cancel(self.confirm_clear)
            self.confirm_clear = None
        self.clear_btn.set_text("Xoá lịch sử")

    def save(self):
        data = {"comments": self.comments, "hide_toxic": self.hide_toxic.get()}
        try:
            HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # không lưu được lịch sử thì app vẫn chạy bình thường

    def on_close(self):
        self.save()
        self.destroy()

