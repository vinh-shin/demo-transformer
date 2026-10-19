"""Hằng số, định dạng số và các widget dùng chung cho giao diện."""

import sys
import tkinter as tk
import tkinter.font as tkfont

LABELS = ["CLEAN", "OFFENSIVE", "HATE"]
LABEL_VI = {"CLEAN": "Bình thường", "OFFENSIVE": "Xúc phạm", "HATE": "Thù ghét"}
LABEL_COLOR = {"CLEAN": "#16a34a", "OFFENSIVE": "#f59e0b", "HATE": "#dc2626"}
BRAND_RGB = (21, 101, 192)
BRAND = "#1565c0"
BRAND_DARK = "#0d47a1"
TEXT = "#0f172a"
MUTED = "#64748b"
BG = "#ffffff"
SURFACE = "#f1f5f9"
SURFACE_HOVER = "#e2e8f0"
BORDER = "#e2e8f0"
DISABLED = "#cbd5e1"

EMPTY_MSG = "Câu trống sau khi làm sạch (chỉ có URL, mention hoặc ký tự đặc biệt)."


# Định dạng số theo chuẩn Việt Nam: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn.
def dec(v: float, digits: int) -> str:
    return f"{v:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(v: float, digits: int = 1) -> str:
    return f"{dec(v * 100, digits)}%"


def display(token: str) -> str:
    return token.replace("_", " ")


def brand_shade(alpha: float) -> str:
    """Màu brand với độ đậm `alpha` trên nền trắng (Tk không hỗ trợ màu trong suốt)."""
    alpha = max(0.0, min(1.0, alpha))
    r, g, b = (round(255 + (c - 255) * alpha) for c in BRAND_RGB)
    return f"#{r:02x}{g:02x}{b:02x}"


def setup_theme(window: tk.Misc):
    """Luôn dùng giao diện sáng, kể cả khi hệ điều hành bật Dark Mode.

    Trên macOS ở Dark Mode, Tk tự lấy màu hệ thống (chữ trắng, ô nhập nền đen), nên chữ
    không đặt màu riêng sẽ biến mất trên nền trắng. Gọi cho cửa sổ chính và mỗi Toplevel.
    """
    if sys.platform == "darwin":
        try:
            window.update_idletasks()  # Tk chỉ đổi được appearance khi cửa sổ đã được tạo
            window.tk.call("::tk::unsupported::MacWindowStyle", "appearance", window._w, "aqua")
        except tk.TclError:  # Tk cũ hơn 8.6.10 không hỗ trợ
            pass
    if isinstance(window, tk.Tk):
        for pattern, value in (
            ("*foreground", TEXT),
            ("*background", BG),
            ("*highlightBackground", BG),
            ("*activeForeground", TEXT),
            ("*disabledForeground", MUTED),
            ("*Text.background", BG),
            ("*Text.insertBackground", TEXT),
            ("*selectBackground", brand_shade(0.25)),
            ("*selectForeground", TEXT),
            ("*TCombobox*Listbox.background", BG),
            ("*TCombobox*Listbox.foreground", TEXT),
        ):
            window.option_add(pattern, value)


def bind_tree(widget, sequence, func):
    """Gắn sự kiện cho widget và mọi widget con (Tk không tự lan sự kiện lên cha)."""
    widget.bind(sequence, func, add="+")
    for child in widget.winfo_children():
        bind_tree(child, sequence, func)


class ScrollFrame(tk.Frame):
    """Frame cuộn dọc; nội dung đặt vào `self.inner`."""

    def __init__(self, master, bg=BG, **kw):
        super().__init__(master, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))

    def scroll(self, step: int):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(step, "units")

    def scroll_to_bottom(self):
        self.update_idletasks()
        self.canvas.yview_moveto(1.0)


def install_mousewheel(root: tk.Misc):
    """Cuộn ScrollFrame nằm dưới con trỏ chuột, kể cả khi con trỏ đang ở trên widget con."""

    def target(event):
        try:
            w = root.winfo_containing(event.x_root, event.y_root)
        except KeyError:  # popup của ttk.Combobox không phải widget Python
            return None
        while w is not None and not isinstance(w, ScrollFrame):
            w = w.master
        return w

    def on_wheel(event):
        if (w := target(event)) is not None:
            # macOS trả delta nhỏ theo pixel, Windows trả bội số của 120.
            w.scroll(-event.delta if sys.platform == "darwin" else -event.delta // 120)

    root.bind_all("<MouseWheel>", on_wheel)
    for seq, step in (("<Button-4>", -1), ("<Button-5>", 1)):  # Linux
        root.bind_all(seq, lambda e, s=step: (w := target(e)) is not None and w.scroll(s))


def round_rect(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Hình chữ nhật bo góc (Tk không có sẵn): đa giác làm mượt với điểm lặp ở mỗi góc."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


class RoundButton(tk.Canvas):
    """Nút phẳng bo góc kiểu web. Không có `command` thì dùng như nhãn (pill)."""

    KINDS = {  # nền, nền khi rê chuột, chữ, viền
        "primary": (BRAND, BRAND_DARK, "white", None),
        "ghost": (BG, SURFACE, TEXT, BORDER),
        "chip": (SURFACE, SURFACE_HOVER, TEXT, BORDER),
    }

    def __init__(self, master, text, command=None, kind="ghost", fill=None, fg=None, font=None,
                 padx=12, pady=5, radius=8):
        super().__init__(master, bg=master.cget("bg"), highlightthickness=0, bd=0)
        base, hover, color, border = self.KINDS[kind]
        self.colors = (fill or base, hover if command else (fill or base), fg or color, border if not fill else None)
        self.font = tkfont.Font(font=font) if font else tkfont.nametofont("TkDefaultFont")
        self.padx, self.pady, self.radius = padx, pady, radius
        self.command, self.enabled, self.hovering = command, True, False
        self.text = text
        if command:
            self.configure(cursor="hand2")
            self.bind("<Enter>", lambda _: self._hover(True))
            self.bind("<Leave>", lambda _: self._hover(False))
            self.bind("<ButtonRelease-1>", self._click)
        self.draw()

    def draw(self):
        w = self.font.measure(self.text) + 2 * self.padx
        h = self.font.metrics("linespace") + 2 * self.pady
        self.configure(width=w, height=h)
        self.delete("all")
        base, hover, color, border = self.colors
        fill = (hover if self.hovering else base) if self.enabled else DISABLED
        round_rect(self, 1, 1, w - 1, h - 1, self.radius, fill=fill, outline=border or fill)
        self.create_text(w / 2, h / 2, text=self.text, font=self.font, fill=color if self.enabled else "white")

    def set_text(self, text):
        self.text = text
        self.draw()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled and self.command else "")
        self.draw()

    def _hover(self, on):
        self.hovering = on
        self.draw()

    def _click(self, event):
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        if self.enabled and inside:
            self.command()


def label_badge(master, label: str, size: int = 9) -> RoundButton:
    return RoundButton(master, label, fill=LABEL_COLOR[label], fg="white",
                       font=("TkDefaultFont", size, "bold"), padx=8, pady=1, radius=99)


class Toggle(tk.Frame):
    """Công tắc bật/tắt kiểu web, gắn với một BooleanVar."""

    W, H = 36, 20

    def __init__(self, master, text, variable: tk.BooleanVar, command=None):
        super().__init__(master, bg=master.cget("bg"), cursor="hand2")
        self.var, self.command = variable, command
        self.switch = tk.Canvas(self, width=self.W, height=self.H, bg=self["bg"], highlightthickness=0)
        self.switch.pack(side="left")
        tk.Label(self, text=text, bg=self["bg"]).pack(side="left", padx=(6, 0))
        bind_tree(self, "<Button-1>", self._toggle)
        self.draw()

    def draw(self):
        c, on = self.switch, self.var.get()
        c.delete("all")
        round_rect(c, 0, 0, self.W, self.H, self.H / 2, fill=BRAND if on else DISABLED, outline="")
        x = self.W - self.H + 2 if on else 2
        c.create_oval(x, 2, x + self.H - 4, self.H - 2, fill="white", outline="")

    def _toggle(self, _):
        self.var.set(not self.var.get())
        self.draw()
        if self.command:
            self.command()


class Avatar(tk.Canvas):
    def __init__(self, master, letter="B", size=36):
        super().__init__(master, width=size, height=size, bg=master.cget("bg"), highlightthickness=0)
        self.create_oval(1, 1, size - 1, size - 1, fill=BRAND, outline="")
        self.create_text(size / 2, size / 2, text=letter, fill="white", font=("TkDefaultFont", 13, "bold"))


class ProbBars(tk.Frame):
    """Thanh xác suất ngang cho 3 lớp."""

    BAR_W, BAR_H = 320, 12

    def __init__(self, master, probs, **kw):
        super().__init__(master, bg=kw.pop("bg", BG), **kw)
        for i, label in enumerate(LABELS):
            tk.Label(self, text=f"{label} · {LABEL_VI[label]}", bg=self["bg"], anchor="w", width=22) \
                .grid(row=i, column=0, sticky="w", pady=3)
            c = tk.Canvas(self, width=self.BAR_W, height=self.BAR_H, bg=self["bg"], highlightthickness=0)
            c.create_rectangle(0, 0, self.BAR_W, self.BAR_H, fill=SURFACE, outline="")
            c.create_rectangle(0, 0, self.BAR_W * probs[i], self.BAR_H, fill=LABEL_COLOR[label], outline="")
            c.grid(row=i, column=1, padx=8)
            tk.Label(self, text=pct(probs[i]), bg=self["bg"], width=7, anchor="e").grid(row=i, column=2)
