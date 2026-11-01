# -*- coding: utf-8 -*-
# =====================================================================
# Tarih          : 27.10.2026
# Ad / Soyad     : Yasin Alioğlu
# Öğrenci No     : 2020250062
# Ders           : ELHT2237 SİNYALLER VE SİSTEMLER
# Kodun Amacı    : Sürekli ve ayrık zamanlı temel sinyalleri üretmek,
#                  zaman kaydırma, zaman ölçekleme, zamanda tersine
#                  çevirme ve genlik ölçekleme dönüşümlerini uygulamak.
# Gereksinimler  : Python 3, numpy, matplotlib
# Kurulum        : pip install numpy matplotlib
#
# DÖNÜŞÜM MODELİ
#   y(t) = K · x( c · (t + t0) ),   c = ±a   (tersine çevirmede c = -a)
#
#   t0 > 0 : sinyal sola kayar        t0 < 0 : sinyal sağa kayar
#   |a| > 1: zaman sıkışır            |a| < 1: zaman genişler
#   K      : genlik ölçekleme
#
#   Sürekli : y(t) her t için x_func ile doğrudan hesaplanır.
#             Pencere [tmin, tmax] -> [tmin/c - t0, tmax/c - t0]'a taşınır.
#   Ayrık   : y[n] = K·x[c·(n+n0)]. Argüman tam sayı değilse örnek
#             tanımsızdır ve 0 alınır (|a|<1 -> araya sıfır ekleme,
#             |a|>1 -> seyreltme / örnek atma).
#   Delta   : δ(c·u) = δ(u)/|c| olduğundan sürekli darbenin şiddeti
#             K·A/|c| olur.
# =====================================================================

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

# ---------------------------------------------------------------------
# Öğrenci bilgileri
# ---------------------------------------------------------------------

AD_SOYAD = "Yasin Alioğlu"
OGRENCI_NO = "2020250062"
DERS_ADI = "ELHT2237 SİNYALLER VE SİSTEMLER"

SIGNAL_TYPES = [
    "Sinüs",
    "Kare",
    "Üçgen",
    "Birim Basamak",
    "Birim Darbe (Delta)",
    "Üstel",
    "Karmaşık Üstel",
]

PERIODIC = ("Sinüs", "Kare", "Üçgen", "Karmaşık Üstel")

MAX_SAMPLES = 200000
MAX_DISCRETE = 2000

# numpy 2.x'te trapz kaldırıldı, trapezoid geldi; ikisini de destekle
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class TransformError(ValueError):
    """Dönüşüm sırasında oluşan, kullanıcıya gösterilecek hata."""


# =====================================================================
# SİNYAL FONKSİYONLARI
# =====================================================================

def make_signal(kind, A, f, phi_deg, a, Ts):
    """Seçilen sinyal tipinin matematiksel fonksiyonunu oluşturur."""

    phi = np.deg2rad(phi_deg)
    eps = Ts * 1e-6

    if kind == "Sinüs":
        return lambda u: A * np.sin(2 * np.pi * f * u + phi)

    if kind == "Kare":
        return lambda u: A * np.where(
            np.sin(2 * np.pi * f * u + phi) >= 0,
            1.0,
            -1.0
        )

    if kind == "Üçgen":
        return lambda u: A * (2 / np.pi) * np.arcsin(
            np.sin(2 * np.pi * f * u + phi)
        )

    if kind == "Birim Basamak":
        return lambda u: A * np.where(u >= -eps, 1.0, 0.0)

    if kind == "Birim Darbe (Delta)":
        return lambda u: A * np.where(np.abs(u) < eps, 1.0, 0.0)

    if kind == "Üstel":
        return lambda u: A * np.exp(a * u)

    if kind == "Karmaşık Üstel":
        return lambda u: A * np.exp(
            1j * (2 * np.pi * f * u + phi)
        )

    raise ValueError("Bilinmeyen sinyal tipi.")


def delta_samples(t, pos, amp, Ts):
    """Belirtilen konuma en yakın örneğe darbe (şiddet = amp) yerleştirir."""

    y = np.zeros(len(t))

    if len(t) == 0:
        return y

    i = int(np.argmin(np.abs(t - pos)))

    if abs(t[i] - pos) <= Ts / 2 + Ts * 1e-9:
        y[i] = amp

    return y


# =====================================================================
# DÖNÜŞÜM ÇEKİRDEĞİ (arayüzden bağımsız, tek başına test edilebilir)
# =====================================================================

def transform_signal(x_func, info, K, t0, a_scale, reverse):
    """
    y(t) = K · x(c·(t + t0)),  c = (-1 if reverse else 1) · a_scale

    Sözlük döndürür:
        t, y        : dönüşmüş sinyalin zaman ekseni ve değerleri
        lo, hi      : dönüşmüş sinyalin zaman penceresi
        c, t0_used  : etkin ölçek katsayısı ve kullanılan kaydırma
        strength    : (yalnızca sürekli delta) darbe şiddeti
        notes       : bilgi notları
        warnings    : uyarılar
    """

    kind = info["kind"]
    discrete = info["discrete"]
    fs = info["fs"]
    Ts = info["Ts"]
    A = info["A"]
    tmin, tmax = info["tmin"], info["tmax"]

    if a_scale == 0:
        raise TransformError("Zaman ölçekleme katsayısı sıfır olamaz.")

    c = (-1.0 if reverse else 1.0) * a_scale
    notes, warnings = [], []
    strength = None

    # -------------------------------------------------------------
    # AYRIK ZAMAN:  y[n] = K · x[c·(n + n0)]
    # -------------------------------------------------------------
    if discrete:

        n0 = int(round(t0 * fs))
        t0_used = n0 / fs

        if abs(t0_used - t0) > 1e-12:
            notes.append(
                f"Ayrık sinyalde kaydırma Ts'nin katı olmalı: "
                f"t0 = {t0:g} s → {t0_used:g} s ({n0:+d} örnek)."
            )

        kmin = int(np.ceil(tmin * fs - 1e-9))
        kmax = int(np.floor(tmax * fs + 1e-9))

        # k = c(n+n0) ∈ [kmin, kmax]  ->  n aralığı
        edges = (kmin / c - n0, kmax / c - n0)
        n_lo = int(np.ceil(min(edges) - 1e-9))
        n_hi = int(np.floor(max(edges) + 1e-9))

        if n_hi < n_lo:
            raise TransformError("Dönüşüm sonrası örnek kalmadı.")

        if n_hi - n_lo + 1 > MAX_DISCRETE:
            raise TransformError(
                "Dönüşüm sonrası ayrık örnek sayısı çok fazla. "
                "Ölçekleme katsayısını büyütün."
            )

        n = np.arange(n_lo, n_hi + 1)
        k = c * (n + n0)
        k_int = np.rint(k)

        # Yalnızca tam sayı indisli ve orijinal pencere içindeki örnekler var
        valid = (
            (np.abs(k - k_int) < 1e-9)
            & (k_int >= kmin)
            & (k_int <= kmax)
        )

        dtype = complex if kind == "Karmaşık Üstel" else float
        y = np.zeros(len(n), dtype=dtype)

        if valid.any():
            y[valid] = K * x_func(k_int[valid] * Ts)

        t_y = n * Ts
        lo, hi = float(t_y[0]), float(t_y[-1])

        ac = abs(c)
        if ac > 1 + 1e-12:
            notes.append(
                f"|a|={ac:g} > 1: seyreltme (decimation) – bazı örnekler atıldı."
            )
        elif ac < 1 - 1e-12:
            notes.append(
                f"|a|={ac:g} < 1: genişletme – tanımsız indislere 0 yazıldı "
                f"(araya sıfır ekleme)."
            )

        frac = ac - round(ac)
        inv = 1 / ac
        if abs(frac) > 1e-9 and abs(inv - round(inv)) > 1e-9:
            notes.append(
                "a tamsayı veya 1/tamsayı değil: x[a·n] yalnızca tamsayı "
                "indislerde tanımlıdır, diğer örnekler 0 alındı."
            )

    # -------------------------------------------------------------
    # SÜREKLİ ZAMAN:  y(t) = K · x(c·(t + t0))
    # -------------------------------------------------------------
    else:

        t0_used = t0

        edges = (tmin / c - t0, tmax / c - t0)
        lo, hi = min(edges), max(edges)

        n_y = int(round((hi - lo) * fs)) + 1

        if n_y > MAX_SAMPLES:
            raise TransformError(
                "Dönüşüm sonrası örnek sayısı çok fazla. "
                "Örnekleme frekansını düşürün veya a değerini artırın."
            )

        n_y = max(n_y, 500)   # çok sıkışan sinyallerde de düzgün çizim
        t_y = np.linspace(lo, hi, n_y)
        dt_y = t_y[1] - t_y[0]

        if kind == "Birim Darbe (Delta)":
            # δ(c(t+t0)) = δ(t+t0)/|c|  -> konum: t = -t0, şiddet: K·A/|c|
            strength = K * A / abs(c)
            y = delta_samples(t_y, -t0, strength, dt_y)
            notes.append(
                f"δ(a·t) = δ(t)/|a| gereği darbe şiddeti K·A/|a| = "
                f"{strength:.4f} oldu (çizimde ok yüksekliği = şiddet)."
            )
        else:
            y = K * x_func(c * (t_y + t0))

        if kind in PERIODIC and info["f"] * abs(c) >= fs / 2:
            warnings.append(
                f"Dönüşmüş sinyalin frekansı f·|a| = {info['f'] * abs(c):g} Hz, "
                f"fs/2 = {fs / 2:g} Hz sınırına ulaştı. "
                "Çizim hatalı görünebilir; fs değerini artırın."
            )

    return {
        "t": t_y, "y": y, "lo": lo, "hi": hi,
        "c": c, "t0_used": t0_used, "strength": strength,
        "notes": notes, "warnings": warnings,
    }


# =====================================================================
# ARAYÜZ
# =====================================================================

class SignalApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Sinyal Üretimi ve Dönüşümleri Simülatörü")
        self.geometry("1280x760")
        self.minsize(1100, 680)

        self.t = None
        self.x = None
        self.x_func = None
        self.x_title = ""
        self.info = {}

        self.show_intro()

    # -----------------------------------------------------------------
    # Giriş ekranı
    # -----------------------------------------------------------------

    def show_intro(self):

        self.intro = ttk.Frame(self, padding=40)
        self.intro.pack(fill="both", expand=True)

        ttk.Label(
            self.intro,
            text="TEKNİK BİLİMLER MESLEK YÜKSEKOKULU",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(30, 4))

        ttk.Label(
            self.intro,
            text="ELEKTRONİK HABERLEŞME TEKNOLOJİSİ PROGRAMI",
            font=("Segoe UI", 11)
        ).pack(pady=4)

        ttk.Separator(self.intro).pack(fill="x", pady=25)

        ttk.Label(
            self.intro,
            text=DERS_ADI,
            font=("Segoe UI", 20, "bold")
        ).pack(pady=8)

        ttk.Label(
            self.intro,
            text="Sinyal Üretimi ve Dönüşümleri Simülatörü",
            font=("Segoe UI", 13)
        ).pack(pady=4)

        box = ttk.Frame(self.intro, padding=20, relief="groove")
        box.pack(pady=30)

        ttk.Label(
            box,
            text=f"Ad / Soyad      :  {AD_SOYAD}",
            font=("Segoe UI", 12)
        ).pack(anchor="w", pady=3)

        ttk.Label(
            box,
            text=f"Öğrenci No     :  {OGRENCI_NO}",
            font=("Segoe UI", 12)
        ).pack(anchor="w", pady=3)

        ttk.Label(
            box,
            text=f"Ders Adı         :  {DERS_ADI}",
            font=("Segoe UI", 12)
        ).pack(anchor="w", pady=3)

        ttk.Button(
            self.intro,
            text="Simülatörü Başlat",
            command=self.start_main
        ).pack(ipadx=20, ipady=8, pady=10)

    def start_main(self):

        self.intro.destroy()
        self.build_main()

    # -----------------------------------------------------------------
    # Ana ekran
    # -----------------------------------------------------------------

    def build_main(self):

        top = ttk.Frame(self, padding=(10, 4))
        top.pack(fill="x")

        ttk.Label(
            top,
            text=f"{AD_SOYAD} | {OGRENCI_NO} | {DERS_ADI}",
            font=("Segoe UI", 9, "italic")
        ).pack(side="left")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=8)
        left.pack(side="left", fill="y")

        # Sinyal üretimi

        gen = ttk.LabelFrame(
            left,
            text="A. Sinyal Üretimi",
            padding=8
        )
        gen.pack(fill="x", pady=4)

        self.var_kind = tk.StringVar(value=SIGNAL_TYPES[0])
        self.var_domain = tk.StringVar(value="Sürekli")
        self.entries = {}

        r = 0

        ttk.Label(
            gen,
            text="Sinyal tipi:"
        ).grid(row=r, column=0, sticky="w", pady=2)

        cb = ttk.Combobox(
            gen,
            textvariable=self.var_kind,
            values=SIGNAL_TYPES,
            state="readonly",
            width=20
        )
        cb.grid(row=r, column=1, pady=2)
        cb.bind("<<ComboboxSelected>>", self.on_type_change)

        r += 1

        ttk.Label(
            gen,
            text="Zaman türü:"
        ).grid(row=r, column=0, sticky="w", pady=2)

        fr = ttk.Frame(gen)
        fr.grid(row=r, column=1, sticky="w")

        for txt in ("Sürekli", "Ayrık"):
            ttk.Radiobutton(
                fr,
                text=txt,
                value=txt,
                variable=self.var_domain,
                command=self.on_domain_change
            ).pack(side="left", padx=2)

        params = [
            ("A", "Genlik (A):", "1"),
            ("f", "Frekans f (Hz):", "2"),
            ("phi", "Faz φ (derece):", "0"),
            ("a", "Üstel katsayı (a):", "-1"),
            ("tmin", "Başlangıç zamanı (s):", "-2"),
            ("tmax", "Bitiş zamanı (s):", "2"),
            ("fs", "Örnekleme frek. fs (Hz):", "1000"),
        ]

        for key, label, default in params:

            r += 1

            ttk.Label(
                gen,
                text=label
            ).grid(row=r, column=0, sticky="w", pady=2)

            e = ttk.Entry(gen, width=22)
            e.insert(0, default)
            e.grid(row=r, column=1, pady=2)

            self.entries[key] = e

        r += 1

        ttk.Button(
            gen,
            text="Sinyali Üret ve Çiz",
            command=self.generate
        ).grid(
            row=r,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 2)
        )

        # Sinyal dönüşümleri

        tr = ttk.LabelFrame(
            left,
            text="B. Sinyal Dönüşümleri",
            padding=8
        )
        tr.pack(fill="x", pady=4)

        tparams = [
            ("t0", "Zaman kaydırma t0 (s):", "1"),
            ("ts", "Zaman ölçekleme (a):", "1"),
            ("K", "Genlik ölçekleme (K):", "1"),
        ]

        rr = 0

        for key, label, default in tparams:

            ttk.Label(
                tr,
                text=label
            ).grid(row=rr, column=0, sticky="w", pady=2)

            e = ttk.Entry(tr, width=22)
            e.insert(0, default)
            e.grid(row=rr, column=1, pady=2)

            self.entries[key] = e
            rr += 1

        self.var_rev = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            tr,
            text="Zamanda tersine çevir",
            variable=self.var_rev
        ).grid(
            row=rr,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2
        )

        rr += 1

        # Eklenti: orijinal sinyali karşılaştırma için gösterme
        self.var_overlay = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            tr,
            text="Orijinali karşılaştırma için göster",
            variable=self.var_overlay
        ).grid(
            row=rr,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2
        )

        rr += 1

        ttk.Label(
            tr,
            text="y(t) = K · x(±a·(t + t0))",
            font=("Consolas", 9)
        ).grid(
            row=rr,
            column=0,
            columnspan=2,
            pady=2
        )

        rr += 1

        ttk.Label(
            tr,
            text="t0>0: sola | t0<0: sağa | a>1: sıkıştır | a<1: genişlet",
            font=("Segoe UI", 8),
            foreground="gray40"
        ).grid(
            row=rr,
            column=0,
            columnspan=2,
            pady=(0, 2)
        )

        rr += 1

        ttk.Button(
            tr,
            text="Dönüşümü Uygula",
            command=self.transform
        ).grid(
            row=rr,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 2)
        )

        ttk.Button(
            left,
            text="Temizle / Sıfırla",
            command=self.reset
        ).pack(fill="x", pady=4)

        # Grafik alanı

        right = ttk.Frame(body, padding=4)
        right.pack(side="left", fill="both", expand=True)

        self.fig = Figure(
            figsize=(7, 5),
            dpi=100,
            constrained_layout=True
        )

        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.draw_empty(self.ax1, "Orijinal Sinyal x(t)")
        self.draw_empty(self.ax2, "Dönüştürülmüş Sinyal y(t)")

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=right
        )

        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        out = ttk.LabelFrame(
            right,
            text="Sayısal Çıktı",
            padding=4
        )
        out.pack(fill="x")

        self.txt = tk.Text(
            out,
            height=11,
            font=("Consolas", 9),
            wrap="none"
        )

        sb = ttk.Scrollbar(
            out,
            command=self.txt.yview
        )

        self.txt.configure(yscrollcommand=sb.set)

        self.txt.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

        self.on_type_change()

    # -----------------------------------------------------------------
    # Yardımcı fonksiyonlar
    # -----------------------------------------------------------------

    def draw_empty(self, ax, title):

        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("Zaman t (s)")
        ax.set_ylabel("Genlik")
        ax.grid(True, alpha=0.4)

    def on_type_change(self, _event=None):

        kind = self.var_kind.get()
        needs_f = kind in PERIODIC
        needs_a = kind == "Üstel"

        self.entries["f"].configure(
            state="normal" if needs_f else "disabled"
        )

        self.entries["phi"].configure(
            state="normal" if needs_f else "disabled"
        )

        self.entries["a"].configure(
            state="normal" if needs_a else "disabled"
        )

    def on_domain_change(self):

        fs = self.entries["fs"]
        fs.delete(0, "end")

        fs.insert(
            0,
            "1000" if self.var_domain.get() == "Sürekli" else "20"
        )

    def get_float(self, key, name, positive=False, nonzero=False):

        raw = self.entries[key].get().strip().replace(",", ".")

        try:
            val = float(raw)
        except ValueError:
            raise ValueError(
                f"'{name}' alanına geçerli bir sayı giriniz."
            )

        if not np.isfinite(val):
            raise ValueError(
                f"'{name}' sonlu bir sayı olmalıdır."
            )

        if positive and val <= 0:
            raise ValueError(
                f"'{name}' sıfırdan büyük olmalıdır."
            )

        if nonzero and val == 0:
            raise ValueError(
                f"'{name}' sıfır olamaz."
            )

        return val

    def reset(self):

        self.t = None
        self.x = None
        self.x_func = None
        self.info = {}

        self.draw_empty(self.ax1, "Orijinal Sinyal x(t)")
        self.draw_empty(self.ax2, "Dönüştürülmüş Sinyal y(t)")

        self.canvas.draw()
        self.txt.delete("1.0", "end")

    @staticmethod
    def common_ylim(*arrays):

        vals = [0.0]

        for arr in arrays:

            vals.append(float(np.min(np.real(arr))))
            vals.append(float(np.max(np.real(arr))))

            if np.iscomplexobj(arr):

                vals.append(float(np.min(np.imag(arr))))
                vals.append(float(np.max(np.imag(arr))))

        lo = min(vals)
        hi = max(vals)

        pad = 0.12 * (hi - lo) if hi > lo else 0.5

        return lo - pad, hi + pad

    # -----------------------------------------------------------------
    # Sinyal üretimi
    # -----------------------------------------------------------------

    def generate(self):

        try:

            kind = self.var_kind.get()
            discrete = self.var_domain.get() == "Ayrık"

            A = self.get_float("A", "Genlik")
            fs = self.get_float(
                "fs",
                "Örnekleme frekansı",
                positive=True
            )

            tmin = self.get_float("tmin", "Başlangıç zamanı")
            tmax = self.get_float("tmax", "Bitiş zamanı")

            if tmax <= tmin:
                raise ValueError(
                    "Bitiş zamanı, başlangıç zamanından büyük olmalıdır."
                )

            f = 1.0
            phi = 0.0
            a = 0.0

            if kind in PERIODIC:

                f = self.get_float(
                    "f",
                    "Frekans",
                    positive=True
                )

                phi = self.get_float("phi", "Faz")

            if kind == "Üstel":

                a = self.get_float("a", "Üstel katsayı")

            n_samples = int((tmax - tmin) * fs) + 1

            if n_samples > MAX_SAMPLES:
                raise ValueError(
                    "Örnek sayısı çok fazla. Örnekleme frekansını düşürün."
                )

            if n_samples < 2:
                raise ValueError(
                    "Çok az örnek var. Zaman aralığını genişletin."
                )

            if discrete and n_samples > MAX_DISCRETE:
                raise ValueError(
                    f"Ayrık sinyal için en fazla {MAX_DISCRETE} örnek kullanılabilir."
                )

            if kind in PERIODIC and fs < 2 * f:

                messagebox.showwarning(
                    "Nyquist Uyarısı",
                    f"fs = {fs:g} Hz, 2f = {2*f:g} Hz değerinden küçük.\n"
                    "Örtüşme (aliasing) oluşabilir."
                )

            if discrete:

                Ts = 1.0 / fs

                n = np.arange(
                    int(np.ceil(tmin * fs - 1e-9)),
                    int(np.floor(tmax * fs + 1e-9)) + 1
                )

                t = n * Ts

            else:

                t = np.linspace(tmin, tmax, n_samples)
                Ts = t[1] - t[0]

            self.x_func = make_signal(
                kind,
                A,
                f,
                phi,
                a,
                Ts
            )

            self.t = t

            if kind == "Birim Darbe (Delta)":

                self.x = delta_samples(t, 0.0, A, Ts)

            else:

                self.x = self.x_func(t)

            self.info = {
                "kind": kind,
                "discrete": discrete,
                "A": A,
                "f": f,
                "phi": phi,
                "a": a,
                "fs": fs,
                "tmin": tmin,
                "tmax": tmax,
                "Ts": Ts
            }

            self.x_title = (
                f"Orijinal Sinyal x(t): {kind} "
                f"({'Ayrık' if discrete else 'Sürekli'} zamanlı)"
            )

            self.plot_signal(
                self.ax1,
                self.t,
                self.x,
                self.x_title
            )

            self.draw_empty(
                self.ax2,
                "Dönüştürülmüş Sinyal y(t)"
            )

            self.canvas.draw()

            self.txt.delete("1.0", "end")

            self.write_report(
                "ORİJİNAL SİNYAL x(t)",
                self.t,
                self.x,
                header=True
            )

        except ValueError as err:

            messagebox.showerror(
                "Hatalı Giriş",
                str(err)
            )

    # -----------------------------------------------------------------
    # Sinyal dönüşümü
    # -----------------------------------------------------------------

    def transform(self):

        if self.x_func is None:
            self.generate()
            if self.x_func is None:
                return

        # Parametreleri oku ve doğrula
        try:
            t0 = self.get_float("t0", "Zaman kaydırma (t0)")
            a_scale = self.get_float(
                "ts", "Zaman ölçekleme (a)", nonzero=True
            )
            K = self.get_float("K", "Genlik ölçekleme (K)")
        except ValueError as err:
            messagebox.showerror("Hatalı Giriş", str(err))
            return

        inf = self.info
        reverse = self.var_rev.get()

        # Dönüşümü hesapla
        try:
            res = transform_signal(
                self.x_func, inf, K, t0, a_scale, reverse
            )
        except TransformError as err:
            messagebox.showerror("Dönüşüm Hatası", str(err))
            return

        t_y, y = res["t"], res["y"]
        lo, hi = res["lo"], res["hi"]
        c, t0_used = res["c"], res["t0_used"]

        for w in res["warnings"]:
            messagebox.showwarning("Uyarı", w)

        if res["notes"] and inf["discrete"]:
            messagebox.showinfo("Bilgi", "\n".join(res["notes"]))

        # Başlık
        parts = []
        if t0_used > 0:
            parts.append(f"{t0_used:g} sola kaydırma")
        elif t0_used < 0:
            parts.append(f"{abs(t0_used):g} sağa kaydırma")
        if abs(abs(c) - 1) > 1e-12:
            parts.append(
                f"zaman {'sıkıştırma' if abs(c) > 1 else 'genişletme'} (a={abs(c):g})"
            )
        if reverse:
            parts.append("zamanda ters")
        if K != 1:
            parts.append(f"genlik ×{K:g}")
        desc = ", ".join(parts) if parts else "dönüşüm yok"

        title = f"Dönüştürülmüş Sinyal y(t) — {desc}"

        # Çizim
        overlay = self.var_overlay.get()

        self.plot_signal(self.ax1, self.t, self.x, self.x_title)

        self.plot_signal(
            self.ax2, t_y, y, title,
            ref=(self.t, self.x) if overlay else None
        )

        ymin, ymax = self.common_ylim(self.x, y)

        # Orijinal grafiğin zaman ekseni sabit kalır.
        pad_o = 0.03 * (inf["tmax"] - inf["tmin"])
        self.ax1.set_xlim(inf["tmin"] - pad_o, inf["tmax"] + pad_o)
        self.ax1.set_ylim(ymin, ymax)

        # Dönüşmüş grafiğin ekseni: kendi penceresi (+ istenirse orijinal)
        x_lo, x_hi = lo, hi
        if overlay:
            x_lo = min(lo, inf["tmin"])
            x_hi = max(hi, inf["tmax"])
        pad_t = 0.03 * (x_hi - x_lo) if x_hi > x_lo else 0.1

        self.ax2.set_xlim(x_lo - pad_t, x_hi + pad_t)
        self.ax2.set_ylim(ymin, ymax)

        self.canvas.draw()

        # Sayısal rapor
        self.txt.delete("1.0", "end")

        self.write_report(
            "ORİJİNAL SİNYAL x(t)",
            self.t,
            self.x,
            header=True
        )

        self.txt.insert("end", "\n")

        self.write_report(
            f"DÖNÜŞTÜRÜLMÜŞ SİNYAL y(t) [{desc}]",
            t_y,
            y,
            delta_strength=res["strength"] is not None
        )

        # Dönüşüm özeti
        w = self.txt.insert
        w("end", "\n--- DÖNÜŞÜM ÖZETİ ---\n")
        w("end", f"y(t) = {K:g} · x({c:+g}·(t {t0_used:+g}))\n")
        w(
            "end",
            f"Pencere: x → [{inf['tmin']:g}, {inf['tmax']:g}] s   "
            f"y → [{lo:.4g}, {hi:.4g}] s\n"
        )

        if not inf["discrete"] and inf["kind"] != "Birim Darbe (Delta)":
            e_x = float(_trapz(np.abs(self.x) ** 2, self.t))
            e_th = K ** 2 * e_x / abs(c)
            w(
                "end",
                f"Kuramsal enerji K²·Ex/|a| = {e_th:.4f}  "
                f"(hesaplanan Ey ile karşılaştırınız)\n"
            )

        for note in res["notes"]:
            w("end", f"* {note}\n")

    # -----------------------------------------------------------------
    # Grafik çizimi
    # -----------------------------------------------------------------

    def plot_signal(self, ax, t, x, title, ref=None):

        ax.clear()

        discrete = self.info["discrete"]
        kind = self.info["kind"]

        is_complex = np.iscomplexobj(x)

        # Karşılaştırma için soluk orijinal sinyal (yalnızca gerçel kısım)
        if ref is not None:

            t_r, x_r = ref
            v_r = np.real(x_r)

            if discrete:
                ax.plot(
                    t_r, v_r, "o", color="0.65", markersize=3,
                    label="Orijinal x(t)"
                )
            elif kind == "Birim Darbe (Delta)":
                nz = np.nonzero(v_r)[0]
                if nz.size:
                    ax.vlines(
                        t_r[nz], 0, v_r[nz], colors="0.65",
                        linestyles="--", linewidth=1.5,
                        label="Orijinal x(t)"
                    )
            else:
                ax.plot(
                    t_r, v_r, "--", color="0.65", linewidth=1,
                    label="Orijinal x(t)"
                )

        def draw(vals, color, label, marker="o"):

            if discrete:

                ml, sl, bl = ax.stem(
                    t,
                    vals,
                    linefmt=color,
                    markerfmt=color + marker,
                    basefmt="k-",
                    label=label
                )

                ml.set_markersize(4)
                sl.set_linewidth(1)

            elif kind == "Birim Darbe (Delta)":

                nz = np.nonzero(vals)[0]

                ax.plot(
                    t,
                    np.zeros_like(t),
                    color=color,
                    label=label
                )

                if nz.size:

                    ax.vlines(
                        t[nz],
                        0,
                        vals[nz],
                        colors=color,
                        linewidth=2
                    )

                    mk = "^" if vals[nz][0] >= 0 else "v"

                    ax.plot(
                        t[nz],
                        vals[nz],
                        color + mk,
                        markersize=8
                    )

            elif kind in ("Kare", "Birim Basamak"):

                ax.step(
                    t,
                    vals,
                    where="post",
                    color=color,
                    label=label
                )

            else:

                ax.plot(
                    t,
                    vals,
                    color=color,
                    label=label
                )

        if is_complex:

            draw(np.real(x), "b", "Gerçel kısım Re{x}")
            draw(np.imag(x), "r", "Sanal kısım Im{x}", marker="s")

        else:

            draw(np.real(x), "b", "y" if ref is not None else "x")

        if is_complex or ref is not None:
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(title, fontsize=10)

        ax.set_xlabel(
            "Zaman t (s)" +
            (" [t = n·Ts]" if discrete else "")
        )

        ax.set_ylabel("Genlik")
        ax.grid(True, alpha=0.4)

    # -----------------------------------------------------------------
    # Sayısal rapor
    # -----------------------------------------------------------------

    def write_report(self, title, t, x, header=False, delta_strength=False):

        w = self.txt.insert
        inf = self.info

        w("end", f"=== {title} ===\n")

        if header:

            w(
                "end",
                f"Tip: {inf['kind']} | "
                f"Zaman: {'Ayrık' if inf['discrete'] else 'Sürekli'} | "
                f"A={inf['A']:g}, f={inf['f']:g} Hz, "
                f"φ={inf['phi']:g}°, a={inf['a']:g}, "
                f"fs={inf['fs']:g} Hz, "
                f"[{inf['tmin']:g}, {inf['tmax']:g}] s\n"
            )

        absx = np.abs(x)
        is_delta = inf["kind"] == "Birim Darbe (Delta)"

        # Enerji ve güç
        if inf["discrete"]:
            # E = Σ|x[n]|²,  P = E / N
            energy = float(np.sum(absx ** 2))
            power = energy / len(x)
            e_txt = f"{energy:.4f}"
            p_txt = f"{power:.4f}"

        elif is_delta:
            e_txt = "∞ (ideal darbe)"
            p_txt = "0"

        else:
            # E = ∫|x(t)|² dt  (yamuk kuralı)
            energy = float(_trapz(absx ** 2, t))
            duration = t[-1] - t[0]
            power = energy / duration if duration > 0 else float("nan")
            e_txt = f"{energy:.4f}"
            p_txt = f"{power:.4f}"

        if np.iscomplexobj(x):

            w(
                "end",
                f"Örnek sayısı N={len(x)} | "
                f"max|x|={absx.max():.4f} | "
                f"ort. Re={np.mean(x.real):.4f}, "
                f"ort. Im={np.mean(x.imag):.4f}\n"
            )

        else:

            w(
                "end",
                f"Örnek sayısı N={len(x)} | "
                f"min={x.min():.4f} | "
                f"max={x.max():.4f} | "
                f"ortalama={np.mean(x):.4f}\n"
            )

        w(
            "end",
            f"Enerji ≈ {e_txt} | "
            f"Ortalama güç ≈ {p_txt}\n"
        )

        if delta_strength:
            w("end", "(Darbe değeri = şiddet / alan)\n")

        count = min(8, len(t))

        idx = np.linspace(
            0,
            len(t) - 1,
            count
        ).astype(int)

        w("end", "   t (s)      |   değer\n")

        for i in idx:

            if np.iscomplexobj(x):

                v = (
                    f"{x[i].real:+.4f} "
                    f"{x[i].imag:+.4f}j"
                )

            else:

                v = f"{x[i]:+.4f}"

            w(
                "end",
                f"  {t[i]:+10.4f}  |  {v}\n"
            )


# =====================================================================
# PROGRAMI BAŞLAT
# =====================================================================

if __name__ == "__main__":
    SignalApp().mainloop()