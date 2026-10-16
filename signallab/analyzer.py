#!/usr/bin/env python3
"""SignalLab: desktop signal classification and visual analysis."""

import datetime
import pickle
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import core

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Utilities
def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Display item for TV
class DisplayItem:
    def __init__(
        self,
        fig: Figure | None = None,
        img=None,
        meta: dict[str, Any] | None = None,
    ):
        self.fig = fig
        self.img = img
        self.meta = meta or {"title": "Untitled", "created": now_ts()}


# Main Application
class SignalAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SignalLab | Signal Analysis & Classification")
        root.geometry("1300x850")
        root.minsize(1100, 700)

        # Data & model state
        self.signal_length = 1024
        self.data: np.ndarray | None = None  # (n, L)
        self.labels: np.ndarray | None = None  # (n,)
        self.features_full: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.all_feature_names: list[str] = []
        self.feature_names: list[str] = []
        self.model: SVC | None = None
        self.scaler: StandardScaler | None = None
        self.selector: SelectKBest | None = None

        # Display / TV
        self.display_items: list[DisplayItem] = []
        self.current_display_index: int = -1
        self.canvas_widget = None
        self.current_fig = None

        # Test signals
        self.test_signals: list[np.ndarray] = []

        # Busy state
        self._is_busy = False
        self.ui_buttons: list[tk.Widget] = []

        # Default parameters (tk vars)
        self.fs_var = tk.DoubleVar(value=1024.0)
        self.band_low = tk.DoubleVar(value=0.0)
        self.band_high = tk.DoubleVar(value=512.0)
        self.input_mode = tk.StringVar(value="synthetic")
        self.num_classes = tk.IntVar(value=3)
        self.num_samples = tk.IntVar(value=200)
        self.target_class = tk.IntVar(value=0)
        self.focus_multiplier = tk.DoubleVar(value=0.5)
        self.mat_x_key = tk.StringVar(value="X")
        self.mat_y_key = tk.StringVar(value="y")

        self.preset_var = tk.StringVar(value="basic")
        self.feature_vars: dict[str, tk.BooleanVar] = {}
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.k_best_var = tk.IntVar(value=0)

        # Build UI
        self._closed = False
        self.root.bind("<Destroy>", self._on_destroy, add="+")
        self._events = queue.Queue()
        self.pipeline = None
        self._last_cm = None
        self.test_labels = []
        self._build_ui()
        self._poll_id = self.root.after(50, self._poll_events)
        self._update_button_states()
        self._log("Application started")

    # Build UI
    def _build_ui(self):
        self.root.title("SignalLab | Signal analysis workstation")
        self.root.minsize(1000, 700)
        self.root.configure(background="#eef2f6")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".", font=("Segoe UI", 10), foreground="#182d43", background="#eef2f6"
        )
        style.configure("TNotebook", background="#eef2f6", borderwidth=0)
        style.configure(
            "TNotebook.Tab", padding=(16, 9), background="#dfe7ee", foreground="#243b50"
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#e8f1f3")],
            foreground=[("selected", "#086f76")],
        )
        style.configure(
            "TButton", padding=(10, 6), background="#ffffff", bordercolor="#bac9d4"
        )
        style.map(
            "TButton", background=[("active", "#e0eef0"), ("disabled", "#e4e9ed")]
        )
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure(
            "Horizontal.TProgressbar", background="#087e84", troughcolor="#dfe7ee"
        )
        style.configure("Accent.TButton", background="#086f76", foreground="white")
        style.map(
            "Accent.TButton",
            background=[("active", "#075b61"), ("disabled", "#526575")],
        )
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        header = ttk.Frame(self.root, padding=(16, 10))
        header.pack(fill="x")
        ttk.Label(header, text="SignalLab", font=("Segoe UI", 20, "bold")).pack(
            side="left"
        )
        ttk.Label(
            header, text="Explore signals. Build features. Evaluate honestly."
        ).pack(side="left", padx=22)
        self.workspace = ttk.Panedwindow(self.root, orient="vertical")
        self.workspace.pack(fill="both", expand=True, padx=12)
        controls = ttk.Frame(self.workspace)
        self.notebook = nb = ttk.Notebook(controls)
        nb.pack(fill="both", expand=True)
        for name, title, builder in [
            ("input", "1  Dataset", self._build_input_tab),
            ("preview", "2  Preview", self._build_preview_tab),
            ("features", "3  Features", self._build_features_tab),
            ("train", "4  Train & evaluate", self._build_train_tab),
            ("test", "5  Predict", self._build_test_tab),
            ("help", "6  Guide & activity", self._build_help_tab),
        ]:
            page = ttk.Frame(nb)
            setattr(self, "tab_" + name, page)
            nb.add(page, text=title)
            canvas = tk.Canvas(
                page, background="#eef2f6", highlightthickness=0, takefocus=True
            )
            scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            content = ttk.Frame(canvas, padding=6)
            window = canvas.create_window((0, 0), window=content, anchor="nw")
            content.bind(
                "<Configure>",
                lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")),
            )
            canvas.bind(
                "<Configure>",
                lambda event, c=canvas, w=window: c.itemconfigure(w, width=event.width),
            )
            canvas.bind(
                "<Prior>",
                lambda event, c=canvas: (c.yview_scroll(-1, "pages"), "break")[-1],
            )
            canvas.bind(
                "<Next>",
                lambda event, c=canvas: (c.yview_scroll(1, "pages"), "break")[-1],
            )
            canvas.bind(
                "<Home>", lambda event, c=canvas: (c.yview_moveto(0), "break")[-1]
            )
            canvas.bind(
                "<End>", lambda event, c=canvas: (c.yview_moveto(1), "break")[-1]
            )
            self._bind_control_scrolling(canvas, content)
            builder(content)
        self.workspace.add(controls, weight=1)
        gallery = ttk.Frame(self.workspace)
        bar = ttk.Frame(gallery, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="PLOT WORKSPACE", font=("Segoe UI", 10, "bold")).pack(
            side="left", padx=(0, 16)
        )
        for title, command in [
            ("Previous", self.display_prev),
            ("Next", self.display_next),
            ("Remove", self.display_delete_current),
            ("Clear gallery", self.display_delete_all),
        ]:
            ttk.Button(bar, text=title, command=command).pack(side="left", padx=3)
        self.gallery_label = ttk.Label(bar, text="No figures yet")
        self.gallery_label.pack(side="right", padx=8)
        self.plot_workspace = ttk.Frame(gallery)
        self.plot_workspace.pack(fill="both", expand=True)
        self.workspace.add(gallery, weight=1)
        footer = ttk.Frame(self.root, padding=(16, 7))
        footer.pack(fill="x")
        self.status_label = ttk.Label(
            footer, text="Ready - load a dataset to begin", anchor="w"
        )
        self.status_label.pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.pack(side="right")
        self._render_current_display()
        self.root.after(150, lambda: self.workspace.sashpos(0, 385))

    def _bind_control_scrolling(self, canvas, content):
        """Keep compressed notebook controls reachable by mouse and keyboard."""

        def belongs(widget):
            while widget is not None:
                if widget is canvas:
                    return True
                widget = getattr(widget, "master", None)
            return False

        def wheel(event):
            hovered = self.root.winfo_containing(event.x_root, event.y_root)
            if not belongs(hovered):
                return
            # Scroll text and list controls themselves when the pointer is over them.
            if isinstance(hovered, (tk.Text, tk.Listbox, ttk.Treeview)):
                return
            if canvas.yview() != (0.0, 1.0):
                units = (
                    (-1 if event.num == 4 else 1)
                    if event.num in (4, 5)
                    else -int(event.delta / 120)
                )
                canvas.yview_scroll(units, "units")
                return "break"

        def reveal(event):
            widget = event.widget
            if widget is canvas or not belongs(widget):
                return
            self.root.update_idletasks()
            top = widget.winfo_rooty() - content.winfo_rooty()
            bottom = top + widget.winfo_height()
            visible_top = canvas.canvasy(0)
            height = canvas.winfo_height()
            total = max(content.winfo_height(), height)
            if top < visible_top:
                canvas.yview_moveto(max(0, top - 10) / total)
            elif bottom > visible_top + height:
                canvas.yview_moveto(max(0, bottom - height + 10) / total)

        self.root.bind("<MouseWheel>", wheel, add="+")
        self.root.bind("<Button-4>", wheel, add="+")
        self.root.bind("<Button-5>", wheel, add="+")
        self.root.bind("<FocusIn>", reveal, add="+")

    # Input Tab
    def _build_input_tab(self, frame):
        ttk.Label(
            frame,
            text="Start with a synthetic dataset, CSV class folders, or a MATLAB file.",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=8, pady=6)

        # mode radios
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(anchor="w", padx=8)
        for txt, val in [
            ("Synthetic", "synthetic"),
            ("Folder (CSV)", "folder"),
            (".mat file", "mat"),
        ]:
            rb = ttk.Radiobutton(
                mode_frame, text=txt, variable=self.input_mode, value=val
            )
            rb.pack(side="left", padx=6)
            self.ui_buttons.append(rb)

        # synthetic frame
        synth = ttk.LabelFrame(frame, text="Synthetic options")
        synth.pack(fill="x", padx=8, pady=6)
        ttk.Label(synth, text="Number of classes:").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(synth, textvariable=self.num_classes, width=6).grid(
            row=0, column=1, padx=6
        )
        ttk.Label(synth, text="Total samples:").grid(
            row=0, column=2, sticky="w", padx=6
        )
        ttk.Entry(synth, textvariable=self.num_samples, width=8).grid(
            row=0, column=3, padx=6
        )
        ttk.Label(synth, text="Target class (0..N):").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(synth, textvariable=self.target_class, width=6).grid(
            row=1, column=1, padx=6
        )
        ttk.Label(synth, text="Focus strength (0..1):").grid(
            row=1, column=2, sticky="w", padx=6
        )
        ttk.Scale(
            synth,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.focus_multiplier,
            length=220,
        ).grid(row=1, column=3, padx=6)

        # fs and band
        misc = ttk.LabelFrame(frame, text="Frequency settings")
        misc.pack(fill="x", padx=8, pady=6)
        ttk.Label(misc, text="Sampling rate fs (Hz):").grid(
            row=0, column=0, sticky="w", padx=6
        )
        ttk.Entry(misc, textvariable=self.fs_var, width=10).grid(
            row=0, column=1, padx=6
        )
        ttk.Label(misc, text="Band low (Hz):").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(misc, textvariable=self.band_low, width=10).grid(
            row=0, column=3, padx=6
        )
        ttk.Label(misc, text="Band high (Hz):").grid(
            row=0, column=4, sticky="w", padx=6
        )
        ttk.Entry(misc, textvariable=self.band_high, width=10).grid(
            row=0, column=5, padx=6
        )

        # mat keys
        matf = ttk.LabelFrame(frame, text=".mat loader keys (if using .mat)")
        matf.pack(fill="x", padx=8, pady=6)
        ttk.Label(matf, text="Signals key:").grid(
            row=0, column=0, padx=6, pady=4, sticky="w"
        )
        ttk.Entry(matf, textvariable=self.mat_x_key, width=12).grid(
            row=0, column=1, padx=6
        )
        ttk.Label(matf, text="Labels key:").grid(
            row=0, column=2, padx=6, pady=4, sticky="w"
        )
        ttk.Entry(matf, textvariable=self.mat_y_key, width=12).grid(
            row=0, column=3, padx=6
        )

        # buttons
        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=8, pady=8)
        b_run = ttk.Button(
            btns,
            text="Load dataset",
            style="Accent.TButton",
            command=self._handle_input,
        )
        b_run.pack(side="left", padx=6)
        b_clear = ttk.Button(btns, text="Clear Data", command=self._clear_data)
        b_clear.pack(side="left", padx=6)
        b_export_feat = ttk.Button(
            btns, text="Export Features CSV", command=self._export_features
        )
        b_export_feat.pack(side="left", padx=6)
        self.ui_buttons.extend([b_run, b_clear, b_export_feat])

        self.input_status = ttk.Label(frame, text="No data loaded", foreground="green")
        self.input_status.pack(anchor="w", padx=8, pady=6)

    # central dispatcher for Run / Load button
    def _handle_input(self):
        if self._check_busy():
            return
        try:
            self._set_ui_busy(True)
            mode = self.input_mode.get()
            if mode == "synthetic":
                self._generate_synthetic()
            elif mode == "folder":
                self._load_from_folder()
            elif mode == "mat":
                self._load_mat_file()
        except Exception as error:
            messagebox.showerror("Input error", str(error))
        finally:
            self._set_ui_busy(False)

    # Synthetic generator
    def _generate_synthetic(self):
        classes, total = int(self.num_classes.get()), int(self.num_samples.get())
        target, focus = int(self.target_class.get()), float(self.focus_multiplier.get())
        if not 2 <= classes <= 10 or not classes * 4 <= total <= 20000:
            raise ValueError(
                "Choose 2-10 classes and at least four signals per class (20,000 total maximum)."
            )
        if not 0 <= target < classes or not 0 <= focus <= 1:
            raise ValueError(
                "Target class must be in range; focus fraction must be 0-1."
            )
        fs = float(self.fs_var.get())
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("Sampling rate must be positive.")
        rng = np.random.default_rng(42)
        counts = np.full(classes, 4, dtype=int)
        remaining = total - classes * 4
        extra = int(round(remaining * focus))
        counts[target] += extra
        others = [c for c in range(classes) if c != target]
        counts[others] += rng.multinomial(
            remaining - extra, np.ones(len(others)) / len(others)
        )
        t = np.arange(self.signal_length) / fs
        rows, labels = [], []
        for cls, count in enumerate(counts):
            for _ in range(count):
                frequency = (
                    fs / self.signal_length * (6 + cls * 9 + rng.uniform(-0.5, 0.5))
                )
                phase = rng.uniform(0, 2 * np.pi)
                row = (
                    np.sin(2 * np.pi * frequency * t + phase)
                    + 0.25 * np.sin(4 * np.pi * frequency * t + phase)
                    + rng.normal(0, 0.2, self.signal_length)
                )
                rows.append(row)
                labels.append(cls)
        order = rng.permutation(total)
        self._install_dataset(np.asarray(rows)[order], np.asarray(labels)[order])
        self.input_status.config(
            text=f"Synthetic: {total} signals / {classes} classes / seed 42"
        )
        self._log(f"Class counts: {counts.tolist()}; sampling rate {fs:g} Hz")

    # Load from folder (CSV)
    def _load_from_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        rows, labels, names = [], [], []
        for entry in sorted(Path(folder).iterdir()):
            if not entry.is_dir():
                continue
            files = sorted(entry.glob("*.csv"))
            if not files:
                continue
            label = len(names)
            names.append(entry.name)
            for file in files:
                raw = np.loadtxt(file, delimiter=",", ndmin=1)
                if raw.ndim > 1 and min(raw.shape) > 1:
                    raise ValueError(f"{file.name}: expected one signal row or column.")
                rows.append(core.signal_matrix(raw.reshape(-1))[0])
                labels.append(label)
        if not rows:
            raise ValueError(
                "Choose a folder containing class subfolders with CSV signals."
            )
        if len({len(row) for row in rows}) != 1:
            raise ValueError(
                "All signals must have equal length. No silent padding or truncation is applied."
            )
        self._install_dataset(np.asarray(rows), np.asarray(labels))
        self.input_status.config(
            text=f"CSV: {len(rows)} signals / {len(names)} classes"
        )
        self._log(
            "Class IDs: " + ", ".join(f"{i} = {name}" for i, name in enumerate(names))
        )

    # Robust MAT loader
    def _load_mat_file(self):
        path = filedialog.askopenfilename(filetypes=[("MATLAB dataset", "*.mat")])
        if not path:
            return
        X, y = core.labelled_mat(
            path, self.mat_x_key.get().strip(), self.mat_y_key.get().strip()
        )
        self._install_dataset(X, y)
        self.input_status.config(
            text=f"{Path(path).name}: {len(X)} signals / {len(np.unique(y))} classes / {X.shape[1]} samples"
        )
        self._log(
            "MAT loaded without invented labels or length changes. Set its actual sampling rate before extraction."
        )

    # Preview tab
    def _build_preview_tab(self, frame):
        ttk.Label(
            frame,
            text="Inspect the waveform before choosing features.",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=8, pady=8)
        actions = ttk.Frame(frame)
        actions.pack(anchor="w", padx=8, pady=8)
        for title, command in [
            ("One sample per class", self._show_preview),
            ("Three samples per class", self._show_three_per_class),
        ]:
            button = ttk.Button(actions, text=title, command=command)
            button.pack(side="left", padx=(0, 8))
            self.ui_buttons.append(button)
        self.custom_frame = ttk.LabelFrame(frame, text="Choose a subset")
        self.custom_frame.pack(fill="x", padx=8, pady=8)
        ttk.Label(self.custom_frame, text="Class labels, separated by commas").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        self.class_entry = ttk.Entry(self.custom_frame, width=30)
        self.class_entry.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(self.custom_frame, text="Samples per class").grid(
            row=1, column=0, sticky="w", padx=6, pady=6
        )
        self.count_entry = ttk.Entry(self.custom_frame, width=8)
        self.count_entry.insert(0, "3")
        self.count_entry.grid(row=1, column=1, sticky="w", padx=6)
        button = ttk.Button(
            self.custom_frame,
            text="Preview selected classes",
            command=self._custom_preview_from_entries,
        )
        button.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        self.ui_buttons.append(button)
        self.preview_status = ttk.Label(
            frame, text="Plots appear in the workspace below."
        )
        self.preview_status.pack(anchor="w", padx=8, pady=8)

    def _show_preview(self):
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data/labels loaded.")
            return

        classes = np.unique(self.labels)
        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        for cls in classes:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                continue
            idx = np.random.choice(idxs)
            ax.plot(self.data[idx], alpha=0.9, label=f"Class {int(cls)}")

        ax.set_title("One random sample per class")
        ax.set_xlabel("Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        self._add_display(fig, title="Preview: 1 per class")

        self.preview_status.config(text="Preview shown")
        self._log("Displayed preview (1 per class)")

    def _show_three_per_class(self):
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data/labels loaded.")
            return

        classes = np.unique(self.labels)
        for cls in classes:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                continue

            sel = np.random.choice(idxs, size=min(3, len(idxs)), replace=False)
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            for i in sel:
                ax.plot(self.data[i], alpha=0.7)
            ax.set_title(f"Class {int(cls)} - {len(sel)} samples")
            fig.tight_layout()
            self._add_display(fig, title=f"Preview Class {int(cls)}")

        self._log("Displayed 3 samples per class")
        self.preview_status.config(text="3 samples per class preview shown")

    def _custom_preview_from_entries(self):
        from tkinter import messagebox

        import numpy as np
        from matplotlib.figure import Figure

        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data/labels loaded.")
            return

        class_str = self.class_entry.get()
        if not class_str:
            return
        try:
            class_list = [int(x.strip()) for x in class_str.split(",")]
        except:
            messagebox.showerror("Error", "Invalid class input.")
            return

        count_str = self.count_entry.get()
        if not count_str:
            return
        try:
            count = int(count_str)
        except:
            messagebox.showerror("Error", "Invalid number.")
            return

        # نمایش نمونه‌ها
        for cls in class_list:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                messagebox.showwarning("Warning", f"No samples found for class {cls}")
                continue

            sel = np.random.choice(idxs, size=min(count, len(idxs)), replace=False)
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            for i in sel:
                ax.plot(self.data[i], alpha=0.7)
            ax.set_title(f"Class {int(cls)} - {len(sel)} samples")
            fig.tight_layout()
            self._add_display(fig, title=f"Custom Preview Class {int(cls)}")

        self.preview_status.config(
            text=f"Custom preview shown for classes {class_list}"
        )
        self._log(
            f"Displayed custom preview for classes {class_list} with {count} samples each"
        )

    # Features tab
    def _build_features_tab(self, frame):
        ttk.Label(
            frame,
            text="Choose features to extract and visualize.",
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=8, pady=6)

        # Presets
        preset_frame = ttk.Frame(frame)
        preset_frame.pack(anchor="w", padx=8, pady=4)
        for txt, val in [
            ("Basic", "basic"),
            ("Advanced", "advanced"),
            ("Custom", "custom"),
        ]:
            rb = ttk.Radiobutton(
                preset_frame,
                text=txt,
                variable=self.preset_var,
                value=val,
                command=self._apply_preset,
            )
            rb.pack(side="left", padx=6)
            self.ui_buttons.append(rb)

        # Features Checkboxes (Time / Freq)
        feat_frame = ttk.LabelFrame(frame, text="Time-domain Features")
        feat_frame.pack(fill="x", padx=8, pady=4)
        time_feats = [
            "mean",
            "median",
            "variance",
            "skewness",
            "kurtosis",
            "rms",
            "ptp",
            "iqr",
            "zero_crossings",
        ]
        for index, f in enumerate(time_feats):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(feat_frame, text=f, variable=var)
            cb.grid(row=index // 4, column=index % 4, sticky="w", padx=12, pady=3)
            self.feature_vars[f] = var
            self.ui_buttons.append(cb)

        freq_frame = ttk.LabelFrame(frame, text="Freq-domain Features")
        freq_frame.pack(fill="x", padx=8, pady=4)
        freq_feats = [
            "spec_centroid",
            "dom_freq",
            "spec_entropy",
            "fft_power",
            "band_power",
            "spectral_rolloff",
            "spectral_roughness",
        ]
        for index, f in enumerate(freq_feats):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(freq_frame, text=f, variable=var)
            cb.grid(row=index // 4, column=index % 4, sticky="w", padx=12, pady=3)
            self.feature_vars[f] = var
            self.ui_buttons.append(cb)

        # Buttons
        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=8, pady=6)
        ttk.Button(btns, text="Feature Info", command=self._show_feature_info).pack(
            side="left", padx=6
        )
        self.extract_button = ttk.Button(
            btns,
            text="Extract features and visualize",
            command=self._extract_and_plot_threaded,
        )
        self.extract_button.pack(side="left", padx=6)
        self.ui_buttons.append(self.extract_button)
        self.features_status = ttk.Label(frame, text="", foreground="green")
        self.features_status.pack(anchor="w", padx=8, pady=6)
        self._apply_preset()

    # Feature Computation
    def _time_features(self, sig):
        return core.time_features(sig)

    def _freq_features(self, sig, fs):
        return core.frequency_features(
            sig,
            core.FeatureConfig(
                fs, float(self.band_low.get()), float(self.band_high.get())
            ),
        )

    def _apply_preset(self):
        preset = self.preset_var.get()
        basic = {"mean", "variance", "rms", "ptp", "fft_power", "dom_freq"}
        if preset != "custom":
            for name, var in self.feature_vars.items():
                var.set(preset == "advanced" or name in basic)
        self._log(f"Feature preset: {preset}")

    # Extract & Plot Threaded
    def _extract_and_plot_threaded(self):
        if self._check_busy():
            return
        if self.data is None:
            messagebox.showerror("Features", "Load signals first.")
            return
        try:
            config = core.FeatureConfig(
                float(self.fs_var.get()),
                float(self.band_low.get()),
                float(self.band_high.get()),
            )
            names = [
                name for name in core.FEATURE_NAMES if self.feature_vars[name].get()
            ]
            data = self.data.copy()

            def done(values):
                self._invalidate_model()
                self._feature_config = config
                self.all_feature_names = list(names)
                self.feature_names = list(names)
                self.features_full = values
                self.features = values.copy()
                n = sum(name in core.TIME_FEATURES for name in names)
                self.feature_names_time = names[:n]
                self.feature_names_freq = names[n:]
                self.features_time = values[:, :n]
                self.features_freq = values[:, n:]
                self._after_extract()

            self._run_job(
                lambda: core.extract_features(data, names, config),
                done,
                "Extracting features",
            )
        except Exception as error:
            messagebox.showerror("Features", str(error))

    def _after_extract(self):
        if self.features is None:
            messagebox.showerror("Error", "No features produced.")
            return
        labels_unique = np.unique(self.labels)

        # Boxplots for time features
        for idx, fname in enumerate(self.feature_names_time):
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            data_by_class = [
                self.features_time[self.labels == cls, idx] for cls in labels_unique
            ]
            ax.boxplot(
                data_by_class,
                tick_labels=[f"Class {int(cls)}" for cls in labels_unique],
            )
            ax.set_title(f"Time Feature: {fname}")
            fig.tight_layout()
            self._add_display(fig, title=f"Time Feature: {fname}", focus=False)

        # Boxplots for freq features
        for idx, fname in enumerate(self.feature_names_freq):
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            data_by_class = [
                self.features_freq[self.labels == cls, idx] for cls in labels_unique
            ]
            ax.boxplot(
                data_by_class,
                tick_labels=[f"Class {int(cls)}" for cls in labels_unique],
            )
            ax.set_title(f"Freq Feature: {fname}")
            fig.tight_layout()
            self._add_display(fig, title=f"Freq Feature: {fname}", focus=False)

        # PCA 2D
        if self.features.shape[1] >= 2:
            try:
                pca = PCA(n_components=2)
                proj = pca.fit_transform(StandardScaler().fit_transform(self.features))
                fig = Figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                for cls in labels_unique:
                    idxs = np.where(self.labels == cls)[0]
                    ax.scatter(
                        proj[idxs, 0],
                        proj[idxs, 1],
                        label=f"Class {int(cls)}",
                        alpha=0.7,
                    )
                ax.legend()
                ax.set_title("PCA (2D)")
                fig.tight_layout()
                self._add_display(fig, title="PCA (2D)", focus=True)
            except Exception as e:
                self._log(f"PCA failed: {e}")

        self.features_status.config(
            text=f"Extracted {len(self.feature_names_time)} time + {len(self.feature_names_freq)} freq features"
        )
        self._log("Feature extraction + display items added")
        self._update_button_states()

    def _show_feature_info(self):
        messagebox.showinfo(
            "Feature definitions",
            "Time: mean, median, population variance, skewness, excess kurtosis, RMS, peak-to-peak, IQR and sign changes. Constant signals use zero skewness/kurtosis by convention.\n\nSpectral power is one-sided FFT mean-square amplitude: fft_power equals RMS squared; band_power sums bins in the selected Hz range. Centroid/rolloff use Hz; entropy uses bits. Spectral roughness measures adjacent-bin probability differences, not temporal spectral flux.\n\nSet the true dataset sampling rate before extracting features.",
        )

    # Train & Eval tab
    def _build_train_tab(self, frame):
        cfg = ttk.Frame(frame)
        cfg.pack(anchor="w", padx=8, pady=8, fill="x")
        # Row 0: test size + CV folds
        ttk.Label(cfg, text="Test size (0-1):").grid(
            row=0, column=0, padx=6, pady=4, sticky="w"
        )
        ttk.Entry(cfg, textvariable=self.test_size_var, width=6).grid(
            row=0, column=1, padx=6, sticky="w"
        )
        ttk.Label(cfg, text="CV folds (grid/cv):").grid(
            row=0, column=2, padx=6, sticky="w"
        )
        self.cv_folds_var = tk.IntVar(value=5)
        ttk.Spinbox(cfg, from_=2, to=20, textvariable=self.cv_folds_var, width=6).grid(
            row=0, column=3, padx=6, sticky="w"
        )

        # Row 1: feature selection k + scaler choice
        ttk.Label(cfg, text="SelectKBest k (0=off):").grid(
            row=1, column=0, padx=6, sticky="w", pady=4
        )
        ttk.Entry(cfg, textvariable=self.k_best_var, width=6).grid(
            row=1, column=1, padx=6, sticky="w"
        )
        ttk.Label(cfg, text="Scaler:").grid(row=1, column=2, padx=6, sticky="w")
        self.scaler_choice = tk.StringVar(value="standard")
        ttk.Combobox(
            cfg,
            textvariable=self.scaler_choice,
            values=["standard", "minmax", "none"],
            width=10,
            state="readonly",
        ).grid(row=1, column=3, padx=6, sticky="w")

        # Row 2: model selection and quick params
        ttk.Label(cfg, text="Classifier:").grid(
            row=2, column=0, padx=6, sticky="w", pady=4
        )
        self.model_choice = tk.StringVar(value="svm")
        ttk.Combobox(
            cfg,
            textvariable=self.model_choice,
            values=["svm", "random_forest", "logistic"],
            width=14,
            state="readonly",
        ).grid(row=2, column=1, padx=6, sticky="w")
        ttk.Label(cfg, text="SVM kernel / C:").grid(row=2, column=2, padx=6, sticky="w")
        self.svm_kernel = tk.StringVar(value="rbf")
        self.svm_C = tk.DoubleVar(value=1.0)
        ttk.Combobox(
            cfg,
            textvariable=self.svm_kernel,
            values=["rbf", "linear", "poly"],
            width=8,
            state="readonly",
        ).grid(row=2, column=3, sticky="w", padx=2)
        ttk.Entry(cfg, textvariable=self.svm_C, width=6).grid(
            row=2, column=4, sticky="w", padx=2
        )

        # Row 3: Grid search toggle + class weight
        self.use_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Use GridSearchCV", variable=self.use_grid_var).grid(
            row=3, column=0, padx=6, sticky="w", pady=6
        )
        ttk.Label(cfg, text="Class weight:").grid(row=3, column=1, padx=6, sticky="w")
        self.class_weight_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            cfg,
            textvariable=self.class_weight_var,
            values=["balanced", "none"],
            width=10,
            state="readonly",
        ).grid(row=3, column=2, padx=6, sticky="w")

        # Buttons row
        btns = ttk.Frame(frame)
        btns.pack(pady=4, anchor="w", padx=8, fill="x")
        # Train / incremental / show confusion / save
        self.train_button = ttk.Button(
            btns, text="Train (with options)", command=self._train_model_threaded
        )
        self.train_button.pack(side="left", padx=6)
        self.ui_buttons.append(self.train_button)

        self.inc_button = ttk.Button(
            btns,
            text="Incremental feature eval",
            command=self._incremental_feature_eval_threaded,
        )
        self.inc_button.pack(side="left", padx=6)
        self.ui_buttons.append(self.inc_button)

        self.show_conf_button = ttk.Button(
            btns, text="Show Last Confusion", command=self._show_last_confusion
        )
        self.show_conf_button.pack(side="left", padx=6)
        self.ui_buttons.append(self.show_conf_button)

        self.load_model_button = ttk.Button(
            btns, text="Load Model", command=self._quick_load_model
        )
        self.load_model_button.pack(side="left", padx=6)
        self.ui_buttons.append(self.load_model_button)

        # Progress & status
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill="x", padx=8)
        self.train_progress = ttk.Progressbar(
            status_frame, mode="indeterminate", length=300
        )
        self.train_progress.pack(side="left", padx=6, pady=4)
        self.train_status = ttk.Label(status_frame, text="", foreground="blue")
        self.train_status.pack(side="right", padx=8)

        # Text area for quick metrics (non-blocking)
        self.train_report_widget = tk.Text(
            frame,
            height=3,
            wrap="word",
            font=("Consolas", 10),
            background="white",
            foreground="#182d43",
        )
        self.train_report_widget.pack(fill="both", expand=True, padx=8, pady=4)
        try:
            self.train_report_widget.configure(state="disabled")
        except Exception:
            pass

    def _train_model_threaded(self):
        if self._check_busy():
            return
        if self.features_full is None:
            messagebox.showerror("Training", "Extract features first.")
            return
        try:
            options = dict(
                test_size=float(self.test_size_var.get()),
                k=int(self.k_best_var.get()),
                scaler=self.scaler_choice.get(),
                model=self.model_choice.get(),
                C=float(self.svm_C.get()),
                kernel=self.svm_kernel.get(),
                class_weight=self.class_weight_var.get(),
                grid=bool(self.use_grid_var.get()),
                folds=int(self.cv_folds_var.get()),
            )
            X, y = self.features_full.copy(), self.labels.copy()
            self._run_job(
                lambda: core.train(X, y, options),
                self._accept_training,
                "Training classifier",
            )
        except Exception as error:
            messagebox.showerror("Training", str(error))

    def _on_train_done(self, acc, report, cm):
        if cm is not None:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.imshow(cm, cmap="Blues")
            classes = getattr(self, "_last_classes", np.arange(len(cm)))
            ax.set(
                xticks=np.arange(len(classes)),
                yticks=np.arange(len(classes)),
                xticklabels=classes,
                yticklabels=classes,
                xlabel="Predicted class",
                ylabel="Actual class",
                title="Reserved holdout confusion matrix",
            )
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#172b4d",
                    )
            fig.tight_layout()
            self._add_display(fig, "Holdout confusion matrix", True)
        self.train_report_widget.configure(state="normal")
        self.train_report_widget.delete("1.0", "end")
        self.train_report_widget.insert(
            "end", f"Holdout accuracy: {acc:.1%}\n\n{report}"
        )
        self.train_report_widget.configure(state="disabled")
        self._log(f"Training complete. Reserved holdout accuracy {acc:.1%}")

    def _quick_load_model(self):
        if self._check_busy():
            return
        if not messagebox.askyesno(
            "Trusted model only",
            "Pickle files can execute code. Only open a model you created or explicitly trust. Continue?",
        ):
            return
        path = filedialog.askopenfilename(filetypes=[("SignalLab model", "*.pkl")])
        if not path:
            return
        try:
            with open(path, "rb") as stream:
                package = pickle.load(stream)
            core.validate_package(package)
            self._invalidate_model()
            self.pipeline = package["pipeline"]
            self.model = self.pipeline
            self._model_names = list(package["features"])
            self._model_config = core.FeatureConfig(**package["config"])
            self._model_length = package["signal_length"]
            self.train_status.config(text="Imported model ready for prediction")
            self._log(f"Imported validated model: {Path(path).name}")
        except Exception as error:
            messagebox.showerror("Model import", str(error))

    def _incremental_feature_eval_threaded(self):
        if self._check_busy():
            return
        if self.features_full is None:
            messagebox.showerror("Evaluation", "Extract features first.")
            return
        try:
            X, y = self.features_full.copy(), self.labels.copy()
            fraction = float(self.test_size_var.get())
            folds = int(self.cv_folds_var.get())
            if folds < 2:
                raise ValueError("Choose at least two CV folds.")
            self._run_job(
                lambda: core.incremental_evaluation(X, y, fraction, folds),
                self._after_incremental_eval,
                "Evaluating feature counts",
            )
        except Exception as error:
            messagebox.showerror("Evaluation", str(error))

    def _after_incremental_eval(self, result):
        self._accept_training(result)
        fig = Figure(figsize=(9, 4))
        ax = fig.add_subplot(111)
        ax.plot(
            range(1, len(result["scores"]) + 1),
            result["scores"],
            marker="o",
            color="#087f8c",
        )
        ax.set(
            xlabel="Feature count (ANOVA selected inside each fold)",
            ylabel="Mean training CV accuracy",
            title="Feature-count cross-validation",
            ylim=(0, 1.02),
        )
        ax.axvline(
            result["best_k"],
            color="#d97706",
            linestyle="--",
            label=f"Selected k = {result['best_k']}",
        )
        ax.legend()
        fig.tight_layout()
        self._add_display(fig, "Incremental cross-validation", True)

    def _show_last_confusion(self):
        if self._last_cm is None:
            messagebox.showinfo("Evaluation", "Train a model first.")
            return
        for idx, item in enumerate(self.display_items):
            if item.meta.get("title") == "Holdout confusion matrix":
                self.current_display_index = idx
                self._render_current_display()
                return
        messagebox.showinfo(
            "Evaluation",
            "This plot was removed from the gallery. Train again to regenerate it.",
        )

    # Test / TV tab
    def _build_test_tab(self, frame):
        top = ttk.Frame(frame)
        top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Button(
            top,
            text="Generate 3 Synthetic Signals",
            command=lambda: self._generate_test_signals(from_dataset=False, n=3),
        ).pack(side="left", padx=6)
        ttk.Button(
            top,
            text="Pick 3 From Dataset",
            command=lambda: self._generate_test_signals(from_dataset=True, n=3),
        ).pack(side="left", padx=6)
        ttk.Button(top, text="Export Model (pickle)", command=self._export_model).pack(
            side="left", padx=6
        )

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Dataset sample index:").pack(side="left", padx=6)
        self.test_index_var = tk.IntVar(value=0)
        ttk.Spinbox(
            top, from_=0, to=999999, textvariable=self.test_index_var, width=6
        ).pack(side="left", padx=4)
        ttk.Button(
            top,
            text="Test Index",
            command=self._test_dataset_index,
        ).pack(side="left", padx=6)

        # TV area with stable subframes
        self.tv_holder = ttk.Frame(frame)
        self.tv_holder.pack(fill="both", expand=True, padx=8, pady=6)
        self.tv_buttons = ttk.Frame(self.tv_holder)
        self.tv_buttons.pack(side="top", fill="x")
        self.tv_plot = ttk.Frame(self.tv_holder)
        self.tv_plot.pack(fill="both", expand=True)
        self.tv_ctrl = ttk.Frame(self.tv_holder)
        self.tv_ctrl.pack(side="bottom", fill="x", pady=6)

        # initial hint
        for w in self.tv_plot.winfo_children():
            w.destroy()
        ttk.Label(
            self.tv_plot,
            text="Select a generated sample above to inspect its prediction. Dataset plots remain in the workspace below.",
        ).pack()

    def _generate_test_signals(self, from_dataset=False, n=3):
        if self._check_busy():
            return
        if self.model is None:
            messagebox.showerror("Prediction", "Train or import a model first.")
            return
        rng = np.random.default_rng()
        if from_dataset:
            if self.data is None:
                messagebox.showerror("Prediction", "Load a dataset first.")
                return
            indices = rng.choice(
                len(self.data), size=min(n, len(self.data)), replace=False
            )
            self.test_signals = [self.data[i].copy() for i in indices]
            self.test_labels = [int(self.labels[i]) for i in indices]
            self._log(
                "Dataset predictions may include training samples; this is not a fresh accuracy estimate."
            )
        else:
            length = self._model_length
            fs = self._model_config.fs
            t = np.arange(length) / fs
            self.test_signals = [
                np.sin(2 * np.pi * rng.uniform(4, 35) * fs / length * t)
                + rng.normal(0, 0.2, length)
                for _ in range(n)
            ]
            self.test_labels = [None] * n
            self._log(
                "Unlabelled synthetic probes are demonstrations, not real-data accuracy measurements."
            )
        self._update_test_buttons()
        self._show_test_signal(0)

    def _update_test_buttons(self):
        for w in self.tv_buttons.winfo_children():
            w.destroy()
        for i in range(len(self.test_signals)):
            ttk.Button(
                self.tv_buttons,
                text=f"Test {i + 1}",
                command=lambda idx=i: self._show_test_signal(idx),
            ).pack(side="left", padx=4)
        ttk.Button(
            self.tv_buttons, text="Clear Tests", command=self._clear_test_signals
        ).pack(side="left", padx=6)

    def _clear_test_signals(self):
        self.test_signals = []
        self.test_labels = []
        for w in self.tv_plot.winfo_children():
            w.destroy()
        for w in self.tv_ctrl.winfo_children():
            w.destroy()
        for w in self.tv_buttons.winfo_children():
            w.destroy()
        self._log("Cleared test signals")

    def _show_test_signal(self, idx):
        if self._check_busy() or not 0 <= idx < len(self.test_signals):
            return
        sig = np.asarray(self.test_signals[idx], dtype=float)
        for parent in (self.tv_plot, self.tv_ctrl):
            for child in parent.winfo_children():
                child.destroy()
        fig = Figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        fs = (
            self._model_config.fs
            if self.model is not None
            else float(self.fs_var.get())
        )
        ax.plot(np.arange(len(sig)) / fs, sig, color="#087f8c", linewidth=0.8)
        ax.set(
            title=f"Prediction probe {idx + 1}", xlabel="Time (s)", ylabel="Amplitude"
        )
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tv_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Button(
            self.tv_ctrl,
            text="Previous",
            command=lambda: self._show_test_signal((idx - 1) % len(self.test_signals)),
        ).pack(side="left", padx=4)
        ttk.Button(
            self.tv_ctrl,
            text="Next",
            command=lambda: self._show_test_signal((idx + 1) % len(self.test_signals)),
        ).pack(side="left", padx=4)
        ttk.Button(
            self.tv_ctrl, text="Delete", command=lambda: self._delete_test_signal(idx)
        ).pack(side="left", padx=4)
        try:
            if self.model is None:
                raise ValueError("Train or import a model first.")
            X = core.prediction_features(
                sig, self._model_names, self._model_config, self._model_length
            )
            info = f"Predicted class: {self.pipeline.predict(X)[0]}"
            if hasattr(self.pipeline, "predict_proba"):
                info += " | " + ", ".join(
                    f"{cls}: {prob:.1%}"
                    for cls, prob in zip(
                        self.pipeline.classes_, self.pipeline.predict_proba(X)[0]
                    )
                )
            label = self.test_labels[idx] if idx < len(self.test_labels) else None
            if label is not None:
                info += f" | Dataset label: {label}"
            ttk.Label(self.tv_ctrl, text=info, wraplength=650).pack(
                side="left", padx=12
            )
            self._log(info)
        except Exception as error:
            ttk.Label(self.tv_ctrl, text=str(error), wraplength=650).pack(
                side="left", padx=8
            )

    def _delete_test_signal(self, idx):
        if not hasattr(self, "test_signals"):
            return
        if 0 <= idx < len(self.test_signals):
            del self.test_signals[idx]
            if hasattr(self, "test_labels") and idx < len(self.test_labels):
                del self.test_labels[idx]
            self._log(f"Deleted test signal {idx + 1}")
            if self.test_signals:
                self._update_test_buttons()
                self._show_test_signal(min(idx, len(self.test_signals) - 1))
            else:
                for w in self.tv_plot.winfo_children():
                    w.destroy()
                for w in self.tv_ctrl.winfo_children():
                    w.destroy()
                for w in self.tv_buttons.winfo_children():
                    w.destroy()

    # helper: test a specific dataset index (used by Test Index button)
    def _test_dataset_index(self, idx=None):
        if self._check_busy():
            return
        try:
            idx = int(self.test_index_var.get()) if idx is None else int(idx)
            if self.data is None:
                raise ValueError("Load a dataset first.")
            if not 0 <= idx < len(self.data):
                raise ValueError(f"Index must be between 0 and {len(self.data) - 1}.")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Dataset prediction", str(error))
            return
        self.test_signals = [self.data[idx].copy()]
        self.test_labels = [int(self.labels[idx])]
        self._update_test_buttons()
        self._show_test_signal(0)

    # Display / gallery helpers
    def _add_display(self, fig: Figure, title: str = "Figure", focus: bool = False):
        item = DisplayItem(fig, img=None, meta={"title": title, "created": now_ts()})
        self.display_items.append(item)
        self._log(f"Added display '{title}'")
        if focus:
            self.current_display_index = len(self.display_items) - 1
            self._render_current_display()
        else:
            # if none focused, show it too but put at end (user can navigate)
            if self.current_display_index == -1:
                self.current_display_index = len(self.display_items) - 1
                self._render_current_display()

    def _render_current_display(self):
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        for widget in self.plot_workspace.winfo_children():
            widget.destroy()
        self.canvas_widget = None
        self.current_fig = None
        if not 0 <= self.current_display_index < len(self.display_items):
            self.gallery_label.configure(text="No figures yet")
            empty = ttk.Frame(self.plot_workspace, padding=28)
            empty.pack(fill="both", expand=True)
            ttk.Label(
                empty, text="Your signal workspace", font=("Segoe UI", 18, "bold")
            ).pack(pady=(12, 8))
            ttk.Label(
                empty,
                text="Load or generate signals, then preview waveforms or extract features.",
            ).pack()
            ttk.Label(
                empty,
                text="Drag the divider above to give plots more room. Use the plot toolbar to zoom, pan, or save.",
            ).pack(pady=8)
            return
        item = self.display_items[self.current_display_index]
        if item.fig is not None:
            self.canvas_widget = FigureCanvasTkAgg(item.fig, master=self.plot_workspace)
            toolbar = NavigationToolbar2Tk(
                self.canvas_widget, self.plot_workspace, pack_toolbar=False
            )
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            self.canvas_widget.draw_idle()
            self.current_fig = item.fig
            title = item.meta.get("title", "Figure")
            self.gallery_label.configure(
                text=f"{self.current_display_index + 1} / {len(self.display_items)}  {title}"
            )

    def display_prev(self):
        if not self.display_items:
            return
        self.current_display_index = max(0, self.current_display_index - 1)
        self._render_current_display()

    def display_next(self):
        if not self.display_items:
            return
        self.current_display_index = min(
            len(self.display_items) - 1, self.current_display_index + 1
        )
        self._render_current_display()

    def display_delete_current(self):
        if self.current_display_index < 0 or self.current_display_index >= len(
            self.display_items
        ):
            return
        title = self.display_items[self.current_display_index].meta.get("title", "")
        del self.display_items[self.current_display_index]
        if not self.display_items:
            self.current_display_index = -1
        else:
            self.current_display_index = min(
                self.current_display_index, len(self.display_items) - 1
            )
        self._log(f"Deleted display item '{title}'")
        self._render_current_display()

    def display_delete_all(self):
        if not self.display_items:
            return
        count = len(self.display_items)
        self.display_items.clear()
        self.current_display_index = -1
        self._log(f"Deleted all {count} display items")
        self._render_current_display()

    # Export / Save
    def _export_features(self):
        if self.features is None or not self.feature_names:
            messagebox.showerror("Error", "No features to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        if len(self.labels) != len(self.features):
            messagebox.showerror(
                "Error", "Labels length does not match features. Cannot export."
            )
            return
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df["label"] = self.labels
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Features saved to {path}")
        self._log(f"Exported features to {path}")

    def _export_model(self):
        if self._check_busy():
            return
        if self.model is None:
            messagebox.showerror("Model export", "Train or import a model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("SignalLab model", "*.pkl")]
        )
        if not path:
            return
        try:
            package = core.model_package(
                self.pipeline, self._model_names, self._model_config, self._model_length
            )
            with open(path, "wb") as stream:
                pickle.dump(package, stream)
            self._log(
                f"Saved model with frozen feature/configuration/version metadata: {Path(path).name}"
            )
            messagebox.showinfo(
                "Saved",
                "Model saved. Load only trusted files in a compatible environment.",
            )
        except Exception as error:
            messagebox.showerror("Model export", str(error))

    # Help & Log tab
    def _build_help_tab(self, frame):
        panes = ttk.Panedwindow(frame, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=1)
        ttk.Label(left, text="WORKFLOW GUIDE", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=6
        )
        help_text = ScrolledText(
            left,
            width=40,
            height=8,
            wrap="word",
            font=("Segoe UI", 10),
            background="white",
            foreground="#182d43",
            relief="flat",
            padx=10,
            pady=10,
        )
        help_text.insert("1.0", self._help_content())
        help_text.configure(state="disabled")
        help_text.pack(fill="both", expand=True)
        ttk.Label(right, text="ACTIVITY LOG", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=6
        )
        self.log_widget = ScrolledText(
            right,
            width=40,
            height=8,
            wrap="word",
            font=("Consolas", 10),
            background="white",
            foreground="#182d43",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_widget.pack(fill="both", expand=True)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Clear log", command=self._clear_log).pack(side="left")
        ttk.Button(buttons, text="Save log", command=self._save_log).pack(
            side="left", padx=8
        )

    def _help_content(self):
        return "SignalLab | signal analysis and classification\n\n1. Load class-labelled MATLAB data, CSV class folders, or reproducible synthetic signals.\n2. Preview signals. Set the real sampling rate (Hz) and band within Nyquist.\n3. Choose features and extract boxplots/PCA. Custom preserves choices.\n4. Train SVM, random forest or logistic regression. Selection and scaling stay inside each training/CV fold.\n5. Incremental evaluation selects k with training cross-validation then reports a reserved holdout score. Repeated experiments can still overfit the holdout.\n6. Test probes and export features or a complete versioned model.\n\nMAT: X plus integer y, or X1, X2, ... class matrices. Unlabelled/struct data belongs in the inspector (python main.py --inspect). No invented labels or silent length changes. MATLAB v7.3 is unsupported: save -v7.\n\nThis app classifies independent signals; it does not perform source separation. Related windows/subjects require group-aware splits outside this app. Probabilities are model estimates, not calibrated certainty. Synthetic success does not establish real-world accuracy.\n\nPickle files execute code: load only trusted files in a matching scikit-learn major/minor environment."

    def _clear_log(self):
        try:
            self.log_widget.delete("1.0", "end")
            self._log("Cleared log")
        except Exception:
            pass

    def _save_log(self):
        try:
            txt = self.log_widget.get("1.0", "end")
            path = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text", "*.txt")]
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Saved", f"Log saved to {path}")
            self._log(f"Saved log to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {e}")

    # Busy helpers & logging
    def _set_ui_busy(self, busy):
        self._is_busy = busy
        if busy:
            self._disabled_widgets = []

            def disable(parent):
                for widget in parent.winfo_children():
                    if isinstance(
                        widget,
                        (
                            ttk.Button,
                            ttk.Entry,
                            ttk.Combobox,
                            ttk.Checkbutton,
                            ttk.Radiobutton,
                            ttk.Spinbox,
                            tk.Button,
                            tk.Entry,
                            tk.Scale,
                        ),
                    ):
                        try:
                            state = str(widget.cget("state"))
                            self._disabled_widgets.append((widget, state))
                            widget.configure(state="disabled")
                        except tk.TclError:
                            pass
                    disable(widget)

            disable(self.root)
            self.progress.start(10)
            if hasattr(self, "train_progress"):
                self.train_progress.start(10)
        else:
            for widget, state in getattr(self, "_disabled_widgets", []):
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
            self._disabled_widgets = []
            self.progress.stop()
            if hasattr(self, "train_progress"):
                self.train_progress.stop()
            self._update_button_states()

    def _check_busy(self):
        return self._is_busy

    def _log(self, msg):
        if threading.current_thread() is not threading.main_thread():
            self._events.put(lambda: self._log(msg))
            return
        full = f"[{now_ts()}] {msg}\n"
        if hasattr(self, "log_widget"):
            self.log_widget.insert("end", full)
            self.log_widget.see("end")
        print(full, end="")

    def _update_button_states(self):
        try:
            if hasattr(self, "extract_button"):
                state = (
                    "normal"
                    if (self.data is not None and self.labels is not None)
                    else "disabled"
                )
                self.extract_button.configure(state=state)
        except Exception:
            pass
        try:
            if hasattr(self, "train_button"):
                state = (
                    "normal"
                    if (self.features_full is not None and self.labels is not None)
                    else "disabled"
                )
                self.train_button.configure(state=state)
                self.inc_button.configure(state=state)
        except Exception:
            pass

    def _clear_data(self):
        if self._check_busy():
            return
        self.data = None
        self.labels = None
        self.features = None
        self.features_full = None
        self.all_feature_names = []
        self.feature_names = []
        self._invalidate_model()
        self._clear_test_signals()
        self.display_items = []
        self.current_display_index = -1
        self._render_current_display()
        self.input_status.config(text="No dataset loaded")
        self._update_button_states()
        self._log("Cleared analysis session")

    def _poll_events(self):
        try:
            while True:
                self._events.get_nowait()()
        except queue.Empty:
            pass
        finally:
            if not self._closed:
                self._poll_id = self.root.after(50, self._poll_events)

    def _on_destroy(self, event):
        if event.widget is self.root:
            self._closed = True
            if hasattr(self, "_poll_id"):
                self.root.after_cancel(self._poll_id)

    def _run_job(self, work, done, title):
        self._set_ui_busy(True)
        self._log(title)

        def execute():
            try:
                result = work()

                def finish():
                    try:
                        done(result)
                    except Exception as error:
                        self._log(str(error))
                        messagebox.showerror(title, str(error))
                    finally:
                        self._set_ui_busy(False)

                self._events.put(finish)
            except Exception as error:
                message = str(error)

                def fail():
                    self._set_ui_busy(False)
                    self._log(message)
                    messagebox.showerror(title, message)

                self._events.put(fail)

        threading.Thread(target=execute, daemon=True).start()

    def _invalidate_model(self):
        self.pipeline = None
        self.model = None
        self.scaler = None
        self.selector = None
        self._last_cm = None
        if hasattr(self, "train_status"):
            self.train_status.config(text="No model trained")
        if hasattr(self, "train_report_widget"):
            self.train_report_widget.configure(state="normal")
            self.train_report_widget.delete("1.0", "end")
            self.train_report_widget.configure(state="disabled")

    def _install_dataset(self, X, y):
        X = core.signal_matrix(X, sample_axis=1)
        if len(X) != len(y):
            raise ValueError("Signal and label counts differ.")
        self._invalidate_model()
        self.data = X
        self.labels = np.asarray(y)
        self.signal_length = X.shape[1]
        self.features = None
        self.features_full = None
        self.all_feature_names = []
        self.feature_names = []
        self._clear_test_signals()
        self.display_items = []
        self.current_display_index = -1
        self._render_current_display()
        self._update_button_states()

    def _accept_training(self, result):
        self.pipeline = result["pipeline"]
        self.model = self.pipeline
        self._model_names = list(self.all_feature_names)
        self._model_config = self._feature_config
        self._model_length = self.signal_length
        self._last_cm = result["cm"]
        self._last_classes = result["classes"]
        selector = self.pipeline.named_steps["select"]
        indices = (
            selector.get_support(indices=True)
            if selector != "passthrough"
            else np.arange(len(self.all_feature_names))
        )
        self.feature_names = [self.all_feature_names[i] for i in indices]
        self.features = self.features_full[:, indices]
        self._on_train_done(result["accuracy"], result["report"], result["cm"])
        self.train_status.config(
            text=f"Holdout: {result['accuracy']:.1%} | train {result['train_count']} / test {result['test_count']}"
        )
        self._log(f"Selected features: {', '.join(self.feature_names)}")
        if result.get("best_params"):
            self._log(f"CV parameters: {result['best_params']}")


# Run
def main():
    root = tk.Tk()
    SignalAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
