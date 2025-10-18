# --- Standard Library Imports ---
import json  # NEW
import logging  # NEW
import os
import sys  # NEW
import threading

# --- Third-Party Imports ---
import tkinter as tk
import webbrowser  # FIX: needed for Help menu links
from pathlib import Path  # NEW
from tkinter import filedialog, messagebox, scrolledtext, ttk  # NEW: dialogs for GenAI prompts
from typing import Any, Callable, Optional, Protocol  # add typing for optional Pillow modules

import ttkbootstrap as tb  # NEW: use ttkbootstrap flatly light theme
from platformdirs import user_config_dir  # NEW

from .gurobi_codegen import GurobiCodeGenerator

# --- Local Imports ---
from .pyopl_core import OPLDataLexer, OPLDataParser, OPLLexer, OPLParser, solve

# NEW: model discovery (provider-specific)
from .pyopl_generative import (
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)
from .scipy_codegen_csc import SciPyCSCCodeGenerator

# NEW: settings storage constants (match sample.py strategy)
APP_NAME = "rhetor"
CONFIG_FILENAME = "settings.json"

# Pillow for image handling (optional)
# Use typed optional aliases to satisfy type checkers
PILImage: Optional[Any]
PILImageTk: Optional[Any]
try:
    from PIL import Image as PILImage
    from PIL import ImageTk as PILImageTk
except ImportError:
    PILImage = None
    PILImageTk = None
    print("Pillow not found. Install it with: pip install Pillow")

# --- Syntax Highlighting Colors ---
# Updated colors for a darker theme
TOKEN_COLORS = {
    "DVAR": "#56b6c2",  # Teal
    "INT": "#61afef",  # Blue
    "FLOAT": "#61afef",  # Blue
    "INT_POS": "#61afef",  # Blue (positive int)
    "FLOAT_POS": "#61afef",  # Blue (positive float)
    "BOOLEAN": "#e5c07b",  # Yellowish
    "BOOLEAN_LITERAL": "#e5c07b",  # Yellowish (literal)
    "RANGE": "#c678dd",  # Purple
    "PARAM": "#e5c07b",  # Yellowish
    "SET": "#e5c07b",  # Yellowish
    "SUBJECT_TO": "#a1c181",  # Greenish
    "MINIMIZE": "#a1c181",  # Greenish
    "MAXIMIZE": "#a1c181",  # Greenish
    "SUM": "#a1c181",  # Greenish
    "FORALL": "#c678dd",  # Purple
    "IN": "#c678dd",  # Purple
    "LE": "#e06c75",  # Reddish
    "GE": "#e06c75",  # Reddish
    "EQ": "#e06c75",  # Reddish
    "NEQ": "#e06c75",  # Reddish (not equal)
    "NUMBER": "#d19a66",  # Orange
    "NAME": "#abb2bf",  # Greyish (default text color)
    "ELLIPSIS": "#5c6370",  # Darker grey
    "DOTDOT": "#5c6370",  # Darker grey
    "DOT": "#5c6370",  # Darker grey (dot)
    "STRING_LITERAL": "#98c379",  # Light green
    "STRING": "#98c379",  # Light green (type keyword)
    "UMINUS": "#e06c75",  # Reddish (unary minus)
    "TUPLE": "#c678dd",  # Purple (tuple keyword)
    "COMMENT": "#5c6370",  # Darker grey (not a token, but for comments)
}


class _CodeGenerator(Protocol):
    def generate_code(self) -> str: ...


class OPLIDE(tk.Tk):
    """
    Main class for the Rhetor IDE. Handles UI setup, event binding, and core logic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Rhetor")
        self.geometry("1000x700")
        self.model_file: Optional[str] = None
        self.data_file: Optional[str] = None
        self.current_font_size = 12  # Default font size
        self.editor_font_family = "Courier New" if os.name == "nt" else "Courier"  # Use Courier for all platforms
        self.solver = tk.StringVar(value="gurobi")  # Solver selection: 'gurobi' or 'scipy'
        self.theme_var = tk.StringVar(value="flatly")  # NEW: current ttkbootstrap theme

        # NEW: GenAI selection state
        self.genai_selection_var = tk.StringVar(value="")  # stores "provider|model"
        self.genai_provider: Optional[str] = None
        self.genai_model: Optional[str] = None
        self._genai_provider_models: dict[str, list[str]] = {}
        self._genai_loading: bool = False  # NEW: avoid concurrent loads

        # NEW: init settings storage and load persisted settings
        self._init_settings_storage()
        loaded_settings = self._load_settings()
        desired_theme = None
        try:
            if isinstance(loaded_settings, dict):
                self.current_font_size = int(loaded_settings.get("font-size", self.current_font_size))
                desired_theme = loaded_settings.get("theme")
        except Exception:
            pass
        # NEW: verbose LLM logs setting (defaults True)
        self.verbose_llm_var = tk.BooleanVar(value=bool(loaded_settings.get("verbose-llm-logs", True)))  # NEW
        # NEW: track font size selection for menu highlighting
        self.font_size_var = tk.IntVar(value=self.current_font_size)

        # NEW: desired GenAI selection from settings (used after model discovery)
        self._desired_genai_provider: Optional[str] = None
        self._desired_genai_model: Optional[str] = None
        try:
            saved_sel = loaded_settings.get("genai-selection")
            if isinstance(saved_sel, str) and "|" in saved_sel:
                p_str, m_str = saved_sel.split("|", 1)
                if p_str and m_str:
                    self._desired_genai_provider = p_str
                    self._desired_genai_model = m_str
                    self.genai_selection_var.set(saved_sel)
            elif isinstance(saved_sel, dict):
                p_dict = saved_sel.get("provider")
                m_dict = saved_sel.get("model")
                if p_dict and m_dict:
                    self._desired_genai_provider = str(p_dict)
                    self._desired_genai_model = str(m_dict)
                    self.genai_selection_var.set(f"{p_dict}|{m_dict}")
        except Exception:
            pass

        # --- General Styling (ttkbootstrap 'flatly' light theme) ---
        self.style = tb.Style(theme="flatly")

        self._set_icon()
        self._setup_menu()
        # NEW: build GenAI model menus asynchronously to avoid blocking UI
        self._build_genai_model_menus_async()
        self._setup_panes()
        self._setup_status_bar()
        self._setup_tag_configs()
        # NEW: apply theme-specific editor/output colors
        self._apply_theme_colors()

        # Apply saved theme (after widgets exist) if different
        if desired_theme in ("flatly", "darkly") and desired_theme != self.theme_var.get():
            self.set_theme(desired_theme)

        # Initial status update
        self._update_caret_position(self.model_text)

        # NEW: global shortcut bindings
        self._bind_shortcuts()

        # NEW: save settings on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI Setup Methods ---
    def _set_icon(self) -> None:
        """Set the application window icon if Pillow is available and icon is present."""
        if PILImage and PILImageTk:
            try:
                import importlib.resources as pkg_resources

                try:
                    # For Python 3.9+, use files().joinpath()
                    from importlib.resources import files

                    icon_path = files("pyopl.icon").joinpath("mindset.png")
                    with icon_path.open("rb") as icon_file:
                        img = PILImage.open(icon_file)
                        photo_image = PILImageTk.PhotoImage(img)
                        self.iconphoto(False, photo_image)
                except Exception:
                    # Fallback for Python 3.7/3.8
                    with pkg_resources.path("pyopl.icon", "mindset.png") as icon_path:
                        img = PILImage.open(icon_path)
                        photo_image = PILImageTk.PhotoImage(img)
                        self.iconphoto(False, photo_image)
            except Exception as e:
                print(f"Error loading icon: {e}")
        else:
            print("Pillow not installed. Cannot set application icon.")

    def _setup_menu(self) -> None:
        """Create the application menu bar."""

        menubar = tk.Menu(self)
        self.menubar = menubar  # NEW: keep reference

        # File Menu
        filemenu = tk.Menu(menubar, tearoff=0)
        # CHANGED: add accelerator to New Model
        filemenu.add_command(label="New Model", command=self.new_model, accelerator=self._accel("N"))
        filemenu.add_separator()
        filemenu.add_command(label="Open Model...", command=self.open_model)
        filemenu.add_command(label="Open Data...", command=self.open_data)
        filemenu.add_separator()
        filemenu.add_command(label="Save", command=self.save_current_buffer, accelerator=self._accel("S"))
        filemenu.add_command(label="Save As...", command=self.save_current_buffer_as)
        filemenu.add_command(label="Export model...", command=self.export_model)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        # Run Menu
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Run Model", command=self.run_model)
        # Solver selection submenu
        solver_menu = tk.Menu(runmenu, tearoff=0)
        solver_menu.add_radiobutton(label="Gurobi", variable=self.solver, value="gurobi")
        solver_menu.add_radiobutton(label="Scipy (HiGHS)", variable=self.solver, value="scipy")
        runmenu.add_cascade(label="Solver", menu=solver_menu)
        menubar.add_cascade(label="Run", menu=runmenu)

        # NEW: GenAI Menu placeholder (populated later)
        self.genai_menu = tk.Menu(menubar, tearoff=0)
        # Initial non-blocking placeholder UI
        self.genai_menu.add_command(label="Loading models...", state="disabled")
        menubar.add_cascade(label="GenAI", menu=self.genai_menu)

        # Settings Menu (renamed from View to avoid macOS system items)
        settings_menu = tk.Menu(menubar, tearoff=0)

        # CHANGED: Font Size submenu uses radiobuttons with typed callbacks
        font_size_menu = tk.Menu(settings_menu, tearoff=0)
        for size, label in zip(
            [10, 12, 14, 16],
            ["Small (10)", "Medium (12)", "Large (14)", "Extra Large (16)"],
        ):
            font_size_menu.add_radiobutton(
                label=label,
                variable=self.font_size_var,
                value=size,
                command=self._make_change_font_cmd(size),
            )
        settings_menu.add_cascade(label="Font Size", menu=font_size_menu)

        # Theme submenu (typed callbacks)
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        theme_menu.add_radiobutton(
            label="Light (Flatly)",
            variable=self.theme_var,
            value="flatly",
            command=self._make_theme_cmd("flatly"),
        )
        theme_menu.add_radiobutton(
            label="Dark (Darkly)",
            variable=self.theme_var,
            value="darkly",
            command=self._make_theme_cmd("darkly"),
        )
        settings_menu.add_cascade(label="Theme", menu=theme_menu)

        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help Menu (typed URL openers)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="User Guide",
            command=lambda: self._open_url("https://github.com/gwr3n/PyOPL/blob/main/docs/PyOPL%20user%20guide.md"),
        )
        help_menu.add_command(
            label="Examples",
            command=lambda: self._open_url("https://github.com/gwr3n/PyOPL/blob/main/docs/PyOPL%20examples%20overview.md"),
        )
        help_menu.add_command(
            label="GitHub",
            command=lambda: self._open_url("https://github.com/gwr3n/PyOPL"),
        )
        # NEW: About dialog
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # NEW: platform-aware accelerator label
    def _accel(self, key: str) -> str:
        return f"{'Cmd' if sys.platform == 'darwin' else 'Ctrl'}+{key}"

    def new_model(self) -> None:
        """Clear the model and data editors, reset file paths, and update UI for a new model."""
        self.model_text.delete(1.0, tk.END)
        self.data_text.delete(1.0, tk.END)
        self.model_file = None
        self.data_file = None

        # Reset tab labels
        self.editor_notebook.tab(self.model_frame, text="Model")
        self.editor_notebook.tab(self.data_frame, text="Data")
        self.editor_notebook.select(self.model_frame)

        self.highlight(self.model_text)
        self.highlight(self.data_text, is_data=True)
        self.status_var.set("New model created. Ready.")

        # Clear output
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "New model created. Ready.\n")
        self.output_text.config(state="disabled")

    def _setup_panes(self) -> None:
        """Set up the main paned window and all subframes/editors/output, using tabs for Model/Data."""
        # Replace left file tree with a single vertical paned layout (Editors over Output)
        editor_output_paned = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            sashrelief=tk.FLAT,
            bd=2,
            bg="#e9ecef",  # CHANGED: light background to match flatly theme
        )
        editor_output_paned.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        self._setup_editors(editor_output_paned)
        self._setup_output(editor_output_paned)

    def _setup_editors(self, parent: tk.PanedWindow) -> None:
        """Create model and data editor frames inside a Notebook (tabs)."""
        editor_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=1)
        parent.add(editor_frame, stretch="always")

        # Notebook for Model/Data tabs
        self.editor_notebook = ttk.Notebook(editor_frame)
        self.editor_notebook.pack(fill=tk.BOTH, expand=1)

        # Model Editor tab
        self.model_frame = ttk.Frame(self.editor_notebook)
        ttk.Label(self.model_frame, text="Model (.mod)").pack(anchor="nw", padx=5, pady=5)  # CHANGED: use ttk.Label
        self.model_text = scrolledtext.ScrolledText(
            self.model_frame,
            wrap=tk.NONE,
            undo=True,
            font=(self.editor_font_family, self.current_font_size),
            bg="#ffffff",
            fg="#212529",
            insertbackground="#212529",
            relief=tk.FLAT,
            bd=0,
        )
        self.model_text.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        # Typed event handlers to avoid mypy lambda inference issue
        def _on_model_changed(event: tk.Event) -> None:
            self._on_text_change(self.model_text, False)

        def _on_data_changed(event: tk.Event) -> None:
            self._on_text_change(self.data_text, True)

        self.model_text.bind("<KeyRelease>", _on_model_changed)
        self.model_text.bind("<ButtonRelease-1>", _on_model_changed)
        self.model_text.bind("<Control-Key-a>", self._select_all_model)

        # Data Editor tab
        self.data_frame = ttk.Frame(self.editor_notebook)
        ttk.Label(self.data_frame, text="Data (.dat)").pack(anchor="nw", padx=5, pady=5)  # CHANGED: use ttk.Label
        self.data_text = scrolledtext.ScrolledText(
            self.data_frame,
            wrap=tk.NONE,
            undo=True,
            font=(self.editor_font_family, self.current_font_size),
            bg="#ffffff",
            fg="#212529",
            insertbackground="#212529",
            relief=tk.FLAT,
            bd=0,
        )
        self.data_text.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.data_text.bind("<KeyRelease>", _on_data_changed)
        self.data_text.bind("<ButtonRelease-1>", _on_data_changed)
        self.data_text.bind("<Control-Key-a>", self._select_all_data)

        # Add tabs
        self.editor_notebook.add(self.model_frame, text="Model")
        self.editor_notebook.add(self.data_frame, text="Data")

        # Update caret/highlighting when switching tabs
        self.editor_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.editor_frame = editor_frame

    def _setup_output(self, parent: tk.PanedWindow) -> None:
        """Create the output panel."""
        output_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=1)
        ttk.Label(output_frame, text="Output").pack(anchor="nw", padx=5, pady=5)  # CHANGED: use ttk.Label
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            height=12,
            font=(self.editor_font_family, self.current_font_size - 1),
            state="disabled",
            bg="#f8f9fa",  # CHANGED: light neutral background
            fg="#212529",  # CHANGED: dark text
            relief=tk.FLAT,
            bd=0,
        )
        self.output_text.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)
        parent.add(output_frame, minsize=150)

    def _setup_status_bar(self) -> None:
        """Create the status bar at the bottom of the window."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 9),
            padding=(8, 0, 0, 2),  # left, top, right, bottom
            relief=tk.FLAT,  # avoid bevel that can look cut off
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_tag_configs(self) -> None:
        """Configure syntax highlighting tags for editors."""
        for token, color in TOKEN_COLORS.items():
            self.model_text.tag_configure(token, foreground=color)
            self.data_text.tag_configure(token, foreground=color)
        # Special tag for error highlighting
        self.model_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        self.data_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        # Special tag for comments (not a token, but used in highlighting)
        self.model_text.tag_configure("COMMENT", font=("Consolas", self.current_font_size, "italic"))

    # --- Event Handlers and Core Logic ---
    def _on_text_change(self, text_widget: tk.Text, is_data: bool = False) -> None:
        """Update caret position and syntax highlighting on text change."""
        self.highlight(text_widget, is_data)
        self._update_caret_position(text_widget)

    def on_tree_select(self, event: Optional[tk.Event]) -> None:
        """Compatibility handler: no file tree in current UI; focus Model editor."""
        self.editor_notebook.select(self.model_frame)
        self.model_text.focus_set()
        self.highlight(self.model_text, is_data=False)
        self._update_caret_position(self.model_text)

    def on_tab_changed(self, event: Optional[tk.Event] = None) -> None:
        """Switch focus and update status/highlighting when the active tab changes."""
        idx = self.editor_notebook.index(self.editor_notebook.select())
        if idx == 0:
            self.model_text.focus_set()
            self.highlight(self.model_text, is_data=False)
            self._update_caret_position(self.model_text)
        else:
            self.data_text.focus_set()
            self.highlight(self.data_text, is_data=True)
            self._update_caret_position(self.data_text)

    # --- File Operations ---
    def open_model(self) -> None:
        """Open a model file and load its contents into the model editor."""
        fname = filedialog.askopenfilename(filetypes=[("Model files", "*.mod"), ("All files", "*.*")])
        if fname:
            with open(fname, "r") as f:
                self.model_text.delete(1.0, tk.END)
                self.model_text.insert(tk.END, f.read())
            self.model_file = fname
            self.highlight(self.model_text)
            self._update_caret_position(self.model_text)

            # Update tab label and switch to Model tab
            self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(fname)}")
            self.editor_notebook.select(self.model_frame)
            self.on_tab_changed(None)

    def open_data(self) -> None:
        """Open a data file and load its contents into the data editor."""
        fname = filedialog.askopenfilename(filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if fname:
            with open(fname, "r") as f:
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, f.read())
            self.data_file = fname
            self.highlight(self.data_text, is_data=True)
            self._update_caret_position(self.data_text)

            # Update tab label and switch to Data tab
            self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(fname)}")
            self.editor_notebook.select(self.data_frame)
            self.on_tab_changed(None)

    def save_model(self) -> None:
        """Save the contents of the model editor to a file."""
        if not self.model_file:
            fname = filedialog.asksaveasfilename(
                defaultextension=".mod",
                filetypes=[("Model files", "*.mod"), ("All files", "*.*")],
            )
            if not fname:
                return
            self.model_file = fname
        content = self.model_text.get(1.0, tk.END).rstrip("\n")
        with open(self.model_file, "w") as f:
            f.write(content)
        # Update tab title to reflect filename
        try:
            self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file)}")
        except Exception:
            pass

    def save_data(self) -> None:
        """Save the contents of the data editor to a file."""
        if not self.data_file:
            fname = filedialog.asksaveasfilename(
                defaultextension=".dat",
                filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
            )
            if not fname:
                return
            self.data_file = fname
        content = self.data_text.get(1.0, tk.END).rstrip("\n")
        with open(self.data_file, "w") as f:
            f.write(content)
        # Update tab title to reflect filename
        try:
            self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file)}")
        except Exception:
            pass

    def save_model_as(self) -> None:
        """Save the model to a new file and update the tab title."""
        fname = filedialog.asksaveasfilename(
            defaultextension=".mod",
            filetypes=[("Model files", "*.mod"), ("All files", "*.*")],
        )
        if not fname:
            return
        self.model_file = fname
        content = self.model_text.get(1.0, tk.END).rstrip("\n")
        with open(self.model_file, "w") as f:
            f.write(content)
        self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file)}")

    def save_data_as(self) -> None:
        """Save the data to a new file and update the tab title."""
        fname = filedialog.asksaveasfilename(
            defaultextension=".dat",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
        )
        if not fname:
            return
        self.data_file = fname
        content = self.data_text.get(1.0, tk.END).rstrip("\n")
        with open(self.data_file, "w") as f:
            f.write(content)
        self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file)}")

    # --- Syntax Highlighting ---
    def highlight(self, text_widget: tk.Text, is_data: bool = False) -> None:
        """Apply syntax highlighting to the given text widget, using both lexer and parser for model and data files."""
        # Remove previous tags
        for previous_tag in TOKEN_COLORS.keys():
            text_widget.tag_remove(previous_tag, "1.0", tk.END)
        text_widget.tag_remove("ERROR", "1.0", tk.END)

        code = text_widget.get("1.0", tk.END)

        # Store the most recent error for status bar display
        self._last_syntax_error = None

        if not is_data:
            lexer = OPLLexer()
            parser = OPLParser()
            tokens = []
            lexer_error = None
            # Lexical analysis
            try:
                tokens = list(lexer.tokenize(code))
            except Exception as e:
                lexer_error = e
                lineno = getattr(e, "lineno", 1)
                if not isinstance(lineno, int) or lineno is None:
                    lineno = 1
                error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                self._last_syntax_error = f"Lexer Error on line {lineno}: {error_message}"
            # Syntax analysis (parsing)
            if not lexer_error:
                try:
                    parser.parse(iter(tokens))
                except Exception as e:
                    lineno = getattr(e, "lineno", 1)
                    if not isinstance(lineno, int) or lineno is None:
                        lineno = 1
                    error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                    text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                    self._last_syntax_error = f"Parser Error on line {lineno}: {error_message}"
            # Apply syntax highlighting regardless of errors
            for token in tokens:
                start_idx = self._index_from_pos(code, token.index)
                end_idx = self._index_from_pos(code, token.index + len(str(token.value)))
                tag = token.type if token.type in TOKEN_COLORS else None
                if tag:
                    text_widget.tag_add(tag, start_idx, end_idx)
        else:
            lexer = OPLDataLexer()
            parser = OPLDataParser()
            tokens = []
            lexer_error = None
            # Lexical analysis for .dat
            try:
                tokens = list(lexer.tokenize(code))
            except Exception as e:
                lexer_error = e
                lineno = getattr(e, "lineno", 1)
                if not isinstance(lineno, int) or lineno is None:
                    lineno = 1
                error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                self._last_syntax_error = f"Lexer Error on line {lineno}: {error_message}"
            # Syntax analysis (parsing)
            if not lexer_error:
                try:
                    parser.parse(iter(tokens), lexer=lexer)
                except Exception as e:
                    lineno = getattr(e, "lineno", 1)
                    if not isinstance(lineno, int) or lineno is None:
                        lineno = 1
                    error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                    text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                    self._last_syntax_error = f"Parser Error on line {lineno}: {error_message}"
            # Apply basic highlighting for .dat regardless of errors
            import re

            for kw in ["param", "set", "true", "false"]:
                for m in re.finditer(r"\b" + kw + r"\b", code):
                    start = self._index_from_pos(code, m.start())
                    end = self._index_from_pos(code, m.end())
                    tag = "PARAM" if kw == "param" else "SET" if kw == "set" else "BOOLEAN"
                    text_widget.tag_add(tag, start, end)
            for m in re.finditer(r"\d+(\.\d+)?", code):
                start = self._index_from_pos(code, m.start())
                end = self._index_from_pos(code, m.end())
                text_widget.tag_add("NUMBER", start, end)

    def _index_from_pos(self, text: str, pos: int) -> str:
        """
        Converts a character offset (pos) in a string to a Tkinter Text widget index (line.char).

        Tkinter's line numbers are 1-based, and character offsets within a line are 0-based.
        This function handles newlines correctly to determine the accurate line and character.
        """
        if pos < 0:
            pos = 0
        if pos > len(text):
            pos = len(text)
        before = text[:pos]
        line = before.count("\n") + 1
        last_nl = before.rfind("\n")
        col = pos if last_nl == -1 else pos - last_nl - 1
        return f"{line}.{col}"

    # --- Font Size ---
    def _change_font_size(self, size: int) -> None:
        """
        Changes the font size of the text editors and output console.
        """
        self.current_font_size = size
        # NEW: keep menu highlight in sync
        try:
            self.font_size_var.set(size)
        except Exception:
            pass
        editor_font = (self.editor_font_family, size)
        output_font = (self.editor_font_family, size - 1 if size > 10 else size)

        self.model_text.config(font=editor_font)
        self.data_text.config(font=editor_font)
        self.output_text.config(font=output_font)

        # Re-apply comment font to reflect new base size
        self.model_text.tag_configure("COMMENT", font=(self.editor_font_family, size, "italic"))

        # Update caret position display after font size change
        self._update_caret_position(self.model_text)  # Assuming model_text is currently active, or last active)

        # NEW: persist settings
        self._save_settings()

    # --- Status Bar ---
    def _update_caret_position(self, text_widget: tk.Text) -> None:
        """
        Updates the status bar with the current caret position (line and column).
        Also, displays the most recent syntax error (if any) alongside caret position.
        """
        if text_widget.winfo_exists():
            try:
                # Get current cursor index
                index = text_widget.index(tk.INSERT)
                index_str = str(index)
                if "." in index_str:
                    caret_line, caret_col = map(int, index_str.split("."))
                else:
                    caret_line, caret_col = 1, 0

                # Gather all error tags and their line numbers
                error_lines = []
                if text_widget.tag_ranges("ERROR"):
                    tag_ranges = list(text_widget.tag_ranges("ERROR"))
                    for tag_start, tag_end in zip(tag_ranges[0::2], tag_ranges[1::2]):
                        tag_start_line = int(str(tag_start).split(".")[0])
                        tag_end_line = int(str(tag_end).split(".")[0])
                        for err_line in range(tag_start_line, tag_end_line + 1):
                            error_lines.append(err_line)

                # Try to get the error message for the current caret line
                error_msg = None
                if error_lines and caret_line in error_lines:
                    last_error = getattr(self, "_last_syntax_error", None)
                    if last_error and f"line {caret_line}" in last_error:
                        error_msg = last_error
                    else:
                        error_msg = f"Syntax Error on line {caret_line}"
                elif error_lines:
                    first_err_line = error_lines[0]
                    last_error = getattr(self, "_last_syntax_error", None)
                    if last_error and f"line {first_err_line}" in last_error:
                        error_msg = last_error
                    else:
                        error_msg = f"Syntax Error on line {first_err_line}"

                caret_msg = f"Ln {caret_line}, Col {caret_col}"
                if error_msg:
                    self.status_var.set(f"{error_msg} | {caret_msg}")
                else:
                    # Treat both model and data editors the same: show syntax status
                    self.status_var.set(f"Syntax OK | {caret_msg}")

            except tk.TclError:
                self.status_var.set("Ready")
            except Exception as e:
                import traceback

                print("[DEBUG] Exception in _update_caret_position:")
                print(f"  Exception: {e}")
                print(f"  type: {type(e)}")
                print(f"  index: {locals().get('index', None)}")
                print(f"  index_str: {locals().get('index_str', None)}")
                print(traceback.format_exc())
                self.status_var.set(f"Error updating status: {e}")
        else:
            self.status_var.set("Ready")

    # --- Editor Shortcuts ---
    def _select_all_model(self, event: Optional[tk.Event] = None) -> str:
        """Select all text in the model editor."""
        self.model_text.tag_add("sel", "1.0", tk.END)
        self.model_text.mark_set(tk.INSERT, "1.0")
        self.model_text.see(tk.INSERT)
        return "break"  # Prevent default Tkinter behavior

    def _select_all_data(self, event: Optional[tk.Event] = None) -> str:
        """Select all text in the data editor."""
        self.data_text.tag_add("sel", "1.0", tk.END)
        self.data_text.mark_set(tk.INSERT, "1.0")
        self.data_text.see(tk.INSERT)
        return "break"  # Prevent default Tkinter behavior

    # --- Model Execution ---
    def run_model(self) -> None:
        """Run the model using the current editor contents, with data file checks and error reporting."""
        import re

        model_code = self.model_text.get(1.0, tk.END).rstrip("\n")
        data_code = self.data_text.get(1.0, tk.END).rstrip("\n")
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Running model...\n")
        self.output_text.config(state="disabled")
        self.status_var.set("Running model...")
        solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"

        # --- Data file presence and validity checks ---
        # 1. Check if model references data (e.g. int X = ...; or similar)
        # 2. If so, check if data file is loaded and non-empty
        # 3. If not, show error in status bar and abort
        # 4. If data file is present, try to parse it, and if error, show error in status bar and abort

        # 1. Find all identifiers in model that are declared as external data (i.e., int X = ...; or similar)
        # Only include variables with '= ...;' or '= ...' (external data), not all declarations
        data_vars = set()
        # Match lines like: int nbSets = ...;
        for m in re.finditer(
            r"\b(?:int|float|boolean|set)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\.\.\.",
            model_code,
        ):
            data_vars.add(m.group(1))
        # Also match arrays: float cost[Sets] = ...;
        for m in re.finditer(
            r"\b(?:int|float|boolean|set)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[.*?\]\s*=\s*\.\.\.",
            model_code,
        ):
            data_vars.add(m.group(1))

        # 2. If model references data, but data file is missing or empty, show error
        if data_vars and (not self.data_file or not os.path.exists(self.data_file) or not data_code.strip()):
            self.status_var.set("Error: Data file missing or empty for required model parameters.")
            self.output_text.config(state="normal")
            self.output_text.insert(
                tk.END,
                "\nError: Data file missing or empty for required model parameters.\n",
            )
            self.output_text.config(state="disabled")
            return

        # 3. Try to parse the data file, if present
        if self.data_file and os.path.exists(self.data_file):
            try:
                from .pyopl_core import OPLDataLexer, OPLDataParser

                lexer = OPLDataLexer()
                parser = OPLDataParser()
                tokens = list(lexer.tokenize(data_code))
                parser.parse(iter(tokens), lexer=lexer)
            except Exception as e:
                self.status_var.set(f"Error: Data file failed to parse: {e}")
                self.output_text.config(state="normal")
                self.output_text.insert(tk.END, f"\nError: Data file failed to parse: {e}\n")
                self.output_text.config(state="disabled")
                return

        # 4. Optionally, check if all required data variables are present in the data file
        # This is a simple check: look for assignments to those variables in the data file
        missing_vars = []
        for var in data_vars:
            # Look for 'var =' or 'var[' (for arrays)
            if not re.search(r"\b" + re.escape(var) + r"\s*(=|\[)", data_code):
                missing_vars.append(var)
        if missing_vars:
            self.status_var.set(f"Error: Data missing for: {', '.join(missing_vars)}")
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, f"\nError: Data missing for: {', '.join(missing_vars)}\n")
            self.output_text.config(state="disabled")
            return

        def run():
            try:
                # Save temp files if not saved
                model_file = self.model_file or "temp_model.mod"
                data_file = self.data_file or "temp_data.dat"
                with open(model_file, "w") as f:
                    f.write(model_code)
                with open(data_file, "w") as f:
                    f.write(data_code)
                results = solve(model_file, data_file, solver=solver_choice)
                self.output_text.config(state="normal")
                self.output_text.insert(tk.END, f"\nSolver: {solver_choice}\n")
                self.output_text.insert(tk.END, "\nStatus: " + results.get("status", "UNKNOWN") + "\n")
                if "objective_value" in results and results["objective_value"] is not None:
                    self.output_text.insert(tk.END, f"Objective: {results['objective_value']}\n")
                if "solution" in results and results["solution"]:
                    self.output_text.insert(tk.END, "Solution:\n")
                    for k, v in results["solution"].items():
                        self.output_text.insert(tk.END, f"  {k}: {v}\n")
                if "stats" in results and results["stats"]:
                    self.output_text.insert(tk.END, "\nSolver Statistics (from 'stats' field):\n")
                    if isinstance(results["stats"], dict):
                        for stat_key, stat_value in results["stats"].items():
                            self.output_text.insert(tk.END, f"  {stat_key}: {stat_value}\n")
                    else:  # If it's a string or other format
                        self.output_text.insert(tk.END, str(results["stats"]) + "\n")
                else:
                    self.output_text.insert(
                        tk.END,
                        "\nNo detailed solver statistics available from pyopl.solve.\n",
                    )

                if "message" in results:
                    self.output_text.insert(tk.END, f"Message: {results['message']}\n")
                self.output_text.config(state="disabled")
                # Set status bar to success or solver message
                msg = results.get("message") or results.get("status", "Done")
                self.status_var.set(msg)
            except Exception as e:
                self.output_text.config(state="normal")
                self.output_text.insert(tk.END, f"\nError: {e}\n")
                self.output_text.config(state="disabled")
                self.status_var.set(f"Error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def export_model(self) -> None:
        """Export the current model as a standalone Python file using the selected solver's code generator."""
        try:
            # Grab editor contents
            model_code = self.model_text.get(1.0, tk.END).rstrip("\n")
            data_code = self.data_text.get(1.0, tk.END).rstrip("\n")

            if not model_code.strip():
                messagebox.showwarning("Export model", "Model editor is empty.")
                return

            # Parse model -> AST
            try:
                m_lexer = OPLLexer()
                m_parser = OPLParser()
                m_tokens = list(m_lexer.tokenize(model_code))
                ast = m_parser.parse(iter(m_tokens))
                if ast is None:
                    raise ValueError("Parser returned no AST.")
            except Exception as e:
                messagebox.showerror("Export model", f"Failed to parse model: {e}")
                return

            # Parse data -> data_dict (if any)
            data_dict = {}
            if data_code.strip():
                try:
                    d_lexer = OPLDataLexer()
                    d_parser = OPLDataParser()
                    d_tokens = list(d_lexer.tokenize(data_code))
                    # Some parsers need lexer=... passed in
                    parsed = d_parser.parse(iter(d_tokens), lexer=d_lexer)
                    if isinstance(parsed, dict):
                        data_dict = parsed
                except Exception as e:
                    messagebox.showerror("Export model", f"Failed to parse data: {e}")
                    return

            # Choose generator by solver selection
            solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"
            generator: _CodeGenerator
            if solver_choice == "gurobi":
                generator = GurobiCodeGenerator(ast, data_dict)
            else:
                generator = SciPyCSCCodeGenerator(ast, data_dict)

            # Generate Python code
            try:
                generated_code = generator.generate_code()
                # Strip the last line of generated_code if it exists
                lines = generated_code.rstrip("\n").split("\n")
                if lines:
                    generated_code = "\n".join(lines[:-1])
            except Exception as e:
                messagebox.showerror("Export model", f"Code generation failed: {e}")
                return

            # Ask for destination file
            default_name = "model_gurobi.py" if solver_choice == "gurobi" else "model_scipy.py"
            if self.model_file:
                base = os.path.splitext(os.path.basename(self.model_file))[0]
                default_name = f"{base}_{'gurobi' if solver_choice == 'gurobi' else 'scipy'}.py"
            dest_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                initialfile=default_name,
                filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            )
            if not dest_path:
                return

            # Write file
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(generated_code)

            self.status_var.set(f"Exported model to {dest_path}")
        except Exception as e:
            messagebox.showerror("Export model", f"Unexpected error: {e}")
            self.status_var.set(f"Export failed: {e}")

    # --- GenAI actions ---
    def _clear_output(self, header: str = "") -> None:
        """Clear the Output panel and optionally write a header line."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        if header:
            self.output_text.insert(tk.END, header + "\n")
        self.output_text.config(state="disabled")

    def _append_output(self, text: str) -> None:
        """Append text to the Output panel safely."""
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")

    def _ask_multiline(self, title: str, prompt: str, initial_text: str = "") -> Optional[str]:
        """Show a resizable multi-line prompt dialog and return the text or None if cancelled."""
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(True, True)

        # Center near parent
        try:
            self.update_idletasks()
            x = self.winfo_rootx() + 40
            y = self.winfo_rooty() + 40
            dlg.geometry(f"+{x}+{y}")
        except Exception:
            pass

        frm = ttk.Frame(dlg, padding=8)
        frm.grid(sticky="nsew")
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(1, weight=1)

        ttk.Label(frm, text=prompt, anchor="w", style="TLabel").grid(row=0, column=0, sticky="ew", pady=(0, 6))
        txt = scrolledtext.ScrolledText(
            frm,
            wrap=tk.WORD,
            width=100,
            height=20,
            font=(self.editor_font_family, self.current_font_size),
        )
        txt.grid(row=1, column=0, sticky="nsew")
        if initial_text:
            txt.insert("1.0", initial_text)
        txt.focus_set()

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(8, 0))
        result = {"text": None}

        def on_ok(event=None):
            result["text"] = txt.get("1.0", tk.END).rstrip()
            dlg.destroy()

        def on_cancel(event=None):
            result["text"] = None
            dlg.destroy()

        ok_btn = ttk.Button(btns, text="OK", command=on_ok)
        cancel_btn = ttk.Button(btns, text="Cancel", command=on_cancel)
        cancel_btn.grid(row=0, column=1, padx=(6, 0))
        ok_btn.grid(row=0, column=0)

        dlg.bind("<Escape>", on_cancel)
        dlg.bind("<Control-Return>", on_ok)
        dlg.bind("<Command-Return>", on_ok)  # macOS shortcut

        dlg.wait_window()
        return result["text"]

    def genai_generate(self) -> None:
        """Prompt user for a problem description and generate model & data via GenAI."""
        # Guard: ensure a model is selected
        if not self.genai_provider or not self.genai_model:
            messagebox.showwarning("GenAI", "No GenAI model selected.")
            return

        prompt = self._ask_multiline(
            "GenAI: Generate Model & Data",
            "Describe the optimization problem:",
            "",
        )
        if not prompt:
            return

        self.status_var.set(f"GenAI: generating with {self.genai_provider} • {self.genai_model} ...")
        self._clear_output("GenAI: Generating model and data...")

        def run():
            try:
                from .pyopl_generative import generative_solve

                # Thread-safe progress hook -> Output panel (always show essential progress)
                def progress(msg: str) -> None:
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"))

                # Optional: bridge module logger to progress (controlled by setting)
                class _ProgressLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        try:
                            text = self.format(record)
                        except Exception:
                            text = record.getMessage()
                        progress(text)

                log = logging.getLogger("pyopl.pyopl_generative")
                handler = None
                old_level = log.level
                if self.verbose_llm_var.get():  # NEW: only attach when verbose enabled
                    handler = _ProgressLogHandler()
                    handler.setLevel(logging.DEBUG)
                    log.addHandler(handler)
                    log.setLevel(logging.DEBUG)

                tmp_dir = os.path.join(os.getcwd(), "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                model_path = os.path.join(tmp_dir, "gen_pyopl_model.mod")
                data_path = os.path.join(tmp_dir, "gen_pyopl_data.dat")

                try:
                    # Pass selected provider/model and progress callback
                    assessment = generative_solve(
                        prompt,
                        model_path,
                        data_path,
                        model_name=self.genai_model,
                        llm_provider=self.genai_provider,
                        progress=progress,
                    )
                finally:
                    if handler is not None:  # NEW
                        try:
                            log.removeHandler(handler)
                            log.setLevel(old_level)
                        except Exception:
                            pass

                with open(model_path, "r") as f:
                    model_code = f.read()
                with open(data_path, "r") as f:
                    data_code = f.read()

                def apply_results():
                    # Load into editors
                    self.model_text.delete("1.0", tk.END)
                    self.model_text.insert(tk.END, model_code)
                    self.data_text.delete("1.0", tk.END)
                    self.data_text.insert(tk.END, data_code)
                    # Update file paths and tabs
                    self.model_file = model_path
                    self.data_file = data_path
                    self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(model_path)}")
                    self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(data_path)}")
                    # Highlight
                    self.highlight(self.model_text, is_data=False)
                    self.highlight(self.data_text, is_data=True)
                    # Output and status
                    self._append_output("\nGenAI: Generation complete.\n")
                    if assessment:
                        self._append_output(f"\nAssessment:\n{assessment}\n")
                    self.status_var.set("GenAI: generation complete")

                self.after(0, apply_results)

            except Exception as e:

                def on_error(e):
                    messagebox.showerror("GenAI Error", str(e))
                    self._append_output(f"\nGenAI Error: {e}\n")
                    self.status_var.set("GenAI: error")

                self.after(0, on_error, e)

        threading.Thread(target=run, daemon=True).start()

    def genai_feedback(self) -> None:
        """Prompt for a question and request feedback/revisions from GenAI for the current model/data."""
        # Guard: ensure a model is selected
        if not self.genai_provider or not self.genai_model:
            messagebox.showwarning("GenAI", "No GenAI model selected.")
            return

        question = self._ask_multiline(
            "GenAI: Ask...",
            "Enter your question about the current model/data (e.g., improvements, changes):",
            "",
        )
        if not question:
            return

        # Ensure we have model/data files to pass to GenAI; save current buffers if needed
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        model_path = self.model_file or os.path.join(tmp_dir, "current_model.mod")
        data_path = self.data_file or os.path.join(tmp_dir, "current_data.dat")
        try:
            with open(model_path, "w") as f:
                f.write(self.model_text.get("1.0", tk.END))
            with open(data_path, "w") as f:
                f.write(self.data_text.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("GenAI Error", f"Failed to save current model/data: {e}")
            return

        self.status_var.set("GenAI: requesting feedback...")
        self._clear_output("GenAI: Requesting feedback...")

        def run():
            try:
                from .pyopl_generative import generative_feedback

                # Thread-safe progress hook -> Output panel (always show essential progress)
                def progress(msg: str) -> None:
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"))

                # Optional: bridge module logger to progress (controlled by setting)
                class _ProgressLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        try:
                            text = self.format(record)
                        except Exception:
                            text = record.getMessage()
                        progress(text)

                log = logging.getLogger("pyopl.pyopl_generative")
                handler = None
                old_level = log.level
                if self.verbose_llm_var.get():  # NEW: only attach when verbose enabled
                    handler = _ProgressLogHandler()
                    handler.setLevel(logging.DEBUG)
                    log.addHandler(handler)
                    log.setLevel(logging.DEBUG)

                try:
                    result = generative_feedback(
                        question,
                        model_path,
                        data_path,
                        model_name=self.genai_model,
                        llm_provider=self.genai_provider,
                        progress=progress,
                    )
                finally:
                    if handler is not None:  # NEW
                        try:
                            log.removeHandler(handler)
                            log.setLevel(old_level)
                        except Exception:
                            pass

                feedback = result.get("feedback", "No feedback returned.")
                revised_model = result.get("revised_model", "")
                revised_data = result.get("revised_data", "")

                def after_feedback():
                    self._append_output(f"\nFeedback:\n{feedback}\n")
                    apply = False
                    if revised_model or revised_data:
                        apply = messagebox.askyesno(
                            "Apply Revisions?", "GenAI returned revised model/data. Apply these revisions to the editors?"
                        )
                    if apply:
                        if revised_model:
                            self.model_text.delete("1.0", tk.END)
                            self.model_text.insert(tk.END, revised_model)
                        if revised_data:
                            self.data_text.delete("1.0", tk.END)
                            self.data_text.insert(tk.END, revised_data)
                        # Keep files pointing to saved paths
                        self.model_file = model_path
                        self.data_file = data_path
                        # Write applied content back to files
                        try:
                            with open(model_path, "w") as f:
                                f.write(self.model_text.get("1.0", tk.END))
                            with open(data_path, "w") as f:
                                f.write(self.data_text.get("1.0", tk.END))
                        except Exception as e2:
                            self._append_output(f"\nWarning: Failed to write revisions to files: {e2}\n")
                        # Update tabs and highlighting
                        self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(model_path)}")
                        self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(data_path)}")
                        self.highlight(self.model_text, is_data=False)
                        self.highlight(self.data_text, is_data=True)
                        self._append_output("\nRevisions applied to editors.\n")
                    self.status_var.set("GenAI: feedback complete")

                self.after(0, after_feedback)

            except Exception as e:

                def on_error(e):
                    messagebox.showerror("GenAI Error", str(e))
                    self._append_output(f"\nGenAI Error: {e}\n")
                    self.status_var.set("GenAI: error")

                self.after(0, on_error, e)

        threading.Thread(target=run, daemon=True).start()

    # NEW: theme switching
    def set_theme(self, theme_name: str) -> None:
        """Switch ttkbootstrap theme and reapply widget colors."""
        if theme_name not in ("flatly", "darkly"):
            return
        self.theme_var.set(theme_name)
        try:
            self.style.theme_use(theme_name)
        except Exception:
            # fallback: recreate style
            self.style = tb.Style(theme=theme_name)
        self._apply_theme_colors()
        # Re-highlight for best contrast
        self.highlight(self.model_text, is_data=False)
        self.highlight(self.data_text, is_data=True)
        # NEW: persist settings
        self._save_settings()

    # NEW: apply text widget colors based on theme
    def _apply_theme_colors(self) -> None:
        theme = self.theme_var.get()
        if theme == "darkly":
            editor_bg = "#2b3035"
            editor_fg = "#e9ecef"
            caret_fg = "#e9ecef"
            output_bg = "#212529"
            output_fg = "#e9ecef"
            error_fg = "white"
        else:
            editor_bg = "#ffffff"
            editor_fg = "#212529"
            caret_fg = "#212529"
            output_bg = "#f8f9fa"
            output_fg = "#212529"
            error_fg = "black"

        # Apply to editors
        if hasattr(self, "model_text"):
            self.model_text.config(bg=editor_bg, fg=editor_fg, insertbackground=caret_fg)
        if hasattr(self, "data_text"):
            self.data_text.config(bg=editor_bg, fg=editor_fg, insertbackground=caret_fg)
        if hasattr(self, "output_text"):
            self.output_text.config(bg=output_bg, fg=output_fg)

        # Adjust ERROR tag for contrast
        if hasattr(self, "model_text"):
            self.model_text.tag_configure("ERROR", background="#e06c75", foreground=error_fg)
        if hasattr(self, "data_text"):
            self.data_text.tag_configure("ERROR", background="#e06c75", foreground=error_fg)

    # NEW: settings helpers (sample.py strategy)
    def _init_settings_storage(self) -> None:
        """Initialize settings storage path."""
        try:
            config_dir = Path(user_config_dir(APP_NAME))
            config_dir.mkdir(parents=True, exist_ok=True)
            self._config_path = config_dir / CONFIG_FILENAME
        except Exception:
            # Fallback to current working directory if platformdirs fails
            self._config_path = Path(os.getcwd()) / CONFIG_FILENAME

    def _load_settings(self) -> dict[str, Any]:
        """Load settings from disk."""
        try:
            if hasattr(self, "_config_path") and self._config_path.exists():
                with open(self._config_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load settings: {e}")
        return {}

    def _save_settings(self) -> None:
        """Save current settings to disk."""
        try:
            payload = {
                "theme": self.theme_var.get() if hasattr(self, "theme_var") else "flatly",
                "font-size": int(getattr(self, "current_font_size", 12)),
                "verbose-llm-logs": bool(self.verbose_llm_var.get()) if hasattr(self, "verbose_llm_var") else True,  # NEW
                # NEW: persist last selected GenAI model
                "genai-selection": (
                    f"{self.genai_provider}|{self.genai_model}"
                    if getattr(self, "genai_provider", None) and getattr(self, "genai_model", None)
                    else ""
                ),
            }
            with open(self._config_path, "w") as f:
                json.dump(payload, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to save settings: {e}")

    def _on_close(self) -> None:
        """Persist settings and close the app."""
        # Flag to refuse further UI updates from background threads
        setattr(self, "_shutting_down", True)
        self._save_settings()
        try:
            self.destroy()
        finally:
            # Ensure mainloop breaks even if destroy raised
            try:
                self.quit()
            except Exception:
                pass

    # NEW: bind Ctrl/Cmd shortcuts
    def _bind_shortcuts(self) -> None:
        # # Save current buffer
        self.bind_all("<Control-s>", self.save_current_buffer)
        self.bind_all("<Command-s>", self.save_current_buffer)
        # New model
        self.bind_all("<Control-n>", self._new_model_shortcut)
        self.bind_all("<Command-n>", self._new_model_shortcut)

    def _new_model_shortcut(self, event: Optional[tk.Event] = None) -> str:
        """Keyboard shortcut handler for creating a new model."""
        self.new_model()
        return "break"

    # NEW: save current tab (model or data)
    def save_current_buffer(self, event: Optional[tk.Event] = None) -> str:
        try:
            idx = self.editor_notebook.index(self.editor_notebook.select())
            if idx == 0:
                self.save_model()
            else:
                self.save_data()
        except Exception:
            # Fallback: try saving both if tab cannot be detected
            self.save_model()
            self.save_data()
        return "break"

    # NEW: save-as for current tab (model or data)
    def save_current_buffer_as(self, event: Optional[tk.Event] = None) -> str:
        try:
            idx = self.editor_notebook.index(self.editor_notebook.select())
            if idx == 0:
                self.save_model_as()
            else:
                self.save_data_as()
        except Exception:
            pass
        return "break"

    # NEW: About dialog handler
    def show_about(self) -> None:
        messagebox.showinfo(
            "About Rhetor",
            "Rhetor\n\n© 2025 Roberto Rossi",
        )

    # NEW: async discovery wrapper to avoid blocking UI at startup and on refresh
    def _build_genai_model_menus_async(self) -> None:
        """Discover models in a background thread and populate the GenAI menu on completion."""
        if self._genai_loading:
            return  # avoid concurrent discovery
        self._genai_loading = True

        # Update placeholder UI
        try:
            self.genai_menu.delete(0, tk.END)
        except Exception:
            pass
        self.genai_menu.add_command(label="Loading models...", state="disabled")
        try:
            self.menubar.entryconfig("GenAI", state="normal")
        except Exception:
            pass

        def discover() -> None:
            provider_models: dict[str, list[str]] = {"openai": [], "google": [], "ollama": []}
            try:
                provider_models["openai"] = list_openai_models()
            except Exception:
                provider_models["openai"] = []
            try:
                provider_models["google"] = list_gemini_models()
            except Exception:
                provider_models["google"] = []
            try:
                provider_models["ollama"] = list_ollama_models()
            except Exception:
                provider_models["ollama"] = []

            def on_done():
                self._genai_loading = False
                self._populate_genai_model_menus(provider_models)

            # Ensure UI updates happen on the main thread
            self.after(0, on_done)

        threading.Thread(target=discover, daemon=True).start()

    # NEW: populate GenAI menu given discovered models (UI-thread only)
    def _populate_genai_model_menus(self, provider_models: dict[str, list[str]]) -> None:
        """Populate the GenAI menu with provider submenus and radio items per model."""
        if getattr(self, "_shutting_down", False):
            return
        # Ensure the GenAI menu exists
        if not hasattr(self, "genai_menu"):
            self.genai_menu = tk.Menu(self.menubar, tearoff=0)
            self.menubar.add_cascade(label="GenAI", menu=self.genai_menu)

        # Clear existing GenAI menu
        try:
            self.genai_menu.delete(0, tk.END)
        except Exception:
            pass

        self._genai_provider_models = provider_models
        any_models = any(len(v) > 0 for v in provider_models.values())
        self._genai_provider_submenus: dict[str, tk.Menu] = {}

        if any_models:
            # Add provider submenus with radio selections
            def add_provider_menu(provider_label: str, provider_key: str, models: list[str]):
                sub = tk.Menu(self.genai_menu, tearoff=0)
                self._genai_provider_submenus[provider_key] = sub
                for m in sorted(models):
                    value = f"{provider_key}|{m}"
                    sub.add_radiobutton(
                        label=m,
                        variable=self.genai_selection_var,
                        value=value,
                        command=self._make_select_model_cmd(provider_key, m),
                    )
                self.genai_menu.add_cascade(label=provider_label, menu=sub)

            if provider_models.get("openai"):
                add_provider_menu("OpenAI", "openai", provider_models["openai"])
            if provider_models.get("google"):
                add_provider_menu("Gemini", "google", provider_models["google"])
            if provider_models.get("ollama"):
                add_provider_menu("Ollama", "ollama", provider_models["ollama"])

            # Actions
            self.genai_menu.add_separator()
            self.genai_menu.add_command(label="Generate Model & Data...", command=self.genai_generate)
            self.genai_menu.add_command(label="Ask...", command=self.genai_feedback)

            # NEW: toggle for verbose LLM progress logs
            self.genai_menu.add_separator()  # NEW
            self.genai_menu.add_checkbutton(  # NEW
                label="Verbose LLM progress logs",
                onvalue=True,
                offvalue=False,
                variable=self.verbose_llm_var,
                command=self._save_settings,
            )

            # Enable GenAI cascade
            try:
                self.menubar.entryconfig("GenAI", state="normal")
            except Exception:
                pass

            # NEW: Prefer saved selection if available and present; otherwise, first available
            preselected = False
            try:
                if self._desired_genai_provider and self._desired_genai_model:
                    models = provider_models.get(self._desired_genai_provider) or []
                    if self._desired_genai_model in models:
                        self.genai_selection_var.set(f"{self._desired_genai_provider}|{self._desired_genai_model}")
                        self._on_select_genai_model(self._desired_genai_provider, self._desired_genai_model)
                        preselected = True
            except Exception:
                pass

            if not preselected and not (self.genai_provider and self.genai_model):
                for pk in ("openai", "google", "ollama"):
                    if provider_models.get(pk):
                        first = provider_models[pk][0]
                        self.genai_selection_var.set(f"{pk}|{first}")
                        self._on_select_genai_model(pk, first)
                        break
        else:
            # No models available
            self.genai_menu.add_command(label="No models available", state="disabled")
            try:
                self.menubar.entryconfig("GenAI", state="disabled")
            except Exception:
                pass

    # NEW: factory to build typed callbacks for menu commands (avoids lambda mypy issue)
    def _make_select_model_cmd(self, provider_key: str, model_name: str) -> Callable[[], None]:
        def _cmd() -> None:
            self._on_select_genai_model(provider_key, model_name)

        return _cmd

    # NEW: typed helpers to replace lambdas in menus
    def _make_change_font_cmd(self, size: int) -> Callable[[], None]:
        def _cmd() -> None:
            self._change_font_size(size)

        return _cmd

    def _make_theme_cmd(self, theme: str) -> Callable[[], None]:
        def _cmd() -> None:
            self.set_theme(theme)

        return _cmd

    def _open_url(self, url: str) -> None:
        webbrowser.open_new(url)

    # NEW: selection handler for GenAI model choice
    def _on_select_genai_model(self, provider_key: str, model_name: str) -> None:
        if getattr(self, "_shutting_down", False):
            return
        self.genai_provider = provider_key
        self.genai_model = model_name
        try:
            # Ensure the variable reflects the current selection
            self.genai_selection_var.set(f"{provider_key}|{model_name}")
            self.status_var.set(f"GenAI selected: {provider_key} • {model_name}")
        except Exception:
            pass
        # NEW: persist selection immediately
        self._save_settings()


if __name__ == "__main__":
    ide = OPLIDE()
    ide.mainloop()
