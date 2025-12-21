# --- Standard Library Imports ---
import json
import logging
import multiprocessing
import os
import queue
import sys
import threading

# --- Third-Party Imports ---
import tkinter as tk
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Optional, Protocol

import ttkbootstrap as tb
from platformdirs import user_config_dir

# Model discovery (provider-specific)
from .genai.pyopl_generative import (
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)
from .gurobi_codegen import GurobiCodeGenerator

# --- Local Imports ---
from .pyopl_core import OPLDataLexer, OPLDataParser, OPLLexer, OPLParser
from .scipy_codegen_csc import SciPyCSCCodeGenerator

# Settings storage (same strategy as sample.py)
APP_NAME = "rhetor"
CONFIG_FILENAME = "settings.json"

# Pillow (optional) for window icon
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


def _solve_wrapper(model_file: str, data_file: str, solver_choice: str, q: multiprocessing.Queue) -> None:
    """Wrapper to run solve in a separate process."""
    try:
        try:
            from .pyopl_core import solve  # package import
        except ImportError:
            from pyopl.pyopl_core import solve  # type: ignore

        results = solve(model_file, data_file, solver=solver_choice)
        q.put(("success", results))
    except Exception as e:
        q.put(("error", f"{e}\n\n{traceback.format_exc()}"))


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
        self.current_font_size = 12
        self.editor_font_family = "Courier New" if os.name == "nt" else "Courier"
        self.solver = tk.StringVar(value="gurobi")  # 'gurobi' or 'scipy'
        self.theme_var = tk.StringVar(value="flatly")

        # Solver process
        self._solver_process: Optional[multiprocessing.Process] = None
        self._solver_queue: Optional[multiprocessing.Queue] = None
        self._current_solver_choice: str = "gurobi"

        # --- Run timer (status bar elapsed time while solving) ---
        self._run_started_at: Optional[float] = None
        self._run_timer_after_id: Optional[str] = None
        self._run_status_base: str = "Running model..."

        # --- Highlight scheduling (prevents UI lag on large files) ---
        self._highlight_debounce_ms = 150  # fast pass while typing
        self._highlight_validate_idle_ms = 800  # expensive lex/parse after idle
        self._highlight_after_ids: dict[tuple[int, str], str] = {}

        # Track last syntax error per editor (prevents cross-editor contamination)
        self._last_syntax_error_by_widget: dict[int, Optional[str]] = {}

        # GenAI selection state
        self.genai_selection_var = tk.StringVar(value="")  # format: "provider|model"
        self.genai_provider: Optional[str] = None
        self.genai_model: Optional[str] = None
        self._genai_provider_models: dict[str, list[str]] = {}
        self._genai_loading: bool = False

        # Output sessions
        self._output_sessions: dict[str, str] = {}
        self._output_session_ids: list[str] = []
        self._output_session_display: dict[str, str] = {}
        self._current_output_session_id: Optional[str] = None
        self._viewing_output_session_id: Optional[str] = None

        # Settings
        self._init_settings_storage()
        loaded_settings = self._load_settings()
        desired_theme = None
        try:
            if isinstance(loaded_settings, dict):
                self.current_font_size = int(loaded_settings.get("font-size", self.current_font_size))
                desired_theme = loaded_settings.get("theme")
        except Exception:
            pass
        # LLM progress logs in Output
        self.verbose_llm_var = tk.BooleanVar(value=bool(loaded_settings.get("verbose-llm-logs", True)))
        # Track font size selection for menu state
        self.font_size_var = tk.IntVar(value=self.current_font_size)

        # GenAI method selection (persisted)
        self._genai_methods: list[tuple[str, str]] = [
            ("Generhetor", "pyopl_generative"),
            ("Standard", "pyopl_standard"),
            ("Chain of Thought", "pyopl_chain_of_thought"),
            ("Tree of Thoughts", "pyopl_tree_of_thoughts"),
            ("CAFA", "pyopl_cafa"),
            ("Chain of Experts", "pyopl_chain_of_experts"),
            ("Reflexion", "pyopl_reflexion"),
        ]
        saved_method = loaded_settings.get("genai-method") if isinstance(loaded_settings, dict) else None
        if not isinstance(saved_method, str) or not saved_method:
            saved_method = "pyopl_generative"
        self.genai_method_var = tk.StringVar(value=saved_method)

        # Desired GenAI selection from settings (applied after model discovery)
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

        # Styling (ttkbootstrap 'flatly' theme by default)
        self.style = tb.Style(theme="flatly")

        self._set_icon()
        self._setup_menu()
        # Build GenAI model menus asynchronously
        self._build_genai_model_menus_async()
        self._setup_panes()
        self._setup_status_bar()
        self._setup_tag_configs()
        # Apply theme-specific colors
        self._apply_theme_colors()

        # Apply saved theme after widgets exist
        if desired_theme in ("flatly", "darkly") and desired_theme != self.theme_var.get():
            self.set_theme(desired_theme)

        # Initial status update
        self._update_caret_position(self.model_text)

        # Global shortcuts
        self._bind_shortcuts()

        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI Setup Methods ---
    def _set_icon(self) -> None:
        """Set the application window icon if Pillow is available and the icon is present."""
        if PILImage and PILImageTk:
            try:
                import importlib.resources as pkg_resources

                try:
                    from importlib.resources import files

                    icon_path = files("pyopl.icon").joinpath("mindset.png")
                    with icon_path.open("rb") as icon_file:
                        img = PILImage.open(icon_file)
                        photo_image = PILImageTk.PhotoImage(img)
                        self.iconphoto(False, photo_image)
                except Exception:
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
        self.menubar = menubar

        # File
        filemenu = tk.Menu(menubar, tearoff=0)
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

        # Edit
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo", command=self._undo, accelerator=self._accel("Z"))
        editmenu.add_command(label="Redo", command=self._redo, accelerator=f"Shift+{self._accel('Z')}")
        menubar.add_cascade(label="Edit", menu=editmenu)

        # Run
        runmenu = tk.Menu(menubar, tearoff=0)
        self.run_menu = runmenu
        runmenu.add_command(
            label="Run Model",
            command=self.run_model,
            accelerator=self._accel("R"),
        )
        solver_menu = tk.Menu(runmenu, tearoff=0)
        solver_menu.add_radiobutton(label="Gurobi", variable=self.solver, value="gurobi")
        solver_menu.add_radiobutton(label="Scipy (HiGHS)", variable=self.solver, value="scipy")
        runmenu.add_cascade(label="Solver", menu=solver_menu)
        menubar.add_cascade(label="Run", menu=runmenu)

        # GenAI (populated after discovery)
        self.genai_menu = tk.Menu(menubar, tearoff=0)
        self.genai_menu.add_command(label="Loading models...", state="disabled")
        menubar.add_cascade(label="GenAI", menu=self.genai_menu)

        # Settings
        settings_menu = tk.Menu(menubar, tearoff=0)

        # Font Size
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

        # Theme
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

        # Help
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="User Guide",
            command=lambda: self._open_url("https://github.com/gwr3n/rhetor/blob/main/docs/PyOPL%20user%20guide.md"),
        )
        help_menu.add_command(
            label="Examples",
            command=lambda: self._open_url("https://github.com/gwr3n/rhetor/blob/main/docs/PyOPL%20examples%20overview.md"),
        )
        help_menu.add_command(
            label="GitHub",
            command=lambda: self._open_url("https://gwr3n.github.io/rhetor/"),
        )
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _find_run_stop_menu_index(self) -> Optional[int]:
        """
        Find the index of the Run/Stop menu entry by its label.
        This avoids relying on a fixed numeric index.
        """
        if not hasattr(self, "run_menu"):
            return None
        try:
            last = self.run_menu.index("end")
            if last is None:
                return None
            for i in range(int(last) + 1):
                try:
                    label = self.run_menu.entrycget(i, "label")
                except Exception:
                    continue
                if label in ("Run Model", "Stop Model"):
                    return i
        except Exception:
            return None
        return None

    def _set_run_menu_running(self, running: bool) -> None:
        """Toggle Run/Stop menu item."""
        idx = self._find_run_stop_menu_index()
        if idx is None:
            return

        if running:
            # Requirement: Stop has no shortcut (clear displayed accelerator)
            self.run_menu.entryconfigure(
                idx,
                label="Stop Model",
                command=self.stop_model,
                accelerator="",
            )
        else:
            self.run_menu.entryconfigure(
                idx,
                label="Run Model",
                command=self.run_model,
                accelerator=self._accel("R"),
            )

    def _accel(self, key: str) -> str:
        """Return platform-aware accelerator label."""
        return f"{'Cmd' if sys.platform == 'darwin' else 'Ctrl'}+{key}"

    def new_model(self) -> None:
        """Clear editors, reset file paths, and prepare for a new model."""
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

        # Clear output with a message
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "New model created. Ready.\n")
        self.output_text.config(state="disabled")

    def _setup_panes(self) -> None:
        """Set up the main paned window with editors on top and output below."""
        editor_output_paned = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            sashrelief=tk.FLAT,
            bd=0,  # Remove bevels
            bg="#e9ecef",  # Will be overridden by _apply_theme_colors
            sashwidth=6,  # Thin, modern sash
            showhandle=False,
            relief=tk.FLAT,
        )
        editor_output_paned.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        # Keep a reference for theme updates
        self.editor_output_paned = editor_output_paned

        self._setup_editors(editor_output_paned)
        self._setup_output(editor_output_paned)

    def _setup_editors(self, parent: tk.PanedWindow) -> None:
        """Create model and data editor frames inside a Notebook."""
        editor_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=0)
        parent.add(editor_frame, stretch="always")

        # Notebook
        self.editor_notebook = ttk.Notebook(editor_frame)
        self.editor_notebook.pack(fill=tk.BOTH, expand=1)

        # Model editor
        # Use a styled frame so we can control its background color
        self.model_frame = ttk.Frame(self.editor_notebook, style="Editor.TFrame")
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

        # Event handlers (typed to avoid lambda inference issues)
        def _on_model_changed(event: tk.Event) -> None:
            self._on_text_change(self.model_text, False)

        def _on_data_changed(event: tk.Event) -> None:
            self._on_text_change(self.data_text, True)

        self.model_text.bind("<KeyRelease>", _on_model_changed)
        self.model_text.bind("<ButtonRelease-1>", _on_model_changed)
        self.model_text.bind("<Control-Key-a>", self._select_all_model)

        # Data editor
        # Use the same styled frame for consistent background
        self.data_frame = ttk.Frame(self.editor_notebook, style="Editor.TFrame")
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

        # Undo/Redo on widgets (overrides Text default bindings)
        self.model_text.bind("<Control-z>", self._undo_shortcut)
        self.model_text.bind("<Control-Shift-Z>", self._redo_shortcut)
        self.model_text.bind("<Control-Shift-z>", self._redo_shortcut)
        self.data_text.bind("<Control-z>", self._undo_shortcut)
        self.data_text.bind("<Control-Shift-Z>", self._redo_shortcut)
        self.data_text.bind("<Control-Shift-z>", self._redo_shortcut)
        if sys.platform == "darwin":
            self.model_text.bind("<Command-z>", self._undo_shortcut)
            self.model_text.bind("<Command-Shift-Z>", self._redo_shortcut)
            self.data_text.bind("<Command-z>", self._undo_shortcut)
            self.data_text.bind("<Command-Shift-Z>", self._redo_shortcut)

        # Tabs
        self.editor_notebook.add(self.model_frame, text="Model")
        self.editor_notebook.add(self.data_frame, text="Data")

        # Update caret/highlighting when switching tabs
        self.editor_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.editor_frame = editor_frame

    def _setup_output(self, parent: tk.PanedWindow) -> None:
        """Create the output panel with a request history list on the right."""
        output_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=0)

        # Split Output (left) and Requests list (right)
        container = ttk.Frame(output_frame)
        container.pack(fill=tk.BOTH, expand=1, padx=5, pady=(0, 5))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.rowconfigure(0, weight=1)

        # Output text
        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew")
        self.output_text = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            height=12,
            font=(self.editor_font_family, self.current_font_size - 1),
            state="disabled",
            bg="#f8f9fa",
            fg="#212529",
            relief=tk.FLAT,
            bd=0,
        )
        self.output_text.pack(fill=tk.BOTH, expand=1)

        # Requests list
        right = ttk.Frame(container, width=220)
        right.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        right.rowconfigure(0, weight=1)

        self.request_listbox = tk.Listbox(
            right,
            exportselection=False,
            height=12,
        )
        # Use classic tk.Scrollbar to match ScrolledText
        request_scroll = tk.Scrollbar(right, orient=tk.VERTICAL, command=self.request_listbox.yview)
        self.request_listbox.configure(yscrollcommand=request_scroll.set)
        self.request_listbox.grid(row=0, column=0, sticky="nsew")
        request_scroll.grid(row=0, column=1, sticky="ns")

        # Selection handler to show previous output
        self.request_listbox.bind("<<ListboxSelect>>", self._on_request_select)

        # Context menu for deleting sessions
        self.request_context_menu = tk.Menu(self, tearoff=0)
        self.request_context_menu.add_command(label="Delete Session", command=self._delete_selected_request)

        # Right-click bindings (support macOS Ctrl+Click)
        self.request_listbox.bind("<Button-3>", self._on_request_right_click)
        if sys.platform == "darwin":
            self.request_listbox.bind("<Button-2>", self._on_request_right_click)
            self.request_listbox.bind("<Control-Button-1>", self._on_request_right_click)

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
            padding=(8, 0, 0, 2),
            relief=tk.FLAT,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_tag_configs(self) -> None:
        """Configure syntax highlighting tags for editors."""
        for token, color in TOKEN_COLORS.items():
            self.model_text.tag_configure(token, foreground=color)
            self.data_text.tag_configure(token, foreground=color)
        # Error tag
        self.model_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        self.data_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        # Comments
        self.model_text.tag_configure("COMMENT", font=("Consolas", self.current_font_size, "italic"))

    # --- Event Handlers and Core Logic ---
    def _on_request_right_click(self, event: Optional[tk.Event]) -> None:
        """Show context menu at right-click position and select the item."""
        if event is None:
            return
        try:
            index = self.request_listbox.nearest(event.y)
            self.request_listbox.selection_clear(0, tk.END)
            if index >= 0:
                self.request_listbox.selection_set(index)
                self.request_listbox.activate(index)
                self._last_request_popup_index = index
            else:
                self._last_request_popup_index = None
            self.request_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self.request_context_menu.grab_release()
            except Exception:
                pass

    def _delete_selected_request(self) -> None:
        """Delete the currently selected output session."""
        try:
            sel = self.request_listbox.curselection()
            index = None
            if sel:
                index = int(sel[0])
            else:
                index = getattr(self, "_last_request_popup_index", None)
            if index is None or index < 0 or index >= len(self._output_session_ids):
                return

            sid = self._output_session_ids[index]
            if not messagebox.askyesno("Delete Session", "Delete the selected session?"):
                return

            # Remove data
            self._output_session_ids.pop(index)
            self._output_sessions.pop(sid, None)
            self._output_session_display.pop(sid, None)

            # Update pointers
            if self._current_output_session_id == sid:
                self._current_output_session_id = None
            if self._viewing_output_session_id == sid:
                self._viewing_output_session_id = None

            # Update UI list
            try:
                self.request_listbox.delete(index)
            except Exception:
                pass

            # Select next available session and show it
            count = self.request_listbox.size()
            if count > 0:
                new_index = min(index, count - 1)
                self.request_listbox.selection_clear(0, tk.END)
                self.request_listbox.selection_set(new_index)
                self.request_listbox.activate(new_index)
                new_sid = self._output_session_ids[new_index]
                self._viewing_output_session_id = new_sid
                self._show_output_session(new_sid)
            else:
                self._viewing_output_session_id = None
                if hasattr(self, "output_text") and self.output_text.winfo_exists():
                    self.output_text.config(state="normal")
                    self.output_text.delete("1.0", tk.END)
                    self.output_text.config(state="disabled")

            self.status_var.set("Session deleted.")
        except Exception:
            pass

    def _on_text_change(self, text_widget: tk.Text, is_data: bool = False) -> None:
        """Update caret position and syntax highlighting on text change."""
        # Keep caret responsive immediately
        self._update_caret_position(text_widget)
        # Debounce highlighting so we don't re-lex/parse on every keystroke
        self._schedule_highlight(text_widget, is_data)

    def _cancel_scheduled_highlight(self, text_widget: tk.Text, kind: str) -> None:
        key = (id(text_widget), kind)
        after_id = self._highlight_after_ids.pop(key, None)
        if after_id:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass

    def _schedule_highlight(self, text_widget: tk.Text, is_data: bool) -> None:
        """Debounce highlight work to keep typing responsive."""
        if getattr(self, "_shutting_down", False):
            return

        # Cancel any pending runs for this widget
        self._cancel_scheduled_highlight(text_widget, "fast")
        self._cancel_scheduled_highlight(text_widget, "validate")

        if is_data:
            # Data: fast regex highlight shortly after typing, then expensive validate after idle
            self._highlight_after_ids[(id(text_widget), "fast")] = self.after(
                self._highlight_debounce_ms,
                self._run_scheduled_highlight,
                text_widget,
                True,
                False,  # validate=False => regex-only path for .dat
            )
            self._highlight_after_ids[(id(text_widget), "validate")] = self.after(
                self._highlight_validate_idle_ms,
                self._run_scheduled_highlight,
                text_widget,
                True,
                True,  # validate=True => lexer+parser
            )
        else:
            # Model: skip any fast pass; only lex/parse after user pauses
            self._highlight_after_ids[(id(text_widget), "validate")] = self.after(
                self._highlight_validate_idle_ms,
                self._run_scheduled_highlight,
                text_widget,
                False,
                True,
            )

    def _run_scheduled_highlight(self, text_widget: tk.Text, is_data: bool, validate: bool) -> None:
        """Run highlight if widget still exists."""
        if getattr(self, "_shutting_down", False):
            return
        try:
            if not text_widget.winfo_exists():
                return
        except Exception:
            return

        self.highlight(text_widget, is_data=is_data, validate=validate)

        # Only refresh the status bar if this is the currently selected editor
        try:
            idx = self.editor_notebook.index(self.editor_notebook.select())
            selected = self.model_text if idx == 0 else self.data_text
            if text_widget is selected:
                self._update_caret_position(text_widget)
        except Exception:
            pass

    def on_tree_select(self, event: Optional[tk.Event]) -> None:
        """Compatibility handler: focus Model editor (no file tree in this UI)."""
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
        """Open a model file into the model editor."""
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
        """Open a data file into the data editor."""
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
        # "end-1c" means "end minus 1 character" (the implicit newline)
        content = self.model_text.get("1.0", "end-1c")
        with open(self.model_file, "w") as f:
            f.write(content)
        # Update tab title
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
        # Update tab title
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
    def highlight(self, text_widget: tk.Text, is_data: bool = False, validate: bool = True) -> None:
        """Apply syntax highlighting to the given text widget."""
        if (not is_data) and (not validate):
            return

        # Remove previous tags
        for previous_tag in TOKEN_COLORS.keys():
            text_widget.tag_remove(previous_tag, "1.0", tk.END)
        text_widget.tag_remove("ERROR", "1.0", tk.END)

        code = text_widget.get("1.0", tk.END)

        # Pre-calculate line start offsets for O(1) or O(log N) lookups
        # This avoids the O(N^2) behavior of calling text[:pos].count('\n') for every token
        line_starts = [0] + [i + 1 for i, char in enumerate(code) if char == "\n"]

        import bisect

        def fast_index(pos: int) -> str:
            if pos < 0:
                pos = 0
            if pos > len(code):
                pos = len(code)
            # Find the line number (bisect_right returns insertion point, so index is i-1)
            line_idx = bisect.bisect_right(line_starts, pos) - 1
            line_start = line_starts[line_idx]
            col = pos - line_start
            return f"{line_idx + 1}.{col}"

        # Store the most recent error for status bar display (per widget)
        self._last_syntax_error_by_widget[id(text_widget)] = None
        # (Keep legacy attribute too, in case other code reads it.)
        self._last_syntax_error = None

        if not is_data:
            lexer = OPLLexer()
            parser = OPLParser()
            tokens = []
            lexer_error = None
            try:
                tokens = list(lexer.tokenize(code))
            except Exception as e:
                lexer_error = e
                lineno = getattr(e, "lineno", 1)
                if not isinstance(lineno, int) or lineno is None:
                    lineno = 1
                error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                msg = f"Lexer Error on line {lineno}: {error_message}"
                self._last_syntax_error_by_widget[id(text_widget)] = msg
                self._last_syntax_error = msg
            if not lexer_error:
                try:
                    parser.parse(iter(tokens))
                except Exception as e:
                    lineno = getattr(e, "lineno", 1)
                    if not isinstance(lineno, int) or lineno is None:
                        lineno = 1
                    error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                    text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                    msg = f"Parser Error on line {lineno}: {error_message}"
                    self._last_syntax_error_by_widget[id(text_widget)] = msg
                    self._last_syntax_error = msg

            # Batch tag application to reduce Tcl overhead
            tag_ranges: dict[str, list[str]] = {}
            for token in tokens:
                # Use the fast lookup instead of self._index_from_pos
                start_idx = fast_index(token.index)
                end_idx = fast_index(token.index + len(str(token.value)))
                tag = token.type if token.type in TOKEN_COLORS else None
                if tag:
                    if tag not in tag_ranges:
                        tag_ranges[tag] = []
                    tag_ranges[tag].extend([start_idx, end_idx])

            for tag, ranges in tag_ranges.items():
                # Apply in chunks to avoid Tcl argument limits
                for i in range(0, len(ranges), 2000):
                    text_widget.tag_add(tag, *ranges[i : i + 2000])

        else:
            import re

            if validate:
                lexer = OPLDataLexer()
                parser = OPLDataParser()
                tokens = []
                lexer_error = None
                try:
                    tokens = list(lexer.tokenize(code))
                except Exception as e:
                    lexer_error = e
                    lineno = getattr(e, "lineno", 1)
                    if not isinstance(lineno, int) or lineno is None:
                        lineno = 1
                    error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                    text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                    msg = f"Lexer Error on line {lineno}: {error_message}"
                    self._last_syntax_error_by_widget[id(text_widget)] = msg
                    self._last_syntax_error = msg
                if not lexer_error:
                    try:
                        parser.parse(iter(tokens), lexer=lexer)
                    except Exception as e:
                        lineno = getattr(e, "lineno", 1)
                        if not isinstance(lineno, int) or lineno is None:
                            lineno = 1
                        error_message = str(e).splitlines()[0] if str(e) else "Unknown syntax error"
                        text_widget.tag_add("ERROR", f"{lineno}.0", f"{lineno}.end")
                        msg = f"Parser Error on line {lineno}: {error_message}"
                        self._last_syntax_error_by_widget[id(text_widget)] = msg
                        self._last_syntax_error = msg

            # Cheap regex highlighting for .dat
            # Batch keyword highlighting
            kw_ranges: dict[str, list[str]] = {"PARAM": [], "SET": [], "BOOLEAN": []}
            for kw in ["param", "set", "true", "false"]:
                tag = "PARAM" if kw == "param" else "SET" if kw == "set" else "BOOLEAN"
                for m in re.finditer(r"\b" + kw + r"\b", code):
                    start = fast_index(m.start())
                    end = fast_index(m.end())
                    kw_ranges[tag].extend([start, end])

            for tag, ranges in kw_ranges.items():
                if ranges:
                    for i in range(0, len(ranges), 2000):
                        text_widget.tag_add(tag, *ranges[i : i + 2000])

            # Batch number highlighting
            number_ranges = []
            for m in re.finditer(r"\d+(\.\d+)?", code):
                start = fast_index(m.start())
                end = fast_index(m.end())
                number_ranges.extend([start, end])

            if number_ranges:
                for i in range(0, len(number_ranges), 2000):
                    text_widget.tag_add("NUMBER", *number_ranges[i : i + 2000])

    def _index_from_pos(self, text: str, pos: int) -> str:
        """
        Convert a character offset in a string to a Tk Text index (line.char).
        Kept for compatibility, though highlight() now uses an internal fast version.
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
        Change font size of editors and output console.
        """
        self.current_font_size = size
        # Sync menu state
        try:
            self.font_size_var.set(size)
        except Exception:
            pass
        editor_font = (self.editor_font_family, size)
        output_font = (self.editor_font_family, size - 1 if size > 10 else size)

        self.model_text.config(font=editor_font)
        self.data_text.config(font=editor_font)
        self.output_text.config(font=output_font)

        # Adjust comment tag to match new size
        self.model_text.tag_configure("COMMENT", font=(self.editor_font_family, size, "italic"))

        # Update caret position after size change
        self._update_caret_position(self.model_text)

        # Persist settings
        self._save_settings()

    # --- Status Bar ---
    def _update_caret_position(self, text_widget: tk.Text) -> None:
        """
        Update status bar with current caret position. If a syntax error is present,
        display its line alongside the caret position.
        """
        if text_widget.winfo_exists():
            try:
                index = text_widget.index(tk.INSERT)
                index_str = str(index)
                if "." in index_str:
                    caret_line, caret_col = map(int, index_str.split("."))
                else:
                    caret_line, caret_col = 1, 0

                # Collect all error lines
                error_lines = []
                if text_widget.tag_ranges("ERROR"):
                    tag_ranges = list(text_widget.tag_ranges("ERROR"))
                    for tag_start, tag_end in zip(tag_ranges[0::2], tag_ranges[1::2]):
                        tag_start_line = int(str(tag_start).split(".")[0])
                        tag_end_line = int(str(tag_end).split(".")[0])
                        for err_line in range(tag_start_line, tag_end_line + 1):
                            error_lines.append(err_line)

                # Use per-widget last error message if available
                last_error = None
                try:
                    last_error = self._last_syntax_error_by_widget.get(id(text_widget))
                except Exception:
                    last_error = getattr(self, "_last_syntax_error", None)

                error_msg = None
                if error_lines and caret_line in error_lines:
                    if last_error and f"line {caret_line}" in last_error:
                        error_msg = last_error
                    else:
                        error_msg = f"Syntax Error on line {caret_line}"
                elif error_lines:
                    first_err_line = error_lines[0]
                    if last_error and f"line {first_err_line}" in last_error:
                        error_msg = last_error
                    else:
                        error_msg = f"Syntax Error on line {first_err_line}"

                caret_msg = f"Ln {caret_line}, Col {caret_col}"
                if error_msg:
                    self.status_var.set(f"{error_msg} | {caret_msg}")
                else:
                    self.status_var.set(f"Syntax OK | {caret_msg}")

            except tk.TclError:
                self.status_var.set("Ready")
            except Exception as e:
                # ...existing code...
                self.status_var.set(f"Error updating status: {e}")
        else:
            self.status_var.set("Ready")

    # --- Editor Shortcuts ---
    def _select_all_model(self, event: Optional[tk.Event] = None) -> str:
        """Select all text in the model editor."""
        self.model_text.tag_add("sel", "1.0", tk.END)
        self.model_text.mark_set(tk.INSERT, "1.0")
        self.model_text.see(tk.INSERT)
        return "break"

    def _select_all_data(self, event: Optional[tk.Event] = None) -> str:
        """Select all text in the data editor."""
        self.data_text.tag_add("sel", "1.0", tk.END)
        self.data_text.mark_set(tk.INSERT, "1.0")
        self.data_text.see(tk.INSERT)
        return "break"

    def _get_active_text_widget(self) -> tk.Text:
        """Return the active editor, falling back to the selected tab."""
        try:
            w = self.focus_get()
        except Exception:
            w = None
        if w is self.model_text or w is self.data_text:
            return w  # type: ignore[return-value]
        idx = self.editor_notebook.index(self.editor_notebook.select())
        return self.model_text if idx == 0 else self.data_text

    def _undo(self) -> None:
        """Undo in the active editor."""
        tw = self._get_active_text_widget()
        try:
            tw.edit_undo()
        except tk.TclError:
            pass
        self._on_text_change(tw, is_data=(tw is self.data_text))

    def _redo(self) -> None:
        """Redo in the active editor."""
        tw = self._get_active_text_widget()
        try:
            tw.edit_redo()
        except tk.TclError:
            pass
        self._on_text_change(tw, is_data=(tw is self.data_text))

    def _start_run_timer(self, base_msg: str = "Running model...") -> None:
        """Start updating the status bar with elapsed solve time (every second)."""
        self._stop_run_timer()
        self._run_status_base = base_msg
        try:
            self._run_started_at = self.tk.call("clock", "seconds")  # integer seconds
        except Exception:
            import time

            self._run_started_at = time.time()
        self._tick_run_timer()

    def _stop_run_timer(self) -> None:
        """Stop elapsed-time updates."""
        if self._run_timer_after_id:
            try:
                self.after_cancel(self._run_timer_after_id)
            except Exception:
                pass
        self._run_timer_after_id = None
        self._run_started_at = None

    def _tick_run_timer(self) -> None:
        """Update status bar with elapsed time so far; reschedule if still running."""
        p = self._solver_process
        if not (p and p.is_alive()):
            self._stop_run_timer()
            return

        # Compute elapsed
        try:
            now = float(self.tk.call("clock", "seconds"))
        except Exception:
            import time

            now = time.time()

        started = self._run_started_at or now
        elapsed = max(0.0, now - started)

        # Format as HH:MM:SS
        total = int(elapsed)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        t = f"{hh:02d}:{mm:02d}:{ss:02d}"

        self.status_var.set(f"{self._run_status_base} (elapsed {t})")

        # Reschedule
        self._run_timer_after_id = self.after(1000, self._tick_run_timer)

    # --- Model Execution ---
    def run_model(self) -> None:
        """Run the model using current editor contents, checking data file presence and validity."""
        if self._solver_process and self._solver_process.is_alive():
            messagebox.showinfo("Run Model", "Model is already running.")
            return

        import re

        model_code = self.model_text.get(1.0, tk.END).rstrip("\n")
        data_code = self.data_text.get(1.0, tk.END).rstrip("\n")

        # Start a new output session
        self._clear_output("Run: Running model...")

        self.status_var.set("Running model...")
        solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"

        # Data file checks
        data_vars = set()
        # Variables like: int nbSets = ...;
        for m in re.finditer(
            r"\b(?:int|float|boolean|set)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\.\.\.",
            model_code,
        ):
            data_vars.add(m.group(1))
        # Arrays: float cost[Sets] = ...;
        for m in re.finditer(
            r"\b(?:int|float|boolean|set)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[.*?\]\s*=\s*\.\.\.",
            model_code,
        ):
            data_vars.add(m.group(1))

        # If model references data but data is missing/empty
        if data_vars and (not self.data_file or not os.path.exists(self.data_file) or not data_code.strip()):
            self.status_var.set("Error: Data file missing or empty for required model parameters.")
            self.output_text.config(state="normal")
            self.output_text.insert(
                tk.END,
                "\nError: Data file missing or empty for required model parameters.\n",
            )
            self.output_text.config(state="disabled")
            return

        # Parse the data file, if present
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

        # Check that all required data variables are present
        missing_vars = []
        for var in data_vars:
            if not re.search(r"\b" + re.escape(var) + r"\s*(=|\[)", data_code):
                missing_vars.append(var)
        if missing_vars:
            self.status_var.set(f"Error: Data missing for: {', '.join(missing_vars)}")
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, f"\nError: Data missing for: {', '.join(missing_vars)}\n")
            self.output_text.config(state="disabled")
            return

        # Save temp files if not saved
        model_file = self.model_file or "temp_model.mod"
        data_file = self.data_file or "temp_data.dat"
        try:
            with open(model_file, "w") as f:
                f.write(model_code)
            with open(data_file, "w") as f:
                f.write(data_code)
        except Exception as e:
            self.status_var.set(f"Error saving temp files: {e}")
            return

        self._current_solver_choice = solver_choice
        self._solver_queue = multiprocessing.Queue()
        self._solver_process = multiprocessing.Process(
            target=_solve_wrapper,
            args=(model_file, data_file, solver_choice, self._solver_queue),
        )
        self._solver_process.start()

        self._set_run_menu_running(True)

        # Start elapsed-time status updates (every second)
        self._start_run_timer("Running model...")

        self.after(100, self._poll_solver)

    def stop_model(self) -> None:
        p = self._solver_process
        q = self._solver_queue

        if p and p.is_alive():
            try:
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive() and hasattr(p, "kill"):
                    p.kill()  # py3.7+ on Unix
                    p.join(timeout=1.0)
            except Exception:
                pass

        self._solver_process = None
        self._solver_queue = None

        # Stop timer updates
        self._stop_run_timer()

        # Best-effort cleanup of queue resources
        try:
            if q is not None:
                q.close()
                # Only join if we expect the process finished cleanly.
                # Since we just killed it, we should skip join_thread or cancel it.
                q.cancel_join_thread()
        except Exception:
            pass

        self._append_output("\nExecution stopped by user.\n")
        self.status_var.set("Execution stopped.")
        self._set_run_menu_running(False)

    def _poll_solver(self) -> None:
        # _poll_solver is scheduled via `after()`, so it can run after `stop_model()`
        # has already nulled these out.
        p = self._solver_process
        q = self._solver_queue
        if not p or not q:
            return

        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            if p.is_alive():
                self.after(100, self._poll_solver)
                return

            # Process ended but no message
            self._set_run_menu_running(False)
            self._append_output("\nError: Solver process terminated unexpectedly.\n")

            # Stop timer updates
            self._stop_run_timer()

            self.status_var.set("Error: Solver process terminated.")
            self._solver_process = None
            self._solver_queue = None
            return

        # Got a message => process should be done
        try:
            p.join(timeout=0.1)
        except Exception:
            pass

        self._solver_process = None
        self._solver_queue = None
        self._set_run_menu_running(False)

        # Stop timer updates
        self._stop_run_timer()

        if kind == "success":
            self._display_solve_results(payload)
        else:
            self._append_output(f"\nError:\n{payload}\n")
            self.status_var.set("Error running model")

    def _display_solve_results(self, results: dict) -> None:
        """Format and display solver results in the output pane."""
        solver_choice = getattr(self, "_current_solver_choice", "gurobi")
        buf = []
        buf.append(f"\nSolver: {solver_choice}\n")
        buf.append("\nStatus: " + results.get("status", "UNKNOWN") + "\n")
        if "objective_value" in results and results["objective_value"] is not None:
            buf.append(f"Objective: {results['objective_value']}\n")
        if "solution" in results and results["solution"]:
            buf.append("Solution:\n")
            for k, v in results["solution"].items():
                buf.append(f"  {k}: {v}\n")
        if "stats" in results and results["stats"]:
            buf.append("\nSolver Statistics (from 'stats' field):\n")
            if isinstance(results["stats"], dict):
                for stat_key, stat_value in results["stats"].items():
                    buf.append(f"  {stat_key}: {stat_value}\n")
            else:
                buf.append(str(results["stats"]) + "\n")
        else:
            buf.append("\nNo detailed solver statistics available from pyopl.solve.\n")
        if "message" in results:
            buf.append(f"Message: {results['message']}\n")

        for s in buf:
            self._append_output(s)
        msg = results.get("message") or results.get("status", "Done")
        self.status_var.set(msg)

    def export_model(self) -> None:
        """Export the current model as a standalone Python file using the selected solver's code generator."""
        try:
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
                messagebox.showerror("Export model", f"Failed to parse model: {type(e).__name__}")
                return

            # Parse data -> data_dict (if any)
            data_dict = {}
            if data_code.strip():
                try:
                    d_lexer = OPLDataLexer()
                    d_parser = OPLDataParser()
                    d_tokens = list(d_lexer.tokenize(data_code))
                    parsed = d_parser.parse(iter(d_tokens), lexer=d_lexer)
                    if isinstance(parsed, dict):
                        data_dict = parsed
                except Exception as e:
                    messagebox.showerror("Export model", f"Failed to parse data: {type(e).__name__}")
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
                # Strip the last line if present
                lines = generated_code.rstrip("\n").split("\n")
                if lines:
                    generated_code = "\n".join(lines[:-1])
            except Exception as e:
                messagebox.showerror("Export model", f"Code generation failed: {type(e).__name__}")
                return

            # Destination file
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

            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(generated_code)

            self.status_var.set(f"Exported model to {dest_path}")
        except Exception as e:
            messagebox.showerror("Export model", f"Unexpected error: {type(e).__name__}")
            self.status_var.set(f"Export failed: {e}")

    # --- GenAI actions ---
    def _clear_output(self, header: str = "") -> None:
        """Start a new output request session and display its header."""
        self._begin_new_output_session(header)

    def _append_output(self, text: str) -> None:
        """Append text to the current output session and update the Output panel if visible."""
        sid = getattr(self, "_current_output_session_id", None)
        if sid:
            self._output_sessions[sid] = self._output_sessions.get(sid, "") + text
        if sid and getattr(self, "_viewing_output_session_id", None) == sid and self.output_text.winfo_exists():
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")

    # Output sessions (history)
    def _begin_new_output_session(self, header: str = "") -> None:
        """Create a new request session, add it to the list, and show it."""
        dt = datetime.now()
        display = dt.strftime("%Y-%m-%d %H:%M:%S")
        session_id = dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        initial = (header + "\n") if header else ""
        self._output_sessions[session_id] = initial
        self._output_session_display[session_id] = display
        self._output_session_ids.insert(0, session_id)

        # Update UI list (most recent at top)
        if hasattr(self, "request_listbox"):
            try:
                self.request_listbox.insert(0, display)
                self.request_listbox.selection_clear(0, tk.END)
                self.request_listbox.selection_set(0)
                self.request_listbox.activate(0)
            except Exception:
                pass

        self._current_output_session_id = session_id
        self._viewing_output_session_id = session_id
        self._show_output_session(session_id)

    def _show_output_session(self, session_id: str) -> None:
        """Display a session's content in the Output panel."""
        content = self._output_sessions.get(session_id, "")
        if hasattr(self, "output_text") and self.output_text.winfo_exists():
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, content)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")

    def _on_request_select(self, event: Optional[tk.Event]) -> None:
        """Handle selection in the Requests list."""
        try:
            sel = self.request_listbox.curselection()
            if not sel:
                return
            index = int(sel[0])
            if index < 0 or index >= len(self._output_session_ids):
                return
            sid = self._output_session_ids[index]
            self._viewing_output_session_id = sid
            self._show_output_session(sid)
        except Exception:
            pass

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
        dlg.bind("<Command-Return>", on_ok)  # macOS

        dlg.wait_window()
        return result["text"]

    def genai_generate(self) -> None:
        """Prompt for a problem description and generate model & data via GenAI."""
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

        # Resolve selected generator module
        gen_module = self._import_selected_genai_module()
        module_logger_name = getattr(gen_module, "__name__", "pyopl.genai.pyopl_generative")

        self.status_var.set(
            f"GenAI: generating with {self.genai_provider} • {self.genai_model} • method={self._label_for_method(self.genai_method_var.get())} ..."
        )
        self._clear_output("GenAI: Generating model and data...")

        # Use the visible request timestamp for filenames
        sid = self._current_output_session_id or ""
        display_ts = self._output_session_display.get(sid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        safe_ts = display_ts.replace(":", "-").replace(" ", "_")

        def run():
            try:
                # Progress hook -> Output panel
                def progress(msg: str) -> None:
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"))

                # Bridge module logger to progress (optional)
                class _ProgressLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        try:
                            text = self.format(record)
                        except Exception:
                            text = record.getMessage()
                        progress(text)

                log = logging.getLogger(module_logger_name)
                handler = None
                old_level = log.level
                if self.verbose_llm_var.get():
                    handler = _ProgressLogHandler()
                    handler.setLevel(logging.DEBUG)
                    log.addHandler(handler)
                    log.setLevel(logging.DEBUG)

                tmp_dir = os.path.join(os.getcwd(), "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                # Unique filenames per request
                base = os.path.join(tmp_dir, f"gen_pyopl_{safe_ts}")
                model_path = base + ".mod"
                data_path = base + ".dat"
                i = 1
                while os.path.exists(model_path) or os.path.exists(data_path):
                    model_path = f"{base}_{i}.mod"
                    data_path = f"{base}_{i}.dat"
                    i += 1

                try:
                    # Dispatch to selected generation method
                    assessment = gen_module.generative_solve(
                        prompt,
                        model_path,
                        data_path,
                        model_name=self.genai_model,
                        llm_provider=self.genai_provider,
                        progress=progress,
                    )
                finally:
                    if handler is not None:
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
                    messagebox.showerror("GenAI Error", type(e).__name__)
                    self._append_output(f"\nGenAI Error: {e}\n")
                    self.status_var.set("GenAI: error")

                self.after(0, on_error, e)

        threading.Thread(target=run, daemon=True).start()

    def genai_feedback(self) -> None:
        """Prompt for a question and request feedback/revisions from GenAI for the current model/data."""
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

        # Ensure we have model/data files; save current buffers if needed
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
            messagebox.showerror("GenAI Error", f"Failed to save current model/data: {type(e).__name__}")
            return

        # Resolve selected generator module
        gen_module = self._import_selected_genai_module()
        module_logger_name = getattr(gen_module, "__name__", "pyopl.genai.pyopl_generative")

        self.status_var.set("GenAI: requesting feedback...")
        self._clear_output("GenAI: Requesting feedback...")

        # Use visible request timestamp for filenames
        sid = self._current_output_session_id or ""
        display_ts = self._output_session_display.get(sid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        safe_ts = display_ts.replace(":", "-").replace(" ", "_")

        # Track if a data file actually existed before this request
        had_data_file = bool(self.data_file and os.path.exists(self.data_file))

        def run():
            try:
                # Progress hook -> Output panel
                def progress(msg: str) -> None:
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"))

                # Bridge module logger to progress (optional)
                class _ProgressLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        try:
                            text = self.format(record)
                        except Exception:
                            text = record.getMessage()
                        progress(text)

                log = logging.getLogger(module_logger_name)
                handler = None
                old_level = log.level
                if self.verbose_llm_var.get():
                    handler = _ProgressLogHandler()
                    handler.setLevel(logging.DEBUG)
                    log.addHandler(handler)
                    log.setLevel(logging.DEBUG)

                try:
                    result = gen_module.generative_feedback(
                        question,
                        model_path,
                        data_path,
                        model_name=self.genai_model,
                        llm_provider=self.genai_provider,
                        progress=progress,
                    )
                finally:
                    if handler is not None:
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
                            "Apply Revisions?",
                            "GenAI returned revised model/data. Apply these revisions to the editors?",
                        )
                    if apply:
                        os.makedirs(tmp_dir, exist_ok=True)

                        # Derive model base name and extension from model_path
                        m_base_name, m_ext = os.path.splitext(os.path.basename(model_path))
                        m_ext = m_ext or ".mod"

                        # If a timestamp suffix already exists, strip it before appending a new one
                        def _strip_ts_suffix(name: str) -> str:
                            import re

                            m = re.match(r"^(.*?)(?:_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_\d+)?$", name)
                            return m.group(1) if m and m.group(1) else name

                        m_base_name = _strip_ts_suffix(m_base_name)

                        model_base = os.path.join(tmp_dir, f"{m_base_name}_{safe_ts}")
                        model_tgt = model_base + m_ext
                        i = 1
                        while os.path.exists(model_tgt):
                            model_tgt = f"{model_base}_{i}{m_ext}"
                            i += 1

                        # Always write a timestamped model (revised or current text)
                        model_content = revised_model if revised_model else self.model_text.get("1.0", tk.END)
                        with open(model_tgt, "w", encoding="utf-8") as f:
                            f.write(model_content)

                        # Data target only if revised_data is present
                        data_tgt = None
                        if revised_data:
                            if had_data_file:
                                d_base_name, d_ext = os.path.splitext(os.path.basename(data_path))
                                d_ext = d_ext or ".dat"
                            else:
                                d_base_name, d_ext = m_base_name, ".dat"

                            # Strip existing timestamp suffix if present
                            d_base_name = _strip_ts_suffix(d_base_name)

                            data_base = os.path.join(tmp_dir, f"{d_base_name}_{safe_ts}")
                            data_tgt = data_base + d_ext
                            j = 1
                            while os.path.exists(data_tgt):
                                data_tgt = f"{data_base}_{j}{d_ext}"
                                j += 1
                            with open(data_tgt, "w", encoding="utf-8") as f:
                                f.write(revised_data)

                        # Update editors with revised content only
                        if revised_model:
                            self.model_text.delete("1.0", tk.END)
                            self.model_text.insert(tk.END, revised_model)
                        if revised_data:
                            self.data_text.delete("1.0", tk.END)
                            self.data_text.insert(tk.END, revised_data)

                        # Point IDE to the new files
                        self.model_file = model_tgt
                        if data_tgt:
                            self.data_file = data_tgt

                        # Update tabs and highlighting
                        self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file)}")
                        if data_tgt:
                            self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file)}")
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

    # --- Theme ---
    def set_theme(self, theme_name: str) -> None:
        """Switch ttkbootstrap theme and reapply widget colors."""
        if theme_name not in ("flatly", "darkly"):
            return
        self.theme_var.set(theme_name)
        try:
            self.style.theme_use(theme_name)
        except Exception:
            self.style = tb.Style(theme=theme_name)
        self._apply_theme_colors()
        # Re-highlight for contrast
        self.highlight(self.model_text, is_data=False)
        self.highlight(self.data_text, is_data=True)
        # Persist settings
        self._save_settings()

    def _apply_theme_colors(self) -> None:
        """Apply text widget colors based on theme."""
        theme = self.theme_var.get()
        if theme == "darkly":
            root_bg = "#212529"
            editor_bg = "#2b3035"
            editor_fg = "#e9ecef"
            caret_fg = "#e9ecef"
            output_bg = "#212529"
            output_fg = "#e9ecef"
            error_fg = "white"
            paned_bg = "#2b3035"
        else:
            root_bg = "#f8f9fa"
            editor_bg = "#ffffff"
            editor_fg = "#212529"
            caret_fg = "#212529"
            output_bg = "#f8f9fa"
            output_fg = "#212529"
            error_fg = "black"
            paned_bg = "#e9ecef"

        # Root background
        try:
            self.configure(bg=root_bg)
        except Exception:
            pass

        # Paned background (and keep it flat)
        if hasattr(self, "editor_output_paned") and self.editor_output_paned.winfo_exists():
            try:
                self.editor_output_paned.config(bg=paned_bg, bd=0, relief=tk.FLAT, sashrelief=tk.FLAT)
            except Exception:
                pass

        # Apply to editors
        if hasattr(self, "model_text"):
            self.model_text.config(bg=editor_bg, fg=editor_fg, insertbackground=caret_fg, relief=tk.FLAT, bd=0)
        if hasattr(self, "data_text"):
            self.data_text.config(bg=editor_bg, fg=editor_fg, insertbackground=caret_fg, relief=tk.FLAT, bd=0)
        if hasattr(self, "output_text"):
            self.output_text.config(bg=output_bg, fg=output_fg, relief=tk.FLAT, bd=0)

        # Ensure the editor frames share the same background as the text area
        try:
            self.style.configure("Editor.TFrame", background=editor_bg)
        except Exception:
            pass

        # Adjust ERROR tag for contrast
        if hasattr(self, "model_text"):
            self.model_text.tag_configure("ERROR", background="#e06c75", foreground=error_fg)
        if hasattr(self, "data_text"):
            self.data_text.tag_configure("ERROR", background="#e06c75", foreground=error_fg)

    # --- Settings ---
    def _init_settings_storage(self) -> None:
        """Initialize settings storage path."""
        try:
            config_dir = Path(user_config_dir(APP_NAME))
            config_dir.mkdir(parents=True, exist_ok=True)
            self._config_path = config_dir / CONFIG_FILENAME
        except Exception:
            # Fallback to current working directory
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
                "verbose-llm-logs": bool(self.verbose_llm_var.get()) if hasattr(self, "verbose_llm_var") else True,
                "genai-selection": (
                    f"{self.genai_provider}|{self.genai_model}"
                    if getattr(self, "genai_provider", None) and getattr(self, "genai_model", None)
                    else ""
                ),
                "genai-method": self.genai_method_var.get() if hasattr(self, "genai_method_var") else "pyopl_generative",
            }
            with open(self._config_path, "w") as f:
                json.dump(payload, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to save settings: {e}")

    def _on_close(self) -> None:
        """Persist settings and close the app."""
        setattr(self, "_shutting_down", True)
        try:
            self.stop_model()  # ensure no stray solver process
        except Exception:
            pass
        self._save_settings()
        try:
            self.destroy()
        finally:
            try:
                self.quit()
            except Exception:
                pass

    # --- Shortcuts ---
    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts."""
        self.bind_all("<Control-s>", self.save_current_buffer)
        self.bind_all("<Control-n>", self._new_model_shortcut)
        self.bind_all("<Control-r>", self._run_model_shortcut)
        self.bind_all("<Control-g>", self._genai_generate_shortcut)
        self.bind_all("<Control-i>", self._genai_feedback_shortcut)

        if sys.platform == "darwin":
            self.bind_all("<Command-s>", self.save_current_buffer)
            self.bind_all("<Command-n>", self._new_model_shortcut)
            self.bind_all("<Command-r>", self._run_model_shortcut)
            self.bind_all("<Command-g>", self._genai_generate_shortcut)
            self.bind_all("<Command-i>", self._genai_feedback_shortcut)

    def _new_model_shortcut(self, event: Optional[tk.Event] = None) -> str:
        """Keyboard shortcut handler for creating a new model."""
        self.new_model()
        return "break"

    def _run_model_shortcut(self, event: Optional[tk.Event] = None) -> str:
        # While running, do not repurpose this shortcut for Stop (Stop has no shortcut).
        if self._solver_process and self._solver_process.is_alive():
            return "break"
        self.run_model()
        return "break"

    def _genai_generate_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self.genai_generate()
        return "break"

    def _genai_feedback_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self.genai_feedback()
        return "break"

    def _undo_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self._undo()
        return "break"

    def _redo_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self._redo()
        return "break"

    def save_current_buffer(self, event: Optional[tk.Event] = None) -> str:
        """Save the current tab (model or data)."""
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

    def save_current_buffer_as(self, event: Optional[tk.Event] = None) -> str:
        """Save-as for the current tab (model or data)."""
        try:
            idx = self.editor_notebook.index(self.editor_notebook.select())
            if idx == 0:
                self.save_model_as()
            else:
                self.save_data_as()
        except Exception:
            pass
        return "break"

    def show_about(self) -> None:
        """About dialog."""
        messagebox.showinfo(
            "About Rhetor",
            "Rhetor\n\n© 2025 Roberto Rossi",
        )

    # --- GenAI model discovery ---
    def _build_genai_model_menus_async(self) -> None:
        """Discover models in a background thread and populate the GenAI menu on completion."""
        if self._genai_loading:
            return
        self._genai_loading = True

        # Placeholder UI
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

            self.after(0, on_done)

        threading.Thread(target=discover, daemon=True).start()

    def _populate_genai_model_menus(self, provider_models: dict[str, list[str]]) -> None:
        """Populate the GenAI menu with provider submenus and radio items per model."""
        if getattr(self, "_shutting_down", False):
            return
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

            # Generation Method submenu
            method_menu = tk.Menu(self.genai_menu, tearoff=0)
            for label, key in self._genai_methods:
                method_menu.add_radiobutton(
                    label=label,
                    variable=self.genai_method_var,
                    value=key,
                    command=self._make_select_genai_method_cmd(key),
                )

            # Actions
            self.genai_menu.add_separator()
            self.genai_menu.add_cascade(label="Method", menu=method_menu)
            self.genai_menu.add_command(
                label="Generate Model & Data...", command=self.genai_generate, accelerator=self._accel("G")
            )
            self.genai_menu.add_command(label="Ask...", command=self.genai_feedback, accelerator=self._accel("I"))

            # Verbose LLM progress logs
            self.genai_menu.add_separator()
            self.genai_menu.add_checkbutton(
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

            # Prefer saved selection if available; otherwise first available
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

    def _make_select_genai_method_cmd(self, key: str) -> Callable[[], None]:
        """Factory for selecting generation method."""

        def _cmd() -> None:
            self._on_select_genai_method(key)

        return _cmd

    def _on_select_genai_method(self, method_key: str) -> None:
        """Update method selection and persist."""
        if getattr(self, "_shutting_down", False):
            return
        try:
            self.genai_method_var.set(method_key)
            self.status_var.set(f"GenAI method: {self._label_for_method(method_key)}")
        except Exception:
            pass
        self._save_settings()

    def _import_selected_genai_module(self):
        """Import the selected generator module, falling back to generative."""
        import importlib

        key = self.genai_method_var.get() or "pyopl_generative"
        try:
            return importlib.import_module(f"pyopl.genai.{key}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Falling back to pyopl_generative due to import error: {e}")
            try:
                self.genai_method_var.set("pyopl_generative")
            except Exception:
                pass
            return importlib.import_module("pyopl.genai.pyopl_generative")

    def _label_for_method(self, key: str) -> str:
        """Return UI label for a method key."""
        for label, k in self._genai_methods:
            if k == key:
                return label
        return "Generhetor"

    def _make_select_model_cmd(self, provider_key: str, model_name: str) -> Callable[[], None]:
        """Factory for selecting a GenAI model."""

        def _cmd() -> None:
            self._on_select_genai_model(provider_key, model_name)

        return _cmd

    def _make_change_font_cmd(self, size: int) -> Callable[[], None]:
        """Factory for changing font size."""

        def _cmd() -> None:
            self._change_font_size(size)

        return _cmd

    def _make_theme_cmd(self, theme: str) -> Callable[[], None]:
        """Factory for changing theme."""

        def _cmd() -> None:
            self.set_theme(theme)

        return _cmd

    def _open_url(self, url: str) -> None:
        """Open a URL in the default browser."""
        try:
            webbrowser.open_new_tab(url)
        except Exception as e:
            messagebox.showerror("Open URL", f"Failed to open URL:\n{url}\n\n{type(e).__name__}")

    def _on_select_genai_model(self, provider_key: str, model_name: str) -> None:
        """Update GenAI model selection and persist."""
        if getattr(self, "_shutting_down", False):
            return
        self.genai_provider = provider_key
        self.genai_model = model_name
        try:
            self.genai_selection_var.set(f"{provider_key}|{model_name}")
            self.status_var.set(f"GenAI selected: {provider_key} • {model_name}")
        except Exception:
            pass
        self._save_settings()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    ide = OPLIDE()
    ide.mainloop()
