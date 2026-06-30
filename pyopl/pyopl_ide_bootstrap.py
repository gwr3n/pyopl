# --- Standard Library Imports ---
import difflib
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import select
import shutil
import sys
import tempfile
import threading

# --- Third-Party Imports ---
import tkinter as tk
import tkinter.font as tkfont
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Literal, Optional, Protocol, cast

import ttkbootstrap as tb
from platformdirs import user_config_dir

# Model discovery (provider-specific)
from .genai.model_discovery import (
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)
from .genai.pyopl_generative import generative_feedback

# --- Local Imports ---
from .linear_problem_highs import export_linear_problem
from .pyopl_core import OPLCompiler, OPLDataLexer, OPLDataParser, OPLLexer, OPLParser
from .scipy_codegen_csc import SciPyCSCCodeGenerator

# Settings storage (same strategy as sample.py)
APP_NAME = "rhetor"
CONFIG_FILENAME = "settings.json"
_SESSION_ARTIFACT_UNSET = object()

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

MAX_HIGHLIGHT_CHARS = 10_000
GENAI_MAX_PDF_PAGES = 5
GENAI_PDF_RENDER_DPI = 180


# GenAI prompt payload shape (kept loose to avoid importing genai modules/types here).
# Either:
#   - str
#   - {"text": str, "images": [{"path": str}, ...]}
_PromptInput = Any


class _QueueTextWriter:
    """File-like stream that forwards solver logs to the IDE process."""

    def __init__(self, q: multiprocessing.Queue) -> None:
        self._queue = q
        self._buffer = ""

    def write(self, text: str) -> int:
        text = str(text)
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._send(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._send(self._buffer)
            self._buffer = ""

    def _send(self, text: str) -> None:
        try:
            self._queue.put(("log", text))
        except Exception:
            pass


class _FdLogRedirector:
    """Redirect process-level stdout/stderr file descriptors to the IDE log queue."""

    def __init__(self, writer: _QueueTextWriter) -> None:
        self._writer = writer
        self._saved_fds: dict[int, int] = {}
        self._pipe_read: Optional[int] = None
        self._pipe_write: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._active = False

    def __enter__(self) -> "_FdLogRedirector":
        if not hasattr(os, "dup"):
            return self
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            pipe_read, pipe_write = os.pipe()
            self._pipe_read = pipe_read
            self._pipe_write = pipe_write
            for fd in (1, 2):
                self._saved_fds[fd] = os.dup(fd)
                os.dup2(pipe_write, fd)
            self._active = True
            self._thread = threading.Thread(target=self._read_pipe, daemon=True)
            self._thread.start()
        except Exception:
            self._restore_fds()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        self._restore_fds()
        if self._pipe_write is not None:
            try:
                os.close(self._pipe_write)
            except Exception:
                pass
            self._pipe_write = None
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None
        if self._pipe_read is not None:
            try:
                os.close(self._pipe_read)
            except Exception:
                pass
            self._pipe_read = None
        self._writer.flush()

    def _restore_fds(self) -> None:
        if not self._saved_fds:
            return
        for fd, saved_fd in list(self._saved_fds.items()):
            try:
                os.dup2(saved_fd, fd)
            except Exception:
                pass
            try:
                os.close(saved_fd)
            except Exception:
                pass
        self._saved_fds.clear()
        self._active = False

    def _read_pipe(self) -> None:
        pipe_read = self._pipe_read
        if pipe_read is None:
            return
        while True:
            try:
                ready, _write_ready, _errors = select.select([pipe_read], [], [], 0.1)
            except Exception:
                return
            if not ready:
                if not self._active:
                    continue
                continue
            try:
                chunk = os.read(pipe_read, 4096)
            except Exception:
                return
            if not chunk:
                return
            try:
                text = chunk.decode(errors="replace")
            except Exception:
                text = str(chunk)
            self._writer.write(text)


def _solve_wrapper(model_file: str, data_file: str, solver_choice: str, q: multiprocessing.Queue) -> None:
    """Wrapper to run solve in a separate process."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_writer = _QueueTextWriter(q)
    try:
        sys.stdout = log_writer
        sys.stderr = log_writer
        try:
            from .pyopl_core import solve  # package import
        except ImportError:
            from pyopl.pyopl_core import solve  # type: ignore

        def _progress(event: dict[str, Any]) -> None:
            try:
                q.put(("progress", event))
            except Exception:
                pass

        with _FdLogRedirector(log_writer):
            results = solve(model_file, data_file, solver=solver_choice, progress_callback=_progress)
        log_writer.flush()
        q.put(("success", results))
    except Exception as e:
        log_writer.flush()
        q.put(("error", f"{e}\n\n{traceback.format_exc()}"))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class _CodeGenerator(Protocol):
    def generate_code(self) -> str: ...


@dataclass
class _ForegroundOperation:
    kind: str
    label: str
    session_id: str
    solver_choice: Optional[str] = None
    model_file: Optional[str] = None
    data_file: Optional[str] = None
    explain_after_solve: bool = False
    cancel_requested: bool = False


class _StatusVarProxy:
    """Adapter that preserves StringVar-like status_var.set(...) writes."""

    def __init__(self, setter: Callable[[str], None], getter: Callable[[], str]) -> None:
        self._setter = setter
        self._getter = getter

    def set(self, value: str) -> None:
        self._setter(value)

    def get(self) -> str:
        return self._getter()


class OPLIDE(tk.Tk):
    """
    Main class for the Rhetor IDE. Handles UI setup, event binding, and core logic.
    """

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        # Whether the IDE was launched with debug/verbose CLI flags
        self.debug = bool(debug)
        self.title("Rhetor")
        self._configure_macos_application_identity("Rhetor")
        self.geometry("1150x700")
        self.model_file: Optional[str] = None
        self.data_file: Optional[str] = None
        self._model_saved_text = ""
        self._data_saved_text = ""
        self.current_font_size = 12
        self.editor_font_family = tkfont.nametofont("TkFixedFont").actual("family")
        self.interface_font_family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.interface_button_font = "TkDefaultFont"
        self.solver = tk.StringVar(value="gurobi")  # 'gurobi' or 'scipy'
        self.theme_var = tk.StringVar(value="flatly")
        self.show_genai_panel_var = tk.BooleanVar(value=True)

        # Solver process
        self._solver_process: Optional[multiprocessing.Process] = None
        self._solver_queue: Optional[multiprocessing.Queue] = None
        self._current_solver_choice: str = "gurobi"
        self._solver_progress_window: Optional[tk.Toplevel] = None
        self._solver_progress_canvas: Optional[tk.Canvas] = None
        self._solver_progress_stats_frame: Optional[ttk.Frame] = None
        self._solver_progress_status_var: Optional[tk.StringVar] = None
        self._solver_progress_stat_vars: dict[str, tk.StringVar] = {}
        self._solver_progress_samples: list[dict[str, Any]] = []
        self._solver_progress_pending_sample: Optional[dict[str, Any]] = None
        self._solver_progress_update_after_id: Optional[str] = None
        self._solver_progress_rolling_seconds = 120.0

        # --- Run timer (status bar elapsed time while solving) ---
        self._run_started_at: Optional[float] = None
        self._run_timer_after_id: Optional[str] = None
        self._run_status_base: str = "Solving model..."
        self._initial_main_pane_ratio_applied = False
        self._initial_genai_panel_width = 300
        self._side_panel_width = 300
        self._genai_panel_visible = True
        self._panel_resize_after_id: Optional[str] = None
        self._genai_diff_preview_window: Optional[tk.Toplevel] = None
        self._genai_diff_preview_notebook: Optional[ttk.Notebook] = None
        self._genai_diff_preview_texts: dict[str, tk.Text] = {}

        # --- Highlight scheduling (prevents UI lag on large files) ---
        self._highlight_debounce_ms = 150  # fast pass while typing
        self._highlight_validate_idle_ms = 800  # expensive lex/parse after idle
        self._highlight_after_ids: dict[tuple[int, str], str] = {}

        # Track last syntax error per editor (prevents cross-editor contamination)
        self._last_syntax_error_by_widget: dict[int, Optional[str]] = {}
        self._last_syntax_error: Optional[str] = None

        # GenAI selection state
        self.genai_selection_var = tk.StringVar(value="")  # format: "provider|model"
        self.genai_provider: Optional[str] = None
        self.genai_model: Optional[str] = None
        self._genai_provider_models: dict[str, list[str]] = {}
        self._genai_loading: bool = False
        self.genai_panel_mode_var = tk.StringVar(value="generate")
        self.genai_attach_output_var = tk.BooleanVar(value=False)
        self.genai_prompt_title_var = tk.StringVar(value="Describe the optimization problem")
        self.genai_submit_label_var = tk.StringVar(value="Generate")
        self.genai_context_var = tk.StringVar(value="No GenAI model selected")
        self.genai_attachment_summary_var = tk.StringVar(value="No visual attachments")
        self.genai_pending_var = tk.StringVar(value="")
        self._genai_attachment_paths: list[str] = []
        self._genai_attachment_display_labels: dict[str, str] = {}
        self._genai_pdf_temp_dir: Optional[str] = None
        self._genai_pending_revisions: Optional[dict[str, Any]] = None

        # Output sessions
        self._output_sessions: dict[str, str] = {}
        self._output_session_ids: list[str] = []
        self._output_session_display: dict[str, str] = {}
        self._output_session_label: dict[str, str] = {}
        self._output_session_timestamp: dict[str, str] = {}
        self._output_session_artifacts: dict[str, dict[str, str]] = {}
        self._current_output_session_id: Optional[str] = None
        self._viewing_output_session_id: Optional[str] = None
        self._active_operation: Optional[_ForegroundOperation] = None

        # Settings
        self._init_settings_storage()
        loaded_settings = self._load_settings()
        desired_theme = None
        try:
            if isinstance(loaded_settings, dict):
                self.current_font_size = int(loaded_settings.get("font-size", self.current_font_size))
                desired_theme = loaded_settings.get("theme")
                saved_solver = loaded_settings.get("solver")
                if saved_solver in ("gurobi", "scipy"):
                    self.solver.set(saved_solver)
                    self._current_solver_choice = saved_solver
        except Exception:
            pass
        # LLM progress logs in Output (off by default unless launched with debug)
        default_verbose = bool(loaded_settings.get("verbose-llm-logs", False))
        if self.debug:
            # If launched with debug, honor saved value but default to True for convenience
            default_verbose = bool(loaded_settings.get("verbose-llm-logs", True))
        self.verbose_llm_var = tk.BooleanVar(value=default_verbose)
        display_solver_progress = loaded_settings.get("display-solver-progress", True)
        if not isinstance(display_solver_progress, bool):
            display_solver_progress = True
        self.display_solver_progress_var = tk.BooleanVar(value=display_solver_progress)
        # Track font size selection for menu state
        self.font_size_var = tk.IntVar(value=self.current_font_size)

        # GenAI method selection (persisted)
        self._genai_methods: list[tuple[str, str]] = [
            ("SyntAGM", "pyopl_generative"),
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

        # Load previous IDE session if present in current working directory
        try:
            self._load_session()
        except Exception:
            pass
        self._mark_editor_baselines_saved()

        # Initial status update
        self._update_caret_position(self.model_text)

        # Global shortcuts
        self._bind_shortcuts()

        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._register_macos_quit_handler()
        self.after(0, self._stabilize_initial_side_panel_width)
        self.bind("<Configure>", self._on_window_resize, add="+")

    def _register_macos_quit_handler(self) -> None:
        """Route macOS app-menu Quit through the IDE close handler."""
        if sys.platform != "darwin":
            return
        try:
            self.createcommand("::pyopl_macos_quit", self._on_close)
            self.tk.call("proc", "::tk::mac::Quit", "", "::pyopl_macos_quit")
        except Exception:
            pass

    def _configure_macos_application_identity(self, app_name: str) -> None:
        """Best-effort macOS app naming so the menu bar does not keep the generic Python label."""
        if sys.platform != "darwin":
            return
        try:
            self.tk.call("tk", "appname", app_name)
        except Exception:
            pass
        try:
            import ctypes

            objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")

            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_msgSend.restype = ctypes.c_void_p

            ns_process_info = objc.objc_getClass(b"NSProcessInfo")
            ns_string = objc.objc_getClass(b"NSString")
            sel_process_info = objc.sel_registerName(b"processInfo")
            sel_set_process_name = objc.sel_registerName(b"setProcessName:")
            sel_string_with_utf8 = objc.sel_registerName(b"stringWithUTF8String:")

            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            process_info = objc.objc_msgSend(ns_process_info, sel_process_info)
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            ns_app_name = objc.objc_msgSend(ns_string, sel_string_with_utf8, app_name.encode("utf-8"))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            objc.objc_msgSend(process_info, sel_set_process_name, ns_app_name)

            ns_bundle = objc.objc_getClass(b"NSBundle")
            sel_main_bundle = objc.sel_registerName(b"mainBundle")
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            main_bundle = objc.objc_msgSend(ns_bundle, sel_main_bundle)
            if main_bundle:
                sel_info_dictionary = objc.sel_registerName(b"infoDictionary")
                sel_set_object = objc.sel_registerName(b"setObject:forKey:")
                info_dict = objc.objc_msgSend(main_bundle, sel_info_dictionary)
                for bundle_key in (b"CFBundleName", b"CFBundleDisplayName"):
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
                    ns_key = objc.objc_msgSend(ns_string, sel_string_with_utf8, bundle_key)
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                    objc.objc_msgSend(info_dict, sel_set_object, ns_app_name, ns_key)
        except Exception:
            pass

    def _apply_macos_theme_appearance(self, theme_name: str) -> None:
        """Best-effort sync of the native macOS window chrome with the active app theme."""
        if sys.platform != "darwin":
            return
        try:
            import ctypes

            objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_msgSend.restype = ctypes.c_void_p

            ns_app_class = objc.objc_getClass(b"NSApplication")
            ns_appearance_class = objc.objc_getClass(b"NSAppearance")
            ns_string_class = objc.objc_getClass(b"NSString")
            if not ns_app_class or not ns_appearance_class or not ns_string_class:
                return

            sel_shared_application = objc.sel_registerName(b"sharedApplication")
            sel_set_appearance = objc.sel_registerName(b"setAppearance:")
            sel_string_with_utf8 = objc.sel_registerName(b"stringWithUTF8String:")
            sel_appearance_named = objc.sel_registerName(b"appearanceNamed:")

            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            ns_app = objc.objc_msgSend(ns_app_class, sel_shared_application)
            if not ns_app:
                return

            appearance_name = b"NSAppearanceNameDarkAqua" if theme_name == "darkly" else b"NSAppearanceNameAqua"
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            ns_appearance_name = objc.objc_msgSend(ns_string_class, sel_string_with_utf8, appearance_name)
            if not ns_appearance_name:
                return

            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            appearance = objc.objc_msgSend(ns_appearance_class, sel_appearance_named, ns_appearance_name)
            objc.objc_msgSend(ns_app, sel_set_appearance, appearance)
        except Exception:
            pass

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
        filemenu.add_command(label="New Session", command=self.new_session)
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
        editmenu.add_separator()
        editmenu.add_command(label="Find...", command=self._open_find_replace_dialog, accelerator=self._accel("F"))
        editmenu.add_command(label="Replace...", command=lambda: self._open_find_replace_dialog(replace=True))
        menubar.add_cascade(label="Edit", menu=editmenu)

        # Run
        runmenu = tk.Menu(menubar, tearoff=0)
        self.run_menu = runmenu
        runmenu.add_command(
            label="Solve Model",
            command=self.run_model,
            accelerator=self._accel("R"),
        )
        solver_menu = tk.Menu(runmenu, tearoff=0)
        solver_menu.add_radiobutton(label="Gurobi", variable=self.solver, value="gurobi", command=self._on_solver_selected)
        solver_menu.add_radiobutton(
            label="Scipy (HiGHS)", variable=self.solver, value="scipy", command=self._on_solver_selected
        )
        solver_menu.add_separator()
        solver_menu.add_checkbutton(
            label="Display Solver Progress",
            onvalue=True,
            offvalue=False,
            variable=self.display_solver_progress_var,
            command=self._on_display_solver_progress_toggled,
        )
        runmenu.add_cascade(label="Solver", menu=solver_menu)
        menubar.add_cascade(label="Solve", menu=runmenu)

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
                if label in ("Solve Model", "Stop Model"):
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
                label="Solve Model",
                command=self.run_model,
                accelerator=self._accel("R"),
            )

    def _ensure_no_active_operation(self, requested_label: str) -> bool:
        """Return True when no foreground operation is active; otherwise notify the user."""
        active = getattr(self, "_active_operation", None)
        if active is None:
            return True

        msg = f"{active.label} is already running. Wait for it to finish before starting {requested_label}."
        try:
            self.status_var.set(msg)
        except Exception:
            pass
        try:
            messagebox.showinfo(requested_label, msg)
        except Exception:
            pass
        return False

    def _start_foreground_operation(
        self,
        *,
        kind: str,
        label: str,
        header: str,
        status: str,
        solver_choice: Optional[str] = None,
        model_file: Optional[str] = None,
        data_file: Optional[str] = None,
        explain_after_solve: bool = False,
    ) -> Optional[_ForegroundOperation]:
        """Create a new output session and bind it to a single foreground operation."""
        if not self._ensure_no_active_operation(label):
            return None

        session_id = self._clear_output(header)
        operation = _ForegroundOperation(
            kind=kind,
            label=label,
            session_id=session_id,
            solver_choice=solver_choice,
            model_file=model_file,
            data_file=data_file,
            explain_after_solve=explain_after_solve,
        )
        self._active_operation = operation
        self.status_var.set(status)
        self._refresh_foreground_operation_ui()
        return operation

    def _finish_foreground_operation(self, operation: Optional[_ForegroundOperation]) -> None:
        """Clear the active foreground operation when the matching request completes."""
        if operation is None:
            return
        if getattr(self, "_active_operation", None) is operation:
            self._active_operation = None
            self._refresh_foreground_operation_ui()

    def _set_editors_locked(self, locked: bool) -> None:
        """Toggle the editor widgets between editable and read-only states."""
        desired = "disabled" if locked else "normal"
        for widget_name in ("model_text", "data_text"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                if str(widget.cget("state")) != desired:
                    widget.config(state=desired)
            except Exception:
                pass

    def _refresh_foreground_operation_ui(self) -> None:
        """Refresh UI elements that depend on foreground-operation state."""
        self._set_editors_locked(getattr(self, "_active_operation", None) is not None)
        if getattr(self, "_genai_loading", False) and not getattr(self, "_genai_provider_models", None):
            try:
                self.genai_menu.delete(0, tk.END)
            except Exception:
                pass
            try:
                self.genai_menu.add_command(label="Loading models...", state="disabled")
                active = getattr(self, "_active_operation", None)
                if active is not None:
                    self.genai_menu.add_separator()
                    self.genai_menu.add_command(label=f"Interrupt {active.label}", command=self.interrupt_active_operation)
                self.menubar.entryconfig("GenAI", state="normal")
            except Exception:
                pass
            return
        if hasattr(self, "genai_menu"):
            self._populate_genai_model_menus(getattr(self, "_genai_provider_models", {}))
        if hasattr(self, "_refresh_genai_panel_state"):
            self._refresh_genai_panel_state()

    def interrupt_active_operation(self) -> None:
        """Interrupt the current foreground operation."""
        active = getattr(self, "_active_operation", None)
        if active is None:
            return
        if active.kind == "solve" and self._solver_process and self._solver_process.is_alive():
            self.stop_model()
            return

        active.cancel_requested = True
        self._append_output("\nOperation interrupted by user.\n", active.session_id)
        self.status_var.set(f"{active.label} interrupted.")
        self._finish_foreground_operation(active)

    def _accel(self, key: str) -> str:
        """Return platform-aware accelerator label."""
        return f"{'Cmd' if sys.platform == 'darwin' else 'Ctrl'}+{key}"

    def new_model(self) -> None:
        """Clear editors, reset file paths, and prepare for a new model."""
        if not self._ensure_no_active_operation("New Model"):
            return
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
        self._mark_editor_baselines_saved()
        self.status_var.set("New model created. Ready.")

        # Clear output with a message
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "New model created. Ready.\n")
        self.output_text.config(state="disabled")

    def new_session(self) -> None:
        """Clear saved and in-memory session history and start a fresh session."""
        if not self._ensure_no_active_operation("New Session"):
            return
        if not messagebox.askyesno(
            "New Session",
            "Are you sure you want to proceed? This will clear the current editors and session history.",
        ):
            return
        try:
            # Also clear editors and reset file state to a blank IDE
            try:
                self.model_text.delete(1.0, tk.END)
            except Exception:
                pass
            try:
                self.data_text.delete(1.0, tk.END)
            except Exception:
                pass
            try:
                self.model_file = None
            except Exception:
                self.model_file = None
            try:
                self.data_file = None
            except Exception:
                self.data_file = None
            self._mark_editor_baselines_saved()

            # Reset tab labels and focus
            try:
                self.editor_notebook.tab(self.model_frame, text="Model")
                self.editor_notebook.tab(self.data_frame, text="Data")
                self.editor_notebook.select(self.model_frame)
            except Exception:
                pass

            # Re-highlight and update caret
            try:
                self.highlight(self.model_text, is_data=False)
                self.highlight(self.data_text, is_data=True)
                self._update_caret_position(self.model_text)
            except Exception:
                pass

            # Clear in-memory session structures
            try:
                self._output_sessions.clear()
            except Exception:
                self._output_sessions = {}
            try:
                self._output_session_ids.clear()
            except Exception:
                self._output_session_ids = []
            try:
                self._output_session_display.clear()
            except Exception:
                self._output_session_display = {}
            try:
                self._output_session_label.clear()
            except Exception:
                self._output_session_label = {}
            try:
                self._output_session_timestamp.clear()
            except Exception:
                self._output_session_timestamp = {}
            try:
                self._output_session_artifacts.clear()
            except Exception:
                self._output_session_artifacts = {}
            self._current_output_session_id = None
            self._viewing_output_session_id = None

            # Remove persisted session file if present
            try:
                path = self._session_file_path()
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                logging.getLogger(__name__).exception("Failed to remove .pyopl_session file during new_session")

            # Clear UI list and output pane
            if hasattr(self, "request_listbox"):
                try:
                    self.request_listbox.delete(0, tk.END)
                except Exception:
                    pass

            # Clear output pane (do NOT create a new timestamped session entry)
            try:
                if hasattr(self, "output_text") and self.output_text.winfo_exists():
                    self.output_text.config(state="normal")
                    self.output_text.delete("1.0", tk.END)
                    self.output_text.insert(tk.END, "Session cleared.\n")
                    self.output_text.config(state="disabled")
            except Exception:
                pass

            try:
                self.status_var.set("Session cleared.")
            except Exception:
                pass
        except Exception as e:
            logging.getLogger(__name__).exception("Error while creating new session")
            try:
                messagebox.showerror("Session Error", f"Failed to clear session: {e}")
            except Exception:
                pass

    def _setup_panes(self) -> None:
        """Set up vertical editor/output rows with independent top and bottom side panels."""
        editor_output_paned = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            sashrelief=tk.FLAT,
            bd=0,
            bg="#e9ecef",
            sashwidth=6,
            showhandle=False,
            relief=tk.FLAT,
        )
        editor_output_paned.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        self.editor_output_paned = editor_output_paned

        top_row_paned = tk.PanedWindow(
            editor_output_paned,
            orient=tk.HORIZONTAL,
            sashrelief=tk.FLAT,
            bd=0,
            bg="#e9ecef",
            sashwidth=6,
            showhandle=False,
            relief=tk.FLAT,
        )
        self.top_row_paned = top_row_paned
        editor_output_paned.add(top_row_paned, stretch="always")

        bottom_row_paned = tk.PanedWindow(
            editor_output_paned,
            orient=tk.HORIZONTAL,
            sashrelief=tk.FLAT,
            bd=0,
            bg="#e9ecef",
            sashwidth=6,
            showhandle=False,
            relief=tk.FLAT,
        )
        self.bottom_row_paned = bottom_row_paned
        editor_output_paned.add(bottom_row_paned, minsize=150)

        self._setup_editors(top_row_paned)
        self._setup_genai_panel(top_row_paned)
        self._setup_output(bottom_row_paned)
        self._setup_sessions_panel(bottom_row_paned)

        self.top_row_paned.bind("<Configure>", self._on_main_paned_configure, add="+")
        self.top_row_paned.bind("<ButtonRelease-1>", self._sync_side_panel_width_from_top, add="+")
        self.bottom_row_paned.bind("<ButtonRelease-1>", self._sync_side_panel_width_from_bottom, add="+")

    def _on_main_paned_configure(self, event: Optional[tk.Event] = None) -> None:
        """Apply the initial side-panel widths once the panes have real sizes."""
        if self._initial_main_pane_ratio_applied:
            return
        self._apply_initial_main_pane_ratio()

    def _apply_initial_main_pane_ratio(self) -> None:
        """Set the initial widths for the GenAI and session side panels."""
        if not hasattr(self, "top_row_paned") or not self.top_row_paned.winfo_exists():
            return
        try:
            self.update_idletasks()
            top_width = int(self.top_row_paned.winfo_width())
            bottom_width = int(self.bottom_row_paned.winfo_width()) if hasattr(self, "bottom_row_paned") else top_width
        except Exception:
            return
        if top_width <= 1 or bottom_width <= 1:
            return
        try:
            self._sync_side_panel_width(self._initial_genai_panel_width)
            self._initial_main_pane_ratio_applied = True
            self.after_idle(self._stabilize_initial_side_panel_width)
        except Exception:
            pass

    def _stabilize_initial_side_panel_width(self, attempts: int = 6) -> None:
        """Re-apply the startup side-panel width until Tk finishes settling geometry."""
        try:
            self.update_idletasks()
            self._sync_side_panel_width(self._initial_genai_panel_width)
            top_width = (
                int(self.genai_panel.winfo_width()) if hasattr(self, "genai_panel") else self._initial_genai_panel_width
            )
            bottom_width = (
                int(self.genai_sessions_panel.winfo_width())
                if hasattr(self, "genai_sessions_panel")
                else self._initial_genai_panel_width
            )
        except Exception:
            return
        if attempts <= 0:
            return
        if top_width != self._initial_genai_panel_width or bottom_width != self._initial_genai_panel_width:
            self.after(10, lambda: self._stabilize_initial_side_panel_width(attempts - 1))

    def _place_side_panel_width(self, paned: tk.PanedWindow, side_width: int) -> None:
        """Place a paned-window sash so the right pane gets a target width."""
        try:
            total_width = int(paned.winfo_width())
        except Exception:
            return
        if total_width <= 1:
            return
        try:
            sash_width = int(paned.cget("sashwidth"))
        except Exception:
            sash_width = 0
        left_width = max(200, total_width - int(side_width) - sash_width)
        try:
            paned.sash_place(0, left_width, 0)
        except Exception:
            pass

    def _sync_side_panel_width(self, side_width: int) -> None:
        """Keep the GenAI and session side panels at the same width."""
        if side_width <= 1:
            return
        self._side_panel_width = int(side_width)
        if self._genai_panel_visible and hasattr(self, "top_row_paned") and self.top_row_paned.winfo_exists():
            self._place_side_panel_width(self.top_row_paned, self._side_panel_width)
        if hasattr(self, "bottom_row_paned") and self.bottom_row_paned.winfo_exists():
            self._place_side_panel_width(self.bottom_row_paned, self._side_panel_width)

    def _sync_side_panel_width_from_top(self, event: Optional[tk.Event] = None) -> None:
        """After dragging the GenAI sash, mirror its width onto the session list."""
        if not self._genai_panel_visible or not hasattr(self, "genai_panel"):
            return
        try:
            self.update_idletasks()
            self._sync_side_panel_width(int(self.genai_panel.winfo_width()))
        except Exception:
            pass

    def _sync_side_panel_width_from_bottom(self, event: Optional[tk.Event] = None) -> None:
        """After dragging the session-list sash, mirror its width onto the GenAI panel."""
        if not hasattr(self, "genai_sessions_panel"):
            return
        try:
            self.update_idletasks()
            self._sync_side_panel_width(int(self.genai_sessions_panel.winfo_width()))
        except Exception:
            pass

    def _on_window_resize(self, event: Optional[tk.Event] = None) -> None:
        """Reapply the current side-panel width after top-level window resizes."""
        if event is not None and getattr(event, "widget", None) is not self:
            return
        if self._panel_resize_after_id is not None:
            try:
                self.after_cancel(self._panel_resize_after_id)
            except Exception:
                pass
        self._panel_resize_after_id = self.after(10, self._apply_panel_resize_sync)

    def _apply_panel_resize_sync(self) -> None:
        """Apply the current shared side-panel width after a debounced resize."""
        self._panel_resize_after_id = None
        try:
            self.update_idletasks()
            self._sync_side_panel_width(self._side_panel_width)
            self._sync_genai_mode_width()
        except Exception:
            pass

    def _on_genai_prompt_configure(self, event: Optional[tk.Event] = None) -> None:
        """Resync the segmented control after prompt widget size changes."""
        self.after_idle(self._sync_genai_mode_width)

    def _sync_genai_mode_width(self) -> None:
        """Match the segmented control width to the prompt text area excluding the scrollbar."""
        if not hasattr(self, "genai_mode_frame") or not hasattr(self, "genai_prompt_text"):
            return
        try:
            prompt_width = int(self.genai_prompt_text.winfo_width())
            scrollbar = getattr(self.genai_prompt_text, "vbar", None)
            scrollbar_width = 0
            if scrollbar is not None:
                scrollbar_width = max(int(scrollbar.winfo_width()), int(scrollbar.winfo_reqwidth()))
            gap_width = 4 + (1 if (prompt_width - 4) % 2 else 0)
            self.genai_mode_frame.columnconfigure(1, minsize=gap_width)
            self.genai_mode_frame.grid_configure(padx=(0, scrollbar_width))
        except Exception:
            pass

    def _get_selected_request_session_id(self) -> Optional[str]:
        """Return the currently selected session id from the request list."""
        try:
            sel = self.request_listbox.curselection()
            index = int(sel[0]) if sel else getattr(self, "_last_request_popup_index", None)
        except Exception:
            index = getattr(self, "_last_request_popup_index", None)
        if index is None or index < 0 or index >= len(self._output_session_ids):
            return None
        return self._output_session_ids[index]

    def _record_output_session_artifacts(
        self,
        session_id: Optional[str],
        model_text: Any = _SESSION_ARTIFACT_UNSET,
        data_text: Any = _SESSION_ARTIFACT_UNSET,
    ) -> None:
        """Associate a session with its persisted model/data snapshot content."""
        if not session_id:
            return
        if not hasattr(self, "_output_session_artifacts") or self._output_session_artifacts is None:
            self._output_session_artifacts = {}
        artifact = dict(self._output_session_artifacts.get(session_id, {}))
        if model_text is not _SESSION_ARTIFACT_UNSET:
            artifact["model_text"] = str(model_text)
        if data_text is not _SESSION_ARTIFACT_UNSET:
            artifact["data_text"] = str(data_text)
        self._output_session_artifacts[session_id] = artifact

    def _snapshot_output_session_artifacts(self, session_id: Optional[str]) -> None:
        """Snapshot the current editor state inline for a session when possible."""
        if not session_id:
            return
        if not hasattr(self, "model_text") or not hasattr(self, "data_text"):
            return
        model_text = self.model_text.get("1.0", tk.END).rstrip("\n")
        data_text = self.data_text.get("1.0", tk.END).rstrip("\n")
        OPLIDE._record_output_session_artifacts(self, session_id, model_text=model_text, data_text=data_text)

    def _get_output_session_artifacts(self, session_id: Optional[str]) -> dict[str, str]:
        """Return the stored model/data snapshot content for a session."""
        if not session_id:
            return {}
        return dict(self._output_session_artifacts.get(session_id, {}))

    def _show_session_model_preview(self) -> None:
        """Preview the selected session's model/data snapshot in a read-only window."""
        session_id = self._get_selected_request_session_id()
        artifacts = self._get_output_session_artifacts(session_id)
        has_model = "model_text" in artifacts
        has_data = "data_text" in artifacts
        model_text = str(artifacts.get("model_text", ""))
        data_text = str(artifacts.get("data_text", ""))
        if not has_model and not has_data:
            messagebox.showinfo("Models", "No saved model/data pair is associated with the selected session.")
            return
        try:
            window = tk.Toplevel(self)
            window.title("Preview Session Model")
            window.geometry("980x700")
            window.transient(self)
            window.rowconfigure(0, weight=1)
            window.columnconfigure(0, weight=1)
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 8))
            for label, content, is_present in (("Model", model_text, has_model), ("Data", data_text, has_data)):
                if not is_present:
                    continue
                frame = ttk.Frame(notebook, padding=(0, 0, 0, 0))
                text_widget = self._create_diff_preview_text(frame)
                text_widget.configure(state="normal")
                text_widget.delete("1.0", tk.END)
                text_widget.insert(tk.END, content)
                text_widget.configure(state="disabled")
                notebook.add(frame, text=label)
            ttk.Button(window, text="Close", command=window.destroy).grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))
        except Exception:
            pass

    def _show_session_model_diff(self) -> None:
        """Diff the selected session's model/data snapshot against the currently loaded editors."""
        session_id = self._get_selected_request_session_id()
        artifacts = self._get_output_session_artifacts(session_id)
        has_model = "model_text" in artifacts
        has_data = "data_text" in artifacts
        selected_model = str(artifacts.get("model_text", ""))
        selected_data = str(artifacts.get("data_text", ""))
        if not has_model and not has_data:
            messagebox.showinfo("Models", "No saved model/data pair is associated with the selected session.")
            return
        try:
            window = tk.Toplevel(self)
            window.title("Diff Against Current Model")
            window.geometry("980x700")
            window.transient(self)
            window.rowconfigure(0, weight=1)
            window.columnconfigure(0, weight=1)
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 8))
            current_model = self.model_text.get("1.0", tk.END)
            current_data = self.data_text.get("1.0", tk.END)
            for label, current_text, selected_text, is_present in (
                ("Model", current_model, selected_model, has_model),
                ("Data", current_data, selected_data, has_data),
            ):
                if not is_present:
                    continue
                frame = ttk.Frame(notebook, padding=(0, 0, 0, 0))
                text_widget = self._create_diff_preview_text(frame)
                self._populate_diff_preview_text(text_widget, selected_text, current_text)
                notebook.add(frame, text=label)
            ttk.Button(window, text="Close", command=window.destroy).grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))
        except Exception:
            pass

    def _restore_session_model(self) -> None:
        """Restore the selected session's model/data snapshot into the editors."""
        session_id = self._get_selected_request_session_id()
        artifacts = self._get_output_session_artifacts(session_id)
        has_model = "model_text" in artifacts
        has_data = "data_text" in artifacts
        model_text = str(artifacts.get("model_text", ""))
        data_text = str(artifacts.get("data_text", ""))
        if not has_model and not has_data:
            messagebox.showinfo("Models", "No saved model/data pair is associated with the selected session.")
            return
        if not messagebox.askyesno("Restore Model", "Restore the selected model/data pair into the editors?"):
            return
        if has_model:
            self.model_text.delete("1.0", tk.END)
            self.model_text.insert(tk.END, model_text)
            self.model_file = None
        if has_data:
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, data_text)
            self.data_file = None
        if has_model:
            self.editor_notebook.tab(self.model_frame, text="Model: session snapshot")
        if has_data:
            self.editor_notebook.tab(self.data_frame, text="Data: session snapshot")
        self.highlight(self.model_text, is_data=False)
        self.highlight(self.data_text, is_data=True)
        self.status_var.set("Model restored from session")

    def _populate_request_context_menu(self, session_id: Optional[str]) -> None:
        """Populate the session list context menu."""
        try:
            self.request_context_menu.delete(0, tk.END)
        except Exception:
            pass
        artifacts = self._get_output_session_artifacts(session_id)
        has_artifacts = "model_text" in artifacts or "data_text" in artifacts
        self.request_context_menu.add_command(
            label="Preview",
            command=self._show_session_model_preview,
            state=("normal" if has_artifacts else "disabled"),
        )
        self.request_context_menu.add_command(
            label="Diff",
            command=self._show_session_model_diff,
            state=("normal" if has_artifacts else "disabled"),
        )
        self.request_context_menu.add_command(
            label="Restore",
            command=self._restore_session_model,
            state=("normal" if has_artifacts else "disabled"),
        )
        self.request_context_menu.add_separator()
        self.request_context_menu.add_command(label="Change label", command=self._rename_selected_request)
        self.request_context_menu.add_command(label="Delete entry", command=self._delete_selected_request)

    def _setup_editors(self, parent: tk.PanedWindow) -> None:
        """Create model and data editor frames inside a Notebook."""
        editor_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=0)
        parent.add(editor_frame, stretch="always")

        # Notebook
        self.editor_notebook = ttk.Notebook(editor_frame, style="Editor.TNotebook")
        self.editor_notebook.pack(fill=tk.BOTH, expand=1)

        # Model editor
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
        self._replace_scrolled_text_vbar(self.model_text)
        self.model_text.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

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
        self._replace_scrolled_text_vbar(self.data_text)
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
        """Create the output panel."""
        output_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=0)
        self.output_frame = output_frame

        container = ttk.Frame(output_frame)
        container.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Output text
        left = ttk.Frame(container)
        self.output_text_container = left
        self._solver_log_text: Optional[scrolledtext.ScrolledText] = None
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
        self._replace_scrolled_text_vbar(self.output_text)
        self.output_text.pack(fill=tk.BOTH, expand=1)

        parent.add(output_frame, minsize=150)

    def _show_solver_log_textbox(self) -> None:
        """Temporarily replace the persistent output pane with solver logs."""
        if not hasattr(self, "output_text_container"):
            return
        try:
            existing_log_text = self._solver_log_text
            if existing_log_text is not None and existing_log_text.winfo_exists():
                existing_log_text.config(state="normal")
                existing_log_text.delete("1.0", tk.END)
                existing_log_text.config(state="disabled")
                return
        except Exception:
            self._solver_log_text = None

        try:
            if hasattr(self, "output_text") and self.output_text.winfo_exists():
                self.output_text.pack_forget()
        except Exception:
            pass

        try:
            log_text = scrolledtext.ScrolledText(
                self.output_text_container,
                wrap=tk.WORD,
                height=12,
                font=(self.editor_font_family, self.current_font_size - 1),
                state="disabled",
                bg="#f8f9fa",
                fg="#212529",
                relief=tk.FLAT,
                bd=0,
            )
            self._replace_scrolled_text_vbar(log_text)
            log_text.pack(fill=tk.BOTH, expand=1)
            self._solver_log_text = log_text
            self._apply_theme_colors()
        except Exception:
            self._solver_log_text = None
            try:
                self._destroy_scrolled_text(log_text)
                self.output_text.pack(fill=tk.BOTH, expand=1)
            except Exception:
                pass

    def _destroy_scrolled_text(self, text_widget: Any) -> None:
        """Destroy a ScrolledText widget and its wrapper frame when present."""
        if text_widget is None:
            return
        try:
            frame = getattr(text_widget, "frame", None)
            if frame is not None and frame.winfo_exists():
                frame.destroy()
                return
        except Exception:
            pass
        try:
            if text_widget.winfo_exists():
                text_widget.destroy()
        except Exception:
            pass

    def _append_solver_log_text(self, text: str) -> None:
        log_text = getattr(self, "_solver_log_text", None)
        if log_text is None:
            return
        try:
            if not log_text.winfo_exists():
                return
            log_text.config(state="normal")
            log_text.insert(tk.END, str(text))
            log_text.see(tk.END)
            log_text.config(state="disabled")
        except Exception:
            pass

    def _restore_output_textbox(self) -> None:
        log_text = getattr(self, "_solver_log_text", None)
        if log_text is not None:
            self._destroy_scrolled_text(log_text)
        self._solver_log_text = None
        try:
            if hasattr(self, "output_text") and self.output_text.winfo_exists():
                self.output_text.pack(fill=tk.BOTH, expand=1)
            if hasattr(self, "output_text") and self.output_text.winfo_exists():
                session_id = getattr(self, "_viewing_output_session_id", None) or getattr(
                    self, "_current_output_session_id", None
                )
                if session_id:
                    content = self._output_sessions.get(session_id, "")
                    self.output_text.config(state="normal")
                    self.output_text.delete("1.0", tk.END)
                    self.output_text.insert(tk.END, content)
                    self.output_text.see(tk.END)
                    self.output_text.config(state="disabled")
        except Exception:
            pass

    def _setup_genai_panel(self, parent: tk.PanedWindow) -> None:
        """Create the GenAI composer panel for the top row."""
        panel = ttk.Frame(parent, style="Sidebar.TFrame", width=300, padding=(5, 10))
        parent.add(panel, minsize=180)
        panel.pack_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.genai_panel = panel

        composer_panel = ttk.Frame(panel, style="Sidebar.TFrame", padding=(0, 0, 0, 0))
        composer_panel.columnconfigure(0, weight=1)
        composer_panel.rowconfigure(0, minsize=24)
        composer_panel.grid(row=0, column=0, sticky="nsew")

        mode_frame = ttk.Frame(composer_panel, style="Sidebar.TFrame")
        mode_frame.grid(row=1, column=0, sticky="ew", pady=(2, 10))
        self.genai_mode_frame = mode_frame
        mode_frame.columnconfigure(0, weight=1, uniform="genai-mode")
        mode_frame.columnconfigure(1, minsize=4)
        mode_frame.columnconfigure(2, weight=1, uniform="genai-mode")
        mode_button_width = max(len("Generate"), len("Ask"))
        self.genai_generate_mode_button = ttk.Button(
            mode_frame,
            text="Generate",
            style="GenaiModeActive.TButton",
            width=mode_button_width,
            command=lambda: self._set_genai_panel_mode("generate"),
        )
        self.genai_generate_mode_button.grid(row=0, column=0, sticky="ew")
        self.genai_ask_mode_button = ttk.Button(
            mode_frame,
            text="Ask",
            style="GenaiMode.TButton",
            width=mode_button_width,
            command=lambda: self._set_genai_panel_mode("ask"),
        )
        self.genai_ask_mode_button.grid(row=0, column=2, sticky="ew")

        composer = ttk.Frame(composer_panel, style="Sidebar.TFrame")
        composer.grid(row=3, column=0, sticky="nsew", pady=(0, 0))
        composer.columnconfigure(0, weight=1)
        composer_panel.rowconfigure(3, weight=1)
        ttk.Label(composer, textvariable=self.genai_prompt_title_var, style="SidebarSection.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.genai_prompt_text = scrolledtext.ScrolledText(
            composer,
            wrap=tk.WORD,
            height=12,
            font=(self.editor_font_family, self.current_font_size),
            relief=tk.FLAT,
            bd=0,
        )
        self._replace_scrolled_text_vbar(self.genai_prompt_text)
        self.genai_prompt_text.grid(row=1, column=0, sticky="nsew")
        self._bind_autohide_vertical_scrollbar(
            self.genai_prompt_text,
            getattr(self.genai_prompt_text, "vbar", None),
            on_toggle=self._sync_genai_mode_width,
        )
        self.genai_prompt_text.bind("<Control-Return>", self._submit_genai_from_event)
        self.genai_prompt_text.bind("<Command-Return>", self._submit_genai_from_event)
        self.genai_prompt_text.bind("<Configure>", self._on_genai_prompt_configure, add="+")
        composer.rowconfigure(1, weight=1)

        self.genai_attach_output_check = ttk.Checkbutton(
            composer,
            text="Attach output",
            variable=self.genai_attach_output_var,
            style="Sidebar.TCheckbutton",
        )
        self.genai_attach_output_check.grid(row=2, column=0, sticky="w", pady=(8, 0))

        attachments = ttk.Frame(composer, style="Sidebar.TFrame")
        attachments.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        attachments.columnconfigure(0, weight=1)
        ttk.Label(attachments, textvariable=self.genai_attachment_summary_var, style="SidebarSubtle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.genai_attachment_listbox = tk.Listbox(attachments, height=4, exportselection=False, activestyle="none")
        self.genai_attachment_listbox.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.genai_attachment_menu = tk.Menu(self, tearoff=0)
        self.genai_attachment_menu.add_command(label="Attach images or short PDFs...", command=self._genai_add_images)
        self.genai_attachment_menu.add_command(label="Remove selected", command=self._genai_remove_selected_image)
        self.genai_attachment_menu.add_separator()
        self.genai_attachment_menu.add_command(label="Clear all", command=self._genai_clear_images)
        self.genai_attachment_listbox.bind("<Button-3>", self._on_genai_attachment_right_click)
        self.genai_attachment_listbox.bind("<Double-Button-1>", lambda _event: self._genai_add_images())
        if sys.platform == "darwin":
            self.genai_attachment_listbox.bind("<Button-2>", self._on_genai_attachment_right_click)
            self.genai_attachment_listbox.bind("<Control-Button-1>", self._on_genai_attachment_right_click)

        pending = ttk.Frame(composer_panel, style="Sidebar.TFrame")
        pending.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        pending.columnconfigure(0, weight=1)
        self.genai_pending_frame = pending
        self.genai_pending_label = ttk.Label(pending, textvariable=self.genai_pending_var, style="SidebarSection.TLabel")
        self.genai_pending_label.grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(pending, style="Sidebar.TFrame")
        actions.grid(row=1, column=0, sticky="e", pady=(6, 0))
        self.genai_preview_button = ttk.Button(actions, text="Review", command=self._show_pending_genai_diff_preview)
        self.genai_preview_button.grid(row=0, column=0)
        self.genai_apply_button = ttk.Button(actions, text="Apply", command=self._apply_pending_genai_revisions)
        self.genai_apply_button.grid(row=0, column=1, padx=(6, 0))
        self.genai_dismiss_button = ttk.Button(actions, text="Dismiss", command=self._clear_pending_genai_revisions)
        self.genai_dismiss_button.grid(row=0, column=2, padx=(6, 0))

        footer = ttk.Frame(composer_panel, style="Sidebar.TFrame")
        footer.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        self.genai_submit_button = ttk.Button(
            footer, textvariable=self.genai_submit_label_var, command=self._submit_genai_request
        )
        self.genai_submit_button.grid(row=0, column=0, sticky="e")

    def _setup_sessions_panel(self, parent: tk.PanedWindow) -> None:
        """Create the output-session list panel for the bottom row."""
        sessions_panel = ttk.Frame(parent, style="Sidebar.TFrame", width=300, padding=0)
        parent.add(sessions_panel, minsize=180)
        sessions_panel.pack_propagate(False)
        sessions_panel.columnconfigure(0, weight=1)
        sessions_panel.rowconfigure(0, weight=1)
        self.genai_sessions_panel = sessions_panel

        sessions_list = tk.Frame(sessions_panel, bd=0, highlightthickness=1)
        sessions_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        sessions_list.columnconfigure(0, weight=1)
        sessions_list.rowconfigure(0, weight=1)
        self.sessions_surface = sessions_list

        self.request_listbox = tk.Listbox(sessions_list, exportselection=False, height=10, activestyle="none")
        request_scroll = ttk.Scrollbar(sessions_list, orient=tk.VERTICAL, command=self.request_listbox.yview)
        self.request_listbox.configure(yscrollcommand=request_scroll.set)
        self.request_listbox.grid(row=0, column=0, sticky="nsew")
        request_scroll.grid(row=0, column=1, sticky="ns")
        self._bind_autohide_vertical_scrollbar(self.request_listbox, request_scroll)

        self.request_listbox.bind("<<ListboxSelect>>", self._on_request_select)

        self.request_context_menu = tk.Menu(self, tearoff=0)
        self.request_listbox.bind("<Button-3>", self._on_request_right_click)
        if sys.platform == "darwin":
            self.request_listbox.bind("<Button-2>", self._on_request_right_click)
            self.request_listbox.bind("<Control-Button-1>", self._on_request_right_click)

        self._clear_pending_genai_revisions()
        self._refresh_genai_panel_state()

    def _set_genai_panel_visible(self, visible: bool) -> None:
        """Show or hide the top-row GenAI composer while keeping the session list visible."""
        visible = bool(visible)
        self.show_genai_panel_var.set(visible)
        if visible == self._genai_panel_visible:
            return
        self._genai_panel_visible = visible
        try:
            if visible:
                self.top_row_paned.add(self.genai_panel, minsize=180)
                self.update_idletasks()
                self._sync_side_panel_width(self._side_panel_width)
            else:
                self.top_row_paned.forget(self.genai_panel)
        except Exception:
            pass

    def _toggle_genai_panel_visibility(self) -> None:
        """Menu command for the GenAI composer visibility toggle."""
        self._set_genai_panel_visible(self.show_genai_panel_var.get())

    def _setup_status_bar(self) -> None:
        """Create a segmented status bar at the bottom of the window."""
        self.status_message_var = tk.StringVar(value="Ready")
        self.status_syntax_var = tk.StringVar(value="Syntax OK")
        self.status_caret_var = tk.StringVar(value="Ln 1, Col 0")
        self.status_solver_var = tk.StringVar(value="Solver: Gurobi")
        self.status_genai_var = tk.StringVar(value="GenAI: none")
        self.status_runtime_var = tk.StringVar(value="")
        self.status_var = _StatusVarProxy(self._set_status_message, lambda: self.status_message_var.get())

        status_bar = ttk.Frame(self, style="StatusBar.TFrame", padding=(6, 1))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.columnconfigure(0, weight=1)
        self.status_bar = status_bar
        self.status_bar_labels: list[ttk.Label] = []

        segments: list[tuple[tk.StringVar, Literal["w", "e"], int]] = [
            (self.status_message_var, "w", 0),
            (self.status_syntax_var, "w", 1),
            (self.status_caret_var, "w", 2),
            (self.status_solver_var, "w", 3),
            (self.status_genai_var, "w", 4),
            (self.status_runtime_var, "e", 5),
        ]

        for idx, (var, anchor, column) in enumerate(segments):
            label = ttk.Label(
                status_bar,
                textvariable=var,
                anchor=anchor,
                style="StatusBar.TLabel" if column == 0 else "StatusBarMeta.TLabel",
            )
            label.grid(row=0, column=idx, sticky=("ew" if column == 0 else "w"), padx=((0, 0) if idx == 0 else (10, 0)))
            self.status_bar_labels.append(label)

        self._refresh_status_context()

    def _set_status_message(self, message: str) -> None:
        """Update the primary status message segment."""
        self.status_message_var.set(str(message or "Ready"))

    def _refresh_status_context(self) -> None:
        """Refresh footer context segments that derive from current IDE state."""
        solver_name = "Gurobi" if getattr(self, "solver", None) and self.solver.get() == "gurobi" else "SciPy"
        self.status_solver_var.set(f"Solver: {solver_name}")
        provider = getattr(self, "genai_provider", None)
        model = getattr(self, "genai_model", None)
        method = self._label_for_method(self.genai_method_var.get()) if hasattr(self, "genai_method_var") else "SyntAGM"
        if provider and model:
            self.status_genai_var.set(f"GenAI: {provider} • {model} • {method}")
        else:
            self.status_genai_var.set("GenAI: none")

    def _on_solver_selected(self) -> None:
        """Persist solver menu selection and refresh dependent status UI."""
        solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"
        if solver_choice in ("gurobi", "scipy"):
            self._current_solver_choice = solver_choice
        self._refresh_status_context()
        self._save_settings()

    def _setup_tag_configs(self) -> None:
        """Configure syntax highlighting tags for editors."""
        for token, color in TOKEN_COLORS.items():
            self.model_text.tag_configure(token, foreground=color)
            self.data_text.tag_configure(token, foreground=color)
        # Error tag
        self.model_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        self.data_text.tag_configure("ERROR", background="#e06c75", foreground="black")
        # Comments
        self.model_text.tag_configure("COMMENT", font=(self.editor_font_family, self.current_font_size, "italic"))

    def _on_genai_mode_changed(self) -> None:
        """Update composer copy when the GenAI panel mode changes."""
        self._refresh_genai_panel_state()

    def _set_genai_panel_mode(self, mode: str) -> None:
        """Switch the GenAI composer mode from the segmented control."""
        if mode not in ("generate", "ask"):
            return
        self.genai_panel_mode_var.set(mode)
        self._refresh_genai_panel_state()

    def _refresh_genai_mode_buttons(self) -> None:
        """Keep the segmented mode buttons visually aligned with the selected mode."""
        mode = self.genai_panel_mode_var.get() if hasattr(self, "genai_panel_mode_var") else "generate"
        if hasattr(self, "genai_generate_mode_button"):
            self.genai_generate_mode_button.configure(
                style=("GenaiModeActive.TButton" if mode == "generate" else "GenaiMode.TButton")
            )
        if hasattr(self, "genai_ask_mode_button"):
            self.genai_ask_mode_button.configure(style=("GenaiModeActive.TButton" if mode == "ask" else "GenaiMode.TButton"))

    def _refresh_genai_panel_state(self) -> None:
        """Refresh docked GenAI panel labels and enabled state."""
        mode = self.genai_panel_mode_var.get() if hasattr(self, "genai_panel_mode_var") else "generate"
        self._refresh_genai_mode_buttons()
        if mode == "ask":
            self.genai_prompt_title_var.set("Ask about the current model and data")
            self.genai_submit_label_var.set("Ask")
        else:
            self.genai_prompt_title_var.set("Describe the optimization problem")
            self.genai_submit_label_var.set("Generate")

        if hasattr(self, "genai_attach_output_check"):
            if mode == "ask":
                self.genai_attach_output_check.grid()
            else:
                self.genai_attach_output_check.grid_remove()

        attachment_count = len(getattr(self, "_genai_attachment_paths", []))
        if attachment_count == 0:
            self.genai_attachment_summary_var.set("No visual attachments")
        elif attachment_count == 1:
            self.genai_attachment_summary_var.set("1 visual attachment")
        else:
            self.genai_attachment_summary_var.set(f"{attachment_count} visual attachments")

        active = getattr(self, "_active_operation", None)
        if hasattr(self, "genai_submit_button"):
            if active is not None and active.kind.startswith("genai"):
                self.genai_submit_label_var.set("Interrupt")
                self.genai_submit_button.configure(command=self.interrupt_active_operation, state="normal")
            else:
                self.genai_submit_button.configure(
                    command=self._submit_genai_request, state=("normal" if active is None else "disabled")
                )
        if hasattr(self, "status_genai_var"):
            self._refresh_status_context()

    def _open_genai_panel(self, mode: str, initial_text: str = "") -> None:
        """Focus the docked GenAI composer and set its mode."""
        if mode not in ("generate", "ask"):
            mode = "generate"
        self._set_genai_panel_visible(True)
        self.genai_panel_mode_var.set(mode)
        self._refresh_genai_panel_state()
        if initial_text and hasattr(self, "genai_prompt_text"):
            self.genai_prompt_text.delete("1.0", tk.END)
            self.genai_prompt_text.insert("1.0", initial_text)
        if hasattr(self, "genai_prompt_text"):
            self.genai_prompt_text.focus_set()

    def _clear_genai_composer(self) -> None:
        """Clear the docked GenAI composer state."""
        if hasattr(self, "genai_prompt_text"):
            self.genai_prompt_text.delete("1.0", tk.END)
        self._genai_attachment_paths.clear()
        self._genai_attachment_display_labels.clear()
        self._cleanup_genai_pdf_temp_dir()
        self._refresh_genai_attachment_list()

    def _label_for_genai_attachment_path(self, path: str) -> str:
        """Return a user-facing label for an attachment path."""
        labels = getattr(self, "_genai_attachment_display_labels", {})
        label = labels.get(path) if isinstance(labels, dict) else None
        return label or path

    def _list_label_for_genai_attachment_path(self, path: str) -> str:
        """Return the compact label shown in the composer attachment list."""
        label = OPLIDE._label_for_genai_attachment_path(self, path)
        if label == path:
            return os.path.basename(path)
        match = re.match(r"^(.*\.pdf)( page \d+)$", label, re.IGNORECASE)
        if match:
            return f"{os.path.basename(match.group(1))}{match.group(2)}"
        return label

    def _refresh_genai_attachment_list(self) -> None:
        """Refresh the composer attachment list."""
        if hasattr(self, "genai_attachment_listbox"):
            self.genai_attachment_listbox.delete(0, tk.END)
            for path in self._genai_attachment_paths:
                self.genai_attachment_listbox.insert(tk.END, self._list_label_for_genai_attachment_path(path))
        self._refresh_genai_panel_state()

    def _genai_add_images(self) -> None:
        """Attach one or more visual files to the docked GenAI composer."""
        try:
            paths = filedialog.askopenfilenames(
                title="Attach images or short PDFs",
                filetypes=[
                    ("Images and short PDFs", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.pdf"),
                    ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*"),
                ],
            )
        except Exception:
            paths = ()
        if not paths:
            return
        for path in paths:
            path_str = str(path)
            if not path_str:
                continue
            if path_str.lower().endswith(".pdf"):
                for rendered_path in self._render_genai_pdf_attachment(path_str):
                    if rendered_path not in self._genai_attachment_paths:
                        self._genai_attachment_paths.append(rendered_path)
            elif path_str not in self._genai_attachment_paths:
                self._genai_attachment_paths.append(path_str)
        self._refresh_genai_attachment_list()

    def _ensure_genai_pdf_temp_dir(self) -> str:
        """Return the temporary directory used for rendered PDF attachment pages."""
        temp_dir = getattr(self, "_genai_pdf_temp_dir", None)
        if temp_dir and os.path.isdir(temp_dir):
            return temp_dir
        temp_dir = tempfile.mkdtemp(prefix="rhetor_syntagm_pdf_")
        self._genai_pdf_temp_dir = temp_dir
        return temp_dir

    def _cleanup_genai_pdf_temp_dir(self) -> None:
        """Remove temporary rendered PDF attachment pages."""
        temp_dir = getattr(self, "_genai_pdf_temp_dir", None)
        if temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._genai_pdf_temp_dir = None

    def _render_genai_pdf_attachment(self, pdf_path: str) -> list[str]:
        """Render a short PDF attachment to image paths accepted by SyntAGM."""
        try:
            import fitz  # type: ignore[import-not-found]
        except Exception:
            messagebox.showerror(
                "GenAI",
                "PDF attachments require PyMuPDF. Install it with: pip install PyMuPDF",
                parent=self,
            )
            return []

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            messagebox.showerror("GenAI", f"Could not open PDF attachment:\n{exc}", parent=self)
            return []

        rendered_paths: list[str] = []
        try:
            raw_page_count = getattr(document, "page_count", None)
            page_count = int(raw_page_count if raw_page_count is not None else len(document))
            if page_count > GENAI_MAX_PDF_PAGES:
                messagebox.showwarning(
                    "GenAI",
                    f"SyntAGM accepts PDFs up to {GENAI_MAX_PDF_PAGES} pages. "
                    "For longer documents, attach screenshots of the relevant formulation.",
                    parent=self,
                )
                return []

            stem = Path(pdf_path).stem or "attachment"
            temp_dir = tempfile.mkdtemp(prefix=f"{stem}_", dir=self._ensure_genai_pdf_temp_dir())
            scale = GENAI_PDF_RENDER_DPI / 72.0
            matrix = fitz.Matrix(scale, scale)
            for page_index in range(page_count):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = os.path.join(temp_dir, f"page_{page_index + 1}.png")
                pixmap.save(image_path)
                self._genai_attachment_display_labels[image_path] = f"{pdf_path} page {page_index + 1}"
                rendered_paths.append(image_path)
        except Exception as exc:
            messagebox.showerror("GenAI", f"Could not render PDF attachment:\n{exc}", parent=self)
            return []
        finally:
            document.close()

        return rendered_paths

    def _genai_remove_selected_image(self) -> None:
        """Remove the selected attachment from the composer."""
        if not hasattr(self, "genai_attachment_listbox"):
            return
        sel = self.genai_attachment_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self._genai_attachment_paths):
            removed_path = self._genai_attachment_paths.pop(idx)
            self._genai_attachment_display_labels.pop(removed_path, None)
        self._refresh_genai_attachment_list()

    def _genai_clear_images(self) -> None:
        """Clear all attached images from the composer."""
        self._genai_attachment_paths.clear()
        self._genai_attachment_display_labels.clear()
        self._cleanup_genai_pdf_temp_dir()
        self._refresh_genai_attachment_list()

    def _on_genai_attachment_right_click(self, event: Optional[tk.Event]) -> None:
        """Show attachment actions from the image list context menu."""
        if event is None or not hasattr(self, "genai_attachment_listbox"):
            return
        try:
            size = self.genai_attachment_listbox.size()
            self.genai_attachment_listbox.selection_clear(0, tk.END)
            selected_index: Optional[int] = None
            if size > 0:
                idx = int(self.genai_attachment_listbox.nearest(event.y))
                bbox = self.genai_attachment_listbox.bbox(idx)
                if bbox is not None:
                    x1, y1, width, height = bbox
                    if y1 <= event.y <= y1 + height and x1 <= event.x <= x1 + width:
                        self.genai_attachment_listbox.selection_set(idx)
                        self.genai_attachment_listbox.activate(idx)
                        selected_index = idx

            has_any = bool(self._genai_attachment_paths)
            self.genai_attachment_menu.entryconfigure(
                "Remove selected", state=("normal" if selected_index is not None else "disabled")
            )
            self.genai_attachment_menu.entryconfigure("Clear all", state=("normal" if has_any else "disabled"))
            self.genai_attachment_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self.genai_attachment_menu.grab_release()
            except Exception:
                pass

    def _collect_genai_prompt_input(self) -> Optional[_PromptInput]:
        """Build PromptInput from the docked composer."""
        if not hasattr(self, "genai_prompt_text"):
            return None
        text_val = self.genai_prompt_text.get("1.0", tk.END).rstrip()
        prompt_input: Optional[_PromptInput]
        if self._genai_attachment_paths:
            if not text_val.strip() and not self._genai_attachment_paths:
                return None
            prompt_input = {"text": text_val, "images": [{"path": p} for p in self._genai_attachment_paths]}
        else:
            if not text_val.strip():
                return None
            prompt_input = text_val

        if (
            self.genai_panel_mode_var.get() == "ask"
            and hasattr(self, "genai_attach_output_var")
            and self.genai_attach_output_var.get()
        ):
            prompt_input = self._append_output_to_prompt_input(prompt_input, self._get_selected_output_session_text())
        return prompt_input

    def _get_selected_output_session_text(self) -> str:
        """Return the currently selected output session content, if any."""
        session_id: Optional[str] = None
        if hasattr(self, "request_listbox"):
            try:
                sel = self.request_listbox.curselection()
                if sel:
                    index = int(sel[0])
                    if 0 <= index < len(self._output_session_ids):
                        session_id = self._output_session_ids[index]
            except Exception:
                session_id = None
        if not session_id:
            session_id = getattr(self, "_viewing_output_session_id", None)
        if not session_id:
            session_id = getattr(self, "_current_output_session_id", None)
        if not session_id:
            return ""
        return str(self._output_sessions.get(session_id, "")).rstrip()

    def _append_output_to_prompt_input(self, prompt_input: _PromptInput, session_output: str) -> _PromptInput:
        """Append selected output history to the prompt text while preserving attachments."""
        output_text = str(session_output or "").rstrip()
        if not output_text:
            return prompt_input
        wrapped_output = f"<session_output>\n{output_text}\n</session_output>"
        if isinstance(prompt_input, dict):
            merged_input = dict(prompt_input)
            prompt_text = str(merged_input.get("text", "")).rstrip()
            merged_input["text"] = f"{prompt_text}\n\n{wrapped_output}" if prompt_text else wrapped_output
            return merged_input
        prompt_text = str(prompt_input).rstrip()
        return f"{prompt_text}\n\n{wrapped_output}" if prompt_text else wrapped_output

    def _submit_genai_from_event(self, event: Optional[tk.Event] = None) -> str:
        """Submit the active GenAI composer via keyboard."""
        self._submit_genai_request()
        return "break"

    def _selected_genai_method_supports_images(self) -> bool:
        """Return whether the selected GenAI method can consume attached images."""
        try:
            return self.genai_method_var.get() == "pyopl_generative"
        except Exception:
            return False

    def _submit_genai_request(self) -> None:
        """Dispatch the docked GenAI composer based on the selected mode."""
        if not self.genai_provider or not self.genai_model:
            messagebox.showwarning("GenAI", "No GenAI model selected.")
            return

        if self._genai_attachment_paths and not self._selected_genai_method_supports_images():
            method_label = self._label_for_method(self.genai_method_var.get())
            messagebox.showwarning(
                "GenAI",
                f"{method_label} does not support image attachments. Use SyntAGM or remove the attached images.",
            )
            return

        prompt_input = self._collect_genai_prompt_input()
        if prompt_input is None:
            self.status_var.set("GenAI: enter a prompt or attach an image")
            return

        if self.genai_panel_mode_var.get() == "ask":
            self._run_genai_feedback(prompt_input)
        else:
            self._run_genai_generate(prompt_input)

    def _set_pending_genai_revisions(self, pending: dict[str, Any]) -> None:
        """Expose pending revised model/data through inline panel actions."""
        self._genai_pending_revisions = pending
        revised_model = bool(pending.get("revised_model"))
        revised_data = bool(pending.get("revised_data"))
        labels = []
        if revised_model:
            labels.append("model")
        if revised_data:
            labels.append("data")
        self.genai_pending_var.set(f"Pending revised {' and '.join(labels)} from Ask")
        if hasattr(self, "genai_pending_frame"):
            self.genai_pending_frame.grid()
        self._show_pending_genai_diff_preview()

    def _clear_pending_genai_revisions(self) -> None:
        """Hide inline revision actions when no revisions are pending."""
        self._genai_pending_revisions = None
        self.genai_pending_var.set("")
        if hasattr(self, "genai_pending_frame"):
            self.genai_pending_frame.grid_remove()
        self._close_pending_genai_diff_preview()

    def _close_pending_genai_diff_preview(self) -> None:
        """Close the pending Ask revision preview window if it is open."""
        window = getattr(self, "_genai_diff_preview_window", None)
        self._genai_diff_preview_texts = {}
        self._genai_diff_preview_notebook = None
        self._genai_diff_preview_window = None
        if window is None:
            return
        try:
            if window.winfo_exists():
                window.destroy()
        except Exception:
            pass

    def _create_diff_preview_text(self, parent: ttk.Frame) -> tk.Text:
        """Create a read-only text widget for Ask revision previews."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        text_widget = tk.Text(
            container,
            wrap=tk.NONE,
            font=(self.editor_font_family, max(10, self.current_font_size - 1)),
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=8,
        )
        y_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
        x_scroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=text_widget.xview)
        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return text_widget

    def _configure_diff_preview_tags(self, text_widget: tk.Text) -> None:
        """Apply VS Code-like diff colors to the preview text widget."""
        if self.theme_var.get() == "darkly":
            bg = "#1f2329"
            fg = "#e6edf3"
            add_bg = "#12261b"
            add_fg = "#7ee787"
            remove_bg = "#30151b"
            remove_fg = "#ffa198"
            header_fg = "#8b949e"
        else:
            bg = "#ffffff"
            fg = "#24292f"
            add_bg = "#e6ffec"
            add_fg = "#1a7f37"
            remove_bg = "#ffebe9"
            remove_fg = "#cf222e"
            header_fg = "#57606a"
        text_widget.configure(bg=bg, fg=fg, insertbackground=fg)
        text_widget.tag_configure("diff_header", foreground=header_fg)
        text_widget.tag_configure("diff_add", background=add_bg, foreground=add_fg)
        text_widget.tag_configure("diff_remove", background=remove_bg, foreground=remove_fg)
        text_widget.tag_configure("diff_context", foreground=fg)

    def _populate_diff_preview_text(self, text_widget: tk.Text, original: str, revised: str) -> None:
        """Render a line-oriented diff preview into a read-only text widget."""
        self._configure_diff_preview_tags(text_widget)
        original_lines = original.splitlines()
        revised_lines = revised.splitlines()
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "--- Historical\n", ("diff_header",))
        text_widget.insert(tk.END, "+++ Current\n\n", ("diff_header",))
        for line in difflib.ndiff(original_lines, revised_lines):
            if line.startswith("? "):
                continue
            if line.startswith("+ "):
                tag = "diff_add"
            elif line.startswith("- "):
                tag = "diff_remove"
            else:
                tag = "diff_context"
            text_widget.insert(tk.END, f"{line}\n", (tag,))
        text_widget.configure(state="disabled")

    def _show_pending_genai_diff_preview(self) -> None:
        """Show a VS Code-like diff preview for pending Ask revisions."""
        pending = self._genai_pending_revisions
        if not pending:
            return
        try:
            if self._genai_diff_preview_window is None or not self._genai_diff_preview_window.winfo_exists():
                window = tk.Toplevel(self)
                window.title("Review Changes")
                window.geometry("980x700")
                window.transient(self)
                window.rowconfigure(0, weight=1)
                window.columnconfigure(0, weight=1)
                notebook = ttk.Notebook(window)
                notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 8))
                footer = ttk.Frame(window)
                footer.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
                footer.columnconfigure(0, weight=1)
                ttk.Button(footer, text="Close", command=window.withdraw).grid(row=0, column=1)
                ttk.Button(footer, text="Apply revisions", command=self._apply_pending_genai_revisions).grid(
                    row=0, column=2, padx=(6, 0)
                )
                window.protocol("WM_DELETE_WINDOW", window.withdraw)
                self._genai_diff_preview_window = window
                self._genai_diff_preview_notebook = notebook
                self._genai_diff_preview_texts = {}
            window = self._genai_diff_preview_window
            preview_notebook = self._genai_diff_preview_notebook
            if window is None or preview_notebook is None:
                return
            notebook = cast(ttk.Notebook, preview_notebook)
            for tab_id in notebook.tabs():
                notebook.forget(tab_id)
            self._genai_diff_preview_texts = {}

            tabs: list[tuple[str, str, str]] = []
            revised_model = str(pending.get("revised_model") or "")
            revised_data = str(pending.get("revised_data") or "")
            current_model = str(pending.get("current_model") or "")
            current_data = str(pending.get("current_data") or "")
            if revised_model:
                tabs.append(("Model", current_model, revised_model))
            if revised_data:
                tabs.append(("Data", current_data, revised_data))
            for label, original, revised in tabs:
                frame = ttk.Frame(notebook, padding=(0, 0, 0, 0))
                text_widget = self._create_diff_preview_text(frame)
                self._populate_diff_preview_text(text_widget, original, revised)
                notebook.add(frame, text=label)
                self._genai_diff_preview_texts[label.lower()] = text_widget
            try:
                window.deiconify()
                window.lift()
                window.focus_force()
            except Exception:
                pass
        except Exception:
            pass

    def _apply_pending_genai_revisions(self) -> None:
        """Apply pending Ask revisions from the docked GenAI panel."""
        pending = self._genai_pending_revisions
        if not pending:
            return

        revised_model = str(pending.get("revised_model") or "")
        revised_data = str(pending.get("revised_data") or "")
        model_path = str(pending.get("model_path") or self.model_file or "")
        data_path = str(pending.get("data_path") or self.data_file or "")
        safe_ts = str(pending.get("safe_ts") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        had_data_file = bool(pending.get("had_data_file"))
        session_id = pending.get("session_id")

        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        m_base_name, m_ext = os.path.splitext(os.path.basename(model_path))
        m_ext = m_ext or ".mod"

        def _strip_ts_suffix(name: str) -> str:
            match = re.match(r"^(.*?)(?:_\d{4}-\d{2}-\d{2}_\d{2}(?:-|_)\d{2}(?:-|_)\d{2})(?:_\d+)?$", name)
            return match.group(1) if match and match.group(1) else name

        m_base_name = _strip_ts_suffix(m_base_name)
        model_base = os.path.join(tmp_dir, f"{m_base_name}_{safe_ts}")
        model_tgt = model_base + m_ext
        i = 1
        while os.path.exists(model_tgt):
            model_tgt = f"{model_base}_{i}{m_ext}"
            i += 1

        model_content = revised_model if revised_model else self.model_text.get("1.0", tk.END)
        with open(model_tgt, "w", encoding="utf-8") as f:
            f.write(model_content)

        data_tgt = None
        if revised_data:
            if had_data_file:
                d_base_name, d_ext = os.path.splitext(os.path.basename(data_path))
                d_ext = d_ext or ".dat"
            else:
                d_base_name, d_ext = m_base_name, ".dat"
            d_base_name = _strip_ts_suffix(d_base_name)
            data_base = os.path.join(tmp_dir, f"{d_base_name}_{safe_ts}")
            data_tgt = data_base + d_ext
            j = 1
            while os.path.exists(data_tgt):
                data_tgt = f"{data_base}_{j}{d_ext}"
                j += 1
            with open(data_tgt, "w", encoding="utf-8") as f:
                f.write(revised_data)

        if revised_model:
            self.model_text.delete("1.0", tk.END)
            self.model_text.insert(tk.END, revised_model)
        if revised_data:
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, revised_data)

        self.model_file = model_tgt
        if data_tgt:
            self.data_file = data_tgt
        self._record_output_session_artifacts(
            session_id,
            model_text=self.model_text.get("1.0", tk.END).rstrip("\n"),
            data_text=self.data_text.get("1.0", tk.END).rstrip("\n"),
        )

        self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file or '')}")
        if data_tgt:
            current_data_file = self.data_file or ""
            self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(current_data_file)}")
        self.highlight(self.model_text, is_data=False)
        self.highlight(self.data_text, is_data=True)
        mark_baselines_saved = getattr(self, "_mark_editor_baselines_saved", None)
        if callable(mark_baselines_saved):
            mark_baselines_saved()
        self._append_output("\nRevisions applied to editors.\n", session_id)
        self.status_var.set("GenAI: revisions applied")
        self._clear_pending_genai_revisions()
        try:
            self._save_session()
        except Exception:
            pass

    def _get_active_editor(self) -> scrolledtext.ScrolledText:
        """Return the currently active editor widget (model or data)."""
        try:
            idx = self.editor_notebook.index(self.editor_notebook.select())
            return self.model_text if idx == 0 else self.data_text
        except Exception:
            return self.model_text

    def _open_find_replace_dialog(self, replace: bool = False) -> None:
        """Open a modal Find/Replace dialog with optional regex support."""
        if not self._ensure_no_active_operation("Find and Replace"):
            return
        dlg = tk.Toplevel(self)
        dlg.title("Find and Replace")
        dlg.transient(self)
        dlg.resizable(False, False)
        dlg.grab_set()

        # Allow undo/redo while the dialog is active by binding the
        # common shortcuts to the IDE handlers on the dialog. These
        # bindings run when focus is inside entry widgets.
        def _dlg_undo(ev: Optional[tk.Event] = None) -> str:
            self._undo()
            return "break"

        def _dlg_redo(ev: Optional[tk.Event] = None) -> str:
            self._redo()
            return "break"

        dlg.bind("<Control-z>", _dlg_undo)
        dlg.bind("<Control-Z>", _dlg_undo)
        dlg.bind("<Control-Shift-Z>", _dlg_redo)
        dlg.bind("<Control-y>", _dlg_redo)
        # macOS Command variants
        if sys.platform == "darwin":
            dlg.bind("<Command-z>", _dlg_undo)
            dlg.bind("<Command-Z>", _dlg_undo)
            dlg.bind("<Command-Shift-Z>", _dlg_redo)
            dlg.bind("<Command-y>", _dlg_redo)

        fg_var = tk.StringVar()
        rp_var = tk.StringVar()
        regex_var = tk.BooleanVar(value=False)
        case_var = tk.BooleanVar(value=False)

        ttk.Label(dlg, text="Find:").grid(row=0, column=0, sticky="w", padx=6, pady=(8, 2))
        find_entry = ttk.Entry(dlg, textvariable=fg_var, width=40)
        find_entry.grid(row=0, column=1, columnspan=3, padx=6, pady=(8, 2))

        ttk.Label(dlg, text="Replace:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        replace_entry = ttk.Entry(dlg, textvariable=rp_var, width=40)
        replace_entry.grid(row=1, column=1, columnspan=3, padx=6, pady=2)

        regex_cb = ttk.Checkbutton(dlg, text="Regex", variable=regex_var)
        regex_cb.grid(row=2, column=1, sticky="w", padx=6)
        case_cb = ttk.Checkbutton(dlg, text="Case sensitive", variable=case_var)
        case_cb.grid(row=2, column=2, sticky="w", padx=6)

        # Buttons
        def _highlight_match(start_idx: str, end_idx: str) -> None:
            w = self._get_active_editor()
            # Clear previous highlights and selection, then mark this match
            w.tag_remove("find_match", "1.0", tk.END)
            w.tag_remove("sel", "1.0", tk.END)
            w.tag_add("find_match", start_idx, end_idx)
            w.tag_add("sel", start_idx, end_idx)
            w.tag_configure("find_match", background="#ffe58f")
            w.mark_set("insert", end_idx)
            w.see(start_idx)

        def _find_next(ev: Optional[tk.Event] = None) -> str:
            pattern = find_entry.get()
            if not pattern:
                return "break"
            w = self._get_active_editor()
            text = w.get("1.0", "end-1c")
            # Choose a sensible start index: after current highlighted match,
            # or the widget's insert if it has focus, otherwise start of buffer.
            if w.tag_ranges("find_match"):
                start = w.index(w.tag_ranges("find_match")[1])
            elif w.focus_get() is w:
                start = w.index("insert")
            else:
                start = "1.0"
            # Convert start to integer offset (chars from 1.0)
            if regex_var.get():
                flags = 0 if case_var.get() else re.IGNORECASE
                try:
                    for m in re.finditer(pattern, text, flags):
                        s_off = m.start()
                        cnt = w.count("1.0", start, "chars")
                        if s_off >= (cnt[0] if cnt else 0):
                            # compute indices
                            s = w.index(f"1.0 + {m.start()} chars")
                            e = w.index(f"1.0 + {m.end()} chars")
                            _highlight_match(s, e)
                            return "break"
                except re.error:
                    messagebox.showerror("Regex error", "Invalid regular expression")
                    return "break"
            else:
                # use Tk text search
                if case_var.get():
                    res = w.search(pattern, start, tk.END)
                    if not res:
                        res = w.search(pattern, "1.0", start)
                else:
                    res = w.search(pattern, start, tk.END, nocase=True)
                    if not res:
                        res = w.search(pattern, "1.0", start, nocase=True)
                if res:
                    end = f"{res}+{len(pattern)}c"
                    _highlight_match(res, end)
            return "break"

        def _find_prev(ev: Optional[tk.Event] = None) -> str:
            # For simplicity, search from top up to current insert and pick last match
            pattern = find_entry.get()
            if not pattern:
                return "break"
            w = self._get_active_editor()
            text = w.get("1.0", "end-1c")
            # Determine the boundary offset.
            # Prefer the highlighted match start, then a selection start,
            # then the widget insert (if focused), else start of buffer.
            ranges = w.tag_ranges("find_match")
            if ranges:
                start_idx = str(ranges[0])
            else:
                if w.tag_ranges("sel"):
                    start_idx = str(w.index("sel.first"))
                elif w.focus_get() is w:
                    start_idx = str(w.index("insert"))
                else:
                    start_idx = "1.0"
            sc = w.count("1.0", start_idx, "chars")
            boundary_off = sc[0] if sc else 0

            matches: list[tuple[int, int]] = []
            if regex_var.get():
                flags = 0 if case_var.get() else re.IGNORECASE
                try:
                    for m in re.finditer(pattern, text, flags):
                        matches.append((m.start(), m.end()))
                except re.error:
                    messagebox.showerror("Regex error", "Invalid regular expression")
                    return "break"
            else:
                # Literal search; use re for case-insensitive, otherwise simple find loop
                if case_var.get():
                    start_pos = 0
                    while True:
                        idx = text.find(pattern, start_pos)
                        if idx == -1:
                            break
                        matches.append((idx, idx + len(pattern)))
                        start_pos = idx + max(1, len(pattern))
                else:
                    esc = re.escape(pattern)
                    for m in re.finditer(esc, text, re.IGNORECASE):
                        matches.append((m.start(), m.end()))

            # Pick the last match strictly before boundary_off; wrap to last match if none
            prev_match: Optional[tuple[int, int]] = None
            for s_off, e_off in matches:
                if s_off < boundary_off:
                    prev_match = (s_off, e_off)
                else:
                    break
            if prev_match is None and matches:
                prev_match = matches[-1]

            if prev_match:
                s_idx = w.index(f"1.0 + {prev_match[0]} chars")
                e_idx = w.index(f"1.0 + {prev_match[1]} chars")
                _highlight_match(s_idx, e_idx)
            return "break"

        def _replace_one() -> None:
            w = self._get_active_editor()
            try:
                sel_start = w.index("sel.first")
                sel_end = w.index("sel.last")
            except Exception:
                # if nothing selected, find next and then replace
                _find_next()
                try:
                    sel_start = w.index("sel.first")
                    sel_end = w.index("sel.last")
                except Exception:
                    # Fallback: use the currently highlighted match if present
                    ranges = w.tag_ranges("find_match")
                    if ranges:
                        sel_start = str(ranges[0])
                        sel_end = str(ranges[1])
                    else:
                        return
            replacement = replace_entry.get()
            if regex_var.get():
                # compute absolute offsets
                # Replace only the selected region using Text.replace so undo groups correctly
                segment = w.get(sel_start, sel_end)
                try:
                    new_segment = re.sub(find_entry.get(), replacement, segment, count=1)
                except re.error:
                    messagebox.showerror("Regex error", "Invalid regular expression")
                    return
                w.replace(sel_start, sel_end, new_segment)
            else:
                # Use Text.replace to make this a single undoable action
                w.replace(sel_start, sel_end, replacement)
            w.tag_remove("find_match", "1.0", tk.END)
            w.tag_remove("sel", "1.0", tk.END)
            # Notify change so UI state (dirty flag) updates
            try:
                self._on_text_change(w, is_data=(w is self.data_text))
            except Exception:
                pass
            w.tag_remove("sel", "1.0", tk.END)

        def _replace_all() -> None:
            w = self._get_active_editor()
            full = w.get("1.0", "end-1c")
            if regex_var.get():
                try:
                    flags = 0 if case_var.get() else re.IGNORECASE
                    new = re.sub(find_entry.get(), replace_entry.get(), full, flags=flags)
                except re.error:
                    messagebox.showerror("Regex error", "Invalid regular expression")
                    return
            else:
                if case_var.get():
                    new = full.replace(find_entry.get(), replace_entry.get())
                else:
                    # case-insensitive replace: do manual loop
                    pat = find_entry.get()
                    if not pat:
                        return
                    new = re.sub(re.escape(pat), lambda m: replace_entry.get(), full, flags=re.IGNORECASE)
            # Replace entire buffer in one operation so undo/redo works as expected
            w.replace("1.0", "end-1c", new)
            w.tag_remove("find_match", "1.0", tk.END)
            w.tag_remove("sel", "1.0", tk.END)
            try:
                self._on_text_change(w, is_data=(w is self.data_text))
            except Exception:
                pass
            w.tag_remove("sel", "1.0", tk.END)

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=8)
        ttk.Button(btn_frame, text="Prev", command=_find_prev).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Next", command=_find_next).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Replace", command=_replace_one).grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text="Replace All", command=_replace_all).grid(row=0, column=3, padx=4)
        ttk.Button(btn_frame, text="Close", command=dlg.destroy).grid(row=0, column=4, padx=4)

        # Keyboard bindings inside dialog
        dlg.bind("<Return>", _find_next)
        dlg.bind("<Shift-Return>", _find_prev)
        find_entry.focus_set()

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
            self._populate_request_context_menu(self._get_selected_request_session_id())
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
            active = getattr(self, "_active_operation", None)
            if active is not None and active.session_id == sid:
                try:
                    messagebox.showinfo(
                        "Delete Session",
                        "The active output session cannot be deleted while its operation is running.",
                    )
                except Exception:
                    pass
                return
            if not messagebox.askyesno("Delete Session", "Delete the selected session?"):
                return

            # Remove data
            self._output_session_ids.pop(index)
            self._output_sessions.pop(sid, None)
            self._output_session_display.pop(sid, None)
            self._output_session_label.pop(sid, None)
            self._output_session_timestamp.pop(sid, None)
            self._output_session_artifacts.pop(sid, None)

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

    def _rename_selected_request(self) -> None:
        """Rename the currently selected output session."""
        session_id = self._get_selected_request_session_id()
        if not session_id:
            return

        timestamp = self._output_session_timestamp.get(session_id, session_id)
        current_label = self._output_session_label.get(session_id, "")
        if not current_label:
            display = self._output_session_display.get(session_id, "")
            prefix = f"{timestamp} • "
            current_label = display[len(prefix) :] if display.startswith(prefix) else (display or "Session")

        new_label = self._ask_short_text(
            title="Change label",
            prompt="Enter a short label (max 50 characters):",
            initial_text=current_label,
        )
        if new_label is None:
            return

        new_label = str(new_label).strip()
        if not new_label:
            messagebox.showerror("Change label", "The session label cannot be empty.")
            return
        if len(new_label) > 50:
            messagebox.showerror("Change label", "The session label cannot exceed 50 characters.")
            return

        display = self._make_output_session_display(timestamp, new_label, exclude_session_id=session_id)
        self._output_session_label[session_id] = new_label
        self._output_session_display[session_id] = display

        try:
            index = self._output_session_ids.index(session_id)
        except ValueError:
            return

        try:
            self.request_listbox.delete(index)
            self.request_listbox.insert(index, display)
            self.request_listbox.selection_clear(0, tk.END)
            self.request_listbox.selection_set(index)
            self.request_listbox.activate(index)
        except Exception:
            pass

        try:
            self._save_session()
        except Exception:
            pass

    def _ask_short_text(self, title: str, prompt: str, initial_text: str = "") -> Optional[str]:
        """Show a themed short-text dialog and return the entered text or None."""
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text=prompt, anchor="w", style="TLabel").grid(row=0, column=0, sticky="ew", pady=(0, 6))

        text_var = tk.StringVar(value=initial_text)
        entry = ttk.Entry(frm, textvariable=text_var, width=48)
        entry.grid(row=1, column=0, sticky="ew")
        entry.focus_set()
        entry.selection_range(0, tk.END)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(10, 0))
        result: dict[str, Optional[str]] = {"value": None}

        def on_ok(event: Optional[tk.Event] = None) -> None:
            result["value"] = text_var.get()
            dlg.destroy()

        def on_cancel(event: Optional[tk.Event] = None) -> None:
            result["value"] = None
            dlg.destroy()

        ok_btn = ttk.Button(btns, text="OK", command=on_ok)
        cancel_btn = ttk.Button(btns, text="Cancel", command=on_cancel)
        cancel_btn.grid(row=0, column=1, padx=(6, 0))
        ok_btn.grid(row=0, column=0)

        dlg.bind("<Return>", on_ok)
        dlg.bind("<Escape>", on_cancel)
        dlg.wait_window()
        return result["value"]

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

    def _text_too_large_for_highlight(self, text_widget: tk.Text) -> bool:
        try:
            count = text_widget.count("1.0", "end-1c", "chars")
            if count:
                return int(count[0]) > MAX_HIGHLIGHT_CHARS
        except Exception:
            pass
        try:
            return len(text_widget.get("1.0", "end-1c")) > MAX_HIGHLIGHT_CHARS
        except Exception:
            return False

    def _clear_highlight_tags(self, text_widget: tk.Text) -> None:
        for previous_tag in TOKEN_COLORS.keys():
            try:
                text_widget.tag_remove(previous_tag, "1.0", tk.END)
            except Exception:
                pass
        try:
            text_widget.tag_remove("ERROR", "1.0", tk.END)
        except Exception:
            pass

    def _disable_highlight_for_large_text(self, text_widget: tk.Text) -> None:
        self._clear_highlight_tags(text_widget)
        self._last_syntax_error_by_widget[id(text_widget)] = None
        self._last_syntax_error = None
        try:
            self.status_syntax_var.set("Syntax validation disabled for large text")
        except Exception:
            pass

    def _schedule_highlight(self, text_widget: tk.Text, is_data: bool) -> None:
        """Debounce highlight work to keep typing responsive."""
        if getattr(self, "_shutting_down", False):
            return

        # Cancel any pending runs for this widget
        self._cancel_scheduled_highlight(text_widget, "fast")
        self._cancel_scheduled_highlight(text_widget, "validate")

        if self._text_too_large_for_highlight(text_widget):
            self._disable_highlight_for_large_text(text_widget)
            return

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
        if not self._ensure_no_active_operation("Open Model"):
            return
        fname = filedialog.askopenfilename(filetypes=[("Model files", "*.mod"), ("All files", "*.*")])
        if fname:
            with open(fname, "r", encoding="utf-8") as f:
                self.model_text.delete(1.0, tk.END)
                self.model_text.insert(tk.END, f.read())
            self.model_file = fname
            self._model_saved_text = self._get_editor_text(self.model_text)
            self.highlight(self.model_text)
            self._update_caret_position(self.model_text)

            # Update tab label and switch to Model tab
            self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(fname)}")
            self.editor_notebook.select(self.model_frame)
            self.on_tab_changed(None)

    def open_data(self) -> None:
        """Open a data file into the data editor."""
        if not self._ensure_no_active_operation("Open Data"):
            return
        fname = filedialog.askopenfilename(filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if fname:
            with open(fname, "r", encoding="utf-8") as f:
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, f.read())
            self.data_file = fname
            self._data_saved_text = self._get_editor_text(self.data_text)
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
        with open(self.model_file, "w", encoding="utf-8") as f:
            f.write(content)
        self._model_saved_text = self._get_editor_text(self.model_text)
        # Update tab title
        try:
            self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file or '')}")
        except Exception:
            pass
        try:
            self._save_session()
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
        with open(self.data_file, "w", encoding="utf-8") as f:
            f.write(content)
        self._data_saved_text = self._get_editor_text(self.data_text)
        # Update tab title
        try:
            self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file)}")
        except Exception:
            pass
        try:
            self._save_session()
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
        with open(self.model_file, "w", encoding="utf-8") as f:
            f.write(content)
        self._model_saved_text = self._get_editor_text(self.model_text)
        self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file or '')}")
        try:
            self._save_session()
        except Exception:
            pass

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
        with open(self.data_file, "w", encoding="utf-8") as f:
            f.write(content)
        self._data_saved_text = self._get_editor_text(self.data_text)
        self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file)}")
        try:
            self._save_session()
        except Exception:
            pass

    # --- Syntax Highlighting ---
    @staticmethod
    def _is_transient_live_parse_error(exc: Exception) -> bool:
        """Return True for parser errors that are expected while typing incomplete text."""
        message = str(exc).splitlines()[0] if str(exc) else ""
        return "end of file (EOF)" in message

    def highlight(self, text_widget: tk.Text, is_data: bool = False, validate: bool = True) -> None:
        """Apply syntax highlighting to the given text widget."""
        if (not is_data) and (not validate):
            return

        if self._text_too_large_for_highlight(text_widget):
            self._disable_highlight_for_large_text(text_widget)
            return

        # Remove previous tags
        self._clear_highlight_tags(text_widget)

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
            lexer_error: Exception | None = None
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
                    if not self._is_transient_live_parse_error(e):
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
        if hasattr(self, "genai_prompt_text"):
            self.genai_prompt_text.config(font=editor_font)

        # Adjust comment tag to match new size
        self.model_text.tag_configure("COMMENT", font=(self.editor_font_family, size, "italic"))

        # Update caret position after size change
        self._update_caret_position(self.model_text)
        self._apply_theme_colors()
        self._sync_genai_mode_width()

        # Persist settings
        self._save_settings()

    # --- Status Bar ---
    def _update_caret_position(self, text_widget: tk.Text) -> None:
        """
        Update status bar with current caret position. If a syntax error is present,
        display its line alongside the caret position.
        """

        def _strip_hint(message: Optional[str]) -> Optional[str]:
            if not message:
                return message
            text = str(message).strip()
            for marker in (" Hint:", "\nHint:"):
                hint_index = text.find(marker)
                if hint_index != -1:
                    return text[:hint_index].rstrip()
            return text

        if text_widget.winfo_exists():
            try:
                index = text_widget.index(tk.INSERT)
                index_str = str(index)
                if "." in index_str:
                    caret_line, caret_col = map(int, index_str.split("."))
                else:
                    caret_line, caret_col = 1, 0

                caret_msg = f"Ln {caret_line}, Col {caret_col}"
                self.status_caret_var.set(caret_msg)
                self._refresh_status_context()

                if self._text_too_large_for_highlight(text_widget):
                    self.status_syntax_var.set("Syntax validation disabled for large text")
                    return

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
                        error_msg = _strip_hint(last_error)
                    else:
                        error_msg = f"Syntax Error on line {caret_line}"
                elif error_lines:
                    first_err_line = error_lines[0]
                    if last_error and f"line {first_err_line}" in last_error:
                        error_msg = _strip_hint(last_error)
                    else:
                        error_msg = f"Syntax Error on line {first_err_line}"
                elif last_error:
                    error_msg = _strip_hint(last_error)

                if error_msg:
                    self.status_syntax_var.set(error_msg)
                else:
                    self.status_syntax_var.set("Syntax OK")

            except tk.TclError:
                self.status_syntax_var.set("Syntax OK")
                self.status_caret_var.set("Ln 1, Col 0")
            except Exception as e:
                self.status_syntax_var.set("Status error")
                self.status_caret_var.set("Ln ?, Col ?")
                self.status_var.set(f"Error updating status: {e}")
        else:
            self.status_syntax_var.set("Syntax OK")
            self.status_caret_var.set("Ln 1, Col 0")

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

    def _start_run_timer(self, base_msg: str = "Solving model...") -> None:
        """Start updating the status bar with elapsed solve time (every second)."""
        self._stop_run_timer()
        self._run_status_base = base_msg
        self.status_runtime_var.set("Elapsed 00:00:00")
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
        if hasattr(self, "status_runtime_var"):
            self.status_runtime_var.set("")

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

        self.status_runtime_var.set(f"Elapsed {t}")

        # Reschedule
        self._run_timer_after_id = self.after(1000, self._tick_run_timer)

    def _solver_tracks_progress(self, solver_choice: Optional[str] = None) -> bool:
        solver_name = solver_choice or getattr(self, "_current_solver_choice", "gurobi")
        return str(solver_name).lower() == "gurobi"

    def _display_solver_progress_enabled(self) -> bool:
        progress_var = getattr(self, "display_solver_progress_var", None)
        if progress_var is None:
            return True
        try:
            return bool(progress_var.get())
        except Exception:
            return True

    def _hide_solver_progress_window(self) -> None:
        if self._solver_progress_update_after_id:
            try:
                self.after_cancel(self._solver_progress_update_after_id)
            except Exception:
                pass
            self._solver_progress_update_after_id = None
        self._solver_progress_pending_sample = None
        if self._solver_progress_window is not None:
            try:
                if self._solver_progress_window.winfo_exists():
                    self._solver_progress_window.withdraw()
            except Exception:
                pass

    def _on_display_solver_progress_toggled(self) -> None:
        self._save_settings()
        if not self._display_solver_progress_enabled():
            self._hide_solver_progress_window()

    def _reset_solver_progress_window(self, solver_choice: str) -> None:
        """Create or reset the solve progress window."""
        if not self._display_solver_progress_enabled():
            self._hide_solver_progress_window()
            return
        if self._solver_progress_update_after_id:
            try:
                self.after_cancel(self._solver_progress_update_after_id)
            except Exception:
                pass
        self._solver_progress_update_after_id = None
        self._solver_progress_samples = []
        self._solver_progress_pending_sample = None
        if self._solver_progress_window is None or not self._solver_progress_window.winfo_exists():
            window = tk.Toplevel(self)
            window.title("Solve Progress")
            window.geometry("760x460")
            window.transient(self)
            window.rowconfigure(1, weight=1)
            window.columnconfigure(0, weight=1)
            window.protocol("WM_DELETE_WINDOW", window.withdraw)

            header = ttk.Frame(window, padding=(12, 10, 12, 4))
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(0, weight=1)
            status_var = tk.StringVar(value="Waiting for solver progress...")
            ttk.Label(header, text="Solve Progress", font=(self.interface_font_family, 13, "bold")).grid(
                row=0, column=0, sticky="w"
            )
            ttk.Label(header, textvariable=status_var).grid(row=1, column=0, sticky="w", pady=(2, 0))

            canvas = tk.Canvas(window, height=260, highlightthickness=1, highlightbackground="#d6d8dc", bg="#ffffff")
            canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 8))
            canvas.bind("<Configure>", lambda _event: self._redraw_solver_progress_chart())

            stats_frame = ttk.Frame(window, padding=(12, 0, 12, 12))
            stats_frame.grid(row=2, column=0, sticky="ew")
            for col in range(4):
                stats_frame.columnconfigure(col, weight=1)

            self._solver_progress_window = window
            self._solver_progress_canvas = canvas
            self._solver_progress_stats_frame = stats_frame
            self._solver_progress_status_var = status_var

        chart_visible = self._solver_tracks_progress(solver_choice)
        if self._solver_progress_window is not None:
            try:
                self._solver_progress_window.geometry("760x460" if chart_visible else "520x220")
                self._solver_progress_window.rowconfigure(1, weight=1 if chart_visible else 0)
            except Exception:
                pass
        if self._solver_progress_canvas is not None:
            try:
                if chart_visible:
                    self._solver_progress_canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 8))
                else:
                    self._solver_progress_canvas.grid_remove()
            except Exception:
                pass

        self._solver_progress_stat_vars = {}
        if self._solver_progress_stats_frame is not None:
            for child in self._solver_progress_stats_frame.winfo_children():
                child.destroy()

        self._set_solver_progress_status(f"{solver_choice}: solving...")
        if chart_visible:
            self._redraw_solver_progress_chart()
        try:
            self._solver_progress_window.deiconify()
            self._solver_progress_window.lift()
        except Exception:
            pass

    def _set_solver_progress_status(self, text: str) -> None:
        if self._solver_progress_status_var is not None:
            self._solver_progress_status_var.set(text)

    def _format_progress_value(self, value: Any) -> str:
        try:
            number = float(value)
            if not math.isfinite(number):
                return "-"
            if abs(number) >= 1_000_000 or (0 < abs(number) < 0.001):
                return f"{number:.3g}"
            return f"{number:.6g}"
        except Exception:
            return "-" if value is None else str(value)

    def _record_solver_progress(self, event: dict[str, Any]) -> None:
        if not self._display_solver_progress_enabled():
            return
        if not self._solver_tracks_progress():
            return
        if not isinstance(event, dict):
            return
        sample = dict(event)
        lb = sample.get("lower_bound")
        ub = sample.get("upper_bound")
        try:
            if lb is None or ub is None:
                return
            lb_float = float(lb)
            ub_float = float(ub)
            if math.isfinite(lb_float) and math.isfinite(ub_float) and ub_float != 0:
                sample["gap"] = abs(ub_float - lb_float) / max(1.0, abs(ub_float))
        except Exception:
            pass
        self._solver_progress_pending_sample = sample
        if self._solver_progress_update_after_id is None:
            self._solver_progress_update_after_id = self.after(1000, self._flush_solver_progress_update)

    def _flush_solver_progress_update(self) -> None:
        self._solver_progress_update_after_id = None
        sample = self._solver_progress_pending_sample
        self._solver_progress_pending_sample = None
        if sample is None:
            return
        self._append_solver_progress_sample(sample)

    def _progress_sample_time(self, sample: dict[str, Any]) -> Optional[float]:
        for key in ("runtime", "time"):
            try:
                raw_value = sample.get(key)
                if raw_value is None:
                    continue
                value = float(raw_value)
            except Exception:
                continue
            if math.isfinite(value):
                return value
        return None

    def _trim_solver_progress_samples(self) -> None:
        latest = None
        for sample in reversed(self._solver_progress_samples):
            latest = self._progress_sample_time(sample)
            if latest is not None:
                break
        if latest is None:
            self._solver_progress_samples = self._solver_progress_samples[-120:]
            return

        cutoff = latest - self._solver_progress_rolling_seconds
        trimmed: list[dict[str, Any]] = []
        for sample in self._solver_progress_samples:
            sample_time = self._progress_sample_time(sample)
            if sample_time is None or sample_time >= cutoff:
                trimmed.append(sample)
        self._solver_progress_samples = trimmed[-180:]

    def _append_solver_progress_sample(self, sample: dict[str, Any]) -> None:
        self._solver_progress_samples.append(sample)
        self._trim_solver_progress_samples()
        self._update_solver_progress_stats(sample)
        self._redraw_solver_progress_chart()

    def _solver_progress_stats(self, sample: dict[str, Any]) -> list[tuple[str, Any]]:
        if self._solver_tracks_progress():
            return [
                ("LB", sample.get("lower_bound")),
                ("UB", sample.get("upper_bound")),
                ("Gap", sample.get("gap")),
                ("Nodes", sample.get("nodes")),
                ("Solutions", sample.get("solutions")),
                ("Runtime", sample.get("runtime")),
            ]

        stats = [
            ("Objective", sample.get("objective_value", sample.get("upper_bound"))),
            ("Runtime", sample.get("runtime")),
            ("Iterations", sample.get("iterations")),
        ]
        return [(label, value) for label, value in stats if value is not None]

    def _update_solver_progress_stats(self, sample: dict[str, Any]) -> None:
        stats = self._solver_progress_stats(sample)
        if self._solver_progress_stats_frame is None:
            return
        for index, (label, value) in enumerate(stats):
            var = self._solver_progress_stat_vars.get(label)
            if var is None:
                var = tk.StringVar(value="-")
                self._solver_progress_stat_vars[label] = var
                frame = ttk.Frame(self._solver_progress_stats_frame)
                frame.grid(row=index // 4, column=index % 4, sticky="ew", padx=(0, 12), pady=(0, 8))
                ttk.Label(frame, text=label, font=(self.interface_font_family, 9, "bold")).grid(row=0, column=0, sticky="w")
                ttk.Label(frame, textvariable=var).grid(row=1, column=0, sticky="w")
            if label == "Gap" and value is not None:
                try:
                    var.set(f"{float(value) * 100:.4g}%")
                    continue
                except Exception:
                    pass
            elif label == "Runtime" and value is not None:
                try:
                    var.set(f"{float(value):.2f}s")
                    continue
                except Exception:
                    pass
            var.set(self._format_progress_value(value))

    def _finish_solver_progress(self, results: Optional[dict[str, Any]] = None, status: str = "complete") -> None:
        if not self._display_solver_progress_enabled():
            self._hide_solver_progress_window()
            return
        if not self._solver_tracks_progress():
            self._reset_solver_progress_window(getattr(self, "_current_solver_choice", "scipy"))
        if isinstance(results, dict):
            sample: dict[str, Any] = {}
            objective = results.get("objective_value")
            if objective is not None:
                sample["objective_value"] = objective
                sample["upper_bound"] = objective
                sample["lower_bound"] = objective
                sample["gap"] = 0.0
            stats = results.get("stats")
            if isinstance(stats, dict):
                sample.update(
                    {
                        "gap": stats.get("MIPGap", sample.get("gap")),
                        "runtime": stats.get("Runtime", stats.get("time")),
                        "nodes": stats.get("NodeCount"),
                        "iterations": stats.get("IterCount", stats.get("nit")),
                    }
                )
            if sample:
                self._update_solver_progress_stats(sample)
                if sample.get("lower_bound") is not None and sample.get("upper_bound") is not None:
                    self._append_solver_progress_sample(sample)
        if self._solver_progress_update_after_id:
            try:
                self.after_cancel(self._solver_progress_update_after_id)
            except Exception:
                pass
            self._solver_progress_update_after_id = None
        self._solver_progress_pending_sample = None
        self._set_solver_progress_status(f"Solve {status}.")

    def _redraw_solver_progress_chart(self) -> None:
        if not self._display_solver_progress_enabled():
            return
        if not self._solver_tracks_progress():
            return
        canvas = self._solver_progress_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        canvas.delete("all")
        width = max(200, canvas.winfo_width())
        height = max(160, canvas.winfo_height())
        pad_left, pad_right, pad_top, pad_bottom = 56, 20, 24, 38
        plot_w = max(1, width - pad_left - pad_right)
        plot_h = max(1, height - pad_top - pad_bottom)
        canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_h, fill="#6b7280")
        canvas.create_line(pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, fill="#6b7280")
        canvas.create_text(pad_left, 10, text="LB / UB", anchor="w", fill="#374151")
        canvas.create_text(
            pad_left + plot_w,
            height - 12,
            text=f"last {int(self._solver_progress_rolling_seconds)}s",
            anchor="e",
            fill="#374151",
        )

        samples = [
            s for s in self._solver_progress_samples if s.get("lower_bound") is not None and s.get("upper_bound") is not None
        ]
        values: list[float] = []
        clean_samples: list[tuple[float, float]] = []
        for sample in samples:
            try:
                lower_bound = sample.get("lower_bound")
                upper_bound = sample.get("upper_bound")
                if lower_bound is None or upper_bound is None:
                    continue
                lb = float(lower_bound)
                ub = float(upper_bound)
            except Exception:
                continue
            if not (math.isfinite(lb) and math.isfinite(ub)):
                continue
            clean_samples.append((lb, ub))
            values.extend([lb, ub])
        if not clean_samples:
            canvas.create_text(width / 2, height / 2, text="Waiting for LB/UB samples...", fill="#6b7280")
            return

        min_y = min(values)
        max_y = max(values)
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0
        span = max_y - min_y

        def point(index: int, value: float) -> tuple[float, float]:
            x = pad_left + (plot_w * index / max(1, len(clean_samples) - 1))
            y = pad_top + plot_h - ((value - min_y) / span * plot_h)
            return x, y

        lb_points: list[float] = []
        ub_points: list[float] = []
        for index, (lb, ub) in enumerate(clean_samples):
            lb_points.extend(point(index, lb))
            ub_points.extend(point(index, ub))
        if len(lb_points) >= 4:
            canvas.create_line(*lb_points, fill="#2563eb", width=2, smooth=True)
            canvas.create_line(*ub_points, fill="#dc2626", width=2, smooth=True)
        else:
            x, y = lb_points
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#2563eb", outline="")
            x, y = ub_points
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#dc2626", outline="")
        canvas.create_text(width - 88, 18, text="LB", fill="#2563eb", anchor="w")
        canvas.create_line(width - 115, 18, width - 94, 18, fill="#2563eb", width=2)
        canvas.create_text(width - 44, 18, text="UB", fill="#dc2626", anchor="w")
        canvas.create_line(width - 71, 18, width - 50, 18, fill="#dc2626", width=2)

    # --- Model Execution ---
    def run_model(
        self,
        explain_after_solve: bool = False,
        model_file_override: Optional[str] = None,
        data_file_override: Optional[str] = None,
    ) -> None:
        """Run the model using current editor contents, checking data file presence and validity."""
        if self._solver_process and self._solver_process.is_alive():
            messagebox.showinfo("Solve Model", "Model is already running.")
            return

        import re

        model_code = self.model_text.get(1.0, tk.END).rstrip("\n")
        data_code = self.data_text.get(1.0, tk.END).rstrip("\n")

        solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"
        operation = self._start_foreground_operation(
            kind="solve",
            label="Solve Model",
            header="Solve: Solving model...",
            status="Solving model...",
            solver_choice=solver_choice,
            model_file=model_file_override,
            data_file=data_file_override,
            explain_after_solve=explain_after_solve,
        )
        if operation is None:
            return

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
                self._append_output(f"\nError: Data file failed to parse: {e}\n", operation.session_id)
                self._finish_foreground_operation(operation)
                return

        # Check that all required data variables are present
        missing_vars = []
        for var in data_vars:
            if not re.search(r"\b" + re.escape(var) + r"\s*(=|\[)", data_code):
                missing_vars.append(var)
        if missing_vars:
            self.status_var.set(f"Error: Data missing for: {', '.join(missing_vars)}")
            self._append_output(f"\nError: Data missing for: {', '.join(missing_vars)}\n", operation.session_id)
            self._finish_foreground_operation(operation)
            return

        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Save temp files if not saved
        model_file = model_file_override or self.model_file or os.path.join(tmp_dir, "temp_model.mod")
        data_file = data_file_override or self.data_file or os.path.join(tmp_dir, "temp_data.dat")
        try:
            with open(model_file, "w", encoding="utf-8") as f:
                f.write(model_code)
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(data_code)
        except Exception as e:
            self.status_var.set(f"Error saving temp files: {e}")
            self._append_output(f"\nError saving temp files: {e}\n", operation.session_id)
            self._finish_foreground_operation(operation)
            return

        self._current_solver_choice = solver_choice
        operation.model_file = model_file
        operation.data_file = data_file
        try:
            self._solver_queue = multiprocessing.Queue()
            self._solver_process = multiprocessing.Process(
                target=_solve_wrapper,
                args=(model_file, data_file, solver_choice, self._solver_queue),
            )
            self._solver_process.start()
        except Exception as e:
            self.status_var.set(f"Error starting solve: {e}")
            self._append_output(f"\nError starting solve: {e}\n", operation.session_id)
            self._solver_process = None
            self._solver_queue = None
            self._finish_foreground_operation(operation)
            return

        self._set_run_menu_running(True)
        self._show_solver_log_textbox()

        if self._display_solver_progress_enabled() and self._solver_tracks_progress(solver_choice):
            self._reset_solver_progress_window(solver_choice)

        # Start elapsed-time status updates (every second)
        self._start_run_timer("Solving model...")

        self.after(100, self._poll_solver, operation)

    def stop_model(self) -> None:
        p = self._solver_process
        active = getattr(self, "_active_operation", None)

        if not (p and p.is_alive()):
            self._cleanup_solver_ipc(cancel_queue_thread=True)
            self._stop_run_timer()
            self._restore_output_textbox()
            self._set_run_menu_running(False)
            if active is not None and active.kind == "solve":
                self._finish_foreground_operation(active)
            return

        try:
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive() and hasattr(p, "kill"):
                p.kill()  # py3.7+ on Unix
                p.join(timeout=1.0)
        except Exception:
            pass

        self._cleanup_solver_ipc(cancel_queue_thread=True)

        # Stop timer updates
        self._stop_run_timer()

        self._restore_output_textbox()
        self._append_output("\nExecution stopped by user.\n", active.session_id if active is not None else None)
        self.status_var.set("Execution stopped.")
        self._finish_solver_progress(status="stopped")
        self._set_run_menu_running(False)
        self._finish_foreground_operation(active)

    def _cleanup_solver_ipc(self, *, cancel_queue_thread: bool) -> None:
        """Release solver process/queue resources to avoid multiprocessing tracker leaks."""
        q = self._solver_queue
        self._solver_process = None
        self._solver_queue = None
        if q is None:
            return
        try:
            q.close()
        except Exception:
            pass
        try:
            if cancel_queue_thread:
                q.cancel_join_thread()
            else:
                q.join_thread()
        except Exception:
            pass

    def _poll_solver(self, operation: Optional[_ForegroundOperation] = None) -> None:
        # _poll_solver is scheduled via `after()`, so it can run after `stop_model()`
        # has already nulled these out.
        p = self._solver_process
        q = self._solver_queue
        if not p or not q:
            return

        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                if p.is_alive():
                    self.after(100, self._poll_solver, operation)
                    return

                # Process ended but no terminal message
                self._set_run_menu_running(False)
                self._restore_output_textbox()
                self._append_output(
                    "\nError: Solver process terminated unexpectedly.\n",
                    operation.session_id if operation is not None else None,
                )

                # Stop timer updates
                self._stop_run_timer()

                self.status_var.set("Error: Solver process terminated.")
                self._finish_solver_progress(status="ended unexpectedly")
                self._cleanup_solver_ipc(cancel_queue_thread=True)
                self._finish_foreground_operation(operation)
                return
            if kind == "progress":
                if isinstance(payload, dict):
                    self._record_solver_progress(payload)
                continue
            if kind == "log":
                self._append_solver_log_text(str(payload))
                continue
            break

        # Got a message => process should be done
        try:
            p.join(timeout=0.1)
        except Exception:
            pass

        self._cleanup_solver_ipc(cancel_queue_thread=False)
        self._set_run_menu_running(False)

        # Stop timer updates
        self._stop_run_timer()
        self._restore_output_textbox()

        if kind == "success":
            self._finish_solver_progress(payload if isinstance(payload, dict) else None, status="complete")
            self._display_solve_results(
                payload,
                session_id=operation.session_id if operation is not None else None,
                solver_choice=operation.solver_choice if operation is not None else None,
            )

            # Determine whether the solver actually produced a usable solution.
            success = False
            try:
                if isinstance(payload, dict):
                    # Prefer an explicit solution payload
                    if payload.get("solution"):
                        success = True
                    else:
                        st = str(payload.get("status", "")).lower()
                        if st in ("optimal", "feasible", "optimal solution", "feasible solution", "success"):
                            success = True
            except Exception:
                success = False

            # If user requested Solve & Explain, only run generative feedback when solve succeeded
            try:
                if operation is not None and operation.explain_after_solve:
                    if not success:
                        # Skip explanation when solve failed
                        try:
                            self._append_output(
                                "\n[GenAI] Skipping explanation because solve did not produce a successful solution.\n",
                                operation.session_id,
                            )
                        except Exception:
                            pass
                        self._finish_foreground_operation(operation)
                    else:
                        # run feedback in background thread to avoid blocking UI
                        def _run_feedback():
                            try:
                                if operation.cancel_requested:
                                    return
                                # Compose solution text as JSON
                                try:
                                    sol_text = json.dumps(payload, indent=2, sort_keys=True, default=str)
                                except Exception:
                                    sol_text = str(payload)

                                feedback_prompt = (
                                    "Translate the following optimization solution into clear, non-technical language targeting a lay user. "
                                    "Include key findings and suggested next steps.\n\nSolution:\n" + sol_text
                                )

                                model_path = operation.model_file
                                data_path = operation.data_file

                                # Notify UI and request feedback
                                if operation.cancel_requested:
                                    return
                                self.after(
                                    0, self._append_output, "\n[GenAI] Requesting explanation...\n", operation.session_id
                                )

                                try:
                                    fb = generative_feedback(
                                        feedback_prompt,
                                        model_path,
                                        data_path,
                                        llm_provider=(self.genai_provider if self.genai_provider else None),
                                        model_name=(self.genai_model if self.genai_model else None),
                                        progress=(None),
                                    )
                                except Exception:
                                    self.after(
                                        0,
                                        self._append_output,
                                        "\n[GenAI] Error requesting explanation:\n",
                                        operation.session_id,
                                    )
                                    self.after(0, self._finish_foreground_operation, operation)
                                    return

                                if operation.cancel_requested:
                                    self.after(0, self._finish_foreground_operation, operation)
                                    return

                                # Format feedback for output
                                try:
                                    if isinstance(fb, dict):
                                        out = fb.get("feedback") or json.dumps(fb, indent=2, sort_keys=True, default=str)
                                    else:
                                        out = str(fb)
                                except Exception:
                                    out = str(fb)

                                # Note: unescaping of double-escaped sequences is handled centrally
                                # in `pyopl.genai.pyopl_generative.generative_feedback`.

                                self.after(
                                    0, self._append_output, "\n[GenAI] Explanation:\n" + out + "\n", operation.session_id
                                )
                                self.after(0, lambda: self.status_var.set("GenAI: explanation complete"))
                                self.after(0, self._finish_foreground_operation, operation)
                            except Exception:
                                try:
                                    self.after(0, self._append_output, "\n[GenAI] Explanation failed.\n", operation.session_id)
                                except Exception:
                                    pass
                                self.after(0, self._finish_foreground_operation, operation)

                        threading.Thread(target=_run_feedback, daemon=True).start()
                        return
            except Exception:
                pass
            self._finish_foreground_operation(operation)
        else:
            self._finish_solver_progress(status="failed")
            self._append_output(f"\nError:\n{payload}\n", operation.session_id if operation is not None else None)
            self.status_var.set("Error running model")
            self._finish_foreground_operation(operation)

    def _display_solve_results(
        self,
        results: dict,
        session_id: Optional[str] = None,
        solver_choice: Optional[str] = None,
    ) -> None:
        """Format and display solver results in the output pane."""
        solver_choice = solver_choice or getattr(self, "_current_solver_choice", "gurobi")
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

        self._append_output("".join(buf), session_id)
        msg = results.get("message") or results.get("status", "Done")
        self.status_var.set(msg)

    def export_model(self) -> None:
        """Export the current model as Python, LP, or MPS using the selected solver/lowering."""
        if not self._ensure_no_active_operation("Export model"):
            return
        try:
            model_code = self.model_text.get(1.0, tk.END).rstrip("\n")
            data_code = self.data_text.get(1.0, tk.END).rstrip("\n")

            if not model_code.strip():
                messagebox.showwarning("Export model", "Model editor is empty.")
                return

            solver_choice = self.solver.get() if hasattr(self, "solver") else "gurobi"

            default_name = "model_gurobi.py" if solver_choice == "gurobi" else "model_scipy.py"
            if self.model_file:
                base = os.path.splitext(os.path.basename(self.model_file or ""))[0]
                default_name = f"{base}_{'gurobi' if solver_choice == 'gurobi' else 'scipy'}.py"

            dest_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                initialfile=default_name,
                filetypes=[
                    ("Python files", "*.py"),
                    ("LP files", "*.lp"),
                    ("MPS files", "*.mps"),
                    ("All files", "*.*"),
                ],
            )
            if not dest_path:
                return

            export_ext = Path(dest_path).suffix.lower()
            if export_ext not in {".py", ".lp", ".mps"}:
                messagebox.showwarning(
                    "Export model",
                    "Choose a supported export extension: .py, .lp, or .mps.",
                )
                return

            # Compile through OPLCompiler so all AST rewrites/validation are applied.
            try:
                compiler = OPLCompiler()
                if export_ext == ".py":
                    _ast, generated_code, _data_dict = compiler.compile_model(
                        model_code,
                        data_code if data_code.strip() else None,
                        solver=solver_choice,
                    )
                    if not generated_code:
                        raise ValueError("Compiler returned no generated code.")

                    # Preserve existing behavior
                    lines = generated_code.rstrip("\n").split("\n")
                    if lines:
                        generated_code = "\n".join(lines[:-1])

                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                else:
                    ast, _generated_code, data_dict = compiler.compile_model(
                        model_code,
                        data_code if data_code.strip() else None,
                        solver="scipy",
                    )
                    problem = SciPyCSCCodeGenerator(ast, data_dict).build_problem()
                    export_linear_problem(problem, dest_path)
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                logging.getLogger(__name__).exception("Export failed")
                self.status_var.set(f"Error: Export failed: {detail}")
                messagebox.showerror("Export model", f"Export failed:\n{detail}")
                return

            self.status_var.set(f"Exported model to {dest_path}")
            print(f"Model exported to {export_ext.lstrip('.').upper()}")
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            logging.getLogger(__name__).exception("Unexpected export error")
            self.status_var.set(f"Unexpected error during export: {detail}")
            messagebox.showerror("Export model", f"Unexpected error:\n{detail}")
            self.status_var.set(f"Export failed: {detail}")

    # --- GenAI actions ---
    def _clear_output(self, header: str = "") -> str:
        """Start a new output request session and display its header."""
        return self._begin_new_output_session(header)

    def _append_output(self, text: str, session_id: Optional[str] = None) -> None:
        """Append text to the current output session and update the Output panel if visible."""
        sid = session_id or getattr(self, "_current_output_session_id", None)
        if sid:
            self._output_sessions[sid] = self._output_sessions.get(sid, "") + text
        if (
            sid
            and getattr(self, "_viewing_output_session_id", None) == sid
            and self.output_text.winfo_exists()
            and not getattr(self, "_solver_log_text", None)
        ):
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")
        # Persist session after any output change
        try:
            self._save_session()
        except Exception:
            pass

    def _format_prompt_for_output(self, label: str, prompt_input: _PromptInput) -> str:
        """Format a GenAI prompt so the request is captured in the output history."""
        if isinstance(prompt_input, dict):
            text = str(prompt_input.get("text", "")).strip()
            images = prompt_input.get("images") or []
            lines = [f"\n{label}:\n"]
            if text:
                lines.append(text + "\n")
            if images:
                lines.append("Attachments:\n")
                for image in images:
                    path = str(image.get("path", "")).strip() if isinstance(image, dict) else str(image).strip()
                    if path:
                        lines.append(f"- {OPLIDE._label_for_genai_attachment_path(self, path)}\n")
            lines.append("\n")
            return "".join(lines)

        text = str(prompt_input).strip()
        return f"\n{label}:\n{text}\n\n"

    def _label_for_output_session(self, header: str) -> str:
        """Derive a short session label from the session header."""
        header_lower = header.strip().lower()
        if header_lower.startswith("genai: generating"):
            return "Generate"
        if header_lower.startswith("genai: requesting feedback"):
            return "Ask"
        if header_lower.startswith("solve:"):
            return "Solve"
        if header_lower.startswith("export model"):
            return "Export"
        if header_lower.startswith("genai:"):
            return "GenAI"
        return "Session"

    def _make_output_session_display(self, timestamp: str, label: str, exclude_session_id: Optional[str] = None) -> str:
        """Return a request-list label that stays unique even for same-second requests."""
        base = f"{timestamp} • {label}"
        existing = {
            display for sid, display in getattr(self, "_output_session_display", {}).items() if sid != exclude_session_id
        }
        if base not in existing:
            return base
        suffix = 2
        while True:
            candidate = f"{base} ({suffix})"
            if candidate not in existing:
                return candidate
            suffix += 1

    # Output sessions (history)
    def _begin_new_output_session(self, header: str = "") -> str:
        """Create a new request session, add it to the list, and show it."""
        dt = datetime.now()
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        if not hasattr(self, "_output_session_timestamp") or self._output_session_timestamp is None:
            self._output_session_timestamp = {}
        if not hasattr(self, "_output_session_artifacts") or self._output_session_artifacts is None:
            self._output_session_artifacts = {}
        if not hasattr(self, "_output_session_label") or self._output_session_label is None:
            self._output_session_label = {}
        label = OPLIDE._label_for_output_session(self, header)
        display = OPLIDE._make_output_session_display(self, timestamp, label)
        session_id = dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        initial = (header + "\n") if header else ""
        self._output_sessions[session_id] = initial
        self._output_session_display[session_id] = display
        self._output_session_label[session_id] = label
        self._output_session_timestamp[session_id] = timestamp
        self._output_session_artifacts.setdefault(session_id, {})
        OPLIDE._snapshot_output_session_artifacts(self, session_id)
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
        # Persist new session immediately
        try:
            self._save_session()
        except Exception:
            pass
        return session_id

    def _show_output_session(self, session_id: str) -> None:
        """Display a session's content in the Output panel."""
        content = self._output_sessions.get(session_id, "")
        if hasattr(self, "output_text") and self.output_text.winfo_exists():
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, content)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")
        # Remember viewing session and persist
        try:
            self._viewing_output_session_id = session_id
            self._current_output_session_id = self._current_output_session_id or session_id
            self._save_session()
        except Exception:
            pass

    # --- Session Persistence (JSON) ---
    def _session_file_path(self) -> str:
        """Return the session file path in the current working directory."""
        try:
            return os.path.join(os.getcwd(), ".pyopl_session")
        except Exception:
            return ".pyopl_session"

    def _save_session(self) -> None:
        """Save the current IDE session (output history, editors, file paths) to JSON.

        This is intentionally tolerant: best-effort persistence without blocking UI.
        """
        try:
            path = self._session_file_path()
            tmp_path = path + ".tmp"
            payload = {
                "output_sessions": self._output_sessions,
                "output_session_ids": self._output_session_ids,
                "output_session_display": self._output_session_display,
                "output_session_label": self._output_session_label,
                "output_session_timestamp": self._output_session_timestamp,
                "output_session_artifacts": self._output_session_artifacts,
                "current_output_session_id": self._current_output_session_id,
                "viewing_output_session_id": self._viewing_output_session_id,
                "model_file": self.model_file,
                "data_file": self.data_file,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            # Write atomically
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp_path, path)
            except Exception:
                # Fallback: try remove and rename
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    os.replace(tmp_path, path)
                except Exception:
                    pass
        except Exception:
            logging.getLogger(__name__).exception("Failed to save .pyopl_session")

    def _load_session(self) -> None:
        """Load session JSON from cwd if present, restoring output history and editors."""
        try:
            path = self._session_file_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            logging.getLogger(__name__).exception("Failed to read .pyopl_session")
            return

        try:
            self._output_sessions = session.get("output_sessions", {}) or {}
            self._output_session_ids = session.get("output_session_ids", []) or []
            self._output_session_display = session.get("output_session_display", {}) or {}
            self._output_session_label = session.get("output_session_label", {}) or {}
            self._output_session_timestamp = session.get("output_session_timestamp", {}) or {}
            raw_artifacts = session.get("output_session_artifacts", {}) or {}
            self._output_session_artifacts = {}
            for sid, artifact in raw_artifacts.items():
                if not isinstance(artifact, dict):
                    continue
                normalized: dict[str, str] = {}
                if "model_text" in artifact:
                    normalized["model_text"] = str(artifact.get("model_text") or "")
                if "data_text" in artifact:
                    normalized["data_text"] = str(artifact.get("data_text") or "")
                self._output_session_artifacts[str(sid)] = normalized
            self._current_output_session_id = session.get("current_output_session_id") or self._current_output_session_id
            self._viewing_output_session_id = session.get("viewing_output_session_id") or self._viewing_output_session_id

            for sid in self._output_session_ids:
                if sid not in self._output_session_timestamp:
                    display = self._output_session_display.get(sid, sid)
                    if " • " in display:
                        self._output_session_timestamp[sid] = display.split(" • ", 1)[0]
                    else:
                        self._output_session_timestamp[sid] = display
                if sid not in self._output_session_label:
                    display = self._output_session_display.get(sid, sid)
                    timestamp = self._output_session_timestamp.get(sid, "")
                    prefix = f"{timestamp} • "
                    if timestamp and display.startswith(prefix):
                        self._output_session_label[sid] = display[len(prefix) :]
                    elif " • " in display:
                        self._output_session_label[sid] = display.split(" • ", 1)[-1]
                    else:
                        self._output_session_label[sid] = str(display)

            # Restore listbox UI
            if hasattr(self, "request_listbox"):
                try:
                    self.request_listbox.delete(0, tk.END)
                    for sid in self._output_session_ids:
                        display = self._output_session_display.get(sid, sid)
                        self.request_listbox.insert(tk.END, display)
                    # Select viewing session if available
                    if self._viewing_output_session_id and self._viewing_output_session_id in self._output_session_ids:
                        idx = self._output_session_ids.index(self._viewing_output_session_id)
                        self.request_listbox.selection_clear(0, tk.END)
                        self.request_listbox.selection_set(idx)
                        self.request_listbox.activate(idx)
                    elif self._output_session_ids:
                        self.request_listbox.selection_clear(0, tk.END)
                        self.request_listbox.selection_set(0)
                        self.request_listbox.activate(0)
                except Exception:
                    pass

            # Restore output_text
            if hasattr(self, "output_text") and self.output_text.winfo_exists():
                try:
                    sid = self._viewing_output_session_id or (
                        self._output_session_ids[0] if self._output_session_ids else None
                    )
                    if sid:
                        content = self._output_sessions.get(sid, "")
                        self._current_output_session_id = self._current_output_session_id or sid
                        self._viewing_output_session_id = sid
                        self.output_text.config(state="normal")
                        self.output_text.delete("1.0", tk.END)
                        self.output_text.insert(tk.END, content)
                        self.output_text.see(tk.END)
                        self.output_text.config(state="disabled")
                except Exception:
                    pass

            # Restore file pointers if present
            try:
                if session.get("model_file"):
                    self.model_file = session.get("model_file")
                    try:
                        with open(self.model_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        self.model_text.delete("1.0", tk.END)
                        self.model_text.insert(tk.END, content)
                    except Exception:
                        self.model_file = None  # Clear pointer if file can't be read
                if session.get("data_file"):
                    self.data_file = session.get("data_file")
                    try:
                        with open(self.data_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        self.data_text.delete("1.0", tk.END)
                        self.data_text.insert(tk.END, content)
                    except Exception:
                        self.data_file = None  # Clear pointer if file can't be read
                # If files were restored, update tab titles to show filenames
                try:
                    if hasattr(self, "editor_notebook"):
                        try:
                            model_label = f"Model: {os.path.basename(self.model_file)}" if self.model_file else "Model"
                            self.editor_notebook.tab(self.model_frame, text=model_label)
                        except Exception:
                            pass
                        try:
                            data_label = f"Data: {os.path.basename(self.data_file)}" if self.data_file else "Data"
                            self.editor_notebook.tab(self.data_frame, text=data_label)
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            logging.getLogger(__name__).exception("Failed to restore session state from .pyopl_session")

    def _ensure_model_data_saved(
        self, model_target: Optional[str] = None, data_target: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Ensure current editor buffers are saved to disk.

        Returns (model_path, data_path) or (None, None) on error.
        Does not change `self.model_file`/`self.data_file` — caller may choose to.
        """
        try:
            tmp_dir = os.path.join(os.getcwd(), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            model_path = model_target or getattr(self, "model_file", None) or os.path.join(tmp_dir, "temp_model.mod")
            data_path = data_target or getattr(self, "data_file", None) or os.path.join(tmp_dir, "temp_data.dat")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(self.model_text.get(1.0, tk.END).rstrip("\n"))
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(self.data_text.get(1.0, tk.END).rstrip("\n"))
            return model_path, data_path
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to write temp files: {e}")
            return None, None

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
        self._replace_scrolled_text_vbar(txt)
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

    def _ask_prompt_with_images(self, title: str, prompt: str, initial_text: str = "") -> Optional[_PromptInput]:
        """
        Show a resizable prompt dialog that supports attaching one or more images.

        Returns:
          - None if cancelled
          - str if OK and no images attached
          - dict {"text": str, "images": [{"path": str}, ...]} if images attached
        """
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

        # Text input
        txt = scrolledtext.ScrolledText(
            frm,
            wrap=tk.WORD,
            width=100,
            height=16,
            font=(self.editor_font_family, self.current_font_size),
        )
        self._replace_scrolled_text_vbar(txt)
        txt.grid(row=1, column=0, sticky="nsew")
        if initial_text:
            txt.insert("1.0", initial_text)
        txt.focus_set()

        # Attachments UI (Label + Listbox, mirroring the prompt label style)
        ttk.Label(frm, text="Attached images:", anchor="w", style="TLabel").grid(row=2, column=0, sticky="ew", pady=(8, 4))

        attachments = ttk.Frame(frm)
        attachments.grid(row=3, column=0, sticky="ew")
        attachments.columnconfigure(0, weight=1)

        file_list = tk.Listbox(attachments, height=4, exportselection=False)
        file_list.grid(row=0, column=0, sticky="ew")
        yscroll = ttk.Scrollbar(attachments, orient=tk.VERTICAL, command=file_list.yview)
        file_list.configure(yscrollcommand=yscroll.set)
        yscroll.grid(row=0, column=1, sticky="ns")

        selected_paths: list[str] = []
        _last_popup_index: Optional[int] = None

        def _refresh_list() -> None:
            try:
                file_list.delete(0, tk.END)
                for p in selected_paths:
                    file_list.insert(tk.END, p)
            except Exception:
                pass

        def _add_images() -> None:
            try:
                paths = filedialog.askopenfilenames(
                    title="Attach images",
                    filetypes=[
                        ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
                        ("All files", "*.*"),
                    ],
                )
            except Exception:
                paths = ()
            if not paths:
                return
            for p in paths:
                ps = str(p)
                if ps and ps not in selected_paths:
                    selected_paths.append(ps)
            _refresh_list()

        def _remove_selected() -> None:
            nonlocal _last_popup_index
            try:
                sel = file_list.curselection()
                idx = int(sel[0]) if sel else (_last_popup_index if _last_popup_index is not None else None)
                if idx is None:
                    return
                if 0 <= idx < len(selected_paths):
                    selected_paths.pop(idx)
                _last_popup_index = None
                _refresh_list()
            except Exception:
                pass

        def _clear_all() -> None:
            nonlocal _last_popup_index
            selected_paths.clear()
            _last_popup_index = None
            _refresh_list()

        # Right-click context menu for managing images
        ctx = tk.Menu(dlg, tearoff=0)
        ctx.add_command(label="Attach Images...", command=_add_images)
        ctx.add_command(label="Remove Selected", command=_remove_selected)
        ctx.add_separator()
        ctx.add_command(label="Clear All", command=_clear_all)

        def _popup_ctx(event: Optional[tk.Event]) -> None:
            nonlocal _last_popup_index
            if event is None:
                return
            try:
                size = file_list.size()
                _last_popup_index = None
                file_list.selection_clear(0, tk.END)
                if size > 0:
                    idx = int(file_list.nearest(event.y))
                    if 0 <= idx < size:
                        file_list.selection_set(idx)
                        file_list.activate(idx)
                        _last_popup_index = idx

                has_any = len(selected_paths) > 0
                has_sel = _last_popup_index is not None
                try:
                    ctx.entryconfigure("Remove Selected", state=("normal" if has_sel else "disabled"))
                    ctx.entryconfigure("Clear All", state=("normal" if has_any else "disabled"))
                except Exception:
                    pass

                ctx.tk_popup(event.x_root, event.y_root)
            finally:
                try:
                    ctx.grab_release()
                except Exception:
                    pass

        # Bind right-click (and macOS Ctrl+Click) on the list (and container for empty-space clicks)
        file_list.bind("<Button-3>", _popup_ctx)
        attachments.bind("<Button-3>", _popup_ctx)
        if sys.platform == "darwin":
            file_list.bind("<Button-2>", _popup_ctx)
            file_list.bind("<Control-Button-1>", _popup_ctx)
            attachments.bind("<Button-2>", _popup_ctx)
            attachments.bind("<Control-Button-1>", _popup_ctx)

        # OK / Cancel
        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, sticky="e", pady=(10, 0))
        result: dict[str, Any] = {"value": None}

        def on_ok(event=None) -> None:
            text_val = txt.get("1.0", tk.END).rstrip()
            if selected_paths:
                result["value"] = {"text": text_val, "images": [{"path": p} for p in selected_paths]}
            else:
                result["value"] = text_val
            dlg.destroy()

        def on_cancel(event=None) -> None:
            result["value"] = None
            dlg.destroy()

        ok_btn = ttk.Button(btns, text="OK", command=on_ok)
        cancel_btn = ttk.Button(btns, text="Cancel", command=on_cancel)
        cancel_btn.grid(row=0, column=1, padx=(6, 0))
        ok_btn.grid(row=0, column=0)

        dlg.bind("<Escape>", on_cancel)
        dlg.bind("<Control-Return>", on_ok)
        dlg.bind("<Command-Return>", on_ok)  # macOS

        dlg.wait_window()
        return result["value"]

    def genai_generate(self) -> None:
        """Open the docked GenAI composer in generation mode."""
        self._open_genai_panel("generate")

    def _run_genai_generate(self, prompt_input: _PromptInput) -> None:
        """Generate model and data from the docked GenAI composer input."""
        if not self.genai_provider or not self.genai_model:
            messagebox.showwarning("GenAI", "No GenAI model selected.")
            return

        # Resolve selected generator module
        gen_module = self._import_selected_genai_module()
        module_logger_name = getattr(gen_module, "__name__", "pyopl.genai.pyopl_generative")

        operation = self._start_foreground_operation(
            kind="genai-generate",
            label="Generate Model & Data",
            header="GenAI: Generating model and data...",
            status=(
                f"GenAI: generating with {self.genai_provider} • {self.genai_model} • "
                f"method={self._label_for_method(self.genai_method_var.get())} ..."
            ),
        )
        if operation is None:
            return
        self._append_output(
            self._format_prompt_for_output("Prompt", prompt_input),
            operation.session_id,
        )
        self._clear_pending_genai_revisions()

        # Use the visible request timestamp for filenames
        display_ts = self._output_session_timestamp.get(operation.session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        safe_ts = display_ts.replace(":", "-").replace(" ", "_")

        def run():
            try:
                # Progress hook -> Output panel
                def progress(msg: str) -> None:
                    if operation.cancel_requested:
                        return
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"), operation.session_id)

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
                    # Dispatch to selected generation method (PromptInput supports images)
                    assessment = gen_module.generative_solve(
                        prompt_input,
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

                with open(model_path, "r", encoding="utf-8") as f:
                    model_code = f.read()
                with open(data_path, "r", encoding="utf-8") as f:
                    data_code = f.read()

                if operation.cancel_requested:
                    return

                def apply_results():
                    if operation.cancel_requested:
                        return
                    self._finish_foreground_operation(operation)
                    # Load into editors
                    self.model_text.delete("1.0", tk.END)
                    self.model_text.insert(tk.END, model_code)
                    self.data_text.delete("1.0", tk.END)
                    self.data_text.insert(tk.END, data_code)
                    # Update file paths and tabs
                    self.model_file = model_path
                    self.data_file = data_path
                    self._record_output_session_artifacts(operation.session_id, model_text=model_code, data_text=data_code)
                    self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(model_path)}")
                    self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(data_path)}")
                    # Highlight
                    self.highlight(self.model_text, is_data=False)
                    self.highlight(self.data_text, is_data=True)
                    self._mark_editor_baselines_saved()
                    # Output and status
                    self._append_output("\nGenAI: Generation complete.\n", operation.session_id)
                    if assessment:
                        self._append_output(f"\nAssessment:\n{assessment}\n", operation.session_id)
                    self.status_var.set("GenAI: generation complete")
                    self._clear_genai_composer()
                    try:
                        self._save_session()
                    except Exception:
                        pass

                self.after(0, apply_results)

            except Exception as e:

                def on_error(e):
                    if operation.cancel_requested:
                        return
                    self._finish_foreground_operation(operation)
                    messagebox.showerror("GenAI Error", type(e).__name__)
                    self._append_output(f"\nGenAI Error: {e}\n", operation.session_id)
                    self.status_var.set("GenAI: error")

                self.after(0, on_error, e)

        threading.Thread(target=run, daemon=True).start()

    def genai_feedback(self) -> None:
        """Open the docked GenAI composer in Ask mode."""
        self._open_genai_panel("ask")

    def _run_genai_feedback(self, prompt_input: _PromptInput) -> None:
        """Request GenAI feedback from the docked composer input."""
        if not self.genai_provider or not self.genai_model:
            messagebox.showwarning("GenAI", "No GenAI model selected.")
            return

        # Track if a data file actually existed before this request
        had_data_file = bool(self.data_file and os.path.exists(self.data_file))

        # Ensure model/data are saved and get paths
        model_path, data_path = self._ensure_model_data_saved()
        if not model_path:
            return

        # Resolve selected generator module
        gen_module = self._import_selected_genai_module()
        module_logger_name = getattr(gen_module, "__name__", "pyopl.genai.pyopl_generative")

        operation = self._start_foreground_operation(
            kind="genai-feedback",
            label="Ask",
            header="GenAI: Requesting feedback...",
            status="GenAI: requesting feedback...",
            model_file=model_path,
            data_file=data_path,
        )
        if operation is None:
            return
        self._append_output(
            self._format_prompt_for_output("Question", prompt_input),
            operation.session_id,
        )
        self._clear_pending_genai_revisions()

        # Use visible request timestamp for filenames
        display_ts = self._output_session_timestamp.get(operation.session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        safe_ts = display_ts.replace(":", "-").replace(" ", "_")

        # Ensure a temp directory exists for any timestamped revised files
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        def run():
            try:
                # Progress hook -> Output panel
                def progress(msg: str) -> None:
                    if operation.cancel_requested:
                        return
                    self.after(0, self._append_output, (msg if msg.endswith("\n") else msg + "\n"), operation.session_id)

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
                    # PromptInput supports images (same shape as Generate)
                    result = gen_module.generative_feedback(
                        prompt_input,
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

                if operation.cancel_requested:
                    return

                def after_feedback():
                    if operation.cancel_requested:
                        return
                    self._finish_foreground_operation(operation)
                    self._append_output(f"\nFeedback:\n{feedback}\n", operation.session_id)
                    if revised_model or revised_data:
                        self._set_pending_genai_revisions(
                            {
                                "revised_model": revised_model,
                                "revised_data": revised_data,
                                "current_model": self.model_text.get("1.0", tk.END),
                                "current_data": self.data_text.get("1.0", tk.END),
                                "model_path": model_path,
                                "data_path": data_path,
                                "safe_ts": safe_ts,
                                "had_data_file": had_data_file,
                                "session_id": operation.session_id,
                            }
                        )
                    self._clear_genai_composer()

                    self.status_var.set("GenAI: feedback complete")
                    try:
                        self._save_session()
                    except Exception:
                        pass

                self.after(0, after_feedback)
            except Exception as e:

                def on_error(e):
                    if operation.cancel_requested:
                        return
                    self._finish_foreground_operation(operation)
                    messagebox.showerror("GenAI Error", str(e))
                    self._append_output(f"\nGenAI Error: {e}\n", operation.session_id)
                    self.status_var.set("GenAI: error")

                self.after(0, on_error, e)

        threading.Thread(target=run, daemon=True).start()

    def _genai_solve_and_explain(self) -> None:
        """Solve the current model/data and ask GenAI to explain the solution in lay terms."""
        # Ensure model/data are saved to files and get paths
        model_path, data_path = self._ensure_model_data_saved()
        if not model_path:
            return

        # Ensure run_model uses these exact files and update tabs/highlighting
        try:
            self.model_file = model_path
            self.data_file = data_path
            try:
                self.editor_notebook.tab(self.model_frame, text=f"Model: {os.path.basename(self.model_file or '')}")
                self.editor_notebook.tab(self.data_frame, text=f"Data: {os.path.basename(self.data_file or '')}")
            except Exception:
                pass
            try:
                self.highlight(self.model_text, is_data=False)
                self.highlight(self.data_text, is_data=True)
            except Exception:
                pass
        except Exception:
            pass

        # Start solve using existing run_model flow (re-uses multiprocessing and UI updates)
        self.run_model(explain_after_solve=True, model_file_override=model_path, data_file_override=data_path)

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
        try:
            self.update_idletasks()
        except Exception:
            pass
        # Re-highlight for contrast
        self.highlight(self.model_text, is_data=False)
        self.highlight(self.data_text, is_data=True)
        # Persist settings
        self._save_settings()

    def _strip_focus_from_ttk_layout(self, layout: Any) -> Any:
        """Remove ttk focus elements so buttons do not render the dotted focus motif."""
        stripped_layout: list[tuple[str, Any]] = []
        for element_name, element_options in layout:
            normalized_name = element_name.lower()
            if normalized_name == "focus" or normalized_name.endswith(".focus"):
                children = element_options.get("children", []) if element_options else []
                stripped_layout.extend(self._strip_focus_from_ttk_layout(children))
                continue

            updated_options = dict(element_options) if element_options else {}
            if "children" in updated_options:
                updated_options["children"] = self._strip_focus_from_ttk_layout(updated_options["children"])
            stripped_layout.append((element_name, updated_options))
        return stripped_layout

    def _configure_tk_scrollbar(
        self,
        scrollbar: Any,
        *,
        thumb_bg: str,
        active_bg: str,
        trough_bg: str,
        border_color: str,
    ) -> None:
        """Apply explicit colors to classic Tk scrollbars that do not follow ttk themes."""
        try:
            if scrollbar is None or not scrollbar.winfo_exists():
                return
            scrollbar.config(
                bg=thumb_bg,
                activebackground=active_bg,
                troughcolor=trough_bg,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
                highlightbackground=border_color,
                highlightcolor=border_color,
                elementborderwidth=0,
                activerelief=tk.FLAT,
            )
        except Exception:
            pass

    def _replace_scrolled_text_vbar(self, text_widget: Any) -> None:
        """Swap ScrolledText's native vertical scrollbar for a ttk one so themes apply on macOS."""
        try:
            if text_widget is None or not text_widget.winfo_exists():
                return
            current_vbar = getattr(text_widget, "vbar", None)
            if current_vbar is None or not current_vbar.winfo_exists():
                return
            if isinstance(current_vbar, ttk.Scrollbar):
                return
            parent = getattr(text_widget, "frame", text_widget.master)
            current_vbar.destroy()
            replacement_vbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=text_widget.yview)
            replacement_vbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.configure(yscrollcommand=replacement_vbar.set)
            text_widget.vbar = replacement_vbar
        except Exception:
            pass

    def _bind_autohide_vertical_scrollbar(
        self,
        widget: Any,
        scrollbar: Any,
        *,
        on_toggle: Optional[Callable[[], None]] = None,
    ) -> None:
        """Hide a vertical scrollbar when the full content is already visible."""
        try:
            if widget is None or scrollbar is None:
                return
            manager = str(scrollbar.winfo_manager())
            pack_restore_kwargs = None
            if manager == "pack":
                try:
                    pack_info = scrollbar.pack_info()
                    pack_restore_kwargs = {
                        "side": pack_info.get("side", tk.RIGHT),
                        "fill": pack_info.get("fill", tk.Y),
                    }
                except Exception:
                    pack_restore_kwargs = {"side": tk.RIGHT, "fill": tk.Y}
            is_visible = {"value": True}

            def _set_visibility(first: str, last: str) -> None:
                try:
                    scrollbar.set(first, last)
                except Exception:
                    pass

                try:
                    first_value = float(first)
                    last_value = float(last)
                except Exception:
                    first_value = 0.0
                    last_value = 1.0
                should_show = not (first_value <= 0.0 and last_value >= 1.0)
                if should_show == is_visible["value"]:
                    return

                if should_show:
                    if manager == "grid":
                        scrollbar.grid()
                    elif manager == "pack" and pack_restore_kwargs is not None:
                        scrollbar.pack(**pack_restore_kwargs)
                else:
                    if manager == "grid":
                        scrollbar.grid_remove()
                    elif manager == "pack":
                        scrollbar.pack_forget()
                is_visible["value"] = should_show
                if on_toggle is not None:
                    try:
                        self.after_idle(on_toggle)
                    except Exception:
                        pass

            widget.configure(yscrollcommand=_set_visibility)
            self.after_idle(lambda: _set_visibility(*(str(value) for value in widget.yview())))
        except Exception:
            pass

    def _apply_theme_colors(self) -> None:
        """Apply text widget colors based on theme."""
        theme = self.theme_var.get()
        self._apply_macos_theme_appearance(theme)
        if theme == "darkly":
            root_bg = "#212529"
            editor_bg = "#2b3035"
            editor_fg = "#e9ecef"
            caret_fg = "#e9ecef"
            sidebar_bg = editor_bg
            output_bg = "#212529"
            output_fg = "#e9ecef"
            error_fg = "white"
            paned_bg = root_bg
            sidebar_fg = editor_fg
            sidebar_muted = "#aab4be"
            list_select_bg = "#334155"
            inset_border = "#495057"
            status_bg = "#212529"
            status_fg = "#cfd6dd"
            status_meta_fg = "#8f9aa3"
            scrollbar_thumb_bg = "#495057"
            scrollbar_active_bg = "#5c636a"
            scrollbar_trough_bg = "#212529"
            ttk_scrollbar_bg = "#495057"
            ttk_scrollbar_active_bg = "#5c636a"
            ttk_scrollbar_trough_bg = "#2b3035"
        else:
            root_bg = "#f8f9fa"
            editor_bg = "#ffffff"
            editor_fg = "#212529"
            caret_fg = "#212529"
            sidebar_bg = editor_bg
            output_bg = "#f8f9fa"
            output_fg = "#212529"
            error_fg = "black"
            paned_bg = root_bg
            sidebar_fg = editor_fg
            sidebar_muted = "#6b7785"
            list_select_bg = "#cfe0ff"
            inset_border = "#ced4da"
            status_bg = "#f8f9fa"
            status_fg = "#364152"
            status_meta_fg = "#7b8794"
            scrollbar_thumb_bg = "#c1c9d0"
            scrollbar_active_bg = "#adb5bd"
            scrollbar_trough_bg = "#f1f3f5"
            ttk_scrollbar_bg = "#c1c9d0"
            ttk_scrollbar_active_bg = "#adb5bd"
            ttk_scrollbar_trough_bg = "#f1f3f5"

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
        if hasattr(self, "top_row_paned") and self.top_row_paned.winfo_exists():
            try:
                self.top_row_paned.config(bg=paned_bg, bd=0, relief=tk.FLAT, sashrelief=tk.FLAT)
            except Exception:
                pass
        if hasattr(self, "bottom_row_paned") and self.bottom_row_paned.winfo_exists():
            try:
                self.bottom_row_paned.config(bg=paned_bg, bd=0, relief=tk.FLAT, sashrelief=tk.FLAT)
            except Exception:
                pass

        # Apply to editors
        if hasattr(self, "model_text"):
            self.model_text.config(
                bg=editor_bg,
                fg=editor_fg,
                insertbackground=caret_fg,
                relief=tk.FLAT,
                bd=0,
                highlightbackground=inset_border,
                highlightcolor=inset_border,
            )
        if hasattr(self, "data_text"):
            self.data_text.config(
                bg=editor_bg,
                fg=editor_fg,
                insertbackground=caret_fg,
                relief=tk.FLAT,
                bd=0,
                highlightbackground=inset_border,
                highlightcolor=inset_border,
            )
        if hasattr(self, "output_text"):
            self.output_text.config(
                bg=output_bg,
                fg=output_fg,
                relief=tk.FLAT,
                bd=0,
                highlightbackground=inset_border,
                highlightcolor=inset_border,
            )

        # Ensure the editor frames share the same background as the text area
        try:
            self.style.configure("Editor.TFrame", background=editor_bg)
            self.style.configure("Sidebar.TFrame", background=sidebar_bg)
            self.style.configure(
                "SidebarHeader.TLabel",
                background=sidebar_bg,
                foreground=sidebar_fg,
                font=(self.interface_font_family, 13, "bold"),
            )
            self.style.configure(
                "SidebarSection.TLabel",
                background=sidebar_bg,
                foreground=sidebar_fg,
                font=(self.interface_font_family, 10, "bold"),
            )
            self.style.configure(
                "SidebarSubtle.TLabel",
                background=sidebar_bg,
                foreground=sidebar_muted,
                font=(self.interface_font_family, 9),
            )
            self.style.configure(
                "GenaiMode.TButton",
                background=paned_bg,
                foreground=sidebar_muted,
                borderwidth=0,
                focusthickness=0,
                padding=(10, 5),
                font=self.interface_button_font,
            )
            self.style.map("GenaiMode.TButton", background=[("active", list_select_bg), ("pressed", list_select_bg)])
            self.style.configure(
                "GenaiModeActive.TButton",
                background=list_select_bg,
                foreground=sidebar_fg,
                borderwidth=0,
                focusthickness=0,
                padding=(10, 5),
                font=self.interface_button_font,
            )
            self.style.map("GenaiModeActive.TButton", background=[("active", list_select_bg), ("pressed", list_select_bg)])
            button_layout = self._strip_focus_from_ttk_layout(self.style.layout("TButton"))
            for button_style in ("TButton", "GenaiMode.TButton", "GenaiModeActive.TButton"):
                self.style.layout(button_style, button_layout)
            self.style.configure("StatusBar.TFrame", background=status_bg)
            self.style.configure(
                "StatusBar.TLabel", background=status_bg, foreground=status_fg, font=self.interface_button_font
            )
            self.style.configure(
                "StatusBarMeta.TLabel",
                background=status_bg,
                foreground=status_meta_fg,
                font=self.interface_button_font,
            )
            self.style.configure(
                "TScrollbar",
                background=ttk_scrollbar_bg,
                troughcolor=ttk_scrollbar_trough_bg,
                bordercolor=ttk_scrollbar_trough_bg,
                darkcolor=ttk_scrollbar_bg,
                lightcolor=ttk_scrollbar_bg,
                arrowcolor=editor_fg,
                gripcount=0,
            )
            self.style.map(
                "TScrollbar",
                background=[("active", ttk_scrollbar_active_bg), ("pressed", ttk_scrollbar_active_bg)],
                arrowcolor=[("disabled", sidebar_muted), ("active", editor_fg)],
            )
        except Exception:
            pass

        if hasattr(self, "status_bar"):
            try:
                self.status_bar.configure(style="StatusBar.TFrame")
            except Exception:
                pass
        if hasattr(self, "editor_notebook"):
            try:
                self.editor_notebook.configure(style="TNotebook", padding=0)
            except Exception:
                pass
        if hasattr(self, "status_bar_labels"):
            for idx, label in enumerate(self.status_bar_labels):
                try:
                    label.configure(style=("StatusBar.TLabel" if idx == 0 else "StatusBarMeta.TLabel"))
                except Exception:
                    pass

        if hasattr(self, "genai_prompt_text"):
            self.genai_prompt_text.config(
                bg=editor_bg,
                fg=editor_fg,
                insertbackground=caret_fg,
                relief=tk.FLAT,
                bd=0,
                highlightbackground=inset_border,
                highlightcolor=inset_border,
            )
            self._configure_tk_scrollbar(
                getattr(self.genai_prompt_text, "vbar", None),
                thumb_bg=scrollbar_thumb_bg,
                active_bg=scrollbar_active_bg,
                trough_bg=scrollbar_trough_bg,
                border_color=inset_border,
            )
        if hasattr(self, "genai_attachment_listbox"):
            self.genai_attachment_listbox.config(
                bg=editor_bg,
                fg=editor_fg,
                selectbackground=list_select_bg,
                selectforeground=editor_fg,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
            )
        if hasattr(self, "sessions_surface"):
            self.sessions_surface.config(bg=inset_border, highlightbackground=inset_border, highlightcolor=inset_border)
            for child in self.sessions_surface.winfo_children():
                if isinstance(child, tk.Scrollbar):
                    self._configure_tk_scrollbar(
                        child,
                        thumb_bg=scrollbar_thumb_bg,
                        active_bg=scrollbar_active_bg,
                        trough_bg=scrollbar_trough_bg,
                        border_color=inset_border,
                    )
        if hasattr(self, "request_listbox"):
            self.request_listbox.config(
                bg=editor_bg,
                fg=editor_fg,
                selectbackground=list_select_bg,
                selectforeground=editor_fg,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
            )
        for text_widget in (
            getattr(self, "model_text", None),
            getattr(self, "data_text", None),
            getattr(self, "output_text", None),
            getattr(self, "genai_prompt_text", None),
        ):
            self._configure_tk_scrollbar(
                getattr(text_widget, "vbar", None),
                thumb_bg=scrollbar_thumb_bg,
                active_bg=scrollbar_active_bg,
                trough_bg=scrollbar_trough_bg,
                border_color=inset_border,
            )

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
                with open(self._config_path, "r", encoding="utf-8") as f:
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
                "solver": (
                    self.solver.get() if hasattr(self, "solver") and self.solver.get() in ("gurobi", "scipy") else "gurobi"
                ),
                "verbose-llm-logs": bool(self.verbose_llm_var.get()) if hasattr(self, "verbose_llm_var") else False,
                "display-solver-progress": (
                    bool(self.display_solver_progress_var.get()) if hasattr(self, "display_solver_progress_var") else True
                ),
                "genai-selection": (
                    f"{self.genai_provider}|{self.genai_model}"
                    if getattr(self, "genai_provider", None) and getattr(self, "genai_model", None)
                    else ""
                ),
                "genai-method": self.genai_method_var.get() if hasattr(self, "genai_method_var") else "pyopl_generative",
            }
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to save settings: {e}")

    def _get_editor_text(self, text_widget: tk.Text) -> str:
        """Return editor content without Tk's implicit trailing newline."""
        return text_widget.get("1.0", "end-1c")

    def _mark_editor_baselines_saved(self) -> None:
        """Record the current model/data editor contents as the clean close baseline."""
        try:
            self._model_saved_text = self._get_editor_text(self.model_text)
        except Exception:
            self._model_saved_text = ""
        try:
            self._data_saved_text = self._get_editor_text(self.data_text)
        except Exception:
            self._data_saved_text = ""

    def _has_unsaved_editor_changes(self) -> bool:
        """Return True when either model or data editor differs from its saved baseline."""
        try:
            if self._get_editor_text(self.model_text) != self._model_saved_text:
                return True
        except Exception:
            pass
        try:
            if self._get_editor_text(self.data_text) != self._data_saved_text:
                return True
        except Exception:
            pass
        return False

    def _confirm_quit_with_unsaved_changes(self) -> bool:
        """Ask whether to quit with unsaved model/data editor changes."""
        return bool(
            messagebox.askyesno(
                "Unsaved Changes",
                "Are you sure you want to quit? There are unsaved changes",
                parent=self,
            )
        )

    def _on_close(self) -> None:
        """Persist settings and close the app."""
        if self._has_unsaved_editor_changes() and not self._confirm_quit_with_unsaved_changes():
            return
        setattr(self, "_shutting_down", True)
        try:
            self.stop_model()  # ensure no stray solver process
        except Exception:
            pass
        self._save_settings()
        try:
            self._save_session()
        except Exception:
            pass
        self._cleanup_genai_pdf_temp_dir()
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
        self.bind_all("<Control-e>", self._genai_solve_and_explain_shortcut)
        self.bind_all("<Control-f>", self._find_shortcut)
        self.bind_all("<Control-q>", self._close_shortcut)

        if sys.platform == "darwin":
            self.bind_all("<Command-s>", self.save_current_buffer)
            self.bind_all("<Command-n>", self._new_model_shortcut)
            self.bind_all("<Command-r>", self._run_model_shortcut)
            self.bind_all("<Command-g>", self._genai_generate_shortcut)
            self.bind_all("<Command-i>", self._genai_feedback_shortcut)
            self.bind_all("<Command-e>", self._genai_solve_and_explain_shortcut)
            self.bind_all("<Command-f>", self._find_shortcut)

    def _close_shortcut(self, event: Optional[tk.Event] = None) -> str:
        """Keyboard shortcut handler for closing the IDE."""
        self._on_close()
        return "break"

    def _new_model_shortcut(self, event: Optional[tk.Event] = None) -> str:
        """Keyboard shortcut handler for creating a new model."""
        self.new_model()
        return "break"

    def _find_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self._open_find_replace_dialog(False)
        return "break"

    def _replace_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self._open_find_replace_dialog(True)
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

    def _genai_solve_and_explain_shortcut(self, event: Optional[tk.Event] = None) -> str:
        self._genai_solve_and_explain()
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
        try:
            # Import version/year from package init if available
            from . import __version__, __year__
        except Exception:
            __version__ = "unknown"
            __year__ = ""

        messagebox.showinfo(
            "About Rhetor",
            f"Rhetor {__version__}\n\n© {__year__} University of Edinburgh",
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
        active = getattr(self, "_active_operation", None)
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
            self.genai_menu.add_checkbutton(
                label="Show GenAI Panel",
                onvalue=True,
                offvalue=False,
                variable=self.show_genai_panel_var,
                command=self._toggle_genai_panel_visibility,
            )
            # Solve & Explain: solve current model/data then ask LLM to explain results
            self.genai_menu.add_separator()
            if active is not None:
                self.genai_menu.add_command(label=f"Interrupt {active.label}", command=self.interrupt_active_operation)
            else:
                self.genai_menu.add_command(
                    label="Solve & Explain", command=self._genai_solve_and_explain, accelerator=self._accel("E")
                )

            # Verbose LLM progress logs (only visible when launched with --debug)
            if getattr(self, "debug", False):
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

            # Preserve the live selection when menus refresh; fall back to the saved
            # selection only when there is no current valid choice.
            preselected = False
            try:
                if self.genai_provider and self.genai_model:
                    models = provider_models.get(self.genai_provider) or []
                    if self.genai_model in models:
                        self.genai_selection_var.set(f"{self.genai_provider}|{self.genai_model}")
                        preselected = True
                if not preselected and self._desired_genai_provider and self._desired_genai_model:
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
            if active is not None:
                self.genai_menu.add_command(label=f"Interrupt {active.label}", command=self.interrupt_active_operation)
            else:
                # No models available
                self.genai_menu.add_command(label="No models available", state="disabled")
            try:
                self.menubar.entryconfig("GenAI", state=("normal" if active is not None else "disabled"))
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
        except Exception:
            pass
        self._refresh_genai_panel_state()
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
        return "SyntAGM"

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
            self.after_idle(lambda: self.set_theme(theme))

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
        self._desired_genai_provider = provider_key
        self._desired_genai_model = model_name
        try:
            self.genai_selection_var.set(f"{provider_key}|{model_name}")
        except Exception:
            pass
        self._refresh_genai_panel_state()
        self._save_settings()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    ide = OPLIDE()
    ide.mainloop()
