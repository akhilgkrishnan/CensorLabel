"""In-app file browser for the server's local disk (used in browser mode, where no OS dialog exists)."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"})


def _size(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def _places() -> list[tuple[str, str, Path]]:
    home = Path.home()
    candidates = [
        ("Home", "home", home),
        ("Desktop", "desktop_mac", home / "Desktop"),
        ("Downloads", "download", home / "Downloads"),
        ("Movies", "movie", home / "Movies"),
        ("Videos", "movie", home / "Videos"),
        ("Volumes", "storage", Path("/Volumes")),
    ]
    return [c for c in candidates if c[2].is_dir()]


class LocalFilePicker(ui.dialog):
    """`path = await LocalFilePicker(start_dir)` returns the chosen file path, or None if cancelled.

    With `save_name`, it works as a "Save as" dialog: pick a folder, edit the file name, press Save.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        extensions: frozenset[str] = VIDEO_EXTENSIONS,
        save_name: str | None = None,
    ):
        super().__init__()
        start = Path(directory).expanduser() if directory else Path.home()
        self.directory = start if start.is_dir() else Path.home()
        self.extensions = extensions
        self.saving = save_name is not None
        self.show_all = False

        with self, ui.card().classes("w-[44rem] max-w-full gap-2 p-4"):
            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                ui.button(icon="arrow_upward", on_click=self._up).props("flat round dense").tooltip("Parent folder")
                self.path_input = ui.input().props("dense outlined").classes("grow font-mono text-sm")
                self.path_input.on("keydown.enter", self._jump)
            with ui.row().classes("w-full gap-1"):
                for name, icon, path in _places():
                    ui.chip(name, icon=icon, on_click=lambda p=path: self._open(p)).props("outline clickable dense")
            self.listing = ui.list().props("dense separator").classes("w-full h-96 overflow-auto rounded border")
            if self.saving:
                self.name_input = ui.input("File name", value=save_name).props("dense outlined").classes("w-full")
                self.name_input.on("keydown.enter", self._save)
            with ui.row().classes("w-full items-center justify-between"):
                ui.switch("Show all files", on_change=self._toggle_all).props("dense")
                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=lambda: self.submit(None)).props("flat")
                    if self.saving:
                        ui.button("Save", icon="save", on_click=self._save)
        self._render()

    def _open(self, path: Path) -> None:
        self.directory = path
        self._render()

    def _save(self) -> None:
        name = (self.name_input.value or "").strip()
        if not name or "/" in name:
            ui.notify("Enter a file name", type="warning")
            return
        self.submit(str(self.directory / name))

    def _pick_file(self, path: Path) -> None:
        if self.saving:
            self.name_input.value = path.name  # choose an existing file to overwrite it
        else:
            self.submit(str(path))

    def _up(self) -> None:
        self._open(self.directory.parent)

    def _jump(self) -> None:
        target = Path(self.path_input.value).expanduser()
        if target.is_file():
            self._pick_file(target)
        elif target.is_dir():
            self._open(target)
        else:
            ui.notify(f"Not found: {target}", type="warning")

    def _toggle_all(self, e) -> None:
        self.show_all = e.value
        self._render()

    def _render(self) -> None:
        self.path_input.value = str(self.directory)
        self.listing.clear()
        try:
            entries = [p for p in self.directory.iterdir() if not p.name.startswith(".")]
        except OSError as exc:
            with self.listing:
                ui.item_label(f"Cannot open folder: {exc.strerror}").classes("p-4 text-red-600")
            return
        folders = sorted((p for p in entries if p.is_dir()), key=lambda p: p.name.lower())
        files = sorted(
            (p for p in entries if p.is_file() and (self.show_all or p.suffix.lower() in self.extensions)),
            key=lambda p: p.name.lower(),
        )
        with self.listing:
            for folder in folders:
                with ui.item(on_click=lambda p=folder: self._open(p)):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder", color="amber-7")
                    ui.item_section(folder.name)
            for file in files:
                with ui.item(on_click=lambda p=file: self._pick_file(p)):
                    with ui.item_section().props("avatar"):
                        ui.icon("movie" if file.suffix.lower() in self.extensions else "description", color="primary")
                    ui.item_section(file.name)
                    with ui.item_section().props("side"):
                        ui.item_label(_size(file.stat().st_size)).props("caption")
            if not folders and not files:
                ui.item_label("No video files here").classes("p-4 opacity-60")
