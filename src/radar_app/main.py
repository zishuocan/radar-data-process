from __future__ import annotations

import tkinter as tk

from .ui.main_window import RadarApp


def main() -> None:
    root = tk.Tk()
    app = RadarApp(root)
    app.run()


if __name__ == "__main__":
    main()
