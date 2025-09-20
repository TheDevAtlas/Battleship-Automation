
"""
Interactive Battleship Heatmap (Tkinter)

- 10x10 grid with heatmap coloring for optimal targeting
- Rocket emoji on the best shot (highest-heat unknown cell)
- Click a tile to cycle: Unknown → Miss → Hit → Unknown; heat recomputes
- Checkboxes to toggle remaining ships by shape (L2,L3,L4,L5,T4,S6)
- Valid target tiles (heat > 0) get a highlighted outline

Notes
- This is a self-contained, fake sandbox for testing targeting logic.
- Takes into account different ship shapes you noted: straight lines plus
  a T4 tetromino and an S6 curve. Each ship is toggled individually.
- Heat counts the number of valid placements of the remaining ship shapes
  covering each unknown cell, given current misses/hits. If any hits exist,
  only placements that include at least one hit are counted (target mode).
"""

import tkinter as tk
from tkinter import ttk
import os


# Board and UI settings
GRID_SIZE = 10
CELL = 46
MARGIN = 12
FONT_CELL = ("Segoe UI Emoji", 16)  # Emoji-capable font (Windows friendly)
FONT_UI = ("Segoe UI", 10)

# Data paths
BIAS_CSV = os.path.join("Logs", "Battleship-Hit-Heatmap.csv")

# Cell states
UNKNOWN = 0
MISS = 1
HIT = 2


# ----- Ship shapes (lines + T4 + S6) -----
def _line(n):
    return {(0, d) for d in range(n)}


SHAPES_BASE = {
    "L2": _line(2),
    "L3": _line(3),
    "L4": _line(4),
    "L5": _line(5),
    "T4": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "S6": {(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)},
}


def _normalize(cells):
    minr = min(r for r, _ in cells)
    minc = min(c for _, c in cells)
    return tuple(sorted([(r - minr, c - minc) for (r, c) in cells]))


def _rot90(cells):
    return {(c, -r) for (r, c) in cells}


def _mirror(cells):
    return {(r, -c) for (r, c) in cells}


def _all_orientations(base):
    variants = set()
    work = [set(base)]
    for _ in range(3):
        work.append(_rot90(work[-1]))
    all_sets = work + [_mirror(s) for s in work]
    for s in all_sets:
        variants.add(_normalize(s))
    return [list(v) for v in variants]


SHAPES_ORIENTED = {name: _all_orientations(cells) for name, cells in SHAPES_BASE.items()}


def compute_heat(board, remaining_ship_names):
    n = len(board)
    heat = [[0 for _ in range(n)] for _ in range(n)]

    # Gather hits and quick lookup helpers
    hits = set()
    misses = set()
    for r in range(n):
        for c in range(n):
            if board[r][c] == HIT:
                hits.add((r, c))
            elif board[r][c] == MISS:
                misses.add((r, c))

    any_hits = len(hits) > 0

    def placement_valid(cells):
        # Cannot touch a known miss
        for rc in cells:
            if rc in misses:
                return False
        return True

    for name in remaining_ship_names:
        variants = SHAPES_ORIENTED.get(name, [])
        for var in variants:
            maxr = max(r for r, _ in var)
            maxc = max(c for _, c in var)
            for br in range(n - maxr):
                for bc in range(n - maxc):
                    cells = [(br + r, bc + c) for (r, c) in var]
                    if not placement_valid(cells):
                        continue
                    cells_set = set(cells)
                    # Target mode: only count placements that include at least one hit
                    if any_hits and not (cells_set & hits):
                        continue
                    # Do not add weight to already-shot cells
                    for (r, c) in cells:
                        if board[r][c] == UNKNOWN:
                            heat[r][c] += 1

    return heat


def max_heat_cell(board, heat, bias=None):
    best = None
    best_val = -1
    best_bias = -1
    n = len(board)
    for r in range(n):
        for c in range(n):
            if board[r][c] != UNKNOWN:
                continue
            v = heat[r][c]
            if v < best_val:
                continue
            b = bias[r][c] if bias and r < len(bias) and c < len(bias[r]) else 0
            if v > best_val:
                best_val = v
                best_bias = b
                best = (r, c)
            elif v == best_val and b > best_bias:
                best_bias = b
                best = (r, c)
    return best, best_val, best_bias


def load_bias_map(path, n):
    # Reads a n x n grid of ints/floats from CSV (comma-separated)
    # Returns zeros if file missing or malformed
    bias = [[0 for _ in range(n)] for _ in range(n)]
    try:
        if not os.path.exists(path):
            return bias
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                row = []
                for p in parts:
                    try:
                        row.append(float(p))
                    except Exception:
                        row.append(0.0)
                rows.append(row)
        # Copy into n x n bias with cropping/padding
        for r in range(min(n, len(rows))):
            row = rows[r]
            for c in range(min(n, len(row))):
                bias[r][c] = float(row[c])
        return bias
    except Exception:
        return bias


def color_from_heat(v, vmax):
    # Blue -> Cyan -> Yellow -> Red gradient
    if vmax <= 0 or v <= 0:
        return "#3c3c3c"  # base for unknown cells with no heat
    t = max(0.0, min(1.0, v / vmax))
    if t < 0.33:
        k = t / 0.33
        b, g, r = 255, int(255 * k), 0
    elif t < 0.66:
        k = (t - 0.33) / 0.33
        b, g, r = int(255 * (1 - k)), 255, int(255 * k)
    else:
        k = (t - 0.66) / 0.34
        b, g, r = 0, int(255 * (1 - k)), 255
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Battleship Heatmap – Sandbox")

        self.board = [[UNKNOWN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.bias = load_bias_map(BIAS_CSV, GRID_SIZE)

        # Remaining ships by shape: S6, L5, L4, T4, L3, L2
        self.ship_names = ["S6", "L5", "L4", "T4", "L3", "L2"]
        self.vars_ships = [tk.BooleanVar(value=True) for _ in self.ship_names]

        self.canvas_w = MARGIN * 2 + GRID_SIZE * CELL
        self.canvas_h = self.canvas_w

        self.build_ui()
        self.recompute()

    def build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, padx=8, pady=8)

        self.canvas = tk.Canvas(left, width=self.canvas_w, height=self.canvas_h, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click_left)

        right = ttk.Frame(container)
        right.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.Y)

        ttk.Label(right, text="Ships Remaining", font=FONT_UI).pack(anchor="w", pady=(0, 4))
        for name, var in zip(self.ship_names, self.vars_ships):
            ttk.Checkbutton(right, text=name, variable=var, command=self.recompute).pack(anchor="w")

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Button(right, text="Reset Board", command=self.reset_board).pack(fill=tk.X)
        ttk.Button(right, text="Auto 15 Moves (MISS)", command=self.auto_15).pack(fill=tk.X, pady=(6, 0))
        ttk.Label(right, text="Click a cell: Unknown → Miss → Hit", font=FONT_UI).pack(anchor="w", pady=(8, 0))
        self.info = ttk.Label(right, text="", font=FONT_UI)
        self.info.pack(anchor="w", pady=(4, 0))

    def remaining_ships(self):
        # Convert checkboxes to a list of selected ship names
        return [name for name, var in zip(self.ship_names, self.vars_ships) if var.get()]

    def reset_board(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.board[r][c] = UNKNOWN
        self.recompute()

    def on_click_left(self, event):
        # Locate cell
        gx = event.x - MARGIN
        gy = event.y - MARGIN
        if gx < 0 or gy < 0:
            return
        c = gx // CELL
        r = gy // CELL
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            # Cycle Unknown -> Miss -> Hit -> Unknown
            self.board[r][c] = (self.board[r][c] + 1) % 3
            self.recompute()

    def recompute(self):
        self.heat = compute_heat(self.board, self.remaining_ships())
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        n = GRID_SIZE
        vmax = max((v for row in self.heat for v in row), default=0)

        # Draw cells
        for r in range(n):
            for c in range(n):
                x1 = MARGIN + c * CELL
                y1 = MARGIN + r * CELL
                x2 = x1 + CELL
                y2 = y1 + CELL

                state = self.board[r][c]
                h = self.heat[r][c]

                if state == UNKNOWN:
                    fill = color_from_heat(h, vmax)
                elif state == MISS:
                    fill = "#303030"
                else:  # HIT
                    fill = "#1d4d2c"  # dark green

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, width=1, outline="#707070")

                # Mark MISS/HIT symbols
                if state == MISS:
                    self.canvas.create_line(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#ffb000", width=2)
                    self.canvas.create_line(x1 + 6, y2 - 6, x2 - 6, y1 + 6, fill="#ffb000", width=2)
                elif state == HIT:
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, outline="#4fff8a", width=2)

        # Outline valid targets (heat > 0) with thicker border
        for r in range(n):
            for c in range(n):
                if self.board[r][c] == UNKNOWN and self.heat[r][c] > 0:
                    x1 = MARGIN + c * CELL
                    y1 = MARGIN + r * CELL
                    x2 = x1 + CELL
                    y2 = y1 + CELL
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ffd400", width=2)

        # Rocket emoji at best unknown cell
        (best_rc, best_val, best_bias) = max_heat_cell(self.board, self.heat, self.bias)
        if best_rc and best_val > 0:
            br, bc = best_rc
            cx = MARGIN + bc * CELL + CELL // 2
            cy = MARGIN + br * CELL + CELL // 2
            self.canvas.create_text(cx, cy, text="🚀", font=FONT_CELL)

        # Info text
        total_valid = sum(1 for r in range(n) for c in range(n) if self.board[r][c] == UNKNOWN and self.heat[r][c] > 0)
        self.info.config(text=f"Valid targets: {total_valid}    Max heat: {best_val if best_val>0 else 0}    Bias: {int(best_bias) if best_val>0 else 0}")

    def auto_15(self):
        moves = []
        for _ in range(15):
            # Ensure heat is current
            self.heat = compute_heat(self.board, self.remaining_ships())
            best_rc, best_val, best_bias = max_heat_cell(self.board, self.heat, self.bias)
            if not best_rc:
                break
            r, c = best_rc
            # Record as x,y (col,row)
            moves.append((c, r))
            # Simulate a click as MISS on the suggested target
            if self.board[r][c] == UNKNOWN:
                self.board[r][c] = MISS
            # Redraw after each move to reflect updates
            self.draw()
            self.root.update_idletasks()
        # Print the 15 x,y locations to stdout, one per line
        print("Auto 15 target sequence (x,y):")
        for (x, y) in moves:
            print(f"{x},{y}")
        # Brief summary in UI
        if moves:
            self.info.config(text=f"Auto ran {len(moves)} moves. Last: {moves[-1][0]},{moves[-1][1]}")


if __name__ == "__main__":
    root = tk.Tk()
    # Use ttk themed widgets
    try:
        from tkinter import TclError
        root.call("source", "sun-valley.tcl")  # if present, load theme (optional)
        root.call("set_theme", "dark")
    except Exception:
        pass
    App(root)
    root.mainloop()
