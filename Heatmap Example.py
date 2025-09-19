import cv2
import numpy as np

# Simple interactive Battleship heatmap demo
# - 10x10 board
# - Ships: lengths [5,4,3,3,2] (line ships)
# - Click a cell to mark it as MISS; heatmap updates immediately
# - Press R to reset; ESC or Q to quit

GRID_SIZE = 10
CELL_PX = 56
MARGIN = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX

UNKNOWN, MISS = 0, 1
SHIPS = [5, 4, 3, 3, 2]


def placement_valid(state, cells):
    for (r, c) in cells:
        if state[r, c] == MISS:
            return False
    return True


def iter_line_placements(length):
    g = GRID_SIZE
    # Horizontal
    for r in range(g):
        for c in range(g - length + 1):
            yield [(r, c + k) for k in range(length)]
    # Vertical
    for c in range(g):
        for r in range(g - length + 1):
            yield [(r + k, c) for k in range(length)]


def heat_for_length(state, length):
    g = GRID_SIZE
    heat = np.zeros((g, g), dtype=np.float32)
    any_valid = False
    for cells in iter_line_placements(length):
        if not placement_valid(state, cells):
            continue
        any_valid = True
        for (r, c) in cells:
            if state[r, c] == UNKNOWN:
                heat[r, c] += 1.0
    return heat, any_valid


def compute_heat(state):
    g = GRID_SIZE
    combined = np.zeros((g, g), dtype=np.float32)
    for L in SHIPS:
        h, ok = heat_for_length(state, L)
        if ok:
            combined += h
    return combined


def heat_color(t):
    # t in [0,1] -> BGR color, blue->cyan->yellow->red
    t = max(0.0, min(1.0, float(t)))
    if t < 0.33:
        k = t / 0.33
        return (255, int(255*k), 0)
    elif t < 0.66:
        k = (t - 0.33) / 0.33
        return (int(255*(1-k)), 255, int(255*k))
    else:
        k = (t - 0.66) / 0.34
        return (0, int(255*(1-k)), 255)


def draw_board(state, heat):
    g = GRID_SIZE
    w = h = MARGIN*2 + g*CELL_PX
    img = np.full((h, w, 3), 30, dtype=np.uint8)

    # Normalize heat for colors
    vmax = float(np.max(heat)) if heat is not None else 0.0

    # Cells
    for r in range(g):
        for c in range(g):
            x1 = MARGIN + c*CELL_PX
            y1 = MARGIN + r*CELL_PX
            x2 = x1 + CELL_PX
            y2 = y1 + CELL_PX

            # Heat fill
            if heat is not None and vmax > 0:
                t = float(heat[r, c]) / vmax
                color = heat_color(t) if state[r, c] == UNKNOWN else (80, 80, 80)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            else:
                base = (60, 60, 60) if state[r, c] == UNKNOWN else (80, 80, 80)
                cv2.rectangle(img, (x1, y1), (x2, y2), base, thickness=-1)

            # Grid lines
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), thickness=1)

            # Mark MISS
            if state[r, c] == MISS:
                cv2.line(img, (x1+6, y1+6), (x2-6, y2-6), (0, 180, 255), 2)
                cv2.line(img, (x1+6, y2-6), (x2-6, y1+6), (0, 180, 255), 2)

    # Legend / instructions
    cv2.putText(img, "Battleship Heatmap Demo (MISS on click)", (MARGIN, 18), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "Ships: 5,4,3,3,2   R: reset   ESC/Q: quit", (MARGIN, h-10), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if vmax > 0:
        cv2.putText(img, f"max={int(vmax)}", (w-100, 18), FONT, 0.55, (240,240,240), 1, cv2.LINE_AA)

    # Draw a rocket emoji-ish icon on the best target (highest UNKNOWN heat)
    if heat is not None and vmax > 0:
        best_val = -1.0
        best_rc = None
        for r in range(g):
            for c in range(g):
                if state[r, c] == UNKNOWN:
                    v = float(heat[r, c])
                    if v > best_val:
                        best_val = v
                        best_rc = (r, c)
        if best_rc is not None and best_val > 0:
            r, c = best_rc
            cx = MARGIN + c*CELL_PX + CELL_PX//2
            cy = MARGIN + r*CELL_PX + CELL_PX//2
            draw_rocket(img, (cx, cy), int(CELL_PX*0.65))
    return img


def draw_rocket(img, center, size):
    """Draw a simple rocket icon centered at center (x,y) with approx height=size."""
    cx, cy = center
    h = size
    w = int(h*0.42)
    body_top = cy - h//2
    body_bot = cy + h//2 - int(h*0.15)
    body_left = cx - w//2
    body_right = cx + w//2

    # Nose cone
    nose_h = int(h*0.22)
    pts_nose = np.array([
        [cx, body_top - 1],
        [body_left, body_top + nose_h],
        [body_right, body_top + nose_h]
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, pts_nose, (0, 0, 230))  # Red cone (BGR)

    # Body
    cv2.rectangle(img, (body_left, body_top + nose_h), (body_right, body_bot), (230, 230, 230), thickness=-1)

    # Window
    win_r = max(3, int(w*0.22))
    cv2.circle(img, (cx, body_top + nose_h + int(h*0.22)), win_r, (230, 180, 60), thickness=-1)  # Yellow ring
    cv2.circle(img, (cx, body_top + nose_h + int(h*0.22)), max(2, win_r-2), (200, 80, 20), thickness=-1)  # Orange glass

    # Fins
    fin_h = int(h*0.18)
    fin_w = int(w*0.6)
    base_y = body_bot
    pts_fin_l = np.array([[body_left, base_y - fin_h], [body_left - fin_w//2, base_y + fin_h//3], [body_left, base_y + fin_h]], np.int32)
    pts_fin_r = np.array([[body_right, base_y - fin_h], [body_right + fin_w//2, base_y + fin_h//3], [body_right, base_y + fin_h]], np.int32)
    cv2.fillConvexPoly(img, pts_fin_l, (0, 0, 220))  # Red fins
    cv2.fillConvexPoly(img, pts_fin_r, (0, 0, 220))

    # Flame
    flame_h = int(h*0.22)
    flame_w = int(w*0.6)
    pts_flame = np.array([
        [cx, body_bot + flame_h],
        [cx - flame_w//2, body_bot],
        [cx + flame_w//2, body_bot]
    ], np.int32)
    cv2.fillConvexPoly(img, pts_flame, (0, 180, 255))  # Orange flame


class Demo:
    def __init__(self):
        self.state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.heat = compute_heat(self.state)
        self.window = "Heatmap"
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Convert pixel -> cell
        gx = x - MARGIN
        gy = y - MARGIN
        if gx < 0 or gy < 0:
            return
        c = gx // CELL_PX
        r = gy // CELL_PX
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            # Toggle to MISS always (as requested)
            self.state[r, c] = MISS
            self.heat = compute_heat(self.state)

    def reset(self):
        self.state.fill(UNKNOWN)
        self.heat = compute_heat(self.state)

    def loop(self):
        while True:
            img = draw_board(self.state, self.heat)
            cv2.imshow(self.window, img)
            k = cv2.waitKey(16) & 0xFF
            if k == 27 or k == ord('q'):
                break
            elif k == ord('r'):
                self.reset()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Demo().loop()
