import time
import cv2
import numpy as np
import mss
import pygetwindow as gw
import win32gui
import win32api
import win32con
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
from queue import Queue
from collections import deque
from dataclasses import dataclass
from typing import Optional
import math

subprocess.Popen([
    r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
    "--instance", "Pie64",
    "--cmd", "launchApp",
    "--package", "de.smuttlewerk.fleetbattle",
    "--source", "desktop_shortcut"
])

time.sleep(5)

WINDOW_TITLE_HINTS = ["BlueStacks", "Pie64", "HD-Player"]
ALPHA_CUTOFF = 10
SHOW_FPS = True
METHOD = cv2.TM_SQDIFF_NORMED

RECT_THICKNESS = 10
PURPLE = (255, 0, 255)
CYAN   = (255, 255, 0)
GREEN  = (0, 200, 0)
ORANGE = (0, 140, 255)

TEMPLATES = [
    {"name":"Start Game Main Menu", "path":r"ScreenElements\Start Game Main Menu.png",
     "sqdiff_thresh":0.05, "uniqueness_min":0.01, "color":PURPLE},
    {"name":"Start Game Ready Up", "path":r"ScreenElements\Start Game Ready Up.png",
     "sqdiff_thresh":0.05, "uniqueness_min":0.01, "color":CYAN},
    {"name":"In Game Marker", "path":r"ScreenElements\In Game Marker.png",
     "sqdiff_thresh":0.15, "uniqueness_min":0.01, "color":GREEN},
    {"name":"Enemy Turn", "path":r"ScreenElements\Enemy Turn.png",
     "sqdiff_thresh":0.65, "uniqueness_min":0.01, "color":ORANGE},
]

MIN_MASK_AREA = 500
MAX_WORKERS = min(4, cpu_count())
SCALE_FACTOR = 0.5
DETECTION_FPS_TARGET = 8
DISPLAY_FPS_TARGET = 60
CAPTURE_FPS_TARGET = 45

cached_monitor = None
cached_window_rect = None
last_rect_check = 0
RECT_CHECK_INTERVAL = 1.0

latest_display_frame = None
display_frame_lock = threading.Lock()

current_matches = []
matches_lock = threading.Lock()

detection_frame_queue = Queue(maxsize=1)
stop_threads = threading.Event()

AUTO_PLAY_ENABLED = True
WAIT_AFTER_CLICK_SEC = 1.0

CLICK_TARGETS = ["Start Game Main Menu", "Start Game Ready Up"]
CLICK_HOLD_SEC = 0.05
VISIBLE_DURATION_SEC = 1.0
_clicked_flags = {name: False for name in CLICK_TARGETS}
_visible_since = {name: None for name in CLICK_TARGETS}
_target_hwnd = None

def hex_to_bgr(h: str):
    h = h.strip().lstrip('#')
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)

def bgr_to_lab(bgr):
    c = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[0,0,:].astype(np.float32)
    return lab

REF_BGR = {
    "HIT":  hex_to_bgr("EC242D"),
    "SUNK": hex_to_bgr("000000"),
    "MISS": hex_to_bgr("3F5887"),
    "BG":   hex_to_bgr("10306B"),
}
REF_LAB = {k: bgr_to_lab(v) for k,v in REF_BGR.items()}

THRESH = {
    "SUNK": 10.0,
    "HIT":  20.0,
    "MISS": 16.0,
    "BG":   18.0,
}

def lab_dist(a, b):
    return float(np.linalg.norm(a - b))

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3
STATE_NAME = {UNKNOWN:"UNK", MISS:"MISS", HIT:"HIT", SUNK:"SUNK"}

@dataclass
class GridParams:
    start_x: int = 1027+16
    start_y: int = 385+16
    dot_diameter: int = 71
    top_left_cx: Optional[int] = None
    top_left_cy: Optional[int] = None
    bottom_right_cx: Optional[int] = None
    bottom_right_cy: Optional[int] = None
    top_right_cx: Optional[int] = None
    top_right_cy: Optional[int] = None
    bottom_left_cx: Optional[int] = None
    bottom_left_cy: Optional[int] = None
    grid_size: int = 10
    sample_size: int = 10
    sample_radius: int = 8
    stable_frames: int = 2

class GridTracker:
    def __init__(self, params: GridParams):
        self.p = params
        g = self.p.grid_size
        self.state = np.full((g, g), UNKNOWN, dtype=np.uint8)
        self.history = [[deque(maxlen=self.p.stable_frames) for _ in range(g)] for __ in range(g)]
        self.last_seen_lab = np.zeros((g, g, 3), dtype=np.float32)
        self.last_reason = ""
        self.locked_miss = set()
        self.clicked_cells = set()
        self._precompute_centers()

    def _precompute_centers(self):
        self.centers = []
        g = self.p.grid_size
        if (self.p.top_left_cx is not None and self.p.top_left_cy is not None):
            tlx = float(self.p.top_left_cx); tly = float(self.p.top_left_cy)
            steps = max(1, g - 1)
            ex_x = ey_x = ex_y = ey_y = None
            if (self.p.top_right_cx is not None and self.p.top_right_cy is not None):
                ex_x = (float(self.p.top_right_cx) - tlx) / steps
                ex_y = (float(self.p.top_right_cy) - tly) / steps
            if (self.p.bottom_left_cx is not None and self.p.bottom_left_cy is not None):
                ey_x = (float(self.p.bottom_left_cx) - tlx) / steps
                ey_y = (float(self.p.bottom_left_cy) - tly) / steps
            if (ex_x is None or ex_y is None) or (ey_x is None or ey_y is None):
                if (self.p.bottom_right_cx is not None and self.p.bottom_right_cy is not None):
                    brx = float(self.p.bottom_right_cx); bry = float(self.p.bottom_right_cy)
                    if ex_x is None or ex_y is None:
                        ex_x = (brx - tlx) / steps
                        ex_y = 0.0
                    if ey_x is None or ey_y is None:
                        ey_x = 0.0
                        ey_y = (bry - tly) / steps
                else:
                    ex_x = float(self.p.dot_diameter); ex_y = 0.0
                    ey_x = 0.0; ey_y = float(self.p.dot_diameter)
            self._dx = math.hypot(ex_x, ex_y)
            self._dy = math.hypot(ey_x, ey_y)
            for row in range(g):
                row_centers = []
                for col in range(g):
                    cx = int(round(tlx + col * ex_x + row * ey_x))
                    cy = int(round(tly + col * ex_y + row * ey_y))
                    row_centers.append((cx, cy))
                self.centers.append(row_centers)
        else:
            r = self.p.dot_diameter // 2
            self._dx = float(self.p.dot_diameter)
            self._dy = float(self.p.dot_diameter)
            for row in range(g):
                row_centers = []
                for col in range(g):
                    cx = self.p.start_x + (col * self.p.dot_diameter) + r
                    cy = self.p.start_y + (row * self.p.dot_diameter) + r
                    row_centers.append((cx, cy))
                self.centers.append(row_centers)

    def _sample_lab(self, frame, row, col):
        h, w = frame.shape[:2]
        cx, cy = self.centers[row][col]
        if self.p.sample_size and self.p.sample_size > 0:
            half = self.p.sample_size // 2
            x1 = max(0, cx - half); x2 = min(w, x1 + self.p.sample_size)
            y1 = max(0, cy - half); y2 = min(h, y1 + self.p.sample_size)
        else:
            sr = self.p.sample_radius
            x1 = max(0, cx - sr); x2 = min(w, cx + sr)
            y1 = max(0, cy - sr); y2 = min(h, cy + sr)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        mean_bgr = tuple(map(int, np.mean(patch.reshape(-1,3), axis=0)))
        return bgr_to_lab(mean_bgr)

    def _classify_lab(self, lab_color):
        if lab_color is None:
            return None
        if lab_color[0] < 18:
            return SUNK
        d = {k: lab_dist(lab_color, REF_LAB[k]) for k in REF_LAB.keys()}
        for name in ["SUNK","HIT","MISS","BG"]:
            if d[name] <= THRESH[name]:
                if name == "SUNK": return SUNK
                if name == "HIT":  return HIT
                if name == "MISS": return MISS
                return UNKNOWN
        return UNKNOWN

    def update(self, frame_bgr):
        g = self.p.grid_size
        for r in range(g):
            for c in range(g):
                labc = self._sample_lab(frame_bgr, r, c)
                if labc is not None:
                    self.last_seen_lab[r, c] = labc
                cls = self._classify_lab(labc)
                if cls is None:
                    continue
                self.history[r][c].append(cls)
                if len(self.history[r][c]) == self.p.stable_frames and len(set(self.history[r][c])) == 1:
                    committed = self.history[r][c][0]
                    if (r, c) in self.clicked_cells or (r, c) in self.locked_miss:
                        if not (
                            (self.state[r, c] in (SUNK, HIT) and committed == UNKNOWN)
                            or (((r, c) in self.locked_miss) and committed == UNKNOWN)
                        ):
                            self.state[r, c] = committed

    def _nbrs4(self, r, c):
        return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]

    def _is_unknown(self, r, c):
        g = self.p.grid_size
        return 0 <= r < g and 0 <= c < g and self.state[r, c] == UNKNOWN

    def _is_miss(self, r, c):
        g = self.p.grid_size
        return 0 <= r < g and 0 <= c < g and self.state[r, c] == MISS

    def _clusters(self, cells):
        gset = set(cells)
        clusters = []
        while gset:
            start = gset.pop()
            comp = [start]
            dq = deque([start])
            while dq:
                r,c = dq.popleft()
                for rr,cc in self._nbrs4(r,c):
                    if (rr,cc) in gset:
                        gset.remove((rr,cc))
                        comp.append((rr,cc))
                        dq.append((rr,cc))
            clusters.append(comp)
        return clusters

    def _best_target(self, candidates):
        best = (-1, None, None)
        why = ""
        for (r,c,label) in candidates:
            unk = sum(self._is_unknown(rr,cc) for rr,cc in self._nbrs4(r,c))
            if unk > best[0]:
                best = (unk, r, c); why = f"{label}, unknown-neighbors={unk}"
        return (best[1], best[2], why)

    def suggest_next(self):
        unknown_cells = list(zip(*np.where(self.state == UNKNOWN)))
        if not unknown_cells:
            self.last_reason = "No UNKNOWN cells; board seems complete."
            return None

        hits = list(zip(*np.where(self.state == HIT)))
        if hits:
            clusters = self._clusters(hits)
            clusters.sort(key=len, reverse=True)
            for cluster in clusters:
                rows = sorted(set([r for r,_ in cluster]))
                cols = sorted(set([c for _,c in cluster]))
                if len(rows) == 1:
                    r = rows[0]
                    minc = min(c for _,c in cluster)
                    maxc = max(c for _,c in cluster)
                    candidates = []
                    if self._is_unknown(r, minc-1): candidates.append((r, minc-1, "extend left"))
                    if self._is_unknown(r, maxc+1): candidates.append((r, maxc+1, "extend right"))
                    if candidates:
                        rc, cc, why = self._best_target(candidates)
                        self.last_reason = f"Extending horizontal hit at row {r}: {why}"
                        return (rc, cc, 100, self.last_reason)
                elif len(cols) == 1:
                    c = cols[0]
                    minr = min(r for r,_ in cluster)
                    maxr = max(r for r,_ in cluster)
                    candidates = []
                    if self._is_unknown(minr-1, c): candidates.append((minr-1, c, "extend up"))
                    if self._is_unknown(maxr+1, c): candidates.append((maxr+1, c, "extend down"))
                    if candidates:
                        rc, cc, why = self._best_target(candidates)
                        self.last_reason = f"Extending vertical hit at col {c}: {why}"
                        return (rc, cc, 100, self.last_reason)
                for (r,c) in cluster:
                    neighbors = [(r-1,c,"up"),(r+1,c,"down"),(r,c-1,"left"),(r,c+1,"right")]
                    cand = [(rr,cc,lab) for (rr,cc,lab) in neighbors if self._is_unknown(rr,cc)]
                    if cand:
                        rc, cc, why = self._best_target(cand)
                        self.last_reason = f"Probing around hit at ({r},{c}): {why}"
                        return (rc, cc, 90, self.last_reason)

        best = (-1e9, None, None)
        for (r,c) in unknown_cells:
            parity = 1 if (r + c) % 2 == 0 else 0
            unk_n = sum(self._is_unknown(rr,cc) for rr,cc in self._nbrs4(r,c))
            miss_close = sum(self._is_miss(rr,cc) for rr,cc in self._nbrs4(r,c))
            score = parity*2 + unk_n*1.5 - miss_close*0.5
            if score > best[0]:
                best = (score, r, c)
        _, br, bc = best
        self.last_reason = "Hunt mode: parity + local unknown-neighbor density."
        return (br, bc, float(best[0]), self.last_reason)

    def render_overlay(self, frame):
        overlay = frame.copy()
        spacing_x = getattr(self, "_dx", float(self.p.dot_diameter))
        spacing_y = getattr(self, "_dy", float(self.p.dot_diameter))
        r = max(6, int(min(spacing_x, spacing_y) // 2) - 16)
        for row in range(self.p.grid_size):
            for col in range(self.p.grid_size):
                cx, cy = self.centers[row][col]
                st = self.state[row, col]
                if st == UNKNOWN: color = (180,180,180)
                elif st == MISS:  color = (200,180,80)
                elif st == HIT:   color = (0,0,255)
                else:             color = (0,0,0)
                cv2.circle(overlay, (cx,cy), r, color, 2)
        sug = self.suggest_next()
        if sug:
            rr, cc, score, _ = sug
            cx, cy = self.centers[rr][cc]
            cv2.drawMarker(overlay, (cx,cy), (0,255,0), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(overlay, f"Next: ({rr},{cc}) score={score:.1f}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(overlay, self.last_reason, (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
        return overlay

    def cell_center_px(self, row, col):
        return self.centers[row][col]

    def force_mark_miss(self, row, col):
        g = self.p.grid_size
        if 0 <= row < g and 0 <= col < g:
            self.state[row, col] = MISS
            self.history[row][col].clear()
            self.history[row][col].extend([MISS] * self.p.stable_frames)
            self.locked_miss.add((row, col))
            self.clicked_cells.add((row, col))

    def mark_clicked(self, row, col):
        g = self.p.grid_size
        if 0 <= row < g and 0 <= col < g:
            self.clicked_cells.add((row, col))

grid_tracker = GridTracker(GridParams(
    top_left_cx=1086,
    top_left_cy=427,
    bottom_right_cx=1752,
    bottom_right_cy=1092,
    grid_size=10,
    sample_size=1,
    stable_frames=2
))

def find_bluestacks_window(title_hints):
    wins = gw.getAllTitles()
    matches = [t for t in wins if any(h.lower() in t.lower() for h in title_hints)]
    if not matches: return None
    matches.sort(key=len, reverse=True)
    win = gw.getWindowsWithTitle(matches[0])[0]
    try: win.activate(); time.sleep(0.05)
    except Exception: pass
    return win

def get_client_rect(win):
    hwnd = win._hWnd
    rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = left + rect[2], top + rect[3]
    return left, top, right, bottom

def load_template_and_mask(path, alpha_cutoff=10, scale_factor=1.0):
    tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        path_alt = path.replace('\\', '/')
        tpl = cv2.imread(path_alt, cv2.IMREAD_UNCHANGED)
        if tpl is None:
            raise FileNotFoundError(f"Could not load template image: {path} (also tried: {path_alt})")
    if tpl.ndim == 3 and tpl.shape[2] == 4:
        bgr = tpl[:, :, :3]
        alpha = tpl[:, :, 3]
        mask = (alpha > alpha_cutoff).astype(np.uint8) * 255
    elif tpl.ndim == 3 and tpl.shape[2] == 3:
        bgr = tpl
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif tpl.ndim == 2:
        bgr = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image format: {tpl.shape} for {path}")
    if scale_factor != 1.0:
        h, w = bgr.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        if new_h > 0 and new_w > 0:
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return bgr, mask

def prepare_templates(scale_factor=1.0):
    prepared = []
    for item in TEMPLATES:
        try:
            import os
            if not os.path.exists(item["path"]):
                alt_path = item["path"].replace('\\', '/')
                if os.path.exists(alt_path):
                    item["path"] = alt_path
                else:
                    print(f"[WARN] Missing template: {item['path']}")
                    continue
            tpl, m = load_template_and_mask(item["path"], ALPHA_CUTOFF, scale_factor)
            th, tw = tpl.shape[:2]
            mask_area = int(np.count_nonzero(m))
            if mask_area >= MIN_MASK_AREA:
                prepared.append({
                    "name": item["name"],
                    "tpl": tpl,
                    "mask": m,
                    "size": (tw, th),
                    "sqdiff_thresh": item["sqdiff_thresh"],
                    "uniqueness_min": item["uniqueness_min"],
                    "color": item["color"],
                    "mask_area": mask_area,
                    "scale_factor": scale_factor,
                    "original_path": item["path"]
                })
        except Exception as e:
            print(f"[ERR] Loading {item['name']}: {e}")
    print(f"Templates loaded: {len(prepared)}/{len(TEMPLATES)}")
    return prepared

def fast_best_and_second_best(result, tw, th):
    min_idx = np.unravel_index(np.argmin(result), result.shape)
    min_val = result[min_idx]
    min_loc = (min_idx[1], min_idx[0])
    y0, x0 = min_idx
    y1 = max(0, y0 - th//4)
    y2 = min(result.shape[0], y0 + th//4 + 1)
    x1 = max(0, x0 - tw//4)
    x2 = min(result.shape[1], x0 + tw//4 + 1)
    result_copy = result.copy()
    result_copy[y1:y2, x1:x2] = 1.0
    min2 = float(np.min(result_copy))
    return min_loc, float(min_val), min2

def process_single_template(frame, template):
    try:
        if frame is None or template["tpl"] is None:
            return {"matched": False, "error": "Invalid input"}
        if frame.shape[2] != 3 or template["tpl"].shape[2] != 3:
            return {"matched": False, "error": "Channel mismatch"}
        res = cv2.matchTemplate(frame, template["tpl"], METHOD, mask=template["mask"])
        if res.size == 0:
            return {"matched": False, "error": "Empty result"}
        top_left, best, second = fast_best_and_second_best(res, *template["size"])
        uniqueness = second - best
        confidence = 1.0 - float(best)
        debug_info = {
            "best": best,
            "second": second,
            "uniqueness": uniqueness,
            "confidence": confidence,
            "thresh": template["sqdiff_thresh"],
            "uniq_min": template["uniqueness_min"]
        }
        if best <= template["sqdiff_thresh"] and uniqueness >= template["uniqueness_min"]:
            scale = template["scale_factor"]
            if scale != 1.0:
                top_left = (int(top_left[0] / scale), int(top_left[1] / scale))
                size = (int(template["size"][0] / scale), int(template["size"][1] / scale))
            else:
                size = template["size"]
            return {
                "name": template["name"],
                "top_left": top_left,
                "size": size,
                "color": template["color"],
                "confidence": confidence,
                "best": best,
                "uniqueness": uniqueness,
                "matched": True,
                "debug": debug_info
            }
        else:
            return {"matched": False, "name": template["name"], "debug": debug_info}
    except Exception as e:
        print(f"Error processing template {template.get('name', 'unknown')}: {e}")
    return {"matched": False, "error": "Unknown error"}

def capture_thread(win):
    global latest_display_frame, cached_monitor, cached_window_rect, last_rect_check
    capture_frame_time = 1.0 / CAPTURE_FPS_TARGET
    with mss.mss() as sct:
        while not stop_threads.is_set():
            capture_start = time.time()
            current_time = time.time()
            if current_time - last_rect_check > RECT_CHECK_INTERVAL:
                try:
                    cached_window_rect = get_client_rect(win)
                    left, top, right, bottom = cached_window_rect
                    w = max(1, right - left)
                    h = max(1, bottom - top)
                    cached_monitor = {"left": left, "top": top, "width": w, "height": h}
                    last_rect_check = current_time
                except Exception:
                    pass
            if cached_monitor is None:
                time.sleep(0.05)
                continue
            try:
                img = np.array(sct.grab(cached_monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                with display_frame_lock:
                    latest_display_frame = frame.copy()
                if SCALE_FACTOR != 1.0:
                    h_orig, w_orig = frame.shape[:2]
                    h_new = int(h_orig * SCALE_FACTOR)
                    w_new = int(w_orig * SCALE_FACTOR)
                    detection_frame = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_AREA)
                else:
                    detection_frame = frame.copy()
                try:
                    detection_frame_queue.put_nowait(detection_frame)
                except:
                    pass
            except Exception as e:
                print(f"Capture thread error: {e}")
            capture_time = time.time() - capture_start
            if capture_time < capture_frame_time:
                time.sleep(capture_frame_time - capture_time)

def detection_thread(templates):
    global current_matches
    if len(templates) == 0:
        print("WARNING: No templates loaded for detection!")
        return
    detection_frame_time = 1.0 / DETECTION_FPS_TARGET
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while not stop_threads.is_set():
            detection_start = time.time()
            try:
                try:
                    frame = detection_frame_queue.get(timeout=0.1)
                except:
                    continue
                while not detection_frame_queue.empty():
                    try:
                        frame = detection_frame_queue.get_nowait()
                    except:
                        break
                if len(templates) > 1:
                    futures = [executor.submit(process_single_template, frame, t) for t in templates]
                    matches = [future.result() for future in futures]
                else:
                    matches = [process_single_template(frame, templates[0])]
                with matches_lock:
                    current_matches.clear()
                    current_matches.extend([m for m in matches if m.get("matched")])
            except Exception as e:
                print(f"Detection thread error: {e}")
            detection_time = time.time() - detection_start
            if detection_time < detection_frame_time:
                time.sleep(detection_frame_time - detection_time)

def save_screenshot(frame_with_rectangles, matches):
    try:
        import os
        from datetime import datetime
        screenshot_dir = "screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_screenshot_{timestamp}.png"
        filepath = os.path.join(screenshot_dir, filename)
        success = cv2.imwrite(filepath, frame_with_rectangles)
        if success:
            print(f"\nSaved: {filepath}")
            print(f"Matches captured: {len(matches)}")
            for i, match in enumerate(matches):
                print(f"  [{i+1}] {match['name']} at {match['top_left']} - conf: {match['confidence']:.3f}")
        else:
            print(f"ERROR: Failed to save screenshot to {filepath}")
    except Exception as e:
        print(f"Error saving screenshot: {e}")

def check_in_game_active(matches):
    for match in matches:
        if match.get("name") == "In Game Marker":
            return True
    return False

def check_enemy_turn(matches):
    for match in matches:
        if match.get("name") == "Enemy Turn":
            return True
    return False

def _click_center_of_match(match):
    global cached_monitor, _target_hwnd
    try:
        if cached_monitor is None:
            return
        tlx, tly = match["top_left"]
        tw, th = match["size"]
        cx = cached_monitor["left"] + int(tlx + tw/2)
        cy = cached_monitor["top"] + int(tly + th/2)
        if _target_hwnd:
            try:
                win32gui.SetForegroundWindow(_target_hwnd)
            except Exception:
                pass
        win32api.SetCursorPos((cx, cy))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(CLICK_HOLD_SEC)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        print(f"Clicked '{match['name']}' at screen ({cx},{cy})")
    except Exception as e:
        print(f"Click error for {match.get('name','?')}: {e}")

def _update_click_state(matches):
    now = time.time()
    present = {name: None for name in CLICK_TARGETS}
    for m in matches:
        n = m.get("name")
        if n in present:
            present[n] = m
    for name in CLICK_TARGETS:
        if _clicked_flags[name]:
            continue
        m = present[name]
        if m is not None:
            if _visible_since[name] is None:
                _visible_since[name] = now
            else:
                if now - _visible_since[name] >= VISIBLE_DURATION_SEC:
                    _click_center_of_match(m)
                    _clicked_flags[name] = True
                    _visible_since[name] = None
        else:
            _visible_since[name] = None

def display_thread():
    global latest_display_frame, current_matches
    print("Display thread started")
    print("[SPACE] screenshot | [N] print next move | [ESC/Q] quit")
    display_frame_time = 1.0 / DISPLAY_FPS_TARGET
    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0
    while not stop_threads.is_set():
        display_start = time.time()
        try:
            display_frame = None
            original_frame = None
            with display_frame_lock:
                if latest_display_frame is not None:
                    original_frame = latest_display_frame.copy()
                    display_frame = latest_display_frame.copy()
            if display_frame is not None and original_frame is not None:
                with matches_lock:
                    matches_to_draw = current_matches.copy()
                in_game_active = check_in_game_active(matches_to_draw)

                _update_click_state(matches_to_draw)

                if in_game_active:
                    grid_tracker.update(original_frame)
                    display_frame = grid_tracker.render_overlay(display_frame)

                for match in matches_to_draw:
                    tw, th = match["size"]
                    br = (match["top_left"][0] + tw, match["top_left"][1] + th)
                    cv2.rectangle(display_frame, match["top_left"], br, match["color"], RECT_THICKNESS)
                    text = f'{match["name"]}: {match["confidence"]:.2f}'
                    cv2.putText(display_frame, text,
                                (match["top_left"][0], max(15, match["top_left"][1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, match["color"], 1, cv2.LINE_AA)

                if SHOW_FPS:
                    frame_count += 1
                    now = time.time()
                    if now - fps_start >= 1.0:
                        display_fps = frame_count / (now - fps_start)
                        frame_count = 0
                        fps_start = now
                    queue_size = detection_frame_queue.qsize()
                    grid_status = "ON" if in_game_active else "OFF"
                    cv2.putText(display_frame,
                               f"Display: {display_fps:.1f} FPS | Matches: {len(matches_to_draw)} | Queue: {queue_size} | Grid: {grid_status}",
                               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame,
                               "SPACE: screenshot   N: print next move   ESC/Q: quit",
                               (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                cv2.imshow("BlueStacks Live - Fully Threaded (Auto-Click)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    stop_threads.set()
                    break
                elif key == ord(' '):
                    save_screenshot(display_frame, matches_to_draw)
                elif key == ord('n'):
                    sug = grid_tracker.suggest_next()
                    if sug:
                        r, c, score, reason = sug
                        px, py = grid_tracker.cell_center_px(r, c)
                        print(f"Next move -> row={r}, col={c}, score={score:.1f}, px=({px},{py}) | {reason}")
                    else:
                        print("No next move: board has no UNKNOWN cells.")
        except Exception as e:
            print(f"Display thread error: {e}")

        display_time = time.time() - display_start
        if display_time < display_frame_time:
            time.sleep(display_frame_time - display_time)

def _click_screen_point(cx, cy, label="cell"):
    global _target_hwnd
    try:
        if _target_hwnd:
            try:
                win32gui.SetForegroundWindow(_target_hwnd)
            except Exception:
                pass
        win32api.SetCursorPos((int(cx), int(cy)))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(CLICK_HOLD_SEC)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        print(f"Clicked {label} at screen ({int(cx)},{int(cy)})")
        return True
    except Exception as e:
        print(f"Click error at ({cx},{cy}): {e}")
        return False

def click_cell(row, col):
    global cached_monitor
    if cached_monitor is None:
        return False
    try:
        cx, cy = grid_tracker.cell_center_px(row, col)
        sx = cached_monitor["left"] + int(cx)
        sy = cached_monitor["top"] + int(cy)
        return _click_screen_point(sx, sy, label=f"cell ({row},{col})")
    except Exception as e:
        print(f"click_cell error for ({row},{col}): {e}")
        return False

def autoplay_thread():
    if not AUTO_PLAY_ENABLED:
        return
    print("Autoplay thread started")
    pending_cell = None
    wait_until = 0.0
    while not stop_threads.is_set():
        try:
            with matches_lock:
                matches = current_matches.copy()

            in_game = check_in_game_active(matches)
            enemy_turn = check_enemy_turn(matches)

            if not in_game:
                pending_cell = None
                time.sleep(0.1)
                continue

            now = time.time()
            if pending_cell is not None:
                if now < wait_until:
                    time.sleep(0.05)
                    continue
                r, c = pending_cell
                st = grid_tracker.state[r, c]
                if st in (HIT, SUNK):
                    print(f"Result: HIT at ({r},{c}) -> taking another shot")
                elif st == MISS:
                    print(f"Result: MISS at ({r},{c}) -> opponent's turn")
                else:
                    print(f"Result: undecided at ({r},{c}) -> assuming MISS and continuing")
                    grid_tracker.force_mark_miss(r, c)
                pending_cell = None

            if enemy_turn:
                time.sleep(0.15)
                continue

            sug = grid_tracker.suggest_next()
            if not sug:
                time.sleep(0.25)
                continue
            r, c, score, reason = sug
            if grid_tracker.state[r, c] != UNKNOWN:
                time.sleep(0.05)
                continue

            grid_tracker.mark_clicked(r, c)
            ok = click_cell(r, c)
            if ok:
                pending_cell = (r, c)
                wait_until = time.time() + WAIT_AFTER_CLICK_SEC
            else:
                time.sleep(0.2)
        except Exception as e:
            print(f"Autoplay error: {e}")
            time.sleep(0.2)

def main():
    global cached_monitor, cached_window_rect, _target_hwnd
    templates = prepare_templates(SCALE_FACTOR)
    templates = [t for t in templates if t["mask_area"] >= MIN_MASK_AREA]
    print(f"Processing {len(templates)} templates at {SCALE_FACTOR}x scale")
    print(f"Targets: display {DISPLAY_FPS_TARGET} FPS | detection {DETECTION_FPS_TARGET} FPS | capture {CAPTURE_FPS_TARGET} FPS")

    win = find_bluestacks_window(WINDOW_TITLE_HINTS)
    if not win:
        raise RuntimeError("Could not find a BlueStacks window. Adjust WINDOW_TITLE_HINTS.")

    _target_hwnd = win._hWnd

    cached_window_rect = get_client_rect(win)
    left, top, right, bottom = cached_window_rect
    w = max(1, right - left)
    h = max(1, bottom - top)
    cached_monitor = {"left": left, "top": top, "width": w, "height": h}

    capture_worker = threading.Thread(target=capture_thread, args=(win,), daemon=True)
    detection_worker = threading.Thread(target=detection_thread, args=(templates,), daemon=True)
    autoplay_worker = threading.Thread(target=autoplay_thread, daemon=True)
    capture_worker.start()
    detection_worker.start()
    autoplay_worker.start()

    try:
        display_thread()
    except KeyboardInterrupt:
        pass
    finally:
        stop_threads.set()
        cv2.destroyAllWindows()
        print("All threads stopped, windows closed")

if __name__ == "__main__":
    main()