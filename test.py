import time
import cv2
import numpy as np
import mss
import pygetwindow as gw
import win32gui
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
from queue import Queue
from collections import deque

# --- Launch BlueStacks (unchanged) ---
subprocess.Popen([
    r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
    "--instance", "Pie64",
    "--cmd", "launchApp",
    "--package", "de.smuttlewerk.fleetbattle",
    "--source", "desktop_shortcut"
])

# === CONFIG ===
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
     "sqdiff_thresh":0.65, "uniqueness_min":0.01, "color":GREEN},
    {"name":"Enemy Turn", "path":r"ScreenElements\Enemy Turn.png",
     "sqdiff_thresh":0.65, "uniqueness_min":0.01, "color":ORANGE},
]

MIN_MASK_AREA = 500
MAX_WORKERS = min(4, cpu_count())
SCALE_FACTOR = 0.5  # Further reduced for even faster processing
DETECTION_FPS_TARGET = 8   # Reduced detection FPS for more stable performance
DISPLAY_FPS_TARGET = 60    # Keep display smooth
CAPTURE_FPS_TARGET = 45    # Dedicated capture thread FPS

# Global variables
cached_monitor = None
cached_window_rect = None
last_rect_check = 0
RECT_CHECK_INTERVAL = 1.0

# Shared data between threads with better separation
latest_display_frame = None
display_frame_lock = threading.Lock()

current_matches = []
matches_lock = threading.Lock()

detection_frame_queue = Queue(maxsize=1)  # Only latest frame for detection
stop_threads = threading.Event()

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
    # Try to load the image
    tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        # Try with forward slashes
        path_alt = path.replace('\\', '/')
        tpl = cv2.imread(path_alt, cv2.IMREAD_UNCHANGED)
        if tpl is None:
            raise FileNotFoundError(f"Could not load template image: {path} (also tried: {path_alt})")
    
    print(f"Loaded template {path}: shape={tpl.shape}, dtype={tpl.dtype}")
    
    # Handle different image formats
    if tpl.ndim == 3 and tpl.shape[2] == 4:
        # RGBA image
        bgr = tpl[:, :, :3]
        alpha = tpl[:, :, 3]
        mask = (alpha > alpha_cutoff).astype(np.uint8) * 255
        print(f"  -> RGBA image, alpha pixels: {np.count_nonzero(alpha > alpha_cutoff)}")
    elif tpl.ndim == 3 and tpl.shape[2] == 3:
        # RGB/BGR image
        bgr = tpl
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        print(f"  -> RGB/BGR image, full mask")
    elif tpl.ndim == 2:
        # Grayscale image
        bgr = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        print(f"  -> Grayscale image, full mask")
    else:
        raise ValueError(f"Unsupported image format: {tpl.shape} for {path}")
    
    if scale_factor != 1.0:
        h, w = bgr.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        if new_h > 0 and new_w > 0:
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            print(f"  -> Scaled to {new_w}x{new_h}")
        else:
            print(f"  -> Warning: Invalid scale factor {scale_factor} for size {w}x{h}")
    
    return bgr, mask

def prepare_templates(scale_factor=1.0):
    prepared = []
    print(f"\n=== Loading Templates with scale factor {scale_factor} ===")
    
    for i, item in enumerate(TEMPLATES):
        try:
            print(f"\n[{i+1}/{len(TEMPLATES)}] Processing: {item['name']}")
            print(f"  Path: {item['path']}")
            
            # Check if file exists
            import os
            if not os.path.exists(item["path"]):
                alt_path = item["path"].replace('\\', '/')
                if not os.path.exists(alt_path):
                    print(f"  ERROR: File not found at {item['path']} or {alt_path}")
                    continue
                else:
                    print(f"  Found at alternate path: {alt_path}")
                    item["path"] = alt_path
            
            tpl, m = load_template_and_mask(item["path"], ALPHA_CUTOFF, scale_factor)
            th, tw = tpl.shape[:2]
            mask_area = int(np.count_nonzero(m))
            
            print(f"  Template size: {tw}x{th}")
            print(f"  Mask area: {mask_area} pixels")
            print(f"  Min mask area threshold: {MIN_MASK_AREA}")
            
            template_data = {
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
            }
            
            if mask_area >= MIN_MASK_AREA:
                prepared.append(template_data)
                print(f"  SUCCESS: Template added to processing list")
            else:
                print(f"  SKIPPED: Mask area {mask_area} < {MIN_MASK_AREA}")
                
        except Exception as e:
            print(f"  ERROR loading {item['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== Template Loading Complete ===")
    print(f"Successfully loaded {len(prepared)} out of {len(TEMPLATES)} templates")
    for i, t in enumerate(prepared):
        print(f"  [{i+1}] {t['name']} - {t['size'][0]}x{t['size'][1]} - {t['mask_area']} mask pixels")
    print("=" * 50)
    
    return prepared

def fast_best_and_second_best(result, tw, th):
    """Optimized version using numpy operations"""
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
    min2 = np.min(result_copy)
    
    return min_loc, min_val, min2

def process_single_template(frame, template):
    """Optimized single template processing with debug info"""
    try:
        # Validate inputs
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

        # Debug info (you can remove this later)
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
            # Return debug info for non-matches too (temporarily)
            return {
                "matched": False,
                "name": template["name"],
                "debug": debug_info
            }
            
    except Exception as e:
        print(f"Error processing template {template.get('name', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
    
    return {"matched": False, "error": str(e) if 'e' in locals() else "Unknown error"}

def capture_thread(win):
    """Dedicated thread for screen capture only"""
    global latest_display_frame, cached_monitor, cached_window_rect, last_rect_check
    
    print("Capture thread started")
    capture_frame_time = 1.0 / CAPTURE_FPS_TARGET
    
    with mss.mss() as sct:
        while not stop_threads.is_set():
            capture_start = time.time()
            
            # Update window rect less frequently
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
                time.sleep(0.1)
                continue
            
            try:
                # Capture frame
                img = np.array(sct.grab(cached_monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Update display frame (thread-safe)
                with display_frame_lock:
                    latest_display_frame = frame.copy()
                
                # Send scaled frame for detection (non-blocking)
                if SCALE_FACTOR != 1.0:
                    h_orig, w_orig = frame.shape[:2]
                    h_new = int(h_orig * SCALE_FACTOR)
                    w_new = int(w_orig * SCALE_FACTOR)
                    detection_frame = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_AREA)
                else:
                    detection_frame = frame.copy()
                
                # Non-blocking put - if detection is busy, skip this frame
                try:
                    detection_frame_queue.put_nowait(detection_frame)
                except:
                    pass  # Detection queue full, skip this frame
                    
            except Exception as e:
                print(f"Capture thread error: {e}")
            
            # Frame rate limiting
            capture_time = time.time() - capture_start
            if capture_time < capture_frame_time:
                time.sleep(capture_frame_time - capture_time)

def detection_thread(templates):
    """Background thread for template detection only"""
    global current_matches
    
    print(f"Detection thread started with {len(templates)} templates")
    if len(templates) == 0:
        print("WARNING: No templates loaded for detection!")
        return
        
    detection_frame_time = 1.0 / DETECTION_FPS_TARGET
    detection_count = 0
    last_debug_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while not stop_threads.is_set():
            detection_start = time.time()
            
            try:
                # Get latest frame for detection (blocking with timeout)
                try:
                    frame = detection_frame_queue.get(timeout=0.1)
                except:
                    continue  # No frame available, continue loop
                
                # Clear queue to always process latest frame
                while not detection_frame_queue.empty():
                    try:
                        frame = detection_frame_queue.get_nowait()
                    except:
                        break
                
                # Process templates in parallel
                if len(templates) > 1:
                    futures = [executor.submit(process_single_template, frame, t) for t in templates]
                    matches = [future.result() for future in futures]
                else:
                    matches = [process_single_template(frame, templates[0])]
                
                # Debug output every 5 seconds
                detection_count += 1
                current_time = time.time()
                if current_time - last_debug_time >= 5.0:
                    print(f"\n=== Detection Debug (frame {detection_count}) ===")
                    for i, match in enumerate(matches):
                        if "debug" in match:
                            debug = match["debug"]
                            status = "MATCH" if match["matched"] else "NO_MATCH"
                            print(f"  [{i+1}] {match.get('name', 'unknown')} - {status}")
                            print(f"      best={debug['best']:.4f} (thresh={debug['thresh']:.4f})")
                            print(f"      uniq={debug['uniqueness']:.4f} (min={debug['uniq_min']:.4f})")
                            print(f"      conf={debug['confidence']:.4f}")
                    last_debug_time = current_time
                
                # Update shared matches data (thread-safe)
                with matches_lock:
                    current_matches.clear()
                    current_matches.extend([m for m in matches if m["matched"]])
                
            except Exception as e:
                print(f"Detection thread error: {e}")
                import traceback
                traceback.print_exc()
            
            # Frame rate limiting for detection
            detection_time = time.time() - detection_start
            if detection_time < detection_frame_time:
                time.sleep(detection_frame_time - detection_time)

def display_thread():
    """Main display thread for smooth video"""
    global latest_display_frame, current_matches
    
    print("Display thread started")
    display_frame_time = 1.0 / DISPLAY_FPS_TARGET
    
    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0
    
    while not stop_threads.is_set():
        display_start = time.time()
        
        try:
            # Get latest display frame (thread-safe)
            display_frame = None
            with display_frame_lock:
                if latest_display_frame is not None:
                    display_frame = latest_display_frame.copy()
            
            if display_frame is not None:
                # Get current matches (thread-safe)
                with matches_lock:
                    matches_to_draw = current_matches.copy()
                
                # Draw matches
                for match in matches_to_draw:
                    tw, th = match["size"]
                    br = (match["top_left"][0] + tw, match["top_left"][1] + th)
                    cv2.rectangle(display_frame, match["top_left"], br, match["color"], RECT_THICKNESS)
                    
                    text = f'{match["name"]}: {match["confidence"]:.2f}'
                    cv2.putText(
                        display_frame, text,
                        (match["top_left"][0], max(15, match["top_left"][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, match["color"], 1, cv2.LINE_AA
                    )
                
                # FPS display
                if SHOW_FPS:
                    frame_count += 1
                    now = time.time()
                    if now - fps_start >= 1.0:
                        display_fps = frame_count / (now - fps_start)
                        frame_count = 0
                        fps_start = now
                    
                    # Show queue status
                    queue_size = detection_frame_queue.qsize()
                    cv2.putText(display_frame, 
                               f"Display: {display_fps:.1f} FPS | Matches: {len(matches_to_draw)} | Queue: {queue_size}", 
                               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                
                cv2.imshow("BlueStacks Live - Fully Threaded", display_frame)
                
        except Exception as e:
            print(f"Display thread error: {e}")
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            stop_threads.set()
            break
        
        # Frame rate limiting for display
        display_time = time.time() - display_start
        if display_time < display_frame_time:
            time.sleep(display_frame_time - display_time)

def main():
    global cached_monitor, cached_window_rect
    
    templates = prepare_templates(SCALE_FACTOR)
    templates = [t for t in templates if t["mask_area"] >= MIN_MASK_AREA]
    print(f"Processing {len(templates)} templates at {SCALE_FACTOR}x scale")
    print(f"Target: {DISPLAY_FPS_TARGET} FPS display, {DETECTION_FPS_TARGET} FPS detection, {CAPTURE_FPS_TARGET} FPS capture")

    win = find_bluestacks_window(WINDOW_TITLE_HINTS)
    if not win:
        raise RuntimeError("Could not find a BlueStacks window. Adjust WINDOW_TITLE_HINTS.")

    # Initialize window rect cache
    cached_window_rect = get_client_rect(win)
    left, top, right, bottom = cached_window_rect
    w = max(1, right - left)
    h = max(1, bottom - top)
    cached_monitor = {"left": left, "top": top, "width": w, "height": h}

    # Start background threads
    capture_worker = threading.Thread(target=capture_thread, args=(win,), daemon=True)
    detection_worker = threading.Thread(target=detection_thread, args=(templates,), daemon=True)
    
    capture_worker.start()
    detection_worker.start()
    
    # Run display in main thread
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