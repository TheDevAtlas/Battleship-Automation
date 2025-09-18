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
