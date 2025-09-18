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
