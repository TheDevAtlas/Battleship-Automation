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

