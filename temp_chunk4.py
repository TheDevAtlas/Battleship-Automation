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
