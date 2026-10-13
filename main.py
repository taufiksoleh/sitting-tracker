"""Sitting Tracker: deteksi customer (YOLO), Re-ID wajah anonim, durasi duduk realtime."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import yaml

from dwell import DwellTracker


def load_zones(path):
    try:
        with open(path, encoding="utf-8") as f:
            zones = json.load(f)["zones"]
    except FileNotFoundError:
        sys.exit(f"Error: file zona '{path}' tidak ditemukan. "
                 "Jalankan dulu: python zone_tool.py --source <video>")
    return [(z["name"], np.array(z["polygon"], np.int32)) for z in zones]


def zone_masks(zones, shape):
    masks = []
    for name, poly in zones:
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [poly], 1)
        masks.append((name, m))
    return masks


def zone_of(box, zones, masks, anchor, min_overlap, prefer=None):
    """Zona untuk satu bbox. anchor: bottom_center | center | overlap.

    overlap: zona dengan porsi bbox tertutup zona terbesar (>= min_overlap). Cocok untuk
    orang yang duduk di samping meja atau membelakangi kamera, di mana satu titik
    bbox sering jatuh di luar polygon meja. prefer (zona saat ini) dipertahankan selama
    overlap-nya masih >= min_overlap, agar orang di antara dua meja tidak berpindah-pindah.
    """
    x1, y1, x2, y2 = box
    if anchor == "overlap":
        h, w = masks[0][1].shape if masks else (0, 0)
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(w, int(x2)), min(h, int(y2))
        area = max(1, (xb - xa) * (yb - ya))
        best, best_frac = None, min_overlap
        for name, m in masks:
            frac = m[ya:yb, xa:xb].sum() / area
            if name == prefer and frac >= min_overlap:
                return name
            if frac >= best_frac:
                best, best_frac = name, frac
        return best
    point = (float((x1 + x2) / 2), float((y1 + y2) / 2 if anchor == "center" else y2))
    for name, poly in zones:
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return name
    return None


def draw_label(frame, text, x, y, color):
    """Label dengan latar terisi di atas kepala, tetap terbaca di video apa pun."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x = max(0, x)
    y = max(th + base + 4, y)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - base - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def fmt_mmss(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="override source di config")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    source = args.source or cfg["source"]
    is_stream = isinstance(source, str) and "://" in source
    out_cfg = cfg["output"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Error: tidak bisa membuka sumber video '{source}'. "
                 "Periksa path file / URL RTSP / kredensial / jaringan.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    zones = load_zones(cfg["zones_file"])

    from ultralytics import YOLO
    model = YOLO(cfg["model"])

    reid = None
    face_cfg = cfg["face"]
    if face_cfg["enabled"]:
        from face_reid import FaceReID
        reid = FaceReID(face_cfg["model_name"], face_cfg["match_threshold"],
                        face_cfg["min_face_px"], cfg["device"])

    dcfg = cfg["dwell"]
    dwell = DwellTracker(dcfg["min_sit_seconds"], dcfg["leave_grace_seconds"],
                         dcfg.get("reattach_dist", 0.5))

    wall_start = datetime.now()

    def fmt_time(t):
        if is_stream:
            return (wall_start + timedelta(seconds=t)).isoformat(timespec="seconds")
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"

    csv_file = open(out_cfg["log_csv"], "w", newline="", encoding="utf-8") if out_cfg.get("log_csv") else None
    writer = csv.writer(csv_file) if csv_file else None
    if writer:
        writer.writerow(["track_id", "customer_id", "zone", "start_time", "end_time", "duration_s"])

    def log_sessions(sessions):
        for s in sessions:
            print(f"Sesi selesai: T{s.track_id} {s.customer_id or '-'} {s.zone} "
                  f"{s.duration_s:.1f}s")
            if writer:
                writer.writerow([s.track_id, s.customer_id or "", s.zone,
                                 fmt_time(s.start_s), fmt_time(s.end_s),
                                 f"{s.duration_s:.2f}"])
        if csv_file:
            csv_file.flush()

    anchor = dcfg.get("anchor", "bottom_center")
    min_overlap = dcfg.get("min_overlap", 0.15)
    masks = None

    video_writer = None
    last_box = {}  # tid -> bbox terakhir, agar label tidak berkedip saat deteksi putus sesaat
    frame_idx = 0
    t0 = time.monotonic()
    t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.monotonic() - t0) if is_stream else frame_idx / fps

            res = model.track(frame, persist=True, classes=[0], tracker=cfg.get("tracker", "bytetrack.yaml"),
                              conf=cfg["person_conf"], imgsz=cfg["imgsz"],
                              device=cfg["device"], verbose=False)[0]

            people = []
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                xyxy = res.boxes.xyxy.cpu().numpy()
                people = list(zip(ids, xyxy))

            if masks is None:
                masks = zone_masks(zones, frame.shape)

            observations, positions = {}, {}
            for tid, (x1, y1, x2, y2) in people:
                observations[tid] = zone_of((x1, y1, x2, y2), zones, masks, anchor, min_overlap,
                                            prefer=dwell.zone(tid))
                positions[tid] = (float((x1 + x2) / 2), float((y1 + y2) / 2), float(y2 - y1))
                last_box[tid] = (x1, y1, x2, y2)

                if (reid and dwell.get_customer(tid) is None
                        and frame_idx % face_cfg["every_n_frames"] == 0):
                    cid = reid.identify(frame, (x1, y1, x2, y2))
                    if cid:
                        dwell.set_customer(tid, cid)

            log_sessions(dwell.update(t, observations, positions))

            if out_cfg["show_window"] or out_cfg["save_video"]:
                if out_cfg.get("draw_zones", False):
                    for name, poly in zones:
                        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                        cv2.putText(frame, name, tuple(int(v) for v in poly[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Semua track yang sedang di zona (termasuk yang hilang sesaat dalam masa grace)
                # diberi timer sejak masuk zona, tanpa menunggu konfirmasi duduk.
                for tid in list(last_box):
                    elapsed = dwell.elapsed(tid, t)
                    if elapsed is None:
                        if tid not in observations:
                            del last_box[tid]
                        continue
                    x1, y1, x2, y2 = (int(v) for v in last_box[tid])
                    color = (0, 0, 220) if dwell.is_sitting(tid) else (0, 160, 0)
                    if out_cfg.get("draw_boxes", True):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    label = fmt_mmss(elapsed)
                    if dwell.get_customer(tid):
                        label = f"{dwell.get_customer(tid)} {label}"
                    draw_label(frame, label, x1, y1 - 2, color)

                if out_cfg["save_video"]:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        video_writer = cv2.VideoWriter(
                            "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    video_writer.write(frame)
                if out_cfg["show_window"]:
                    cv2.imshow("sitting-tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_idx += 1
    except KeyboardInterrupt:
        print("Dihentikan pengguna.")
    finally:
        log_sessions(dwell.flush(t))
        cap.release()
        if video_writer:
            video_writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
