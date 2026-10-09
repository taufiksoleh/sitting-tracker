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


def zone_of(point, zones):
    for name, poly in zones:
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return name
    return None


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
    dwell = DwellTracker(dcfg["min_sit_seconds"], dcfg["leave_grace_seconds"])

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

    video_writer = None
    track_customer = {}
    frame_idx = 0
    t0 = time.monotonic()
    t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.monotonic() - t0) if is_stream else frame_idx / fps

            res = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml",
                              conf=cfg["person_conf"], imgsz=cfg["imgsz"],
                              device=cfg["device"], verbose=False)[0]

            people = []
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                xyxy = res.boxes.xyxy.cpu().numpy()
                people = list(zip(ids, xyxy))

            observations = {}
            for tid, (x1, y1, x2, y2) in people:
                anchor = (float((x1 + x2) / 2), float(y2))
                observations[tid] = zone_of(anchor, zones)

                if (reid and tid not in track_customer
                        and frame_idx % face_cfg["every_n_frames"] == 0):
                    cid = reid.identify(frame, (x1, y1, x2, y2))
                    if cid:
                        track_customer[tid] = cid
                        dwell.set_customer(tid, cid)

            log_sessions(dwell.update(t, observations))

            if out_cfg["show_window"] or out_cfg["save_video"]:
                for name, poly in zones:
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                    cv2.putText(frame, name, tuple(int(v) for v in poly[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                for tid, (x1, y1, x2, y2) in people:
                    sitting = dwell.is_sitting(tid)
                    color = (0, 0, 255) if sitting else (0, 200, 0)
                    label = f"T{tid} {track_customer.get(tid, '')}".strip()
                    if sitting:
                        label += f" duduk {fmt_mmss(dwell.current_duration(tid, t))}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
