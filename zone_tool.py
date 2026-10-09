"""Tool interaktif untuk menandai zona kursi/meja pada frame pertama video/RTSP.

Kontrol: klik kiri = tambah titik, n = selesaikan zona, u = undo,
s = simpan ke zones.json, q = keluar.
"""
import argparse
import json
import sys

import cv2
import numpy as np


def read_first_frame(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Error: tidak bisa membuka sumber video '{source}'. "
                 "Periksa path file atau URL RTSP.")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        sys.exit(f"Error: gagal membaca frame dari '{source}'.")
    return frame


def draw(base, zones, current):
    img = base.copy()
    for z in zones:
        pts = np.array(z["polygon"], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    for i, p in enumerate(current):
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
        if i:
            cv2.line(img, tuple(current[i - 1]), tuple(p), (0, 0, 255), 2)
    cv2.putText(img, "klik=titik  n=selesai zona  u=undo  s=simpan  q=keluar",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Tandai zona duduk pada frame pertama.")
    ap.add_argument("--source", required=True, help="path video atau URL RTSP")
    ap.add_argument("--out", default="zones.json")
    args = ap.parse_args()

    base = read_first_frame(args.source)
    zones, current = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    win = "zone_tool"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, draw(base, zones, current))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            if len(current) >= 3:
                zones.append({"name": f"table_{len(zones) + 1}", "polygon": list(current)})
                current.clear()
            else:
                print("Zona butuh minimal 3 titik.")
        elif key == ord("u"):
            if current:
                current.pop()
            elif zones:
                zones.pop()
        elif key == ord("s"):
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"zones": zones}, f, indent=2)
            print(f"Tersimpan {len(zones)} zona ke {args.out}")
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
