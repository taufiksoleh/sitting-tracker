"""Re-ID wajah anonim: ID otomatis C001, C002, ... tanpa nama dan tanpa foto.

Embedding hanya disimpan di memori (galeri centroid); tidak ada yang ditulis ke disk.
"""
from typing import List, Optional

import numpy as np


class FaceReID:
    def __init__(self, model_name: str = "buffalo_s", threshold: float = 0.45,
                 min_face_px: int = 40, device: str = "cpu"):
        self.model_name = model_name
        self.threshold = threshold
        self.min_face_px = min_face_px
        self.device = device
        self._app = None
        self._centroids: List[np.ndarray] = []
        self._counts: List[int] = []

    def _get_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis  # import lazy
            if self.device.startswith("cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
            app = FaceAnalysis(name=self.model_name, providers=providers,
                               allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            self._app = app
        return self._app

    def _embed(self, frame, person_bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y1 + int((int(person_bbox[3]) - int(person_bbox[1])) * 0.45))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        faces = self._get_app().get(crop)
        faces = [f for f in faces
                 if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= self.min_face_px]
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None

    def match_or_create(self, emb: np.ndarray) -> str:
        """Cocokkan embedding ternormalisasi ke galeri, atau buat ID baru."""
        if self._centroids:
            sims = np.stack(self._centroids) @ emb
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                n = self._counts[best]
                c = (self._centroids[best] * n + emb) / (n + 1)  # running average
                self._centroids[best] = c / np.linalg.norm(c)
                self._counts[best] = n + 1
                return f"C{best + 1:03d}"
        self._centroids.append(emb.copy())
        self._counts.append(1)
        return f"C{len(self._centroids):03d}"

    def identify(self, frame, person_bbox) -> Optional[str]:
        emb = self._embed(frame, person_bbox)
        if emb is None:
            return None
        return self.match_or_create(emb)
