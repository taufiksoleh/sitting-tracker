"""Logika durasi duduk (murni, tanpa dependensi CV).

Setiap track berada di sebuah zona atau di luar zona (None). Sesi duduk dimulai
setelah track berada di zona >= min_sit_seconds (dihitung dari saat masuk zona)
dan ditutup pada waktu terakhir terlihat di zona, setelah hilang/keluar zona
lebih lama dari leave_grace_seconds, atau saat pindah zona.

Tracker sering memberi ID baru untuk orang yang sama (terhalang, deteksi berkedip).
Jika posisi diberikan, ID baru yang muncul di zona dan posisi yang hampir sama dengan
track yang baru hilang (masih dalam masa grace) mewarisi sesinya, sehingga timer tidak
kembali ke nol.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Pos = Tuple[float, float, float]  # (cx, cy, tinggi bbox)


@dataclass
class Session:
    track_id: int
    customer_id: Optional[str]
    zone: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class _State:
    __slots__ = ("zone", "enter_t", "last_in_t", "confirmed", "pos")

    def __init__(self, zone: str, t: float, pos: Optional[Pos] = None):
        self.zone = zone
        self.enter_t = t      # waktu masuk zona (awal sesi)
        self.last_in_t = t    # terakhir terlihat di zona
        self.confirmed = False
        self.pos = pos        # posisi terakhir di zona


class DwellTracker:
    def __init__(self, min_sit_seconds: float = 3.0, leave_grace_seconds: float = 2.0,
                 reattach_dist: float = 0.5):
        self.min_sit = min_sit_seconds
        self.grace = leave_grace_seconds
        self.reattach_dist = reattach_dist  # jarak pusat maks, relatif ke tinggi bbox
        self._states: Dict[int, _State] = {}
        self._customers: Dict[int, str] = {}

    def set_customer(self, track_id: int, customer_id: str) -> None:
        self._customers[track_id] = customer_id

    def get_customer(self, track_id: int) -> Optional[str]:
        return self._customers.get(track_id)

    def zone(self, track_id: int) -> Optional[str]:
        st = self._states.get(track_id)
        return st.zone if st else None

    def is_sitting(self, track_id: int) -> bool:
        st = self._states.get(track_id)
        return st is not None and st.confirmed

    def current_duration(self, track_id: int, t: float) -> float:
        """Durasi sesi berjalan (0 jika belum dianggap duduk)."""
        st = self._states.get(track_id)
        if st is None or not st.confirmed:
            return 0.0
        return max(0.0, t - st.enter_t)

    def elapsed(self, track_id: int, t: float) -> Optional[float]:
        """Lama sejak masuk zona, termasuk sebelum dikonfirmasi duduk (None jika tidak di zona)."""
        st = self._states.get(track_id)
        if st is None:
            return None
        return max(0.0, t - st.enter_t)

    def _close(self, track_id: int, st: _State) -> Optional[Session]:
        if not st.confirmed:
            return None
        return Session(track_id, self._customers.get(track_id), st.zone,
                       st.enter_t, st.last_in_t)

    def _reattach(self, tid: int, zone: str, t: float, pos: Optional[Pos],
                  observations: Dict[int, Optional[str]]) -> Optional[_State]:
        """Ambil alih sesi track lain di zona yang sama yang baru hilang di posisi yang sama."""
        if pos is None:
            return None
        best, best_d = None, None
        for old, st in self._states.items():
            if (old == tid or old in observations or st.zone != zone or st.pos is None
                    or t - st.last_in_t > self.grace):
                continue
            d = ((st.pos[0] - pos[0]) ** 2 + (st.pos[1] - pos[1]) ** 2) ** 0.5
            if d <= self.reattach_dist * max(st.pos[2], pos[2]) and (best_d is None or d < best_d):
                best, best_d = old, d
        if best is None:
            return None
        st = self._states.pop(best)
        if best in self._customers and tid not in self._customers:
            self._customers[tid] = self._customers[best]
        return st

    def update(self, t: float, observations: Dict[int, Optional[str]],
               positions: Optional[Dict[int, Pos]] = None) -> List[Session]:
        """observations: {track_id: nama_zona | None} untuk track yang terlihat pada waktu t.
        positions (opsional): {track_id: (cx, cy, tinggi)} untuk menyambung ID yang berganti.

        Mengembalikan sesi yang baru selesai.
        """
        finished: List[Session] = []
        positions = positions or {}

        for tid, zone in observations.items():
            st = self._states.get(tid)
            pos = positions.get(tid)
            if zone is None:
                continue  # keluar zona: dinilai lewat grace di bawah
            if st is not None and st.zone != zone:
                # pindah zona: tutup sesi lama, mulai kandidat baru
                s = self._close(tid, st)
                if s:
                    finished.append(s)
                st = None
            if st is None:
                st = self._reattach(tid, zone, t, pos, observations)
                if st is None:
                    self._states[tid] = _State(zone, t, pos)
                    continue
                self._states[tid] = st
            st.last_in_t = t
            st.pos = pos
            if not st.confirmed and t - st.enter_t >= self.min_sit:
                st.confirmed = True

        # tutup track yang sudah di luar zona / tidak terlihat lebih dari grace
        for tid in list(self._states):
            st = self._states[tid]
            if observations.get(tid) == st.zone:
                continue
            if t - st.last_in_t > self.grace:
                s = self._close(tid, st)
                if s:
                    finished.append(s)
                del self._states[tid]

        return finished

    def flush(self, t: float) -> List[Session]:
        """Tutup semua sesi yang masih terbuka (akhir video/stream)."""
        finished = []
        for tid, st in self._states.items():
            s = self._close(tid, st)
            if s:
                finished.append(s)
        self._states.clear()
        return finished
