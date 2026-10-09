"""Logika durasi duduk (murni, tanpa dependensi CV).

Setiap track berada di sebuah zona atau di luar zona (None). Sesi duduk dimulai
setelah track berada di zona >= min_sit_seconds (dihitung dari saat masuk zona)
dan ditutup pada waktu terakhir terlihat di zona, setelah hilang/keluar zona
lebih lama dari leave_grace_seconds, atau saat pindah zona.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    __slots__ = ("zone", "enter_t", "last_in_t", "confirmed")

    def __init__(self, zone: str, t: float):
        self.zone = zone
        self.enter_t = t      # waktu masuk zona (awal sesi)
        self.last_in_t = t    # terakhir terlihat di zona
        self.confirmed = False


class DwellTracker:
    def __init__(self, min_sit_seconds: float = 3.0, leave_grace_seconds: float = 2.0):
        self.min_sit = min_sit_seconds
        self.grace = leave_grace_seconds
        self._states: Dict[int, _State] = {}
        self._customers: Dict[int, str] = {}

    def set_customer(self, track_id: int, customer_id: str) -> None:
        self._customers[track_id] = customer_id

    def get_customer(self, track_id: int) -> Optional[str]:
        return self._customers.get(track_id)

    def is_sitting(self, track_id: int) -> bool:
        st = self._states.get(track_id)
        return st is not None and st.confirmed

    def current_duration(self, track_id: int, t: float) -> float:
        """Durasi sesi berjalan (0 jika belum dianggap duduk)."""
        st = self._states.get(track_id)
        if st is None or not st.confirmed:
            return 0.0
        return max(0.0, t - st.enter_t)

    def _close(self, track_id: int, st: _State) -> Optional[Session]:
        if not st.confirmed:
            return None
        return Session(track_id, self._customers.get(track_id), st.zone,
                       st.enter_t, st.last_in_t)

    def update(self, t: float, observations: Dict[int, Optional[str]]) -> List[Session]:
        """observations: {track_id: nama_zona | None} untuk track yang terlihat pada waktu t.

        Mengembalikan sesi yang baru selesai.
        """
        finished: List[Session] = []

        for tid, zone in observations.items():
            st = self._states.get(tid)
            if zone is None:
                continue  # keluar zona: dinilai lewat grace di bawah
            if st is not None and st.zone != zone:
                # pindah zona: tutup sesi lama, mulai kandidat baru
                s = self._close(tid, st)
                if s:
                    finished.append(s)
                st = None
            if st is None:
                self._states[tid] = _State(zone, t)
                continue
            st.last_in_t = t
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
