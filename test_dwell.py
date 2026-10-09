import pytest

from dwell import DwellTracker


def run(tracker, steps):
    """steps: list of (t, {track_id: zone|None}); kumpulkan semua sesi selesai."""
    done = []
    for t, obs in steps:
        done += tracker.update(t, obs)
    return done


def frames(start, end, obs, dt=0.5):
    n = int(round((end - start) / dt))
    return [(start + i * dt, obs) for i in range(n + 1)]


def make():
    return DwellTracker(min_sit_seconds=3.0, leave_grace_seconds=2.0)


def test_short_visit_not_recorded():
    d = make()
    done = run(d, frames(0, 2, {1: "table_1"}))
    done += run(d, frames(2.5, 10, {1: None}))
    done += d.flush(10)
    assert done == []


def test_session_duration():
    d = make()
    done = run(d, frames(0, 10, {1: "table_1"}))
    done += run(d, frames(10.5, 14, {1: None}))
    assert len(done) == 1
    s = done[0]
    assert (s.track_id, s.zone) == (1, "table_1")
    assert s.start_s == 0
    assert s.end_s == 10
    assert s.duration_s == pytest.approx(10)


def test_occlusion_within_grace_keeps_session():
    d = make()
    steps = frames(0, 5, {1: "table_1"})
    steps += frames(5.5, 6.5, {})              # hilang 1.5 s (<= grace)
    steps += frames(7, 10, {1: "table_1"})
    done = run(d, steps)
    assert done == []                           # tidak terputus
    done = d.flush(10)
    assert len(done) == 1
    assert done[0].start_s == 0 and done[0].end_s == 10


def test_gap_beyond_grace_closes_session():
    d = make()
    steps = frames(0, 5, {1: "table_1"})
    steps += frames(5.5, 9, {})                 # hilang > grace
    done = run(d, steps)
    assert len(done) == 1
    assert done[0].end_s == 5                   # waktu terakhir terlihat di zona
    assert done[0].duration_s == pytest.approx(5)


def test_leave_zone_beyond_grace_closes_session():
    d = make()
    steps = frames(0, 5, {1: "table_1"}) + frames(5.5, 9, {1: None})
    done = run(d, steps)
    assert len(done) == 1 and done[0].end_s == 5


def test_customer_id_in_log():
    d = make()
    d.set_customer(7, "C001")
    steps = frames(0, 5, {7: "table_1"}) + frames(5.5, 9, {})
    done = run(d, steps)
    assert len(done) == 1
    assert done[0].customer_id == "C001"
    assert done[0].track_id == 7


def test_customer_id_set_mid_session():
    d = make()
    run(d, frames(0, 2, {3: "table_1"}))
    d.set_customer(3, "C002")
    done = run(d, frames(2.5, 5, {3: "table_1"}))
    done += d.flush(5)
    assert done[0].customer_id == "C002"


def test_zone_switch_splits_session():
    d = make()
    steps = frames(0, 5, {1: "table_1"}) + frames(5.5, 10, {1: "table_2"})
    done = run(d, steps)
    assert len(done) == 1
    assert done[0].zone == "table_1" and done[0].end_s == 5
    done += d.flush(10)
    assert len(done) == 2
    assert done[1].zone == "table_2"
    assert done[1].start_s == 5.5
    assert done[1].end_s == 10


def test_current_duration():
    d = make()
    run(d, frames(0, 1, {1: "table_1"}))
    assert d.current_duration(1, 1) == 0        # belum dianggap duduk
    run(d, frames(1.5, 6, {1: "table_1"}))
    assert d.current_duration(1, 6) == pytest.approx(6)
    assert d.current_duration(99, 6) == 0


def test_flush_closes_open_sessions():
    d = make()
    run(d, frames(0, 6, {1: "table_1", 2: "table_2"}))
    done = d.flush(6)
    assert {s.track_id for s in done} == {1, 2}
    assert d.flush(7) == []
