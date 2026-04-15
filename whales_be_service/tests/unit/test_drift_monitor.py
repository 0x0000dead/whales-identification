"""Unit tests for the DriftMonitor — rolling window + alarm heuristic."""

from whales_be_service.monitoring.drift import DriftMonitor, get_drift_monitor


class TestDriftMonitor:
    def test_empty_stats(self):
        m = DriftMonitor(window_size=100)
        s = m.stats()
        assert s["n"] == 0
        assert s["score_mean"] == 0.0

    def test_record_and_stats(self):
        m = DriftMonitor(window_size=100)
        for v in [0.8, 0.9, 0.7, 0.85]:
            m.record(v, 0.5)
        s = m.stats()
        assert s["n"] == 4
        assert 0.7 <= s["score_mean"] <= 0.9

    def test_rolling_window_drops_old(self):
        m = DriftMonitor(window_size=3)
        for v in [0.1, 0.2, 0.3, 0.4]:  # 0.1 should be dropped
            m.record(v, 0.5)
        s = m.stats()
        assert s["n"] == 3
        # mean of [0.2, 0.3, 0.4] = 0.3
        assert abs(s["score_mean"] - 0.3) < 0.01

    def test_alarm_when_below_baseline(self):
        m = DriftMonitor(window_size=100, baseline_mean=0.9, alarm_drop=0.1)
        # need >=50 samples for alarm to trigger
        for _ in range(60):
            m.record(0.5, 0.5)  # well below baseline
        s = m.stats()
        assert s["alarms_total"] >= 1

    def test_no_alarm_when_no_baseline(self):
        m = DriftMonitor(window_size=100, baseline_mean=None)
        for _ in range(100):
            m.record(0.1, 0.1)
        assert m.stats()["alarms_total"] == 0

    def test_no_alarm_when_too_few_samples(self):
        m = DriftMonitor(window_size=100, baseline_mean=0.9, alarm_drop=0.1)
        for _ in range(10):
            m.record(0.1, 0.1)
        assert m.stats()["alarms_total"] == 0

    def test_singleton_accessor(self):
        a = get_drift_monitor()
        b = get_drift_monitor()
        assert a is b
