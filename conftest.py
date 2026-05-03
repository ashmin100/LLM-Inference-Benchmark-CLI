"""Pytest plugin: live progress with immediate failure details."""
import time

_config = None
_total = 0
_current = 0
_t0 = 0.0


def pytest_configure(config):
    global _config
    _config = config


def pytest_collection_finish(session):
    global _total
    _total = len(session.items)


def pytest_runtest_logstart(nodeid, location):
    global _current, _t0
    _current += 1
    _t0 = time.monotonic()


def pytest_runtest_logreport(report):
    tw = _config.get_terminal_writer()

    if report.when == "call":
        elapsed_ms = (time.monotonic() - _t0) * 1000
        progress = f"[{_current}/{_total}]"

        if report.passed:
            tw.line(f"  {progress}  {elapsed_ms:.0f}ms", green=True)
        elif report.failed:
            tw.line(f"  {progress}  {elapsed_ms:.0f}ms  -- failure details:", red=True)
            if report.longrepr:
                for line in str(report.longrepr).strip().splitlines():
                    tw.line(f"      {line}", red=True)
            tw.line()

    elif report.when == "setup" and report.failed:
        tw.line(f"  [{_current}/{_total}]  setup error:", red=True)
        if report.longrepr:
            for line in str(report.longrepr).strip().splitlines():
                tw.line(f"      {line}", red=True)
        tw.line()
