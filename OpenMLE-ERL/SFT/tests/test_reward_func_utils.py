from __future__ import annotations

from tts_search.reward_func_utils import get_clear_log


def test_get_clear_log_keeps_sandbox_stdout_and_stderr() -> None:
    run_log = """\
--- SANDBOX STDOUT START ---
started
[HB]temporary heartbeat noise[HB]
finished
--- SANDBOX STDOUT END ---
--- SANDBOX STDERR START ---
Traceback (most recent call last):
  File "/workspace/main.py", line 9, in <module>
FileNotFoundError: train.csv
--- SANDBOX STDERR END ---
"""

    clear_log = get_clear_log(run_log)

    assert "started" in clear_log
    assert "finished" in clear_log
    assert "Traceback (most recent call last)" in clear_log
    assert "FileNotFoundError: train.csv" in clear_log
    assert "temporary heartbeat noise" not in clear_log
    assert "SANDBOX STDOUT" not in clear_log
    assert "SANDBOX STDERR" not in clear_log
