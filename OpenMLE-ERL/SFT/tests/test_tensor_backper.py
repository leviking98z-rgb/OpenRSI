from __future__ import annotations

import torch

from slime.utils.tensor_backper import TensorBackuper


def test_pageable_backup_can_be_forced(monkeypatch):
    monkeypatch.setenv("SLIME_TENSOR_BACKUP_PIN_MEMORY", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    source = torch.arange(8, dtype=torch.float32)
    backuper = TensorBackuper.create(
        source_getter=lambda: [("weight", source)],
        single_tag=None,
    )

    backuper.backup("actor")

    backup = backuper.get("actor")["weight"]
    assert backup.device.type == "cpu"
    assert not backup.is_pinned()
    torch.testing.assert_close(backup, source)


def test_pinned_allocation_falls_back_to_pageable(monkeypatch, caplog):
    monkeypatch.setenv("SLIME_TENSOR_BACKUP_PIN_MEMORY", "1")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    source = torch.arange(8, dtype=torch.float32)
    original_empty_like = torch.empty_like

    def empty_like_with_failed_pin(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise torch.AcceleratorError("synthetic pinned-memory failure")
        return original_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", empty_like_with_failed_pin)
    backuper = TensorBackuper.create(
        source_getter=lambda: [("weight", source)],
        single_tag=None,
    )

    backuper.backup("ref")

    backup = backuper.get("ref")["weight"]
    assert backup.device.type == "cpu"
    assert not backup.is_pinned()
    torch.testing.assert_close(backup, source)
    assert "falling back to pageable memory" in caplog.text
