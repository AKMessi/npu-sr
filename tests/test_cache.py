from npu_sr.cache import ContextCache


def test_context_cache_integrity_and_version_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr("npu_sr.cache.device_signature", lambda: {"architecture": "test"})
    cache = ContextCache(tmp_path, "model-hash", "ort-1", "qnn-1")
    assert not cache.valid()
    cache.pending.write_bytes(b"test embedded context")
    cache.publish()
    assert cache.valid()
    assert not ContextCache(tmp_path, "model-hash", "ort-1", "qnn-2").valid()
    assert not ContextCache(tmp_path, "model-hash", "ort-1", "qnn-1", "burst").valid()
    cache.path.write_bytes(b"tampered")
    assert not cache.valid()
    cache.invalidate()
    assert not cache.metadata.exists()
    cache.metadata.write_text("[]")
    assert not cache.valid()
