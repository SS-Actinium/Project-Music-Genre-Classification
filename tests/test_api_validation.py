"""API validation helpers (no live server required for suffix/size rules)."""

from pathlib import Path

from music_genre.config import ALLOWED_AUDIO_SUFFIXES, MAX_UPLOAD_BYTES


def test_allowed_suffixes_include_wav_mp3():
    assert ".wav" in ALLOWED_AUDIO_SUFFIXES
    assert ".mp3" in ALLOWED_AUDIO_SUFFIXES


def test_max_upload_positive():
    assert MAX_UPLOAD_BYTES >= 1_000_000


def test_reject_exe_suffix_logic():
    suffix = Path("malware.exe").suffix.lower()
    assert suffix not in ALLOWED_AUDIO_SUFFIXES
