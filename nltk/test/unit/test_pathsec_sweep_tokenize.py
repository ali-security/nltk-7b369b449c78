"""
Attack tests for the bare-``open()`` path-traversal hardening
(GHSA-8mgp-746c-j5xp) in the tokenize package:

* ``nltk/tokenize/punkt.py``: ``save_punkt_params`` (4 param-file writes; its
  ``dir`` defaults to a fresh private temp) and ``PunktSentenceTokenizer.dump``
  (a debug write that now targets a fresh private temp, not a guessable /tmp).

Each patched sink is driven with a path OUTSIDE every allowed NLTK data root
and must raise ``PermissionError`` under pathsec ``ENFORCE``, writing nothing
outside the sandbox.
"""

import os
import pathlib
import shutil

import pytest

import nltk.data
import nltk.pathsec as pathsec

# The ``sandbox`` fixture is provided by nltk/test/unit/conftest.py.


def test_negative_control(sandbox):
    """A plain write to a path outside all allowed roots must be refused; proof
    that the sandbox is genuinely enforcing in this process."""
    target = sandbox / "control.txt"
    with pytest.raises(PermissionError):
        pathsec.open(str(target), "w")
    assert not target.exists()


def test_save_punkt_params_refuses_outside_dir(sandbox):
    """``save_punkt_params(dir=<outside>)`` must refuse the caller-controlled
    directory before creating it or writing any of its 4 param files."""
    from nltk.tokenize.punkt import PunktParameters, save_punkt_params

    outside_dir = sandbox / "punkt_tab"
    params = PunktParameters()
    with pytest.raises(PermissionError):
        save_punkt_params(params, dir=str(outside_dir))
    # validate_path rejects up front, so neither the dir nor any file is made.
    assert not outside_dir.exists()
    assert list(sandbox.iterdir()) == []


def test_save_punkt_params_default_is_private_dir_not_tmp(sandbox):
    """The default destination is a fresh private (0700), unpredictably-named
    directory (returned by the call), never the old guessable ``/tmp``."""
    from nltk.tokenize.punkt import PunktParameters, save_punkt_params

    out = save_punkt_params(PunktParameters())
    try:
        # is_private_dir (0700, user-owned) is the real "not shared /tmp" check;
        # a fresh mkdtemp lives under /tmp on Linux, so assert privacy plus the
        # unpredictable mkdtemp name, not that the path avoids /tmp.
        assert pathsec.is_private_dir(out)
        assert os.path.basename(out).startswith("nltk_punkt_params_")
        assert (pathlib.Path(out) / "collocations.tab").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_punkt_dump_writes_private_temp_not_hardcoded_tmp(sandbox):
    """The ``dump`` debug scaffold must write to a fresh private (0700) temp file
    and return its path, never the old guessable ``/tmp/punkt.new``."""
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktToken

    tok = PunktSentenceTokenizer()
    out = tok.dump(iter([PunktToken("Hello"), PunktToken("world.")]))
    try:
        # is_private_dir on the parent is the real "not shared /tmp" check; a
        # fresh mkdtemp lives under /tmp on Linux, so assert privacy plus the
        # unpredictable mkdtemp name, not that the path avoids /tmp.
        assert os.path.basename(out) == "punkt.new"
        assert os.path.basename(os.path.dirname(out)).startswith("nltk_punkt_dump_")
        assert pathsec.is_private_dir(os.path.dirname(out))
        assert os.path.exists(out)
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_load_lang_resets_save_dir(sandbox, monkeypatch):
    """Reusing a PunktTokenizer across languages must not keep writing into the
    first language's staging dir; load_lang() invalidates the memoized save_dir
    so each language gets its own directory."""
    import nltk.tokenize.punkt as punkt_mod
    from nltk.tokenize.punkt import PunktParameters, PunktTokenizer

    # Avoid needing installed punkt_tab data: stub what load_lang loads.
    monkeypatch.setattr("nltk.data.find", lambda p: "unused")
    monkeypatch.setattr(punkt_mod, "load_punkt_params", lambda d: PunktParameters())

    tok = PunktTokenizer("english")
    first = tok.save_dir
    assert os.path.basename(first).startswith("nltk_punkt_english_")
    tok.load_lang("german")  # switching language must refresh the dir
    second = tok.save_dir
    assert second != first, "load_lang must invalidate the memoized save_dir"
    assert os.path.basename(second).startswith("nltk_punkt_german_")
    for d in (first, second):
        shutil.rmtree(d, ignore_errors=True)


def test_save_punkt_params_round_trip(sandbox):
    """Functional check: params written by save_punkt_params reload identically
    via load_punkt_params (given a PathPointer, the API load_lang uses), so the
    save/load pair still works end-to-end under enforcement after the hardening."""
    from nltk.data import FileSystemPathPointer
    from nltk.tokenize.punkt import (
        PunktParameters,
        load_punkt_params,
        save_punkt_params,
    )

    params = PunktParameters()
    params.abbrev_types = {"dr", "mr", "vs"}
    params.collocations = {("new", "york"), ("los", "angeles")}
    params.sent_starters = {"however", "the"}
    params.ortho_context["foo"] = 5
    params.ortho_context["bar"] = 12

    out = save_punkt_params(params)
    try:
        reloaded = load_punkt_params(FileSystemPathPointer(out))
        assert reloaded.abbrev_types == params.abbrev_types
        assert reloaded.collocations == params.collocations
        assert reloaded.sent_starters == params.sent_starters
        assert dict(reloaded.ortho_context) == dict(params.ortho_context)
    finally:
        shutil.rmtree(out, ignore_errors=True)
