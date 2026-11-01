# Natural Language Toolkit: save_dir compatibility tests for the staged saves
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""``save_dir`` on PerceptronTagger, Maxent_NE_Chunker and PunktTokenizer is a
lazily allocated private staging dir inside a data root (GHSA-8mgp-746c-j5xp).

Two compatibility paths around it are pinned here:

* ``PerceptronTagger.save_dir`` used to be a plain attribute, so it stays
  assignable; whatever is assigned is still validated when the model is saved.
* An object pickled before ``save_dir`` became a property carries no
  ``_save_dir`` (a tagger carries the old ``save_dir`` string instead); after
  unpickling it gets a fresh private staging dir rather than an AttributeError,
  and never the stale shared-temp path.
"""

import os
import tempfile
import types

import pytest

from nltk import pathsec
from nltk.data import FileSystemPathPointer

_SENTENCES = [[("today", "NN"), ("is", "VBZ"), ("good", "JJ")]]


def _as_if_unpickled(cls, state):
    """Rebuild an instance the way the default unpickler does: ``__new__`` and
    then the saved ``__dict__``, with no ``__init__``."""
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def _inside(path, root):
    real, real_root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([real, real_root]) == real_root


def _tagger():
    from nltk.tag.perceptron import PerceptronTagger

    tagger = PerceptronTagger(load=False)
    tagger.model.weights = {"feat": {"NN": 1.0}}
    tagger.tagdict = {"today": "NN"}
    tagger.classes = {"NN"}
    return tagger


# --- PerceptronTagger.save_dir stays assignable --------------------------------


def test_assigned_outside_save_dir_is_refused(pathsec_sandbox):
    root, outside = pathsec_sandbox
    target = outside / "model_dir"
    tagger = _tagger()
    tagger.save_dir = str(target)
    assert tagger.save_dir == str(target)

    with pytest.raises(PermissionError):
        tagger.save_to_json(lang="eng")
    with pytest.raises(PermissionError):
        tagger.train(_SENTENCES, save_loc=tagger.save_dir, nr_iter=1)
    assert not target.exists(), "an outside save_dir was created"


def test_assigned_in_root_save_dir_saves_and_reloads(restricted_sandbox):
    from nltk.tag.perceptron import PerceptronTagger

    target = os.path.join(restricted_sandbox, "assigned_model")
    tagger = _tagger()
    tagger.save_dir = target
    tagger.save_to_json(lang="eng")
    assert sorted(os.listdir(target)) == sorted(tagger.param_files("eng"))

    back = PerceptronTagger(load=False)
    back.load_from_json(lang="eng", loc=FileSystemPathPointer(target))
    assert back.model.weights == {"feat": {"NN": 1.0}}
    assert back.tagdict == {"today": "NN"}


# --- objects unpickled from before save_dir became a property -------------------


def test_unpickled_tagger_gets_a_fresh_private_save_dir(restricted_sandbox):
    from nltk.tag.perceptron import PerceptronTagger

    stale = os.path.join(tempfile.gettempdir(), "averaged_perceptron_tagger_eng")
    state = dict(vars(_tagger()))
    del state["_save_dir"]
    state["save_dir"] = stale
    tagger = _as_if_unpickled(PerceptronTagger, state)

    out = tagger.save_dir
    assert out != stale
    assert _inside(out, restricted_sandbox)
    assert os.path.basename(out).startswith("nltk_averaged_perceptron_tagger_eng_")
    assert pathsec.is_private_dir(out)

    tagger.save_to_json(lang="eng")
    assert sorted(os.listdir(out)) == sorted(tagger.param_files("eng"))


def test_unpickled_ne_chunker_gets_a_fresh_private_save_dir(
    restricted_sandbox, monkeypatch
):
    import nltk.chunk.named_entity as ne

    captured = {}

    def fake_save(wgt, mpg, lab, aon, tab_dir=None):
        captured["tab_dir"] = tab_dir

    monkeypatch.setattr(ne, "save_maxent_params", fake_save)
    tagger = types.SimpleNamespace(
        _classifier=types.SimpleNamespace(
            _encoding=types.SimpleNamespace(_mapping={}, _labels=[], _alwayson={}),
            _weights=[],
        )
    )
    chunker = _as_if_unpickled(
        ne.Maxent_NE_Chunker, {"_fmt": "multiclass", "_tagger": tagger}
    )

    out = chunker.save_dir
    assert _inside(out, restricted_sandbox)
    assert os.path.basename(out).startswith("nltk_ne_chunker_multiclass_")
    assert pathsec.is_private_dir(out)
    assert chunker.save_params() == out == captured["tab_dir"]


def test_unpickled_punkt_tokenizer_gets_a_fresh_private_save_dir(restricted_sandbox):
    from nltk.tokenize.punkt import PunktParameters, PunktTokenizer

    tokenizer = _as_if_unpickled(
        PunktTokenizer, {"_lang": "english", "_params": PunktParameters()}
    )

    out = tokenizer.save_dir
    assert _inside(out, restricted_sandbox)
    assert os.path.basename(out).startswith("nltk_punkt_english_")
    assert pathsec.is_private_dir(out)
    assert tokenizer.save_params() == out
    assert sorted(os.listdir(out)) == [
        "abbrev_types.txt",
        "collocations.tab",
        "ortho_context.tab",
        "sent_starters.txt",
    ]
