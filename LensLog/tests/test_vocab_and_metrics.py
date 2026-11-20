import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from category_vocab import load_categories
from perception_metrics import ranking_f1,category_metrics

def test_canonical_vocab_adapter_exact_labels(monkeypatch):
    module=ModuleType('vocab');module.load=lambda:([],[],['Official A','Official B'],lambda v,n:False)
    monkeypatch.setitem(sys.modules,'vocab',module)
    result=load_categories(SimpleNamespace(categories=()),require_canonical=True)
    assert result.canonical and result.labels==('Official A','Official B')
    with pytest.raises(ValueError,match='conflict'):load_categories(SimpleNamespace(categories=('different',)))

def test_canonical_vocab_required_without_actual_module(monkeypatch):
    import category_vocab
    def missing(name):
        error=ModuleNotFoundError("No module named 'vocab'");error.name='vocab';raise error
    monkeypatch.setattr(category_vocab.importlib,'import_module',missing)
    with pytest.raises(RuntimeError,match='BLOCKED'):load_categories(SimpleNamespace(categories=('dev',)),True)
    result=load_categories(SimpleNamespace(categories=('dev',)),False)
    assert not result.canonical and result.source=='development-json'

@pytest.mark.parametrize('labels',[[],['A','A'],[''],[3]])
def test_invalid_canonical_vocab_rejected(monkeypatch,labels):
    module=ModuleType('vocab');module.load=lambda:([],[],labels,None)
    monkeypatch.setitem(sys.modules,'vocab',module)
    with pytest.raises(ValueError):load_categories(SimpleNamespace(categories=()))

def test_ranking_proxy_known_answers_and_ties():
    perfect=ranking_f1([0,1,0,0],[0,1,0,0],.25)
    wrong=ranking_f1([0,1,0,0],[1,0,0,0],.25)
    assert perfect['f1']==1 and wrong['f1']==0
    assert ranking_f1([0,0],[0,0],.5)['f1']==1  # Explicit earlier-index tie rule, not quality evidence.
    assert not perfect['budget_enforced']
    with pytest.raises(ValueError):ranking_f1([0,np.nan],[0,1])

def test_confusion_matrix_sample_counts_and_vocabulary():
    result=category_metrics(['A','B'],['A','A','B'],['A','B','B'])
    assert result['accuracy']==pytest.approx(2/3)
    assert result['confusion_matrix']==[[1,1],[0,1]]
    assert result['per_category']['A']['count']==2
    with pytest.raises(ValueError):category_metrics(['A'],['A'],['unsupported'])

def test_m4_module_present_without_kit_uses_explicit_development_labels(tmp_path,monkeypatch):
    module=ModuleType('vocab')
    class VocabError(RuntimeError):pass
    module.VocabError=VocabError
    module.config=SimpleNamespace(vocab_path=lambda:tmp_path/'missing-kit.json')
    def load():raise VocabError('Kit not installed')
    module.load=load
    monkeypatch.setitem(sys.modules,'vocab',module)
    result=load_categories(SimpleNamespace(categories=('dev-label',)),require_canonical=False)
    assert result.labels==('dev-label',) and not result.canonical
    with pytest.raises(VocabError):load_categories(SimpleNamespace(categories=('dev-label',)),require_canonical=True)


def test_malformed_installed_m4_vocab_never_silently_uses_dev_labels(tmp_path,monkeypatch):
    path=tmp_path/'bad-kit.json';path.write_text('broken')
    module=ModuleType('vocab')
    class VocabError(RuntimeError):pass
    module.VocabError=VocabError;module.config=SimpleNamespace(vocab_path=lambda:path)
    def load():raise VocabError('Malformed vocabulary')
    module.load=load;monkeypatch.setitem(sys.modules,'vocab',module)
    with pytest.raises(VocabError):load_categories(SimpleNamespace(categories=('dev-label',)),False)


def test_unrelated_vocab_failure_is_not_masked(tmp_path,monkeypatch):
    module=ModuleType('vocab');module.config=SimpleNamespace(vocab_path=lambda:tmp_path/'missing')
    module.VocabError=RuntimeError
    def load():raise TypeError('Implementation error')
    module.load=load;monkeypatch.setitem(sys.modules,'vocab',module)
    with pytest.raises(TypeError):load_categories(SimpleNamespace(categories=('dev',)),False)
