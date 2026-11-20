"""M2 adapter: canonical M4 vocab.load() preferred; explicit dev JSON is labeled."""
from dataclasses import dataclass
import importlib
import os
from pathlib import Path

@dataclass(frozen=True)
class CategoryVocabulary:
    labels: tuple[str,...]
    source: str
    canonical: bool


def validate_labels(labels):
    if not isinstance(labels,(list,tuple)) or not labels or any(not isinstance(x,str) or not x.strip() for x in labels) or len(set(labels))!=len(labels):
        raise ValueError('Categories require unique nonempty exact strings.')
    return tuple(labels)


def load_categories(cfg, require_canonical=None):
    if require_canonical is None:
        require_canonical=os.environ.get('ACTIVA_REQUIRE_CANONICAL_VOCAB','0')=='1'
    try:
        module=importlib.import_module('vocab')
    except ModuleNotFoundError as error:
        if error.name!='vocab':raise
        module=None
    if module is not None:
        try:
            loaded=module.load()
        except Exception as error:
            # Module presence does not mean the kit has been installed. Only the
            # declared M4 error plus a demonstrably missing configured file may
            # use explicit development labels; malformed installed files fail.
            declared_error=getattr(module,'VocabError',None)
            path_getter=getattr(getattr(module,'config',None),'vocab_path',None)
            if (require_canonical or not isinstance(declared_error,type)
                    or not isinstance(error,declared_error) or not callable(path_getter)):
                raise
            if Path(path_getter()).exists():
                raise
            module=None
    if module is not None:
        if not isinstance(loaded,(tuple,list)) or len(loaded)!=4:
            raise ValueError('M4 vocab.load() must return (verbs,nouns,categories,is_valid_pair).')
        canonical=validate_labels(loaded[2])
        if cfg.categories and tuple(cfg.categories)!=canonical:
            raise ValueError('Development category labels conflict with canonical M4 vocabulary.')
        return CategoryVocabulary(canonical,'M4 vocab.load()',True)
    if require_canonical:
        raise RuntimeError('Canonical M4 vocab.load() unavailable; official compatibility is BLOCKED.')
    if not cfg.categories:
        raise ValueError('Set ACTIVA_CATEGORIES for explicitly labeled development use, or supply canonical M4 vocab.load().')
    return CategoryVocabulary(validate_labels(cfg.categories),'development-json',False)
