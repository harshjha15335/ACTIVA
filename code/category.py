"""PLACEHOLDER - owned by M2. Replace with the m2-perception version.

Keep the signature: classify_video(info, clip, imp) -> CategoryResult.
"""
import vocab
from contracts import CategoryResult


def classify_video(info, clip, imp) -> CategoryResult:
    return CategoryResult(label=vocab.load().categories[0])
