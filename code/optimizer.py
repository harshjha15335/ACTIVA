"""PLACEHOLDER - owned by M3. Replace with the m3-selector version.

Keep the signature: select_highlights(info, actions, imp, clip) -> list[Highlight],
sorted, non-overlapping and within the budget.
"""
from fallbacks import uniform_highlights


def select_highlights(info, actions, imp, clip):
    return uniform_highlights(info)
