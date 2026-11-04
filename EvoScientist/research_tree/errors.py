"""Errors shared by the research-tree modules."""

from __future__ import annotations


class TreeError(RuntimeError):
    """A request the tree refuses; the message says what to do instead."""
