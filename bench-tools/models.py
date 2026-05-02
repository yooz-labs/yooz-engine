"""Data models for the benchmark pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawEntry(BaseModel):
    """A raw transcription entry from any source."""

    id: str
    source: str
    raw_transcription: str
    word_count: int
    metadata: dict = Field(default_factory=dict)


class Annotation(BaseModel):
    """Gold-standard annotation produced by Claude."""

    proofread: str
    rewrite: str
    difficulty: str  # easy, medium, hard
    error_types: list[str] = Field(default_factory=list)
    domain: str = "unknown"  # casual, technical, business, dictation


class GoldEntry(BaseModel):
    """A fully annotated benchmark entry."""

    id: str
    source: str
    raw_transcription: str
    proofread: str
    rewrite: str
    word_count: int
    metadata: dict = Field(default_factory=dict)
