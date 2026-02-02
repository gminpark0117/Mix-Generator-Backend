from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AnalysisResult:
    """Result from analyzing a track."""
    metadata_json: dict
    analysis_json: dict


class MixAnalyzer(Protocol):
    """Protocol for audio analysis."""
    async def analyze(self, abs_path: str, metadata_json: dict) -> AnalysisResult:
        ...


class PlaceholderMixAnalyzer:
    """
    Placeholder analyzer:
    - reads no actual audio
    - returns minimal analysis_json stub
    """

    async def analyze(self, abs_path: str, metadata_json: dict) -> AnalysisResult:
        """
        Analyze a track file.
        
        Args:
            abs_path: absolute path to audio file
            metadata_json: track metadata (song_name, artist_name, etc.)
        
        Returns:
            AnalysisResult with modified metadata_json and analysis_json dict
        """
        # Make a copy to avoid mutating the original
        meta = dict(metadata_json)
        
        # Placeholder: add some analysis fields to metadata
        meta["analyzed"] = True
        
        # Placeholder: return empty analysis
        # Later, this can extract tempo, key, energy, etc. from audio
        analysis = {
            "placeholder": True,
        }
        
        return AnalysisResult(
            metadata_json=meta,
            analysis_json=analysis,
        )
