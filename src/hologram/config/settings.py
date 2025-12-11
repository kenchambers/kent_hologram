"""
Application settings using Pydantic.

This module provides runtime configuration with environment variable support.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hologram.config.constants import (
    CONVERGENCE_THRESHOLD,
    CROSS_VALIDATION_AGREEMENT_THRESHOLD,
    DEFAULT_DIMENSIONS,
    DEFAULT_PERSIST_PATH,
    MAX_RESONATOR_ITERATIONS,
    NUM_QUERY_VARIATIONS,
    REFUSAL_CONFIDENCE_THRESHOLD,
    RESPONSE_CONFIDENCE_THRESHOLD,
    VSA_MODEL,
)


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Settings can be overridden via environment variables prefixed with HOLOGRAM_
    For example: HOLOGRAM_DIMENSIONS=5000
    """

    model_config = SettingsConfigDict(
        env_prefix="HOLOGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow other env vars without error
    )

    # Vector Space
    dimensions: int = Field(
        default=DEFAULT_DIMENSIONS,
        description="Hypervector dimensionality",
        ge=1000,
        le=100000,
    )

    vsa_model: str = Field(
        default=VSA_MODEL,
        description="Vector Symbolic Architecture model (MAP, FHRR, BSC)",
    )

    # Confidence Thresholds
    response_threshold: float = Field(
        default=RESPONSE_CONFIDENCE_THRESHOLD,
        description="Minimum confidence for responding",
        ge=0.0,
        le=1.0,
    )

    refusal_threshold: float = Field(
        default=REFUSAL_CONFIDENCE_THRESHOLD,
        description="Below this, refuse to answer",
        ge=0.0,
        le=1.0,
    )

    # Resonator
    max_resonator_iterations: int = Field(
        default=MAX_RESONATOR_ITERATIONS,
        description="Maximum resonator iterations",
        ge=1,
        le=1000,
    )

    convergence_threshold: float = Field(
        default=CONVERGENCE_THRESHOLD,
        description="Convergence similarity threshold",
        ge=0.0,
        le=1.0,
    )

    # Triangulation
    num_query_variations: int = Field(
        default=NUM_QUERY_VARIATIONS,
        description="Number of query variations for triangulation",
        ge=1,
        le=10,
    )

    cross_validation_agreement: float = Field(
        default=CROSS_VALIDATION_AGREEMENT_THRESHOLD,
        description="Required agreement between variations",
        ge=0.0,
        le=1.0,
    )

    # Persistence
    persist_path: Path = Field(
        default=Path(DEFAULT_PERSIST_PATH),
        description="Directory for Faiss index storage",
    )

    # Runtime
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )

    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        # Ensure persist path exists
        self.persist_path.mkdir(parents=True, exist_ok=True)


# Global settings instance (can be overridden for testing)
settings = Settings()
