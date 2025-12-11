"""
HDC and system constants.

These values are based on hyperdimensional computing research and will be
validated empirically during testing.
"""

# Vector Space Configuration
DEFAULT_DIMENSIONS = 10000
"""Default hypervector dimensionality. Higher dimensions provide more capacity
but increase computation cost. 10,000D is standard in HDC research."""

VSA_MODEL = "MAP"
"""Vector Symbolic Architecture model. MAP (Multiply-Add-Permute) provides
good capacity and is well-supported by torchhd."""

# Confidence Thresholds (Calibrated for Holographic Storage)
RESPONSE_CONFIDENCE_THRESHOLD = 0.20
"""Minimum cosine similarity to provide a confident answer.
Holographic storage with bundled facts produces similarities of 0.24-0.37
for known facts due to interference. Threshold set to 0.20 to match this reality.
Above this: respond with high confidence."""

REFUSAL_CONFIDENCE_THRESHOLD = 0.10
"""Below this threshold, refuse to answer ('I don't know').
Values below 0.10 indicate noise or truly unknown queries.
Between refusal (0.10) and response (0.20): hedge/low confidence."""

SIMILARITY_THRESHOLD = 0.5
"""General threshold for considering two vectors 'similar'."""

CONTRADICTION_THRESHOLD = 0.5
"""Threshold for detecting contradictions in devil's advocate checks."""

# Resonator Network
MAX_RESONATOR_ITERATIONS = 100
"""Maximum iterations for resonator convergence."""

CONVERGENCE_THRESHOLD = 0.95
"""Cosine similarity threshold for considering resonator converged."""

# Memory Capacity
ESTIMATED_CAPACITY_DIVISOR = 100
"""Rough estimate: capacity ≈ dimensions / divisor.
For 10,000D, ~100 facts before degradation.
This is UNPROVEN and will be validated empirically."""

MAX_SEQUENCE_LENGTH = 1000
"""Maximum tokens in a sequence for positional encoding."""

# Triangulation
NUM_QUERY_VARIATIONS = 3
"""Number of query variations for triangulation cross-validation."""

CROSS_VALIDATION_AGREEMENT_THRESHOLD = 0.7
"""Minimum agreement ratio between query variations."""

# Noise Tolerance
NOISE_TOLERANCE = 0.3
"""Maximum noise ratio for graceful degradation testing."""

# Grammar Roles
ROLES = ["SUBJECT", "VERB", "OBJECT", "MODIFIER", "COMPLEMENT"]
"""Standard grammatical roles for template-based generation."""

# File Persistence
DEFAULT_PERSIST_PATH = "./data/hologram"
"""Default path for Faiss index and metadata storage."""

# ==============================================================================
# Resonant Cavity Architecture
# ==============================================================================

# Divergence Calculator Thresholds
DIVERGENCE_ACCEPT_THRESHOLD = 0.6
"""Above this similarity: accept token without correction."""

DIVERGENCE_SOFT_THRESHOLD = 0.4
"""Between soft and accept: accept with correction signal."""

DIVERGENCE_HARD_THRESHOLD = 0.2
"""Below this: reject token and resample."""

# Sesame Modulator (Style Layer)
CREATIVITY_TEMPERATURE = 0.2
"""Creativity temperature (0.0 to 1.0). Higher = more stylistic variation."""

STYLE_WEIGHT = 0.1
"""Weight for style injection in target tensor (0.0 to 0.3). Nudges, not overrides."""

LAMBDA_STYLE = 0.2
"""Style influence strength (0.0 to 0.5). Style nudges, not overrides."""

DISFLUENCY_THRESHOLD = 0.35
"""Below this confidence: inject filler tokens (um, uh, ...)."""

# Style Word Markers (for style vector construction)
STYLE_FORMAL_WORDS = ["therefore", "subsequently", "moreover", "thus", "hence"]
"""Words associated with formal style."""

STYLE_CASUAL_WORDS = ["cool", "yeah", "like", "okay", "sure", "awesome"]
"""Words associated with casual style."""

STYLE_URGENT_WORDS = ["now", "immediately", "urgent", "critical", "asap"]
"""Words associated with urgent style."""

# Generation Parameters
MAX_GENERATION_TOKENS = 100
"""Maximum tokens per generation."""

CANDIDATE_K = 5
"""Number of candidate tokens to evaluate per position."""

# Oscillation Detection
OSCILLATION_WINDOW = 5
"""Window size for detecting resonator oscillation."""

# ==============================================================================
# Conversational Chatbot
# ==============================================================================

# Intent Classification
INTENT_CONFIDENCE_THRESHOLD = 0.20
"""Minimum similarity to classify intent (below = UNKNOWN).
Set lower (0.20) because bundled prototypes have reduced similarity
to individual examples due to holographic interference.
Intent prototypes are learned from examples, not hardcoded keywords."""

# Entity Extraction
ENTITY_MATCH_THRESHOLD = 0.6
"""Minimum similarity for entity matching against vocabulary."""

# Conversation Memory
MAX_CONVERSATION_TURNS = 10
"""Maximum turns to keep in session memory."""

CONTEXT_LOOKBACK = 3
"""Number of recent turns to include in context vector."""

# Response Selection
PATTERN_MATCH_THRESHOLD = 0.05
"""Minimum similarity to match a response pattern.
Set very low to allow repetition penalty and boosting logic to work.
The selector applies additional scoring adjustments after initial matching."""

# Learning
LEARNING_STRENGTHEN_FACTOR = 1.5
"""Factor to strengthen successful patterns (Hebbian learning)."""

LEARNING_WEAKEN_FACTOR = 0.8
"""Factor to weaken failed patterns."""

STYLE_ADAPTATION_MIN_MESSAGES = 3
"""Minimum messages before inferring user style."""

# ==============================================================================
# Surprise Learning (Titans-inspired)
# ==============================================================================

SURPRISE_THRESHOLD = 0.1
"""Minimum surprise to trigger learning (0.0-1.0).
Below this, facts are considered 'already known' and skipped."""

SURPRISE_LEARNING_RATE = 0.5
"""Base learning rate for surprise-weighted updates.
Higher = faster learning, but more noise from novel facts."""

SURPRISE_DECAY = 0.99
"""Optional: Decay factor for old memories.
Enables forgetting of rarely-accessed facts (Titans momentum)."""

SURPRISE_MOMENTUM_DECAY = 0.9
"""How quickly momentum fades (exponential moving average).
Higher = longer memory of recent learning direction."""

# ==============================================================================
# LLM Models (Ventriloquist)
# ==============================================================================

DEFAULT_REASONING_MODEL = "zai-org/glm-4.6v"
"""GLM-4.6v via Novita API - larger reasoning model with multi-step deduction.
Supports: Function Calling, Structured Output, Reasoning."""

DEFAULT_FLUENCY_MODEL = "moonshotai/kimi-k2-thinking"
"""Kimi K2 via Novita API - current SLM for fluent natural language generation."""

ENABLE_REASONING_CHAIN = True
"""Enable chain-of-thought reasoning for complex queries.
When True, the Ventriloquist uses GLM-4.6v to generate step-by-step reasoning."""
