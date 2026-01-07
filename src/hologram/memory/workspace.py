"""
Global Workspace Selection (Baars Bottleneck)

Implements the attentional bottleneck from Global Workspace Theory:
- Only ~50 high-salience items can enter the "workspace" at once
- Selection based on salience, novelty, and relevance
- Prevents saturation from bulk ingestion

Based on cognitive science research:
- Baars (1988): Global Workspace Theory
- Dehaene et al.: Conscious Processing and Global Neuronal Workspace

The workspace acts as a filter between episodic buffer and consolidation,
ensuring only the most important episodes get consolidated into long-term
semantic memory.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceItem:
    """An item selected for the global workspace."""

    value_label: str
    salience: float
    source_id: str
    timestamp: float

    # For ranking
    novelty_score: float = 0.5
    relevance_score: float = 0.5

    @property
    def combined_score(self) -> float:
        """Combined score for workspace ranking."""
        # Weights: salience 50%, novelty 30%, relevance 20%
        return (
            0.5 * self.salience +
            0.3 * self.novelty_score +
            0.2 * self.relevance_score
        )


class GlobalWorkspace:
    """
    Implements the Baars Global Workspace bottleneck.

    The workspace has limited capacity (~50 items) and only the most
    salient episodes are selected for consolidation. This prevents
    memory saturation from bulk ingestion (e.g., Gutenberg books).

    Key properties:
    - Bounded capacity (default 50)
    - Priority by salience + novelty + relevance
    - User input prioritized over bulk sources
    - Automatic eviction of low-priority items

    Example:
        >>> workspace = GlobalWorkspace(capacity=50)
        >>> selected = workspace.select(pending_facts)
        >>> # selected contains top 50 items by combined score
    """

    def __init__(
        self,
        capacity: int = 50,
        salience_threshold: float = 0.3,
        user_priority_boost: float = 0.2,
    ):
        """
        Initialize the global workspace.

        Args:
            capacity: Maximum items in workspace (default 50)
            salience_threshold: Minimum salience to be considered (default 0.3)
            user_priority_boost: Extra score for user input (default 0.2)
        """
        self.capacity = capacity
        self.salience_threshold = salience_threshold
        self.user_priority_boost = user_priority_boost

        self._current_workspace: List[WorkspaceItem] = []
        self._selection_count = 0

    def select(
        self,
        candidates: List,
        existing_facts: Optional[List] = None,
        user_focus: Optional[str] = None,
    ) -> List[WorkspaceItem]:
        """
        Select top items for the global workspace.

        Implements the attentional bottleneck: only the most salient
        candidates pass through to consolidation.

        Args:
            candidates: List of PendingFact objects from episodic buffer
            existing_facts: List of existing semantic facts (for novelty scoring)
            user_focus: Current user focus/topic (for relevance scoring)

        Returns:
            List of WorkspaceItem objects selected for consolidation
        """
        if not candidates:
            return []

        # Score each candidate
        scored_items = []
        for candidate in candidates:
            # Basic salience from candidate
            salience = getattr(candidate, 'salience', 0.5)

            # Skip if below threshold
            if salience < self.salience_threshold:
                continue

            # Compute novelty (simplified: check if similar fact exists)
            novelty = self._compute_novelty(candidate, existing_facts)

            # Compute relevance to user focus
            relevance = self._compute_relevance(candidate, user_focus)

            # Apply user priority boost
            source_id = getattr(candidate, 'source_id', '')
            if source_id.startswith('user'):
                salience = min(1.0, salience + self.user_priority_boost)

            item = WorkspaceItem(
                value_label=getattr(candidate, 'value_label', str(candidate)),
                salience=salience,
                source_id=source_id,
                timestamp=getattr(candidate, 'timestamp', 0.0),
                novelty_score=novelty,
                relevance_score=relevance,
            )
            scored_items.append(item)

        # Sort by combined score (descending)
        scored_items.sort(key=lambda x: x.combined_score, reverse=True)

        # Take top N items
        selected = scored_items[:self.capacity]

        # Update internal state
        self._current_workspace = selected
        self._selection_count += 1

        logger.debug(
            f"Workspace selection #{self._selection_count}: "
            f"{len(selected)}/{len(candidates)} items selected "
            f"(threshold={self.salience_threshold})"
        )

        return selected

    def _compute_novelty(
        self,
        candidate,
        existing_facts: Optional[List],
    ) -> float:
        """
        Compute novelty score for a candidate.

        High novelty = fact doesn't exist in semantic memory yet.
        Low novelty = fact is already known.

        Args:
            candidate: PendingFact to score
            existing_facts: Existing semantic facts

        Returns:
            Novelty score 0-1 (higher = more novel)
        """
        if not existing_facts:
            return 0.8  # Assume novel if no existing facts

        value_label = getattr(candidate, 'value_label', '').lower()

        # Check if any existing fact contains similar content
        for fact in existing_facts:
            fact_label = getattr(fact, 'value_label',
                        getattr(fact, 'object', '')).lower()
            if fact_label and value_label:
                # Simple overlap check
                if fact_label in value_label or value_label in fact_label:
                    return 0.2  # Low novelty - already exists

        return 0.8  # High novelty - not found

    def _compute_relevance(
        self,
        candidate,
        user_focus: Optional[str],
    ) -> float:
        """
        Compute relevance to current user focus.

        High relevance = matches what user is interested in.
        Low relevance = unrelated to current conversation.

        Args:
            candidate: PendingFact to score
            user_focus: Current user topic/focus (optional)

        Returns:
            Relevance score 0-1 (higher = more relevant)
        """
        if not user_focus:
            return 0.5  # Neutral if no focus specified

        value_label = getattr(candidate, 'value_label', '').lower()
        user_focus_lower = user_focus.lower()

        # Simple keyword matching
        focus_words = set(user_focus_lower.split())
        label_words = set(value_label.split())

        if not focus_words or not label_words:
            return 0.5

        overlap = len(focus_words & label_words)
        max_possible = min(len(focus_words), len(label_words))

        if max_possible == 0:
            return 0.5

        return 0.3 + 0.7 * (overlap / max_possible)

    def get_current_workspace(self) -> List[WorkspaceItem]:
        """Get the current workspace contents."""
        return self._current_workspace.copy()

    def get_stats(self) -> dict:
        """Get workspace statistics."""
        return {
            'capacity': self.capacity,
            'current_size': len(self._current_workspace),
            'selection_count': self._selection_count,
            'salience_threshold': self.salience_threshold,
            'avg_salience': (
                sum(item.salience for item in self._current_workspace) /
                len(self._current_workspace)
                if self._current_workspace else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear the workspace."""
        self._current_workspace = []


def select_for_workspace(
    episodic_buffer: List,
    workspace_capacity: int = 50,
    salience_threshold: float = 0.3,
    existing_facts: Optional[List] = None,
    user_focus: Optional[str] = None,
) -> Tuple[List[WorkspaceItem], 'GlobalWorkspace']:
    """
    Convenience function to select items for workspace.

    Args:
        episodic_buffer: List of PendingFact objects
        workspace_capacity: Maximum items to select (default 50)
        salience_threshold: Minimum salience to be considered
        existing_facts: Existing semantic facts for novelty scoring
        user_focus: Current user topic for relevance scoring

    Returns:
        Tuple of (selected items, workspace instance)
    """
    workspace = GlobalWorkspace(
        capacity=workspace_capacity,
        salience_threshold=salience_threshold,
    )

    selected = workspace.select(
        candidates=episodic_buffer,
        existing_facts=existing_facts,
        user_focus=user_focus,
    )

    return selected, workspace
