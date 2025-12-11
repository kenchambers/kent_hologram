"""
CitationEnforcer: Fact tracing for every output.

Ensures every claim can be traced back to a stored fact with source.
No unsupported statements allowed - this is critical for bounded hallucination.
"""

from typing import Optional

from hologram.memory.fact_store import Fact, FactStore


class CitationEnforcer:
    """
    Enforces citations for all outputs.

    Every claim must trace back to a stored fact. This prevents the system
    from making unsupported statements - a key component of bounded hallucination.

    The principle: If we can't cite it, we can't claim it.

    Attributes:
        fact_store: FactStore to validate against

    Example:
        >>> enforcer = CitationEnforcer(fact_store)
        >>> fact = enforcer.find_supporting_fact("France", "capital", "Paris")
        >>> enforcer.format_citation(fact)
        '[Wikipedia] France --capital--> Paris'
    """

    def __init__(self, fact_store: FactStore):
        """
        Initialize citation enforcer.

        Args:
            fact_store: FactStore containing ground truth facts
        """
        self.fact_store = fact_store

    def find_supporting_fact(
        self,
        subject: str,
        predicate: str,
        claimed_object: str
    ) -> Optional[Fact]:
        """
        Find fact that supports a claim.

        Searches stored facts for one matching the claim.
        If found, this fact can be cited as evidence.

        Args:
            subject: Claimed subject
            predicate: Claimed predicate
            claimed_object: Claimed object/value

        Returns:
            Matching Fact if found, None otherwise

        Example:
            >>> enforcer = CitationEnforcer(fact_store)
            >>> fact = enforcer.find_supporting_fact("Earth", "shape", "round")
            >>> fact.source
            'NASA'
        """
        # Get all facts with matching subject and predicate
        matching_facts = [
            f for f in self.fact_store._facts
            if f.subject.lower() == subject.lower()
            and f.predicate.lower() == predicate.lower()
        ]

        # Find exact match for object
        for fact in matching_facts:
            if fact.object.lower() == claimed_object.lower():
                return fact

        return None

    def validate_claim(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> tuple[bool, Optional[Fact]]:
        """
        Validate that a claim is supported by stored facts.

        Args:
            subject: Subject of claim
            predicate: Predicate of claim
            obj: Object of claim

        Returns:
            Tuple of (is_valid, supporting_fact)

        Example:
            >>> enforcer = CitationEnforcer(fact_store)
            >>> valid, fact = enforcer.validate_claim("France", "capital", "Paris")
            >>> valid
            True
            >>> fact.source
            'Wikipedia'
        """
        fact = self.find_supporting_fact(subject, predicate, obj)
        return (fact is not None, fact)

    def format_citation(self, fact: Fact) -> str:
        """
        Format a fact as a citation.

        Args:
            fact: Fact to cite

        Returns:
            Citation string

        Example:
            >>> fact = Fact("Earth", "shape", "round", source="NASA")
            >>> enforcer.format_citation(fact)
            '[NASA] Earth --shape--> round'
        """
        source_str = f"[{fact.source}] " if fact.source else ""
        return f"{source_str}{fact.subject} --{fact.predicate}--> {fact.object}"

    def attach_citations(
        self,
        response: str,
        facts: list[Fact]
    ) -> str:
        """
        Attach citation block to response.

        Args:
            response: Response text
            facts: Facts to cite

        Returns:
            Response with citation block appended

        Example:
            >>> enforcer = CitationEnforcer(fact_store)
            >>> response = "Paris is the capital of France."
            >>> facts = [Fact("France", "capital", "Paris", source="Wikipedia")]
            >>> print(enforcer.attach_citations(response, facts))
            Paris is the capital of France.

            Citations:
            - [Wikipedia] France --capital--> Paris
        """
        if not facts:
            return response

        citation_block = "\n\nCitations:\n"
        for fact in facts:
            citation_block += f"- {self.format_citation(fact)}\n"

        return response + citation_block

    def get_all_citations(self) -> list[str]:
        """
        Get formatted citations for all stored facts.

        Returns:
            List of citation strings

        Example:
            >>> enforcer = CitationEnforcer(fact_store)
            >>> citations = enforcer.get_all_citations()
            >>> len(citations)
            10
        """
        return [self.format_citation(f) for f in self.fact_store._facts]

    def __repr__(self) -> str:
        return (
            f"CitationEnforcer(facts={self.fact_store.fact_count}, "
            f"vocabulary={self.fact_store.vocabulary_size})"
        )
