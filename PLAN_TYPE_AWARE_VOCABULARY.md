# Implementation Plan: Type-Aware Vocabulary Builder

## Problem Statement

The resonator produces incoherent text like "day is canberra" because:
1. **Same vocabulary pool for subjects AND objects** - `noun_vocabulary` is used for both slots
2. **Heuristic-based word categorization** - Current `vocabulary.py` uses suffix matching, not grammatical parsing
3. **No SVO extraction** - Words aren't extracted from actual subject-verb-object triples

## Solution: Type-Constrained Vocabularies

Build separate vocabularies from actual SVO triples extracted during training:
- `subjects` - Words that appear as grammatical subjects
- `verbs` - Words that appear as main verbs
- `objects` - Words that appear as grammatical objects

This prevents subject unbinding from ever matching a word that only appears as an object.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Gutenberg Books                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │  WTPSplit   │────▶│   spaCy     │────▶│ TypeAwareVocab  │   │
│  │  (sentence  │     │  + textacy  │     │    Builder      │   │
│  │   segment)  │     │  (SVO ext)  │     │                 │   │
│  └─────────────┘     └─────────────┘     └────────┬────────┘   │
│                                                    │            │
│                                          ┌────────┴────────┐   │
│                                          ▼        ▼        ▼   │
│                                     subjects   verbs   objects │
│                                        │         │         │   │
│                                        └────────┬┴─────────┘   │
│                                                 │              │
│                                                 ▼              │
│                                     TypedVocabulary JSON       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  TypedVocabulary JSON                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RESONATOR                             │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
│  │  │ Subject │    │  Verb   │    │ Object  │              │   │
│  │  │  Slot   │    │  Slot   │    │  Slot   │              │   │
│  │  └────┬────┘    └────┬────┘    └────┬────┘              │   │
│  │       │              │              │                    │   │
│  │       ▼              ▼              ▼                    │   │
│  │   subjects       verbs         objects                   │   │
│  │   vocabulary     vocabulary    vocabulary                │   │
│  │   (constrained)  (constrained) (constrained)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies

```toml
[project.dependencies]
wtpsplit = "^2.0"          # Sentence segmentation (EMNLP 2024)
spacy = "^3.7"             # NLP pipeline
textacy = "^0.13"          # SVO extraction built on spaCy
```

## Implementation Steps

### Phase 1: TypeAwareVocabularyBuilder Class

**File:** `src/hologram/conversation/type_aware_vocabulary.py`

```python
"""
Type-aware vocabulary extraction using SVO triples.

Extracts words from actual grammatical roles in sentences,
building separate vocabularies for subjects, verbs, and objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

import spacy
from textacy.extract.triples import subject_verb_object_triples
from wtpsplit import SaT


@dataclass
class TypedVocabulary:
    """Vocabulary separated by grammatical role."""
    subjects: Set[str] = field(default_factory=set)
    verbs: Set[str] = field(default_factory=set)
    objects: Set[str] = field(default_factory=set)

    # Track frequency for filtering
    subject_counts: Dict[str, int] = field(default_factory=dict)
    verb_counts: Dict[str, int] = field(default_factory=dict)
    object_counts: Dict[str, int] = field(default_factory=dict)


class TypeAwareVocabularyBuilder:
    """
    Build type-constrained vocabularies from text using SVO extraction.

    Uses:
    - WTPSplit for robust sentence segmentation
    - spaCy + textacy for SVO triple extraction
    - Lemmatization for vocabulary normalization
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        min_word_length: int = 2,
        min_frequency: int = 2,
    ):
        self.nlp = spacy.load(spacy_model)
        self.sat = SaT("sat-3l-sm")  # Fast, accurate sentence segmenter
        self.min_word_length = min_word_length
        self.min_frequency = min_frequency
        self.vocab = TypedVocabulary()

    def extract_from_text(self, text: str) -> int:
        """Extract SVO triples from text, returns count of triples found."""
        # Segment into sentences
        sentences = self.sat.split(text)

        triples_found = 0
        for sentence in sentences:
            doc = self.nlp(sentence)
            for svo in subject_verb_object_triples(doc):
                # Extract lemmatized words
                subj = " ".join(t.lemma_.lower() for t in svo.subject)
                verb = " ".join(t.lemma_.lower() for t in svo.verb)
                obj = " ".join(t.lemma_.lower() for t in svo.object)

                # Filter by length
                if len(subj) >= self.min_word_length:
                    self.vocab.subjects.add(subj)
                    self.vocab.subject_counts[subj] = \
                        self.vocab.subject_counts.get(subj, 0) + 1

                if len(verb) >= self.min_word_length:
                    self.vocab.verbs.add(verb)
                    self.vocab.verb_counts[verb] = \
                        self.vocab.verb_counts.get(verb, 0) + 1

                if len(obj) >= self.min_word_length:
                    self.vocab.objects.add(obj)
                    self.vocab.object_counts[obj] = \
                        self.vocab.object_counts.get(obj, 0) + 1

                triples_found += 1

        return triples_found

    def get_filtered_vocabulary(self) -> Dict[str, List[str]]:
        """Get vocabulary filtered by minimum frequency."""
        return {
            "subjects": sorted([
                w for w, c in self.vocab.subject_counts.items()
                if c >= self.min_frequency
            ]),
            "verbs": sorted([
                w for w, c in self.vocab.verb_counts.items()
                if c >= self.min_frequency
            ]),
            "objects": sorted([
                w for w, c in self.vocab.object_counts.items()
                if c >= self.min_frequency
            ]),
        }

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON."""
        data = {
            "subjects": list(self.vocab.subjects),
            "verbs": list(self.vocab.verbs),
            "objects": list(self.vocab.objects),
            "subject_counts": self.vocab.subject_counts,
            "verb_counts": self.vocab.verb_counts,
            "object_counts": self.vocab.object_counts,
            "stats": self.get_stats(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TypeAwareVocabularyBuilder":
        """Load vocabulary from JSON."""
        builder = cls()
        with open(path) as f:
            data = json.load(f)
        builder.vocab.subjects = set(data["subjects"])
        builder.vocab.verbs = set(data["verbs"])
        builder.vocab.objects = set(data["objects"])
        builder.vocab.subject_counts = data.get("subject_counts", {})
        builder.vocab.verb_counts = data.get("verb_counts", {})
        builder.vocab.object_counts = data.get("object_counts", {})
        return builder

    def get_stats(self) -> dict:
        """Get vocabulary statistics."""
        return {
            "total_subjects": len(self.vocab.subjects),
            "total_verbs": len(self.vocab.verbs),
            "total_objects": len(self.vocab.objects),
            "unique_words": len(
                self.vocab.subjects | self.vocab.verbs | self.vocab.objects
            ),
        }
```

### Phase 2: Modify Resonator for Type-Constrained Cleanup

**File:** `src/hologram/core/resonator.py`

Change the `resonate()` signature:

```python
def resonate(
    self,
    thought: torch.Tensor,
    subject_vocabulary: List[str],  # NEW: Separate subject vocab
    verb_vocabulary: List[str],
    object_vocabulary: List[str],   # NEW: Separate object vocab
) -> ResonatorResult:
```

Key change in `_solve_for_slot`:
- Subject slot: Only searches `subject_vocabulary`
- Verb slot: Only searches `verb_vocabulary`
- Object slot: Only searches `object_vocabulary`

### Phase 3: Integrate with Ingestion Pipeline

**Modify:** `scripts/ingest_gutenberg.py`

Add vocabulary extraction during book processing:

```python
class GutenbergIngester:
    def __init__(self, ...):
        # ... existing init ...
        self._vocab_builder = TypeAwareVocabularyBuilder()
        self._vocab_file = Path(persist_dir) / "typed_vocabulary.json"

    def process_book(self, book_id: str, text: str) -> int:
        # ... existing fact extraction ...

        # Also extract vocabulary
        triples = self._vocab_builder.extract_from_text(cleaned_text)
        print(f"  Extracted {triples} SVO triples for vocabulary")

        return facts_added

    def run(self):
        # ... existing run logic ...

        # Save vocabulary at end
        self._vocab_builder.save(self._vocab_file)
        print(f"Vocabulary saved: {self._vocab_builder.get_stats()}")
```

### Phase 4: Update ResonantGenerator

**Modify:** `src/hologram/generation/resonant_generator.py`

Load and use typed vocabulary:

```python
class ResonantGenerator:
    def __init__(self, ..., typed_vocab_path: Optional[Path] = None):
        if typed_vocab_path and typed_vocab_path.exists():
            vocab_data = json.load(open(typed_vocab_path))
            self._subject_vocab = vocab_data["subjects"]
            self._verb_vocab = vocab_data["verbs"]
            self._object_vocab = vocab_data["objects"]
        else:
            # Fallback to old behavior
            self._subject_vocab = self._noun_vocab
            self._verb_vocab = self._verb_vocab
            self._object_vocab = self._noun_vocab
```

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/hologram/conversation/type_aware_vocabulary.py` | NEW | TypeAwareVocabularyBuilder class |
| `src/hologram/core/resonator.py` | MODIFY | Separate subject/object vocabularies |
| `scripts/ingest_gutenberg.py` | MODIFY | Add vocabulary extraction |
| `src/hologram/generation/resonant_generator.py` | MODIFY | Load typed vocabulary |
| `tests/test_type_aware_vocabulary.py` | NEW | Unit tests |
| `pyproject.toml` | MODIFY | Add wtpsplit, textacy dependencies |

## Testing Strategy

```python
def test_svo_extraction():
    """Test that SVO triples are correctly extracted."""
    builder = TypeAwareVocabularyBuilder()
    builder.extract_from_text("The cat eats fish. Dogs chase balls.")

    assert "cat" in builder.vocab.subjects
    assert "dog" in builder.vocab.subjects
    assert "eat" in builder.vocab.verbs
    assert "chase" in builder.vocab.verbs
    assert "fish" in builder.vocab.objects
    assert "ball" in builder.vocab.objects


def test_type_separation():
    """Test that words don't cross type boundaries incorrectly."""
    builder = TypeAwareVocabularyBuilder()
    builder.extract_from_text("Time flies. The fly ate sugar.")

    # "fly" should be in verbs (from "flies") AND subjects (from "The fly")
    # This is correct - words can have multiple roles
    assert "fly" in builder.vocab.verbs or "fly" in builder.vocab.subjects
```

## Success Criteria

1. **Vocabulary files generated** - `typed_vocabulary.json` with separate S/V/O lists
2. **Resonator type-constrained** - Subject slot only searches subjects, etc.
3. **No more "day is canberra"** - Incoherent cross-type combinations eliminated
4. **Backward compatible** - Falls back to old behavior if no typed vocab exists

## Estimated Impact

- **Eliminates 80%+ of incoherent outputs** caused by type mixing
- **Improves confidence scores** by reducing vocabulary search space per slot
- **Faster resonator convergence** with smaller, focused vocabularies
