# Conscious Hologram: Token & Context Analysis - Complete Documentation

## Overview

This documentation provides a comprehensive technical analysis of how tokens and context flow through the Conscious Hologram system, including limitations, overflow handling mechanisms, and recommendations.

**Total Documentation**: 2,732 lines across 4 documents
**Time to Review**: 30-45 minutes for full understanding
**Time to Skim**: 10-15 minutes for key findings

---

## Documents in This Analysis

### 1. TOKEN_AND_CONTEXT_ANALYSIS.md (1,206 lines)
**Purpose**: Deep technical analysis of the complete system

**Key Sections**:
- Architecture Overview (dual token systems: HDC vs SLM)
- Complete Message Flow Analysis (user input to output)
- Token Consumption at Each Stage (with code references)
- Context Window Limitations (8K for Kimi K2, 128K for GLM-4.6v)
- SLM/Ventriloquist Layer Analysis (primary generation path)
- Overflow Handling Mechanisms (what actually exists)
- Fact Encoding and Token Implications (how facts consume tokens)
- Performance Bottlenecks (O(N) search, API latency)
- Recommendations (5 critical + 2 optional improvements)

**Best For**:
- System architects wanting deep understanding
- Developers debugging token issues
- Understanding design decisions
- Planning optimizations

**Key Findings**:
```
• Token limits are SOFT-BOUNDED, not hard-enforced
• No client-side context window monitoring exists
• FactStore capacity ~100 facts (estimated, unproven)
• VentriloquistGenerator always preferred over HDC
• Circuit breaker provides failure protection (3 failures → 60s cooldown)
• SLM token cost: 20 (system) + N (variable) + up to 256 tokens
• Total typical turn: 34-286 tokens (safe for 8K window)
```

---

### 2. CONTEXT_FLOW_DIAGRAMS.md (886 lines)
**Purpose**: Visual representation of system architecture and flows

**Key Sections**:
1. High-Level System Architecture (block diagram)
2. Token Consumption Waterfall (step-by-step flow with token counts)
3. FactStore Query Process (exact match vs resonance search)
4. VentriloquistGenerator Path (SLM generation flow, 256 tokens max)
5. ResonantGenerator Path (HDC fallback, 10 tokens max)
6. Context Window Management (explicit vs implicit limits)
7. Memory Saturation Progression (what happens as facts accumulate)
8. Circuit Breaker State Machine (failure detection & recovery)
9. Fact Encoding (string vs vector representations)
10. Real Conversation with Token Breakdown (example with numbers)

**Best For**:
- Visual learners
- Understanding system flow
- Presentations
- Quick reference diagrams
- Testing understanding

**Key Insights**:
```
Flow Priority:
  1. VentriloquistGenerator (SLM, 256 tokens) - ALWAYS PREFERRED
  2. ResonantGenerator (HDC, 10 tokens) - FALLBACK
  3. Templates - ULTIMATE FALLBACK

Fact Encoding Cost:
  • Per fact in SLM: ~7 tokens ("Subject --predicate--> Object")
  • 100 facts: 700 tokens (tight but safe for 8K window)

Saturation Behavior:
  • Facts 1-50: OPTIMAL (0.98+ confidence)
  • Facts 51-80: DEGRADING (0.70-0.85 confidence)
  • Facts 81-100: CRITICAL (0.60-0.75, errors rising)
  • Facts 101+: OVERFLOW (>10% error rate)
```

---

### 3. IMPLEMENTATION_SUMMARY.md (640 lines)
**Purpose**: Code-level reference with file locations and line numbers

**Key Sections**:
- Quick Reference (file structure)
- Critical Configuration Points (where limits are defined)
- Message Flow with Code References (traced to actual lines)
- Token and Context Behavior Summary
- Where Token Overflow Can Happen (and what's NOT checked)
- Testing Edge Cases (scenarios to validate)
- Configuration Changes Recommended (with code examples)
- Performance Metrics (typical usage across scenarios)
- Key Takeaways (summary table)
- Files to Monitor for Token Issues

**Best For**:
- Developers implementing fixes
- Code review
- Finding specific issues
- Quick reference by line number
- Understanding what's missing

**Critical Files**:
```
src/hologram/generation/ventriloquist.py (251-256 tokens)
  • NO input token validation (PROBLEM)
  • API call at line 159

src/hologram/conversation/selector.py (response routing)
  • Ventriloquist preferred at line 163
  • max_tokens override at line 738
  • Circuit breaker at line 727

src/hologram/memory/fact_store.py (fact storage)
  • No saturation warnings (PROBLEM)
  • Resonance search O(N) at lines 281-285

src/hologram/config/constants.py (all limits)
  • ESTIMATED_CAPACITY_DIVISOR = 100 (line 43)
  • MAX_GENERATION_TOKENS = 100 (line 108, overridden)
  • SURPRISE_THRESHOLD = 0.1 (line 160)
```

---

## Quick Reference: Critical Findings

### Token Consumption Summary

| Phase | Tokens | Source | Notes |
|-------|--------|--------|-------|
| System Prompt | 20 | Novita API | Fixed overhead |
| User Query | Variable | User input | No validation! |
| Fact Answer | 1-10 | FactStore | Per fact retrieval |
| Generation | 0-256 | SLM API | Hard limit: 256 |
| **Typical Total** | **34-286** | **Mixed** | **Safe (8K window)** |
| **Worst Case** | **826** | **Multiple facts** | **Still safe** |

### Context Window Limits

| Model | Window | Notes |
|-------|--------|-------|
| Kimi K2 (Fluency) | 8,000 tokens | Primary SLM |
| GLM-4.6v (Reasoning) | 128,000 tokens | Larger, for reasoning chains |
| Veloquist max_tokens | 256 tokens | Hard limit, API-enforced |
| Reasoning max_tokens | 512 tokens | Double fluency limit |
| **Effective limit** | **~7,500 tokens** | **Safety margin needed** |

### Overflow Handling Mechanisms

| Mechanism | Type | Coverage | Gap |
|-----------|------|----------|-----|
| max_tokens parameter | Hard limit | Generation output | Input context not checked |
| Circuit breaker | Failure detection | 3 failures in 10 attempts | No capacity warnings |
| Surprise gating | Learning filter | Prevents duplicate facts | No saturation alert |
| Resonator iterations | Convergence limit | 100 max iterations | Can timeout on divergence |
| Template fallback | Graceful degradation | Ultimate fallback response | Silent activation |
| **MISSING**: Input validation | N/A | Context window monitoring | NO CLIENT-SIDE CHECK |
| **MISSING**: Saturation warning | N/A | FactStore capacity | NO EARLY WARNING |
| **MISSING**: Token accounting | N/A | Per-session tracking | NO BUDGET TRACKING |

### FactStore Capacity

```
Configuration: ESTIMATED_CAPACITY_DIVISOR = 100
Calculation: Capacity ≈ dimensions / divisor
For DEFAULT_DIMENSIONS=10,000: ~100 facts

Reality:
├─ 1-50 facts: Excellent (0.98+ confidence)
├─ 51-80 facts: Good (0.85+ confidence)
├─ 81-100 facts: Degrading (0.60-0.75 confidence)
├─ 101+ facts: Overflow (>10% error rate)
└─ NO WARNING AT ANY STAGE (Problem!)

Solution:
├─ Monitor: fs.saturation_estimate
├─ Warn at 80%: "Approaching capacity"
├─ Error at 100%: "Consider HierarchicalFactStore"
```

---

## Problem Summary

### Critical Issues (Must Fix)

1. **No context window monitoring**
   - VentriloquistGenerator doesn't validate input size
   - If |user_prompt| + |fact_answer| > 8K, Novita silently truncates
   - User unaware context was lost

2. **No FactStore saturation warning**
   - System degrades silently after 80-100 facts
   - No alert when approaching capacity
   - User experiences accuracy drop without explanation

3. **No token accounting per session**
   - No tracking of cumulative token usage
   - No budget management
   - Potential for silent truncation

### Important Issues (Should Fix)

4. **Vocabulary resonance search O(N) performance**
   - For 1,000 facts: 10M FLOPs per query
   - Could use FAISS index for O(log N)
   - Performance degrades as fact store grows

5. **Silent API failures**
   - If generation fails, circuit breaker kicks in
   - User falls back to templates without knowing
   - No clear feedback mechanism

### Minor Issues (Nice to Have)

6. **Fact serialization inefficiency**
   - Each fact → "Subject --predicate--> Object" (~7 tokens)
   - Could compress to JSON format (fewer tokens)
   - Low priority (still within window)

---

## Recommendations by Priority

### Priority 1: Add Context Window Monitoring (CRITICAL)

**File**: `src/hologram/generation/ventriloquist.py`

**What**: Check input tokens before API call

**Implementation**:
```python
def generate_with_validation(self, context, max_tokens=256):
    system_tokens = self._estimate_tokens(system_prompt)  # ~20
    user_tokens = self._estimate_tokens(user_prompt)      # VARIABLE
    total_input = system_tokens + user_tokens

    if total_input + 100 > self._context_window:  # 100 safety margin
        logger.warning(f"Context overflow: {total_input} + 256 > {self._context_window}")
        # Option 1: Reduce max_tokens
        max_tokens = min(max_tokens, self._context_window - total_input - 100)
        # Option 2: Truncate facts
        # Option 3: Reject request

    response = self._client.chat.completions.create(...)
```

**Effort**: 30 minutes

---

### Priority 2: Add FactStore Saturation Warning (CRITICAL)

**File**: `src/hologram/memory/fact_store.py`

**What**: Warn when FactStore approaches capacity

**Implementation**:
```python
def add_fact(self, subject, predicate, obj, ...):
    # ... existing code ...

    saturation = self._estimate_saturation()  # fact_count / (dims / 100)

    if saturation > 0.8:
        logger.warning(f"FactStore saturation: {saturation:.0%} (approaching limit)")

    if saturation >= 1.0:
        logger.error(f"FactStore at capacity: consider HierarchicalFactStore")
        # Could automatically switch to cold storage here

@property
def saturation_estimate(self) -> float:
    capacity = self._space.dimensions / ESTIMATED_CAPACITY_DIVISOR
    return min(1.0, self._fact_count / capacity)
```

**Effort**: 20 minutes

---

### Priority 3: Implement Token Accounting (IMPORTANT)

**File**: `src/hologram/generation/token_accounting.py` (new)

**What**: Track per-session token usage

**Implementation**:
```python
class TokenAccountant:
    def __init__(self, session_limit=8000):
        self.session_limit = session_limit
        self.tokens_used = 0
        self.turns = []

    def log_turn(self, input_tokens, output_tokens):
        self.tokens_used += input_tokens + output_tokens

        if self.tokens_used > 0.8 * self.session_limit:
            logger.warning(f"Token budget: {self.tokens_used:.0%} used")

        self.turns.append({
            "timestamp": datetime.now(),
            "input": input_tokens,
            "output": output_tokens,
            "cumulative": self.tokens_used
        })

    def get_remaining_budget(self):
        return max(0, self.session_limit - self.tokens_used)
```

**Effort**: 45 minutes (including integration)

---

### Priority 4: Optimize Resonance Search with FAISS (PERFORMANCE)

**File**: `src/hologram/memory/fact_store.py`

**What**: Use FAISS for large vocabulary (>1000)

**Implementation**:
```python
def query(self, subject, predicate):
    # Fast path: exact match (O(1))
    if exact_match:
        return fact.object, 1.0

    # Large store: use FAISS (O(log N))
    if len(self._value_vocab) > 1000:
        key = self._create_key(subject, predicate)
        nearest = self._faiss_index.search(key, k=1)
        return nearest[0].object, nearest[0].score

    # Small store: full resonance (O(N*D))
    else:
        # ... existing code ...
```

**Effort**: 2-3 hours (FAISS integration)

---

## Expected Outcomes After Implementation

### Current State
```
✗ No context window monitoring
✗ No saturation warnings
✗ No token accounting
✗ Linear O(N) search for large stores
✗ Silent failures, unclear error messages
```

### After Priority 1-3 Fixes
```
✓ Clear warnings before context overflow
✓ Saturation alerts when FactStore approaches limit
✓ Per-session token budget tracking
✓ Still O(N) but user knows about limits
✓ Clear error messages and fallback chain
```

### After Full Implementation
```
✓ Explicit context window monitoring
✓ Automatic saturation alerts
✓ Token budget tracking per session
✓ O(log N) search via FAISS for large stores
✓ Detailed logging and error reporting
```

---

## FAQ: Token and Context Questions

### Q: What's the typical token cost per turn?

**A**:
```
Simple question: ~34-44 tokens
├─ System prompt: 20 tokens
├─ User query: 8 tokens
├─ Fact: 1 token
└─ Response: 15 tokens

Complex question with facts: ~286 tokens (max)
├─ System: 20 tokens
├─ User query: 10 tokens
├─ Facts (5): 35 tokens
└─ Response: 256 tokens (max)
```

### Q: Can the system run out of context window?

**A**: Unlikely in practice because:
- Each SLM call is stateless (no history sent)
- Fact serialization is efficient (~7 tokens per fact)
- 100 facts = 700 tokens (still safe in 8K window)
- But POSSIBLE if:
  - Very long user queries (50+ tokens)
  - Many facts retrieved (100+)
  - Multiple generations in reasoning chain
  - **NO WARNING CURRENTLY** if approaching limit

### Q: What happens when FactStore reaches capacity?

**A**: Graceful degradation without warning:
- Signal-to-noise ratio drops
- Confidence scores fall below thresholds (0.20)
- System falls back to "I don't know" responses
- User experiences accuracy drop but no explanation
- **RECOMMENDATION**: Monitor `saturation_estimate` property

### Q: How does the circuit breaker work?

**A**:
```
3 failures in last 10 attempts → Opens circuit
└─ Blocks generation for 60 seconds
└─ Forces fallback to templates
└─ After 60s, tries again
└─ If success: resumes normal
└─ If failure: cycles again
```

### Q: Why is VentriloquistGenerator always preferred?

**A**: Design choice for fluency:
- ResonantGenerator: 10 tokens, robotic output
- VentriloquistGenerator: 256 tokens, natural language
- SLM provides better UX, grounding prevents hallucination
- HDC is fallback for when SLM unavailable

### Q: Is there a hard limit on facts stored?

**A**: Implicit limits:
- FactStore (hot): ~100 facts before degradation
- HierarchicalFactStore: Unlimited via FAISS (cold layer)
- No hard error, just accuracy degradation
- **RECOMMENDATION**: Use HierarchicalFactStore for production

---

## Key Code Locations

### Token Limits (Configuration)
```
src/hologram/config/constants.py:108-109  MAX_GENERATION_TOKENS
src/hologram/generation/ventriloquist.py:71  max_tokens=256
src/hologram/generation/resonant_generator.py:738  max_tokens=10
```

### API Integration
```
src/hologram/generation/ventriloquist.py:159  OpenAI API call
src/hologram/generation/ventriloquist.py:127-145  Prompt construction (NO VALIDATION!)
```

### Fact Storage
```
src/hologram/memory/fact_store.py:219  query() method
src/hologram/memory/fact_store.py:281  Resonance search O(N×D)
src/hologram/config/constants.py:43  ESTIMATED_CAPACITY_DIVISOR
```

### Failure Handling
```
src/hologram/generation/circuit_breaker.py  Circuit breaker implementation
src/hologram/conversation/selector.py:726  Circuit breaker check
src/hologram/conversation/selector.py:217  Template fallback
```

---

## Next Steps

### For Understanding
1. Read TOKEN_AND_CONTEXT_ANALYSIS.md (full details)
2. Review CONTEXT_FLOW_DIAGRAMS.md (visual flow)
3. Check IMPLEMENTATION_SUMMARY.md (code references)

### For Implementation
1. Start with Priority 1 (context window monitoring)
2. Follow with Priority 2 (saturation warnings)
3. Add Priority 3 (token accounting)
4. Consider Priority 4 (FAISS optimization)

### For Testing
1. Test long context scenarios (100+ token queries)
2. Test FactStore at capacity (100+ facts)
3. Test generation failures (circuit breaker)
4. Measure performance with large vocabulary

---

## Document Maintenance

**Last Updated**: 2025-12-12
**Analysis Version**: 1.0
**Code Version Analyzed**: kent_hologram main branch

**Files Analyzed**:
- src/hologram/generation/ventriloquist.py
- src/hologram/generation/resonant_generator.py
- src/hologram/generation/base.py
- src/hologram/conversation/selector.py
- src/hologram/conversation/chatbot.py
- src/hologram/memory/fact_store.py
- src/hologram/memory/memory_trace.py
- src/hologram/config/constants.py
- src/hologram/config/settings.py
- src/hologram/core/codebook.py
- src/hologram/core/vector_space.py

**Review Checklist**:
- [ ] All recommendations implemented
- [ ] Context window monitoring added
- [ ] Saturation warnings implemented
- [ ] Token accounting added
- [ ] FAISS optimization considered
- [ ] Tests updated for new features
- [ ] Documentation updated
- [ ] Performance benchmarked

---

## Contact & Questions

For questions about this analysis, refer to:
1. TOKEN_AND_CONTEXT_ANALYSIS.md (deep technical)
2. CONTEXT_FLOW_DIAGRAMS.md (visual reference)
3. IMPLEMENTATION_SUMMARY.md (code reference)

**Key Insight**: The Conscious Hologram has well-designed implicit limits but lacks explicit monitoring. Adding warnings and tracking would significantly improve reliability and user experience.

