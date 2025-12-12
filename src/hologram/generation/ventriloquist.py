"""
VentriloquistGenerator: SLM-based generation for fluent natural language.

Uses a Small Language Model (via Novita API) to generate fluent responses
based on facts retrieved by the Conscious Hologram.

This implements the "Voice Box" metaphor: the Hologram (Brain) retrieves facts,
and the SLM (Voice) speaks them naturally.

Enhanced with code verification and dual retrieval for coding tasks.
"""

import ast
import os
import json
from typing import Optional, List, Dict, Any, Set, Tuple

from openai import OpenAI

from hologram.generation.base import GenerationContext, Generator
from hologram.generation.resonant_generator import (
    GenerationResult,
    GenerationTrace,
    GenerationMetrics,
)
from hologram.cavity.divergence import DivergenceAction
from hologram.config.constants import (
    DEFAULT_REASONING_MODEL,
    DEFAULT_FLUENCY_MODEL,
    ENABLE_REASONING_CHAIN,
)


class VentriloquistGenerator:
    """
    Dual-model generator using Novita API.
    
    Takes facts retrieved by the Hologram and generates fluent natural language
    responses. Supports two modes:
    1. Fluency mode: Uses Kimi K2 for conversational responses
    2. Reasoning mode: Uses GLM-4.6v for multi-step deduction
    
    This solves the "fluency" problem while preserving HDC's bounded
    hallucination (only speaks what the Brain found).
    
    Attributes:
        _client: OpenAI-compatible client for Novita API
        _fluency_model: Model for fluent responses (default: kimi-k2-thinking)
        _reasoning_model: Model for reasoning chains (default: glm-4.6v)
        _temperature: Sampling temperature (default: 0.7)
        _max_tokens: Maximum tokens per response (default: 256)
        _enable_reasoning: Whether to use reasoning chains (default: True)
    
    Example:
        >>> generator = VentriloquistGenerator()
        >>> context = GenerationContext(
        ...     query_text="What is the capital of France?",
        ...     fact_answer="Paris",
        ...     ...
        ... )
        >>> result = generator.generate_with_validation(context)
        >>> print(result.text)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        fluency_model: str = DEFAULT_FLUENCY_MODEL,
        reasoning_model: str = DEFAULT_REASONING_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 256,
        enable_reasoning: bool = ENABLE_REASONING_CHAIN,
    ):
        """
        Initialize Ventriloquist generator.
        
        Args:
            api_key: Novita API key (default: from NOVITA_API_KEY env var)
            fluency_model: Model for fluent responses (default: from constants)
            reasoning_model: Model for reasoning chains (default: from constants)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens per response (default: 256)
            enable_reasoning: Enable chain-of-thought reasoning (default: True)
        """
        api_key = api_key or os.getenv("NOVITA_API_KEY")
        if not api_key:
            raise ValueError(
                "NOVITA_API_KEY not found. Set it in .env file or pass api_key parameter."
            )
        
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.novita.ai/openai"
        )
        self._fluency_model = fluency_model
        self._reasoning_model = reasoning_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_reasoning = enable_reasoning
        # Simple, model-agnostic context windows (tokens); adjust if models change
        self._fluency_context_window = 8000
        self._reasoning_context_window = 128000
        self._safety_buffer = 256

    def _estimate_tokens(self, text: str) -> int:
        """Crude token estimator (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _budget_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        requested_max_tokens: int,
        context_window: int,
        safety_buffer: int,
    ) -> Optional[int]:
        """
        Cap max_tokens so input+output fits in the assumed context window.

        Returns capped max_tokens, or None if there is no room for any output.
        """
        input_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        available_for_output = context_window - input_tokens - safety_buffer
        if available_for_output <= 0:
            return None
        return max(1, min(requested_max_tokens, available_for_output))
    
    def generate_with_validation(
        self,
        context: GenerationContext,
        max_tokens: Optional[int] = None,
    ) -> Optional[GenerationResult]:
        """
        Generate fluent response using SLM.
        
        Constructs a prompt that includes:
        - User's query
        - Retrieved fact (if available)
        - Style instructions
        
        The SLM generates a natural language response based on this context.
        
        Args:
            context: GenerationContext with query, facts, and style
            max_tokens: Override default max_tokens (optional)
            
        Returns:
            GenerationResult with fluent text, or None if generation fails
        """
        if max_tokens is None:
            max_tokens = self._max_tokens
        
        # Build prompt based on whether we have a fact answer
        if context.fact_answer:
            # Fact-based response: Hologram found the answer, SLM phrases it
            system_prompt = (
                "You are a helpful assistant. Answer questions naturally and fluently "
                "using the provided facts. Be conversational but accurate."
            )
            episodes_text = ""
            if context.episodes:
                episodes_text = "Recent episodes:\n" + "\n".join(f"- {e}" for e in context.episodes[:3]) + "\n\n"
            user_prompt = (
                f"Question: {context.query_text}\n\n"
                f"Fact: {context.fact_answer}\n\n"
                f"{episodes_text}"
                f"Answer the question using the fact above. Be natural and conversational."
            )
        else:
            # Conversational response: No specific fact, use general knowledge
            system_prompt = (
                "You are a helpful assistant. Respond naturally and conversationally. "
                "If you don't know something, say so politely."
            )
            episodes_text = ""
            if context.episodes:
                episodes_text = "Recent episodes:\n" + "\n".join(f"- {e}" for e in context.episodes[:3]) + "\n\n"
            user_prompt = episodes_text + context.query_text
        
        # Add style instructions
        style_instructions = {
            "formal": "Use formal, professional language.",
            "casual": "Use casual, friendly language.",
            "urgent": "Be concise and direct.",
            "neutral": "Use neutral, clear language.",
        }
        style_key = context.style.value if hasattr(context.style, 'value') else str(context.style)
        if style_key in style_instructions:
            system_prompt += f" {style_instructions[style_key]}"
        
        try:
            # Apply simple context budgeting for the fluency model
            capped_max_tokens = self._budget_tokens(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                requested_max_tokens=max_tokens,
                context_window=self._fluency_context_window,
                safety_buffer=self._safety_buffer,
            )
            if capped_max_tokens is None:
                return None

            # Call Novita API (using fluency model)
            response = self._client.chat.completions.create(
                model=self._fluency_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=capped_max_tokens,
                temperature=self._temperature,
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content.strip()
            
            if not generated_text:
                return None
            
            # Validate: Check if fact_answer appears in response (if provided)
            if context.fact_answer:
                fact_lower = context.fact_answer.lower()
                text_lower = generated_text.lower()

                # First, check for exact substring match (handles names like "Emperor Vex")
                if fact_lower in text_lower:
                    # Exact match found - validation passed
                    pass
                else:
                    # Fall back to word-level matching for longer phrases
                    # Keep words >= 2 chars (was > 2, which excluded 3-letter words like "Vex")
                    fact_words = [w for w in fact_lower.split() if len(w) >= 2]
                    if fact_words:
                        matches = sum(1 for word in fact_words if word in text_lower)
                        if matches < len(fact_words) * 0.5:
                            # Fact not properly incorporated - reject
                            return None
            
            # Create GenerationResult (SLM doesn't provide trace/metrics)
            tokens = generated_text.split()
            trace = [
                GenerationTrace(
                    token=token,
                    type="TOKEN",
                    similarity=1.0,  # SLM doesn't have similarity scores
                    action=DivergenceAction.ACCEPT,  # SLM output is always accepted
                    role="SLM_OUTPUT",
                )
                for token in tokens
            ]
            
            metrics = GenerationMetrics(
                total_tokens=len(tokens),
                accepted_first_try=len(tokens),
                average_similarity=1.0,  # SLM doesn't have similarity
            )
            
            return GenerationResult(
                text=generated_text,
                tokens=tokens,
                trace=trace,
                divergence_history=[],  # SLM doesn't have divergence history
                metrics=metrics,
                resonator_result=None,  # SLM doesn't use resonator
            )
            
        except Exception as e:
            # Generation failed - return None to fall back to templates
            return None
    
    def generate(
        self,
        context: GenerationContext,
        max_tokens: Optional[int] = None,
    ) -> Optional[GenerationResult]:
        """
        Generate response without validation (for compatibility).
        
        Args:
            context: GenerationContext
            max_tokens: Override default max_tokens
            
        Returns:
            GenerationResult or None
        """
        return self.generate_with_validation(context, max_tokens)
    
    def generate_reasoning_chain(
        self,
        query: str,
        facts: List[str],
        fact_store: Optional[Any] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate step-by-step reasoning chain using GLM-4.6v.
        
        This method uses the larger reasoning model to perform multi-step
        deductions based on retrieved facts. Each step is verified against
        the FactStore to prevent hallucination.
        
        Args:
            query: The question to answer
            facts: List of retrieved facts from FactStore
            fact_store: Optional FactStore for verification (can be None)
            max_tokens: Override default max_tokens (optional)
            
        Returns:
            Dict with:
            - reasoning_steps: List of reasoning steps
            - conclusion: Final answer
            - confidence: Confidence score based on verification
            Or None if reasoning fails
            
        Example:
            >>> facts = ["France --capital--> Paris", "Paris --population--> 2.2M"]
            >>> result = generator.generate_reasoning_chain(
            ...     query="How many people live in France's capital?",
            ...     facts=facts
            ... )
            >>> result["conclusion"]
            "2.2 million people"
        """
        if not self._enable_reasoning:
            return None
        
        if max_tokens is None:
            max_tokens = self._max_tokens * 2  # Reasoning needs more tokens
        
        # Build reasoning prompt
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        
        system_prompt = (
            "You are a logical reasoning assistant. Given facts, deduce the answer "
            "step-by-step. Output your reasoning as JSON with this structure:\n"
            "{\n"
            '  "reasoning_steps": [\n'
            '    "Step 1: ...",\n'
            '    "Step 2: ..."\n'
            "  ],\n"
            '  "conclusion": "Final answer"\n'
            "}\n"
            "Only use information from the provided facts. Do not invent information."
        )
        
        user_prompt = (
            f"Question: {query}\n\n"
            f"Known Facts:\n{facts_text}\n\n"
            f"Provide step-by-step reasoning to answer the question using ONLY the facts above."
        )
        
        try:
            capped_max_tokens = self._budget_tokens(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                requested_max_tokens=max_tokens,
                context_window=self._reasoning_context_window,
                safety_buffer=self._safety_buffer * 2,
            )
            if capped_max_tokens is None:
                return None

            # Call reasoning model (GLM-4.6v)
            response = self._client.chat.completions.create(
                model=self._reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=capped_max_tokens,
                temperature=0.3,  # Lower temperature for reasoning
                response_format={"type": "json_object"},  # Request JSON output
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            reasoning_chain = json.loads(content)
            
            # Verify the reasoning chain
            if fact_store is not None:
                verification_result = self._verify_deduction(reasoning_chain, fact_store, facts)
                reasoning_chain["verified"] = verification_result["verified"]
                reasoning_chain["confidence"] = verification_result["confidence"]
            else:
                reasoning_chain["verified"] = True  # Skip verification if no fact_store
                reasoning_chain["confidence"] = 0.8  # Default confidence
            
            return reasoning_chain
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Reasoning failed - return None
            return None
    
    def _verify_deduction(
        self,
        reasoning_chain: Dict[str, Any],
        fact_store: Any,
        known_facts: List[str],
    ) -> Dict[str, Any]:
        """
        Verify that reasoning steps don't contradict known facts.
        
        This is the "Holographic Grounding" guardrail - ensures that all
        deductions are traceable back to facts in the FactStore.
        
        Args:
            reasoning_chain: The reasoning chain from GLM-4.6v
            fact_store: FactStore instance for verification
            known_facts: List of facts that were provided to the reasoner
            
        Returns:
            Dict with:
            - verified: True if no contradictions found
            - confidence: Score based on how well grounded the reasoning is
            - issues: List of any problems found
        """
        issues = []
        reasoning_steps = reasoning_chain.get("reasoning_steps", [])
        conclusion = reasoning_chain.get("conclusion", "")
        
        # Check 1: Conclusion must reference at least one known fact
        conclusion_lower = conclusion.lower()
        known_fact_strings = [f.lower() for f in known_facts]
        
        # Extract key terms from facts
        fact_terms = set()
        for fact in known_fact_strings:
            # Simple extraction: split on common delimiters
            parts = fact.replace("--", " ").replace("-->", " ").split()
            fact_terms.update([p.strip() for p in parts if len(p.strip()) > 3])
        
        # Check if conclusion uses fact terms
        conclusion_words = set(conclusion_lower.split())
        grounded_terms = fact_terms.intersection(conclusion_words)
        
        if len(grounded_terms) == 0:
            issues.append("Conclusion does not reference any known facts")
        
        # Check 2: Each reasoning step should reference known information
        for i, step in enumerate(reasoning_steps):
            step_lower = step.lower()
            step_words = set(step_lower.split())
            step_grounded = fact_terms.intersection(step_words)
            
            if len(step_grounded) == 0:
                issues.append(f"Step {i+1} may not be grounded in facts")
        
        # Calculate confidence based on grounding
        if len(issues) == 0:
            confidence = 0.9  # High confidence - fully grounded
        elif len(issues) == 1:
            confidence = 0.6  # Medium confidence - partially grounded
        else:
            confidence = 0.3  # Low confidence - poorly grounded
        
        return {
            "verified": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
        }
    
    def verify_code(
        self,
        code_str: str,
        fact_store: Optional[Any] = None,
        concept_store: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verify generated code against indexed facts.
        
        This prevents hallucination by checking that:
        1. All function calls reference known functions (from code index)
        2. Function signatures match stored signatures
        3. Classes inherit from known base classes
        
        Args:
            code_str: Generated code to verify
            fact_store: Project-specific code index (FactStore)
            concept_store: General coding concepts (FactStore)
            
        Returns:
            Dict with:
            - verified: True if code passes verification
            - issues: List of verification issues found
            - confidence: Overall confidence score
            - unknown_functions: Set of referenced functions not in index
            - unknown_classes: Set of referenced classes not in index
        """
        issues = []
        unknown_functions = set()
        unknown_classes = set()
        
        # Parse the code
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return {
                "verified": False,
                "issues": [f"Syntax error: {e}"],
                "confidence": 0.0,
                "unknown_functions": unknown_functions,
                "unknown_classes": unknown_classes,
            }
        
        # Extract function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name:
                    # Check if function exists in code index
                    if fact_store:
                        func_facts = fact_store.get_facts_by_subject(func_name)
                        func_exists = any(f.predicate == "type" and f.object == "function" 
                                        for f in func_facts)
                        
                        if not func_exists:
                            # Check in concept store (for standard library/patterns)
                            if concept_store:
                                concept_facts = concept_store.get_facts_by_subject(func_name)
                                concept_exists = len(concept_facts) > 0
                                if not concept_exists:
                                    unknown_functions.add(func_name)
                            else:
                                unknown_functions.add(func_name)
            
            # Extract class references
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Check base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        
                        if fact_store:
                            base_facts = fact_store.get_facts_by_subject(base_name)
                            base_exists = any(f.predicate == "type" and f.object == "class" 
                                            for f in base_facts)
                            
                            if not base_exists:
                                if concept_store:
                                    concept_facts = concept_store.get_facts_by_subject(base_name)
                                    concept_exists = len(concept_facts) > 0
                                    if not concept_exists:
                                        unknown_classes.add(base_name)
                                else:
                                    unknown_classes.add(base_name)
        
        # Generate issues
        if unknown_functions:
            issues.append(f"Unknown functions: {', '.join(unknown_functions)}")
        if unknown_classes:
            issues.append(f"Unknown classes: {', '.join(unknown_classes)}")
        
        # Calculate confidence
        total_unknowns = len(unknown_functions) + len(unknown_classes)
        if total_unknowns == 0:
            confidence = 1.0
        elif total_unknowns <= 2:
            confidence = 0.7  # Some unknowns, but might be stdlib
        else:
            confidence = 0.3  # Many unknowns - likely hallucination
        
        verified = len(issues) == 0
        
        return {
            "verified": verified,
            "issues": issues,
            "confidence": confidence,
            "unknown_functions": unknown_functions,
            "unknown_classes": unknown_classes,
        }
    
    def retrieve_dual_context(
        self,
        query: str,
        fact_store: Optional[Any] = None,
        concept_store: Optional[Any] = None,
        max_facts: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieve both general concepts and project-specific facts.
        
        This enables the generator to combine:
        - General coding knowledge (design patterns, algorithms)
        - Project-specific context (API signatures, class hierarchy)
        
        Args:
            query: Query string (e.g., "implement factory pattern for User class")
            fact_store: Project-specific code index
            concept_store: General coding concepts
            max_facts: Maximum facts to retrieve from each store
            
        Returns:
            Tuple of (concept_facts, project_facts) as formatted strings
        """
        concept_facts = []
        project_facts = []
        
        # Extract key terms from query (simple keyword extraction)
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if len(word) > 3]
        
        # Retrieve from concept store (design patterns, algorithms, etc.)
        if concept_store:
            for keyword in keywords:
                # Try exact match first
                facts = concept_store.get_facts_by_subject(keyword.capitalize())
                for fact in facts[:max_facts]:
                    fact_str = f"{fact.subject} --{fact.predicate}--> {fact.object}"
                    if fact_str not in concept_facts:
                        concept_facts.append(fact_str)
        
        # Retrieve from project fact store (classes, functions, etc.)
        if fact_store:
            for keyword in keywords:
                # Try exact match
                facts = fact_store.get_facts_by_subject(keyword)
                for fact in facts[:max_facts]:
                    fact_str = f"{fact.subject} --{fact.predicate}--> {fact.object}"
                    if fact_str not in project_facts:
                        project_facts.append(fact_str)
                
                # Also try capitalized version (for class names)
                facts = fact_store.get_facts_by_subject(keyword.capitalize())
                for fact in facts[:max_facts]:
                    fact_str = f"{fact.subject} --{fact.predicate}--> {fact.object}"
                    if fact_str not in project_facts:
                        project_facts.append(fact_str)
        
        return concept_facts[:max_facts], project_facts[:max_facts]
    
    def generate_code_with_context(
        self,
        prompt: str,
        fact_store: Optional[Any] = None,
        concept_store: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        verify: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate code with dual context retrieval and verification.
        
        Combines general coding knowledge with project-specific facts
        to generate verified, hallucination-free code.
        
        Args:
            prompt: Coding task description
            fact_store: Project-specific code index
            concept_store: General coding concepts
            max_tokens: Maximum tokens to generate
            verify: Whether to verify generated code
            
        Returns:
            Dict with:
            - code: Generated code string
            - verification: Verification results
            - concepts_used: List of concepts retrieved
            - project_facts_used: List of project facts retrieved
            Or None if generation fails
        """
        if max_tokens is None:
            max_tokens = self._max_tokens * 2  # Code needs more tokens
        
        # Retrieve dual context
        concept_facts, project_facts = self.retrieve_dual_context(
            prompt,
            fact_store=fact_store,
            concept_store=concept_store
        )
        
        # Build enhanced prompt with context
        system_prompt = (
            "You are an expert Python programmer. Generate clean, correct code "
            "using the provided context. Use the design patterns and project APIs "
            "shown below. Do NOT invent functions or classes that aren't listed."
        )
        
        context_parts = []
        
        if concept_facts:
            context_parts.append("General Coding Knowledge:")
            for fact in concept_facts:
                context_parts.append(f"  - {fact}")
        
        if project_facts:
            context_parts.append("\nProject-Specific Context:")
            for fact in project_facts:
                context_parts.append(f"  - {fact}")
        
        context_str = "\n".join(context_parts) if context_parts else "No specific context available."
        
        user_prompt = f"{context_str}\n\nTask: {prompt}\n\nGenerate the code:"
        
        try:
            capped_max_tokens = self._budget_tokens(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                requested_max_tokens=max_tokens,
                context_window=self._reasoning_context_window,
                safety_buffer=self._safety_buffer * 2,
            )
            if capped_max_tokens is None:
                return None

            # Generate code using reasoning model (GLM-4.6v)
            response = self._client.chat.completions.create(
                model=self._reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=capped_max_tokens,
                temperature=0.3,  # Lower temperature for code generation
            )
            
            generated_code = response.choices[0].message.content.strip()
            
            # Extract code block if wrapped in markdown
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            # Verify code if requested
            verification = None
            if verify:
                verification = self.verify_code(
                    generated_code,
                    fact_store=fact_store,
                    concept_store=concept_store
                )
            
            return {
                "code": generated_code,
                "verification": verification,
                "concepts_used": concept_facts,
                "project_facts_used": project_facts,
            }
            
        except Exception as e:
            return None
    
    def __repr__(self) -> str:
        return (
            f"VentriloquistGenerator("
            f"fluency={self._fluency_model}, "
            f"reasoning={self._reasoning_model}, "
            f"temperature={self._temperature})"
        )

    # ------------------------------------------------------------------ #
    # Long-form pipeline: outline -> expand -> verify
    # ------------------------------------------------------------------ #
    def generate_long_form(
        self,
        query: str,
        facts: Optional[List[str]] = None,
        episodes: Optional[List[str]] = None,
        sections: int = 4,
        section_tokens: int = 256,
    ) -> Optional[str]:
        """
        Produce longer responses in two passes:
        1) Outline in JSON with reasoning model.
        2) Expand each section with the fluency model.
        """
        facts = facts or []
        episodes = episodes or []

        # --- Pass 1: Outline
        outline_system = (
            "You are an expert planner. Create a JSON outline for a response.\n"
            'Respond ONLY as JSON: {"sections": ["title 1", "title 2", ...]}.'
        )
        context_lines = []
        if facts:
            context_lines.append("Known facts:")
            context_lines.extend(f"- {f}" for f in facts[:8])
        if episodes:
            context_lines.append("Recent episodes:")
            context_lines.extend(f"- {e}" for e in episodes[:3])
        outline_user = "\n".join(context_lines + [f"Question: {query}"])

        capped_outline_tokens = self._budget_tokens(
            system_prompt=outline_system,
            user_prompt=outline_user,
            requested_max_tokens=128,
            context_window=self._reasoning_context_window,
            safety_buffer=self._safety_buffer * 2,
        )
        if capped_outline_tokens is None:
            return None

        try:
            outline_resp = self._client.chat.completions.create(
                model=self._reasoning_model,
                messages=[
                    {"role": "system", "content": outline_system},
                    {"role": "user", "content": outline_user},
                ],
                max_tokens=capped_outline_tokens,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            outline_content = outline_resp.choices[0].message.content.strip()
            outline_json = json.loads(outline_content)
            section_titles = outline_json.get("sections", [])
        except Exception:
            return None

        if not section_titles:
            return None

        # --- Pass 2: Expand each section
        assembled_sections = []
        fact_terms = self._extract_fact_terms(facts)
        for title in section_titles[:sections]:
            expand_system = (
                "You are a helpful assistant. Write a concise section that stays grounded "
                "in the provided facts/episodes. Avoid inventing new facts."
            )
            expand_user_parts = [f"Section: {title}", f"Question: {query}"]
            if facts:
                expand_user_parts.append("Facts:")
                expand_user_parts.extend(f"- {f}" for f in facts[:8])
            if episodes:
                expand_user_parts.append("Episodes:")
                expand_user_parts.extend(f"- {e}" for e in episodes[:3])
            expand_user = "\n".join(expand_user_parts)

            capped_section_tokens = self._budget_tokens(
                system_prompt=expand_system,
                user_prompt=expand_user,
                requested_max_tokens=section_tokens,
                context_window=self._fluency_context_window,
                safety_buffer=self._safety_buffer,
            )
            if capped_section_tokens is None:
                continue

            try:
                section_resp = self._client.chat.completions.create(
                    model=self._fluency_model,
                    messages=[
                        {"role": "system", "content": expand_system},
                        {"role": "user", "content": expand_user},
                    ],
                    max_tokens=capped_section_tokens,
                    temperature=self._temperature,
                )
                section_text = section_resp.choices[0].message.content.strip()
            except Exception:
                continue

            if not section_text:
                continue

            # Lightweight grounding check: ensure some fact terms appear
            if fact_terms:
                lower = section_text.lower()
                matches = sum(1 for term in fact_terms if term in lower)
                if matches == 0:
                    # Skip ungrounded section
                    continue

            assembled_sections.append(f"{title}\n{section_text}")

        if not assembled_sections:
            return None

        return "\n\n".join(assembled_sections)

    def _extract_fact_terms(self, facts: List[str]) -> set:
        terms = set()
        for fact in facts:
            for token in fact.lower().replace("--", " ").replace("-->", " ").split():
                if len(token) > 3:
                    terms.add(token)
        return terms

