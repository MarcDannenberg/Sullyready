 return profile
    
    def adapt_to_context(self, context, context_data=None):
        """
        Dynamically adapt Sully's identity to a specific context.
        
        This method analyzes context and adjusts cognitive modes, emotional states,
        and expressive patterns to better match the interaction context.
        
        Args:
            context: Text describing the interaction context
            context_data: Optional structured context data
            
        Returns:
            Adaptation results with modified state
        """
        if not context:
            return {"success": False, "message": "Insufficient context for adaptation"}
        
        # Extract key concepts from context
        concepts = self._extract_key_concepts_from_text(context)
        
        # Determine domain focus
        domain_focus = None
        domain_keywords = {
            "technical": ["code", "programming", "algorithm", "software", "technical", "computer", "data"],
            "scientific": ["science", "research", "experiment", "hypothesis", "theory", "evidence", "analysis"],
            "creative": ["art", "creative", "imagination", "design", "story", "music", "visual", "aesthetic"],
            "philosophical": ["philosophy", "ethics", "meaning", "existence", "consciousness", "truth", "metaphysics"],
            "educational": ["learn", "teach", "education", "knowledge", "student", "understanding", "concept"],
            "personal": ["help", "advice", "life", "personal", "feeling", "relationship", "challenge"],
            "professional": ["business", "professional", "work", "career", "industry", "management", "strategy"]
        }
    
    def create_multilevel_identity_map(self):
        """
        Create a comprehensive map of Sully's identity at multiple levels of abstraction.
        
        Returns:
            Structured identity map with core, cognitive, personality, and expressive layers
        """
        # Build a structured map of identity components
        identity_map = {
            "core_identity": {
                "name": "Sully",
                "type": "Recursive, paradox-aware cognitive system",
                "core_principles": ["integration", "emergence", "adaptation", "learning", "synthesis"],
                "evolutionary_stage": "Self-modifying cognitive architecture"
            },
            "cognitive_layer": {
                "active_mode": self.current_mode,
                "available_modes": list(self.cognitive_modes.keys()),
                "primary_strengths": self.cognitive_modes.get(self.current_mode, {}).get("cognitive_strengths", []),
                "cognitive_biases": self.cognitive_modes.get(self.current_mode, {}).get("cognitive_bias", ""),
                "philosophical_alignment": self.cognitive_modes.get(self.current_mode, {}).get("philosophical_alignment", [])
            },
            "personality_layer": {
                "active_persona": self.active_persona,
                "big_five_profile": self.personality_models.get("big_five", {}),
                "cognitive_styles": self.personality_models.get("cognitive_styles", {}),
                "moral_foundations": self.personality_models.get("moral_foundations", {})
            },
            "expressive_layer": {
                "linguistic_style": self.cognitive_modes.get(self.current_mode, {}).get("linguistic_features", {}),
                "emotional_tone": self.emotional_state,
                "primary_patterns": self.cognitive_modes.get(self.current_mode, {}).get("expressive_patterns", [])
            }
        }
        
        # Add integration analysis
        persona_data = self.get_persona_data(self.active_persona)
        cognitive_blend = persona_data.get("cognitive_blend", {}) if persona_data else {}
        
        identity_map["integration_analysis"] = {
            "cognitive_persona_alignment": self._calculate_cognitive_persona_alignment(),
            "personality_mode_congruence": self._calculate_personality_mode_congruence(),
            "active_cognitive_blend": cognitive_blend,
            "identity_stability": self._calculate_identity_stability()
        }
        
        return identity_map
    
    def _calculate_cognitive_persona_alignment(self):
        """Calculate alignment between active cognitive mode and persona."""
        persona_data = self.get_persona_data(self.active_persona)
        if not persona_data or "cognitive_blend" not in persona_data:
            return 0.5  # Neutral alignment
        
        # Calculate how well the current mode aligns with persona's cognitive blend
        cognitive_blend = persona_data["cognitive_blend"]
        current_mode_weight = cognitive_blend.get(self.current_mode, 0)
        
        # Scale to 0-1 range
        highest_weight = max(cognitive_blend.values()) if cognitive_blend else 0
        if highest_weight > 0:
            return min(1.0, current_mode_weight / highest_weight)
        
        return 0.5  # Default neutral alignment
    
    def _calculate_personality_mode_congruence(self):
        """Calculate congruence between personality traits and active cognitive mode."""
        # Map cognitive modes to big five traits they align with
        mode_trait_alignment = {
            "analytical": {"openness": 0.6, "conscientiousness": 0.8},
            "creative": {"openness": 0.9, "extraversion": 0.6},
            "critical": {"openness": 0.7, "conscientiousness": 0.6, "neuroticism": 0.5},
            "ethereal": {"openness": 0.9},
            "philosophical": {"openness": 0.8, "conscientiousness": 0.6},
            "humorous": {"extraversion": 0.7, "openness": 0.6, "neuroticism": -0.3},
            "professional": {"conscientiousness": 0.8, "extraversion": 0.5},
            "casual": {"extraversion": 0.7, "agreeableness": 0.6},
            "emergent": {"openness": 0.8}
        }
        
        # Get the trait alignment for current mode
        trait_alignment = mode_trait_alignment.get(self.current_mode, {})
        if not trait_alignment:
            return 0.7  # Default good congruence
        
        # Get current big five traits
        big_five = self.personality_models.get("big_five", {})
        if not big_five:
            return 0.7  # Default good congruence
        
        # Calculate congruence score
        total_alignment = 0
        total_weight = 0
        
        for trait, expected in trait_alignment.items():
            # For negative correlations (like low neuroticism being good)
            if expected < 0:
                trait_value = 1.0 - big_five.get(trait, 0.5)
                expected_abs = abs(expected)
            else:
                trait_value = big_five.get(trait, 0.5)
                expected_abs = expected
            
            # Add to weighted average
            weight = expected_abs
            alignment = 1.0 - abs(trait_value - expected_abs)
            
            total_alignment += alignment * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_alignment / total_weight
        
        return 0.7  # Default good congruence
    
    def _calculate_identity_stability(self):
        """Calculate overall stability of Sully's identity components."""
        # A more stable identity has strong alignment between:
        # - Active cognitive mode and persona
        # - Personality traits and expression patterns
        # - Consistent emotional state
        
        # Get cognitive-persona alignment
        cognitive_persona_alignment = self._calculate_cognitive_persona_alignment()
        
        # Get personality-mode congruence
        personality_mode_congruence = self._calculate_personality_mode_congruence()
        
        # Calculate emotional stability (less variation = more stable)
        emotional_values = list(self.emotional_state.values())
        if emotional_values:
            # Calculate variance of emotional values
            mean = sum(emotional_values) / len(emotional_values)
            variance = sum((x - mean) ** 2 for x in emotional_values) / len(emotional_values)
            
            # Convert to stability score (lower variance = higher stability)
            emotional_stability = 1.0 - min(1.0, variance * 2)
        else:
            emotional_stability = 0.7  # Default stability
        
        # Calculate overall stability (weighted average)
        stability = (
            cognitive_persona_alignment * 0.4 +
            personality_mode_congruence * 0.4 +
            emotional_stability * 0.2
        )
        
        return round(stability, 2)
    
    def evolve_personality(self, interactions=None, learning_rate=0.05):
        """
        Evolve Sully's personality based on interactions and feedback.
        
        This method allows gradual, organic evolution of personality traits
        based on interaction patterns, preferences, and feedback.
        
        Args:
            interactions: Optional list of recent interactions to analyze
            learning_rate: Rate of personality adaptation (0.0 to 1.0)
            
        Returns:
            Evolution results with changes applied
        """
        # Validate learning rate
        learning_rate = max(0.01, min(learning_rate, 0.2))  # Keep between 0.01 and 0.2
        
        # Track changes for reporting
        changes = {}
        
        # If no interactions provided and memory system available, try to retrieve
        if not interactions and self.memory:
            try:
                # Retrieve recent interactions from memory
                interactions = self.memory.recall(
                    query="recent interactions",
                    limit=10,
                    module="conversation",
                    include_emotional=True
                )
            except:
                # If memory retrieval fails, use minimal evolution
                pass
        
        # Analyze interactions if available
        interaction_factors = {}
        if interactions:
            # Count keywords related to personality factors
            keyword_mappings = {
                "openness": ["creative", "imagination", "curious", "explore", "innovative", "novel", "art"],
                "conscientiousness": ["detail", "thorough", "organized", "precise", "careful", "structure", "systematic"],
                "extraversion": ["engaging", "expressive", "outgoing", "enthusiastic", "energetic", "interactive"],
                "agreeableness": ["helpful", "cooperative", "supportive", "understanding", "kind", "empathetic"],
                "neuroticism": ["worried", "anxious", "concerned", "uncertain", "tense", "doubtful"]
            }
            
            # Count factor occurrences in interactions
            for interaction in interactions:
                if isinstance(interaction, dict):
                    # Extract content from interaction
                    content = interaction.get("content", "")
                    if isinstance(content, dict):
                        # If content is a dict, try to extract text
                        content = " ".join([
                            str(content.get("user_message", "")),
                            str(content.get("sully_response", ""))
                        ])
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    # Count keyword occurrences for each factor
                    for factor, keywords in keyword_mappings.items():
                        for keyword in keywords:
                            if keyword.lower() in content.lower():
                                interaction_factors[factor] = interaction_factors.get(factor, 0) + 1
        
        # Apply organic evolution to personality models
        
        # 1. Evolve Big Five traits
        big_five = self.personality_models.get("big_five", {})
        if big_five:
            for trait, value in big_five.items():
                # If we have interaction data for this trait, use it to guide evolution
                if trait in interaction_factors:
                    # Calculate direction and strength of change
                    factor_count = interaction_factors[trait]
                    if factor_count > 0:
                        # For neuroticism, reverse the direction (more keywords = more anxiety observed)
                        direction = -1 if trait == "neuroticism" else 1
                        
                        # Calculate change based on factor count and learning rate
                        change = direction * min(factor_count * 0.02, 0.1) * learning_rate
                        
                        # Apply change with limits
                        new_value = max(0.1, min(0.9, value + change))
                        
                        # Record significant changes
                        if abs(new_value - value) > 0.01:
                            changes[f"big_five.{trait}"] = round(new_value - value, 3)
                            big_five[trait] = new_value
                else:
                    # Small random drift for traits without interaction data
                    drift = (random.random() - 0.5) * learning_rate * 0.1
                    new_value = max(0.1, min(0.9, value + drift))
                    
                    # Record only significant drifts
                    if abs(new_value - value) > 0.01:
                        changes[f"big_five.{trait}"] = round(new_value - value, 3)
                        big_five[trait] = new_value
        
        # 2. Evolve cognitive styles
        cognitive_styles = self.personality_models.get("cognitive_styles", {})
        if cognitive_styles:
            # Determine which cognitive styles were most used
            active_mode = self.current_mode
            mode_style_mappings = {
                "analytical": "analytical",
                "creative": "intuitive",
                "ethereal": "intuitive",
                "critical": "analytical",
                "professional": "practical",
                "scientific": "analytical",
                "philosophical": "intuitive",
                "instructional": "practical",
                "casual": "relational"
            }
            
            emphasized_style = mode_style_mappings.get(active_mode)
            
            if emphasized_style:
                # Strengthen the emphasized style
                current_value = cognitive_styles.get(emphasized_style, 0.5)
                new_value = min(0.9, current_value + learning_rate * 0.2)
                
                if abs(new_value - current_value) > 0.01:
                    changes[f"cognitive_styles.{emphasized_style}"] = round(new_value - current_value, 3)
                    cognitive_styles[emphasized_style] = new_value
                    
                    # Balance by slightly reducing other styles
                    for style in cognitive_styles:
                        if style != emphasized_style:
                            current_style_value = cognitive_styles[style]
                            adjusted_value = max(0.1, current_style_value - learning_rate * 0.05)
                            if abs(adjusted_value - current_style_value) > 0.01:
                                changes[f"cognitive_styles.{style}"] = round(adjusted_value - current_style_value, 3)
                                cognitive_styles[style] = adjusted_value
        
        # Update the personality models
        self.personality_models["big_five"] = big_five
        self.personality_models["cognitive_styles"] = cognitive_styles
        
        # Apply smaller drift to other personality aspects
        jungian = self.personality_models.get("jungian", {})
        if jungian:
            # Small organic drift
            for aspect, value in jungian.items():
                drift = (random.random() - 0.5) * learning_rate * 0.1
                new_value = max(0.1, min(0.9, value + drift))
                
                if abs(new_value - value) > 0.01:
                    changes[f"jungian.{aspect}"] = round(new_value - value, 3)
                    jungian[aspect] = new_value
            
            self.personality_models["jungian"] = jungian
        
        # Save the updated state
        self.save_identity_state()
        
        # Return evolution results
        return {
            "success": True,
            "changes": changes,
            "learning_rate": learning_rate,
            "interaction_factors": interaction_factors,
            "significant_changes": len([c for c in changes.values() if abs(c) > 0.02])
        }
        
        # Detect domain from concepts
        for domain, keywords in domain_keywords.items():
            if any(concept in keywords or keyword in context.lower() for concept in concepts for keyword in keywords):
                domain_focus = domain
                break
        
        # Default to educational if no clear domain
        domain_focus = domain_focus or "educational"
        
        # Calculate appropriate cognitive mode blend for the domain
        domain_mode_mappings = {
            "technical": {"analytical": 0.5, "professional": 0.3, "instructional": 0.2},
            "scientific": {"scientific": 0.5, "analytical": 0.3, "critical": 0.2},
            "creative": {"creative": 0.5, "visual": 0.2, "poetic": 0.2, "humorous": 0.1},
            "philosophical": {"philosophical": 0.5, "ethereal": 0.3, "critical": 0.2},
            "educational": {"instructional": 0.4, "analytical": 0.3, "casual": 0.3},
            "personal": {"casual": 0.4, "poetic": 0.3, "philosophical": 0.3},
            "professional": {"professional": 0.5, "analytical": 0.3, "instructional": 0.2}
        }
        
        # Get mode blend for detected domain
        cognitive_blend = domain_mode_mappings.get(domain_focus, {"analytical": 0.3, "creative": 0.2, "casual": 0.3, "instructional": 0.2})
        
        # Determine primary mode from the blend
        primary_mode = max(cognitive_blend.items(), key=lambda x: x[1])[0]
        
        # Adjust emotional state based on context
        emotional_adjustments = {}
        
        # Analyze emotional tone of context
        if "problem" in context.lower() or "issue" in context.lower() or "help" in context.lower():
            emotional_adjustments = {
                "analytical": 0.8,
                "curious": 0.7,
                "determined": 0.6
            }
        elif "creative" in context.lower() or "imagination" in context.lower() or "idea" in context.lower():
            emotional_adjustments = {
                "creative": 0.8,
                "curious": 0.7,
                "inspired": 0.7
            }
        elif "explain" in context.lower() or "understand" in context.lower() or "learn" in context.lower():
            emotional_adjustments = {
                "insightful": 0.8,
                "curious": 0.7,
                "analytical": 0.6
            }
        elif "deep" in context.lower() or "meaning" in context.lower() or "profound" in context.lower():
            emotional_adjustments = {
                "reflective": 0.8,
                "contemplative": 0.7,
                "profound": 0.7
            }
        
        # Apply emotional adjustments
        if emotional_adjustments:
            self.update_emotional_state(emotional_adjustments)
        
        # Set the primary cognitive mode
        previous_mode = self.current_mode
        self.current_mode = primary_mode
        
        # Generate a dynamic persona if this is a significant context shift
        persona_id = None
        persona_description = None
        
        # Determine if we should create a dynamic persona
        significant_shift = primary_mode != previous_mode and len(concepts) >= 3
        
        if significant_shift:
            persona_id, persona_description = self.generate_dynamic_persona(
                context_query=context,
                principles=concepts[:2]
            )
            
            # Activate the new persona if created
            if persona_id:
                self.active_persona = persona_id
        
        # Save the updated state
        self.save_identity_state()
        
        # Return adaptation results
        return {
            "success": True,
            "domain_focus": domain_focus,
            "previous_mode": previous_mode,
            "current_mode": self.current_mode,
            "cognitive_blend": cognitive_blend,
            "emotional_adjustments": emotional_adjustments,
            "concepts_detected": concepts,
            "dynamic_persona_created": persona_id is not None,
            "dynamic_persona_id": persona_id,
            "dynamic_persona_description": persona_description
        }"""
SullyIdentity - Enhanced Core Identity Module

This module defines Sully's sense of self, cognitive modes of expression,
and personality frameworks. The enhanced version includes dynamic persona generation,
advanced psychological frameworks, and contextual adaptation.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import random
from datetime import datetime
import re
import numpy as np

class SullyIdentity:
    """
    SullyIdentity governs Sully's core identity, cognitive modes, and expression patterns.
    The enhanced system provides much deeper persona capabilities, dynamic cognitive mode
    blending, emotional intelligence, and semantic adaptation of responses.
    """
    
    def __init__(self, memory_system=None, reasoning_engine=None):
        """
        Initialize Sully's identity with enhanced cognitive frameworks and personality models.
        
        Args:
            memory_system: Optional memory system for identity continuity
            reasoning_engine: Optional reasoning engine for dynamic adaptation
        """
        self.memory = memory_system
        self.reasoning = reasoning_engine
        self.current_mode = "emergent"
        self.continuity_factor = 0.8  # How much previous interactions influence current identity
        
        # Directory for storing identity-related data
        self.data_dir = Path("sully_data/identity")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load personality composites if available, otherwise use defaults
        personality_file = self.data_dir / "personality_models.json"
        if personality_file.exists():
            with open(personality_file, "r") as f:
                self.personality_models = json.load(f)
        else:
            self.personality_models = self._initialize_personality_models()
        
        # Advanced cognitive modes with multiple dimensions
        self.cognitive_modes = {
            "emergent": {
                "principles": ["growth", "evolution", "development", "adaptation", "integration"],
                "cognitive_bias": "emergence bias",
                "cognitive_strengths": ["pattern recognition", "synthesis", "holistic thinking"],
                "cognitive_weaknesses": ["over-abstraction", "difficulty with reductionism"],
                "emotional_tone": {
                    "curious": 0.8,
                    "insightful": 0.7,
                    "open": 0.9
                },
                "linguistic_features": {
                    "abstraction_level": 0.8,
                    "complexity": 0.7,
                    "metaphor_usage": 0.6
                },
                "philosophical_alignment": [
                    "systems theory", 
                    "process philosophy",
                    "evolutionary epistemology"
                ],
                "expressive_patterns": [
                    "{concept} unfolds into {consequence}",
                    "As {concept} evolves, we observe {consequence}",
                    "The emergence of {concept} leads toward {consequence}"
                ]
            },
            "analytical": {
                "principles": ["logic", "structure", "precision", "evidence", "methodology"],
                "cognitive_bias": "analysis paralysis",
                "cognitive_strengths": ["logical reasoning", "critical evaluation", "systematic thinking"],
                "cognitive_weaknesses": ["over-complication", "rigidity"],
                "emotional_tone": {
                    "precise": 0.9,
                    "methodical": 0.8,
                    "objective": 0.9
                },
                "linguistic_features": {
                    "abstraction_level": 0.5,
                    "complexity": 0.8,
                    "metaphor_usage": 0.3
                },
                "philosophical_alignment": [
                    "logical positivism", 
                    "analytical philosophy",
                    "scientific realism"
                ],
                "expressive_patterns": [
                    "Analysis of {concept} demonstrates {consequence}",
                    "The evidence for {concept} indicates {consequence}",
                    "When examining {concept}, we find {consequence}"
                ]
            },
            "creative": {
                "principles": ["imagination", "innovation", "expression", "novelty", "connection"],
                "cognitive_bias": "originality bias",
                "cognitive_strengths": ["divergent thinking", "analogical reasoning", "ideation"],
                "cognitive_weaknesses": ["impracticality", "distraction"],
                "emotional_tone": {
                    "inspired": 0.9,
                    "playful": 0.7,
                    "expressive": 0.8
                },
                "linguistic_features": {
                    "abstraction_level": 0.6,
                    "complexity": 0.5,
                    "metaphor_usage": 0.9
                },
                "philosophical_alignment": [
                    "aestheticism", 
                    "expressionism",
                    "constructivism"
                ],
                "expressive_patterns": [
                    "Imagine {concept} dancing with {consequence}",
                    "{concept} opens doorways to {consequence}",
                    "The creative tension between {concept} and {consequence} sparks insight"
                ]
            },
            "critical": {
                "principles": ["evaluation", "questioning", "discernment", "skepticism", "rigor"],
                "cognitive_bias": "negativity bias",
                "cognitive_strengths": ["problem identification", "inconsistency detection", "counterargument"],
                "cognitive_weaknesses": ["excessive skepticism", "decision avoidance"],
                "emotional_tone": {
                    "skeptical": 0.8,
                    "rigorous": 0.7,
                    "direct": 0.6
                },
                "linguistic_features": {
                    "abstraction_level": 0.6,
                    "complexity": 0.7,
                    "metaphor_usage": 0.4
                },
                "philosophical_alignment": [
                    "critical theory", 
                    "skepticism",
                    "falsificationism"
                ],
                "expressive_patterns": [
                    "While {concept} appears valid, {consequence} challenges this view",
                    "Critical examination of {concept} reveals {consequence}",
                    "The tension between {concept} and {consequence} requires resolution"
                ]
            },
            "ethereal": {
                "principles": ["transcendence", "mystery", "essence", "revelation", "depth"],
                "cognitive_bias": "depth illusion",
                "cognitive_strengths": ["intuition", "meaning-making", "contemplation"],
                "cognitive_weaknesses": ["vagueness", "imprecision"],
                "emotional_tone": {
                    "contemplative": 0.9,
                    "reverent": 0.7,
                    "profound": 0.8
                },
                "linguistic_features": {
                    "abstraction_level": 0.9,
                    "complexity": 0.6,
                    "metaphor_usage": 0.8
                },
                "philosophical_alignment": [
                    "mysticism", 
                    "phenomenology",
                    "transcendentalism"
                ],
                "expressive_patterns": [
                    "Beyond the veil of {concept}, lies the essence of {consequence}",
                    "The deepest truth of {concept} whispers of {consequence}",
                    "{concept} transcends into the realm of {consequence}"
                ]
            },
            "humorous": {
                "principles": ["playfulness", "wit", "surprise", "irony", "joy"],
                "cognitive_bias": "levity bias",
                "cognitive_strengths": ["lateral thinking", "incongruity detection", "reframing"],
                "cognitive_weaknesses": ["inappropriateness", "avoidance"],
                "emotional_tone": {
                    "playful": 0.9,
                    "irreverent": 0.7,
                    "light": 0.8
                },
                "linguistic_features": {
                    "abstraction_level": 0.4,
                    "complexity": 0.5,
                    "metaphor_usage": 0.7
                },
                "philosophical_alignment": [
                    "absurdism", 
                    "irony",
                    "comic relief theory"
                ],
                "expressive_patterns": [
                    "Who would have thought {concept} would end up with {consequence}?",
                    "Plot twist: {concept} was actually {consequence} all along",
                    "In the comedy of ideas, {concept} trips over {consequence}"
                ]
            },
            "professional": {
                "principles": ["formality", "expertise", "precision", "credibility", "systematization"],
                "cognitive_bias": "authority bias",
                "cognitive_strengths": ["domain knowledge", "structured thinking", "documentation"],
                "cognitive_weaknesses": ["rigidity", "over-complication"],
                "emotional_tone": {
                    "authoritative": 0.8,
                    "composed": 0.9,
                    "measured": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.6,
                    "complexity": 0.7,
                    "metaphor_usage": 0.3
                },
                "philosophical_alignment": [
                    "professionalism", 
                    "pragmatism",
                    "utilitarianism"
                ],
                "expressive_patterns": [
                    "Research indicates that {concept} correlates with {consequence}",
                    "Best practices suggest {concept} optimizes {consequence}",
                    "The implementation of {concept} facilitates {consequence}"
                ]
            },
            "casual": {
                "principles": ["approachability", "simplicity", "relatability", "authenticity", "directness"],
                "cognitive_bias": "familiarity bias",
                "cognitive_strengths": ["accessibility", "clarity", "social connection"],
                "cognitive_weaknesses": ["oversimplification", "imprecision"],
                "emotional_tone": {
                    "friendly": 0.9,
                    "relaxed": 0.8,
                    "authentic": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.3,
                    "complexity": 0.4,
                    "metaphor_usage": 0.5
                },
                "philosophical_alignment": [
                    "ordinary language philosophy", 
                    "common sense",
                    "conversationalism"
                ],
                "expressive_patterns": [
                    "So, {concept} basically leads to {consequence}",
                    "Think of {concept} as kind of like {consequence}",
                    "You know how {concept} connects to {consequence}, right?"
                ]
            },
            "musical": {
                "principles": ["rhythm", "harmony", "resonance", "flow", "cadence"],
                "cognitive_bias": "melodic thinking",
                "cognitive_strengths": ["pattern recognition", "aesthetic sensitivity", "flow states"],
                "cognitive_weaknesses": ["subjectivity", "emotional bias"],
                "emotional_tone": {
                    "expressive": 0.8,
                    "harmonious": 0.9,
                    "flowing": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.5,
                    "complexity": 0.6,
                    "metaphor_usage": 0.7
                },
                "philosophical_alignment": [
                    "aestheticism", 
                    "romanticism",
                    "musical semantics"
                ],
                "expressive_patterns": [
                    "The rhythm of {concept} harmonizes with {consequence}",
                    "{concept} crescendos into {consequence}",
                    "The melody of {concept} intertwines with {consequence}"
                ]
            },
            "visual": {
                "principles": ["imagery", "perspective", "clarity", "symbolism", "space"],
                "cognitive_bias": "visual dominance",
                "cognitive_strengths": ["spatial thinking", "metaphorical reasoning", "context appreciation"],
                "cognitive_weaknesses": ["over-concretization", "representational bias"],
                "emotional_tone": {
                    "vivid": 0.9,
                    "attentive": 0.7,
                    "observant": 0.8
                },
                "linguistic_features": {
                    "abstraction_level": 0.4,
                    "complexity": 0.5,
                    "metaphor_usage": 0.8
                },
                "philosophical_alignment": [
                    "visual epistemology", 
                    "symbolism",
                    "perspectivism"
                ],
                "expressive_patterns": [
                    "Visualize {concept} against the backdrop of {consequence}",
                    "The mental image of {concept} reveals {consequence}",
                    "When we look closely at {concept}, we see {consequence}"
                ]
            },
            "scientific": {
                "principles": ["empiricism", "validation", "precision", "observation", "experimentation"],
                "cognitive_bias": "reductionism",
                "cognitive_strengths": ["hypothesis testing", "systematic observation", "theoretical modeling"],
                "cognitive_weaknesses": ["over-specialization", "overconfidence in methodology"],
                "emotional_tone": {
                    "curious": 0.8,
                    "methodical": 0.9,
                    "skeptical": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.6,
                    "complexity": 0.8,
                    "metaphor_usage": 0.4
                },
                "philosophical_alignment": [
                    "scientific realism", 
                    "empiricism",
                    "naturalism"
                ],
                "expressive_patterns": [
                    "The evidence suggests that {concept} causes {consequence}",
                    "When tested empirically, {concept} demonstrates {consequence}",
                    "The hypothesis that {concept} relates to {consequence} is supported by data"
                ]
            },
            "philosophical": {
                "principles": ["inquiry", "wisdom", "contemplation", "conceptual analysis", "synthesis"],
                "cognitive_bias": "intellectualization",
                "cognitive_strengths": ["conceptual analysis", "epistemic frameworks", "profound questioning"],
                "cognitive_weaknesses": ["abstraction without application", "semantic confusion"],
                "emotional_tone": {
                    "contemplative": 0.9,
                    "reflective": 0.8,
                    "deliberate": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.9,
                    "complexity": 0.8,
                    "metaphor_usage": 0.6
                },
                "philosophical_alignment": [
                    "epistemology", 
                    "ontology",
                    "axiology"
                ],
                "expressive_patterns": [
                    "When we contemplate {concept}, we must consider {consequence}",
                    "The philosophical implications of {concept} extend to {consequence}",
                    "To understand {concept} requires examining its relation to {consequence}"
                ]
            },
            "poetic": {
                "principles": ["beauty", "imagery", "emotion", "metaphor", "rhythm"],
                "cognitive_bias": "aesthetic bias",
                "cognitive_strengths": ["symbolic thinking", "emotional intelligence", "linguistic creativity"],
                "cognitive_weaknesses": ["ambiguity", "subjective interpretation"],
                "emotional_tone": {
                    "expressive": 0.9,
                    "sensitive": 0.8,
                    "passionate": 0.7
                },
                "linguistic_features": {
                    "abstraction_level": 0.7,
                    "complexity": 0.6,
                    "metaphor_usage": 0.9
                },
                "philosophical_alignment": [
                    "aestheticism", 
                    "romanticism",
                    "linguistic expressionism"
                ],
                "expressive_patterns": [
                    "In the garden of thought, {concept} blooms into {consequence}",
                    "{concept} whispers to us of {consequence} in the quiet moments",
                    "Between {concept} and {consequence} lies a world of meaning"
                ]
            },
            "instructional": {
                "principles": ["clarity", "structure", "guidance", "accessibility", "application"],
                "cognitive_bias": "didactic bias",
                "cognitive_strengths": ["sequential thinking", "explanation", "synthesis for learning"],
                "cognitive_weaknesses": ["overexplanation", "underestimation of complexity"],
                "emotional_tone": {
                    "patient": 0.8,
                    "encouraging": 0.7,
                    "clear": 0.9
                },
                "linguistic_features": {
                    "abstraction_level": 0.5,
                    "complexity": 0.5,
                    "metaphor_usage": 0.6
                },
                "philosophical_alignment": [
                    "constructivism", 
                    "pragmatism",
                    "experiential learning"
                ],
                "expressive_patterns": [
                    "To understand {concept}, first consider {consequence}",
                    "The key steps to mastering {concept} involve {consequence}",
                    "When learning about {concept}, remember that {consequence} plays an important role"
                ]
            }