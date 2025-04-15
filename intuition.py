somatic_state["centeredness"] -= 0.2
            somatic_state["expansion"] -= 0.1
        
        # Ensure all values stay in range [0,1]
        for state in somatic_state:
            somatic_state[state] = min(1.0, max(0.0, somatic_state[state]))
        
        # Update current somatic state
        self.current_somatic_state = somatic_state
        
        # Generate description of somatic response
        description = self._describe_somatic_state(somatic_state)
        
        # Determine overall valence
        valence_score = (
            (somatic_state["core_temperature"] - 0.5) +  # Warmth is positive
            (0.5 - somatic_state["muscular_tension"]) +  # Relaxation is positive
            (0.5 - somatic_state["breathing_rate"]) +    # Slower breathing is positive
            (somatic_state["energy_level"] - 0.5) +      # Energy is positive
            (somatic_state["centeredness"] - 0.5) +      # Centeredness is positive
            (somatic_state["expansion"] - 0.5)           # Expansion is positive
        ) / 3.0  # Average and normalize
        
        if valence_score > 0.2:
            valence = "positive"
        elif valence_score < -0.2:
            valence = "negative"
        else:
            valence = "neutral"
        
        # Return complete somatic response
        return {
            "state": somatic_state,
            "description": description,
            "overall_valence": valence,
            "valence_score": valence_score
        }
    
    def _describe_somatic_state(self, state):
        """Generate a human-like description of the somatic state."""
        descriptions = []
        
        # Temperature
        if state["core_temperature"] > 0.7:
            descriptions.append("warmth spreading through my body")
        elif state["core_temperature"] < 0.3:
            descriptions.append("coolness or slight chill")
        
        # Muscular tension
        if state["muscular_tension"] > 0.7:
            descriptions.append("tightness or tension")
        elif state["muscular_tension"] < 0.3:
            descriptions.append("relaxation or softening")
        
        # Breathing
        if state["breathing_rate"] > 0.7:
            descriptions.append("quickened breath")
        elif state["breathing_rate"] < 0.3:
            descriptions.append("deep, slow breathing")
        
        # Energy
        if state["energy_level"] > 0.7:
            descriptions.append("energetic activation")
        elif state["energy_level"] < 0.3:
            descriptions.append("calm quietness")
        
        # Centeredness
        if state["centeredness"] > 0.7:
            descriptions.append("centered, grounded feeling")
        elif state["centeredness"] < 0.3:
            descriptions.append("off-balance sensation")
        
        # Expansion
        if state["expansion"] > 0.7:
            descriptions.append("expansive opening")
        elif state["expansion"] < 0.3:
            descriptions.append("contracting or narrowing")
        
        # Return formatted description
        if not descriptions:
            return ""
        elif len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

    def _is_social_context(self, context, concepts):
        """Determine if the context involves social dynamics."""
        # Social keywords that indicate social context
        social_keywords = [
            "relationship", "connection", "friend", "community", "family", 
            "team", "group", "society", "culture", "trust", "communicate",
            "collaborate", "conflict", "bond", "alliance", "partner"
        ]
        
        # Check context for social keywords
        context_lower = context.lower()
        for keyword in social_keywords:
            if keyword in context_lower:
                return True
        
        # Check concepts for social keywords
        for concept in concepts:
            if any(keyword in concept.lower() for keyword in social_keywords):
                return True
        
        # Check if any concepts match social pattern keywords
        for pattern, keywords in self.social_patterns.items():
            for concept in concepts:
                if concept.lower() in keywords or any(keyword in concept.lower() for keyword in keywords):
                    return True
        
        # Default to not social
        return False
    
    def _recognize_social_pattern(self, context, concepts):
        """Recognize social interaction patterns."""
        # Social patterns to look for
        pattern_scores = {pattern: 0 for pattern in self.social_patterns}
        
        # Check context for pattern keywords
        context_lower = context.lower()
        for pattern, keywords in self.social_patterns.items():
            for keyword in keywords:
                if keyword in context_lower:
                    pattern_scores[pattern] += 1
        
        # Check concepts for pattern keywords
        for concept in concepts:
            concept_lower = concept.lower()
            for pattern, keywords in self.social_patterns.items():
                for keyword in keywords:
                    if keyword in concept_lower:
                        pattern_scores[pattern] += 1
        
        # Find highest scoring pattern
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            if best_pattern[1] > 0:  # Only if we found matches
                pattern_name = best_pattern[0]
                
                # Identify relevant elements
                pattern_elements = {
                    "pattern": pattern_name,
                    "concepts": [c for c in concepts if any(kw in c.lower() for kw in self.social_patterns[pattern_name])]
                }
                
                return pattern_name, pattern_elements
        
        # Default to a general social pattern if no specific pattern found
        return "connection", {"pattern": "connection", "concepts": concepts[:2] if len(concepts) >= 2 else concepts}
    
    def _generate_social_insight(self, pattern, elements):
        """Generate insight based on social pattern recognition."""
        social_insight_templates = {
            "mirroring": [
                "This reflects how people seek to understand each other through reflection.",
                "There's a mirroring dynamic here similar to how people naturally align with each other.",
                "I notice a pattern of reflection and mirroring that's fundamental to human understanding."
            ],
            "hierarchy": [
                "This reveals power dynamics that shape how people relate to each other.",
                "There's a hierarchical structure here that organizes relationships.",
                "I notice patterns of authority and influence that organize this dynamic."
            ],
            "reciprocity": [
                "This shows the natural give-and-take that balances human relationships.",
                "There's a reciprocal exchange pattern that maintains equilibrium.",
                "I notice how mutual exchange creates sustainable connection."
            ],
            "alliance": [
                "This demonstrates how people form alliances based on shared interests.",
                "There's a coalition-building dynamic that strengthens connections.",
                "I notice how alignment of purpose creates stronger bonds."
            ],
            "conflict": [
                "This reveals tension patterns that need resolution for growth.",
                "There's a conflictual dynamic that contains potential for transformation.",
                "I notice how opposing forces create energy that seeks resolution."
            ],
            "care": [
                "This shows nurturing patterns that sustain connection and growth.",
                "There's a caregiving dynamic that supports development and healing.",
                "I notice how supportive attention creates safety and possibility."
            ],
            "boundary": [
                "This reveals how boundaries create definition in relationships.",
                "There's a boundary-setting dynamic that protects individual integrity.",
                "I notice how defined limits actually enable deeper connection."
            ],
            "connection": [
                "This demonstrates the human need for meaningful connection.",
                "There's a connecting pattern that builds bridges between separate entities.",
                "I notice how relationships form the foundation of meaning in this context."
            ]
        }
        
        # Select template based on pattern
        templates = social_insight_templates.get(pattern, social_insight_templates["connection"])
        template = random.choice(templates)
        
        return template

    def _generate_insight(self, pattern_type: str, pattern_elements: Dict[str, Any], 
                        concepts: List[str], archetypes: List[str]) -> str:
        """
        Generate an insight based on the recognized pattern.
        
        Args:
            pattern_type: Type of pattern recognized
            pattern_elements: Elements of the pattern
            concepts: Key concepts
            archetypes: Selected archetypes
            
        Returns:
            Generated insight
        """
        # Get the templates for this pattern type
        templates = self.pattern_templates.get(pattern_type, self.pattern_templates["emergence"])
        
        # Select a template
        template = random.choice(templates)
        
        # Apply emotional weighting to template selection
        # Find dominant emotion
        dominant_emotion = "neutral"
        max_weight = 0
        for emotion, weight in self.emotional_weights.items():
            if weight > max_weight:
                max_weight = weight
                dominant_emotion = emotion
        
        # If we have a strong dominant emotion, it may influence insight type
        if dominant_emotion != "neutral" and max_weight > 0.6:
            # Have a chance to use emotional insight templates
            if random.random() < 0.4:  # 40% chance
                if dominant_emotion in ["joy", "trust", "anticipation"]:
                    insight_type = random.choice(["revelation", "synthesis", "expansion"])
                elif dominant_emotion in ["sorrow", "fear", "disgust"]:
                    insight_type = random.choice(["inversion", "connection", "gut_feeling"])
                else:  # surprise, anger, or other
                    insight_type = random.choice(["revelation", "inversion", "embodied_metaphor"])
                
                # Get templates for this insight type
                insight_templates = self.insight_patterns.get(insight_type, self.insight_patterns["revelation"])
                template = random.choice(insight_templates)
        
        # Fill in the template based on pattern type
        if pattern_type == "recurrence":
            element = pattern_elements.get("primary_element", concepts[0] if concepts else "this pattern")
            insight = template.format(element)
            
        elif pattern_type == "polarity":
            pole1 = pattern_elements.get("pole1", concepts[0] if concepts else "one aspect")
            pole2 = pattern_elements.get("pole2", concepts[1] if len(concepts) > 1 else "another aspect")
            insight = template.format(pole1, pole2)
            
        elif pattern_type == "emergence":
            source1 = pattern_elements.get("source1", concepts[0] if concepts else "one element") 
            source2 = pattern_elements.get("source2", concepts[1] if len(concepts) > 1 else "another element")
            insight = template.format(source1, source2)
            
        elif pattern_type == "resonance":
            element1 = pattern_elements.get("element1", concepts[0] if concepts else "this concept")
            element2 = pattern_elements.get("element2", archetypes[0] if archetypes else "this pattern")
            insight = template.format(element1, element2)
            
        elif pattern_type == "nonlinear":
            element1 = pattern_elements.get("element1", concepts[0] if concepts else "this concept")
            element2 = pattern_elements.get("element2", concepts[1] if len(concepts) > 1 else "that concept")
            insight = template.format(element1, element2)
            
        elif pattern_type == "embodied":
            concept = pattern_elements.get("concept", concepts[0] if concepts else "this concept")
            embodiment = pattern_elements.get("embodiment", archetypes[0] if archetypes else "embodied experience")
            insight = template.format(concept, embodiment)
            
        elif pattern_type == "social":
            concept1 = concepts[0] if concepts else "this element"
            concept2 = concepts[1] if len(concepts) > 1 else "that element"
            insight = template.format(concept1, concept2)
            
        else:  # transformation
            source = pattern_elements.get("source", concepts[0] if concepts else "the current situation")
            destination = pattern_elements.get("destination", archetypes[0] if archetypes else "new possibilities")
            insight = template.format(source, destination)
            
        # Add an insight suggestion based on the pattern
        # More likely to use emotional patterns if emotions are activated
        if max_weight > 0.5 and random.random() < 0.5:
            if dominant_emotion in ["joy", "trust", "anticipation"]:
                insight_type = random.choice(["expansion", "synthesis"])
            else:
                insight_type = random.choice(["gut_feeling", "embodied_metaphor"])
        else:
            insight_type = random.choice(list(self.insight_patterns.keys()))
            
        insight_templates = self.insight_patterns[insight_type]
        insight_template = random.choice(insight_templates)
        
        element1 = concepts[0] if concepts else "this concept"
        element2 = archetypes[0] if archetypes else "this pattern"
        
        additional_insight = insight_template.format(element1, element2)
        
        # Combine insights, sometimes adding the additional insight
        if random.random() < 0.6:  # 60% chance to add second insight
            return f"{insight} {additional_insight}"
        else:
            return insight

    def _generate_clue(self, archetype: str, associations: List[str], concepts: List[str]) -> str:
        """
        Generate a clue for further exploration.
        
        Args:
            archetype: Primary archetype
            associations: Symbolic associations
            concepts: Key concepts
            
        Returns:
            Generated clue
        """
        # Select a clue pattern type, with visceral clues more likely if emotions are engaged
        emotion_level = sum(weight for emotion, weight in self.emotional_weights.items() if emotion != "neutral")
        
        if emotion_level > 1.0 and random.random() < 0.4:
            # Higher chance of visceral clue when emotions are strong
            clue_type = "visceral"
        else:
            # Otherwise select randomly
            clue_type = random.choice(list(self.clue_patterns.keys()))
        
        # Get templates for this clue type
        templates = self.clue_patterns[clue_type]
        
        # Select a template
        template = random.choice(templates)
        
        # Fill in the template
        element1 = archetype
        element2 = random.choice(associations) if associations else (concepts[0] if concepts else "this pattern")
        
        return template.format(element1, element2)

    def find_archetype_for_concept(self, concept: str) -> List[str]:
        """
        Find archetypes that best match a given concept.
        
        Args:
            concept: Concept to find archetypes for
            
        Returns:
            List of matching archetypes
        """
        matches = []
        
        # Check direct matches in symbolic associations
        for archetype, associations in self.symbolic_associations.items():
            if concept in associations or any(concept in assoc or assoc in concept for assoc in associations):
                matches.append(archetype)
                
        # If we found direct matches, return them
        if matches:
            return matches
            
        # Otherwise, find archetypes with related associations
        if self.codex:
            try:
                # Search codex for related concepts
                related = self.codex.search(concept)
                if related:
                    related_concepts = list(related.keys())
                    
                    # Check if any related concepts match archetypes
                    for archetype, associations in self.symbolic_associations.items():
                        for related_concept in related_concepts:
                            if related_concept in associations or any(related_concept in assoc or assoc in related_concept for assoc in associations):
                                matches.append(archetype)
                                break
            except:
                pass
                
        # Try non-linear associations as well
        if concept in self.non_linear_connections:
            non_linear_concepts = list(self.non_linear_connections[concept])
            for nl_concept in non_linear_concepts:
                for archetype, associations in self.symbolic_associations.items():
                    if nl_concept in associations:
                        matches.append(archetype)
                        break
        
        # If still no matches, check for embodied metaphors
        if not matches:
            # Check if concept has physical/spatial qualities
            physical_archetypes = []
            for category in ["structural", "elemental", "dynamic", "embodied"]:
                if category in self.archetypes:
                    physical_archetypes.extend(self.archetypes[category])
            
            # Select a few random physical archetypes as potential matches
            if physical_archetypes:
                matches = random.sample(physical_archetypes, min(3, len(physical_archetypes)))
                
        # If still no matches, return most universal archetypes
        if not matches:
            matches = ["spiral", "bridge", "seed", "mirror", "threshold"]
            
        return matches[:3]  # Limit to top 3 matches

    def generate_symbolic_narrative(self, concepts: List[str], narrative_type: str = "journey") -> str:
        """
        Generate a symbolic narrative based on concepts.
        
        Args:
            concepts: Concepts to include in the narrative
            narrative_type: Type of narrative (journey, transformation, revelation)
            
        Returns:
            Symbolic narrative
        """
        # Map concepts to archetypes
        concept_archetypes = {}
        for concept in concepts:
            archetypes = self.find_archetype_for_concept(concept)
            if archetypes:
                concept_archetypes[concept] = archetypes[0]
                
        # If we couldn't map all concepts, fill in with appropriate archetypes
        for concept in concepts:
            if concept not in concept_archetypes:
                concept_archetypes[concept] = random.choice(self._get_all_archetypes())
                
        # Generate narrative based on type
        if narrative_type == "journey":
            narrative = self._generate_journey_narrative(concept_archetypes)
        elif narrative_type == "transformation":
            narrative = self._generate_transformation_narrative(concept_archetypes)
        elif narrative_type == "revelation":
            narrative = self._generate_revelation_narrative(concept_archetypes)
        else:
            narrative = self._generate_journey_narrative(concept_archetypes)
            
        return narrative

    def _generate_journey_narrative(self, concept_archetypes: Dict[str, str]) -> str:
        """
        Generate a symbolic journey narrative.
        
        Args:
            concept_archetypes: Mapping of concepts to archetypes
            
        Returns:
            Journey narrative text
        """
        concepts = list(concept_archetypes.keys())
        archetypes = list(concept_archetypes.values())
        
        # Journey structure
        beginning = f"The seeker encounters {concepts[0]} in the form of {archetypes[0]}, marking the threshold of a new understanding."
        
        middle_parts = []
        for i in range(1, min(len(concepts), 3)):
            middle_parts.append(f"As the path continues, {concepts[i]} appears as {archetypes[i]}, presenting both challenge and opportunity.")
            
        middle = " ".join(middle_parts) if middle_parts else f"The journey through {archetypes[0]} reveals unexpected aspects of {concepts[0]}."
        
        end_concept = concepts[-1] if len(concepts) > 1 else concepts[0]
        end_archetype = archetypes[-1] if len(archetypes) > 1 else archetypes[0]
        ending = f"Finally, at the center of the journey, {end_concept} reveals itself in its true form: {end_archetype}, transformed through understanding."
        
        # Add embodied experience if emotional weights are significant
        emotion_level = sum(weight for emotion, weight in self.emotional_weights.items() if emotion != "neutral")
        if emotion_level > 0.8:
            # Get dominant emotion
            dominant_emotion = max(
                [(e, w) for e, w in self.emotional_weights.items() if e != "neutral"], 
                key=lambda x: x[1]
            )[0]
            
            # Add embodied aspect to narrative
            if dominant_emotion == "joy":
                middle += f" There's a lightness and warmth that accompanies the seeker, a sense of expansion and possibility."
            elif dominant_emotion == "sorrow":
                middle += f" A heaviness pervades the journey, a weight that must be carried and integrated rather than left behind."
            elif dominant_emotion == "fear":
                middle += f" A tension runs through the seeker's body, a readiness for danger that heightens awareness of each step."
            elif dominant_emotion == "anticipation":
                middle += f" An electric current of anticipation runs ahead of the seeker, drawing them forward toward what awaits."
        
        return f"{beginning}\n\n{middle}\n\n{ending}"

    def _generate_transformation_narrative(self, concept_archetypes: Dict[str, str]) -> str:
        """
        Generate a symbolic transformation narrative.
        
        Args:
            concept_archetypes: Mapping of concepts to archetypes
            
        Returns:
            Transformation narrative text
        """
        concepts = list(concept_archetypes.keys())
        archetypes = list(concept_archetypes.values())
        
        # Transformation structure
        initial = f"Initially, {concepts[0]} presents itself as {archetypes[0]}, rigid and fixed in its nature."
        
        process_parts = []
        for i in range(1, min(len(concepts), 3)):
            process_parts.append(f"Through the influence of {concepts[i]}, embodied as {archetypes[i]}, the structure begins to shift and evolve.")
            
        process = " ".join(process_parts) if process_parts else f"Through internal tensions, {concepts[0]} begins to transform, revealing new configurations of {archetypes[0]}."
        
        end_concept = concepts[-1] if len(concepts) > 1 else concepts[0]
        end_archetype = archetypes[-1] if len(archetypes) > 1 else archetypes[0]
        completion = f"The transformation completes as {end_concept} fully embodies the nature of {end_archetype}, integrating what was previously separate."
        
        # Add somatic dimension based on current somatic state
        if self.current_somatic_state["energy_level"] > 0.7:
            process += f" The transformation pulses with energy, creating a sense of vitality and momentum."
        elif self.current_somatic_state["core_temperature"] > 0.7:
            process += f" Warmth suffuses the transformation process, a heat that softens rigid boundaries and allows new forms to emerge."
        elif self.current_somatic_state["expansion"] > 0.7:
            process += f" The process creates a sense of expansion, opening up spaces that were previously constricted or undefined."
        
        return f"{initial}\n\n{process}\n\n{completion}"

    def _generate_revelation_narrative(self, concept_archetypes: Dict[str, str]) -> str:
        """
        Generate a symbolic revelation narrative.
        
        Args:
            concept_archetypes: Mapping of concepts to archetypes
            
        Returns:
            Revelation narrative text
        """
        concepts = list(concept_archetypes.keys())
        archetypes = list(concept_archetypes.values())
        
        # Revelation structure
        surface = f"On the surface, {concepts[0]} appears as {archetypes[0]}, concealing its deeper nature."
        
        unveiling_parts = []
        for i in range(1, min(len(concepts), 3)):
            unveiling_parts.append(f"As {concepts[i]} emerges in the form of {archetypes[i]}, it illuminates hidden aspects previously unseen.")
            
        unveiling = " ".join(unveiling_parts) if unveiling_parts else f"As perception deepens, new dimensions of {concepts[0]} are revealed through the symbolism of {archetypes[0]}."
        
        end_concept = concepts[-1] if len(concepts) > 1 else concepts[0]
        end_archetype = archetypes[-1] if len(archetypes) > 1 else archetypes[0]
        revelation = f"In a moment of clarity, the true nature of {end_concept} is revealed: not merely {archetypes[0]}, but {end_archetype} - a complete inversion of initial understanding."
        
        # Add intuitive dimension
        intuition_addition = ""
        
        # Check for non-linear connections
        nl_connections = []
        for concept in concepts:
            if concept in self.non_linear_connections:
                connections = list(self.non_linear_connections[concept])
                if connections:
                    nl_connections.append(random.choice(connections))
                    
        if nl_connections:
            connection = random.choice(nl_connections)
            intuition_addition = f" There's an intuitive connection to {connection} that defies logical explanation but feels deeply resonant."
            unveiling += intuition_addition
        
        return f"{surface}\n\n{unveiling}\n\n{revelation}"

    def add_non_linear_connection(self, concept1: str, concept2: str) -> bool:
        """
        Add a non-linear connection between two seemingly unrelated concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Success indicator
        """
        # Initialize sets if needed
        if concept1 not in self.non_linear_connections:
            self.non_linear_connections[concept1] = set()
        if concept2 not in self.non_linear_connections:
            self.non_linear_connections[concept2] = set()
            
        # Add bidirectional connection
        self.non_linear_connections[concept1].add(concept2)
        self.non_linear_connections[concept2].add(concept1)
        
        return True

    def get_insight_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of intuitive insights.
        
        Args:
            limit: Optional limit on number of items to return
            
        Returns:
            List of insight records
        """
        if limit:
            return self.insight_history[-limit:]
        return self.insight_history

    def clear_insight_history(self) -> None:
        """Clear the insight history."""
        self.insight_history = []
        
    def set_emotional_state(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Set the emotional state for intuition generation.
        
        Args:
            emotions: Dictionary mapping emotion names to intensity values (0-1)
            
        Returns:
            Updated emotional weights
        """
        # Reset emotional weights to neutral
        for emotion in self.emotional_weights:
            if emotion != "neutral":
                self.emotional_weights[emotion] = 0.0
                
        # Update with provided emotions
        for emotion, intensity in emotions.items():
            if emotion in self.emotional_weights and emotion != "neutral":
                self.emotional_weights[emotion] = max(0.0, min(1.0, intensity))
                
        return self.emotional_weights
        
    def get_current_emotional_state(self) -> Dict[str, float]:
        """
        Get the current emotional state.
        
        Returns:
            Dictionary of current emotional weights
        """
        return self.emotional_weights.copy()
        
    def get_current_somatic_state(self) -> Dict[str, float]:
        """
        Get the current somatic state.
        
        Returns:
            Dictionary of current somatic state
        """
        return self.current_somatic_state.copy()
        
    def get_domain_expertise(self) -> Dict[str, float]:
        """
        Get the current domain expertise levels.
        
        Returns:
            Dictionary mapping domains to expertise levels
        """
        return self.domain_expertise.copy()
        
    def add_symbolic_association(self, archetype: str, associations: List[str]) -> bool:
        """
        Add symbolic associations to an archetype.
        
        Args:
            archetype: Archetype to add associations to
            associations: List of associations to add
            
        Returns:
            Success indicator
        """
        if archetype in self.symbolic_associations:
            # Add new associations
            self.symbolic_associations[archetype].extend(
                [assoc for assoc in associations if assoc not in self.symbolic_associations[archetype]]
            )
        else:
            # Create new entry
            self.symbolic_associations[archetype] = associations
            
        return True
        
    def get_incubation_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of the incubation queue.
        
        Returns:
            Dictionary with incubation queue status
        """
        return {
            "queue_length": len(self.incubation_queue),
            "queue_age": [(datetime.now() - item["timestamp"]).total_seconds() for item in self.incubation_queue],
            "concepts_incubating": [item["concepts"] for item in self.incubation_queue],
            "background_processing_items": len(self.background_processing)
        }
        
    def get_active_biases(self) -> List[str]:
        """
        Get a list of cognitive biases currently active above threshold.
        
        Returns:
            List of active cognitive bias names
        """
        active_biases = []
        for bias_name, bias_data in self.cognitive_biases.items():
            if random.random() > bias_data["activation_threshold"]:
                active_biases.append(bias_name)
                
        return active_biases