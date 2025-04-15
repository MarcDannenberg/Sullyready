# sully_engine/codex.py
# 📚 Sully's Symbolic Codex (Knowledge Repository)

from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Union, Set

class SullyCodex:
    """
    Stores and organizes Sully's symbolic knowledge, concepts, and their relationships.
    Functions as both a lexicon and a semantic network of interconnected meanings.
    """

    def __init__(self):
        self.entries = {}
        self.terms = {}  # For word definitions
        self.associations = {}  # For tracking relationships between concepts

    def record(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Records new symbolic knowledge under a topic name.

        Args:
            topic: The symbolic topic or name
            data: Associated symbolic data or metadata
        """
        normalized_topic = topic.lower()
        self.entries[normalized_topic] = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create associations with existing concepts
        self._create_associations(normalized_topic, data)

    def _create_associations(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Creates semantic associations between concepts based on shared attributes.
        
        Args:
            topic: The topic to create associations for
            data: The data containing potential association points
        """
        # Extract potential keywords from the data
        keywords = set()
        for value in data.values():
            if isinstance(value, str):
                # Split text into words, filter out very short words
                words = [w.lower() for w in str(value).split() if len(w) > 3]
                keywords.update(words)
        
        # Look for matches in existing entries
        for existing_topic in self.entries:
            if existing_topic == topic:
                continue  # Skip self-association
                
            # Check for keyword overlap in topic name
            if any(keyword in existing_topic for keyword in keywords):
                self._add_association(topic, existing_topic, "keyword_match")
                
            # Check for keyword overlap in data values
            existing_data = self.entries[existing_topic]
            existing_keywords = set()
            for value in existing_data.values():
                if isinstance(value, str):
                    words = [w.lower() for w in str(value).split() if len(w) > 3]
                    existing_keywords.update(words)
                    
            common_keywords = keywords.intersection(existing_keywords)
            if common_keywords:
                self._add_association(topic, existing_topic, "shared_concepts", list(common_keywords))

    def _add_association(self, topic1: str, topic2: str, type_: str, details: Any = None) -> None:
        """
        Adds a bidirectional association between two topics.
        
        Args:
            topic1: First topic
            topic2: Second topic
            type_: Type of association (e.g., "keyword_match", "shared_concepts")
            details: Optional details about the association
        """
        if topic1 not in self.associations:
            self.associations[topic1] = {}
            
        if topic2 not in self.associations:
            self.associations[topic2] = {}
            
        # Add bidirectional association
        self.associations[topic1][topic2] = {"type": type_, "details": details}
        self.associations[topic2][topic1] = {"type": type_, "details": details}

    def add_word(self, term: str, meaning: str) -> None:
        """
        Adds a new word definition to Sully's vocabulary.
        
        Args:
            term: The word or concept to define
            meaning: The definition or meaning of the term
        """
        normalized_term = term.lower()
        self.terms[normalized_term] = {
            "meaning": meaning,
            "created": datetime.now().isoformat(),
            "contexts": []  # Tracks different contexts where the term appears
        }
        
        # Also add to entries for searchability
        self.record(normalized_term, {
            "type": "term",
            "definition": meaning
        })

    def add_context(self, term: str, context: str) -> None:
        """
        Adds a usage context for a term to enrich its understanding.
        
        Args:
            term: The term to add context for
            context: A sample sentence or context where the term is used
        """
        normalized_term = term.lower()
        if normalized_term in self.terms:
            self.terms[normalized_term]["contexts"].append(context)
            # Update the timestamp
            self.terms[normalized_term]["updated"] = datetime.now().isoformat()

    def search(self, phrase: str, case_sensitive: bool = False, semantic: bool = True) -> Dict[str, Any]:
        """
        Searches the codex for entries matching a phrase, with optional
        semantic expansion to related concepts.

        Args:
            phrase: The search keyword
            case_sensitive: Match case when scanning
            semantic: Whether to include semantically related results

        Returns:
            Dictionary of matching entries (topic -> data)
        """
        results = {}
        phrase_check = phrase if case_sensitive else phrase.lower()

        # Direct matches in entries
        for topic, data in self.entries.items():
            topic_check = topic if case_sensitive else topic.lower()
            values = [str(v) for v in data.values() if v is not None]

            if phrase_check in topic_check or any(phrase_check in (v.lower() if not case_sensitive else v) for v in values):
                results[topic] = data

        # Search in term definitions
        for term, data in self.terms.items():
            term_check = term if case_sensitive else term.lower()
            meaning = data.get("meaning", "")
            meaning_check = meaning if case_sensitive else meaning.lower()
            
            if phrase_check in term_check or phrase_check in meaning_check:
                if term not in results:  # Avoid duplication with entries
                    results[term] = {
                        "type": "term",
                        "definition": meaning,
                        "contexts": data.get("contexts", [])
                    }

        # Expand to semantically related topics if requested
        if semantic and results:
            semantic_results = {}
            for topic in list(results.keys()):
                if topic in self.associations:
                    for related_topic, relation in self.associations[topic].items():
                        if related_topic not in results:
                            semantic_results[related_topic] = {
                                **self.entries.get(related_topic, {}),
                                "related_to": topic,
                                "relation": relation
                            }
            
            # Add semantic results with a note about their relationship
            results.update(semantic_results)

        return results

    def get(self, topic: str) -> Dict[str, Any]:
        """
        Gets a codex entry by topic name.
        
        Args:
            topic: The topic name to retrieve
            
        Returns:
            The entry data or a message if not found
        """
        normalized_topic = topic.lower()
        
        # Check entries first
        entry = self.entries.get(normalized_topic)
        if entry:
            # If it exists in entries, also check for associations
            result = dict(entry)
            if normalized_topic in self.associations:
                result["associations"] = {
                    related: info for related, info in self.associations[normalized_topic].items()
                }
            return result
            
        # Then check terms
        term_data = self.terms.get(normalized_topic)
        if term_data:
            return {
                "type": "term",
                "definition": term_data.get("meaning", ""),
                "contexts": term_data.get("contexts", [])
            }
            
        return {"message": "🔍 No codex entry found."}

    def list_topics(self) -> List[str]:
        """
        Returns a list of all topic names currently in the codex.
        
        Returns:
            List of topic names
        """
        # Combine entries and terms (avoiding duplicates)
        all_topics = set(self.entries.keys())
        all_topics.update(self.terms.keys())
        return sorted(list(all_topics))

    def get_related_concepts(self, topic: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Gets concepts related to a given topic up to a specified depth of relationships.
        
        Args:
            topic: The topic to find related concepts for
            max_depth: How many relationship steps to traverse
            
        Returns:
            Dictionary of related concepts with their relationship paths
        """
        normalized_topic = topic.lower()
        if normalized_topic not in self.associations:
            return {}
            
        # Start with direct associations
        related = {
            related_topic: {"path": [normalized_topic], "relation": info}
            for related_topic, info in self.associations[normalized_topic].items()
        }
        
        # For depth > 1, traverse the graph further
        if max_depth > 1:
            current_level = list(related.keys())
            for depth in range(1, max_depth):
                next_level = []
                for current_topic in current_level:
                    if current_topic in self.associations:
                        for related_topic, info in self.associations[current_topic].items():
                            # Skip if already encountered to prevent cycles
                            if related_topic not in related and related_topic != normalized_topic:
                                path = related[current_topic]["path"] + [current_topic]
                                related[related_topic] = {
                                    "path": path,
                                    "relation": info,
                                    "depth": depth + 1
                                }
                                next_level.append(related_topic)
                current_level = next_level
                if not current_level:
                    break  # No more connections to explore
                    
        return related

    def export(self) -> Dict[str, Any]:
        """
        Returns all codex entries for backup, JSON export, or UI rendering.
        
        Returns:
            Dictionary containing all codex data
        """
        return {
            "entries": self.entries,
            "terms": self.terms,
            "associations": self.associations
        }

    def import_data(self, data: Dict[str, Any]) -> None:
        """
        Imports codex data from a previously exported format.
        
        Args:
            data: Dictionary containing codex data (entries, terms, associations)
        """
        if "entries" in data:
            self.entries.update(data["entries"])
        if "terms" in data:
            self.terms.update(data["terms"])
        if "associations" in data:
            self.associations.update(data["associations"])

    def __len__(self) -> int:
        """
        Returns the total number of unique concepts in the codex.
        
        Returns:
            Count of unique concepts (entries + terms)
        """
        # Get unique set of all concepts (terms might overlap with entries)
        all_concepts = set(self.entries.keys())
        all_concepts.update(self.terms.keys())
        return len(all_concepts)
        
    def batch_process(self, text: str) -> List[Dict[str, Any]]:
        """
        Processes a text to extract and record potential concepts and their relationships.
        
        Args:
            text: Text to analyze for concepts
            
        Returns:
            List of newly identified and recorded concepts
        """
        # This would typically use NLP to extract entities and concepts
        # For now, we'll implement a simple approach
        import re
        from collections import Counter
        
        # Extract potential concept phrases (sequences of 1-3 words)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        if not words:
            return []
            
        # Count word frequencies to identify important terms
        word_counts = Counter(words)
        important_words = [word for word, count in word_counts.most_common(10) if count > 1]
        
        # Record these as potential concepts
        new_concepts = []
        for word in important_words:
            # Extract a context for this word
            context_match = re.search(r'[^.!?]*\b' + re.escape(word) + r'\b[^.!?]*[.!?]', text)
            context = context_match.group(0).strip() if context_match else ""
            
            # Create a basic definition based on context
            definition = f"Concept extracted from text context: '{context}'"
            
            # Record in codex
            self.add_word(word, definition)
            if context:
                self.add_context(word, context)
                
            new_concepts.append({
                "term": word,
                "definition": definition,
                "context": context
            })
            
        return new_concepts