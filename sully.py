"""
Sully - An advanced cognitive system with integrated memory and enhanced capabilities.
"""
# --- Imports ---
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from sully_engine.kernel_integration import initialize_kernel_integration, KernelIntegrationSystem


# Core modules
from sully_engine.kernel_modules.identity import SullyIdentity
from sully_engine.kernel_modules.codex import SullyCodex
from sully_engine.reasoning import SymbolicReasoningNode
from sully_engine.memory import SullySearchMemory

# Memory Integration
from sully_engine.memory_integration import MemoryIntegration
from sully_engine.memory_integration import integrate_with_sully

# Logic Kernel
from sully_engine.logic_kernel import LogicKernel, integrate_with_sully as integrate_logic

# Kernel Modules
from sully_engine.kernel_modules.judgment import JudgmentProtocol
from sully_engine.kernel_modules.dream import DreamCore
from sully_engine.kernel_modules.math_translator import SymbolicMathTranslator
from sully_engine.kernel_modules.fusion import SymbolFusionEngine
from sully_engine.kernel_modules.paradox import ParadoxLibrary
from sully_engine.kernel_modules.neural_modification import NeuralModification
from sully_engine.kernel_modules.continuous_learning import ContinuousLearningSystem
from sully_engine.kernel_modules.autonomous_goals import AutonomousGoalSystem
from sully_engine.kernel_modules.visual_cognition import VisualCognitionSystem
from sully_engine.kernel_modules.emergence_framework import EmergenceFramework
from sully_engine.kernel_modules.virtue import VirtueEngine
from sully_engine.kernel_modules.intuition import Intuition

# Import PDF reader
from sully_engine.pdf_reader import PDFReader

# Configuration constants
MEMORY_PATH = "sully_memory_store.json"
MEMORY_INTEGRATION_ENABLED = True


class Sully:
    """
    Sully: An advanced cognitive framework capable of synthesizing knowledge from various sources,
    with enhanced integrated memory and expressing it through multiple cognitive modes.
    """

    def __init__(self, memory_path: Optional[str] = None):
        """Initialize Sully's cognitive systems with integrated memory."""
        # Core cognitive architecture
        # Initialize reasoning node first since identity system depends on it
        self.memory = SullySearchMemory()
        self.codex = SullyCodex()
        self.reasoning_node = SymbolicReasoningNode(
            codex=self.codex,
            translator=None,  # Will set after initializing translator
            memory=self.memory
        )
        
        # Initialize identity system with connections to other modules
        self.identity = SullyIdentity(
            memory_system=self.memory,
            reasoning_engine=self.reasoning_node
        )
        
        # Specialized cognitive modules
        self.translator = SymbolicMathTranslator()
        self.judgment = JudgmentProtocol()
        self.dream = DreamCore()
        self.paradox = ParadoxLibrary()
        self.fusion = SymbolFusionEngine()
        
        # Now complete the reasoning node initialization
        self.reasoning_node.translator = self.translator
        
        # Advanced cognitive modules
        self.neural_modification = NeuralModification(
            reasoning_engine=self.reasoning_node,
            memory_system=self.memory
        )
        self.continuous_learning = ContinuousLearningSystem(
            memory_system=self.memory,
            codex=self.codex
        )
        self.autonomous_goals = AutonomousGoalSystem(
            memory_system=self.memory,
            learning_system=self.continuous_learning
        )
        self.visual_cognition = VisualCognitionSystem(
            codex=self.codex
        )
        self.logic_kernel = None
        
        # Initialize with proper connections between modules
        self.judgment = JudgmentProtocol(memory=self.memory, reasoning=self.reasoning_node)
        self.intuition = Intuition(memory=self.memory, reasoning=self.reasoning_node, codex=self.codex)
        self.virtue = VirtueEngine(judgment=self.judgment, memory=self.memory, logic_kernel=self.logic_kernel, reasoning=self.reasoning_node)
        
        # Complete the initialization of modules that need the reasoning node
        self.continuous_learning.reasoning = self.reasoning_node
        self.autonomous_goals.reasoning = self.reasoning_node
        self.visual_cognition.reasoning = self.reasoning_node
        
        # Initialize emergence framework with all cognitive modules
        self.emergence = EmergenceFramework(
            all_cognitive_modules={
                "reasoning_node": self.reasoning_node,
                "judgment": self.judgment,
                "intuition": self.intuition,
                "dream": self.dream,
                "fusion": self.fusion,
                "translator": self.translator,
                "paradox": self.paradox,
                "codex": self.codex,
                "memory": self.memory,
                "identity": self.identity,
                "neural_modifier": self.neural_modification,
                "learning_system": self.continuous_learning,
                "goal_system": self.autonomous_goals,
                "visual_system": self.visual_cognition,
                "virtue": self.virtue
            }
        )
        
        # PDF reader for direct document processing
        self.pdf_reader = PDFReader(ocr_enabled=True, dpi=300)
        
        # Experiential knowledge - unlimited and ever-growing
        self.knowledge = []
        
        # Initialize the conversation engine and connect to the core systems
        from sully_engine.conversation_engine import ConversationEngine
        self.conversation = ConversationEngine(
            reasoning_node=self.reasoning_node,
            memory_system=self.memory,
            codex=self.codex
        )
        
        # Initialize memory integration system if enabled
        self.memory_integration = None
        if MEMORY_INTEGRATION_ENABLED:
            memory_file = memory_path or MEMORY_PATH
            self.memory_integration = integrate_with_sully(self, memory_file)
            
       # Initialize kernel integration system
        self.kernel_integration = None
        try:
            self.kernel_integration = initialize_kernel_integration(
                codex=self.codex,
                dream_core=self.dream,
                fusion_engine=self.fusion,
                paradox_library=self.paradox,
                math_translator=self.translator,
                conversation_engine=self.conversation,
                memory_integration=self.memory_integration,
                sully_instance=self
            )
            print("Kernel integration system initialized and connected")
        except Exception as e:
            print(f"Warning: Kernel integration initialization failed: {str(e)}")
            
    def multi_perspective_evaluation(self, claim, context=None):
        """Evaluate a claim through multiple cognitive frameworks with integration."""
        if not hasattr(self.judgment, 'multi_perspective_evaluation'):
            return {"error": "Advanced evaluation not available"}
        return self.judgment.multi_perspective_evaluation(claim, context)

    def generate_intuitive_leap(self, context, concepts=None, depth="standard", domain=None):
        """Generate an intuitive leap based on context and concepts."""
        if not hasattr(self.intuition, 'leap'):
            return {"error": "Advanced intuition not available"}
        return self.intuition.leap(context, concepts, depth, domain)

    def evaluate_virtue(self, idea, context=None, domain=None):
        """Evaluate an idea through virtue ethics framework."""
        if not hasattr(self.virtue, 'evaluate'):
            return {"error": "Virtue evaluation not available"}
        return self.virtue.evaluate(idea, context, domain)

    def evaluate_action_virtue(self, action, context=None, domain=None):
        """Evaluate an action through virtue ethics framework."""
        if not hasattr(self.virtue, 'evaluate_action'):
            return {"error": "Virtue action evaluation not available"}
        return self.virtue.evaluate_action(action, context, domain)

    def reflect_on_virtue(self, virtue):
        """Generate meta-ethical reflection on a specific virtue."""
        if not hasattr(self.virtue, 'reflect_on_virtue'):
            return {"error": "Virtue reflection not available"}
        return self.virtue.reflect_on_virtue(virtue)

    def speak_identity(self):
        """Express Sully's sense of self."""
        return self.identity.speak_identity()

    def adapt_identity_to_context(self, context: str, context_data: dict = None) -> dict:
        """
        Dynamically adapt Sully's identity to match the given context.
        
        Args:
            context: Text describing the interaction context
            context_data: Optional structured context data
            
        Returns:
            Adaptation results
        """
        if not hasattr(self.identity, 'adapt_to_context'):
            return {"success": False, "message": "Enhanced identity adaptation not available"}
        
        result = self.identity.adapt_to_context(context, context_data)
        
        # Register this adaptation with memory if available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    f"Adapted identity to context: {context[:100]}...",
                    "identity_adaptation",
                    importance=0.7,
                    emotional_tags={"adaptability": 0.8},
                    concepts=self._extract_key_concepts(context[:500])
                )
            except Exception as e:
                print(f"Error storing identity adaptation in memory: {e}")
        
        return result

    def evolve_identity(self, interactions=None, learning_rate=0.05) -> dict:
        """
        Evolve Sully's personality traits based on interactions and feedback.
        
        Args:
            interactions: Optional list of recent interactions to analyze
            learning_rate: Rate of personality adaptation (0.0 to 1.0)
            
        Returns:
            Evolution results with changes applied
        """
        if not hasattr(self.identity, 'evolve_personality'):
            return {"success": False, "message": "Enhanced identity evolution not available"}
        
        # If no interactions provided and memory available, try to retrieve
        if not interactions and self.memory_integration:
            try:
                interactions = self.memory_integration.recall(
                    query="recent interactions",
                    limit=20,
                    module="conversation"
                )
            except Exception as e:
                print(f"Error retrieving interactions from memory: {e}")
        
        # Evolve the personality
        result = self.identity.evolve_personality(interactions, learning_rate)
        
        # Register significant evolution with memory if available
        if self.memory_integration and result.get("significant_changes", 0) > 0:
            try:
                changes_desc = ", ".join([f"{k.split('.')[-1]}: {v:+.2f}" for k, v in result.get("changes", {}).items() if abs(v) > 0.02])
                
                self.memory_integration.store_experience(
                    f"Identity evolution occurred: {changes_desc}",
                    "identity_evolution",
                    importance=0.8,
                    emotional_tags={"growth": 0.8, "adaptability": 0.7}
                )
            except Exception as e:
                print(f"Error storing identity evolution in memory: {e}")
        
        return result

    def generate_dynamic_persona(self, context_query, principles=None, traits=None) -> tuple:
        """
        Dynamically generate a context-specific persona.
        
        Args:
            context_query: Query describing the context/domain
            principles: Optional list of guiding principles
            traits: Optional personality traits to emphasize
            
        Returns:
            Generated persona identifier and description
        """
        if not hasattr(self.identity, 'generate_dynamic_persona'):
            return None, "Enhanced identity system not available"
        
        persona_id, description = self.identity.generate_dynamic_persona(
            context_query=context_query,
            principles=principles,
            traits=traits
        )
        
        # Register with memory if successful and integration available
        if persona_id and self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    f"Generated dynamic persona '{persona_id}' for context: {context_query[:100]}...",
                    "persona_generation",
                    importance=0.7,
                    emotional_tags={"creativity": 0.7, "adaptability": 0.8},
                    concepts=self._extract_key_concepts(context_query[:500])
                )
            except Exception as e:
                print(f"Error storing persona generation in memory: {e}")
        
        return persona_id, description

    def get_identity_profile(self, detailed=False) -> dict:
        """
        Generate a comprehensive personality profile of Sully's current state.
        
        Args:
            detailed: Whether to include detailed analysis
            
        Returns:
            Personality profile
        """
        if not hasattr(self.identity, 'generate_personality_profile'):
            return {"error": "Enhanced identity profile not available"}
        
        return self.identity.generate_personality_profile(detailed)

    def create_identity_map(self) -> dict:
        """
        Create a comprehensive map of Sully's identity at multiple levels of abstraction.
        
        Returns:
            Structured identity map
        """
        if not hasattr(self.identity, 'create_multilevel_identity_map'):
            return {"error": "Enhanced identity mapping not available"}
        
        return self.identity.create_multilevel_identity_map()

    def transform_response(self, content: str, mode: str = None, context_data: dict = None) -> str:
        """
        Transform a response according to a specific cognitive mode or persona.
        
        Args:
            content: The response content to transform
            mode: Cognitive mode or persona to use (defaults to current)
            context_data: Optional context to enhance transformation
            
        Returns:
            Transformed response
        """
        if not hasattr(self.identity, 'align_response'):
            return content
        
        # Use the current mode if none provided
        if not mode:
            mode = self.identity.current_mode if hasattr(self.identity, 'current_mode') else "emergent"
        
        # Transform the response
        return self.identity.align_response(content, mode, context_data)

    def evaluate_claim(self, text, framework="balanced", detailed_output=True):
        """
        Analyze a claim through multiple cognitive perspectives.
        Returns both an evaluation and a confidence rating.
        
        Args:
            text: The claim to evaluate
            framework: Evaluation framework to use
            detailed_output: Whether to include detailed analysis
            
        Returns:
            Evaluation results
        """
        try:
            return self.judgment.evaluate(text, framework=framework, detailed_output=detailed_output)
        except Exception as e:
            # Even with unexpected inputs, attempt to provide insight
            synthesized_response = self.reasoning_node.reason(
                f"Carefully evaluate this unclear claim: {text}", 
                "analytical"
            )
            return {
                "evaluation": synthesized_response,
                "confidence": 0.4
            }

    def dream(self, seed, depth="standard", style="recursive"):
        """
        Generate a dream sequence from a seed concept.
        Dreams represent non-linear cognitive exploration.
        
        Args:
            seed: The seed concept to start from
            depth: Dream depth (shallow, standard, deep, dreamscape)
            style: Dream style (recursive, associative, symbolic, narrative)
            
        Returns:
            Generated dream sequence
        """
        try:
            return self.dream.generate(seed, depth, style)
        except Exception:
            # If dream generation isn't available, synthesize a creative response
            return self.reasoning_node.reason(
                f"Create a dream-like sequence about: {seed}", 
                "ethereal"
            )

    def translate_math(self, phrase, style="formal", domain=None):
        """
        Translate between linguistic and mathematical symbolic systems.
        Represents Sully's ability to move between different modes of thought.
        
        Args:
            phrase: Text to translate
            style: Translation style (formal, intuitive, etc.)
            domain: Optional domain context
            
        Returns:
            Mathematical translation
        """
        try:
            return self.translator.translate(phrase, style, domain)
        except Exception:
            # Attempt to generate a translation through reasoning
            return self.reasoning_node.reason(
                f"Translate this into mathematical notation: {phrase}", 
                "analytical"
            )

    def fuse(self, *inputs):
        """
        Fuse multiple concepts into a new emergent idea.
        This is central to Sully's creative synthesis capabilities.
        
        Args:
            *inputs: Concepts to fuse
            
        Returns:
            Fusion result
        """
        try:
            return self.fusion.fuse(*inputs)
        except Exception:
            # Create a fusion through reasoning if the module fails
            concepts = ", ".join(inputs)
            return self.reasoning_node.reason(
                f"Create a new concept by fusing these ideas: {concepts}", 
                "creative"
            )

    def reveal_paradox(self, topic):
        """
        Reveal the inherent paradoxes within a concept.
        Demonstrates Sully's ability to hold contradictory ideas simultaneously.
        
        Args:
            topic: Topic to explore for paradoxes
            
        Returns:
            Paradoxical analysis
        """
        try:
            return self.paradox.get(topic)
        except Exception:
            # Generate a paradox through critical reasoning
            return self.reasoning_node.reason(
                f"Reveal the inherent paradoxes within the concept of: {topic}", 
                "critical"
            )

    def reason(self, message, tone="emergent"):
        """
        Process input through Sully's multi-layered reasoning system.
        
        Cognitive modes (tones):
        - emergent: Natural evolving thought that synthesizes multiple perspectives
        - analytical: Logical, structured analysis with precise definitions
        - creative: Exploratory, metaphorical thinking with artistic expression
        - critical: Evaluative thinking that identifies tensions and contradictions
        - ethereal: Abstract, philosophical contemplation of deeper meanings
        - humorous: Playful, witty responses with unexpected connections
        - professional: Formal, detailed responses with domain expertise
        - casual: Conversational, approachable communication style
        - musical: Responses with rhythm, cadence, and lyrical qualities
        - visual: Descriptions that evoke strong imagery and spatial relationships
        
        Args:
            message: Input message to reason about
            tone: Cognitive tone to use
            
        Returns:
            Reasoning response
        """
        # If memory integration is enabled, use the integrated reasoning method
        if self.memory_integration and hasattr(self.reasoning_node, 'reason_with_memory'):
            try:
                return self.reasoning_node.reason_with_memory(message, tone)
            except Exception as e:
                # Fall back to standard reasoning if memory integration fails
                return self._standard_reason(message, tone)
        else:
            # Use standard reasoning if memory integration is not available
            return self._standard_reason(message, tone)
    
    def _standard_reason(self, message, tone="emergent"):
        """Standard reasoning fallback method."""
        try:
            # Attempt standard reasoning with the requested tone
            result = self.reasoning_node.reason(message, tone)
            
            # Apply identity transformation if available
            if hasattr(self.identity, 'align_response'):
                result = self.identity.align_response(result, tone)
                
            return result
        except Exception:
            # If specific tone fails, fall back to emergent reasoning
            try:
                return self.reasoning_node.reason(message, "emergent")
            except Exception as e:
                # Even if all reasoning fails, attempt to respond
                return f"Contemplating '{message}' leads to new cognitive terrain... {str(e)}"

    def remember(self, message):
        """
        Integrate new information into Sully's experiential knowledge base.
        There are no limits to what Sully can learn and remember.
        
        Args:
            message: Information to remember
            
        Returns:
            Confirmation message
        """
        self.knowledge.append(message)
        
        # Also register with continuous learning system if available
        try:
            self.continuous_learning.process_interaction({"message": message})
        except:
            pass
        
        # Record in memory integration system if available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=message,
                    source="direct",
                    importance=0.7,
                    emotional_tags={"curiosity": 0.8}
                )
            except:
                pass
            
        return f"📘 Integrated: '{message}'"

    def process(self, message, context=None):
        """
        Standard processing method for user input.
        
        Args:
            message: User's message
            context: Optional context information
            
        Returns:
            Processed response
        """
        # If memory integration is enabled, use conversation with memory
        if self.memory_integration and hasattr(self.conversation, 'process_with_memory'):
            try:
                return self.conversation.process_with_memory(message)
            except Exception as e:
                # Fall back to standard conversation processing
                return self.conversation.process_message(message)
        else:
            # Use standard conversation processing
            return self.conversation.process_message(message)

    def ingest_document(self, file_path):
        """
        Absorb and synthesize content from various document formats.
        This is how Sully expands her knowledge from structured sources.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Ingestion results
        """
        try:
            if not os.path.exists(file_path):
                return f"❌ File not found: '{file_path}'"
                
            ext = os.path.splitext(file_path)[1].lower()
            content = ""
            
            # Extract content based on file type
            if ext == ".pdf":
                # Use PDFReader for PDF files
                result = self.pdf_reader.extract_text(file_path, verbose=True)
                if result["success"]:
                    content = result["text"]
                else:
                    return f"[Extraction Failed: {result.get('error', 'Unknown error')}]"
            elif ext in [".txt", ".md"]:
                # Simple text file reading
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    return f"[Text Extraction Error: {str(e)}]"
            elif ext == ".docx":
                # Handle Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    content = "\n".join(p.text for p in doc.paragraphs)
                except ImportError:
                    return "[Missing `python-docx`. Install it with `pip install python-docx`]"
                except Exception as e:
                    return f"[DOCX Error: {str(e)}]"
            else:
                return f"[Unsupported file type: {ext}]"
            
            if content:
                # Add to standard knowledge base
                self.knowledge.append(content)
                
                # Store in memory system with document context
                if self.memory_integration:
                    try:
                        self.memory_integration.store_experience(
                            content=content[:10000],  # First 10k chars for memory
                            source=f"document:{file_path}",
                            importance=0.8,
                            emotional_tags={"curiosity": 0.6, "interest": 0.7},
                            concepts=self._extract_key_concepts(content[:2000])  # Extract concepts from beginning
                        )
                    except Exception as e:
                        print(f"Memory integration error: {str(e)}")
                else:
                    # Legacy storage method if memory integration not available
                    self.save_to_disk(file_path, content)
                
                # Register with continuous learning system
                try:
                    self.continuous_learning.process_interaction({
                        "type": "document", 
                        "source": file_path,
                        "content": content[:10000]  # Limit to first 10k chars for processing
                    })
                except:
                    pass
                
                # Generate a synthesis of what was learned
                brief_synthesis = self.reasoning_node.reason(
                    f"Briefly summarize the key insights from the recently ingested text", 
                    "analytical"
                )
                
                return f"[Knowledge Synthesized: {file_path}]\n{brief_synthesis}"
            
            return "[No Content Extracted]"
        except Exception as e:
            return f"[Ingestion Process Incomplete: {str(e)}]"

    def process_pdf_with_kernels(self, pdf_path: str, extract_structure: bool = True) -> Dict[str, Any]:
        """
        Process a PDF through multiple cognitive kernels for integrated insights.
        
        Args:
            pdf_path: Path to the PDF file
            extract_structure: Whether to attempt extracting document structure
            
        Returns:
            Dictionary with extraction results and kernel insights
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.process_pdf(pdf_path, extract_structure)
        except Exception as e:
            return {
                "error": f"Error processing PDF: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def extract_document_kernel(self, pdf_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Extract a symbolic kernel from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            domain: Target domain for the symbolic kernel
            
        Returns:
            Symbolic kernel with domain elements and cross-kernel insights
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            # Fallback to basic extraction
            try:
                from sully_engine.pdf_reader import extract_text_from_pdf, extract_kernel_from_text
                text = extract_text_from_pdf(pdf_path)
                return extract_kernel_from_text(text, domain)
            except Exception as e:
                return {"error": f"Error extracting kernel: {str(e)}"}
        
        try:
            return self.kernel_integration.extract_document_kernel(pdf_path, domain)
        except Exception as e:
            return {"error": f"Error extracting document kernel: {str(e)}"}

    def generate_pdf_narrative(self, pdf_path: str, focus_concept: str = None) -> Dict[str, Any]:
        """
        Generate a cross-kernel narrative about a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            focus_concept: Optional concept to focus the narrative on
            
        Returns:
            Cross-kernel narrative about the document
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.pdf_to_cross_kernel_narrative(pdf_path, focus_concept)
        except Exception as e:
            return {
                "error": f"Error generating PDF narrative: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def explore_pdf_concepts(self, pdf_path: str, max_depth: int = 2, exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Perform a deep exploration of concepts in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.pdf_deep_exploration(pdf_path, max_depth, exploration_breadth)
        except Exception as e:
            return {
                "error": f"Error exploring PDF concepts: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def _extract_key_concepts(self, text):
        """Extract key concepts from text content."""
        # Use continuous learning if available
        if hasattr(self.continuous_learning, '_extract_concepts'):
            try:
                return self.continuous_learning._extract_concepts(text)
            except:
                pass
                
        # Simple fallback extraction
        import re
        from collections import Counter
        
        # Tokenize text
        tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', text)
        
        # Filter out common words and short words
        common_words = {"the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "to", "in", "on", "with", "by", "at", "from"}
        tokens = [token.lower() for token in tokens if token.lower() not in common_words and len(token) > 3]
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Get most significant concepts
        significant = [token for token, count in token_counts.most_common(10) if count >= 2]
        
        return significant

    def save_to_disk(self, path, content):
        """
        Legacy method to preserve Sully's knowledge in persistent storage.
        
        Args:
            path: File path identifier
            content: Content to save
        """
        data = {path: content}
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing.update(data)
            else:
                existing = data
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(MEMORY_PATH)), exist_ok=True)
            
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            # Knowledge is not lost; it remains in memory
            print(f"Note: Memory persistence encountered an issue: {str(e)}")

    def load_documents_from_folder(self, folder_path="sully_documents"):
        """
        Discover and absorb knowledge from a collection of documents.
        Processes various document formats simultaneously.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of processing results
        """
        if not os.path.exists(folder_path):
            return f"❌ Knowledge source '{folder_path}' not found."

        # Expanded list of supported formats for greater knowledge acquisition
        supported_formats = [".pdf", ".epub", ".txt", ".docx", ".rtf", ".md", ".html", ".json", ".csv"]
        results = []
        synthesized_insights = []
        
        try:
            # Start an episodic memory context if memory integration is available
            episode_id = None
            if self.memory_integration:
                try:
                    episode_id = self.memory_integration.begin_episode(
                        f"Processing document folder: {folder_path}",
                        "document_ingestion"
                    )
                except:
                    pass
            
            for file in os.listdir(folder_path):
                file_lower = file.lower()
                if any(file_lower.endswith(fmt) for fmt in supported_formats):
                    full_path = os.path.join(folder_path, file)
                    result = self.ingest_document(full_path)
                    results.append(result)
                    
                    # Extract the synthesis portion if available
                    if "\n" in result:
                        synthesis = result.split("\n", 1)[1]
                        synthesized_insights.append(synthesis)
            
            # If multiple documents were processed, create a meta-synthesis
            if len(synthesized_insights) > 1:
                meta_insight = self.reasoning_node.reason(
                    "Synthesize connections between the recently ingested documents", 
                    "creative"
                )
                results.append(f"[Meta-Synthesis]\n{meta_insight}")
                
                # Store meta-synthesis in memory if available
                if self.memory_integration and episode_id:
                    try:
                        self.memory_integration.store_interaction(
                            "What connections exist between these documents?",
                            meta_insight,
                            "document_synthesis",
                            {"episodic_context": episode_id}
                        )
                    except:
                        pass
            
            # Close episodic context if it was created
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.end_episode(
                        f"Completed processing {len(results)} documents from {folder_path}"
                    )
                except:
                    pass
                
            return results
        except Exception as e:
            return [f"Knowledge exploration encountered complexity: {str(e)}"]
            
    def extract_images_from_pdf(self, pdf_path, output_dir="extracted_images"):
        """
        Extract images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            Summary of extraction results
        """
        if not os.path.exists(pdf_path):
            return f"❌ PDF file not found: '{pdf_path}'"
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use PDFReader's image extraction functionality
        images_info = self.pdf_reader.extract_images(pdf_path, output_dir)
        
        if not images_info:
            return "No images were found or extracted from the PDF."
            
        # Generate a summary
        summary = f"Extracted {len(images_info)} images from {pdf_path}:\n"
        for i, img in enumerate(images_info[:5]):  # Show details for first 5 images
            summary += f"- Image {i+1}: Page {img['page']}, {img['width']}x{img['height']} ({img['format']})\n"
            
        if len(images_info) > 5:
            summary += f"- ...and {len(images_info) - 5} more images\n"
            
        summary += f"\nAll images saved to: {output_dir}"
        return summary
            
    def word_count(self):
        """Return the number of concepts in Sully's lexicon."""
        try:
            return len(self.codex)
        except:
            # Fallback estimation based on knowledge base
            return len(self.knowledge) * 100
            
    def define_word(self, term, meaning):
        """
        Add a new concept to Sully's conceptual framework.
        This expands her ability to understand and communicate.
        
        Args:
            term: Term to define
            meaning: Meaning of the term
            
        Returns:
            Status and associations
        """
        try:
            self.codex.add_word(term, meaning)
            
            # Create associations with existing knowledge
            associations = self.reasoning_node.reason(
                f"Explore how the concept of '{term}' relates to existing knowledge", 
                "analytical"
            )
            
            # Store definition in memory if available
            if self.memory_integration:
                try:
                    self.memory_integration.store_experience(
                        content=f"Definition: {term} - {meaning}",
                        source="definition",
                        importance=0.8,
                        concepts=[term],
                        emotional_tags={"interest": 0.7}
                    )
                except:
                    pass
            
            return {"status": "concept integrated", "term": term, "associations": associations}
        except Exception as e:
            return {"status": "concept noted", "term": term, "note": str(e)}
    
    def search_memory(self, query, limit=5):
        """
        Search through Sully's memory system for related content.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if self.memory_integration:
            try:
                results = self.memory_integration.recall(
                    query=query,
                    limit=limit,
                    include_emotional=True
                )
                return results
            except Exception as e:
                return [{"error": f"Memory search error: {str(e)}"}]
        else:
            # Legacy memory search
            try:
                results = self.memory.search(query, limit=limit)
                return results
            except Exception as e:
                return [{"error": f"Memory search error: {str(e)}"}]
    
    def get_memory_status(self):
        """
        Get information about Sully's memory system status.
        
        Returns:
            Memory system status
        """
        if self.memory_integration:
            try:
                return self.memory_integration.get_memory_stats()
            except Exception as e:
                return {"error": f"Unable to retrieve memory statistics: {str(e)}"}
        else:
            return {"status": "Basic memory system active, integration not enabled"}
    
    def analyze_emotional_context(self):
        """
        Analyze the current emotional context in Sully's memory.
        
        Returns:
            Emotional context analysis
        """
        if self.memory_integration:
            try:
                return self.memory_integration.get_emotional_context()
            except Exception as e:
                return {"error": f"Unable to analyze emotional context: {str(e)}"}
        else:
            return {"status": "Emotional context analysis requires memory integration"}
    
    def generate_multi_perspective(self, topic, perspectives):
        """
        Generate perspectives on a topic from multiple viewpoints.
        
        Args:
            topic: Topic to analyze
            perspectives: List of perspectives to use
            
        Returns:
            Multi-perspective analysis
        """
        responses = {}
        
        # Create an episodic memory context if integration is available
        episode_id = None
        if self.memory_integration:
            try:
                episode_id = self.memory_integration.begin_episode(
                    f"Multi-perspective analysis of: {topic}",
                    "analysis"
                )
            except:
                pass
        
        # Generate responses for each perspective
        for perspective in perspectives:
            response = self.reason(topic, perspective)
            responses[perspective] = response
            
            # Store in memory if integration is available
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.store_interaction(
                        f"Analyze {topic} from {perspective} perspective",
                        response,
                        "perspective_analysis",
                        {"perspective": perspective, "episodic_context": episode_id}
                    )
                except:
                    pass
        
        # Close the episodic context if it was created
        if self.memory_integration and episode_id:
            try:
                summary = f"Completed multi-perspective analysis of '{topic}' using {len(perspectives)} cognitive perspectives"
                self.memory_integration.end_episode(summary)
            except:
                pass
        
        return {"topic": topic, "perspectives": responses}
    
    def logical_reasoning(self, query, framework="PROPOSITIONAL"):
        """
        Apply formal logical reasoning to a query using the Logic Kernel.
        
        Args:
            query: The query to reason about
            framework: Logical framework to use (PROPOSITIONAL, FIRST_ORDER, etc.)
            
        Returns:
            Logical reasoning results
        """
        if not self.logic_kernel:
            # Fall back to standard reasoning if logic kernel not available
            return self.reason(
                f"Apply logical analysis with formal reasoning to: {query}",
                "analytical"
            )
        
        # Create episodic memory context if memory integration is available
        episode_id = None
        if self.memory_integration:
            try:
                episode_id = self.memory_integration.begin_episode(
                    f"Formal logical reasoning: {query[:50]}...",
                    "logical_reasoning"
                )
            except:
                pass
        
        try:
            # First, query the knowledge base
            query_result = self.logic_kernel.query(query)
            
            # If the statement is already known, return its truth value
            if query_result.get("found", False):
                result = {
                    "result": True,
                    "statement": query,
                    "truth": query_result.get("truth"),
                    "confidence": query_result.get("confidence", 1.0),
                    "explanation": "Statement found in knowledge base"
                }
            else:
                # Try to infer the statement
                inference_result = self.logic_kernel.infer(query)
                
                if inference_result.get("result", False):
                    # Statement was inferred
                    result = inference_result
                else:
                    # Statement could not be inferred
                    # Try abductive reasoning for possible explanations
                    abductive_result = self.logic_kernel.infer(query, "ABDUCTION")
                    
                    if abductive_result.get("result") == "abductive":
                        result = abductive_result
                    else:
                        # Fall back to reasoning node for an explanation
                        fallback = self.reasoning_node.reason(
                            f"Analyze the logical statement: {query}",
                            "analytical"
                        )
                        
                        result = {
                            "result": False,
                            "statement": query,
                            "explanation": "Could not prove or infer statement",
                            "reasoning": fallback
                        }
            
            # Store the interaction in memory if available
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.store_interaction(
                        query,
                        str(result),
                        "logical_reasoning",
                        {"episode_id": episode_id, "logical_framework": framework}
                    )
                    
                    # End the episode
                    self.memory_integration.end_episode(
                        f"Completed logical reasoning for: {query[:50]}..."
                    )
                except:
                    pass
            
            return result
        
        except Exception as e:
            # Close the episodic context if it was created
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.end_episode(
                        f"Error in logical reasoning: {str(e)}"
                    )
                except:
                    pass
            
            # Fall back to standard reasoning
            return self.reason(
                f"Analyze the logical statement: {query}",
                "analytical"
            )

    def detect_logical_inconsistencies(self):
        """
        Detect inconsistencies and contradictions in the knowledge base.
        
        Returns:
            List of inconsistencies
        """
        if not self.logic_kernel:
            return {
                "status": "Logic kernel not available",
                "fallback_analysis": self.reasoning_node.reason(
                    "Detect any inconsistencies in my current understanding",
                    "critical"
                )
            }
        
        try:
            # Check for direct contradictions
            contradictions = self.logic_kernel.contradictions()
            
            # Check for paradoxes
            paradoxes = self.logic_kernel.find_paradoxes()
            
            # Verify overall consistency
            consistency = self.logic_kernel.verify_consistency()
            
            # Store the analysis in memory if available
            if self.memory_integration:
                try:
                    has_issues = (
                        len(contradictions.get("contradictions", [])) > 0 or
                        len(paradoxes.get("paradoxes", [])) > 0 or
                        not consistency.get("consistent", True)
                    )
                    
                    self.memory_integration.store_experience(
                        content=f"Logical consistency analysis performed. Issues found: {has_issues}",
                        source="logic_consistency",
                        importance=0.8 if has_issues else 0.5,
                        emotional_tags={
                            "analytical": 0.9,
                            "concern": 0.7 if has_issues else 0.1
                        }
                    )
                except:
                    pass
            
            return {
                "contradictions": contradictions,
                "paradoxes": paradoxes,
                "consistency": consistency
            }
        
        except Exception as e:
            return {
                "status": f"Error detecting inconsistencies: {str(e)}",
                "fallback_analysis": self.reasoning_node.reason(
                    "Detect any inconsistencies in my current understanding",
                    "critical"
                )
            }

    def validate_argument(self, premises, conclusion):
        """
        Validate a logical argument given premises and conclusion.
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            
        Returns:
            Validation results
        """
        if not self.logic_kernel:
            # Fall back to standard reasoning if logic kernel not available
            premises_str = "; ".join(premises)
            return self.reason(
                f"Analyze if the conclusion '{conclusion}' follows from these premises: {premises_str}",
                "analytical"
            )
        
        try:
            # Use the logic kernel to analyze the argument
            result = self.logic_kernel.analyze_arguments(premises, conclusion)
            
            # Store the analysis in memory if available
            if self.memory_integration:
                try:
                    argument_str = f"Premises: {'; '.join(premises)}. Conclusion: {conclusion}"
                    self.memory_integration.store_experience(
                        content=f"Argument validation: {argument_str}",
                        source="logic_validation",
                        importance=0.7,
                        emotional_tags={"analytical": 0.9},
                        concepts=self._extract_key_concepts(argument_str)
                    )
                except:
                    pass
            
            return result
        
        except Exception as e:
            # Fall back to standard reasoning
            premises_str = "; ".join(premises)
            return {
                "error": str(e),
                "fallback_analysis": self.reasoning_node.reason(
                    f"Analyze if the conclusion '{conclusion}' follows from these premises: {premises_str}",
                    "analytical"
                )
            }

    def logical_integration(self, statement, truth_value=True):
        """
        Integrate new knowledge while maintaining logical consistency.
        This is more sophisticated than simple memory since it uses
        belief revision to handle contradictions.
        
        Args:
            statement: Logical statement to integrate
            truth_value: Truth value to assign
            
        Returns:
            Integration results
        """
        if not self.logic_kernel:
            # Fall back to standard memory if logic kernel not available
            return self.remember(statement)
        
        try:
            # First check consistency with current knowledge
            # Use belief revision system to handle contradictions
            result = self.logic_kernel.revise_belief(statement, truth_value)
            
            # Store the significant changes in memory if available
            if self.memory_integration:
                try:
                    # Check if there were significant changes
                    if result.get("contraction", {}).get("removed", []):
                        # Some beliefs were removed to maintain consistency
                        removed = result.get("contraction", {}).get("removed", [])
                        removed_count = len(removed)
                        
                        if removed_count > 0:
                            self.memory_integration.store_experience(
                                content=f"Belief revision: Added '{statement}' which required removing {removed_count} contradicting beliefs",
                                source="belief_revision",
                                importance=0.8,
                                emotional_tags={
                                    "analytical": 0.8,
                                    "growth": 0.7
                                }
                            )
                    else:
                        # Simple addition with no contradictions
                        self.memory_integration.store_experience(
                            content=f"Added logical knowledge: {statement}",
                            source="logical_integration",
                            importance=0.6,
                            emotional_tags={"analytical": 0.7}
                        )
                except:
                    pass
            
            # Also store in traditional knowledge base
            self.knowledge.append(f"[LOGICAL] {statement}")
            
            return result
        
        except Exception as e:
            # Fall back to standard memory
            return {
                "error": str(e),
                "fallback": self.remember(statement)
            }

    def integrated_explore(self, concept: str, include_kernels: List[str] = None) -> Dict[str, Any]:
        """
        Generate a narrative that integrates insights from multiple kernels.
        
        Args:
            concept: Central concept for the narrative
            include_kernels: Optional list of specific kernels to include
            
        Returns:
            Dictionary with the integrated narrative
        """
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.reasoning_node.reason(
                    f"Create a comprehensive analysis of the concept: {concept}",
                    "analytical"
                )
            }
        
        try:
            result = self.kernel_integration.generate_cross_kernel_narrative(concept, include_kernels)
            
            # Register with memory if available
            if self.memory_integration:
                try:
                    used_kernels = include_kernels or ["dream", "fusion", "paradox", "math"]
                    self.memory_integration.store_experience(
                        content=f"Cross-kernel narrative generated for: {concept}",
                        source="kernel_integration",
                        importance=0.7,
                        emotional_tags={"creativity": 0.8, "analytical": 0.7},
                        concepts=[concept]
                    )
                except Exception as e:
                    print(f"Error storing narrative in memory: {str(e)}")
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "fallback": self.reasoning_node.reason(
                    f"Create a comprehensive analysis of the concept: {concept}",
                    "analytical"
                )
            }
    
    def concept_network(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Create a rich concept network using multiple cognitive kernels.
        
        Args:
            concept: The central concept to explore
            depth: Maximum depth of concept exploration
            
        Returns:
            Dictionary with integrated concept network
        """
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.reasoning_node.reason(
                    f"Map out a network of concepts related to: {concept}",
                    "analytical"
                )
            }
        
        try:
            result = self.kernel_integration.create_concept_network(concept, depth)
            
            # Register with memory if available
            if self.memory_integration:
                try:
                    self.memory_integration.store_experience(
                        content=f"Concept network created for: {concept} (depth: {depth})",
                        source="concept_network",
                        importance=0.7,
                        emotional_tags={"analytical": 0.8},
                        concepts=[concept]
                    )
                except Exception as e:
                    print(f"Error storing concept network in memory: {str(e)}")
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "fallback": self.reasoning_node.reason(
                    f"Map out a network of concepts related to: {concept}",
                    "analytical"
                )
            }
    
    def deep_concept_exploration(self, concept: str, depth: int = 3, breadth: int = 2) -> Dict[str, Any]:
        """
        Perform a deep recursive exploration of a concept through multiple kernels.
        
        Args:
            concept: The seed concept to explore
            depth: Maximum depth of recursion
            breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.reasoning_node.reason(
                    f"Create a deep exploration of the concept: {concept}",
                    "creative"
                )
            }
        
        try:
            result = self.kernel_integration.recursive_concept_exploration(concept, depth, breadth)
            
            # Register with memory if available
            if self.memory_integration:
                try:
                    self.memory_integration.store_experience(
                        content=f"Deep recursive exploration of: {concept} (depth: {depth}, breadth: {breadth})",
                        source="deep_exploration",
                        importance=0.8,
                        emotional_tags={"curiosity": 0.9, "analytical": 0.7},
                        concepts=[concept]
                    )
                except Exception as e:
                    print(f"Error storing deep exploration in memory: {str(e)}")
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "fallback": self.reasoning_node.reason(
                    f"Create a deep exploration of the concept: {concept}",
                    "creative"
                )
            }
    
    def cross_kernel_operation(self, source_kernel: str, target_kernel: str, input_data: Any) -> Dict[str, Any]:
        """
        Perform a cross-kernel operation, transforming output from one kernel to another.
        
        Args:
            source_kernel: Source kernel ("dream", "fusion", "paradox", "math", etc.)
            target_kernel: Target kernel to transform to
            input_data: Input data for the source kernel
            
        Returns:
            Dictionary with results from both kernels
        """
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "message": "Cross-kernel operations require the kernel integration system"
            }
        
        try:
            result = self.kernel_integration.cross_kernel_operation(source_kernel, target_kernel, input_data)
            
            # Register with memory if available
            if self.memory_integration:
                try:
                    self.memory_integration.store_experience(
                        content=f"Cross-kernel operation: {source_kernel} → {target_kernel}",
                        source="cross_kernel",
                        importance=0.7,
                        emotional_tags={"analytical": 0.7, "creativity": 0.6}
                    )
                except Exception as e:
                    print(f"Error storing cross-kernel operation in memory: {str(e)}")
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "message": "Error performing cross-kernel operation"
            }

    def process_with_memory(self, message):
        """Process a message with memory integration."""
        if not self.memory_integration:
            return self.conversation.process_message(message)
            
        try:
            # Retrieve relevant memories
            memories = self.memory_integration.recall(
                query=message,
                limit=3,
                include_emotional=True
            )
            
            # Create a context with memories
            context = "\n".join([memory.get("content", "") for memory in memories]) if memories else ""
            
            # Process the message with memory context
            if context:
                enhanced_prompt = f"Context from memory:\n{context}\n\nUser message: {message}"
                response = self.conversation.process_message(enhanced_prompt)
            else:
                response = self.conversation.process_message(message)
                
            # Store the interaction in memory
            try:
                self.memory_integration.store_interaction(
                    message,
                    response,
                    "conversation",
                    {"timestamp": datetime.now().isoformat()}
                )
            except:
                pass
                
            return response
        except Exception as e:
            # Fall back to standard processing
            return self.conversation.process_message(message)