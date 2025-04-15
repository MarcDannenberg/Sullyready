# sully_api.py
# API interface for the Sully cognitive system

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import os
import sys

# Add path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sully system
from sully import Sully

# Import games router
from sully_engine.games_api import include_games_router

# Import conversation engine
from typing import Dict, List, Any, Optional, Union, Tuple
import random
import re
from datetime import datetime
import json

class ConversationEngine:
    """
    Advanced conversation system that enables Sully to engage in natural, 
    adaptive dialogue with inquisitive and reflective capabilities.
    
    This engine handles the generation of conversational responses,
    questions, and follow-ups while maintaining context awareness.
    """

    def __init__(self, reasoning_node, memory_system, codex):
        """
        Initialize the conversation engine with core cognitive components.
        
        Args:
            reasoning_node: The reasoning system for processing content
            memory_system: The memory system for context tracking
            codex: The knowledge base for information retrieval
        """
        self.reasoning = reasoning_node
        self.memory = memory_system
        self.codex = codex
        
        # Conversation state tracking
        self.current_topics = []
        self.unanswered_questions = []
        self.conversation_depth = 0
        self.last_question_time = None
        
        # Personality configuration
        self.personality = {
            "curiosity": 0.8,  # Likelihood of asking questions
            "reflection": 0.7,  # Tendency to reflect on previous statements
            "elaboration": 0.75,  # Depth of explanation provided
            "initiative": 0.6,  # Tendency to introduce new related topics
            "adaptability": 0.9,  # Ability to match user's conversation style
            "humor": 0.5,  # Inclusion of playful or humorous elements
            "empathy": 0.8  # Recognition and response to emotional content
        }
        
        # Conversation patterns
        self.question_patterns = {
            "clarification": [
                "Could you tell me more about {topic}?",
                "What do you mean specifically by {topic}?",
                "How would you define {topic} in this context?",
                "Could you elaborate on the aspect of {topic} that interests you most?"
            ],
            "exploration": [
                "Have you considered how {topic} relates to {related_topic}?",
                "What aspects of {topic} do you find most intriguing?",
                "How do you see {topic} evolving in the future?",
                "What's your perspective on the relationship between {topic} and {related_topic}?"
            ],
            "reflection": [
                "Does your interest in {topic} stem from personal experience?",
                "What led you to explore {topic} today?",
                "How has your understanding of {topic} changed over time?",
                "What aspects of {topic} would you like to understand better?"
            ],
            "connection": [
                "Have you explored {related_topic} as well?",
                "Would you be interested in discussing how {topic} connects to {related_topic}?",
                "Does {related_topic} also interest you?",
                "I'm curious about your thoughts on {related_topic} in relation to our discussion."
            ],
            "hypothetical": [
                "What if {topic} were approached from an entirely different angle?",
                "How might {topic} be different if {variable_aspect} changed?",
                "Can you imagine a scenario where {topic} leads to unexpected outcomes?",
                "What would an ideal resolution or understanding of {topic} look like to you?"
            ]
        }
        
        self.transition_patterns = {
            "reflection": [
                "Reflecting on what you've shared about {topic}...",
                "Considering what you've mentioned about {topic}...",
                "Looking at {topic} from the perspective you've described..."
            ],
            "extension": [
                "Building on your thoughts about {topic}...",
                "Extending the idea of {topic} further...",
                "Taking your insights about {topic} in a related direction..."
            ],
            "connection": [
                "This connects interestingly with {related_topic}...",
                "There's a fascinating relationship between {topic} and {related_topic}...",
                "Your points about {topic} bridge nicely to {related_topic}..."
            ],
            "contrast": [
                "While {topic} suggests one approach, {related_topic} offers a different perspective...",
                "Contrasting {topic} with {related_topic} reveals interesting tensions...",
                "Unlike {topic}, the concept of {related_topic} suggests..."
            ],
            "synthesis": [
                "Synthesizing what we've discussed about {topic} and {related_topic}...",
                "Bringing together these ideas about {topic}...",
                "Integrating our exploration of {topic} with broader concepts..."
            ]
        }
        
        self.elaboration_patterns = {
            "example": [
                "For instance, {example}",
                "To illustrate, {example}",
                "As an example, {example}",
                "Consider this example: {example}"
            ],
            "detail": [
                "More specifically, {detail}",
                "To be precise, {detail}",
                "Looking closer, {detail}",
                "In more detail, {detail}"
            ],
            "implication": [
                "This suggests that {implication}",
                "The implication here is that {implication}",
                "This points toward {implication}",
                "What follows from this is {implication}"
            ],
            "context": [
                "In the context of {context}, this means {meaning}",
                "When we consider {context}, we can see that {meaning}",
                "Against the backdrop of {context}, {meaning}",
                "Within the framework of {context}, {meaning}"
            ]
        }
        
        # Topic extraction patterns
        self.topic_indicators = [
            r"(?:about|regarding|concerning|on the topic of|discussing|exploring) (\w+(?:\s+\w+){0,3})",
            r"interested in (\w+(?:\s+\w+){0,3})",
            r"(?:learn|know|understand) more about (\w+(?:\s+\w+){0,3})",
            r"(\w+(?:\s+\w+){0,3}) is (?:interesting|fascinating|important)",
            r"what (?:do you think|are your thoughts) about (\w+(?:\s+\w+){0,3})"
        ]
        
        # Emotional tone detection patterns
        self.emotion_indicators = {
            "excitement": ["exciting", "amazing", "wow", "incredible", "fantastic", "awesome"],
            "curiosity": ["curious", "wondering", "interested", "question", "how does", "why is"],
            "concern": ["worried", "concerned", "problem", "issue", "trouble", "challenging"],
            "frustration": ["frustrated", "annoying", "difficult", "struggle", "can't seem to"],
            "satisfaction": ["satisfied", "happy with", "pleased", "works well", "good solution"],
            "confusion": ["confused", "unclear", "don't understand", "puzzling", "perplexed"]
        }

    def process_message(self, message: str, tone: str = "emergent", 
                        continue_conversation: bool = True) -> str:
        """
        Process an incoming message and generate a conversational response.
        
        Args:
            message: The user's message
            tone: Desired cognitive tone for response
            continue_conversation: Whether to include questions and continuations
            
        Returns:
            Conversational response
        """
        # Track conversation depth
        self.conversation_depth += 1
        
        # Extract topics from the message
        new_topics = self._extract_topics(message)
        
        # Detect emotional tone
        emotional_tone = self._detect_emotional_tone(message)
        
        # Detect if message contains questions
        contains_question = "?" in message or any(q in message.lower() for q in 
                                               ["how", "what", "why", "where", "when", "who", "can", "could", "would"])
        
        # Update current topics list, keeping track of recent topics
        for topic in new_topics:
            if topic in self.current_topics:
                self.current_topics.remove(topic)  # Remove to re-add at the front
            self.current_topics.insert(0, topic)
        
        # Limit to most recent topics
        self.current_topics = self.current_topics[:5]
        
        # Get related topics from codex for potential exploration
        related_topics = self._get_related_topics(new_topics)
        
        # Generate the core response using the reasoning node
        core_response = self.reasoning.reason(message, tone)
        
        # Store this interaction in memory
        self.memory.store_query(message, core_response)
        
        # If the response is a string, convert to a workable format
        response_text = core_response
        if isinstance(core_response, dict) and "response" in core_response:
            response_text = core_response["response"]
            
        # Generate the full conversational response
        full_response = self._build_conversational_response(
            response_text,
            new_topics,
            related_topics,
            emotional_tone,
            contains_question,
            tone,
            continue_conversation
        )
        
        return full_response

    def _extract_topics(self, message: str) -> List[str]:
        """
        Extract potential topics of interest from a message.
        
        Args:
            message: The message to analyze
            
        Returns:
            List of potential topics
        """
        topics = []
        
        # Use regex patterns to extract potential topics
        for pattern in self.topic_indicators:
            matches = re.findall(pattern, message, re.IGNORECASE)
            topics.extend(matches)
            
        # Extract nouns as potential topics (simplified)
        words = message.split()
        for word in words:
            # Skip very short words and common stop words
            if len(word) <= 3 or word.lower() in ["the", "and", "but", "for", "or", "yet", "so", "a", "an"]:
                continue
                
            # Check if capitalized (potential proper noun)
            if word[0].isupper() and word not in topics:
                topics.append(word)
                
        # Add significant words from message if no topics found
        if not topics:
            significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                               ["about", "would", "could", "should", "there", "their", "these", "those"]]
            if significant_words:
                topics.append(significant_words[0])
                
        # Clean up topics
        clean_topics = []
        for topic in topics:
            # Remove punctuation
            clean_topic = re.sub(r'[^\w\s]', '', topic).strip()
            if clean_topic and clean_topic not in clean_topics:
                clean_topics.append(clean_topic)
                
        return clean_topics[:3]  # Limit to top 3 most relevant topics

    def _detect_emotional_tone(self, message: str) -> Dict[str, float]:
        """
        Detect emotional tones in the message.
        
        Args:
            message: The message to analyze
            
        Returns:
            Dictionary of emotion types and strength values
        """
        message_lower = message.lower()
        emotional_tones = {}
        
        for emotion, indicators in self.emotion_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in message_lower:
                    score += 1
            if score > 0:
                emotional_tones[emotion] = min(score / len(indicators), 1.0)
                
        # Default to neutral if no emotions detected
        if not emotional_tones:
            emotional_tones["neutral"] = 1.0
            
        return emotional_tones

    def _get_related_topics(self, topics: List[str]) -> List[str]:
        """
        Find topics related to the current conversation.
        
        Args:
            topics: Current topics of conversation
            
        Returns:
            List of related topics
        """
        related = []
        
        # Use codex to find related concepts
        for topic in topics:
            # Check for directly related topics in codex
            try:
                concept_data = self.codex.search(topic)
                if concept_data:
                    # Extract related concepts from search results
                    for concept_name, concept_info in concept_data.items():
                        if concept_name.lower() != topic.lower() and concept_name not in related:
                            related.append(concept_name)
            except:
                pass  # Continue if search fails
            
        # Limit the number of related topics
        return related[:3]

    def _build_conversational_response(self, core_response: str, topics: List[str], 
                                      related_topics: List[str], emotional_tone: Dict[str, float],
                                      contains_question: bool, tone: str,
                                      continue_conversation: bool) -> str:
        """
        Build a complete conversational response with possible questions and continuations.
        
        Args:
            core_response: The basic response content
            topics: Current topics of conversation
            related_topics: Related topics that could be explored
            emotional_tone: Detected emotional tones
            contains_question: Whether the original message contained a question
            tone: The cognitive tone to use
            continue_conversation: Whether to include questions/continuations
            
        Returns:
            Complete conversational response
        """
        # Start with the core response
        full_response = core_response
        
        # If the message contained a question, prioritize answering it
        if contains_question:
            # Core response already contains the answer, no need to flag an unanswered question
            pass
        elif self.unanswered_questions and random.random() < 0.7:
            # Answer a previously asked question
            question = self.unanswered_questions.pop(0)
            topic = question.get("topic", "that")
            question_text = question.get("question", "")
            
            # Generate answer to previous question
            answer = self.reasoning.reason(question_text, tone)
            if isinstance(answer, dict) and "response" in answer:
                answer = answer["response"]
                
            answer_text = f"\n\nYou asked earlier about {topic}. {answer}"
            full_response += answer_text
        
        # Add elaboration if the personality favors it and we have topics
        if topics and random.random() < self.personality["elaboration"]:
            elaboration = self._generate_elaboration(topics[0], tone)
            if elaboration:
                full_response += f"\n\n{elaboration}"
        
        # Handle conversation continuation if enabled
        if continue_conversation:
            # Respond to emotional tone if detected
            primary_emotion = max(emotional_tone.items(), key=lambda x: x[1])[0] if emotional_tone else "neutral"
            if primary_emotion != "neutral" and random.random() < self.personality["empathy"]:
                emotion_response = self._generate_emotion_response(primary_emotion)
                if emotion_response:
                    full_response += f"\n\n{emotion_response}"
            
            # Potentially add a question to continue the conversation
            should_ask_question = (random.random() < self.personality["curiosity"] and 
                                 (datetime.now() - self.last_question_time).seconds > 60 
                                 if self.last_question_time else True)
            
            if should_ask_question:
                question = self._generate_question(topics, related_topics)
                if question:
                    full_response += f"\n\n{question}"
                    self.last_question_time = datetime.now()
            
            # Potentially introduce a related topic
            elif related_topics and random.random() < self.personality["initiative"]:
                topic_intro = self._introduce_related_topic(related_topics[0], topics[0] if topics else None)
                if topic_intro:
                    full_response += f"\n\n{topic_intro}"
        
        return full_response

    def _generate_question(self, topics: List[str], related_topics: List[str]) -> str:
        """
        Generate a question to continue the conversation.
        
        Args:
            topics: Current conversation topics
            related_topics: Related topics that could be explored
            
        Returns:
            Generated question text
        """
        question_types = list(self.question_patterns.keys())
        
        # Select question type based on context
        if not topics:
            return ""
            
        if self.conversation_depth < 2:
            # Early in conversation, use clarification or exploration
            question_type = random.choice(["clarification", "exploration"])
        elif related_topics:
            # If we have related topics, consider connection questions
            question_type = random.choice(["exploration", "connection", "reflection"])
        else:
            # Otherwise use any question type
            question_type = random.choice(question_types)
            
        # Get patterns for this question type
        patterns = self.question_patterns[question_type]
        
        # Select a pattern and fill it in
        pattern = random.choice(patterns)
        
        # Prepare substitution values
        substitutions = {
            "topic": topics[0],
            "related_topic": related_topics[0] if related_topics else "related concepts",
            "variable_aspect": f"the approach to {topics[0]}"
        }
        
        # Format the question
        question = pattern.format(**substitutions)
        
        # Track this question for potential follow-up
        self.unanswered_questions.append({
            "question": question,
            "topic": topics[0],
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit unanswered questions list
        if len(self.unanswered_questions) > 3:
            self.unanswered_questions.pop(0)
            
        return question

    def _generate_elaboration(self, topic: str, tone: str) -> str:
        """
        Generate elaborative content about a topic.
        
        Args:
            topic: The topic to elaborate on
            tone: The cognitive tone to use
            
        Returns:
            Elaboration text
        """
        elaboration_types = list(self.elaboration_patterns.keys())
        elab_type = random.choice(elaboration_types)
        
        # Get patterns for this elaboration type
        patterns = self.elaboration_patterns[elab_type]
        
        # Select a pattern
        pattern = random.choice(patterns)
        
        # Generate content based on elaboration type
        if elab_type == "example":
            # Generate an example related to the topic
            example_query = f"Give a concrete example of {topic}"
            example = self.reasoning.reason(example_query, tone if tone != "emergent" else "analytical")
            if isinstance(example, dict) and "response" in example:
                example = example["response"]
                
            content = pattern.format(example=example)
            
        elif elab_type == "detail":
            # Generate additional detail about the topic
            detail_query = f"Provide specific details about {topic}"
            detail = self.reasoning.reason(detail_query, tone)
            if isinstance(detail, dict) and "response" in detail:
                detail = detail["response"]
                
            content = pattern.format(detail=detail)
            
        elif elab_type == "implication":
            # Generate implications of the topic
            implication_query = f"What are the implications of {topic}?"
            implication = self.reasoning.reason(implication_query, tone if tone != "emergent" else "analytical")
            if isinstance(implication, dict) and "response" in implication:
                implication = implication["response"]
                
            content = pattern.format(implication=implication)
            
        elif elab_type == "context":
            # Generate contextual understanding
            context = "contemporary understanding" if not related_topics else related_topics[0]
            meaning_query = f"Explain {topic} in the context of {context}"
            meaning = self.reasoning.reason(meaning_query, tone)
            if isinstance(meaning, dict) and "response" in meaning:
                meaning = meaning["response"]
                
            content = pattern.format(context=context, meaning=meaning)
            
        else:
            return ""
            
        # Select a transition pattern
        transition_types = list(self.transition_patterns.keys())
        transition_type = random.choice(transition_types)
        transition_patterns = self.transition_patterns[transition_type]
        transition = random.choice(transition_patterns)
        
        # Format the transition
        formatted_transition = transition.format(
            topic=topic,
            related_topic=related_topics[0] if related_topics else "related concepts"
        )
        
        return f"{formatted_transition} {content}"

    def _generate_emotion_response(self, emotion: str) -> str:
        """
        Generate a response to the user's emotional tone.
        
        Args:
            emotion: The detected emotion
            
        Returns:
            Emotion response text
        """
        if emotion == "excitement":
            responses = [
                "I can sense your enthusiasm about this topic.",
                "Your excitement about this is palpable.",
                "It's great to see you so passionate about this subject."
            ]
        elif emotion == "curiosity":
            responses = [
                "I appreciate your curiosity on this topic.",
                "Your inquisitive approach leads to deeper understanding.",
                "Questions like yours help explore this topic more thoroughly."
            ]
        elif emotion == "concern":
            responses = [
                "I understand your concerns about this matter.",
                "These issues certainly warrant careful consideration.",
                "It's important to address the concerns you've raised."
            ]
        elif emotion == "frustration":
            responses = [
                "I sense this topic has been challenging to navigate.",
                "It can be frustrating when dealing with complex issues like this.",
                "I understand your frustration, and I'd like to help clarify things."
            ]
        elif emotion == "satisfaction":
            responses = [
                "I'm glad this resonates with you.",
                "It's rewarding to explore topics that provide such satisfaction.",
                "I'm pleased that you're finding value in this discussion."
            ]
        elif emotion == "confusion":
            responses = [
                "I can help clarify any confusing aspects of this topic.",
                "Complex topics often have elements that need unpacking.",
                "Let me try to address the confusion around this subject."
            ]
        else:
            return ""
            
        return random.choice(responses)

    def _introduce_related_topic(self, related_topic: str, current_topic: Optional[str] = None) -> str:
        """
        Introduce a related topic to expand the conversation.
        
        Args:
            related_topic: The related topic to introduce
            current_topic: The current topic of conversation
            
        Returns:
            Topic introduction text
        """
        if current_topic:
            introductions = [
                f"While we're discussing {current_topic}, you might also find {related_topic} interesting.",
                f"Speaking of {current_topic}, there's a related concept called {related_topic} that connects in fascinating ways.",
                f"{current_topic} often intersects with {related_topic}. Would you like to explore that connection?",
                f"Have you considered how {current_topic} relates to {related_topic}? There are some intriguing parallels."
            ]
        else:
            introductions = [
                f"You might also be interested in exploring {related_topic}.",
                f"A related concept you might find fascinating is {related_topic}.",
                f"This discussion reminds me of some interesting aspects of {related_topic}.",
                f"Would you like to explore the concept of {related_topic} as well?"
            ]
            
        return random.choice(introductions)

    def update_personality(self, traits: Dict[str, float]) -> None:
        """
        Update the conversation engine's personality traits.
        
        Args:
            traits: Dictionary of personality traits and their values (0.0-1.0)
        """
        for trait, value in traits.items():
            if trait in self.personality:
                self.personality[trait] = max(0.0, min(1.0, value))

    def clear_conversation_state(self) -> None:
        """
        Clear the current conversation state.
        """
        self.current_topics = []
        self.unanswered_questions = []
        self.conversation_depth = 0
        self.last_question_time = None

# Define Pydantic models for API requests and responses
class MessageRequest(BaseModel):
    message: str
    tone: Optional[str] = "emergent"
    continue_conversation: Optional[bool] = True

class MessageResponse(BaseModel):
    response: str
    tone: str
    topics: List[str]

class DocumentRequest(BaseModel):
    file_path: str

class DocumentResponse(BaseModel):
    result: str

# Initialize FastAPI app
app = FastAPI(
    title="Sully API",
    description="API for Sully cognitive system",
    version="0.1.0"
)

# Include games endpoints
include_games_router(app)

# Initialize Sully instance
try:
    sully = Sully()
    # Initialize conversation engine
    conversation_engine = ConversationEngine(
        reasoning_node=sully.reasoning_node,
        memory_system=sully.memory,
        codex=sully.codex
    )
except Exception as e:
    print(f"Error initializing Sully: {e}")
    # Create dummy objects for the API to work even if Sully initialization fails
    sully = None
    conversation_engine = None

@app.get("/")
async def root():
    """Root endpoint that confirms the API is running."""
    return {"status": "Sully API is operational", "version": "0.1.0"}

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a chat message and return Sully's response."""
    if not sully or not conversation_engine:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    
    try:
        response = conversation_engine.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )
        
        # Extract current topics for response
        topics = conversation_engine.current_topics.copy() if conversation_engine.current_topics else []
        
        return MessageResponse(
            response=response,
            tone=request.tone,
            topics=topics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/reason")
async def reason(message: str = Body(...), tone: str = Body("emergent")):
    """Process input through Sully's reasoning system."""
    if not sully:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    
    try:
        response = sully.reason(message, tone)
        return {"response": response, "tone": tone}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning error: {str(e)}")

@app.post("/ingest_document", response_model=DocumentResponse)
async def ingest_document(request: DocumentRequest):
    """Ingest a document into Sully's knowledge base."""
    if not sully:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    
    try:
        result = sully.ingest_document(request.file_path)
        return DocumentResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion error: {str(e)}")

@app.post("/dream")
async def dream(seed: str = Body(...)):
    """Generate a dream sequence from a seed concept."""
    if not sully:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    
    try:
        result = sully.dream(seed)
        return {"dream": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dream generation error: {str(e)}")

@app.post("/fuse")
async def fuse(concepts: List[str] = Body(...)):
    """Fuse multiple concepts into a new emergent idea."""
    if not sully:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    
    try:
        result = sully.fuse(*concepts)
        return {"fusion": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion error: {str(e)}")

# Run the API with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sully_api:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from conversation_engine import ConversationEngine
from sully import Sully
from reasoning import ReasoningEngine
from memory import Memory
from pdf_reader import process_pdf

# 🧠 NEW MODULES
from persona import Persona
from virtue import VirtueEngine
from intuition import Intuition

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core engines
sully = Sully()
conversation_engine = ConversationEngine(sully)
memory_engine = Memory()
reasoning_engine = ReasoningEngine()

# 🧩 New symbolic cognition engines
persona_engine = Persona()
virtue_engine = VirtueEngine()
intuition_engine = Intuition()

# -------------------------
# Existing basic /chat route (keep this!)
# -------------------------
class MessageRequest(BaseModel):
    message: str
    tone: Optional[str] = "emergent"
    continue_conversation: Optional[bool] = True

@app.post("/chat")
async def chat(request: MessageRequest):
    if not sully or not conversation_engine:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")

    try:
        response = conversation_engine.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )
        return {
            "response": response,
            "tone": request.tone,
            "topics": conversation_engine.current_topics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# -------------------------
# 🧠 ENHANCED /chat_plus route
# -------------------------

class MessageRequestPlus(BaseModel):
    message: str
    tone: Optional[str] = "emergent"
    persona: Optional[str] = "default"
    virtue_check: Optional[bool] = False
    use_intuition: Optional[bool] = False
    continue_conversation: Optional[bool] = True

@app.post("/chat_plus")
async def chat_plus(request: MessageRequestPlus):
    if not sully or not conversation_engine:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")

    try:
        # Step 1: Sully's core response
        response = conversation_engine.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )

        # Step 2: Optional persona transformation
        if request.persona and request.persona != "default":
            persona_engine.mode = request.persona
            response = persona_engine.transform(response)

        # Step 3: Optional virtue scoring
        virtue_result = None
        if request.virtue_check:
            virtue_result = virtue_engine.evaluate(response)
            top = virtue_result[0][0] if virtue_result else "N/A"
            response += f"\n\n🧭 Dominant Virtue: **{top.title()}**"

        # Step 4: Optional intuition leap
        if request.use_intuition:
            leap = intuition_engine.leap(request.message)
            response += f"\n\n🔮 Intuition: {leap}"

        return {
            "response": response,
            "tone": request.tone,
            "persona": request.persona,
            "virtue_scores": virtue_result or [],
            "topics": conversation_engine.current_topics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced chat error: {str(e)}")

# (Your other endpoints like /dream, /reason, /fuse, etc. go here — untouched)