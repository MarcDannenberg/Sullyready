# sully_api.py
# Complete API interface for the Sully cognitive system

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import os
import sys
import json
import uuid
from datetime import datetime, timedelta
import asyncio
from enum import Enum

# Add path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sully system
from sully import Sully

# Import games router
from sully_engine.games_api import include_games_router

# Enum definitions for various parameter options
class CognitiveMode(str, Enum):
    emergent = "emergent"
    analytical = "analytical"
    creative = "creative"
    critical = "critical"
    ethereal = "ethereal"
    humorous = "humorous"
    professional = "professional"
    casual = "casual"
    musical = "musical"
    visual = "visual"
    scientific = "scientific"
    philosophical = "philosophical"
    poetic = "poetic"
    instructional = "instructional"

class DreamDepth(str, Enum):
    shallow = "shallow"
    standard = "standard"
    deep = "deep"
    dreamscape = "dreamscape"

class DreamStyle(str, Enum):
    recursive = "recursive"
    associative = "associative"
    symbolic = "symbolic"
    narrative = "narrative"

class MathStyle(str, Enum):
    formal = "formal"
    intuitive = "intuitive"
    applied = "applied"
    creative = "creative"

class LogicalFramework(str, Enum):
    PROPOSITIONAL = "PROPOSITIONAL"
    FIRST_ORDER = "FIRST_ORDER"
    MODAL = "MODAL"
    TEMPORAL = "TEMPORAL"
    FUZZY = "FUZZY"

class CoreKernel(str, Enum):
    dream = "dream"
    fusion = "fusion"
    paradox = "paradox"
    math = "math"
    reasoning = "reasoning"
    conversation = "conversation"
    memory = "memory"

class EvaluationFramework(str, Enum):
    balanced = "balanced"
    logical = "logical"
    ethical = "ethical"
    practical = "practical"
    scientific = "scientific"
    creative = "creative"
    combined = "combined"

# Pydantic models for API requests and responses
class MessageRequest(BaseModel):
    message: str
    tone: Optional[CognitiveMode] = CognitiveMode.emergent
    continue_conversation: Optional[bool] = True

class MessageResponse(BaseModel):
    response: str
    tone: str
    topics: List[str]

class DocumentRequest(BaseModel):
    file_path: str

class DocumentResponse(BaseModel):
    result: str

class DreamRequest(BaseModel):
    seed: str
    depth: Optional[DreamDepth] = DreamDepth.standard
    style: Optional[DreamStyle] = DreamStyle.recursive

class DreamResponse(BaseModel):
    dream: str
    seed: str
    depth: str
    style: str

class FuseRequest(BaseModel):
    concepts: List[str]
    style: Optional[CognitiveMode] = CognitiveMode.creative

class FuseResponse(BaseModel):
    fusion: str
    concepts: List[str]
    style: str

class ParadoxRequest(BaseModel):
    topic: str
    perspectives: Optional[List[str]] = None

class MathTranslationRequest(BaseModel):
    phrase: str
    style: Optional[MathStyle] = MathStyle.formal
    domain: Optional[str] = None

class ClaimEvaluationRequest(BaseModel):
    text: str
    framework: Optional[EvaluationFramework] = EvaluationFramework.balanced
    detailed_output: Optional[bool] = True

class MultiFrameworkEvaluationRequest(BaseModel):
    text: str
    frameworks: List[EvaluationFramework]

class LogicalStatementRequest(BaseModel):
    statement: str
    truth_value: Optional[bool] = True

class LogicalRuleRequest(BaseModel):
    premises: List[str]
    conclusion: str

class LogicalQueryRequest(BaseModel):
    query: str
    framework: Optional[LogicalFramework] = LogicalFramework.PROPOSITIONAL

class CrossKernelRequest(BaseModel):
    source_kernel: CoreKernel
    target_kernel: CoreKernel
    input_data: Any

class ConceptNetworkRequest(BaseModel):
    concept: str
    depth: Optional[int] = 2

class DeepExplorationRequest(BaseModel):
    concept: str
    depth: Optional[int] = 3
    breadth: Optional[int] = 2

class NarrativeRequest(BaseModel):
    concept: str
    include_kernels: Optional[List[CoreKernel]] = None

class PDFProcessRequest(BaseModel):
    pdf_path: str
    extract_structure: Optional[bool] = True

class DocumentKernelRequest(BaseModel):
    pdf_path: str
    domain: Optional[str] = "general"

class PDFNarrativeRequest(BaseModel):
    pdf_path: str
    focus_concept: Optional[str] = None

class PDFConceptsRequest(BaseModel):
    pdf_path: str
    max_depth: Optional[int] = 2
    exploration_breadth: Optional[int] = 2

class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    include_emotional: Optional[bool] = True

class StoreExperienceRequest(BaseModel):
    content: str
    source: str
    importance: Optional[float] = 0.7
    emotional_tags: Optional[Dict[str, float]] = None
    concepts: Optional[List[str]] = None

class BeginEpisodeRequest(BaseModel):
    description: str
    context_type: str

class StoreInteractionRequest(BaseModel):
    query: str
    response: str
    interaction_type: str
    metadata: Optional[Dict[str, Any]] = None

class PersonaRequest(BaseModel):
    name: str
    traits: Dict[str, float]
    description: Optional[str] = None

class BlendedPersonaRequest(BaseModel):
    name: str
    personas: List[str]
    weights: Optional[List[float]] = None
    description: Optional[str] = None

class AdaptIdentityRequest(BaseModel):
    context: str
    context_data: Optional[Dict[str, Any]] = None

class EvolveIdentityRequest(BaseModel):
    interactions: Optional[List[Dict[str, Any]]] = None
    learning_rate: Optional[float] = 0.05

class DynamicPersonaRequest(BaseModel):
    context_query: str
    principles: Optional[List[str]] = None
    traits: Optional[Dict[str, float]] = None

class TransformRequest(BaseModel):
    content: str
    mode: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

class GoalRequest(BaseModel):
    goal: str
    priority: Optional[float] = 0.7
    domain: Optional[str] = None
    deadline: Optional[datetime] = None

class InterestRequest(BaseModel):
    topic: str
    engagement_level: Optional[float] = 0.8
    context: Optional[str] = None

class VisualProcessRequest(BaseModel):
    image_path: str
    analysis_depth: Optional[str] = "standard"
    include_objects: Optional[bool] = True
    include_scene: Optional[bool] = True

class VirtueEvaluationRequest(BaseModel):
    content: str
    context: Optional[str] = None
    domain: Optional[str] = None

class ActionVirtueRequest(BaseModel):
    action: str
    context: Optional[str] = None
    domain: Optional[str] = None

class IntuitionRequest(BaseModel):
    context: str
    concepts: Optional[List[str]] = None
    depth: Optional[str] = "standard"
    domain: Optional[str] = None

class MultiPerspectiveRequest(BaseModel):
    topic: str
    perspectives: List[CognitiveMode]

class EmergenceDetectionRequest(BaseModel):
    module_interactions: Optional[List[str]] = None
    threshold: Optional[float] = 0.7

class MessageRequestPlus(BaseModel):
    message: str
    tone: Optional[CognitiveMode] = CognitiveMode.emergent
    persona: Optional[str] = "default"
    virtue_check: Optional[bool] = False
    use_intuition: Optional[bool] = False
    continue_conversation: Optional[bool] = True

# Initialize FastAPI app
app = FastAPI(
    title="Sully API",
    description="API for Sully cognitive system - an advanced cognitive framework capable of synthesizing knowledge from various sources",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include games endpoints
include_games_router(app)

# Initialize Sully instance
try:
    sully = Sully()
    print("Sully system initialized successfully")
except Exception as e:
    print(f"Warning: Core Sully system initialization failed: {str(e)}")
    print("Starting with limited functionality")
    sully = None

# Initialize separate modules if not available in main Sully instance
try:
    from sully_engine.conversation_engine import ConversationEngine
    conversation_engine = ConversationEngine(
        reasoning_node=sully.reasoning_node if sully else None,
        memory_system=sully.memory if sully else None,
        codex=sully.codex if sully else None
    )
except Exception as e:
    print(f"Warning: Conversation engine initialization failed: {str(e)}")
    conversation_engine = None

try:
    from sully_engine.kernel_modules.persona import Persona
    persona_engine = Persona()
except Exception as e:
    print(f"Warning: Persona engine initialization failed: {str(e)}")
    persona_engine = None

try:
    from sully_engine.kernel_modules.virtue import VirtueEngine
    virtue_engine = VirtueEngine()
except Exception as e:
    print(f"Warning: Virtue engine initialization failed: {str(e)}")
    virtue_engine = None

try:
    from sully_engine.kernel_modules.intuition import Intuition
    intuition_engine = Intuition()
except Exception as e:
    print(f"Warning: Intuition engine initialization failed: {str(e)}")
    intuition_engine = None

# Helper function to check if Sully is initialized
def check_sully():
    if not sully:
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    return sully

# Helper function to check if the conversation engine is initialized
def check_conversation():
    if not conversation_engine:
        raise HTTPException(status_code=500, detail="Conversation engine not properly initialized")
    return conversation_engine

# Helper function to get a module from Sully if it exists
def get_sully_module(module_name):
    s = check_sully()
    if not hasattr(s, module_name):
        raise HTTPException(status_code=501, detail=f"Module '{module_name}' not available")
    return getattr(s, module_name)

# --- Core Interaction Routes ---

@app.get("/")
async def root():
    """Root endpoint that confirms the API is running."""
    return {
        "status": "Sully API is operational",
        "version": "1.0.0",
        "capabilities": [
            "Core Interaction",
            "Creative Functions",
            "Analytical Functions", 
            "Logical Reasoning",
            "Kernel Integration",
            "Memory Integration",
            "Identity & Personality",
            "Emergent Systems"
        ]
    }

@app.get("/system_status")
async def system_status():
    """Get comprehensive system status."""
    if not sully:
        return {
            "status": "limited",
            "message": "Core Sully system not initialized",
            "available_modules": []
        }
    
    # Get list of available modules
    modules = []
    for module_name in dir(sully):
        if not module_name.startswith("_") and not callable(getattr(sully, module_name)):
            modules.append(module_name)
    
    # Get memory status if available
    memory_status = None
    if hasattr(sully, "get_memory_status"):
        try:
            memory_status = sully.get_memory_status()
        except:
            memory_status = {"status": "error", "message": "Unable to get memory status"}
    
    # Get kernel integration status if available
    kernel_status = None
    if hasattr(sully, "kernel_integration") and sully.kernel_integration:
        try:
            if hasattr(sully.kernel_integration, "get_stats"):
                kernel_status = sully.kernel_integration.get_stats()
            else:
                kernel_status = {"status": "active", "details": "Statistics not available"}
        except:
            kernel_status = {"status": "error", "message": "Unable to get kernel integration status"}
    
    return {
        "status": "operational",
        "modules": modules,
        "memory": memory_status,
        "kernel_integration": kernel_status,
        "system_time": datetime.now().isoformat()
    }

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a chat message and return Sully's response."""
    ce = check_conversation()
    
    try:
        response = ce.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )
        
        # Extract current topics for response
        topics = ce.current_topics.copy() if ce.current_topics else []
        
        return MessageResponse(
            response=response,
            tone=request.tone,
            topics=topics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual processing error: {str(e)}")

# --- Ethics & Intuition Routes ---

@app.post("/virtue/evaluate")
async def virtue_evaluate(request: VirtueEvaluationRequest):
    """Evaluate content using virtue ethics."""
    s = check_sully()
    
    try:
        result = s.evaluate_virtue(request.content, request.context, request.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Virtue evaluation error: {str(e)}")

@app.post("/virtue/evaluate_action")
async def virtue_evaluate_action(request: ActionVirtueRequest):
    """Evaluate an action through virtue ethics framework."""
    s = check_sully()
    
    try:
        result = s.evaluate_action_virtue(request.action, request.context, request.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action virtue evaluation error: {str(e)}")

@app.post("/virtue/reflect")
async def virtue_reflect(virtue: str = Body(...)):
    """Generate meta-ethical reflection on a specific virtue."""
    s = check_sully()
    
    try:
        if hasattr(s, "reflect_on_virtue"):
            result = s.reflect_on_virtue(virtue)
            return result
        
        # Fallback using reasoning
        reflection = s.reason(
            f"Provide a deep philosophical reflection on the virtue of {virtue}",
            "philosophical"
        )
        
        return {
            "virtue": virtue,
            "reflection": reflection,
            "note": "Generated through reasoning, not formal virtue engine"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Virtue reflection error: {str(e)}")

@app.post("/intuition/leap")
async def intuition_leap(request: IntuitionRequest):
    """Generate intuitive leaps."""
    s = check_sully()
    
    try:
        result = s.generate_intuitive_leap(
            request.context,
            request.concepts,
            request.depth,
            request.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intuitive leap error: {str(e)}")

@app.post("/course_of_thought")
async def course_of_thought(request: MultiPerspectiveRequest):
    """Generate multi-perspective thought."""
    s = check_sully()
    
    try:
        result = s.generate_multi_perspective(request.topic, request.perspectives)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-perspective thought error: {str(e)}")

@app.post("/neural/analyze")
async def neural_analyze(module: str = Body(...)):
    """Analyze module performance."""
    s = check_sully()
    
    try:
        if hasattr(s, "neural_modification") and hasattr(s.neural_modification, "analyze_module"):
            result = s.neural_modification.analyze_module(module)
            return result
        
        # Fallback response if method not available
        return {
            "module": module,
            "message": "Neural analysis not available for this module",
            "status": "limited"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural analysis error: {str(e)}")

# Run the API with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sully_api:app", host="0.0.0.0", port=8000, reload=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/chat_plus")
async def chat_plus(request: MessageRequestPlus):
    """Enhanced chat with optional persona, virtue checking, and intuition."""
    ce = check_conversation()

    try:
        # Step 1: Sully's core response
        response = ce.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )

        # Step 2: Optional persona transformation
        if request.persona and request.persona != "default" and persona_engine:
            try:
                persona_engine.mode = request.persona
                response = persona_engine.transform(response)
            except Exception as e:
                print(f"Persona transformation error: {str(e)}")

        # Step 3: Optional virtue scoring
        virtue_result = None
        if request.virtue_check and virtue_engine:
            try:
                virtue_result = virtue_engine.evaluate(response)
                top = virtue_result[0][0] if virtue_result else "N/A"
                response += f"\n\n🧭 Dominant Virtue: **{top.title()}**"
            except Exception as e:
                print(f"Virtue evaluation error: {str(e)}")

        # Step 4: Optional intuition leap
        if request.use_intuition and intuition_engine:
            try:
                leap = intuition_engine.leap(request.message)
                response += f"\n\n🔮 Intuition: {leap}"
            except Exception as e:
                print(f"Intuition error: {str(e)}")

        return {
            "response": response,
            "tone": request.tone,
            "persona": request.persona,
            "virtue_scores": virtue_result or [],
            "topics": ce.current_topics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced chat error: {str(e)}")

@app.post("/reason")
async def reason(message: str = Body(...), tone: CognitiveMode = Body(CognitiveMode.emergent)):
    """Process input through Sully's reasoning system."""
    s = check_sully()
    
    try:
        response = s.reason(message, tone)
        return {"response": response, "tone": tone}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning error: {str(e)}")

@app.post("/reason/with_memory")
async def reason_with_memory(message: str = Body(...), tone: CognitiveMode = Body(CognitiveMode.emergent)):
    """Process input through Sully's reasoning system with memory integration."""
    s = check_sully()
    
    if not hasattr(s.reasoning_node, 'reason_with_memory'):
        return await reason(message, tone)
    
    try:
        response = s.reasoning_node.reason_with_memory(message, tone)
        return {"response": response, "tone": tone, "memory_enhanced": True}
    except Exception as e:
        # Fall back to standard reasoning
        response = s.reason(message, tone)
        return {"response": response, "tone": tone, "memory_enhanced": False, "fallback_reason": str(e)}

@app.post("/ingest_document", response_model=DocumentResponse)
async def ingest_document(request: DocumentRequest):
    """Ingest a document into Sully's knowledge base."""
    s = check_sully()
    
    try:
        result = s.ingest_document(request.file_path)
        return DocumentResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion error: {str(e)}")

@app.post("/ingest_folder")
async def ingest_folder(folder_path: str = Body(..., embed=True)):
    """Process multiple documents from a folder."""
    s = check_sully()
    
    try:
        results = s.load_documents_from_folder(folder_path)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder ingestion error: {str(e)}")

# --- Creative Functions Routes ---

@app.post("/dream", response_model=DreamResponse)
async def dream(request: DreamRequest):
    """Generate a dream sequence from a seed concept."""
    s = check_sully()
    
    try:
        result = s.dream(request.seed, request.depth, request.style)
        return DreamResponse(
            dream=result,
            seed=request.seed,
            depth=request.depth,
            style=request.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dream generation error: {str(e)}")

@app.post("/dream/advanced")
async def dream_advanced(
    seed: str = Body(...),
    depth: DreamDepth = Body(DreamDepth.standard),
    style: DreamStyle = Body(DreamStyle.recursive),
    filter_concepts: Optional[List[str]] = Body(None),
    emotional_tone: Optional[str] = Body(None),
    narrative_structure: Optional[bool] = Body(False)
):
    """Generate a dream with advanced control options."""
    s = check_sully()
    dream_module = get_sully_module("dream")
    
    try:
        # First generate base dream
        dream_result = s.dream(seed, depth, style)
        
        # Perform advanced processing if requested
        if filter_concepts or emotional_tone or narrative_structure:
            # If dream module has advanced methods, use them
            if hasattr(dream_module, "apply_emotional_tone") and emotional_tone:
                dream_result = dream_module.apply_emotional_tone(dream_result, emotional_tone)
            
            if hasattr(dream_module, "apply_narrative_structure") and narrative_structure:
                dream_result = dream_module.apply_narrative_structure(dream_result)
            
            if hasattr(dream_module, "filter_through_concepts") and filter_concepts:
                dream_result = dream_module.filter_through_concepts(dream_result, filter_concepts)
            
            # Fallback using reasoning if direct methods not available
            if not hasattr(dream_module, "apply_emotional_tone") and emotional_tone:
                dream_result = s.reason(
                    f"Rewrite this dream with a {emotional_tone} emotional tone: {dream_result}", 
                    "creative"
                )
            
            if not hasattr(dream_module, "apply_narrative_structure") and narrative_structure:
                dream_result = s.reason(
                    f"Restructure this dream with a clearer narrative arc: {dream_result}", 
                    "creative"
                )
        
        return {
            "dream": dream_result,
            "seed": seed,
            "depth": depth,
            "style": style,
            "advanced_processing": {
                "emotional_tone": emotional_tone,
                "narrative_structure": narrative_structure,
                "filter_concepts": filter_concepts
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced dream generation error: {str(e)}")

@app.post("/fuse", response_model=FuseResponse)
async def fuse(request: FuseRequest):
    """Fuse multiple concepts into a new emergent idea."""
    s = check_sully()
    
    try:
        result = s.fuse(*request.concepts)
        return FuseResponse(
            fusion=result,
            concepts=request.concepts,
            style=request.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion error: {str(e)}")

@app.post("/fuse/advanced")
async def fuse_advanced(
    concepts: List[str] = Body(...),
    style: CognitiveMode = Body(CognitiveMode.creative),
    weighting: Optional[List[float]] = Body(None),
    domain_context: Optional[str] = Body(None),
    output_format: Optional[str] = Body("prose")
):
    """Advanced concept fusion with style and cognitive mode options."""
    s = check_sully()
    fusion_module = get_sully_module("fusion")
    
    try:
        # If fusion module has advanced methods, use them
        if hasattr(fusion_module, "fuse_with_weights") and weighting:
            result = fusion_module.fuse_with_weights(concepts, weighting)
        elif hasattr(fusion_module, "fuse_in_context") and domain_context:
            result = fusion_module.fuse_in_context(concepts, domain_context)
        else:
            # Fall back to standard fusion
            result = s.fuse(*concepts)
        
        # Post-process the fusion result according to style and format
        if style != CognitiveMode.creative or output_format != "prose":
            # Transform the result to the requested style
            instruction = f"Transform this concept fusion into {output_format} format:"
            result = s.reason(f"{instruction} {result}", style)
            
        return {
            "fusion": result,
            "concepts": concepts,
            "style": style,
            "domain_context": domain_context,
            "output_format": output_format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced fusion error: {str(e)}")

# --- Analytical Functions Routes ---

@app.post("/evaluate")
async def evaluate_claim(request: ClaimEvaluationRequest):
    """Analyze a claim through multiple cognitive perspectives."""
    s = check_sully()
    
    try:
        result = s.evaluate_claim(request.text, request.framework, request.detailed_output)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/evaluate/multi")
async def evaluate_multi_framework(request: MultiFrameworkEvaluationRequest):
    """Analyze a claim through multiple evaluation frameworks simultaneously."""
    s = check_sully()
    judgment = get_sully_module("judgment")
    
    results = {}
    try:
        # Try to use multi-perspective evaluation if available
        if hasattr(s, "multi_perspective_evaluation"):
            return s.multi_perspective_evaluation(request.text)
        
        # Otherwise evaluate with each framework individually
        for framework in request.frameworks:
            results[framework] = s.evaluate_claim(request.text, framework)
            
        # Create a synthesis of the frameworks
        synthesis = s.reason(
            f"Synthesize these different perspectives on the claim: '{request.text}'",
            "analytical"
        )
        
        return {
            "claim": request.text,
            "framework_evaluations": results,
            "synthesis": synthesis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-framework evaluation error: {str(e)}")

@app.post("/translate/math")
async def translate_math(request: MathTranslationRequest):
    """Translate between language and mathematics."""
    s = check_sully()
    
    try:
        result = s.translate_math(request.phrase, request.style, request.domain)
        return {
            "input": request.phrase,
            "translation": result,
            "style": request.style,
            "domain": request.domain
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Math translation error: {str(e)}")

@app.post("/paradox")
async def paradox(request: ParadoxRequest):
    """Reveal paradoxes from different perspectives."""
    s = check_sully()
    
    try:
        # Basic paradox analysis
        result = s.reveal_paradox(request.topic)
        
        # Add perspectives if requested
        if request.perspectives:
            perspectives = {}
            for perspective in request.perspectives:
                perspectives[perspective] = s.reason(
                    f"Analyze the paradox of {request.topic} from a {perspective} perspective",
                    "critical"
                )
            
            return {
                "topic": request.topic,
                "paradox": result,
                "perspectives": perspectives
            }
        
        return {
            "topic": request.topic,
            "paradox": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paradox exploration error: {str(e)}")

# --- Logical Reasoning Routes ---

@app.post("/logic/assert")
async def logic_assert(request: LogicalStatementRequest):
    """Assert a logical statement into the knowledge base."""
    s = check_sully()
    
    try:
        result = s.logical_integration(request.statement, request.truth_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical assertion error: {str(e)}")

@app.post("/logic/rule")
async def logic_rule(request: LogicalRuleRequest):
    """Define logical rules with premises and conclusion."""
    s = check_sully()
    
    try:
        # Format the rule in a standard way
        rule_text = " & ".join(request.premises) + " -> " + request.conclusion
        
        # Integrate the rule
        result = s.logical_integration(rule_text)
        result["rule"] = {
            "premises": request.premises,
            "conclusion": request.conclusion
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical rule definition error: {str(e)}")

@app.post("/logic/infer")
async def logic_infer(request: LogicalQueryRequest):
    """Perform logical inference on a statement."""
    s = check_sully()
    
    try:
        result = s.logical_reasoning(request.query, request.framework)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical inference error: {str(e)}")

@app.post("/logic/proof")
async def logic_proof(query: str = Body(...), framework: LogicalFramework = Body(LogicalFramework.PROPOSITIONAL)):
    """Generate a formal logical proof."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the proof method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "prove"):
            result = s.logic_kernel.prove(query, framework)
            return result
        
        # Fallback to generating a proof through reasoning
        proof = s.reason(
            f"Generate a formal logical proof for this statement using {framework} logic: {query}",
            "analytical"
        )
        
        return {
            "query": query,
            "framework": framework,
            "proof": proof,
            "note": "Generated through reasoning, not formal logic kernel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical proof generation error: {str(e)}")

@app.post("/logic/equivalence")
async def logic_equivalence(statement1: str = Body(...), statement2: str = Body(...)):
    """Check logical equivalence of two statements."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the equivalence method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "check_equivalence"):
            result = s.logic_kernel.check_equivalence(statement1, statement2)
            return result
        
        # Fallback using reasoning
        analysis = s.reason(
            f"Determine if these two logical statements are equivalent:\n1. {statement1}\n2. {statement2}",
            "analytical"
        )
        
        return {
            "statement1": statement1,
            "statement2": statement2,
            "analysis": analysis,
            "note": "Generated through reasoning, not formal logic kernel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical equivalence error: {str(e)}")

@app.post("/logic/query")
async def logic_db_query(request: LogicalQueryRequest):
    """Query the logical knowledge base."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the query method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "query"):
            result = s.logic_kernel.query(request.query)
            return result
        
        # Fallback using reasoning
        response = s.reason(
            f"Based on logical knowledge, analyze this query: {request.query}",
            "analytical"
        )
        
        return {
            "query": request.query,
            "response": response,
            "note": "Generated through reasoning, not formal logic kernel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical query error: {str(e)}")

@app.get("/logic/paradoxes")
async def logic_paradoxes():
    """Find logical paradoxes in the knowledge base."""
    s = check_sully()
    
    try:
        result = s.detect_logical_inconsistencies()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical paradox detection error: {str(e)}")

@app.post("/logic/undecidable")
async def logic_undecidable(statement: str = Body(...)):
    """Check if a statement is potentially undecidable."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the undecidable method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "check_undecidable"):
            result = s.logic_kernel.check_undecidable(statement)
            return result
        
        # Fallback using reasoning
        analysis = s.reason(
            f"Analyze whether this statement might be logically undecidable: {statement}",
            "analytical"
        )
        
        return {
            "statement": statement,
            "analysis": analysis,
            "note": "Generated through reasoning, not formal logic kernel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Undecidability analysis error: {str(e)}")

@app.post("/logic/argument")
async def logic_argument(premises: List[str] = Body(...), conclusion: str = Body(...)):
    """Analyze the validity of a logical argument."""
    s = check_sully()
    
    try:
        result = s.validate_argument(premises, conclusion)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Argument validation error: {str(e)}")

@app.get("/logic/consistency")
async def logic_consistency():
    """Verify consistency of logical knowledge."""
    s = check_sully()
    
    try:
        result = s.detect_logical_inconsistencies()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logical consistency check error: {str(e)}")

@app.post("/logic/revise")
async def logic_revise(statement: str = Body(...), truth_value: bool = Body(True)):
    """Revise beliefs while maintaining consistency."""
    s = check_sully()
    
    try:
        result = s.logical_integration(statement, truth_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Belief revision error: {str(e)}")

@app.post("/logic/retract")
async def logic_retract(statement: str = Body(...)):
    """Remove a belief from the knowledge base."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the retract method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "retract_belief"):
            result = s.logic_kernel.retract_belief(statement)
            return result
        
        # Fallback response if method not available
        return {
            "statement": statement,
            "status": "error",
            "message": "Belief retraction not available in current logic kernel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Belief retraction error: {str(e)}")

@app.get("/logic/stats")
async def logic_stats():
    """Get statistics about the logical reasoning system."""
    s = check_sully()
    
    try:
        # Check if logic kernel has the stats method
        if hasattr(s, "logic_kernel") and s.logic_kernel and hasattr(s.logic_kernel, "get_stats"):
            result = s.logic_kernel.get_stats()
            return result
        
        # Fallback response if method not available
        return {
            "status": "limited",
            "message": "Full logic statistics not available in current logic kernel",
            "beliefs": "Unknown",
            "rules": "Unknown",
            "consistency": "Unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logic statistics error: {str(e)}")

# --- Kernel Integration Routes ---

@app.post("/kernel_integration/narrative")
async def kernel_integration_narrative(request: NarrativeRequest):
    """Generate integrated cross-kernel narratives."""
    s = check_sully()
    
    try:
        result = s.integrated_explore(request.concept, request.include_kernels)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-kernel narrative error: {str(e)}")

@app.post("/kernel_integration/concept_network")
async def kernel_integration_concept_network(request: ConceptNetworkRequest):
    """Create multi-modal concept networks."""
    s = check_sully()
    
    try:
        result = s.concept_network(request.concept, request.depth)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concept network error: {str(e)}")

@app.post("/kernel_integration/deep_exploration")
async def kernel_integration_deep_exploration(request: DeepExplorationRequest):
    """Perform recursive concept exploration."""
    s = check_sully()
    
    try:
        result = s.deep_concept_exploration(request.concept, request.depth, request.breadth)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep exploration error: {str(e)}")

@app.post("/kernel_integration/cross_kernel")
async def kernel_integration_cross_kernel(request: CrossKernelRequest):
    """Execute cross-kernel operations."""
    s = check_sully()
    
    try:
        result = s.cross_kernel_operation(request.source_kernel, request.target_kernel, request.input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-kernel operation error: {str(e)}")

@app.post("/kernel_integration/process_pdf")
async def kernel_integration_process_pdf(request: PDFProcessRequest):
    """Process documents through multiple cognitive kernels."""
    s = check_sully()
    
    try:
        result = s.process_pdf_with_kernels(request.pdf_path, request.extract_structure)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

@app.post("/kernel_integration/extract_document_kernel")
async def kernel_integration_extract_document_kernel(request: DocumentKernelRequest):
    """Extract symbolic kernels from documents."""
    s = check_sully()
    
    try:
        result = s.extract_document_kernel(request.pdf_path, request.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document kernel extraction error: {str(e)}")

@app.post("/kernel_integration/generate_pdf_narrative")
async def kernel_integration_generate_pdf_narrative(request: PDFNarrativeRequest):
    """Generate narratives about document content."""
    s = check_sully()
    
    try:
        result = s.generate_pdf_narrative(request.pdf_path, request.focus_concept)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF narrative generation error: {str(e)}")

@app.post("/kernel_integration/explore_pdf_concepts")
async def kernel_integration_explore_pdf_concepts(request: PDFConceptsRequest):
    """Explore concepts from PDFs."""
    s = check_sully()
    
    try:
        result = s.explore_pdf_concepts(request.pdf_path, request.max_depth, request.exploration_breadth)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF concept exploration error: {str(e)}")

@app.get("/kernel_integration/status")
async def kernel_integration_status():
    """Get the status of the kernel integration system."""
    s = check_sully()
    
    try:
        if not s.kernel_integration:
            return {"status": "not_available", "message": "Kernel integration system not initialized"}
        
        if hasattr(s.kernel_integration, "get_stats"):
            return s.kernel_integration.get_stats()
        
        # Basic info if stats not available
        return {
            "status": "active",
            "message": "Kernel integration system is active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kernel status error: {str(e)}")

@app.get("/kernel_integration/available_kernels")
async def kernel_integration_available_kernels():
    """List available kernels for integration."""
    s = check_sully()
    
    try:
        if not s.kernel_integration:
            return {"kernels": [], "message": "Kernel integration system not initialized"}
        
        # Get available kernels
        available_kernels = []
        if hasattr(s, "dream"):
            available_kernels.append("dream")
        if hasattr(s, "fusion"):
            available_kernels.append("fusion")
        if hasattr(s, "paradox"):
            available_kernels.append("paradox")
        if hasattr(s, "translator"):
            available_kernels.append("math")
        if hasattr(s, "reasoning_node"):
            available_kernels.append("reasoning")
        if hasattr(s, "conversation"):
            available_kernels.append("conversation")
        if hasattr(s, "memory"):
            available_kernels.append("memory")
        
        # Get detailed info if available
        if hasattr(s.kernel_integration, "get_kernel_info"):
            kernels_info = s.kernel_integration.get_kernel_info()
            return {"kernels": available_kernels, "details": kernels_info}
        
        return {"kernels": available_kernels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Available kernels error: {str(e)}")

# --- Memory Integration Routes ---

@app.post("/memory/search")
async def memory_search(request: MemorySearchRequest):
    """Search memory system."""
    s = check_sully()
    
    try:
        results = s.search_memory(request.query, request.limit)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search error: {str(e)}")

@app.get("/memory/status")
async def memory_status():
    """Get memory system status."""
    s = check_sully()
    
    try:
        result = s.get_memory_status()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory status error: {str(e)}")

@app.get("/memory/emotional")
async def memory_emotional():
    """Get emotional context from memory."""
    s = check_sully()
    
    try:
        result = s.analyze_emotional_context()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotional context analysis error: {str(e)}")

@app.post("/memory/begin_episode")
async def memory_begin_episode(request: BeginEpisodeRequest):
    """Begin new episodic memory context."""
    s = check_sully()
    
    try:
        if not hasattr(s, "memory_integration") or not s.memory_integration:
            return {"status": "error", "message": "Memory integration not enabled"}
        
        episode_id = s.memory_integration.begin_episode(
            request.description,
            request.context_type
        )
        
        return {"status": "success", "episode_id": episode_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Begin episode error: {str(e)}")

@app.post("/memory/end_episode")
async def memory_end_episode(summary: str = Body(...)):
    """End current episodic memory context."""
    s = check_sully()
    
    try:
        if not hasattr(s, "memory_integration") or not s.memory_integration:
            return {"status": "error", "message": "Memory integration not enabled"}
        
        s.memory_integration.end_episode(summary)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End episode error: {str(e)}")

@app.post("/memory/store")
async def memory_store_interaction(request: StoreInteractionRequest):
    """Store an interaction in memory."""
    s = check_sully()
    
    try:
        if not hasattr(s, "memory_integration") or not s.memory_integration:
            return {"status": "error", "message": "Memory integration not enabled"}
        
        s.memory_integration.store_interaction(
            request.query,
            request.response,
            request.interaction_type,
            request.metadata
        )
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store interaction error: {str(e)}")

@app.post("/memory/recall")
async def memory_recall(
    query: str = Body(...),
    limit: int = Body(5),
    include_emotional: bool = Body(True),
    module: Optional[str] = Body(None)
):
    """Recall relevant memories for a query."""
    s = check_sully()
    
    try:
        if not hasattr(s, "memory_integration") or not s.memory_integration:
            # Fall back to basic memory search
            return {"memories": s.search_memory(query, limit)}
        
        kwargs = {
            "query": query,
            "limit": limit,
            "include_emotional": include_emotional
        }
        
        if module:
            kwargs["module"] = module
            
        memories = s.memory_integration.recall(**kwargs)
        return {"memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory recall error: {str(e)}")

@app.post("/memory/store_experience")
async def memory_store_experience(request: StoreExperienceRequest):
    """Store general knowledge in memory."""
    s = check_sully()
    
    try:
        if not hasattr(s, "memory_integration") or not s.memory_integration:
            # Fall back to basic memory
            s.remember(request.content)
            return {"status": "basic_storage", "message": "Stored in basic memory system"}
        
        s.memory_integration.store_experience(
            content=request.content,
            source=request.source,
            importance=request.importance,
            emotional_tags=request.emotional_tags,
            concepts=request.concepts
        )
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store experience error: {str(e)}")

@app.post("/chat/with_memory")
async def chat_with_memory(message: str = Body(...), tone: CognitiveMode = Body(CognitiveMode.emergent)):
    """Process chat with memory enhancement."""
    s = check_sully()
    
    try:
        if hasattr(s, "process_with_memory"):
            response = s.process_with_memory(message)
            return {"response": response, "memory_enhanced": True}
        
        # Fall back to regular chat
        response = s.process(message)
        return {"response": response, "memory_enhanced": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory-enhanced chat error: {str(e)}")

@app.post("/process/integrated")
async def process_integrated(
    message: str = Body(...),
    tone: CognitiveMode = Body(CognitiveMode.emergent),
    retrieve_memories: bool = Body(True),
    cross_modal: bool = Body(False)
):
    """Multi-modal processing with memory."""
    s = check_sully()
    
    try:
        response_parts = {}
        
        # Step 1: Retrieve relevant memories if requested
        memories = []
        if retrieve_memories and hasattr(s, "search_memory"):
            memories = s.search_memory(message, limit=3)
            response_parts["memories"] = memories
        
        # Step 2: Process the message
        if hasattr(s, "process_with_memory") and retrieve_memories:
            response = s.process_with_memory(message)
        else:
            response = s.process(message)
        
        response_parts["response"] = response
        
        # Step 3: Add cross-modal processing if requested
        if cross_modal and hasattr(s, "kernel_integration") and s.kernel_integration:
            # Try to get cross-kernel insights
            if hasattr(s.kernel_integration, "dynamic_process"):
                insight = s.kernel_integration.dynamic_process(message, "insight")
                response_parts["cross_modal_insight"] = insight
            
            # Try to add dream perspective
            if hasattr(s, "dream"):
                dream_sequence = s.dream(message, "shallow", "symbolic")
                response_parts["dream_perspective"] = dream_sequence
        
        return response_parts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integrated processing error: {str(e)}")

# --- Identity & Personality Routes ---

@app.get("/speak_identity")
async def speak_identity():
    """Express Sully's sense of self."""
    s = check_sully()
    
    try:
        result = s.speak_identity()
        return {"identity_expression": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity expression error: {str(e)}")

@app.post("/personas/custom")
async def personas_custom(request: PersonaRequest):
    """Create custom persona."""
    s = check_sully()
    
    try:
        # Check if identity module has the create_persona method
        if hasattr(s, "identity") and hasattr(s.identity, "create_persona"):
            result = s.identity.create_persona(
                request.name,
                request.traits,
                request.description
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "error",
            "message": "Custom persona creation not available in current identity system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom persona creation error: {str(e)}")

@app.post("/personas/composite")
async def personas_composite(request: BlendedPersonaRequest):
    """Create blended persona."""
    s = check_sully()
    
    try:
        # Check if identity module has the blend_personas method
        if hasattr(s, "identity") and hasattr(s.identity, "blend_personas"):
            result = s.identity.blend_personas(
                request.name,
                request.personas,
                request.weights,
                request.description
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "error",
            "message": "Persona blending not available in current identity system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blended persona creation error: {str(e)}")

@app.get("/personas")
async def personas_list():
    """List available personas."""
    s = check_sully()
    
    try:
        # Check if identity module has the get_personas method
        if hasattr(s, "identity") and hasattr(s.identity, "get_personas"):
            result = s.identity.get_personas()
            return {"personas": result}
        
        # Fallback response if method not available
        return {
            "personas": ["default"],
            "message": "Extended persona system not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona listing error: {str(e)}")

@app.post("/identity/evolve")
async def identity_evolve(request: EvolveIdentityRequest):
    """Evolve personality traits based on interactions."""
    s = check_sully()
    
    try:
        result = s.evolve_identity(request.interactions, request.learning_rate)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity evolution error: {str(e)}")

@app.post("/identity/adapt")
async def identity_adapt(request: AdaptIdentityRequest):
    """Adapt identity to specific context."""
    s = check_sully()
    
    try:
        result = s.adapt_identity_to_context(request.context, request.context_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity adaptation error: {str(e)}")

@app.get("/identity/profile")
async def identity_profile(detailed: bool = Query(False)):
    """Get comprehensive personality profile."""
    s = check_sully()
    
    try:
        result = s.get_identity_profile(detailed)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity profile error: {str(e)}")

@app.post("/identity/generate_persona")
async def identity_generate_persona(request: DynamicPersonaRequest):
    """Dynamically generate a context-specific persona."""
    s = check_sully()
    
    try:
        persona_id, description = s.generate_dynamic_persona(
            request.context_query,
            request.principles,
            request.traits
        )
        
        return {
            "persona_id": persona_id,
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona generation error: {str(e)}")

@app.get("/identity/map")
async def identity_map():
    """Get a comprehensive multi-level map of Sully's identity."""
    s = check_sully()
    
    try:
        result = s.create_identity_map()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity mapping error: {str(e)}")

@app.post("/transform")
async def transform(request: TransformRequest):
    """Transform a response according to a specific cognitive mode or persona."""
    s = check_sully()
    
    try:
        result = s.transform_response(
            request.content,
            request.mode,
            request.context_data
        )
        
        return {
            "original": request.content,
            "transformed": result,
            "mode": request.mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation error: {str(e)}")

# --- Emergent Systems Routes ---

@app.post("/emergence/detect")
async def emergence_detect(request: EmergenceDetectionRequest):
    """Detect emergent patterns in cognitive system."""
    s = check_sully()
    
    try:
        # Check if emergence framework has the detect_emergence method
        if hasattr(s, "emergence") and hasattr(s.emergence, "detect_emergence"):
            result = s.emergence.detect_emergence(
                request.module_interactions,
                request.threshold
            )
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            "Analyze the current system state for emergent cognitive patterns",
            "analytical"
        )
        
        return {
            "emergent_patterns": [],
            "analysis": insight,
            "note": "Generated through reasoning, not formal emergence detection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergence detection error: {str(e)}")

@app.get("/emergence/properties")
async def emergence_properties():
    """View detected emergent properties."""
    s = check_sully()
    
    try:
        # Check if emergence framework has the get_properties method
        if hasattr(s, "emergence") and hasattr(s.emergence, "get_properties"):
            result = s.emergence.get_properties()
            return result
        
        # Fallback response if method not available
        return {
            "properties": [],
            "message": "Formal emergence property detection not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergence properties error: {str(e)}")

# --- Learning & Knowledge Routes ---

@app.post("/learning/process")
async def learning_process(interaction: Dict[str, Any] = Body(...)):
    """Process interaction for learning."""
    s = check_sully()
    
    try:
        # Check if learning system has process_interaction method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "process_interaction"):
            s.continuous_learning.process_interaction(interaction)
            return {"status": "success"}
        
        # Fallback to basic memory
        s.remember(str(interaction))
        return {"status": "basic_storage", "message": "Stored in basic memory system"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning process error: {str(e)}")

@app.post("/learning/consolidate")
async def learning_consolidate():
    """Consolidate experience into knowledge."""
    s = check_sully()
    
    try:
        # Check if learning system has consolidate_knowledge method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "consolidate_knowledge"):
            result = s.continuous_learning.consolidate_knowledge()
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            "Synthesize and consolidate recent experiences into deeper understanding",
            "analytical"
        )
        
        return {
            "consolidated_insights": insight,
            "note": "Generated through reasoning, not formal knowledge consolidation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge consolidation error: {str(e)}")

@app.get("/learning/statistics")
async def learning_statistics():
    """Get learning statistics."""
    s = check_sully()
    
    try:
        # Check if learning system has get_statistics method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "get_statistics"):
            result = s.continuous_learning.get_statistics()
            return result
        
        # Fallback basic stats
        return {
            "status": "limited",
            "message": "Detailed learning statistics not available",
            "knowledge_items": len(s.knowledge) if hasattr(s, "knowledge") else "unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning statistics error: {str(e)}")

@app.get("/codex/search")
async def codex_search(term: str = Query(...), limit: int = Query(10)):
    """Search knowledge codex."""
    s = check_sully()
    
    try:
        # Check if codex has search method
        if hasattr(s, "codex") and hasattr(s.codex, "search"):
            result = s.codex.search(term)
            
            # Limit results if needed
            if isinstance(result, dict) and len(result) > limit:
                result = dict(list(result.items())[:limit])
                
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            f"Share knowledge about the concept: {term}",
            "analytical"
        )
        
        return {
            term: {
                "definition": insight,
                "note": "Generated through reasoning, not from formal codex"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Codex search error: {str(e)}")

# --- Autonomous Goals Routes ---

@app.post("/goals/establish")
async def goals_establish(request: GoalRequest):
    """Establish new autonomous goal."""
    s = check_sully()
    
    try:
        # Check if goals system has establish_goal method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "establish_goal"):
            result = s.autonomous_goals.establish_goal(
                request.goal,
                request.priority,
                request.domain,
                request.deadline
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "acknowledged",
            "goal": request.goal,
            "message": "Goal acknowledged but autonomous goals system not fully available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Goal establishment error: {str(e)}")

@app.get("/goals/active")
async def goals_active():
    """View active goals."""
    s = check_sully()
    
    try:
        # Check if goals system has get_active_goals method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "get_active_goals"):
            result = s.autonomous_goals.get_active_goals()
            return {"goals": result}
        
        # Fallback response if method not available
        return {
            "goals": [],
            "message": "Autonomous goals system not fully available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Active goals error: {str(e)}")

@app.post("/interests/register")
async def interests_register(request: InterestRequest):
    """Register topic engagement."""
    s = check_sully()
    
    try:
        # Check if goals system has register_interest method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "register_interest"):
            result = s.autonomous_goals.register_interest(
                request.topic,
                request.engagement_level,
                request.context
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "acknowledged",
            "topic": request.topic,
            "message": "Interest acknowledged but autonomous goals system not fully available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interest registration error: {str(e)}")

# --- Visual Cognition Routes ---

@app.post("/visual/process")
async def visual_process(request: VisualProcessRequest):
    """Process and understand images."""
    s = check_sully()
    
    try:
        # Check if visual cognition system is available
        if hasattr(s, "visual_cognition") and hasattr(s.visual_cognition, "process_image"):
            result = s.visual_cognition.process_image(
                request.image_path,
                request.analysis_depth,
                request.include_objects,
                request.include_scene
            )
            return result
        
        # Fallback for simple image extraction from PDFs
        if request.image_path.lower().endswith(".pdf") and hasattr(s, "extract_images_from_pdf"):
            temp_dir = f"temp_img_{uuid.uuid4().hex[:8]}"
            extraction_result = s.extract_images_from_pdf(request.image_path, temp_dir)
            
            return {
                "status": "limited",
                "message": "Full visual cognition not available, extracted images from PDF",
                "extraction_result": extraction_result
            }
        
        # Complete fallback if no visual processing available
        return {
            "status": "error",
            "message": "Visual processing not available in current system configuration"
        }
    except Exception as e: