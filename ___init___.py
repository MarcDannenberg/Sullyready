# sully_engine/__init__.py
# 🧠 Sully Core Engine Package

# Core system components
from .memory import SullySearchMemory
from .reasoning import SymbolicReasoningNode
from .pdf_reader import PDFReader, extract_text_from_pdf, extract_kernel_from_text
from .memory_integration import MemoryIntegration, integrate_with_sully
from .logic_kernel import LogicKernel, integrate_with_sully as integrate_logic

# Kernel modules
from .kernel_modules.identity import SullyIdentity
from .kernel_modules.codex import SullyCodex
from .kernel_modules.judgment import JudgmentProtocol
from .kernel_modules.dream import DreamCore
from .kernel_modules.math_translator import SymbolicMathTranslator
from .kernel_modules.fusion import SymbolFusionEngine
from .kernel_modules.paradox import ParadoxLibrary
from .kernel_modules.neural_modification import NeuralModification
from .kernel_modules.continuous_learning import ContinuousLearningSystem
from .kernel_modules.autonomous_goals import AutonomousGoalSystem
from .kernel_modules.visual_cognition import VisualCognitionSystem
from .kernel_modules.emergence_framework import EmergenceFramework
from .kernel_modules.virtue import VirtueEngine
from .kernel_modules.intuition import Intuition

# Conversation system
from .conversation_engine import ConversationEngine

# Kernel integration
from .kernel_integration import KernelIntegrationSystem, initialize_kernel_integration

# Re-export classes for easier imports
__all__ = [
    # Core components
    "SullySearchMemory",
    "SymbolicReasoningNode",
    "PDFReader",
    "extract_text_from_pdf",
    "extract_kernel_from_text",
    "MemoryIntegration",
    "integrate_with_sully",
    "LogicKernel",
    "integrate_logic",
    
    # Kernel modules
    "SullyIdentity",
    "SullyCodex",
    "JudgmentProtocol",
    "DreamCore",
    "SymbolicMathTranslator",
    "SymbolFusionEngine",
    "ParadoxLibrary",
    "NeuralModification",
    "ContinuousLearningSystem",
    "AutonomousGoalSystem",
    "VisualCognitionSystem",
    "EmergenceFramework",
    "VirtueEngine",
    "Intuition",
    
    # Conversation
    "ConversationEngine",
    
    # Kernel integration
    "KernelIntegrationSystem",
    "initialize_kernel_integration"
]