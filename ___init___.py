# sully_engine/kernel_modules/__init__.py
# 🧠 Sully Core Cognitive Kernel Modules

from .judgment import JudgmentProtocol
from .dream import DreamCore
from .math_translator import SymbolicMathTranslator
from .fusion import SymbolFusionEngine
from .paradox import ParadoxLibrary
from .identity import SullyIdentity
from .codex import SullyCodex

# New autonomous brain modules
from .neural_modification import NeuralModification
from .continuous_learning import ContinuousLearningSystem
from .autonomous_goals import AutonomousGoalSystem
from .visual_cognition import VisualCognitionSystem
from .emergence_framework import EmergenceFramework

# Missing integrations now added
from .persona import PersonaManager
from .intuition import SomaticIntuition, SocialIntuition
from .logic_kernel import LogicKernel

# New: Games Kernel
from .games import SullyGames

__all__ = [
    "JudgmentProtocol",
    "DreamCore",
    "SymbolicMathTranslator",
    "SymbolFusionEngine",
    "ParadoxLibrary",
    "SullyIdentity",
    "SullyCodex",
    # New modules
    "NeuralModification",
    "ContinuousLearningSystem",
    "AutonomousGoalSystem",
    "VisualCognitionSystem",
    "EmergenceFramework",
    # Added modules
    "PersonaManager",
    "SomaticIntuition",
    "SocialIntuition",
    "LogicKernel",
    # Games
    "SullyGames"
]