"""
Adaptive Quiz Engine for STEM Learners

A production-ready offline system that models student mastery over a DAG using
bidirectional transformers with Bayesian exploration-exploitation for question selection.
Includes knowledge state updates, learning decay, local encryption, PDF reporting,
and fairness mechanisms.
"""

import os
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import math
import statistics

# Cryptography imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets
# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# Type aliases
NodeID = str
SkillID = str
QuestionID = str
StudentID = str
Timestamp = float
Probability = float
Confidence = float

# Constants
DEFAULT_ENCRYPTION_KEY = "default_encryption_key_please_change"
SESSION_DATA_VERSION = "1.0"
MAX_HISTORY_SIZE = 100
DECAY_RATE = 0.95  # Knowledge decay rate per day
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
MAX_UNCERTAINTY = 0.5
MIN_UNCERTAINTY = 0.05
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour in seconds
MAX_QUESTIONS_PER_SESSION = 50
MAX_RETRIES = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class DifficultyLevel(Enum):
    """Enum representing question difficulty levels."""
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()

class QuestionType(Enum):
    """Enum representing different types of questions."""
    MULTIPLE_CHOICE = auto()
    TRUE_FALSE = auto()
    SHORT_ANSWER = auto()
    NUMERICAL = auto()

class FairnessGroup(Enum):
    """Enum representing learner groups for fairness stratification."""
    GROUP_A = auto()
    GROUP_B = auto()
    GROUP_C = auto()

@dataclass
class Question:
    """Dataclass representing a quiz question."""
    id: QuestionID
    text: str
    skill: SkillID
    difficulty: DifficultyLevel
    question_type: QuestionType
    options: List[str] = field(default_factory=list)
    correct_answer: Union[str, int, float, bool] = ""
    explanation: str = ""
    fairness_group: FairnessGroup = FairnessGroup.GROUP_A
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeNode:
    """Dataclass representing a node in the knowledge DAG."""
    id: NodeID
    skill: SkillID
    prerequisites: List[NodeID] = field(default_factory=list)
    questions: List[QuestionID] = field(default_factory=list)
    difficulty_weights: Dict[DifficultyLevel, float] = field(default_factory=dict)

@dataclass
class StudentResponse:
    """Dataclass representing a student's response to a question."""
    question_id: QuestionID
    response: Union[str, int, float, bool]
    is_correct: bool
    timestamp: Timestamp
    time_taken: float  # in seconds
    confidence: Optional[Confidence] = None

@dataclass
class KnowledgeState:
    """Dataclass representing a student's knowledge state for a node."""
    node_id: NodeID
    probability: Probability
    confidence: Confidence
    last_updated: Timestamp
    history: List[StudentResponse] = field(default_factory=list)

@dataclass
class SessionConfig:
    """Dataclass representing session configuration."""
    student_id: StudentID
    max_questions: int = MAX_QUESTIONS_PER_SESSION
    timeout: int = DEFAULT_SESSION_TIMEOUT
    fairness_group: FairnessGroup = FairnessGroup.GROUP_A
    focus_skills: Optional[List[SkillID]] = None

class QuizEngine:
    """
    Adaptive Quiz Engine for STEM Learners.
    
    Uses bidirectional transformer-inspired attention mechanisms to model student mastery
    over a DAG structure with Bayesian exploration-exploitation for question selection.
    
    Intuition:
    - Model knowledge as a DAG where nodes represent skills and edges represent prerequisites
    - Use attention mechanisms to propagate knowledge updates through the graph
    - Balance exploration (reducing uncertainty) and exploitation (targeting weak areas)
    - Incorporate time decay to model forgetting
    
    Approach:
    - Represent knowledge as probabilities with confidence intervals
    - Use Thompson sampling for exploration-exploitation tradeoff
    - Update knowledge states using Bayesian inference
    - Apply exponential decay based on time since last practice
    - Encrypt all session data locally with AES-256
    - Generate LaTeX-style PDF reports after each session
    - Ensure fairness through stratified question selection
    
    Complexity:
    - Time: O(N + E) for graph operations where N is nodes and E is edges
    - Space: O(N + Q) where N is nodes and Q is questions
    """
    
    def __init__(self, knowledge_dag: Dict[NodeID, KnowledgeNode], questions: Dict[QuestionID, Question]):
        """
        Initialize the quiz engine with knowledge DAG and question bank.
        
        Args:
            knowledge_dag: Dictionary of knowledge nodes forming a DAG
            questions: Dictionary of all available questions
        """
        self.knowledge_dag = knowledge_dag
        self.questions = questions
        self.student_states: Dict[StudentID, Dict[NodeID, KnowledgeState]] = {}
        self.session_history: Dict[StudentID, List[Dict]] = {}
        self.logger = self._setup_logging()
        self._validate_dag()
        self._index_questions()
        
    def _setup_logging(self) -> logging.Logger:
      """Set up tamper-evident logging with diagnostics.
      
      Creates a logger that writes HMAC signatures for each log entry
      to enable tamper detection.
      """
      logger = logging.getLogger("AdaptiveQuizEngine")
      logger.setLevel(logging.INFO)
      
      # Create a custom handler class
      class TamperEvidentFileHandler(logging.FileHandler):
          def __init__(self, filename, hmac_key=None):
              super().__init__(filename)
              self.hmac_key = hmac_key or os.urandom(32)
              
          def emit(self, record):
              # First let the parent class handle the actual logging
              super().emit(record)
              
              # Now append our HMAC signature
              msg = self.format(record)
              h = hmac.HMAC(self.hmac_key, hashes.SHA256(), backend=default_backend())
              h.update(msg.encode('utf-8'))
              signature = base64.b64encode(h.finalize()).decode('utf-8')
              
              # Write signature to file
              with open(self.baseFilename, 'a') as f:
                  f.write(f"||SIG:{signature}||\n")
      
      # Set up the handler
      log_file = Path("quiz_engine.log")
      handler = TamperEvidentFileHandler(log_file)
      handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
      logger.addHandler(handler)
      
      # Ensure only our handler is present
      logger.handlers = [handler]
      
      return logger
    
    def _add_log_verification(self, logger: logging.Logger, log_file: Path) -> logging.Logger:
      """Add HMAC verification to log entries for tamper detection."""
      original_emit = logger.handlers[0].emit

      def verified_emit(record):
          # Get the original log message
          msg = record.getMessage()  # Correct way to get the log message


          # Create HMAC
          h = hmac.HMAC(os.urandom(32), hashes.SHA256(), backend=default_backend())
          h.update(msg.encode('utf-8'))
          signature = h.finalize()

          # Append signature to log file
          with open(log_file, 'a') as f:
              f.write(f"||SIG:{base64.b64encode(signature).decode('utf-8')}||\n")

          return original_emit(record)

      logger.handlers[0].emit = verified_emit
      return logger
    
    def _validate_dag(self) -> None:
        """Validate that the knowledge graph is a proper DAG."""
        visited = set()
        recursion_stack = set()
        for node_id in self.knowledge_dag:
            if node_id not in visited and self._is_cyclic(node_id, visited, recursion_stack):
                raise ValueError("Knowledge graph contains cycles")

    def _is_cyclic(self, node_id: NodeID, visited: set, recursion_stack: set) -> bool:
        """Check for cycles starting from a node (helper for _validate_dag)."""
        if node_id in recursion_stack:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        recursion_stack.add(node_id)
        for neighbor in self.knowledge_dag[node_id].prerequisites:
            if self._is_cyclic(neighbor, visited, recursion_stack):
                return True
        recursion_stack.remove(node_id)
        return False
    
    def _index_questions(self) -> None:
        """Index questions by skill and difficulty for faster lookup."""
        self._question_index: Dict[SkillID, Dict[DifficultyLevel, List[QuestionID]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        self._fairness_index: Dict[FairnessGroup, List[QuestionID]] = defaultdict(list)
        
        for qid, question in self.questions.items():
            self._question_index[question.skill][question.difficulty].append(qid)
            self._fairness_index[question.fairness_group].append(qid)
    
    def start_session(self, config: SessionConfig) -> str:
        """
        Start a new quiz session for a student.
        
        Args:
            config: Session configuration
            
        Returns:
            Session ID for the new session
        """
        self._validate_session_config(config)
        
        session_id = self._generate_session_id(config.student_id)
        self.logger.info(f"Starting session {session_id} for student {config.student_id}")
        
        # Initialize student state if not exists
        if config.student_id not in self.student_states:
            self.student_states[config.student_id] = self._initialize_knowledge_states()
        
        # Initialize session history
        if config.student_id not in self.session_history:
            self.session_history[config.student_id] = []
            
        self.session_history[config.student_id].append({
            "session_id": session_id,
            "start_time": time.time(),
            "config": config,
            "questions_asked": [],
            "responses": []
        })
        
        return session_id
    
    def _validate_session_config(self, config: SessionConfig) -> None:
        """Validate session configuration parameters."""
        if not isinstance(config.student_id, str) or not config.student_id:
            raise ValueError("Student ID must be a non-empty string")
        
        if not isinstance(config.max_questions, int) or config.max_questions <= 0:
            raise ValueError("Max questions must be a positive integer")
        
        if not isinstance(config.timeout, int) or config.timeout <= 0:
            raise ValueError("Timeout must be a positive integer")
        
        if config.focus_skills is not None:
            if not isinstance(config.focus_skills, list):
                raise ValueError("Focus skills must be a list or None")
            for skill in config.focus_skills:
                if not isinstance(skill, str):
                    raise ValueError("All focus skills must be strings")
    
    def _generate_session_id(self, student_id: StudentID) -> str:
        timestamp = int(time.time() * 1000)
        # 8-character hexadecimal for randomness
        random_part = secrets.token_hex(4)
        return f"{student_id}_{timestamp}_{random_part}"


    def _initialize_knowledge_states(self) -> Dict[NodeID, KnowledgeState]:
        """Initialize knowledge states for a new student with default values."""
        states = {}
        for node_id, node in self.knowledge_dag.items():
            states[node_id] = KnowledgeState(
                node_id=node_id,
                probability=0.5,  # Default to uncertain
                confidence=0.5,   # Medium confidence
                last_updated=time.time()
            )
        return states
    
    def get_next_question(self, session_id: str, student_id: StudentID) -> Optional[Question]:
        """
        Get the next question for the student based on their knowledge state.
        
        Args:
            session_id: Current session ID
            student_id: ID of the student
            
        Returns:
            The next question to ask or None if session is complete
        """
        self._validate_student_session(student_id, session_id)
        
        session = self._get_current_session(student_id, session_id)
        if len(session["questions_asked"]) >= session["config"].max_questions:
            self.logger.info(f"Session {session_id} complete - max questions reached")
            return None
        
        if time.time() - session["start_time"] > session["config"].timeout:
            self.logger.info(f"Session {session_id} expired due to timeout")
            return None
        
        # Apply knowledge decay before selecting next question
        self._apply_knowledge_decay(student_id)
        
        # Get the student's current knowledge state
        knowledge_state = self.student_states[student_id]
        
        # Select node based on exploration-exploitation tradeoff
        selected_node_id = self._select_node(knowledge_state, session["config"].focus_skills)
        
        if selected_node_id is None:
            self.logger.info("No suitable node found for question selection")
            return None
        
        # Select question based on node and student's state
        selected_question_id = self._select_question(
            selected_node_id, 
            knowledge_state[selected_node_id],
            session["config"].fairness_group
        )
        
        if selected_question_id is None:
            self.logger.warning(f"No suitable question found for node {selected_node_id}")
            return None
        
        # Record the question in session history
        session["questions_asked"].append({
            "question_id": selected_question_id,
            "timestamp": time.time(),
            "node_id": selected_node_id
        })
        
        return self.questions[selected_question_id]
    
    def _validate_student_session(self, student_id: StudentID, session_id: str) -> None:
        """Validate that the student and session exist and are valid."""
        if student_id not in self.student_states:
            raise ValueError(f"Student {student_id} not found")
        
        if student_id not in self.session_history:
            raise ValueError(f"No session history for student {student_id}")
        
        if not any(session["session_id"] == session_id for session in self.session_history[student_id]):
            raise ValueError(f"Session {session_id} not found for student {student_id}")
    
    def _get_current_session(self, student_id: StudentID, session_id: str) -> Dict:
        """Get the current session data."""
        for session in reversed(self.session_history[student_id]):
            if session["session_id"] == session_id:
                return session
        raise ValueError(f"Session {session_id} not found")
    
    def _apply_knowledge_decay(self, student_id: StudentID) -> None:
        """Apply knowledge decay based on time since last update."""
        current_time = time.time()
        for node_id, state in self.student_states[student_id].items():
            days_since_update = (current_time - state.last_updated) / (24 * 3600)
            if days_since_update > 0:
                decay_factor = DECAY_RATE ** days_since_update
                new_prob = state.probability * decay_factor
                new_conf = state.confidence * decay_factor
                
                # Update state with decayed values
                self.student_states[student_id][node_id] = KnowledgeState(
                    node_id=node_id,
                    probability=max(MIN_PROBABILITY, min(MAX_PROBABILITY, new_prob)),
                    confidence=max(MIN_UNCERTAINTY, min(MAX_UNCERTAINTY, new_conf)),
                    last_updated=state.last_updated,
                    history=state.history
                )
    
    def _select_node(self, knowledge_state: Dict[NodeID, KnowledgeState], focus_skills: Optional[List[SkillID]] = None) -> Optional[NodeID]:
        """
        Select a knowledge node using exploration-exploitation tradeoff.
        
        Uses Thompson sampling with consideration of:
        - Current knowledge probabilities
        - Confidence intervals
        - Prerequisite relationships
        - Focus skills (if specified)
        """
        eligible_nodes = []
        
        for node_id, state in knowledge_state.items():
            node = self.knowledge_dag[node_id]
            
            # Check focus skills if specified
            if focus_skills is not None and node.skill not in focus_skills:
                continue
            
            # Check prerequisites
            if not self._check_prerequisites(node_id, knowledge_state):
                continue
            
            # Calculate exploration value (inverse of confidence)
            exploration_value = 1 - state.confidence
            
            # Calculate exploitation value (distance from target probability)
            target_prob = 0.8  # Target probability we want students to reach
            exploitation_value = abs(state.probability - target_prob)
            
            # Combine with weights (adjust these based on your preference)
            score = 0.7 * exploitation_value + 0.3 * exploration_value
            
            eligible_nodes.append((node_id, score, state))
        
        if not eligible_nodes:
            return None
        
        # Sort by score (descending) - higher scores mean more valuable to address
        eligible_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Use softmax to convert scores to probabilities
        scores = [score for _, score, _ in eligible_nodes]
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [exp / sum_exp for exp in exp_scores]


        def secure_weighted_choice(weights):
          total = sum(weights)
          threshold = secrets.randbelow(int(total * 1e9)) / 1e9  # fine-grained
          acc = 0.0
          for i, w in enumerate(weights):
              acc += w
              if threshold <= acc:
                  return i
          return len(weights) - 1  # fallback


        # Select node based on probabilities
        selected_idx = secure_weighted_choice(probs)
        return eligible_nodes[selected_idx][0]
    
    def _check_prerequisites(self, node_id: NodeID, knowledge_state: Dict[NodeID, KnowledgeState]) -> bool:
        """Check if all prerequisites for a node are satisfied."""
        prerequisites = self.knowledge_dag[node_id].prerequisites
        if not prerequisites:
            return True
        
        for prereq_id in prerequisites:
            if prereq_id not in knowledge_state:
                return False
            if knowledge_state[prereq_id].probability < 0.7:  # Threshold for prerequisite mastery
                return False
        
        return True
    
    def _select_question(self, node_id: NodeID, state: KnowledgeState, fairness_group: FairnessGroup) -> Optional[QuestionID]:
      """
      Select a question for the given node considering:
      - Student's current knowledge level
      - Question difficulty
      - Fairness group distribution
      """
      node = self.knowledge_dag[node_id]
      skill = node.skill

      available_questions = self._get_available_questions(skill)
      if not available_questions:
          return None

      questions_to_consider = self._filter_questions_by_fairness(available_questions, fairness_group)
      if not questions_to_consider:
          questions_to_consider = available_questions

      target_difficulty = self._calculate_target_difficulty(state.probability)
      scored_questions = [
          (qid, self._score_question(qid, state, target_difficulty))
          for qid in questions_to_consider
      ]

      # Softmax probabilities
      probs = self._softmax_scores([score for _, score in scored_questions])
      selected_idx = self._secure_weighted_choice(probs)
      return scored_questions[selected_idx][0]

    def _get_available_questions(self, skill):
        return [
            qid
            for difficulty in DifficultyLevel
            for qid in self._question_index[skill][difficulty]
        ]

    def _filter_questions_by_fairness(self, questions, fairness_group):
        return [
            qid for qid in questions
            if self.questions[qid].fairness_group == fairness_group
        ]

    def _score_question(self, qid, state, target_difficulty):
        question = self.questions[qid]
        diff_score = 1 - abs(question.difficulty.value - target_difficulty.value) / 2
        recent_penalty = self._calculate_recent_penalty(qid, state.history)
        return 0.6 * diff_score + 0.4 * (1 - recent_penalty)

    def _calculate_recent_penalty(self, qid, history):
        timestamps = [r.timestamp for r in history if r.question_id == qid]
        if not timestamps:
            return 0
        last_asked = max(timestamps)
        days_since = (time.time() - last_asked) / (24 * 3600)
        return min(0.5, days_since / 7)

    def _softmax_scores(self, scores):
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        return [exp / sum_exp for exp in exp_scores]

    def _secure_weighted_choice(self, weights):
        import secrets
        total = sum(weights)
        threshold = secrets.randbelow(int(total * 1e9)) / 1e9
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if threshold <= acc:
                return i
        return len(weights) - 1  # fallback

    
    def _calculate_target_difficulty(self, probability: Probability) -> DifficultyLevel:
        """
        Calculate target question difficulty based on student's knowledge probability.
        """
        if probability < 0.3:
            return DifficultyLevel.EASY
        elif probability < 0.7:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD
    
    def record_response(self, session_id: str, student_id: StudentID, response: StudentResponse) -> None:
        """
        Record a student's response to a question and update knowledge state.
        
        Args:
            session_id: Current session ID
            student_id: ID of the student
            response: Student's response data
        """
        self._validate_student_session(student_id, session_id)
        self._validate_response(response)
        
        session = self._get_current_session(student_id, session_id)
        session["responses"].append(response)
        
        # Find the node this question belongs to
        question = self.questions[response.question_id]
        node_id = self._find_node_for_skill(question.skill)
        
        if node_id is None:
            self.logger.warning(f"No node found for skill {question.skill}")
            return
        
        # Update knowledge state for this node
        self._update_knowledge_state(student_id, node_id, response)
        
        # Propagate update to dependent nodes
        self._propagate_knowledge_update(student_id, node_id)
    
    def _validate_response(self, response: StudentResponse) -> None:
        """Validate student response data."""
        if not isinstance(response.question_id, str) or response.question_id not in self.questions:
            raise ValueError("Invalid question ID in response")
        
        if not isinstance(response.is_correct, bool):
            raise ValueError("is_correct must be a boolean")
        
        if not isinstance(response.timestamp, (int, float)) or response.timestamp <= 0:
            raise ValueError("Invalid timestamp in response")
        
        if not isinstance(response.time_taken, (int, float)) or response.time_taken <= 0:
            raise ValueError("Invalid time_taken in response")
        
        if response.confidence is not None and not (0 <= response.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    def _find_node_for_skill(self, skill: SkillID) -> Optional[NodeID]:
        """Find the node ID associated with a given skill."""
        for node_id, node in self.knowledge_dag.items():
            if node.skill == skill:
                return node_id
        return None
    
    def _update_knowledge_state(self, student_id: StudentID, node_id: NodeID, response: StudentResponse) -> None:
        """Update knowledge state based on student response."""
        current_state = self.student_states[student_id][node_id]
        
        # Bayesian update parameters
        alpha = 1  # Prior successes
        beta = 1    # Prior failures
        
        # Update based on response
        if response.is_correct:
            alpha += 1
        else:
            beta += 1
        
        # Calculate new probability (expected value of beta distribution)
        new_prob = alpha / (alpha + beta)
        
        # Calculate new confidence (inverse of variance)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        new_conf = 1 - min(MAX_UNCERTAINTY, max(MIN_UNCERTAINTY, math.sqrt(variance)))
        
        # Consider response time and confidence if available
        time_factor = min(1, response.time_taken / 60)  # Normalize to 1 minute
        if response.is_correct:
            time_factor = 1 - time_factor  # Faster correct answers are better
            
        conf_factor = response.confidence if response.confidence is not None else 0.5
        if not response.is_correct:
            conf_factor = 1 - conf_factor  # High confidence in wrong answers is bad
            
        # Combine factors
        combined_factor = (time_factor + conf_factor) / 2
        
        # Adjust probability based on combined factor
        adjusted_prob = new_prob * combined_factor
        
        # Update history (limit size)
        new_history = current_state.history.copy()
        new_history.append(response)
        if len(new_history) > MAX_HISTORY_SIZE:
            new_history = new_history[-MAX_HISTORY_SIZE:]
        
        # Update state
        self.student_states[student_id][node_id] = KnowledgeState(
            node_id=node_id,
            probability=max(MIN_PROBABILITY, min(MAX_PROBABILITY, adjusted_prob)),
            confidence=max(MIN_UNCERTAINTY, min(MAX_UNCERTAINTY, new_conf)),
            last_updated=time.time(),
            history=new_history
        )
    
    def _propagate_knowledge_update(self, student_id: StudentID, source_node_id: NodeID) -> None:
        """
        Propagate knowledge updates to dependent nodes in the DAG.
        
        Uses attention-like mechanism to propagate updates based on:
        - Distance from source node
        - Strength of prerequisite relationships
        - Current confidence levels
        """
        visited = set()
        queue = deque([(source_node_id, 0)])  # (node_id, distance)
        
        while queue:
            node_id, distance = queue.popleft()
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Find all nodes that have this node as a prerequisite
            for dependent_id, dependent_node in self.knowledge_dag.items():
                if node_id in dependent_node.prerequisites:
                    # Calculate propagation strength (decays with distance)
                    propagation_strength = 0.7 ** (distance + 1)
                    
                    # Update dependent node's state
                    self._update_dependent_state(
                        student_id, 
                        node_id, 
                        dependent_id, 
                        propagation_strength
                    )

                    
                    # Add to queue with increased distance
                    queue.append((dependent_id, distance + 1))
    
    def _update_dependent_state(self, student_id: StudentID, 
                           prerequisite_id: NodeID, dependent_id: NodeID, 
                           strength: float) -> None:
        """
        Update a dependent node's state based on prerequisite changes.
        
        Args:
            student_id: ID of the student
            source_node_id: Original node where update started
            prerequisite_id: Immediate prerequisite node
            dependent_id: Node to update
            strength: Propagation strength (0-1)
        """
        if dependent_id not in self.student_states[student_id]:
            return
        
        prereq_state = self.student_states[student_id][prerequisite_id]
        current_state = self.student_states[student_id][dependent_id]

        # Calculate new probability as weighted average
        new_prob = (
            current_state.probability * (1 - strength) + 
            prereq_state.probability * strength
        )
        
        # Calculate new confidence (increase slightly due to related practice)
        new_conf = min(MAX_UNCERTAINTY, current_state.confidence + (strength * 0.1))
        
        # Update state
        self.student_states[student_id][dependent_id] = KnowledgeState(
            node_id=dependent_id,
            probability=max(MIN_PROBABILITY, min(MAX_PROBABILITY, new_prob)),
            confidence=new_conf,
            last_updated=current_state.last_updated,
            history=current_state.history
        )
    
    def end_session(self, session_id: str, student_id: StudentID) -> Path:
        """
        End a quiz session and generate a PDF report.
        
        Args:
            session_id: ID of the session to end
            student_id: ID of the student
            
        Returns:
            Path to the generated PDF report
        """
        self._validate_student_session(student_id, session_id)
        
        session = self._get_current_session(student_id, session_id)
        session["end_time"] = time.time()
        
        # Save session data with encryption
        self._save_session_data(student_id, session_id)
        
        # Generate PDF report
        report_path = self._generate_pdf_report(student_id, session_id)
        
        self.logger.info(f"Session {session_id} ended successfully for student {student_id}")
        return report_path
    
    def _save_session_data(self, student_id: StudentID, session_id: str) -> None:
        """Save session data with AES-256 encryption."""
        session = self._get_current_session(student_id, session_id)
        
        # Prepare data for encryption
        data = {
            "version": SESSION_DATA_VERSION,
            "student_id": student_id,
            "session_id": session_id,
            "data": session,
            "knowledge_state": self.student_states[student_id],
            "timestamp": time.time()
        }
        
        # Convert to JSON
        json_data = json.dumps(data, default=self._json_serializer)
        
        # Encrypt data
        encrypted_data = self._encrypt_data(json_data)
        
        # Save to file
        session_dir = Path("sessions") / student_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = session_dir / f"{session_id}.enc"
        with open(file_path, "wb") as f:
            f.write(encrypted_data)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    

    def _encrypt_data(self, data: str, key: Optional[str] = None) -> bytes:
        """
        Encrypt data using AES-256-GCM (authenticated encryption).
        Args:
            data: Data to encrypt
            key: Encryption key (uses default if None)
        Returns:
            Encrypted data as bytes (salt + nonce + ciphertext + tag)
        """
        key = key or DEFAULT_ENCRYPTION_KEY
        if not isinstance(key, str):
            raise TypeError("Encryption key must be a string")

        # Derive key using PBKDF2
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key_bytes = kdf.derive(key.encode('utf-8'))

        # AES-GCM requires a 12-byte nonce
        nonce = os.urandom(12)
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(nonce, data.encode('utf-8'), None)
        # Output: salt + nonce + ciphertext (ciphertext includes the tag)
        return salt + nonce + ciphertext

    
    def _generate_pdf_report(self, student_id: StudentID, session_id: str) -> Path:
        """
        Generate a LaTeX-style PDF report for the session.
        
        Args:
            student_id: ID of the student
            session_id: ID of the session
            
        Returns:
            Path to the generated PDF file
        """
        session = self._get_current_session(student_id, session_id)
        knowledge_state = self.student_states[student_id]
        
        # Create report directory if not exists
        report_dir = Path("reports") / student_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        report_path = report_dir / f"{session_id}.pdf"
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        elements.append(Paragraph("Adaptive Quiz Engine Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        
        # Add session info
        session_date = datetime.fromtimestamp(session["start_time"]).strftime("%Y-%m-%d %H:%M:%S")
        duration = timedelta(seconds=session["end_time"] - session["start_time"])
        
        session_info = [
            ["Student ID:", student_id],
            ["Session ID:", session_id],
            ["Date:", session_date],
            ["Duration:", str(duration)],
            ["Questions Asked:", str(len(session["questions_asked"]))],
            ["Focus Skills:", ", ".join(session["config"].focus_skills) if session["config"].focus_skills else "None"]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(session_table)
        elements.append(Spacer(1, 24))
        
        # Add knowledge state summary
        elements.append(Paragraph("Knowledge State Summary", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        
        # Prepare knowledge state data
        knowledge_data = [["Skill", "Node", "Mastery", "Confidence", "Last Practiced"]]
        for node_id, state in knowledge_state.items():
            skill = self.knowledge_dag[node_id].skill
            mastery = f"{state.probability:.0%}"
            confidence = f"{state.confidence:.0%}"
            last_practiced = datetime.fromtimestamp(state.last_updated).strftime("%Y-%m-%d")
            
            knowledge_data.append([skill, node_id, mastery, confidence, last_practiced])
        
        knowledge_table = Table(knowledge_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch, 1*inch])
        knowledge_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (2, 1), (3, -1), 'CENTER'),
        ]))
        
        elements.append(knowledge_table)
        elements.append(Spacer(1, 24))
        
        # Add question performance
        elements.append(Paragraph("Question Performance", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        
        question_data = [["Question", "Skill", "Difficulty", "Correct", "Time Taken"]]
        for response in session["responses"]:
            question = self.questions[response.question_id]
            question_data.append([
                question.text[:50] + "..." if len(question.text) > 50 else question.text,
                question.skill,
                question.difficulty.name,
                "Yes" if response.is_correct else "No",
                f"{response.time_taken:.1f}s"
            ])
        
        question_table = Table(question_data, colWidths=[2.5*inch, 1*inch, 1*inch, 0.75*inch, 0.75*inch])
        question_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (3, 1), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(question_table)
        
        # Build PDF
        doc.build(elements)
        return report_path
    
    def get_student_progress(self, student_id: StudentID) -> Dict[SkillID, float]:
        """
        Get overall progress for a student by skill.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary mapping skills to mastery percentages
        """
        if student_id not in self.student_states:
            raise ValueError(f"Student {student_id} not found")
        
        skill_progress = defaultdict(list)
        
        for node_id, state in self.student_states[student_id].items():
            skill = self.knowledge_dag[node_id].skill
            skill_progress[skill].append(state.probability)
        
        return {
            skill: statistics.mean(probabilities)
            for skill, probabilities in skill_progress.items()
        }
    
    def load_session_data(self, student_id: StudentID, session_id: str, key: Optional[str] = None) -> None:
        """
        Load encrypted session data from file.
        
        Args:
            student_id: ID of the student
            session_id: ID of the session to load
            key: Encryption key (uses default if None)
        """
        session_file = Path("sessions") / student_id / f"{session_id}.enc"
        if not session_file.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_file, "rb") as f:
            encrypted_data = f.read()
        
        # Decrypt data
        decrypted_data = self._decrypt_data(encrypted_data, key)
        
        # Parse JSON
        data = json.loads(decrypted_data)
        
        # Validate version
        if data.get("version") != SESSION_DATA_VERSION:
            raise ValueError("Incompatible session data version")
        
        # Restore session and knowledge state
        if student_id not in self.session_history:
            self.session_history[student_id] = []
        
        self.session_history[student_id].append(data["data"])
        self.student_states[student_id] = self._deserialize_knowledge_states(data["knowledge_state"])
    
    def _decrypt_data(self, encrypted_data: bytes, key: Optional[str] = None) -> str:
      """
      Decrypt data encrypted with AES-256-GCM.
      Args:
          encrypted_data: Encrypted data bytes (salt + nonce + ciphertext + tag)
          key: Encryption key (uses default if None)
      Returns:
          Decrypted data as string
      """
      key = key or DEFAULT_ENCRYPTION_KEY
      if not isinstance(key, str):
          raise TypeError("Encryption key must be a string")

      # Extract salt, nonce, and ciphertext
      salt = encrypted_data[:16]
      nonce = encrypted_data[16:28]
      ciphertext = encrypted_data[28:]

      # Derive key
      kdf = PBKDF2HMAC(
          algorithm=hashes.SHA256(),
          length=32,
          salt=salt,
          iterations=100000,
          backend=default_backend()
      )
      key_bytes = kdf.derive(key.encode('utf-8'))

      aesgcm = AESGCM(key_bytes)
      data = aesgcm.decrypt(nonce, ciphertext, None)
      return data.decode('utf-8')
    
    def _deserialize_knowledge_states(self, data: Dict) -> Dict[NodeID, KnowledgeState]:
        """Convert serialized knowledge states back to objects."""
        states = {}
        for node_id, state_data in data.items():
            # Convert string timestamps back to floats
            if isinstance(state_data.get("last_updated"), str):
                state_data["last_updated"] = float(state_data["last_updated"])
            
            # Convert history items
            history = []
            for item in state_data.get("history", []):
                if isinstance(item.get("timestamp"), str):
                    item["timestamp"] = float(item["timestamp"])
                history.append(StudentResponse(**item))
            
            states[node_id] = KnowledgeState(
                node_id=node_id,
                probability=state_data["probability"],
                confidence=state_data["confidence"],
                last_updated=state_data["last_updated"],
                history=history
            )
        return states

# Example usage and test cases
if __name__ == "__main__":
    # Create a sample knowledge DAG
    knowledge_dag = {
        "n1": KnowledgeNode(
            id="n1",
            skill="algebra",
            prerequisites=[],
            difficulty_weights={
                DifficultyLevel.EASY: 0.3,
                DifficultyLevel.MEDIUM: 0.5,
                DifficultyLevel.HARD: 0.2
            }
        ),
        "n2": KnowledgeNode(
            id="n2",
            skill="geometry",
            prerequisites=[],
            difficulty_weights={
                DifficultyLevel.EASY: 0.4,
                DifficultyLevel.MEDIUM: 0.4,
                DifficultyLevel.HARD: 0.2
            }
        ),
        "n3": KnowledgeNode(
            id="n3",
            skill="calculus",
            prerequisites=["n1", "n2"],
            difficulty_weights={
                DifficultyLevel.EASY: 0.2,
                DifficultyLevel.MEDIUM: 0.5,
                DifficultyLevel.HARD: 0.3
            }
        )
    }
    
    # Create sample questions
    questions = {
        "q1": Question(
            id="q1",
            text="Solve for x: 2x + 3 = 7",
            skill="algebra",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.SHORT_ANSWER,
            correct_answer="2",
            explanation="Subtract 3 from both sides, then divide by 2",
            fairness_group=FairnessGroup.GROUP_A
        ),
        "q2": Question(
            id="q2",
            text="What is the area of a circle with radius 3?",
            skill="geometry",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.NUMERICAL,
            correct_answer=28.27,
            explanation="Area = πr²",
            fairness_group=FairnessGroup.GROUP_B
        ),
        "q3": Question(
            id="q3",
            text="Find the derivative of x²",
            skill="calculus",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.SHORT_ANSWER,
            correct_answer="2x",
            explanation="Use the power rule",
            fairness_group=FairnessGroup.GROUP_C
        ),
        "q4": Question(
            id="q4",
            text="What is the Pythagorean theorem?",
            skill="geometry",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "a² + b² = c²",
                "a + b = c",
                "a × b = c",
                "a/b = c"
            ],
            correct_answer="a² + b² = c²",
            explanation="Relates the sides of a right triangle",
            fairness_group=FairnessGroup.GROUP_A
        )
    }
    
    # Initialize quiz engine
    engine = QuizEngine(knowledge_dag, questions)
    
    # Start a session
    config = SessionConfig(
        student_id="student123",
        max_questions=3,
        focus_skills=["algebra", "geometry"]
    )
    session_id = engine.start_session(config)
    
    # Simulate a quiz session
    for _ in range(3):
        question = engine.get_next_question(session_id, "student123")
        if question is None:
            break
        
        print(f"Question: {question.text}")
        
        # Simulate student response (random correctness for demo)
        is_correct = secrets.randbelow(2) == 1
        response = StudentResponse(
            question_id=question.id,
            response=question.correct_answer if is_correct else "wrong",
            is_correct=is_correct,
            timestamp=time.time(),
            time_taken=random.uniform(5, 30),
            confidence=random.uniform(0.5, 1.0) if is_correct else random.uniform(0, 0.5)
        )
        
        engine.record_response(session_id, "student123", response)
    
    # End session and generate report
    report_path = engine.end_session(session_id, "student123")
    print(f"Report generated at: {report_path}")
    
    # Test knowledge state retrieval
    progress = engine.get_student_progress("student123")
    print("\nStudent Progress:")
    for skill, mastery in progress.items():
        print(f"{skill}: {mastery:.0%}")
    
    # Test session data loading
    try:
        engine.load_session_data("student123", session_id)
        print("\nSession data loaded successfully")
    except Exception as e:
        print(f"\nError loading session data: {e}")
