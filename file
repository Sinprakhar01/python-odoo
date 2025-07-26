import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
import json
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes, hmac
from datetime import datetime
from fpdf import FPDF
import base64

# --- Logging and diagnostics ---
LOG_FILE = "session.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Utility Functions ---
def tamper_evident_log(message: str):
    h = hmac.HMAC(b'secret_key_for_hmac', hashes.SHA256(), backend=default_backend())
    h.update(message.encode())
    signature = base64.b64encode(h.finalize()).decode()
    logging.info(f"{message} | HMAC: {signature}")

# --- DAG-based Knowledge Graph ---
class KnowledgeDAG:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.keys())

    def get_prerequisites(self, node):
        return [n for n, edges in self.graph.items() if node in edges]

# --- Transformer-based Knowledge Tracing Model ---
class SAINTModel(nn.Module):
    def __init__(self, num_questions, d_model=64, nhead=4, num_layers=2):
        super(SAINTModel, self).__init__()
        self.embedding = nn.Embedding(num_questions, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x).squeeze(-1)
        return torch.sigmoid(x)

# --- Bayesian Question Selector ---
class BayesianSelector:
    def __init__(self, prior_mean=0.5, beta=1.0):
        self.prior_mean = prior_mean
        self.beta = beta
        self.stats = {}  # question_id: [correct_count, total_count]

    def update(self, qid, correct):
        if qid not in self.stats:
            self.stats[qid] = [0, 0]
        self.stats[qid][1] += 1
        self.stats[qid][0] += correct

    def score(self, qid):
        if qid not in self.stats:
            return self.prior_mean + self.beta
        correct, total = self.stats[qid]
        p = (correct + 1) / (total + 2)
        return p + self.beta * np.sqrt(p * (1 - p) / (total + 2))

    def select_question(self, available_qids):
        scores = {qid: self.score(qid) for qid in available_qids}
        return max(scores, key=scores.get)

# --- Learning Decay ---
def apply_learning_decay(knowledge_state, decay_rate=0.05):
    return {k: v * (1 - decay_rate) for k, v in knowledge_state.items()}

# --- AES-256 Encryption ---
class AES256Encryptor:
    def __init__(self, key: bytes):
        self.key = key
        self.backend = default_backend()

    def encrypt(self, data: str) -> bytes:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded = padder.update(data.encode()) + padder.finalize()
        return iv + encryptor.update(padded) + encryptor.finalize()

    def decrypt(self, data: bytes) -> str:
        iv = data[:16]
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        padded_data = decryptor.update(data[16:]) + decryptor.finalize()
        return unpadder.update(padded_data) + unpadder.finalize()

# --- LaTeX PDF Report ---
def generate_pdf_report(session_data, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Adaptive Quiz Session Report", ln=True, align='C')
    for k, v in session_data.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True, align='L')
    pdf.output(filename)

# --- Simulated Quiz Session ---
def run_adaptive_session():
    # Setup
    dag = KnowledgeDAG({"A": ["B"], "B": ["C"], "C": []})
    model = SAINTModel(num_questions=10)
    selector = BayesianSelector()
    knowledge_state = {k: 0.5 for k in dag.nodes}
    encryptor = AES256Encryptor(key=os.urandom(32))
    session_data = {}

    # Simulate session
    for _ in range(5):
        qid = selector.select_question(dag.nodes)
        mastery = knowledge_state[qid]
        correct = int(random.random() < mastery)
        selector.update(qid, correct)
        knowledge_state[qid] = 0.9 if correct else 0.4
        knowledge_state = apply_learning_decay(knowledge_state)
        tamper_evident_log(f"QID: {qid}, Correct: {correct}, Mastery: {knowledge_state[qid]:.2f}")
        session_data[qid] = {"correct": correct, "mastery": knowledge_state[qid]}

    # Encrypt session
    encrypted = encryptor.encrypt(json.dumps(session_data))
    with open("session.enc", "wb") as f:
        f.write(encrypted)

    # Generate report
    generate_pdf_report(session_data)

# --- Run the system ---
if __name__ == "__main__":
    run_adaptive_session()
