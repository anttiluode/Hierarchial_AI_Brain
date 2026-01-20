"""
THE ALCHEMIST: INVERSE REASONING VIA DIFFERENTIABLE PHYSICS
===========================================================
Task: Factorization and Decomposition (Reverse Math).
Input: "Target: 21"
Output: "3 * 7"

Mechanism:
1. System 1 (Laboratory) is a frozen, differentiable math engine.
2. System 2 (Alchemist) generates operands (a, b).
3. We backpropagate (Target - Result) THROUGH the Physics Engine to update the Alchemist.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. SYSTEM 1: THE DIFFERENTIABLE LABORATORY (Physics)
# ============================================================================
class DifferentiablePhysics(nn.Module):
    def __init__(self, vocab=97, dim=128):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        
        # Two Circuits
        self.col_add = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*4, 
                                     batch_first=True, norm_first=True, activation='gelu'),
            num_layers=2
        )
        self.col_mult = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*4, 
                                     batch_first=True, norm_first=True, activation='gelu'),
            num_layers=2
        )
        self.head = nn.Linear(dim, vocab)

    def execute_soft(self, soft_a, soft_b, op_name):
        """
        Executes math on PROBABILITY DISTRIBUTIONS (Soft Tensors).
        This allows gradients to flow backwards.
        Input: soft_a [Batch, Vocab] (Probability of being each number)
        """
        # Materialize Soft Embeddings
        # E[x] = Sum(prob_i * emb_i)
        vec_a = torch.matmul(soft_a, self.emb.weight)
        vec_b = torch.matmul(soft_b, self.emb.weight)
        
        # Structure Input
        seq = torch.stack([vec_a, vec_b], dim=1) + self.pos
        
        # Run Physics
        if op_name == 'ADD':
            h = self.col_add(seq)
        elif op_name == 'MULT':
            h = self.col_mult(seq)
            
        # Return Logits (The result distribution)
        return self.head(h[:, -1])

# ============================================================================
# 2. SYSTEM 2: THE ALCHEMIST (Generator)
# ============================================================================
class Alchemist(nn.Module):
    def __init__(self, physics_model):
        super().__init__()
        self.physics = physics_model
        # Freeze Physics! The laws of the universe don't change.
        for p in self.physics.parameters():
            p.requires_grad = False
            
        dim = self.physics.emb.embedding_dim
        vocab = self.physics.vocab
        
        # The Alchemist's Brain
        # Input: Target Embedding
        # Output: Two Soft Distributions (a, b)
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim*2),
            nn.GELU()
        )
        
        # Two Heads: One for Operand A, One for Operand B
        self.head_a = nn.Linear(dim*2, vocab)
        self.head_b = nn.Linear(dim*2, vocab)

    def imagine(self, target_idx):
        # 1. Perceive Target
        target_emb = self.physics.emb(target_idx)
        
        # 2. Think
        hidden = self.net(target_emb)
        
        # 3. Propose Ingredients (Softmax = Differentiable Choice)
        # We use Gumbel-Softmax during training to encourage distinct numbers
        logits_a = self.head_a(hidden)
        logits_b = self.head_b(hidden)
        
        return logits_a, logits_b

# ============================================================================
# 3. TRAINING INFRASTRUCTURE
# ============================================================================
def make_physics_data(p=97):
    # Standard training data for System 1
    x_add, y_add = [], []
    x_mul, y_mul = [], []
    for a in range(p):
        for b in range(p):
            x_add.append([a, b]); y_add.append((a+b)%p)
            x_mul.append([a, b]); y_mul.append((a*b)%p)
    return (torch.tensor(x_add), torch.tensor(y_add)), (torch.tensor(x_mul), torch.tensor(y_mul))

def train_laboratory(lab, device, p=97):
    print("--- PHASE 1: BUILDING THE LABORATORY (Training Physics) ---")
    opt = optim.AdamW(lab.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    (xa, ya), (xm, ym) = make_physics_data(p)
    xa, ya, xm, ym = xa.to(device), ya.to(device), xm.to(device), ym.to(device)
    
    for epoch in range(400):
        lab.train(); opt.zero_grad()
        
        # Train Add (Using Soft Execution logic with One-Hot for robustness)
        # We simulate soft inputs to make it robust for the Alchemist later
        oh_a = F.one_hot(xa[:,0], p).float()
        oh_b = F.one_hot(xa[:,1], p).float()
        pred_a = lab.execute_soft(oh_a, oh_b, 'ADD')
        loss_a = crit(pred_a, ya)
        
        # Train Mult
        oh_ma = F.one_hot(xm[:,0], p).float()
        oh_mb = F.one_hot(xm[:,1], p).float()
        pred_m = lab.execute_soft(oh_ma, oh_mb, 'MULT')
        loss_m = crit(pred_m, ym)
        
        (loss_a + loss_m).backward()
        opt.step()
        
        if epoch % 100 == 0:
            acc_a = (pred_a.argmax(-1) == ya).float().mean()
            acc_m = (pred_m.argmax(-1) == ym).float().mean()
            print(f"Ep {epoch} | Physics Acc: Add {acc_a:.1%} | Mult {acc_m:.1%}")

def train_alchemist(alchemist, device, p=97):
    print("\n--- PHASE 2: THE ALCHEMIST (Discovering Formulas) ---")
    print("Goal: Given T, find A and B such that A * B = T")
    print("Method: Backprop through the frozen Physics Engine")
    
    # We only optimize the Alchemist
    opt = optim.AdamW(filter(lambda p: p.requires_grad, alchemist.parameters()), lr=1e-3)
    
    # Targets: We want to factorize all numbers from 0 to 96
    # Note: Prime numbers are hard (1 * P), Composites have multiple solutions.
    targets = torch.arange(p, device=device)
    
    history = []
    
    for epoch in range(1000):
        alchemist.train(); opt.zero_grad()
        
        # 1. Alchemist Proposes A and B
        logits_a, logits_b = alchemist.imagine(targets)
        
        # Use Gumbel Softmax to make it "Real" (Discrete-ish)
        # Anneal temperature
        temp = max(0.5, 2.0 - epoch * 0.002)
        soft_a = F.gumbel_softmax(logits_a, tau=temp, hard=False)
        soft_b = F.gumbel_softmax(logits_b, tau=temp, hard=False)
        
        # 2. Laboratory Checks the Result (A * B)
        # We force it to solve MULTIPLICATION (Factorization)
        # The 'result_logits' is the distribution of the product
        result_logits = alchemist.physics.execute_soft(soft_a, soft_b, 'MULT')
        
        # 3. The Loss: Does Result == Target?
        # We want the result distribution to peak at the Target index
        loss = nn.CrossEntropyLoss()(result_logits, targets)
        
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0:
            # Check success rate
            val_a = logits_a.argmax(-1)
            val_b = logits_b.argmax(-1)
            
            # Verify using real math
            real_product = (val_a * val_b) % p
            correct = (real_product == targets).float().mean()
            history.append(correct.item())
            print(f"Ep {epoch} (Temp {temp:.1f}) | Factorization Acc: {correct:.1%}")
            
            # Show a few examples
            print(f"  Target 12 -> {val_a[12].item()} * {val_b[12].item()} = {(val_a[12]*val_b[12])%p}")
            print(f"  Target 21 -> {val_a[21].item()} * {val_b[21].item()} = {(val_a[21]*val_b[21])%p}")
            print(f"  Target 42 -> {val_a[42].item()} * {val_b[42].item()} = {(val_a[42]*val_b[42])%p}")

    plt.plot(history)
    plt.title("The Alchemist: Learning to Factorize via Physics")
    plt.ylabel("Accuracy")
    plt.savefig("alchemist.png")
    print("Saved alchemist.png")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Init System 1
    lab = DifferentiablePhysics(vocab=97, dim=128).to(device)
    train_laboratory(lab, device)
    
    # 2. Init System 2
    merlin = Alchemist(lab).to(device)
    train_alchemist(merlin, device)