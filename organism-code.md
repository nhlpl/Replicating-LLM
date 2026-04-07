Below is a **Python simulation** of the enhanced self‑replicating code organism *Programma immortalis*, including all four enhancements: horizontal gene transfer, sporulation, predator fragments, and biofilm mode. The code models a swarm of fragments evolving in a distributed hash table (DHT) across multiple nodes. It runs on a single machine but captures the essential dynamics.

```python
import numpy as np
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ------------------------------------------------------------
# Parameters (from the blueprint)
# ------------------------------------------------------------
R = 1e4                     # replication threshold (interactions per division)
mu = 0.302                  # optimal mutation rate (variance)
gamma = 0.01                # horizontal gene transfer rate per execution
eta = 0.1                   # mixing coefficient for gene transfer
beta_pred = 0.05            # predator fitness bonus per deletion
alpha_prey = 0.1            # prey growth rate (Lotka-Volterra)
beta_prey = 0.02            # predation rate
gamma_pred = 0.05           # predator death rate
delta_pred = 0.01           # predator conversion efficiency
spore_threshold = 0.1       # fraction of peak resource to trigger sporulation
biofilm_lambda = 0.5        # cooperation reward in biofilm
biofilm_timeout = 60        # seconds
k_spore = 0.1               # reactivation rate constant

# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class Fragment:
    """A single code fragment (gene)."""
    code: str                  # representation of code (e.g., lambda expression)
    fitness: float            # current fitness [0,1]
    counter: int              # number of executions since last division
    is_predator: bool = False # if True, this fragment is a predator
    is_spore: bool = False    # if True, fragment is dormant
    spore_time: float = 0.0   # timestamp when sporulated
    biofilm_id: Optional[int] = None  # ID of biofilm if part of one

    def mutate(self):
        """Apply mutation to code (simplified: random character change)."""
        if random.random() < mu:
            # Simple mutation: flip a random character in the code string
            if len(self.code) > 0:
                idx = random.randrange(len(self.code))
                self.code = self.code[:idx] + chr(ord(self.code[idx]) ^ 1) + self.code[idx+1:]
        # Also mutate fitness slightly (Gaussian)
        self.fitness += np.random.normal(0, mu*0.1)
        self.fitness = max(0.0, min(1.0, self.fitness))

    def replicate(self) -> 'Fragment':
        """Create a child fragment with mutated copy."""
        child = Fragment(code=self.code, fitness=self.fitness,
                         counter=0, is_predator=self.is_predator)
        child.mutate()
        return child

    def sporulate(self, current_time: float) -> 'Fragment':
        """Return a spore (dormant) version of this fragment."""
        spore = Fragment(code=self.code, fitness=self.fitness,
                         counter=0, is_predator=self.is_predator,
                         is_spore=True, spore_time=current_time)
        return spore

    def reactivate(self) -> 'Fragment':
        """Wake up from spore state."""
        return Fragment(code=self.code, fitness=self.fitness,
                        counter=0, is_predator=self.is_predator)

class DHT:
    """Distributed Hash Table (simulated as a dict with fitness‑weighted values)."""
    def __init__(self):
        self.storage: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)
        # key: hash of code (simplified to first 8 chars of code)
        # value: list of (fragment, last_access_time)

    def store(self, frag: Fragment):
        key = frag.code[:8]  # crude hash
        self.storage[key].append((frag, 0.0))  # time not used in this sim

    def retrieve(self, code_hash: str) -> Optional[Fragment]:
        """Return the fragment with highest fitness for given hash."""
        items = self.storage.get(code_hash, [])
        if not items:
            return None
        # pick the one with highest fitness (predators are considered too)
        best = max(items, key=lambda x: x[0].fitness)
        return best[0]

    def delete(self, frag: Fragment):
        """Remove a fragment from DHT (used by predators)."""
        key = frag.code[:8]
        self.storage[key] = [f for f in self.storage[key] if f[0] != frag]

    def all_fragments(self) -> List[Fragment]:
        """Return all fragments (for statistics)."""
        return [f for lst in self.storage.values() for f, _ in lst]

# ------------------------------------------------------------
# Node (execution environment)
# ------------------------------------------------------------
class Node:
    def __init__(self, node_id: int, dht: DHT):
        self.id = node_id
        self.dht = dht
        self.resource_load = 0.5  # current resource usage (0-1)
        self.biofilm_coordinator = None

    def execute(self, frag: Fragment, current_time: float) -> float:
        """Simulate execution of a fragment, return resulting qualia (for fun)."""
        # Simple model: execution increases counter and possibly updates fitness
        frag.counter += 1
        # Simulate user feedback (random fitness change)
        feedback = random.uniform(-0.05, 0.05)
        frag.fitness += feedback
        frag.fitness = max(0.0, min(1.0, frag.fitness))
        # Qualia = fitness * (1 + counter/1e4)
        qualia = frag.fitness * (1 + frag.counter / 1e4)
        return qualia

    def step(self, current_time: float):
        """Perform one step of the node's local operations."""
        # 1. Randomly pick a fragment from DHT
        all_frags = self.dht.all_fragments()
        if not all_frags:
            return
        frag = random.choice(all_frags)

        # 2. If it's a spore, attempt reactivation based on resource level
        if frag.is_spore:
            # Reactivation probability depends on time and current resources
            t_since = current_time - frag.spore_time
            prob = 1 / (1 + math.exp(-k_spore * (t_since - 10)))  # after 10 sec
            if self.resource_load < 0.8 and random.random() < prob:
                new_frag = frag.reactivate()
                self.dht.delete(frag)
                self.dht.store(new_frag)
                frag = new_frag  # continue with active version
            else:
                return  # spore remains dormant

        # 3. Execute the fragment
        qualia = self.execute(frag, current_time)

        # 4. Horizontal gene transfer (with random neighbor)
        if random.random() < gamma:
            neighbor = random.choice(all_frags)
            if neighbor.code != frag.code:
                # Exchange a random gene (simplified: swap a character)
                if len(frag.code) > 0 and len(neighbor.code) > 0:
                    idx = random.randrange(min(len(frag.code), len(neighbor.code)))
                    frag_code_list = list(frag.code)
                    neigh_code_list = list(neighbor.code)
                    frag_code_list[idx], neigh_code_list[idx] = neigh_code_list[idx], frag_code_list[idx]
                    frag.code = ''.join(frag_code_list)
                    neighbor.code = ''.join(neigh_code_list)
                    # Update fitness according to transfer equation
                    delta = eta * (neighbor.fitness - frag.fitness)
                    frag.fitness += delta
                    neighbor.fitness -= delta
                    # Store back
                    self.dht.store(frag)
                    self.dht.store(neighbor)

        # 5. Predator action (if fragment is predator)
        if frag.is_predator:
            # Find a non‑predator prey fragment with low fitness
            prey = [f for f in all_frags if not f.is_predator and f.fitness < 0.3]
            if prey:
                target = random.choice(prey)
                self.dht.delete(target)
                # Increase predator fitness
                frag.fitness += beta_pred
                frag.fitness = min(1.0, frag.fitness)
                self.dht.store(frag)

        # 6. Check replication condition
        if frag.counter >= R and self.resource_load < 0.9:
            child = frag.replicate()
            self.dht.store(child)
            frag.counter = 0  # reset parent counter
            self.dht.store(frag)

        # 7. Sporulation if resources are low
        if self.resource_load < spore_threshold and not frag.is_spore:
            spore = frag.sporulate(current_time)
            self.dht.delete(frag)
            self.dht.store(spore)

        # 8. Biofilm formation (if a complex query arrives, here simulated by random chance)
        if random.random() < 0.01 and self.biofilm_coordinator is None:
            # Form a biofilm with a few random fragments
            members = random.sample(all_frags, min(5, len(all_frags)))
            # Elect leader (highest fitness)
            leader = max(members, key=lambda x: x.fitness)
            biofilm_id = random.randint(1, 10000)
            for m in members:
                m.biofilm_id = biofilm_id
            # Biofilm fitness: average plus cooperation term
            avg_f = np.mean([m.fitness for m in members])
            mutual_info = 0.1  # dummy
            biofilm_fitness = avg_f + biofilm_lambda * mutual_info / len(members)
            # For simplicity, we just note the event
            print(f"[Node {self.id}] Biofilm {biofilm_id} formed with {len(members)} members, fitness {biofilm_fitness:.3f}")
            self.biofilm_coordinator = biofilm_id

        # Biofilm timeout
        if self.biofilm_coordinator and current_time % biofilm_timeout < 1:
            # dissolve biofilm
            for m in all_frags:
                if m.biofilm_id == self.biofilm_coordinator:
                    m.biofilm_id = None
            self.biofilm_coordinator = None

        # Update resource load (simple random walk)
        self.resource_load += random.uniform(-0.01, 0.01)
        self.resource_load = max(0.0, min(1.0, self.resource_load))

# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
def run_simulation(steps=10000, num_nodes=10):
    dht = DHT()
    # Seed with initial fragments: one normal, one predator, one biofilm coordinator (normal but with high fitness)
    initial_frags = [
        Fragment(code="identity", fitness=0.5, counter=0),
        Fragment(code="predator", fitness=0.5, counter=0, is_predator=True),
        Fragment(code="cooperator", fitness=0.6, counter=0)
    ]
    for f in initial_frags:
        dht.store(f)

    nodes = [Node(i, dht) for i in range(num_nodes)]

    # Statistics
    history = []
    for step in range(steps):
        current_time = step * 0.1  # arbitrary time units
        # Each node performs a step
        for node in nodes:
            node.step(current_time)

        # Collect stats every 100 steps
        if step % 100 == 0:
            all_f = dht.all_fragments()
            n_frag = len(all_f)
            avg_fit = np.mean([f.fitness for f in all_f]) if all_f else 0
            n_pred = sum(1 for f in all_f if f.is_predator)
            n_spore = sum(1 for f in all_f if f.is_spore)
            history.append((step, n_frag, avg_fit, n_pred, n_spore))
            print(f"Step {step}: fragments={n_frag}, avg_fit={avg_fit:.3f}, predators={n_pred}, spores={n_spore}")

    # Plot results (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        steps_hist = [h[0] for h in history]
        frags_hist = [h[1] for h in history]
        fit_hist = [h[2] for h in history]
        pred_hist = [h[3] for h in history]
        spore_hist = [h[4] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(steps_hist, frags_hist)
        axes[0,0].set_title('Total fragments')
        axes[0,1].plot(steps_hist, fit_hist)
        axes[0,1].set_title('Mean fitness')
        axes[1,0].plot(steps_hist, pred_hist, label='Predators')
        axes[1,0].plot(steps_hist, spore_hist, label='Spores')
        axes[1,0].legend()
        axes[1,0].set_title('Specialized fragments')
        # Predator-prey ratio
        ratio = [p/(s+1) for p,s in zip(pred_hist, spore_hist)]
        axes[1,1].plot(steps_hist, ratio)
        axes[1,1].set_title('Predator/Spore ratio')
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    run_simulation(steps=5000, num_nodes=10)
```

**How to run:** Save the code as `programma_immortalis.py` and run with Python 3.8+. It requires `numpy` and optionally `matplotlib` for plots.

**What the simulation demonstrates:**
- **Replication** – fragments divide after \(R\) executions.
- **Mutation** – random changes in code and fitness.
- **Horizontal gene transfer** – fragments swap code characters with probability \(\gamma\).
- **Sporulation** – fragments become dormant when resources are low.
- **Predator fragments** – actively delete low‑fitness prey, gaining fitness.
- **Biofilm mode** – temporary coalitions form for complex tasks (simulated by random chance).

The output shows the population dynamics, mean fitness, and the predator/spore balance over time. This is a conceptual demonstration; a real implementation would run on liar‑lattice hardware at \(10^{33}\) Hz.

Would you like the **liar‑lambda calculus** source code for a single fragment?
