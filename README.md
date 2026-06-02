# hodge-belief-c

Every belief you hold has three ingredients: what you've seen, what makes sense, and what you assume. Hodge decomposition makes these three ingredients mathematically precise.

Given any belief network — a graph of relationships between propositions, each edge weighted by belief strength — Hodge decomposition splits every belief into three orthogonal components:

```
ω = exact     + co-exact     + harmonic
  = evidence  + coherence    + prior
  = δβ         + δ*γ          + h    where Δh = 0
```

This is a C11 library that performs this decomposition. It takes your belief network, builds the simplicial complex, constructs the Hodge Laplacian, and projects every belief onto these three subspaces. The result: you learn which of your beliefs come from data, which come from logical consistency, and which come from the topology of how your concepts are connected.

## The Ah-ha

When your ML model's predictions are wrong, is it because the data is insufficient, the model is incoherent, or the priors are wrong? Hodge decomposition tells you which.

Feature importance tells you *what* matters. Hodge decomposition tells you *why* it matters.

## A Concrete Example

Imagine a medical diagnosis network — five symptoms arranged in a ring (cycle), each connected to the next. You observe belief strengths along each edge: how strongly each symptom predicts the next.

```c
#include "hodge_belief.h"

/* A cycle of 5 symptoms — each connected to its neighbor */
SimplicialComplex *sc = sc_alloc(5, 5, 0);
for (int i = 0; i < 5; i++) {
    sc->edges[i][0] = i;
    sc->edges[i][1] = (i + 1) % 5;
}

/* Observed belief strengths between symptoms */
double observations[] = {2.0, -1.0, 3.0, -0.5, 1.5};
BeliefState *bs = belief_from_evidence(sc, observations);

/* Decompose */
BeliefHodge *bh = belief_hodge_decompose(bs);

/* The three components */
const double *evidence  = evidence_component(bh);   // what the data says
const double *coherence = coherence_component(bh);   // what consistency demands
const double *prior     = prior_component(bh);       // what the topology assumes
```

On a cycle, there are no triangles — so the coherence component is zero. The decomposition splits into just evidence and prior:

```
Edge 0: ω = 2.0    → evidence: 1.0   prior: 1.0
Edge 1: ω = -1.0   → evidence: -1.6  prior: 0.6
Edge 2: ω = 3.0    → evidence: 2.6   prior: 0.4
Edge 3: ω = -0.5   → evidence: -0.2  prior: -0.3
Edge 4: ω = 1.5    → evidence: 0.2   prior: 1.3
```

The prior component is *uniform circulation* around the cycle — it's the part of belief that flows consistently in one direction, independent of the data. It exists because the cycle topology creates a "hole" in the network, and the harmonic form wraps around that hole.

Remove an edge, breaking the cycle into a path, and the prior vanishes. The same data now decomposes differently — all evidence, no prior. The topology of your concept graph shapes what you believe independently of what you observe.

## The Harmonic Component

The harmonic component is the most stable part of any belief system. It survives when all the evidence changes. That's because it's topologically protected.

Harmonic forms live in the kernel of the Hodge Laplacian: Δh = 0. They are neither the gradient of any potential (not exact) nor the curl of any higher form (not co-exact). They circulate around topological holes in the belief network. On a cycle graph, that's uniform flow around the ring. On a more complex network, it's flow around every independent cycle.

The dimension of the harmonic space equals the first Betti number β₁ — the number of independent cycles in the graph. The library computes this directly:

```c
int cycles = topological_prior_strength(sc);  // β₁
```

More cycles means more dimensions of belief that are immune to evidence. This is a topological invariant: you can perturb the edge weights however you like, and the harmonic space retains its dimension. The only way to change it is to change the topology — add or remove edges, fill in triangles.

## What Each Component Means

### Evidence (Exact Component)

This is the belief that can be derived from a scalar potential function on the vertices. If you assign each vertex a "truth value" and take differences along edges, you get the exact component. It's the part of belief that's fully explained by the data at each node — the gradient of the evidence landscape.

**Properties:** Changes when observations change. Zero if all vertex values are equal. This is the responsive, data-driven part of belief.

### Coherence (Co-exact Component)

This is the belief required by local consistency. When triangles exist in the network (three mutually related propositions), the co-exact component enforces that beliefs around each triangle are compatible. It's the discrete analogue of the curl — it captures rotational structure in the belief field.

**Properties:** Zero if there are no triangles. Enforces logical consistency between mutually related beliefs. More triangles means more coherence constraints.

### Prior (Harmonic Component)

This is the belief that comes from the shape of the network itself. It exists in the kernel of the Laplacian — it satisfies no equation, fits no data, enforces no consistency. It simply *is*, because the topology allows it.

**Properties:** Topologically protected. Persists under perturbation. Dimension equals the number of independent cycles. This is the prior that doesn't come from any distribution — it comes from the geometry of how your concepts are connected.

## Applications

**Model Interpretability.** Decompose a neural network's internal representations into evidence-driven, coherence-driven, and prior-driven components. Understand not just which features matter, but why they matter — is the model relying on data, internal consistency, or structural bias?

**Belief Revision.** When new evidence arrives, only the exact component needs to update. The coherence and prior components are projections onto orthogonal subspaces. This means you can update beliefs efficiently without recomputing the entire decomposition.

**Scientific Theory Evaluation.** A scientific theory is a network of relationships between concepts. Hodge decomposition reveals which parts of the theory are supported by evidence, which parts are required by internal consistency, and which parts are assumptions baked into the theoretical framework. The harmonic component tells you exactly how much of the theory is immune to empirical disconfirmation.

**Agent Alignment.** When building agents that reason over knowledge graphs, the harmonic component reveals the agent's inescapable priors — the beliefs it cannot update away because they're built into the structure of its concept space.

## The Three Components are Orthogonal

A key property of the Hodge decomposition: the three components are mutually orthogonal. For any decomposed belief ω:

```
⟨exact, co-exact⟩  = 0
⟨exact, harmonic⟩  = 0
⟨co-exact, harmonic⟩ = 0
```

This means the decomposition is unique and energy-preserving:

```
‖ω‖² = ‖exact‖² + ‖co-exact‖² + ‖harmonic‖²
```

Every bit of belief energy is accounted for. No double-counting, no overlap. The library tests this orthogonality to machine precision.

## Classifying Individual Beliefs

Each edge (each individual belief in the network) can be classified:

```c
BeliefClassification c = classify_belief(bh, edge_idx);
// → BELIEF_EVIDENCE_DRIVEN   (the data dominates)
// → BELIEF_COHERENCE_DRIVEN  (consistency dominates)
// → BELIEF_PRIOR_DRIVEN      (topology dominates)
// → BELIEF_MIXED             (no single component dominates)
```

And the overall belief state gets a stability score:

```c
BeliefInterpretation interp = interpret_belief(bh);
interp.stability;          // fraction of energy in harmonic component
interp.evidence_fraction;  // how much comes from data
interp.coherence_fraction; // how much comes from consistency
interp.prior_fraction;     // how much comes from topology
```

High stability means the belief is resistant to evidence change. Low stability means it's fragile — it shifts with the data. Neither is inherently good or bad. The question is whether you *want* that particular belief to be stable.

## Building

```bash
make        # builds libhodgebelief.a
make test   # runs 28 tests covering all decomposition properties
```

Requires a C11 compiler and libm. No external dependencies.

## Architecture

```
src/
├── hodge.c            — Simplicial complexes, boundary/coboundary operators,
│                        Hodge Laplacian, sparse linear algebra (CG solver),
│                        core decomposition
├── belief.c           — Belief state construction, component accessors,
│                        conflict measurement, classification, stability,
│                        susceptibility analysis
└── interpretability.c — Energy spectrum, per-edge component fractions
```

The decomposition uses conjugate gradient to solve the least-squares projection problems: first project onto the image of the coboundary (exact), then project the residual onto the image of the co-coboundary (co-exact), and the remainder is harmonic by construction.

## Key Mathematical Objects

| Object | Code | Meaning |
|--------|------|---------|
| Boundary operator ∂₁ | `sc_boundary_1()` | Maps edges to their endpoint vertices |
| Coboundary operator δ₀ | `sc_coboundary_0()` | Discrete gradient: vertex values → edge differences |
| Coboundary operator δ₁ | `sc_coboundary_1()` | Discrete curl: edge values → triangle circulations |
| Hodge Laplacian Δ₀ | `sc_hodge_laplacian_0()` | Graph Laplacian on vertices |
| Hodge Laplacian Δ₁ | `sc_hodge_laplacian_1()` | Edge Laplacian: Δ₁ = δ₀δ₀* + δ₁*δ₁ |
| Betti number β₁ | `topological_prior_strength()` | Dimension of harmonic space |

## Quick Start

```c
#include "hodge_belief.h"

/* Build your belief network */
SimplicialComplex *sc = sc_alloc(3, 3, 1);
sc->edges[0] = (int[2]){0, 1};
sc->edges[1] = (int[2]){1, 2};
sc->edges[2] = (int[2]){0, 2};
sc->triangles[0] = (int[3]){0, 1, 2};

/* What you observe */
double obs[] = {0.8, 0.5, 0.3};
BeliefState *bs = belief_from_evidence(sc, obs);

/* Decompose */
BeliefHodge *bh = belief_hodge_decompose(bs);

/* Read the result */
BeliefInterpretation interp = interpret_belief(bh);
printf("Evidence: %.1f%%  Coherence: %.1f%%  Prior: %.1f%%\n",
       interp.evidence_fraction * 100,
       interp.coherence_fraction * 100,
       interp.prior_fraction * 100);

belief_hodge_free(bh);
bs_free(bs);
sc_free(sc);
```

## License

MIT
