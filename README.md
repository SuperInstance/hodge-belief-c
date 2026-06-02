# hodge-belief-c

Hodge decomposition for agent belief states — a C11 library that decomposes beliefs into interpretable components using topological data analysis.

## What is Hodge Decomposition?

The [Hodge decomposition theorem](https://en.wikipedia.org/wiki/Hodge_theory) states that any differential form on a manifold (or cochain on a simplicial complex) can be uniquely decomposed into three orthogonal components:

```
ω = exact + co-exact + harmonic
```

This library applies this decomposition to **agent belief states** represented as 1-cochains on graph structures, where each edge represents a relationship or proposition the agent has a belief about.

### The Three Components

| Component | Mathematical | Belief Interpretation |
|-----------|-------------|----------------------|
| **Exact** | α = δβ (gradient of a potential) | **Evidence-driven**: beliefs directly supported by observed data |
| **Co-exact** | β = δ*γ (curl of a 2-form) | **Coherence-driven**: beliefs required by logical consistency |
| **Harmonic** | Δh = 0 (in kernel of Laplacian) | **Prior-driven**: beliefs imposed by topology, independent of data |

### Why This Matters for Interpretability

- **Evidence-driven beliefs** change when data changes. They're responsive but fragile.
- **Coherence-driven beliefs** come from the agent's internal logical structure. They enforce consistency.
- **Harmonic beliefs** are topological invariants. They persist regardless of what data arrives — they represent the agent's structural priors.

The dimension of the harmonic space equals the first Betti number β₁ of the graph, which counts independent cycles. More cycles → more topological bias → stronger prior influence.

## Building

```bash
make
make test
```

Requires: C11 compiler, libm.

## Usage

```c
#include "hodge_belief.h"

/* Create a simplicial complex (triangle with 3 vertices, 3 edges, 1 triangle) */
SimplicialComplex *sc = sc_alloc(3, 3, 1);
sc->edges[0] = (int[2]){0, 1};
sc->edges[1] = (int[2]){1, 2};
sc->edges[2] = (int[2]){0, 2};
sc->triangles[0] = (int[3]){0, 1, 2};

/* Create belief state from observations */
double observations[] = {0.8, 0.5, 0.3};
BeliefState *bs = belief_from_evidence(sc, observations);

/* Decompose */
BeliefHodge *bh = belief_hodge_decompose(bs);

/* Access components */
const double *evidence  = evidence_component(bh);   // exact
const double *coherence = coherence_component(bh);   // co-exact
const double *prior     = prior_component(bh);       // harmonic

/* Interpret */
BeliefInterpretation interp = interpret_belief(bh);
printf("Evidence fraction:  %.2f\n", interp.evidence_fraction);
printf("Coherence fraction: %.2f\n", interp.coherence_fraction);
printf("Prior fraction:     %.2f\n", interp.prior_fraction);
printf("Stability:          %.2f\n", interp.stability);

/* Topological prior strength */
int betti1 = topological_prior_strength(sc);

belief_hodge_free(bh);
bs_free(bs);
sc_free(sc);
```

## Architecture

```
src/
├── hodge.c            — Simplicial complexes, boundary/coboundary operators,
                         Hodge Laplacian, sparse linear algebra, decomposition
├── belief.c           — Agent belief states, evidence construction,
                         component accessors, conflict measurement,
                         interpretability classification
└── interpretability.c — Energy spectrum analysis, per-edge fractions
```

## Key Concepts

- **Simplicial Complex**: Vertices, edges, and triangles forming a topological space
- **Coboundary Operator δ**: The discrete analogue of exterior derivative (gradient/curl)
- **Hodge Laplacian Δ_k**: Combines δ and δ* to capture both boundary and co-boundary information
- **Belief State**: A 1-cochain assigning values to edges (relationships between concepts)
- **Harmonic Forms**: Elements of ker(Δ_k) — they "circulate" around holes in the complex

## License

MIT
