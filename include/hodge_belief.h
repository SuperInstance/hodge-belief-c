#ifndef HODGE_BELIEF_H
#define HODGE_BELIEF_H

#include <stddef.h>
#include <stdbool.h>

/*
 * Hodge Decomposition for Agent Belief States
 *
 * Decomposes k-cochains on simplicial complexes into:
 *   - Exact (evidence-driven): α = δβ
 *   - Co-exact (coherence-driven): β = δ*γ
 *   - Harmonic (prior-driven): Δh = 0
 *
 * Based on the Hodge decomposition theorem:
 *   C^k = im(δ_{k-1}) ⊕ im(δ*_k) ⊕ ker(Δ_k)
 */

/* ── Sparse matrix ── */

typedef struct {
    int    rows, cols;
    int   *row_ptr;   /* CSR: size rows+1 */
    int   *col_idx;   /* CSR: nnz entries */
    double *values;   /* CSR: nnz entries */
} SparseMatrix;

SparseMatrix *sm_alloc(int rows, int cols, int nnz_hint);
void           sm_free(SparseMatrix *m);
SparseMatrix *sm_identity(int n);
SparseMatrix *sm_transpose(const SparseMatrix *m);
SparseMatrix *sm_multiply(const SparseMatrix *a, const SparseMatrix *b);
void           sm_to_dense(const SparseMatrix *m, double *buf);
int            sm_set_entry(SparseMatrix *m, int r, int c, double val);

/* ── Simplicial complex ── */

typedef struct {
    int n_vertices;
    int n_edges;
    int n_triangles;
    int (*edges)[2];      /* [n_edges][2]: vertex pairs */
    int (*triangles)[3];  /* [n_triangles][3]: vertex triples */
} SimplicialComplex;

SimplicialComplex *sc_alloc(int nv, int ne, int nt);
void               sc_free(SimplicialComplex *sc);

/* Boundary and coboundary operators */
/* B1: (n_edges x n_vertices) edge->vertex boundary */
SparseMatrix *sc_boundary_1(const SimplicialComplex *sc);
/* B2: (n_triangles x n_edges) triangle->edge boundary */
SparseMatrix *sc_boundary_2(const SimplicialComplex *sc);

/* Coboundary = transpose of boundary */
/* δ_0: (n_vertices x n_edges) = B1^T */
SparseMatrix *sc_coboundary_0(const SimplicialComplex *sc);
/* δ_1: (n_edges x n_triangles) = B2^T */
SparseMatrix *sc_coboundary_1(const SimplicialComplex *sc);

/* Hodge Laplacian: Δ_k = δ_{k-1}δ*_{k-1} + δ*_k δ_k */
/* Δ_0 = δ*_0 δ_0 = B1^T B1 (vertex Laplacian) */
/* Δ_1 = δ_0 δ*_0 + δ*_1 δ_1 = B1 B1^T + B2^T B2 (edge Laplacian) */
SparseMatrix *sc_hodge_laplacian_0(const SimplicialComplex *sc);
SparseMatrix *sc_hodge_laplacian_1(const SimplicialComplex *sc);

/* ── Linear algebra utilities ── */

/* Solve Ax = b via conjugate gradient (for symmetric positive semi-definite A) */
int la_cg_solve(const SparseMatrix *A, const double *b, double *x,
                int n, int max_iter, double tol);

/* Solve least-squares: find x minimizing ||Ax - b||_2 via normal equations + CG */
int la_least_squares(const SparseMatrix *A, const double *b, double *x,
                     int rows, int cols, int max_iter, double tol);

/* Matrix-vector product: y = A*x */
void la_spmv(const SparseMatrix *A, const double *x, double *y);

/* Dot product */
double la_dot(const double *a, const double *b, int n);

/* L2 norm */
double la_norm(const double *v, int n);

/* ── Hodge decomposition ── */

typedef struct {
    double *exact;     /* α = δβ   (evidence-driven) */
    double *coexact;   /* β = δ*γ  (coherence-driven) */
    double *harmonic;  /* h: Δh=0  (prior-driven, topological) */
    double *residual;  /* ω - (exact + coexact + harmonic) */
    int     dim;       /* dimension of the cochain space */
    double  exact_norm;
    double  coexact_norm;
    double  harmonic_norm;
} HodgeDecomposition;

/*
 * Decompose a 1-cochain (edge values) on the simplicial complex.
 * ω = exact + coexact + harmonic
 *   exact    = δ₀ β  where β minimizes ||ω - δ₀β||²  (project onto im δ₀)
 *   coexact  = δ₁* γ where γ minimizes ||(ω-exact) - δ₁*γ||²  (project onto im δ₁*)
 *   harmonic = ω - exact - coexact
 */
HodgeDecomposition *hodge_decompose_1(const SimplicialComplex *sc,
                                       const double *omega);
void                 hodge_decomposition_free(HodgeDecomposition *hd);

/* ── Agent belief states ── */

typedef struct {
    double *edge_values;  /* 1-cochain: belief strength on each edge */
    int     n_edges;
    const SimplicialComplex *sc;
} BeliefState;

BeliefState *bs_alloc(const SimplicialComplex *sc);
void         bs_free(BeliefState *bs);

/* Construct beliefs from raw evidence (observation weights per edge) */
BeliefState *belief_from_evidence(const SimplicialComplex *sc,
                                  const double *observations);

/* Full Hodge decomposition of a belief state */
typedef struct {
    HodgeDecomposition *hd;
    const BeliefState  *bs;
} BeliefHodge;

BeliefHodge *belief_hodge_decompose(const BeliefState *bs);
void         belief_hodge_free(BeliefHodge *bh);

/* Component accessors */
const double *evidence_component(const BeliefHodge *bh);   /* exact */
const double *coherence_component(const BeliefHodge *bh);  /* co-exact */
const double *prior_component(const BeliefHodge *bh);      /* harmonic */

/* Measure disagreement/conflict between components */
double belief_conflict(const BeliefHodge *bh);

/* ── Belief interpretability ── */

typedef enum {
    BELIEF_EVIDENCE_DRIVEN,
    BELIEF_COHERENCE_DRIVEN,
    BELIEF_PRIOR_DRIVEN,
    BELIEF_MIXED
} BeliefClassification;

typedef struct {
    BeliefClassification classification;
    double stability;           /* 0-1, higher = more stable */
    double susceptibility;      /* how much exact component changes with perturbation */
    double evidence_fraction;   /* ||exact|| / ||ω|| */
    double coherence_fraction;  /* ||coexact|| / ||ω|| */
    double prior_fraction;      /* ||harmonic|| / ||ω|| */
} BeliefInterpretation;

/* Classify a single edge's belief */
BeliefClassification classify_belief(const BeliefHodge *bh, int edge_idx);

/* Overall belief state stability (fraction of energy in harmonic component) */
double belief_stability(const BeliefHodge *bh);

/* Susceptibility: rate of change of evidence component with perturbation */
double susceptibility(const BeliefState *bs);

/* Dimension of harmonic space = amount of topological bias */
int topological_prior_strength(const SimplicialComplex *sc);

/* Full interpretation */
BeliefInterpretation interpret_belief(const BeliefHodge *bh);

#endif /* HODGE_BELIEF_H */
