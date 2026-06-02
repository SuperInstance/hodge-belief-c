#include "hodge_belief.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ_DBL(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (tol)) { \
        printf("  FAIL %s:%d: %.6f != %.6f (tol %.1e)\n", __FILE__, __LINE__, _a, _b, (tol)); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

#define TOL 1e-6

/* Helper: build a triangle complex (3 vertices, 3 edges, 1 triangle) */
static SimplicialComplex *make_triangle(void)
{
    SimplicialComplex *sc = sc_alloc(3, 3, 1);
    sc->edges[0][0] = 0; sc->edges[0][1] = 1;
    sc->edges[1][0] = 1; sc->edges[1][1] = 2;
    sc->edges[2][0] = 0; sc->edges[2][1] = 2;
    sc->triangles[0][0] = 0; sc->triangles[0][1] = 1; sc->triangles[0][2] = 2;
    return sc;
}

/* Cycle graph C_n: n vertices, n edges, no triangles */
static SimplicialComplex *make_cycle(int n)
{
    SimplicialComplex *sc = sc_alloc(n, n, 0);
    for (int i = 0; i < n; i++) {
        sc->edges[i][0] = i;
        sc->edges[i][1] = (i + 1) % n;
    }
    return sc;
}

/* Path graph: n vertices, n-1 edges */
static SimplicialComplex *make_path(int n)
{
    SimplicialComplex *sc = sc_alloc(n, n - 1, 0);
    for (int i = 0; i < n - 1; i++) {
        sc->edges[i][0] = i;
        sc->edges[i][1] = i + 1;
    }
    return sc;
}

/* Complete graph K_n */
static SimplicialComplex *make_complete(int n)
{
    int ne = n * (n - 1) / 2;
    SimplicialComplex *sc = sc_alloc(n, ne, 0);
    int e = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            sc->edges[e][0] = i;
            sc->edges[e][1] = j;
            e++;
        }
    return sc;
}

/* ═══════════════════════════════════════════════
   Test 1-5: Basic simplicial complex operations
   ═══════════════════════════════════════════════ */

static void test_triangle_boundary_1(void)
{
    printf("test_triangle_boundary_1...\n");
    SimplicialComplex *sc = make_triangle();
    SparseMatrix *b1 = sc_boundary_1(sc);
    double d[9];
    sm_to_dense(b1, d);
    /* Row 0: edge (0,1) → -1 at col 0, +1 at col 1 */
    ASSERT_EQ_DBL(d[0*3+0], -1.0, TOL);
    ASSERT_EQ_DBL(d[0*3+1],  1.0, TOL);
    ASSERT_EQ_DBL(d[0*3+2],  0.0, TOL);
    /* Row 2: edge (0,2) → -1 at col 0, +1 at col 2 */
    ASSERT_EQ_DBL(d[2*3+0], -1.0, TOL);
    ASSERT_EQ_DBL(d[2*3+1],  0.0, TOL);
    ASSERT_EQ_DBL(d[2*3+2],  1.0, TOL);
    sm_free(b1);
    sc_free(sc);
}

static void test_coboundary_0_is_gradient(void)
{
    printf("test_coboundary_0_is_gradient...\n");
    SimplicialComplex *sc = make_triangle();
    /* B1 acting on vertex values gives edge differences (gradient) */
    SparseMatrix *b1 = sc_boundary_1(sc);
    double f[3] = {1.0, 3.0, 5.0};
    double grad[3];
    la_spmv(b1, f, grad);
    /* edge (0,1): 3-1=2, edge (1,2): 5-3=2, edge (0,2): 5-1=4 */
    ASSERT_EQ_DBL(grad[0], 2.0, TOL);
    ASSERT_EQ_DBL(grad[1], 2.0, TOL);
    ASSERT_EQ_DBL(grad[2], 4.0, TOL);
    sm_free(b1);
    sc_free(sc);
}

static void test_boundary_2_triangle(void)
{
    printf("test_boundary_2_triangle...\n");
    SimplicialComplex *sc = make_triangle();
    if (sc->n_triangles > 0) {
        SparseMatrix *b2 = sc_boundary_2(sc);
        double d[3];
        sm_to_dense(b2, d);
        /* Triangle (0,1,2): edges are (0,1)=0, (1,2)=1, (0,2)=2
           signs: +1 for (0,1), +1 for (1,2), -1 for (0,2) */
        ASSERT_EQ_DBL(d[0],  1.0, TOL);
        ASSERT_EQ_DBL(d[1],  1.0, TOL);
        ASSERT_EQ_DBL(d[2], -1.0, TOL);
        sm_free(b2);
    }
    sc_free(sc);
}

static void test_hodge_laplacian_0(void)
{
    printf("test_hodge_laplacian_0...\n");
    SimplicialComplex *sc = make_triangle();
    SparseMatrix *l0 = sc_hodge_laplacian_0(sc);
    /* L = B1^T B1, for triangle: L[i,i] = degree, L[i,j] = -1 if adjacent */
    double d[9];
    sm_to_dense(l0, d);
    ASSERT_EQ_DBL(d[0*3+0], 2.0, TOL); /* deg(0) = 2 */
    ASSERT_EQ_DBL(d[1*3+1], 2.0, TOL); /* deg(1) = 2 */
    ASSERT_EQ_DBL(d[2*3+2], 2.0, TOL); /* deg(2) = 2 */
    sm_free(l0);
    sc_free(sc);
}

static void test_hodge_laplacian_1_triangle(void)
{
    printf("test_hodge_laplacian_1_triangle...\n");
    SimplicialComplex *sc = make_triangle();
    SparseMatrix *l1 = sc_hodge_laplacian_1(sc);
    double d[9];
    sm_to_dense(l1, d);
    /* For a filled triangle, Δ₁ should have rank 2 (β₁=0 for disk topology) */
    /* All diagonal entries should be > 0 since triangle has no holes */
    ASSERT_TRUE(d[0] > 1.0);
    ASSERT_TRUE(d[4] > 1.0);
    ASSERT_TRUE(d[8] > 1.0);
    sm_free(l1);
    sc_free(sc);
}

/* ═══════════════════════════════════════════════
   Test 6-10: Hodge decomposition fundamentals
   ═══════════════════════════════════════════════ */

static void test_exact_form_decomposition(void)
{
    printf("test_exact_form_decomposition...\n");
    SimplicialComplex *sc = make_triangle();
    /* ω = δ₀ β: gradient of vertex values → pure exact form */
    double beta[3] = {1.0, 2.0, 4.0};
    SparseMatrix *b1 = sc_boundary_1(sc);
    double omega[3];
    la_spmv(b1, beta, omega);  /* B1 * beta = gradient */

    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);
    /* Should be entirely in exact component */
    ASSERT_TRUE(hd->exact_norm > 0.5);
    ASSERT_EQ_DBL(hd->coexact_norm, 0.0, 1e-4);
    ASSERT_EQ_DBL(hd->harmonic_norm, 0.0, 1e-4);

    /* Verify exact matches omega */
    for (int i = 0; i < 3; i++)
        ASSERT_EQ_DBL(hd->exact[i], omega[i], 1e-4);

    hodge_decomposition_free(hd);
    sm_free(b1);
    sc_free(sc);
}

static void test_harmonic_form_cycle(void)
{
    printf("test_harmonic_form_cycle...\n");
    /* Cycle C4 has β₁=1, so there should be a harmonic 1-form */
    SimplicialComplex *sc = make_cycle(4);
    /* Uniform flow around cycle: ω = (1,1,1,1) — this is harmonic */
    double omega[4] = {1.0, 1.0, 1.0, 1.0};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);

    /* On a cycle with no triangles, co-exact is always 0.
       Uniform flow should be mostly harmonic (not exact because no potential) */
    ASSERT_TRUE(hd->harmonic_norm > 0.5);
    ASSERT_EQ_DBL(hd->coexact_norm, 0.0, 1e-4);

    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_harmonic_satisfies_laplacian(void)
{
    printf("test_harmonic_satisfies_laplacian...\n");
    SimplicialComplex *sc = make_cycle(5);
    double omega[5] = {1.0, 2.0, 1.0, 2.0, 1.0};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);

    /* Verify Δ₁ h ≈ 0 */
    SparseMatrix *l1 = sc_hodge_laplacian_1(sc);
    double lh[5];
    la_spmv(l1, hd->harmonic, lh);
    for (int i = 0; i < 5; i++)
        ASSERT_EQ_DBL(lh[i], 0.0, 1e-4);

    sm_free(l1);
    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_orthogonality(void)
{
    printf("test_orthogonality...\n");
    /* On a triangle, decompose a mixed form and check orthogonality */
    SimplicialComplex *sc = make_triangle();
    double omega[3] = {3.0, -1.0, 2.0};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);

    /* exact ⊥ co-exact */
    double ec = la_dot(hd->exact, hd->coexact, 3);
    ASSERT_EQ_DBL(ec, 0.0, 1e-4);

    /* exact ⊥ harmonic */
    double eh = la_dot(hd->exact, hd->harmonic, 3);
    ASSERT_EQ_DBL(eh, 0.0, 1e-4);

    /* co-exact ⊥ harmonic */
    double ch = la_dot(hd->coexact, hd->harmonic, 3);
    ASSERT_EQ_DBL(ch, 0.0, 1e-4);

    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_decomposition_preserves_norm(void)
{
    printf("test_decomposition_preserves_norm...\n");
    SimplicialComplex *sc = make_triangle();
    double omega[3] = {1.5, -2.3, 0.7};
    double omega_norm_sq = la_dot(omega, omega, 3);

    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);
    double comp_norm_sq = la_dot(hd->exact, hd->exact, 3)
                        + la_dot(hd->coexact, hd->coexact, 3)
                        + la_dot(hd->harmonic, hd->harmonic, 3);
    ASSERT_EQ_DBL(comp_norm_sq, omega_norm_sq, 1e-3);

    hodge_decomposition_free(hd);
    sc_free(sc);
}

/* ═══════════════════════════════════════════════
   Test 11-15: Belief states
   ═══════════════════════════════════════════════ */

static void test_belief_from_evidence(void)
{
    printf("test_belief_from_evidence...\n");
    SimplicialComplex *sc = make_triangle();
    double obs[3] = {0.8, 0.5, 0.3};
    BeliefState *bs = belief_from_evidence(sc, obs);
    ASSERT_TRUE(bs != NULL);
    for (int i = 0; i < 3; i++)
        ASSERT_EQ_DBL(bs->edge_values[i], obs[i], TOL);
    bs_free(bs);
    sc_free(sc);
}

static void test_belief_pure_evidence(void)
{
    printf("test_belief_pure_evidence...\n");
    SimplicialComplex *sc = make_triangle();
    /* Construct beliefs as gradient → all in exact component */
    double obs[3] = {2.0, 3.0, 5.0}; /* = gradient of (1,3,6) */
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    BeliefInterpretation interp = interpret_belief(bh);
    ASSERT_TRUE(interp.evidence_fraction > 0.95);

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_belief_pure_prior(void)
{
    printf("test_belief_pure_prior...\n");
    SimplicialComplex *sc = make_cycle(4);
    /* Uniform flow on cycle → harmonic (prior-driven) */
    double obs[4] = {1.0, 1.0, 1.0, 1.0};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    BeliefInterpretation interp = interpret_belief(bh);
    ASSERT_TRUE(interp.prior_fraction > 0.8);

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_mixed_belief_decomposition(void)
{
    printf("test_mixed_belief_decomposition...\n");
    SimplicialComplex *sc = make_triangle();
    double obs[3] = {2.0, -1.0, 3.5};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    /* Should have non-trivial components */
    ASSERT_TRUE(bh->hd->exact_norm > 0.1);
    /* Verify reconstruction */
    for (int i = 0; i < 3; i++) {
        double recon = bh->hd->exact[i] + bh->hd->coexact[i] + bh->hd->harmonic[i];
        ASSERT_EQ_DBL(recon, obs[i], 1e-3);
    }

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_belief_conflict(void)
{
    printf("test_belief_conflict...\n");
    SimplicialComplex *sc = make_triangle();
    /* Orthogonal components → high conflict (low alignment) */
    double obs[3] = {1.0, -2.0, 3.0};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    double conflict = belief_conflict(bh);
    ASSERT_TRUE(conflict >= 0.0 && conflict <= 2.0);

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

/* ═══════════════════════════════════════════════
   Test 16-20: Interpretability
   ═══════════════════════════════════════════════ */

static void test_classify_belief_evidence(void)
{
    printf("test_classify_belief_evidence...\n");
    SimplicialComplex *sc = make_triangle();
    double obs[3] = {2.0, 3.0, 5.0}; /* gradient */
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    for (int i = 0; i < 3; i++) {
        BeliefClassification c = classify_belief(bh, i);
        ASSERT_TRUE(c == BELIEF_EVIDENCE_DRIVEN);
    }

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_classify_belief_prior(void)
{
    printf("test_classify_belief_prior...\n");
    SimplicialComplex *sc = make_cycle(4);
    double obs[4] = {1.0, 1.0, 1.0, 1.0};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    for (int i = 0; i < 4; i++) {
        BeliefClassification c = classify_belief(bh, i);
        ASSERT_TRUE(c == BELIEF_PRIOR_DRIVEN);
    }

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_belief_stability_harmonic(void)
{
    printf("test_belief_stability_harmonic...\n");
    SimplicialComplex *sc = make_cycle(4);
    double obs[4] = {1.0, 1.0, 1.0, 1.0};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    double stab = belief_stability(bh);
    ASSERT_TRUE(stab > 0.8); /* mostly harmonic → high stability */

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_belief_stability_evidence(void)
{
    printf("test_belief_stability_evidence...\n");
    SimplicialComplex *sc = make_triangle();
    double obs[3] = {2.0, 3.0, 5.0}; /* pure gradient */
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh = belief_hodge_decompose(bs);

    double stab = belief_stability(bh);
    ASSERT_TRUE(stab < 0.1); /* mostly evidence → low stability */

    belief_hodge_free(bh);
    bs_free(bs);
    sc_free(sc);
}

static void test_topological_prior_strength(void)
{
    printf("test_topological_prior_strength...\n");
    /* Triangle (filled): β₁ = 0 */
    SimplicialComplex *tri = make_triangle();
    int tps_tri = topological_prior_strength(tri);
    ASSERT_EQ_DBL(tps_tri, 0.0, 0.5);

    /* Cycle C4: β₁ = 1 */
    SimplicialComplex *c4 = make_cycle(4);
    int tps_c4 = topological_prior_strength(c4);
    ASSERT_EQ_DBL(tps_c4, 1.0, 0.5);

    /* Cycle C5: β₁ = 1 */
    SimplicialComplex *c5 = make_cycle(5);
    int tps_c5 = topological_prior_strength(c5);
    ASSERT_EQ_DBL(tps_c5, 1.0, 0.5);

    sc_free(tri);
    sc_free(c4);
    sc_free(c5);
}

/* ═══════════════════════════════════════════════
   Test 21-28: Edge cases and additional coverage
   ═══════════════════════════════════════════════ */

static void test_single_edge(void)
{
    printf("test_single_edge...\n");
    SimplicialComplex *sc = sc_alloc(2, 1, 0);
    sc->edges[0][0] = 0;
    sc->edges[0][1] = 1;

    double omega[1] = {3.14};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);
    /* Single edge: should be exact (gradient of (0, 3.14)) */
    ASSERT_TRUE(hd->exact_norm > 2.0);
    ASSERT_EQ_DBL(hd->harmonic_norm, 0.0, 1e-4);
    ASSERT_EQ_DBL(hd->coexact_norm, 0.0, 1e-4);

    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_single_vertex(void)
{
    printf("test_single_vertex...\n");
    SimplicialComplex *sc = sc_alloc(1, 0, 0);
    ASSERT_TRUE(sc != NULL);
    ASSERT_EQ_DBL(sc->n_vertices, 1.0, TOL);
    ASSERT_EQ_DBL(sc->n_edges, 0.0, TOL);
    sc_free(sc);
}

static void test_path_no_harmonic(void)
{
    printf("test_path_no_harmonic...\n");
    /* Path graph has β₁ = 0, no harmonic forms */
    SimplicialComplex *sc = make_path(4);
    double omega[3] = {1.0, -2.0, 3.0};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);

    ASSERT_EQ_DBL(hd->harmonic_norm, 0.0, 1e-4);

    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_stability_unchanged_under_perturbation(void)
{
    printf("test_stability_unchanged_under_perturbation...\n");
    SimplicialComplex *sc = make_cycle(5);
    double obs[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    BeliefState *bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh0 = belief_hodge_decompose(bs);

    double harmonic0[5];
    memcpy(harmonic0, prior_component(bh0), 5 * sizeof(double));

    /* Perturb */
    obs[0] += 0.1;
    bs_free(bs);
    bs = belief_from_evidence(sc, obs);
    BeliefHodge *bh1 = belief_hodge_decompose(bs);

    /* Harmonic component should be relatively stable */
    const double *harm1 = prior_component(bh1);
    double h0_norm = la_norm(harmonic0, 5);
    if (h0_norm > 1e-10) {
        double diff[5];
        for (int i = 0; i < 5; i++) diff[i] = harmonic0[i] - harm1[i];
        double rel_change = la_norm(diff, 5) / h0_norm;
        ASSERT_TRUE(rel_change < 0.5); /* harmonic shouldn't change too much */
    }

    belief_hodge_free(bh1);
    belief_hodge_free(bh0);
    bs_free(bs);
    sc_free(sc);
}

static void test_complete_graph(void)
{
    printf("test_complete_graph...\n");
    SimplicialComplex *sc = make_complete(4);
    /* K4 has 6 edges, no triangles in our model */
    ASSERT_EQ_DBL(sc->n_edges, 6.0, TOL);

    double omega[6] = {1.0, 2.0, 3.0, -1.0, -2.0, 0.5};
    HodgeDecomposition *hd = hodge_decompose_1(sc, omega);

    /* Should decompose without error */
    ASSERT_TRUE(hd != NULL);

    /* Verify reconstruction */
    for (int i = 0; i < 6; i++) {
        double recon = hd->exact[i] + hd->coexact[i] + hd->harmonic[i];
        ASSERT_EQ_DBL(recon, omega[i], 1e-3);
    }

    hodge_decomposition_free(hd);
    sc_free(sc);
}

static void test_sparse_matrix_identity(void)
{
    printf("test_sparse_matrix_identity...\n");
    SparseMatrix *id = sm_identity(4);
    double d[16];
    sm_to_dense(id, d);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ASSERT_EQ_DBL(d[i*4+j], (i == j) ? 1.0 : 0.0, TOL);
    sm_free(id);
}

static void test_sparse_matrix_transpose(void)
{
    printf("test_sparse_matrix_transpose...\n");
    SimplicialComplex *sc = make_triangle();
    SparseMatrix *b1 = sc_boundary_1(sc);
    SparseMatrix *b1t = sm_transpose(b1);

    double d1[9], d1t[9];
    sm_to_dense(b1, d1);
    sm_to_dense(b1t, d1t);

    /* Transpose property: d1t[j][i] = d1[i][j] */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ASSERT_EQ_DBL(d1t[j*3+i], d1[i*3+j], TOL);

    sm_free(b1);
    sm_free(b1t);
    sc_free(sc);
}

static void test_sparse_matrix_multiply(void)
{
    printf("test_sparse_matrix_multiply...\n");
    SparseMatrix *id = sm_identity(3);
    SimplicialComplex *sc = make_triangle();
    SparseMatrix *b1 = sc_boundary_1(sc);

    SparseMatrix *prod = sm_multiply(id, b1);
    double d_orig[9], d_prod[9];
    sm_to_dense(b1, d_orig);
    sm_to_dense(prod, d_prod);

    for (int i = 0; i < 9; i++)
        ASSERT_EQ_DBL(d_prod[i], d_orig[i], TOL);

    sm_free(id);
    sm_free(b1);
    sm_free(prod);
    sc_free(sc);
}

static void test_cycle_vs_triangle_topology(void)
{
    printf("test_cycle_vs_triangle_topology...\n");
    /* Triangle (filled) has β₁ = 0, cycle has β₁ = 1 */
    SimplicialComplex *tri = make_triangle();
    SimplicialComplex *c3 = make_cycle(3);

    int tps_tri = topological_prior_strength(tri);
    int tps_c3  = topological_prior_strength(c3);

    ASSERT_EQ_DBL(tps_tri, 0.0, 0.5);
    ASSERT_EQ_DBL(tps_c3,  1.0, 0.5);

    sc_free(tri);
    sc_free(c3);
}

/* ═══════════════════════════════════════════════
   Main
   ═══════════════════════════════════════════════ */

int main(void)
{
    printf("═══ Hodge Belief Tests ═══\n\n");

    /* Basic operations */
    test_triangle_boundary_1();
    test_coboundary_0_is_gradient();
    test_boundary_2_triangle();
    test_hodge_laplacian_0();
    test_hodge_laplacian_1_triangle();

    /* Decomposition */
    test_exact_form_decomposition();
    test_harmonic_form_cycle();
    test_harmonic_satisfies_laplacian();
    test_orthogonality();
    test_decomposition_preserves_norm();

    /* Belief states */
    test_belief_from_evidence();
    test_belief_pure_evidence();
    test_belief_pure_prior();
    test_mixed_belief_decomposition();
    test_belief_conflict();

    /* Interpretability */
    test_classify_belief_evidence();
    test_classify_belief_prior();
    test_belief_stability_harmonic();
    test_belief_stability_evidence();
    test_topological_prior_strength();

    /* Edge cases */
    test_single_edge();
    test_single_vertex();
    test_path_no_harmonic();
    test_stability_unchanged_under_perturbation();
    test_complete_graph();
    test_sparse_matrix_identity();
    test_sparse_matrix_transpose();
    test_sparse_matrix_multiply();
    test_cycle_vs_triangle_topology();

    printf("\n═══ Results: %d passed, %d failed ═══\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
