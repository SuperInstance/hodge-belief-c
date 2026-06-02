#include "hodge_belief.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Belief state ── */

BeliefState *bs_alloc(const SimplicialComplex *sc)
{
    BeliefState *bs = calloc(1, sizeof(*bs));
    bs->sc = sc;
    bs->n_edges = sc->n_edges;
    bs->edge_values = calloc(sc->n_edges, sizeof(double));
    return bs;
}

void bs_free(BeliefState *bs)
{
    if (!bs) return;
    free(bs->edge_values);
    free(bs);
}

BeliefState *belief_from_evidence(const SimplicialComplex *sc,
                                  const double *observations)
{
    BeliefState *bs = bs_alloc(sc);
    memcpy(bs->edge_values, observations, sc->n_edges * sizeof(double));
    return bs;
}

/* ── Belief Hodge ── */

BeliefHodge *belief_hodge_decompose(const BeliefState *bs)
{
    BeliefHodge *bh = calloc(1, sizeof(*bh));
    bh->bs = bs;
    bh->hd = hodge_decompose_1(bs->sc, bs->edge_values);
    return bh;
}

void belief_hodge_free(BeliefHodge *bh)
{
    if (!bh) return;
    hodge_decomposition_free(bh->hd);
    free(bh);
}

const double *evidence_component(const BeliefHodge *bh)
{
    return bh->hd->exact;
}

const double *coherence_component(const BeliefHodge *bh)
{
    return bh->hd->coexact;
}

const double *prior_component(const BeliefHodge *bh)
{
    return bh->hd->harmonic;
}

/* Conflict = angle between evidence and coherence components */
double belief_conflict(const BeliefHodge *bh)
{
    int n = bh->hd->dim;
    double ed = la_dot(bh->hd->exact, bh->hd->coexact, n);
    double en = la_norm(bh->hd->exact, n);
    double cn = la_norm(bh->hd->coexact, n);
    if (en < 1e-15 || cn < 1e-15) return 0.0;
    /* Return 1 - |cos(angle)|; 0 = aligned, 1 = orthogonal */
    double cos_a = ed / (en * cn);
    if (cos_a > 1.0) cos_a = 1.0;
    if (cos_a < -1.0) cos_a = -1.0;
    return 1.0 - fabs(cos_a);
}

/* ── Interpretability ── */

BeliefClassification classify_belief(const BeliefHodge *bh, int edge_idx)
{
    double e = fabs(bh->hd->exact[edge_idx]);
    double c = fabs(bh->hd->coexact[edge_idx]);
    double h = fabs(bh->hd->harmonic[edge_idx]);
    double total = e + c + h;
    if (total < 1e-15) return BELIEF_MIXED;

    double thresh = 0.6;
    if (e / total > thresh) return BELIEF_EVIDENCE_DRIVEN;
    if (c / total > thresh) return BELIEF_COHERENCE_DRIVEN;
    if (h / total > thresh) return BELIEF_PRIOR_DRIVEN;
    return BELIEF_MIXED;
}

double belief_stability(const BeliefHodge *bh)
{
    double total = la_norm(bh->bs->edge_values, bh->hd->dim);
    if (total < 1e-15) return 1.0;
    return bh->hd->harmonic_norm / total;
}

double susceptibility(const BeliefState *bs)
{
    /* Perturb each edge by epsilon and measure change in evidence component */
    double eps = 1e-4;
    int ne = bs->n_edges;

    BeliefHodge *bh0 = belief_hodge_decompose(bs);
    double *orig_exact = malloc(ne * sizeof(double));
    memcpy(orig_exact, bh0->hd->exact, ne * sizeof(double));

    double total_change = 0.0;
    for (int i = 0; i < ne; i++) {
        double *perturbed = malloc(ne * sizeof(double));
        memcpy(perturbed, bs->edge_values, ne * sizeof(double));
        perturbed[i] += eps;

        BeliefState *pbs = bs_alloc(bs->sc);
        memcpy(pbs->edge_values, perturbed, ne * sizeof(double));
        BeliefHodge *bh1 = belief_hodge_decompose(pbs);

        double diff = 0.0;
        for (int j = 0; j < ne; j++)
            diff += (bh1->hd->exact[j] - orig_exact[j]) * (bh1->hd->exact[j] - orig_exact[j]);
        total_change += sqrt(diff) / eps;

        belief_hodge_free(bh1);
        bs_free(pbs);
        free(perturbed);
    }

    belief_hodge_free(bh0);
    free(orig_exact);

    return total_change / ne;
}

int topological_prior_strength(const SimplicialComplex *sc)
{
    /* Dimension of harmonic space for 1-forms:
       β₁ = dim(ker Δ₁) = n_edges - rank(Δ₁)
       β₁ = dim H₁ = number of independent cycles = n_edges - n_vertices + n_components
       For connected graph: β₁ = n_edges - n_vertices + 1 */
    if (sc->n_vertices == 0) return 0;
    /* Compute via Laplacian: count zero eigenvalues */
    SparseMatrix *l1 = sc_hodge_laplacian_1(sc);
    int ne = sc->n_edges;
    double tol = 1e-8;

    /* Use power iteration to find eigenvalues, or just check row sums */
    /* For the harmonic space dimension, use: nullity = ne - rank.
       Rank can be estimated via row reduction on the dense form. */
    double *d = calloc(ne * ne, sizeof(double));
    sm_to_dense(l1, d);

    /* Gaussian elimination to find rank */
    int rank = 0;
    double *pivot_row = calloc(ne, sizeof(double));
    for (int col = 0; col < ne; col++) {
        /* Find pivot */
        int prow = -1;
        for (int row = rank; row < ne; row++) {
            if (fabs(d[row * ne + col]) > tol) { prow = row; break; }
        }
        if (prow < 0) continue;
        /* Swap */
        if (prow != rank) {
            for (int j = 0; j < ne; j++) {
                double tmp = d[rank * ne + j];
                d[rank * ne + j] = d[prow * ne + j];
                d[prow * ne + j] = tmp;
            }
        }
        /* Eliminate */
        double pv = d[rank * ne + col];
        for (int row = rank + 1; row < ne; row++) {
            double factor = d[row * ne + col] / pv;
            for (int j = col; j < ne; j++)
                d[row * ne + j] -= factor * d[rank * ne + j];
        }
        rank++;
    }

    free(d);
    free(pivot_row);
    sm_free(l1);
    return ne - rank;
}

BeliefInterpretation interpret_belief(const BeliefHodge *bh)
{
    BeliefInterpretation interp = {0};
    int n = bh->hd->dim;
    double total = la_norm(bh->bs->edge_values, n);

    if (total < 1e-15) {
        interp.classification = BELIEF_MIXED;
        return interp;
    }

    interp.evidence_fraction  = bh->hd->exact_norm / total;
    interp.coherence_fraction = bh->hd->coexact_norm / total;
    interp.prior_fraction     = bh->hd->harmonic_norm / total;
    interp.stability          = interp.prior_fraction;

    /* Overall classification */
    double thresh = 0.5;
    if (interp.evidence_fraction > thresh)
        interp.classification = BELIEF_EVIDENCE_DRIVEN;
    else if (interp.coherence_fraction > thresh)
        interp.classification = BELIEF_COHERENCE_DRIVEN;
    else if (interp.prior_fraction > thresh)
        interp.classification = BELIEF_PRIOR_DRIVEN;
    else
        interp.classification = BELIEF_MIXED;

    return interp;
}
