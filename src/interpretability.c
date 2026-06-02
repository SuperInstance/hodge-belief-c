#include "hodge_belief.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Interpretability (separated for modularity) ── */

/* Additional analysis functions that depend on both hodge and belief layers */

/* Compute energy spectrum: how much total energy is in each component */
void belief_energy_spectrum(const BeliefHodge *bh,
                            double *evidence_energy,
                            double *coherence_energy,
                            double *prior_energy)
{
    int n = bh->hd->dim;
    *evidence_energy  = la_dot(bh->hd->exact, bh->hd->exact, n);
    *coherence_energy = la_dot(bh->hd->coexact, bh->hd->coexact, n);
    *prior_energy     = la_dot(bh->hd->harmonic, bh->hd->harmonic, n);
}

/* Per-edge component fractions */
void edge_component_fractions(const BeliefHodge *bh, int edge,
                               double *ev_frac, double *co_frac, double *pr_frac)
{
    double e = fabs(bh->hd->exact[edge]);
    double c = fabs(bh->hd->coexact[edge]);
    double h = fabs(bh->hd->harmonic[edge]);
    double total = e + c + h;
    if (total < 1e-15) {
        *ev_frac = *co_frac = *pr_frac = 0.0;
        return;
    }
    *ev_frac = e / total;
    *co_frac = c / total;
    *pr_frac = h / total;
}
