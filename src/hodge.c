#include "hodge_belief.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Sparse matrix (CSR) ── */

SparseMatrix *sm_alloc(int rows, int cols, int nnz_hint)
{
    SparseMatrix *m = calloc(1, sizeof(*m));
    m->rows = rows;
    m->cols = cols;
    int cap = nnz_hint > 0 ? nnz_hint : 64;
    m->row_ptr = calloc(rows + 1, sizeof(int));
    m->col_idx = calloc(cap, sizeof(int));
    m->values  = calloc(cap, sizeof(double));
    /* We'll build entries then compress. For simplicity, use dense build then convert. */
    /* Instead: use a dynamic approach with reallocation. */
    return m;
}

void sm_free(SparseMatrix *m)
{
    if (!m) return;
    free(m->row_ptr);
    free(m->col_idx);
    free(m->values);
    free(m);
}

/* Build a dense matrix then compress to CSR */
static SparseMatrix *sm_from_dense(int rows, int cols, const double *d)
{
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++)
        if (d[i] != 0.0) nnz++;

    SparseMatrix *m = calloc(1, sizeof(*m));
    m->rows = rows;
    m->cols = cols;
    m->row_ptr = calloc(rows + 1, sizeof(int));
    m->col_idx = calloc(nnz, sizeof(int));
    m->values  = calloc(nnz, sizeof(double));

    int idx = 0;
    for (int i = 0; i < rows; i++) {
        m->row_ptr[i] = idx;
        for (int j = 0; j < cols; j++) {
            double v = d[i * cols + j];
            if (v != 0.0) {
                m->col_idx[idx] = j;
                m->values[idx]  = v;
                idx++;
            }
        }
    }
    m->row_ptr[rows] = idx;
    return m;
}

SparseMatrix *sm_identity(int n)
{
    double *d = calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++) d[i * n + i] = 1.0;
    SparseMatrix *m = sm_from_dense(n, n, d);
    free(d);
    return m;
}

SparseMatrix *sm_transpose(const SparseMatrix *m)
{
    /* Build dense, transpose, compress */
    int r = m->rows, c = m->cols;
    double *d = calloc(r * c, sizeof(double));
    for (int i = 0; i < r; i++)
        for (int p = m->row_ptr[i]; p < m->row_ptr[i + 1]; p++)
            d[i * c + m->col_idx[p]] = m->values[p];

    double *dt = calloc(c * r, sizeof(double));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            dt[j * r + i] = d[i * c + j];

    SparseMatrix *mt = sm_from_dense(c, r, dt);
    free(d);
    free(dt);
    return mt;
}

SparseMatrix *sm_multiply(const SparseMatrix *a, const SparseMatrix *b)
{
    /* A: (ar x ac), B: (br x bc), ac == br */
    int ar = a->rows, bc = b->cols;
    double *d = calloc(ar * bc, sizeof(double));

    for (int i = 0; i < ar; i++) {
        for (int p = a->row_ptr[i]; p < a->row_ptr[i + 1]; p++) {
            int k = a->col_idx[p];
            double a_ik = a->values[p];
            for (int q = b->row_ptr[k]; q < b->row_ptr[k + 1]; q++) {
                int j = b->col_idx[q];
                d[i * bc + j] += a_ik * b->values[q];
            }
        }
    }

    SparseMatrix *c = sm_from_dense(ar, bc, d);
    free(d);
    return c;
}

void sm_to_dense(const SparseMatrix *m, double *buf)
{
    memset(buf, 0, m->rows * m->cols * sizeof(double));
    for (int i = 0; i < m->rows; i++)
        for (int p = m->row_ptr[i]; p < m->row_ptr[i + 1]; p++)
            buf[i * m->cols + m->col_idx[p]] = m->values[p];
}

int sm_set_entry(SparseMatrix *m, int r, int c, double val)
{
    /* Only works before compression for now; no-op placeholder */
    (void)m; (void)r; (void)c; (void)val;
    return 0;
}

/* ── Simplicial complex ── */

SimplicialComplex *sc_alloc(int nv, int ne, int nt)
{
    SimplicialComplex *sc = calloc(1, sizeof(*sc));
    sc->n_vertices  = nv;
    sc->n_edges     = ne;
    sc->n_triangles = nt;
    if (ne > 0) sc->edges    = calloc(ne, sizeof(int[2]));
    if (nt > 0) sc->triangles = calloc(nt, sizeof(int[3]));
    return sc;
}

void sc_free(SimplicialComplex *sc)
{
    if (!sc) return;
    free(sc->edges);
    free(sc->triangles);
    free(sc);
}

/* Boundary operator B1: (n_edges x n_vertices)
   For edge (u,v), row has -1 at u and +1 at v (oriented). */
SparseMatrix *sc_boundary_1(const SimplicialComplex *sc)
{
    double *d = calloc(sc->n_edges * sc->n_vertices, sizeof(double));
    for (int e = 0; e < sc->n_edges; e++) {
        int u = sc->edges[e][0];
        int v = sc->edges[e][1];
        d[e * sc->n_vertices + u] = -1.0;
        d[e * sc->n_vertices + v] =  1.0;
    }
    SparseMatrix *m = sm_from_dense(sc->n_edges, sc->n_vertices, d);
    free(d);
    return m;
}

/* Boundary operator B2: (n_triangles x n_edges)
   For triangle (a,b,c) with edges ordered:
   Find which edge corresponds to (a,b), (b,c), (a,c) and assign signs. */
SparseMatrix *sc_boundary_2(const SimplicialComplex *sc)
{
    double *d = calloc(sc->n_triangles * sc->n_edges, sizeof(double));

    for (int t = 0; t < sc->n_triangles; t++) {
        int a = sc->triangles[t][0];
        int b = sc->triangles[t][1];
        int c = sc->triangles[t][2];
        /* Edges of triangle: (a,b), (b,c), (a,c) with oriented signs */
        /* Sign convention: edge (u,v) appears with sign based on orientation */
        int pairs[3][2] = {{a,b}, {b,c}, {a,c}};
        int signs[3] = {1, 1, -1}; /* standard orientation */

        for (int k = 0; k < 3; k++) {
            int pu = pairs[k][0], pv = pairs[k][1];
            int s = signs[k];
            /* Find matching edge */
            for (int e = 0; e < sc->n_edges; e++) {
                int eu = sc->edges[e][0], ev = sc->edges[e][1];
                if ((eu == pu && ev == pv)) {
                    d[t * sc->n_edges + e] += s;
                    break;
                }
                if ((eu == pv && ev == pu)) {
                    d[t * sc->n_edges + e] -= s;
                    break;
                }
            }
        }
    }
    SparseMatrix *m = sm_from_dense(sc->n_triangles, sc->n_edges, d);
    free(d);
    return m;
}

SparseMatrix *sc_coboundary_0(const SimplicialComplex *sc)
{
    SparseMatrix *b1 = sc_boundary_1(sc);
    SparseMatrix *d0 = sm_transpose(b1);
    sm_free(b1);
    return d0;
}

SparseMatrix *sc_coboundary_1(const SimplicialComplex *sc)
{
    SparseMatrix *b2 = sc_boundary_2(sc);
    SparseMatrix *d1 = sm_transpose(b2);
    sm_free(b2);
    return d1;
}

/* Δ_0 = B1^T B1 */
SparseMatrix *sc_hodge_laplacian_0(const SimplicialComplex *sc)
{
    SparseMatrix *b1 = sc_boundary_1(sc);
    SparseMatrix *b1t = sm_transpose(b1);
    SparseMatrix *l0 = sm_multiply(b1t, b1);
    sm_free(b1);
    sm_free(b1t);
    return l0;
}

/* Δ_1 = B1 B1^T + B2^T B2 */
SparseMatrix *sc_hodge_laplacian_1(const SimplicialComplex *sc)
{
    SparseMatrix *b1 = sc_boundary_1(sc);
    SparseMatrix *b1t = sm_transpose(b1);
    SparseMatrix *term1 = sm_multiply(b1, b1t);
    sm_free(b1);
    sm_free(b1t);

    SparseMatrix *term2 = NULL;
    if (sc->n_triangles > 0) {
        SparseMatrix *b2 = sc_boundary_2(sc);
        SparseMatrix *b2t = sm_transpose(b2);
        term2 = sm_multiply(b2t, b2);
        sm_free(b2);
        sm_free(b2t);
    } else {
        /* Zero matrix: allocate proper ne×ne dense buffer instead of
         * compound literal which would be a zero-length array (UB). */
        double *zd = calloc((size_t)sc->n_edges * sc->n_edges, sizeof(double));
        term2 = sm_from_dense(sc->n_edges, sc->n_edges, zd);
        free(zd);
    }

    /* Add term1 + term2 (both same dims: n_edges x n_edges) */
    int ne = sc->n_edges;
    double *d = calloc(ne * ne, sizeof(double));
    sm_to_dense(term1, d);
    double *d2 = calloc(ne * ne, sizeof(double));
    sm_to_dense(term2, d2);
    for (int i = 0; i < ne * ne; i++) d[i] += d2[i];
    SparseMatrix *l1 = sm_from_dense(ne, ne, d);
    free(d);
    free(d2);
    sm_free(term1);
    sm_free(term2);
    return l1;
}

/* ── Linear algebra ── */

void la_spmv(const SparseMatrix *A, const double *x, double *y)
{
    for (int i = 0; i < A->rows; i++) {
        y[i] = 0.0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
            y[i] += A->values[p] * x[A->col_idx[p]];
    }
}

double la_dot(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

double la_norm(const double *v, int n)
{
    return sqrt(la_dot(v, v, n));
}

/* Conjugate gradient for symmetric positive (semi-)definite systems.
   Handles semi-definite by projecting out null space iteratively. */
int la_cg_solve(const SparseMatrix *A, const double *b, double *x,
                int n, int max_iter, double tol)
{
    double *r  = calloc(n, sizeof(double));
    double *p  = calloc(n, sizeof(double));
    double *ap = calloc(n, sizeof(double));
    double *x0 = calloc(n, sizeof(double));

    /* Start from zero */
    memset(x, 0, n * sizeof(double));
    la_spmv(A, x, r);
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    memcpy(p, r, n * sizeof(double));

    double rs_old = la_dot(r, r, n);
    if (sqrt(rs_old) < tol) { free(r); free(p); free(ap); free(x0); return 0; }

    for (int iter = 0; iter < max_iter; iter++) {
        la_spmv(A, p, ap);
        double pAp = la_dot(p, ap, n);
        if (fabs(pAp) < 1e-30) break; /* in null space */
        double alpha = rs_old / pAp;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        double rs_new = la_dot(r, r, n);
        if (sqrt(rs_new) < tol) break;
        double beta = rs_new / rs_old;
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }

    free(r); free(p); free(ap); free(x0);
    return 0;
}

/* Least squares: min ||Ax - b||² via normal equations A^T A x = A^T b */
int la_least_squares(const SparseMatrix *A, const double *b, double *x,
                     int rows, int cols, int max_iter, double tol)
{
    SparseMatrix *at = sm_transpose(A);
    SparseMatrix *ata = sm_multiply(at, A);

    double *atb = calloc(cols, sizeof(double));
    double *tmp = calloc(rows, sizeof(double));
    /* Compute A^T b properly */
    memset(atb, 0, cols * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
            atb[A->col_idx[p]] += A->values[p] * b[i];

    int ret = la_cg_solve(ata, atb, x, cols, max_iter, tol);

    free(atb); free(tmp);
    sm_free(at); sm_free(ata);
    return ret;
}

/* ── Hodge decomposition ── */

HodgeDecomposition *hodge_decompose_1(const SimplicialComplex *sc,
                                       const double *omega)
{
    int ne = sc->n_edges;
    int nv = sc->n_vertices;

    HodgeDecomposition *hd = calloc(1, sizeof(*hd));
    hd->dim = ne;
    hd->exact    = calloc(ne, sizeof(double));
    hd->coexact  = calloc(ne, sizeof(double));
    hd->harmonic = calloc(ne, sizeof(double));
    hd->residual = calloc(ne, sizeof(double));

    /* Step 1: Exact component = δ₀ β, find β minimizing ||ω - δ₀ β|| */
    SparseMatrix *d0 = sc_coboundary_0(sc);  /* nv x ne */
    /* We want β (nv-dim) such that δ₀ β ≈ ω. δ₀ = B1^T is (nv x ne).
       So δ₀ maps from edges to vertices? No: δ₀ maps 0-cochains to 1-cochains.
       δ₀ = B1^T is (nv x ne)? No.
       B1 is (ne x nv). δ₀ = B1^T is (nv x ne).
       Wait, δ₀ maps C⁰ → C¹. C⁰ has dim nv, C¹ has dim ne.
       So δ₀ is (ne x nv). That's just B1^T transposed? No.
       B1: (ne x nv). δ₀ = B1^T: (nv x ne).
       But δ₀ should be ne x nv to map nv-vector to ne-vector.
       Actually δ₀ = B1^T means: if B1 is ne×nv, then B1^T is nv×ne.
       But the coboundary δ maps C^k → C^{k+1}, so δ₀: C⁰(nv) → C¹(ne).
       So δ₀ should be ne×nv. That IS B1^T: B1 is ne×nv, B1^T is nv×ne.

       Hmm, let me reconsider. The boundary ∂₁: C₁ → C₀ maps edges to vertices.
       So ∂₁ is (nv × ne) or... conventionally boundary maps chains to chains.
       ∂₁ maps 1-chains (edges) to 0-chains (vertices).
       If we represent chains as vectors (1-cochain = edge values as column),
       then ∂₁ acting on a 1-chain gives a 0-chain.
       The matrix representation: ∂₁ is (nv × ne).

       The coboundary δ₀ = ∂₁^T is (ne × nv).
       δ₀ maps 0-cochains (nv-dim) to 1-cochains (ne-dim).

       So: exact component = δ₀ β where β is 0-cochain (nv-dim).
       We find β minimizing ||ω - δ₀ β||².
       δ₀ is (ne × nv), β is nv-dim, δ₀ β is ne-dim.
       Normal equations: δ₀^T δ₀ β = δ₀^T ω.  (nv × nv system)
    */

    /* Rebuild coboundary properly: δ₀ is (ne × nv) = ∂₁^T where ∂₁ is (nv × ne)? 
       Let me just use the boundary matrix directly.
       B1 as I defined it is (ne × nv). So ∂₁ matrix in row form is B1: ne×nv.
       Actually in my code, B1 = sc_boundary_1 is (n_edges × n_vertices).
       The standard convention: ∂₁ acting on basis gives ∂(edge_i) = v_i - v_j.
       As a matrix, ∂₁ maps 1-chains to 0-chains: ∂₁ is (nv × ne).
       But I defined B1 as (ne × nv)... let me just use it consistently.
       
       B1 = sc_boundary_1: (ne × nv). Row e has -1 at u, +1 at v.
       This is the transpose of the boundary matrix in standard form.
       But that's fine as long as I'm consistent.
       
       δ₀ should be B1^T: (nv × ne). But δ₀ maps C⁰ → C¹ (nv → ne).
       So δ₀ should be (ne × nv). Which is B1 itself? 
       
       OK let me think about this differently. 
       (B1 x)_e = x_v - x_u for edge (u,v). That IS the coboundary of x.
       So δ₀ = B1: (ne × nv). The exact 1-forms are {B1 β : β ∈ R^nv}.
       
       Good. So exact = B1 β, minimize ||ω - B1 β||².
    */
    sm_free(d0);
    /* Use B1 directly */
    SparseMatrix *B1 = sc_boundary_1(sc);  /* ne × nv */

    double *beta = calloc(nv, sizeof(double));
    la_least_squares(B1, omega, beta, ne, nv, 1000, 1e-12);

    /* exact = B1 * beta */
    la_spmv(B1, beta, hd->exact);

    /* Step 2: Co-exact component from residual ω - exact */
    double *omega2 = calloc(ne, sizeof(double));
    for (int i = 0; i < ne; i++) omega2[i] = omega[i] - hd->exact[i];

    /* Co-exact = δ₁* γ where δ₁ = B2^T? Or δ₁ maps C¹ → C² (ne → nt).
       δ₁ should be (nt × ne). In my code B2 = sc_boundary_2 is (nt × ne).
       δ₁ = B2: (nt × ne)? Let me check.
       
       B2 row t: for triangle t with edges, gives boundary.
       (B2 x)_t = sum of signed edge values = boundary of 1-chain.
       That's ∂₂: (nt × ne). So δ₁ = B2^T: (ne × nt).
       δ₁ maps C¹ (ne-dim) to C² (nt-dim)... wait.
       δ₁ maps C¹ → C². C¹ has dim ne, C² has dim nt. So δ₁ is (nt × ne).
       That's B2^T? No, B2 is (nt × ne), so B2^T is (ne × nt).
       
       Hmm. Let me be careful.
       ∂₂: C₂ → C₁, so ∂₂ is (ne × nt) or (nt × ne)?
       If we think of chains as column vectors: ∂₂ maps nt-vector to ne-vector.
       Matrix: (ne × nt).
       
       But I defined B2 = sc_boundary_2 as (n_triangles × n_edges) = (nt × ne).
       That's ∂₂^T in the standard convention.
       
       OK I'm going in circles. Let me just use consistent algebra.
       δ₁ (coboundary, C¹ → C²) = (boundary ∂₂)^T.
       If ∂₂ is the boundary of a triangle expressed as edge differences, 
       then in my code B2 = (nt × ne) represents this.
       So δ₁ = B2: (nt × ne) maps ne-vector to nt-vector? 
       No: (nt × ne) matrix times (ne × 1) vector = (nt × 1). Yes!
       But δ₁ should map C¹(ne) → C²(nt). So δ₁ = B2 works? (nt × ne) × (ne × 1) = (nt × 1). Yes!
       
       Wait no. B2 is defined as boundary_2 which maps triangles to edges.
       That's ∂₂: C₂ → C₁? No, ∂₂ maps 2-chains (triangles) to 1-chains (edges).
       So ∂₂ should be (ne × nt). But B2 = (nt × ne) is ∂₂^T.
       
       The coboundary δ₁ = ∂₂^T = B2^T^T = B2? No, δ₁ = (∂₂)^T.
       If ∂₂ is (ne × nt), then δ₁ = ∂₂^T = (nt × ne).
       And B2 = (nt × ne), so B2 = δ₁. 
       
       But B2 was defined as boundary_2 which gives boundary of triangles in terms of edges.
       In my code: (B2 * x)_t = sum over edges of triangle t of signed x_e.
       This is the boundary of the 2-simplex, which should be ∂₂. 
       ∂₂ maps 2-simplices (basis) to 1-chains. As a matrix: ∂₂_{e,t} = ±1 if edge e is in triangle t.
       That would be (ne × nt). But my B2 is (nt × ne) = ∂₂^T.
       
       This is getting confusing. Let me just define things operationally:
       - "coboundary_0" δ₀ = B1: (ne × nv), maps vertex values to edge values.
         (B1 * f)_e = f(v) - f(u) for edge (u,v). ✓ This is the gradient.
       - "coboundary_1" δ₁: should map edge values to triangle values.
         If there are no triangles, co-exact component is zero.
         With triangles, δ₁ = B2^T (ne × nt)... no, let me define it as:
         δ₁ should produce a 2-cochain from a 1-cochain.
         (δ₁ * g)_t = g(edge_ab) + g(edge_bc) - g(edge_ac) for triangle (a,b,c).
         This is the curl. My B2 already does this: (B2 * g)_t = signed sum of edges of t.
         So δ₁ = B2: (nt × ne) maps ne-vector to nt-vector. ✓
       
       The co-exact component is δ₁* γ = δ₁^T γ where γ is a 2-cochain (nt-dim).
       δ₁^T = B2^T: (ne × nt).
       So co-exact = B2^T γ, find γ minimizing ||omega2 - B2^T γ||².
    */

    if (sc->n_triangles > 0) {
        SparseMatrix *B2t = sc_coboundary_1(sc); /* δ₁^T = B2^T: (ne × nt) */
        double *gamma = calloc(sc->n_triangles, sizeof(double));
        la_least_squares(B2t, omega2, gamma, ne, sc->n_triangles, 1000, 1e-12);
        la_spmv(B2t, gamma, hd->coexact);
        free(gamma);
        sm_free(B2t);
    }

    /* Step 3: Harmonic = ω - exact - co-exact */
    for (int i = 0; i < ne; i++)
        hd->harmonic[i] = omega[i] - hd->exact[i] - hd->coexact[i];

    /* Residual (should be ~0) */
    for (int i = 0; i < ne; i++)
        hd->residual[i] = omega[i] - hd->exact[i] - hd->coexact[i] - hd->harmonic[i];

    hd->exact_norm    = la_norm(hd->exact, ne);
    hd->coexact_norm  = la_norm(hd->coexact, ne);
    hd->harmonic_norm = la_norm(hd->harmonic, ne);

    free(beta);
    free(omega2);
    sm_free(B1);
    return hd;
}

void hodge_decomposition_free(HodgeDecomposition *hd)
{
    if (!hd) return;
    free(hd->exact);
    free(hd->coexact);
    free(hd->harmonic);
    free(hd->residual);
    free(hd);
}
