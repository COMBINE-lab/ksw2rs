#include <stdint.h>
#include <stdlib.h>
#include "ksw2.h"

typedef struct {
    uint32_t max;
    int32_t zdropped;
    int32_t max_q;
    int32_t max_t;
    int32_t mqe;
    int32_t mqe_t;
    int32_t mte;
    int32_t mte_q;
    int32_t score;
    int32_t reach_end;
    uint32_t n_cigar;
    uint64_t cigar_hash;
} ksw2rs_extz_plain;

static uint64_t ksw2rs_hash_cigar(const uint32_t *cigar, int n)
{
    uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
    int i;
    for (i = 0; i < n; ++i) {
        h ^= (uint64_t)cigar[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void ksw2rs_extz2_sse_ref(
    int qlen,
    const uint8_t *query,
    int tlen,
    const uint8_t *target,
    int8_t m,
    const int8_t *mat,
    int8_t q,
    int8_t e,
    int w,
    int zdrop,
    int end_bonus,
    int flag,
    ksw2rs_extz_plain *out)
{
    ksw_extz_t ez;
    ksw_reset_extz(&ez);
    ez.cigar = 0;
    ez.m_cigar = ez.n_cigar = 0;

    ksw_extz2_sse(0, qlen, query, tlen, target, m, mat, q, e, w, zdrop, end_bonus, flag, &ez);

    out->max = ez.max;
    out->zdropped = ez.zdropped;
    out->max_q = ez.max_q;
    out->max_t = ez.max_t;
    out->mqe = ez.mqe;
    out->mqe_t = ez.mqe_t;
    out->mte = ez.mte;
    out->mte_q = ez.mte_q;
    out->score = ez.score;
    out->reach_end = ez.reach_end;
    out->n_cigar = (uint32_t)ez.n_cigar;
    out->cigar_hash = ez.cigar ? ksw2rs_hash_cigar(ez.cigar, ez.n_cigar) : 0;

    free(ez.cigar);
}
