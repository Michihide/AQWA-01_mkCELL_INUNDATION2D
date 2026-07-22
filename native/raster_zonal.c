#include "mkcell_native.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

void mkcell_accumulate_codes(
    const int *codes,
    const double *weights,
    size_t n,
    double *out_areas,
    int out_len,
    int min_code,
    int max_code
) {
    if (!out_areas || out_len <= 0) {
        return;
    }
    memset(out_areas, 0, (size_t)out_len * sizeof(double));
    for (size_t k = 0; k < n; ++k) {
        int code = codes[k];
        if (code < min_code || code > max_code) {
            continue;
        }
        int idx = code - min_code;
        if (idx >= 0 && idx < out_len) {
            out_areas[idx] += weights[k];
        }
    }
}

void mkcell_elevation_stats(
    const double *values,
    size_t n,
    double *out_mean,
    double *out_median,
    double *out_std,
    double *out_p05,
    double *out_p95
) {
    if (n == 0) {
        if (out_mean) *out_mean = 0.0;
        if (out_median) *out_median = 0.0;
        if (out_std) *out_std = 0.0;
        if (out_p05) *out_p05 = 0.0;
        if (out_p95) *out_p95 = 0.0;
        return;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += values[i];
    }
    double mean = sum / (double)n;

    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) {
        return;
    }
    memcpy(buf, values, n * sizeof(double));
    qsort(buf, n, sizeof(double), cmp_double);

    double median;
    if (n % 2 == 1) {
        median = buf[n / 2];
    } else {
        median = 0.5 * (buf[n / 2 - 1] + buf[n / 2]);
    }

    size_t i05 = (size_t)floor(0.05 * (double)(n - 1));
    size_t i95 = (size_t)floor(0.95 * (double)(n - 1));
    double p05 = buf[i05];
    double p95 = buf[i95];

    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = values[i] - mean;
        var += d * d;
    }
    double stdv = sqrt(var / (double)n);

    if (out_mean) *out_mean = mean;
    if (out_median) *out_median = median;
    if (out_std) *out_std = stdv;
    if (out_p05) *out_p05 = p05;
    if (out_p95) *out_p95 = p95;

    free(buf);
}

void mkcell_slab_dem_stats(
    const float *slab,
    const unsigned char *valid,
    const unsigned char *domain,
    size_t n,
    float nodata,
    int need_relief,
    double *out_std,
    double *out_relief,
    int *out_count,
    int *out_full_cover
) {
    size_t dom_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (domain[i]) {
            dom_count++;
        }
    }

    double *vals = (double *)malloc(n * sizeof(double));
    if (!vals) {
        return;
    }

    size_t m = 0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!valid[i]) {
            continue;
        }
        float v = slab[i];
        if (v == nodata) {
            continue;
        }
        vals[m++] = (double)v;
        sum += (double)v;
    }

    if (out_count) *out_count = (int)m;
    if (out_full_cover) *out_full_cover = (dom_count > 0 && m >= dom_count) ? 1 : 0;

    if (m == 0) {
        if (out_std) *out_std = -1.0;
        if (out_relief) *out_relief = -1.0;
        free(vals);
        return;
    }

    double mean = sum / (double)m;
    double var = 0.0;
    for (size_t i = 0; i < m; ++i) {
        double d = vals[i] - mean;
        var += d * d;
    }
    if (out_std) *out_std = sqrt(var / (double)m);

    if (need_relief && out_relief) {
        qsort(vals, m, sizeof(double), cmp_double);
        size_t i05 = (size_t)floor(0.05 * (double)(m - 1));
        size_t i95 = (size_t)floor(0.95 * (double)(m - 1));
        *out_relief = vals[i95] - vals[i05];
    } else if (out_relief) {
        *out_relief = -1.0;
    }

    free(vals);
}
