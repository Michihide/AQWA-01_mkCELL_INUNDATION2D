#ifndef MKCELL_NATIVE_H
#define MKCELL_NATIVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void mkcell_accumulate_codes(
    const int *codes,
    const double *weights,
    size_t n,
    double *out_areas,
    int out_len,
    int min_code,
    int max_code
);

void mkcell_elevation_stats(
    const double *values,
    size_t n,
    double *out_mean,
    double *out_median,
    double *out_std,
    double *out_p05,
    double *out_p95
);

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
);

int mkcell_quadtree_refine(
    const float *dem,
    const unsigned char *valid,
    const unsigned char *domain_mask,
    int height,
    int width,
    double transform_a,
    double transform_e,
    double transform_c,
    double transform_f,
    float nodata,
    double origin_x,
    double origin_y,
    double base_cell,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double std_threshold,
    double relief_threshold,
    double min_area,
    int max_depth,
    int *out_levels,
    int *out_is,
    int *out_js,
    int max_out,
    int *out_count
);

int mkcell_grid_clip_candidates(
    double origin_x,
    double origin_y,
    double base_cell,
    int max_level,
    double poly_minx,
    double poly_miny,
    double poly_maxx,
    double poly_maxy,
    int *out_levels,
    int *out_is,
    int *out_js,
    int max_out,
    int *out_count
);

#ifdef __cplusplus
}
#endif

#endif
