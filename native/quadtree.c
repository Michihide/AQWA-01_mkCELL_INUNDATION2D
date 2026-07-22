#include "mkcell_native.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int level;
    int i;
    int j;
} QtCell;

typedef struct {
    QtCell *items;
    size_t len;
    size_t cap;
} QtCellList;

static void qtcl_push(QtCellList *list, int level, int i, int j) {
    if (list->len >= list->cap) {
        size_t new_cap = list->cap ? list->cap * 2 : 64;
        QtCell *p = (QtCell *)realloc(list->items, new_cap * sizeof(QtCell));
        if (!p) return;
        list->items = p;
        list->cap = new_cap;
    }
    list->items[list->len].level = level;
    list->items[list->len].i = i;
    list->items[list->len].j = j;
    list->len++;
}

static int slab_has_domain(
    const unsigned char *domain_mask,
    int height,
    int width,
    int r0, int r1, int c0, int c1
) {
    for (int r = r0; r < r1; ++r) {
        for (int c = c0; c < c1; ++c) {
            if (domain_mask[r * width + c]) {
                return 1;
            }
        }
    }
    return 0;
}

static void bounds_to_slice(
    double origin_x,
    double origin_y,
    double base_cell,
    int level,
    int i,
    int j,
    int height,
    int width,
    double transform_a,
    double transform_e,
    double transform_c,
    double transform_f,
    int *r0,
    int *r1,
    int *c0,
    int *c1
) {
    double step = base_cell / (double)(1 << level);
    double minx = origin_x + i * step;
    double miny = origin_y + j * step;
    double maxx = minx + step;
    double maxy = miny + step;

    int col0 = (int)floor((minx - transform_c) / transform_a);
    int col1 = (int)ceil((maxx - transform_c) / transform_a);
    int row0 = (int)floor((maxy - transform_f) / transform_e);
    int row1 = (int)ceil((miny - transform_f) / transform_e);

    if (row0 > row1) {
        int t = row0; row0 = row1; row1 = t;
    }
    if (col0 > col1) {
        int t = col0; col0 = col1; col1 = t;
    }

    if (row0 < 0) row0 = 0;
    if (col0 < 0) col0 = 0;
    if (row1 > height) row1 = height;
    if (col1 > width) col1 = width;

    *r0 = row0;
    *r1 = row1;
    *c0 = col0;
    *c1 = col1;
}

static void eval_cell(
    const float *dem,
    const unsigned char *valid,
    const unsigned char *domain,
    int height,
    int width,
    float nodata,
    int r0,
    int r1,
    int c0,
    int c1,
    double std_threshold,
    double relief_threshold,
    int *should_split,
    int *reason_std
) {
    *should_split = 0;
    *reason_std = 0;
    if (r1 <= r0 || c1 <= c0) {
        return;
    }

    size_t n = (size_t)(r1 - r0) * (size_t)(c1 - c0);
    float *slab = (float *)malloc(n * sizeof(float));
    unsigned char *v = (unsigned char *)malloc(n);
    unsigned char *d = (unsigned char *)malloc(n);
    if (!slab || !v || !d) {
        free(slab); free(v); free(d);
        return;
    }

    size_t k = 0;
    for (int r = r0; r < r1; ++r) {
        for (int c = c0; c < c1; ++c) {
            size_t idx = (size_t)r * (size_t)width + (size_t)c;
            slab[k] = dem[idx];
            v[k] = valid[idx];
            d[k] = domain[idx];
            k++;
        }
    }

    double stdv, relief;
    int count, full_cover;
    mkcell_slab_dem_stats(slab, v, d, n, nodata, 1, &stdv, &relief, &count, &full_cover);
    free(slab); free(v); free(d);

    if (count <= 0) {
        return;
    }
    if (stdv > std_threshold) {
        *should_split = 1;
        *reason_std = 1;
    } else if (relief > relief_threshold) {
        *should_split = 1;
        *reason_std = 0;
    }
}

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
) {
    QtCellList leaves = {0};
    int i_min = (int)floor((xmin - origin_x) / base_cell);
    int i_max = (int)floor((xmax - origin_x) / base_cell);
    int j_min = (int)floor((ymin - origin_y) / base_cell);
    int j_max = (int)floor((ymax - origin_y) / base_cell);

    for (int i = i_min; i <= i_max; ++i) {
        for (int j = j_min; j <= j_max; ++j) {
            int r0, r1, c0, c1;
            bounds_to_slice(origin_x, origin_y, base_cell, 0, i, j,
                            height, width, transform_a, transform_e, transform_c, transform_f,
                            &r0, &r1, &c0, &c1);
            if (slab_has_domain(domain_mask, height, width, r0, r1, c0, c1)) {
                qtcl_push(&leaves, 0, i, j);
            }
        }
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        QtCellList pending = {0};
        for (size_t k = 0; k < leaves.len; ++k) {
            qtcl_push(&pending, leaves.items[k].level, leaves.items[k].i, leaves.items[k].j);
        }

        int min_level = max_depth;
        for (size_t k = 0; k < pending.len; ++k) {
            if (pending.items[k].level < min_level) {
                min_level = pending.items[k].level;
            }
        }
        if (min_level >= max_depth) {
            free(pending.items);
            break;
        }

        QtCellList to_split = {0};
#ifdef _OPENMP
#pragma omp parallel
        {
            QtCellList local = {0};
#pragma omp for schedule(dynamic)
            for (size_t k = 0; k < pending.len; ++k) {
                int level = pending.items[k].level;
                int i = pending.items[k].i;
                int j = pending.items[k].j;
                if (level >= max_depth) continue;

                double step = base_cell / (double)(1 << level);
                double cell_area = step * step;
                if (cell_area < min_area) continue;

                int r0, r1, c0, c1;
                bounds_to_slice(origin_x, origin_y, base_cell, level, i, j,
                                height, width, transform_a, transform_e, transform_c, transform_f,
                                &r0, &r1, &c0, &c1);

                int should_split = 0, reason_std = 0;
                eval_cell(dem, valid, domain_mask, height, width, nodata,
                          r0, r1, c0, c1, std_threshold, relief_threshold,
                          &should_split, &reason_std);
                if (should_split) {
                    qtcl_push(&local, level, i, j);
                }
            }
#pragma omp critical
            {
                for (size_t k = 0; k < local.len; ++k) {
                    qtcl_push(&to_split, local.items[k].level, local.items[k].i, local.items[k].j);
                }
            }
            free(local.items);
        }
#else
        for (size_t k = 0; k < pending.len; ++k) {
            int level = pending.items[k].level;
            int i = pending.items[k].i;
            int j = pending.items[k].j;
            if (level >= max_depth) continue;

            double step = base_cell / (double)(1 << level);
            double cell_area = step * step;
            if (cell_area < min_area) continue;

            int r0, r1, c0, c1;
            bounds_to_slice(origin_x, origin_y, base_cell, level, i, j,
                            height, width, transform_a, transform_e, transform_c, transform_f,
                            &r0, &r1, &c0, &c1);

            int should_split = 0, reason_std = 0;
            eval_cell(dem, valid, domain_mask, height, width, nodata,
                      r0, r1, c0, c1, std_threshold, relief_threshold,
                      &should_split, &reason_std);
            if (should_split) {
                qtcl_push(&to_split, level, i, j);
            }
        }
#endif

        if (to_split.len == 0) {
            free(pending.items);
            free(to_split.items);
            break;
        }

        for (size_t s = 0; s < to_split.len; ++s) {
            int level = to_split.items[s].level;
            int i = to_split.items[s].i;
            int j = to_split.items[s].j;
            int child_level = level + 1;
            int children[4][2] = {
                {2 * i, 2 * j}, {2 * i + 1, 2 * j}, {2 * i, 2 * j + 1}, {2 * i + 1, 2 * j + 1}
            };

            for (int c = 0; c < 4; ++c) {
                int ci = children[c][0];
                int cj = children[c][1];
                int r0, r1, c0, c1;
                bounds_to_slice(origin_x, origin_y, base_cell, child_level, ci, cj,
                                height, width, transform_a, transform_e, transform_c, transform_f,
                                &r0, &r1, &c0, &c1);
                if (slab_has_domain(domain_mask, height, width, r0, r1, c0, c1)) {
                    qtcl_push(&leaves, child_level, ci, cj);
                    changed = 1;
                }
            }

            for (size_t k = 0; k < leaves.len; ) {
                if (leaves.items[k].level == level &&
                    leaves.items[k].i == i &&
                    leaves.items[k].j == j) {
                    leaves.items[k] = leaves.items[leaves.len - 1];
                    leaves.len--;
                } else {
                    k++;
                }
            }
        }

        free(pending.items);
        free(to_split.items);
    }

    size_t nout = leaves.len < (size_t)max_out ? leaves.len : (size_t)max_out;
    for (size_t k = 0; k < nout; ++k) {
        out_levels[k] = leaves.items[k].level;
        out_is[k] = leaves.items[k].i;
        out_js[k] = leaves.items[k].j;
    }
    *out_count = (int)nout;
    free(leaves.items);
    return 0;
}

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
) {
    int count = 0;
    for (int level = 0; level <= max_level; ++level) {
        double step = base_cell / (double)(1 << level);
        int i0 = (int)floor((poly_minx - origin_x) / step);
        int i1 = (int)floor((poly_maxx - origin_x) / step);
        int j0 = (int)floor((poly_miny - origin_y) / step);
        int j1 = (int)floor((poly_maxy - origin_y) / step);
        for (int i = i0; i <= i1; ++i) {
            for (int j = j0; j <= j1; ++j) {
                double cx0 = origin_x + i * step;
                double cy0 = origin_y + j * step;
                double cx1 = cx0 + step;
                double cy1 = cy0 + step;
                if (cx1 <= poly_minx || cx0 >= poly_maxx || cy1 <= poly_miny || cy0 >= poly_maxy) {
                    continue;
                }
                if (count >= max_out) {
                    *out_count = count;
                    return 0;
                }
                out_levels[count] = level;
                out_is[count] = i;
                out_js[count] = j;
                count++;
            }
        }
    }
    *out_count = count;
    return 0;
}
