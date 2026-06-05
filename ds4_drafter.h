#ifndef DS4_DRAFTER_H
#define DS4_DRAFTER_H

#include "ds4.h"

#include <stdio.h>
#include <sys/types.h>

typedef enum {
    DS4_DRAFTER_BACKEND_PYTHON = 0,
    DS4_DRAFTER_BACKEND_NATIVE = 1,
} ds4_drafter_backend;

typedef struct {
    ds4_drafter_backend backend;
    const char *model;
    const char *python;
    const char *script;
    const char *tokenizer;
    int score_lookahead;
    int score_pool_kernel;
    float keep_pct;
    int sink;
    int tail;
    int chunk;
    const char *process_name;
} ds4_drafter_options;

typedef struct {
    ds4_drafter_backend active_backend;
    pid_t pid;
    int in_fd;
    FILE *out_fp;
    void *native;
    char ready_detail[256];
} ds4_drafter;

typedef struct {
    double total_ms;
    double tokenize_ms;
    double score_ms;
    double align_ms;
} ds4_drafter_score_stats;

void ds4_drafter_init(ds4_drafter *d);
void ds4_drafter_stop(ds4_drafter *d);

int ds4_drafter_start(ds4_drafter *d,
                      const ds4_drafter_options *opt,
                      char *err,
                      size_t errlen);

int ds4_drafter_score(ds4_drafter *d,
                      const ds4_drafter_options *opt,
                      ds4_engine *engine,
                      const ds4_tokens *prompt,
                      float **scores_out,
                      int *scores_len_out,
                      ds4_drafter_score_stats *stats_out,
                      char *err,
                      size_t errlen);

#endif
