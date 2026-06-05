#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "ds4_drafter.h"
#if defined(DS4_DRAFTER_HAS_METAL)
#include "ds4_drafter_metal.h"
#endif

#include <errno.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/wait.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    int quant_bits;
    int quant_group_size;
    int full_attention_layers;
    int linear_attention_layers;
    int tokenizer_vocab_size;
    int tokenizer_merges;
    int safetensors_tensors;
} ds4_drafter_native_config;

enum { DS4_DRAFTER_MAX_DIMS = 8 };

typedef struct {
    char *name;
    char dtype[8];
    int n_dims;
    int64_t shape[DS4_DRAFTER_MAX_DIMS];
    uint64_t begin;
    uint64_t end;
    float *dequant_f32;
    int dequant_rows;
    int dequant_cols;
} ds4_drafter_tensor_meta;

typedef struct {
    const char *ptr;
    uint64_t len;
} native_str;

typedef struct {
    native_str key;
    int value;
    bool used;
} native_str_i32_entry;

typedef struct {
    native_str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} native_str_i32_table;

typedef struct {
    int *ids;
    uint32_t *offsets;
    int len;
    int cap;
} native_token_vec;

typedef struct {
    char *content;
    int id;
} native_added_token;

typedef struct {
    char **token;
    uint32_t *token_len;
    int n_vocab_slots;
    native_added_token *added;
    int n_added;
    int cap_added;
    char **merge_storage;
    int n_merge_storage;
    int cap_merge_storage;
    native_str_i32_table token_to_id;
    native_str_i32_table merge_rank;
} native_tokenizer;

#if defined(DS4_DRAFTER_HAS_METAL)
typedef struct {
    int initialized;
    const uint8_t *input_norm_data;
    uint64_t input_norm_bytes;
    const uint8_t *post_norm_data;
    uint64_t post_norm_bytes;
    const uint8_t *conv_data;
    uint64_t conv_bytes;
    const uint8_t *linear_norm_data;
    uint64_t linear_norm_bytes;
    const uint8_t *a_log_data;
    uint64_t a_log_bytes;
    const uint8_t *dt_bias_data;
    uint64_t dt_bias_bytes;
    ds4_drafter_metal_affine_job qkv_job;
    ds4_drafter_metal_affine_job z_job;
    ds4_drafter_metal_affine_job b_job;
    ds4_drafter_metal_affine_job a_job;
    ds4_drafter_metal_affine_job linear_out_job;
    ds4_drafter_metal_affine_job mlp_gate_job;
    ds4_drafter_metal_affine_job mlp_up_job;
    ds4_drafter_metal_affine_job mlp_down_job;
} native_metal_linear_layer_cache;

typedef struct {
    int initialized;
    int decoder_initialized;
    const uint8_t *input_norm_data;
    uint64_t input_norm_bytes;
    const uint8_t *post_norm_data;
    uint64_t post_norm_bytes;
    const uint8_t *q_norm_data;
    uint64_t q_norm_bytes;
    const uint8_t *k_norm_data;
    uint64_t k_norm_bytes;
    ds4_drafter_metal_affine_job q_job;
    ds4_drafter_metal_affine_job k_job;
    ds4_drafter_metal_affine_job v_job;
    ds4_drafter_metal_affine_job o_job;
    ds4_drafter_metal_affine_job mlp_gate_job;
    ds4_drafter_metal_affine_job mlp_up_job;
    ds4_drafter_metal_affine_job mlp_down_job;
} native_metal_full_layer_cache;
#endif

typedef struct {
    ds4_drafter_native_config cfg;
    char *safetensors_path;
    int safetensors_fd;
    uint8_t *safetensors_map;
    size_t safetensors_size;
    uint64_t safetensors_data_base;
    ds4_drafter_tensor_meta *tensors;
    int n_tensors;
    int cap_tensors;
    native_str_i32_table tensor_to_index;
    native_tokenizer tokenizer;
#if defined(DS4_DRAFTER_HAS_METAL)
    native_metal_linear_layer_cache metal_linear_cache[24];
    native_metal_full_layer_cache metal_full_cache[24];
#endif
} ds4_drafter_native_model;

typedef struct {
    float *keys;
    float *values;
    int len;
    int cap;
    int metal_cache_valid;
    int metal_cache_cap;
} native_full_attention_state;

typedef struct {
    float *linear_conv[24];
    float *linear_delta[24];
    native_full_attention_state full[24];
    int position;
} native_qwen_runtime;

typedef struct {
    double embed_ms;
    double linear_ms;
    double full_ms;
    double final_norm_ms;
    double resident_ms;
    int tokens;
    int linear_layers;
    int full_layers;
    int resident_tokens;
} native_qwen_step_profile;

void ds4_drafter_init(ds4_drafter *d) {
    if (!d) return;
    d->active_backend = DS4_DRAFTER_BACKEND_PYTHON;
    d->pid = 0;
    d->in_fd = -1;
    d->out_fp = NULL;
    d->native = NULL;
    d->ready_detail[0] = '\0';
}

static uint64_t native_next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static uint64_t native_hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool native_str_eq(native_str a, native_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, (size_t)a.len) == 0;
}

static int native_table_init(native_str_i32_table *t, uint64_t expected) {
    t->cap = native_next_pow2(expected * 2u + 16u);
    t->used = 0;
    t->entry = calloc((size_t)t->cap, sizeof(t->entry[0]));
    return t->entry ? 0 : -1;
}

static void native_table_free(native_str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void native_table_put(native_str_i32_table *t, native_str key, int value) {
    uint64_t mask = t->cap - 1u;
    uint64_t i = native_hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (native_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1u) & mask;
    }
    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool native_table_get(const native_str_i32_table *t,
                             const char *ptr,
                             uint64_t len,
                             int *value) {
    if (t->cap == 0) return false;
    uint64_t mask = t->cap - 1u;
    uint64_t i = native_hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        native_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, (size_t)len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1u) & mask;
    }
    return false;
}

static int native_token_vec_push(native_token_vec *v,
                                 int token,
                                 uint32_t start,
                                 uint32_t end) {
    if (v->len == v->cap) {
        int next = v->cap ? v->cap * 2 : 64;
        int *ids = realloc(v->ids, (size_t)next * sizeof(ids[0]));
        if (!ids) return -1;
        v->ids = ids;
        uint32_t *offsets = realloc(v->offsets, (size_t)next * 2u * sizeof(offsets[0]));
        if (!offsets) return -1;
        v->offsets = offsets;
        v->cap = next;
    }
    v->ids[v->len] = token;
    v->offsets[(size_t)v->len * 2u + 0u] = start;
    v->offsets[(size_t)v->len * 2u + 1u] = end;
    v->len++;
    return 0;
}

static void native_token_vec_free(native_token_vec *v) {
    free(v->ids);
    free(v->offsets);
    memset(v, 0, sizeof(*v));
}

static void native_tokenizer_free(native_tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->n_vocab_slots; i++) {
        free(tok->token[i]);
    }
    for (int i = 0; i < tok->n_added; i++) {
        free(tok->added[i].content);
    }
    for (int i = 0; i < tok->n_merge_storage; i++) {
        free(tok->merge_storage[i]);
    }
    free(tok->token);
    free(tok->token_len);
    free(tok->added);
    free(tok->merge_storage);
    native_table_free(&tok->token_to_id);
    native_table_free(&tok->merge_rank);
    memset(tok, 0, sizeof(*tok));
}

static int native_tokenizer_ensure_slot(native_tokenizer *tok, int id) {
    if (id < 0) return -1;
    if (id < tok->n_vocab_slots) return 0;
    int next = tok->n_vocab_slots ? tok->n_vocab_slots : 262144;
    while (id >= next) next *= 2;
    char **token = realloc(tok->token, (size_t)next * sizeof(token[0]));
    if (!token) return -1;
    tok->token = token;
    uint32_t *token_len = realloc(tok->token_len, (size_t)next * sizeof(token_len[0]));
    if (!token_len) return -1;
    tok->token_len = token_len;
    for (int i = tok->n_vocab_slots; i < next; i++) {
        tok->token[i] = NULL;
        tok->token_len[i] = 0;
    }
    tok->n_vocab_slots = next;
    return 0;
}

static int native_tokenizer_add_token(native_tokenizer *tok,
                                      char *content,
                                      int id,
                                      bool added) {
    if (native_tokenizer_ensure_slot(tok, id) != 0) return -1;
    free(tok->token[id]);
    tok->token[id] = content;
    tok->token_len[id] = (uint32_t)strlen(content);
    native_table_put(&tok->token_to_id,
                     (native_str){tok->token[id], tok->token_len[id]},
                     id);
    if (added) {
        if (tok->n_added == tok->cap_added) {
            int next = tok->cap_added ? tok->cap_added * 2 : 64;
            native_added_token *v =
                realloc(tok->added, (size_t)next * sizeof(v[0]));
            if (!v) return -1;
            tok->added = v;
            tok->cap_added = next;
        }
        tok->added[tok->n_added].content = malloc((size_t)tok->token_len[id] + 1u);
        if (!tok->added[tok->n_added].content) return -1;
        memcpy(tok->added[tok->n_added].content,
               tok->token[id],
               (size_t)tok->token_len[id] + 1u);
        tok->added[tok->n_added].id = id;
        tok->n_added++;
    }
    return 0;
}

static void native_model_free(ds4_drafter_native_model *m) {
    if (!m) return;
#if defined(DS4_DRAFTER_HAS_METAL)
    ds4_drafter_metal_shutdown();
#endif
    if (m->safetensors_map && m->safetensors_size > 0) {
        munmap(m->safetensors_map, m->safetensors_size);
    }
    if (m->safetensors_fd >= 0) close(m->safetensors_fd);
    for (int i = 0; i < m->n_tensors; i++) {
        free(m->tensors[i].name);
        free(m->tensors[i].dequant_f32);
    }
    native_table_free(&m->tensor_to_index);
    free(m->tensors);
    free(m->safetensors_path);
    native_tokenizer_free(&m->tokenizer);
    free(m);
}

static int write_all_fd(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static char *render_tokens_text_with_spans(ds4_engine *engine,
                                           const ds4_tokens *tokens,
                                           size_t *out_len,
                                           uint32_t **spans_out) {
    size_t len = 0;
    size_t cap = 4096;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    uint32_t *spans = NULL;
    if (spans_out) {
        spans = malloc((size_t)tokens->len * 2u * sizeof(spans[0]));
        if (!spans) {
            free(buf);
            return NULL;
        }
    }
    for (int i = 0; i < tokens->len; i++) {
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &piece_len);
        if (!piece && piece_len > 0) {
            free(spans);
            free(buf);
            return NULL;
        }
        const size_t start = len;
        if (len + piece_len + 1 > cap) {
            while (len + piece_len + 1 > cap) cap *= 2;
            char *next = realloc(buf, cap);
            if (!next) {
                free(piece);
                free(spans);
                free(buf);
                return NULL;
            }
            buf = next;
        }
        if (piece_len > 0) memcpy(buf + len, piece, piece_len);
        len += piece_len;
        if (spans) {
            spans[(size_t)i * 2u + 0u] =
                start > UINT32_MAX ? UINT32_MAX : (uint32_t)start;
            spans[(size_t)i * 2u + 1u] =
                len > UINT32_MAX ? UINT32_MAX : (uint32_t)len;
        }
        free(piece);
    }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    if (spans_out) *spans_out = spans;
    return buf;
}

void ds4_drafter_stop(ds4_drafter *d) {
    if (!d) return;
    if (d->active_backend == DS4_DRAFTER_BACKEND_NATIVE) {
        native_model_free((ds4_drafter_native_model *)d->native);
        d->native = NULL;
        d->active_backend = DS4_DRAFTER_BACKEND_PYTHON;
        d->ready_detail[0] = '\0';
        return;
    }
    if (d->in_fd > 0) {
        (void)write_all_fd(d->in_fd, "QUIT\n", 5);
        close(d->in_fd);
        d->in_fd = -1;
    }
    if (d->out_fp) {
        fclose(d->out_fp);
        d->out_fp = NULL;
    }
    if (d->pid > 0) {
        waitpid(d->pid, NULL, 0);
        d->pid = 0;
    }
    d->ready_detail[0] = '\0';
}

static char *read_file_text(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = malloc((size_t)n + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

static char *read_safetensors_header(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    unsigned char raw_len[8];
    if (fread(raw_len, 1, sizeof(raw_len), fp) != sizeof(raw_len)) {
        fclose(fp);
        return NULL;
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) {
        n = (n << 8) | (uint64_t)raw_len[i];
    }
    if (n == 0 || n > 256u * 1024u * 1024u) {
        fclose(fp);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

static int json_int_after_key(const char *json, const char *key, int *out) {
    const char *p = strstr(json, key);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p || v < 0 || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int count_layer_types(const char *json, int *full_out, int *linear_out) {
    const char *p = strstr(json, "\"layer_types\"");
    if (!p) return -1;
    const char *end = strchr(p, ']');
    if (!end) return -1;
    int full = 0;
    int linear = 0;
    const char *q = p;
    while ((q = strstr(q, "\"full_attention\"")) && q < end) {
        full++;
        q++;
    }
    q = p;
    while ((q = strstr(q, "\"linear_attention\"")) && q < end) {
        linear++;
        q++;
    }
    *full_out = full;
    *linear_out = linear;
    return 0;
}

static int path_join(char *dst, size_t dstlen, const char *dir, const char *file) {
    if (!dst || !dir || !file) return -1;
    size_t n = strlen(dir);
    const char *sep = (n > 0 && dir[n - 1] == '/') ? "" : "/";
    int rc = snprintf(dst, dstlen, "%s%s%s", dir, sep, file);
    return (rc > 0 && (size_t)rc < dstlen) ? 0 : -1;
}

static const char *json_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static void native_utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static int json_hex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        unsigned char c = (unsigned char)p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static char *json_parse_string_dup(const char **pp) {
    const char *p = json_ws(*pp);
    if (*p != '"') return NULL;
    p++;
    const char *start = p;
    bool esc = false;
    for (; *p; p++) {
        if (esc) {
            esc = false;
        } else if (*p == '\\') {
            esc = true;
        } else if (*p == '"') {
            break;
        }
    }
    if (*p != '"') return NULL;
    size_t n = (size_t)(p - start);
    char *out = malloc(n * 4u + 1u);
    if (!out) return NULL;
    char *w = out;
    const char *r = start;
    while (r < p) {
        unsigned char c = (unsigned char)*r++;
        if (c != '\\') {
            *w++ = (char)c;
            continue;
        }
        if (r >= p) {
            free(out);
            return NULL;
        }
        char esc_ch = *r++;
        switch (esc_ch) {
        case '"': *w++ = '"'; break;
        case '\\': *w++ = '\\'; break;
        case '/': *w++ = '/'; break;
        case 'b': *w++ = '\b'; break;
        case 'f': *w++ = '\f'; break;
        case 'n': *w++ = '\n'; break;
        case 'r': *w++ = '\r'; break;
        case 't': *w++ = '\t'; break;
        case 'u': {
            if (p - r < 4) {
                free(out);
                return NULL;
            }
            uint32_t cp = 0;
            if (json_hex4(r, &cp) != 0) {
                free(out);
                return NULL;
            }
            r += 4;
            if (cp >= 0xd800 && cp <= 0xdbff && p - r >= 6 &&
                r[0] == '\\' && r[1] == 'u') {
                uint32_t lo = 0;
                if (json_hex4(r + 2, &lo) == 0 && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                    r += 6;
                }
            }
            native_utf8_put(&w, cp);
            break;
        }
        default:
            free(out);
            return NULL;
        }
    }
    *w = '\0';
    *pp = p + 1;
    return out;
}

static int json_skip_value(const char **pp) {
    const char *p = json_ws(*pp);
    if (*p == '"') {
        char *s = json_parse_string_dup(&p);
        if (!s) return -1;
        free(s);
        *pp = p;
        return 0;
    }
    if (*p == '{' || *p == '[') {
        char open = *p++;
        char close = open == '{' ? '}' : ']';
        int depth = 1;
        bool in_string = false;
        bool esc = false;
        for (; *p; p++) {
            if (in_string) {
                if (esc) esc = false;
                else if (*p == '\\') esc = true;
                else if (*p == '"') in_string = false;
                continue;
            }
            if (*p == '"') in_string = true;
            else if (*p == open) depth++;
            else if (*p == close && --depth == 0) {
                *pp = p + 1;
                return 0;
            }
        }
        return -1;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    *pp = p;
    return 0;
}

static int json_expect_char(const char **pp, char c) {
    const char *p = json_ws(*pp);
    if (*p != c) return -1;
    *pp = p + 1;
    return 0;
}

static int json_parse_i64_value(const char **pp, int64_t *out) {
    const char *p = json_ws(*pp);
    char *end = NULL;
    long long v = strtoll(p, &end, 10);
    if (end == p) return -1;
    *out = (int64_t)v;
    *pp = end;
    return 0;
}

static uint32_t native_gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static char *native_byte_encode(native_str in,
                                uint32_t base_offset,
                                uint32_t **byte_offsets_out,
                                uint64_t *out_len) {
    char *out = malloc((size_t)in.len * 4u + 1u);
    uint32_t *byte_offsets = malloc((size_t)in.len * 4u * sizeof(byte_offsets[0]));
    if (!out || !byte_offsets) {
        free(out);
        free(byte_offsets);
        return NULL;
    }
    char *p = out;
    uint32_t *op = byte_offsets;
    for (uint64_t i = 0; i < in.len; i++) {
        char *before = p;
        native_utf8_put(&p, native_gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
        while (before < p) {
            *op++ = base_offset + (uint32_t)i;
            before++;
        }
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    *byte_offsets_out = byte_offsets;
    return out;
}

static int native_utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
    uint32_t start;
    uint32_t end;
} native_owned_symbol;

static native_owned_symbol native_owned_symbol_copy(const char *ptr,
                                                    uint64_t len,
                                                    uint32_t start,
                                                    uint32_t end) {
    native_owned_symbol s;
    s.ptr = malloc((size_t)len);
    if (s.ptr) memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    s.start = start;
    s.end = end;
    return s;
}

static int native_bpe_rank(const native_tokenizer *tok,
                           const native_owned_symbol *a,
                           const native_owned_symbol *b) {
    uint64_t len = a->len + 1u + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : malloc((size_t)len);
    if (!buf) return -1;
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1u, b->ptr, (size_t)b->len);
    int rank = -1;
    native_table_get(&tok->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static int native_bpe_emit_piece(const native_tokenizer *tok,
                                 native_str raw_piece,
                                 uint32_t base_offset,
                                 native_token_vec *out,
                                 char *err,
                                 size_t errlen) {
    uint64_t encoded_len = 0;
    uint32_t *byte_offsets = NULL;
    char *encoded = native_byte_encode(raw_piece, base_offset,
                                       &byte_offsets, &encoded_len);
    if (!encoded) {
        snprintf(err, errlen, "native drafter tokenizer out of memory");
        return -1;
    }

    int n_sym = 0;
    int cap_sym = 32;
    native_owned_symbol *sym = calloc((size_t)cap_sym, sizeof(sym[0]));
    if (!sym) {
        free(byte_offsets);
        free(encoded);
        snprintf(err, errlen, "native drafter tokenizer out of memory");
        return -1;
    }

    for (uint64_t off = 0; off < encoded_len;) {
        int n = native_utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            native_owned_symbol *next =
                realloc(sym, (size_t)cap_sym * sizeof(sym[0]));
            if (!next) {
                for (int i = 0; i < n_sym; i++) free(sym[i].ptr);
                free(sym);
                free(byte_offsets);
                free(encoded);
                snprintf(err, errlen, "native drafter tokenizer out of memory");
                return -1;
            }
            sym = next;
        }
        uint32_t start = byte_offsets[off];
        uint32_t end = byte_offsets[off + (uint64_t)n - 1u] + 1u;
        sym[n_sym] = native_owned_symbol_copy(encoded + off, (uint64_t)n, start, end);
        if (!sym[n_sym].ptr) {
            for (int i = 0; i < n_sym; i++) free(sym[i].ptr);
            free(sym);
            free(byte_offsets);
            free(encoded);
            snprintf(err, errlen, "native drafter tokenizer out of memory");
            return -1;
        }
        n_sym++;
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = native_bpe_rank(tok, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        native_owned_symbol merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = malloc((size_t)merged.len);
        if (!merged.ptr) {
            for (int i = 0; i < n_sym; i++) free(sym[i].ptr);
            free(sym);
            free(byte_offsets);
            free(encoded);
            snprintf(err, errlen, "native drafter tokenizer out of memory");
            return -1;
        }
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len,
               sym[best_i + 1].ptr,
               (size_t)sym[best_i + 1].len);
        merged.start = sym[best_i].start;
        merged.end = sym[best_i + 1].end;

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (!native_table_get(&tok->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            snprintf(err, errlen, "native drafter tokenizer missing BPE token");
            for (int j = 0; j < n_sym; j++) free(sym[j].ptr);
            free(sym);
            free(byte_offsets);
            free(encoded);
            return -1;
        }
        if (native_token_vec_push(out, token, sym[i].start, sym[i].end) != 0) {
            snprintf(err, errlen, "native drafter tokenizer out of memory");
            for (int j = 0; j < n_sym; j++) free(sym[j].ptr);
            free(sym);
            free(byte_offsets);
            free(encoded);
            return -1;
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(byte_offsets);
    free(encoded);
    return 0;
}

static bool native_ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool native_ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool native_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool native_ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool native_qwen_letter_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return native_ascii_alpha(c);
    if (pos + 3u <= len &&
        (uint8_t)s[pos] == 0xef &&
        (uint8_t)s[pos + 1u] == 0xbd &&
        (uint8_t)s[pos + 2u] == 0x9c) {
        return false;
    }
    return true;
}

static uint64_t native_next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = native_utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static uint64_t native_qwen_consume_letters(const char *s,
                                            uint64_t len,
                                            uint64_t pos) {
    while (pos < len && native_qwen_letter_at(s, len, pos)) {
        pos = native_next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool native_match_ascii_ci(const char *s,
                                  uint64_t len,
                                  uint64_t pos,
                                  const char *pat) {
    uint64_t n = (uint64_t)strlen(pat);
    if (pos + n > len) return false;
    for (uint64_t i = 0; i < n; i++) {
        unsigned char a = (unsigned char)s[pos + i];
        unsigned char b = (unsigned char)pat[i];
        if (a >= 'A' && a <= 'Z') a = (unsigned char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (unsigned char)(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

static uint64_t native_qwen_contraction_len(const char *s,
                                            uint64_t len,
                                            uint64_t pos) {
    static const char *const pats[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
    };
    for (size_t i = 0; i < sizeof(pats) / sizeof(pats[0]); i++) {
        if (native_match_ascii_ci(s, len, pos, pats[i])) return strlen(pats[i]);
    }
    return 0;
}

static int native_added_token_at(const native_tokenizer *tok,
                                 const char *text,
                                 uint64_t len,
                                 uint64_t pos,
                                 int *id_out,
                                 uint64_t *match_len_out) {
    int best_id = -1;
    uint64_t best_len = 0;
    for (int i = 0; i < tok->n_added; i++) {
        uint64_t n = (uint64_t)strlen(tok->added[i].content);
        if (n <= best_len || pos + n > len) continue;
        if (memcmp(text + pos, tok->added[i].content, (size_t)n) == 0) {
            best_id = tok->added[i].id;
            best_len = n;
        }
    }
    if (best_id < 0) return 0;
    *id_out = best_id;
    *match_len_out = best_len;
    return 1;
}

static int native_tokenizer_encode_text(const native_tokenizer *tok,
                                        const char *text,
                                        size_t text_len,
                                        native_token_vec *out,
                                        char *err,
                                        size_t errlen) {
    const uint64_t len = (uint64_t)text_len;
    uint64_t pos = 0;
    while (pos < len) {
        int added_id = -1;
        uint64_t added_len = 0;
        if (native_added_token_at(tok, text, len, pos, &added_id, &added_len)) {
            if (native_token_vec_push(out, added_id, (uint32_t)pos,
                                      (uint32_t)(pos + added_len)) != 0) {
                snprintf(err, errlen, "native drafter tokenizer out of memory");
                return -1;
            }
            pos += added_len;
            continue;
        }

        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];
        uint64_t clen = native_qwen_contraction_len(text, len, pos);
        if (clen > 0) {
            pos += clen;
        } else if (native_qwen_letter_at(text, len, pos)) {
            pos = native_qwen_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   native_qwen_letter_at(text, len, pos + 1)) {
            pos++;
            pos = native_qwen_consume_letters(text, len, pos);
        } else if (!native_ascii_space(c) &&
                   !native_ascii_digit(c) &&
                   !native_qwen_letter_at(text, len, pos)) {
            uint64_t next = native_next_utf8_char(text, len, pos);
            if (next >= len || !native_qwen_letter_at(text, len, next)) {
                pos = start;
                goto native_qwen_non_word_piece;
            }
            pos = next;
            pos = native_qwen_consume_letters(text, len, pos);
        } else if (native_ascii_digit(c)) {
            pos++;
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   !native_ascii_space((uint8_t)text[pos + 1]) &&
                   !native_ascii_digit((uint8_t)text[pos + 1]) &&
                   !native_qwen_letter_at(text, len, pos + 1)) {
            pos++;
            while (pos < len &&
                   !native_ascii_space((uint8_t)text[pos]) &&
                   !native_ascii_digit((uint8_t)text[pos]) &&
                   !native_qwen_letter_at(text, len, pos)) {
                int boundary_id = -1;
                uint64_t boundary_len = 0;
                if (pos > start &&
                    native_added_token_at(tok, text, len, pos,
                                          &boundary_id, &boundary_len)) {
                    break;
                }
                pos = native_next_utf8_char(text, len, pos);
            }
            while (pos < len && native_ascii_newline((uint8_t)text[pos])) pos++;
        } else if (!native_ascii_space(c) &&
                   !native_ascii_digit(c) &&
                   !native_qwen_letter_at(text, len, pos)) {
native_qwen_non_word_piece:
            while (pos < len &&
                   !native_ascii_space((uint8_t)text[pos]) &&
                   !native_ascii_digit((uint8_t)text[pos]) &&
                   !native_qwen_letter_at(text, len, pos)) {
                int boundary_id = -1;
                uint64_t boundary_len = 0;
                if (pos > start &&
                    native_added_token_at(tok, text, len, pos,
                                          &boundary_id, &boundary_len)) {
                    break;
                }
                pos = native_next_utf8_char(text, len, pos);
            }
            while (pos < len && native_ascii_newline((uint8_t)text[pos])) pos++;
        } else if (native_ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && native_ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (native_ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = native_next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = native_next_utf8_char(text, len, pos);
        if (native_bpe_emit_piece(tok,
                                  (native_str){text + start, pos - start},
                                  (uint32_t)start,
                                  out,
                                  err,
                                  errlen) != 0) {
            return -1;
        }
    }
    return 0;
}

static int native_tensor_push(ds4_drafter_native_model *m,
                              ds4_drafter_tensor_meta meta) {
    if (m->n_tensors == m->cap_tensors) {
        int next = m->cap_tensors ? m->cap_tensors * 2 : 256;
        ds4_drafter_tensor_meta *v =
            realloc(m->tensors, (size_t)next * sizeof(v[0]));
        if (!v) return -1;
        m->tensors = v;
        m->cap_tensors = next;
    }
    m->tensors[m->n_tensors++] = meta;
    return 0;
}

static int parse_safetensors_shape_array(const char **pp,
                                         ds4_drafter_tensor_meta *meta) {
    if (json_expect_char(pp, '[') != 0) return -1;
    int nd = 0;
    const char *p = json_ws(*pp);
    while (*p && *p != ']') {
        if (nd >= DS4_DRAFTER_MAX_DIMS) return -1;
        int64_t dim = 0;
        if (json_parse_i64_value(&p, &dim) != 0 || dim < 0) return -1;
        meta->shape[nd++] = dim;
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    if (*p != ']') return -1;
    meta->n_dims = nd;
    *pp = p + 1;
    return 0;
}

static int parse_safetensors_offsets_array(const char **pp,
                                           ds4_drafter_tensor_meta *meta) {
    if (json_expect_char(pp, '[') != 0) return -1;
    int64_t begin = 0;
    int64_t end = 0;
    if (json_parse_i64_value(pp, &begin) != 0) return -1;
    if (json_expect_char(pp, ',') != 0) return -1;
    if (json_parse_i64_value(pp, &end) != 0) return -1;
    if (json_expect_char(pp, ']') != 0) return -1;
    if (begin < 0 || end < begin) return -1;
    meta->begin = (uint64_t)begin;
    meta->end = (uint64_t)end;
    return 0;
}

static int parse_safetensors_tensor_object(const char **pp,
                                           ds4_drafter_tensor_meta *meta) {
    if (json_expect_char(pp, '{') != 0) return -1;
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;
    const char *p = json_ws(*pp);
    while (*p && *p != '}') {
        char *key = json_parse_string_dup(&p);
        if (!key) return -1;
        if (json_expect_char(&p, ':') != 0) {
            free(key);
            return -1;
        }
        if (strcmp(key, "dtype") == 0) {
            char *dtype = json_parse_string_dup(&p);
            if (!dtype) {
                free(key);
                return -1;
            }
            snprintf(meta->dtype, sizeof(meta->dtype), "%s", dtype);
            free(dtype);
            have_dtype = true;
        } else if (strcmp(key, "shape") == 0) {
            if (parse_safetensors_shape_array(&p, meta) != 0) {
                free(key);
                return -1;
            }
            have_shape = true;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (parse_safetensors_offsets_array(&p, meta) != 0) {
                free(key);
                return -1;
            }
            have_offsets = true;
        } else if (json_skip_value(&p) != 0) {
            free(key);
            return -1;
        }
        free(key);
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    if (*p != '}') return -1;
    *pp = p + 1;
    return have_dtype && have_shape && have_offsets ? 0 : -1;
}

static uint64_t native_tensor_elems(const ds4_drafter_tensor_meta *t) {
    uint64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) {
        if (t->shape[i] <= 0) return 0;
        n *= (uint64_t)t->shape[i];
    }
    return n;
}

static uint64_t native_dtype_size(const char *dtype) {
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "U32") == 0) return 4;
    return 0;
}

static const ds4_drafter_tensor_meta *native_find_tensor(
        const ds4_drafter_native_model *m,
        const char *name) {
    int index = -1;
    if (m->tensor_to_index.cap > 0 &&
        native_table_get(&m->tensor_to_index, name, strlen(name), &index) &&
        index >= 0 && index < m->n_tensors) {
        return &m->tensors[index];
    }
    for (int i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    return NULL;
}

static int native_build_tensor_index(ds4_drafter_native_model *m,
                                     char *err,
                                     size_t errlen) {
    if (m->tensor_to_index.cap > 0) return 0;
    if (native_table_init(&m->tensor_to_index,
                          (uint64_t)(m->n_tensors > 0 ? m->n_tensors : 1)) != 0) {
        snprintf(err, errlen, "native drafter out of memory indexing tensors");
        return -1;
    }
    for (int i = 0; i < m->n_tensors; i++) {
        native_table_put(&m->tensor_to_index,
                         (native_str){m->tensors[i].name,
                                      (uint64_t)strlen(m->tensors[i].name)},
                         i);
    }
    return 0;
}

static uint16_t native_read_le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t native_read_le32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float native_bf16_to_f32(uint16_t v) {
    uint32_t raw = (uint32_t)v << 16;
    float out;
    memcpy(&out, &raw, sizeof(out));
    return out;
}

static uint16_t native_f32_to_bf16(float v) {
    uint32_t raw;
    memcpy(&raw, &v, sizeof(raw));
    uint32_t lsb = (raw >> 16) & 1u;
    raw += 0x7fffu + lsb;
    return (uint16_t)(raw >> 16);
}

static const uint8_t *native_tensor_data_ptr(const ds4_drafter_native_model *m,
                                             const ds4_drafter_tensor_meta *t,
                                             char *err,
                                             size_t errlen) {
    if (!m->safetensors_map || m->safetensors_size == 0) {
        snprintf(err, errlen, "native drafter safetensors is not mapped");
        return NULL;
    }
    uint64_t begin = m->safetensors_data_base + t->begin;
    uint64_t end = m->safetensors_data_base + t->end;
    if (end < begin || end > (uint64_t)m->safetensors_size) {
        snprintf(err, errlen, "native drafter tensor %s is outside mapped file", t->name);
        return NULL;
    }
    return m->safetensors_map + begin;
}

static int native_map_safetensors(ds4_drafter_native_model *m,
                                  char *err,
                                  size_t errlen) {
    if (m->safetensors_map) return 0;
    int fd = open(m->safetensors_path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, errlen, "native drafter failed to open %s: %s",
                 m->safetensors_path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        snprintf(err, errlen, "native drafter failed to stat %s: %s",
                 m->safetensors_path, strerror(errno));
        close(fd);
        return -1;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        snprintf(err, errlen, "native drafter failed to mmap %s: %s",
                 m->safetensors_path, strerror(errno));
        close(fd);
        return -1;
    }
    m->safetensors_fd = fd;
    m->safetensors_map = map;
    m->safetensors_size = (size_t)st.st_size;
    return 0;
}

static int native_quant_sibling_name(char *dst,
                                     size_t dstlen,
                                     const char *weight_name,
                                     const char *suffix) {
    const char *tail = ".weight";
    size_t n = strlen(weight_name);
    size_t tail_n = strlen(tail);
    if (n <= tail_n || strcmp(weight_name + n - tail_n, tail) != 0) return -1;
    int rc = snprintf(dst, dstlen, "%.*s%s", (int)(n - tail_n), weight_name, suffix);
    return (rc > 0 && (size_t)rc < dstlen) ? 0 : -1;
}

static int native_dequant_affine_u32_row(const ds4_drafter_native_model *m,
                                         const char *weight_name,
                                         int row,
                                         float *out,
                                         int out_len,
                                         char *err,
                                         size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
    if (!t || strcmp(t->dtype, "U32") != 0 || t->n_dims != 2) {
        snprintf(err, errlen, "native drafter expected U32 rank-2 weight tensor: %s",
                 weight_name);
        return -1;
    }

    char scales_name[1024];
    char biases_name[1024];
    if (native_quant_sibling_name(scales_name, sizeof(scales_name),
                                  weight_name, ".scales") != 0 ||
        native_quant_sibling_name(biases_name, sizeof(biases_name),
                                  weight_name, ".biases") != 0) {
        snprintf(err, errlen, "native drafter bad quantized weight name: %s", weight_name);
        return -1;
    }
    const ds4_drafter_tensor_meta *scales = native_find_tensor(m, scales_name);
    const ds4_drafter_tensor_meta *biases = native_find_tensor(m, biases_name);
    if (!scales || !biases ||
        strcmp(scales->dtype, "BF16") != 0 ||
        strcmp(biases->dtype, "BF16") != 0 ||
        scales->n_dims != 2 ||
        biases->n_dims != 2) {
        snprintf(err, errlen, "native drafter missing BF16 scales/biases for %s",
                 weight_name);
        return -1;
    }

    const int bits = m->cfg.quant_bits;
    const int group_size = m->cfg.quant_group_size;
    if (bits <= 0 || 32 % bits != 0 || group_size <= 0) {
        snprintf(err, errlen, "native drafter invalid affine quantization bits/group");
        return -1;
    }
    const int pack = 32 / bits;
    const int rows = (int)t->shape[0];
    const int packed_cols = (int)t->shape[1];
    const int cols = packed_cols * pack;
    const int groups = cols / group_size;
    if (row < 0 || row >= rows ||
        cols != out_len ||
        cols % group_size != 0 ||
        scales->shape[0] != rows ||
        biases->shape[0] != rows ||
        scales->shape[1] != groups ||
        biases->shape[1] != groups) {
        snprintf(err, errlen,
                 "native drafter quantized tensor shape mismatch for %s "
                 "(rows=%d cols=%d groups=%d out_len=%d)",
                 weight_name, rows, cols, groups, out_len);
        return -1;
    }

    const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
    const uint8_t *s_data = native_tensor_data_ptr(m, scales, err, errlen);
    const uint8_t *b_data = native_tensor_data_ptr(m, biases, err, errlen);
    if (!w_data || !s_data || !b_data) return -1;

    const uint8_t *w_row = w_data + (size_t)row * (size_t)packed_cols * 4u;
    const uint8_t *s_row = s_data + (size_t)row * (size_t)groups * 2u;
    const uint8_t *b_row = b_data + (size_t)row * (size_t)groups * 2u;
    const uint32_t mask = (1u << bits) - 1u;
    for (int pc = 0; pc < packed_cols; pc++) {
        uint32_t packed = native_read_le32(w_row + (size_t)pc * 4u);
        for (int lane = 0; lane < pack; lane++) {
            int col = pc * pack + lane;
            int g = col / group_size;
            float scale = native_bf16_to_f32(native_read_le16(s_row + (size_t)g * 2u));
            float bias = native_bf16_to_f32(native_read_le16(b_row + (size_t)g * 2u));
            uint32_t q = (packed >> (lane * bits)) & mask;
            out[col] = (float)q * scale + bias;
        }
    }
    return 0;
}

static int native_load_bf16_vector(const ds4_drafter_native_model *m,
                                   const char *name,
                                   float *out,
                                   int out_len,
                                   char *err,
                                   size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, name);
    if (!t || strcmp(t->dtype, "BF16") != 0 || t->n_dims != 1 ||
        t->shape[0] != out_len) {
        snprintf(err, errlen, "native drafter expected BF16 vector tensor: %s", name);
        return -1;
    }
    const uint8_t *data = native_tensor_data_ptr(m, t, err, errlen);
    if (!data) return -1;
    for (int i = 0; i < out_len; i++) {
        out[i] = native_bf16_to_f32(native_read_le16(data + (size_t)i * 2u));
    }
    return 0;
}

static int native_drafter_threads(void) {
    static int cached = 0;
    if (cached > 0) return cached;
    int n = 1;
    const char *env = getenv("DS4_DRAFTER_THREADS");
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && v > 0 && v <= 64) n = (int)v;
    } else {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online > 1) n = online > 12 ? 12 : (int)online;
    }
    if (n < 1) n = 1;
    cached = n;
    return cached;
}

static int native_load_f32_vector(const ds4_drafter_native_model *m,
                                  const char *name,
                                  float *out,
                                  int out_len,
                                  char *err,
                                  size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, name);
    if (!t || strcmp(t->dtype, "F32") != 0 || t->n_dims != 1 ||
        t->shape[0] != out_len) {
        snprintf(err, errlen, "native drafter expected F32 vector tensor: %s", name);
        return -1;
    }
    const uint8_t *data = native_tensor_data_ptr(m, t, err, errlen);
    if (!data) return -1;
    for (int i = 0; i < out_len; i++) {
        uint32_t bits = native_read_le32(data + (size_t)i * 4u);
        memcpy(out + i, &bits, sizeof(bits));
    }
    return 0;
}

typedef struct {
    const uint8_t *w_data;
    const uint8_t *s_data;
    const uint8_t *b_data;
    const float *f32_data;
    const float *x;
    float *out;
    int row_begin;
    int row_end;
    int packed_cols;
    int groups;
    int bits;
    int group_size;
} native_matvec_task;

static bool native_dequant_cache_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_DEQUANT_CACHE");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_accelerate_enabled(void) {
#if defined(__APPLE__)
    const char *env = getenv("DS4_DRAFTER_ACCELERATE");
    return env && env[0] == '1' && env[1] == '\0';
#else
    return false;
#endif
}

static bool native_attention_accelerate_enabled(void) {
#if defined(__APPLE__)
    const char *env = getenv("DS4_DRAFTER_ATTENTION_ACCELERATE");
    return !(env && env[0] == '0' && env[1] == '\0');
#else
    return false;
#endif
}

static bool native_drafter_profile_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_PROFILE");
    return env && env[0] == '1' && env[1] == '\0';
}

#if defined(DS4_DRAFTER_HAS_METAL)
static bool native_metal_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_metal_strict(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_STRICT");
    return env && env[0] == '1' && env[1] == '\0';
}

static bool native_metal_fused_linear_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_FUSED_LINEAR");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_metal_fused_full_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_FUSED_FULL");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_metal_resident_token_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_RESIDENT_TOKEN");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_metal_resident_cpu_kv_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_RESIDENT_CPU_KV");
    if (env && env[0]) return !(env[0] == '0' && env[1] == '\0');
    return false;
}

static bool native_metal_skip_cpu_kv_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_SKIP_CPU_KV");
    if (env && env[0]) return !(env[0] == '0' && env[1] == '\0');
    return native_metal_strict();
}

static bool native_metal_importance_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_IMPORTANCE");
    return !(env && env[0] == '0' && env[1] == '\0');
}

static bool native_metal_prefill_cache_sync_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_PREFILL_CACHE_SYNC");
    return env && !(env[0] == '0' && env[1] == '\0');
}

static bool native_metal_argmax_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_ARGMAX");
    return env && env[0] == '1' && env[1] == '\0';
}
#endif

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    int initialized;
    int stop;
    int n_workers;
    int generation;
    int task_count;
    int active;
    int busy;
    native_matvec_task *tasks;
    pthread_t threads[63];
} native_matvec_pool;

typedef struct {
    native_matvec_pool *pool;
    int index;
} native_matvec_pool_arg;

static native_matvec_pool g_native_matvec_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .start_cond = PTHREAD_COND_INITIALIZER,
    .done_cond = PTHREAD_COND_INITIALIZER,
};
static native_matvec_pool_arg g_native_matvec_pool_args[63];

static void native_quant_affine_u32_matvec_rows(const native_matvec_task *task) {
    if (task->f32_data) {
        const int cols = task->packed_cols * (32 / task->bits);
        for (int r = task->row_begin; r < task->row_end; r++) {
            const float *w_row = task->f32_data + (size_t)r * (size_t)cols;
            float acc = 0.0f;
            for (int col = 0; col < cols; col++) {
                acc += task->x[col] * w_row[col];
            }
            task->out[r] = acc;
        }
        return;
    }
    const int pack = 32 / task->bits;
    const uint32_t mask = (1u << task->bits) - 1u;
    for (int r = task->row_begin; r < task->row_end; r++) {
        const uint8_t *w_row = task->w_data + (size_t)r * (size_t)task->packed_cols * 4u;
        const uint8_t *s_row = task->s_data + (size_t)r * (size_t)task->groups * 2u;
        const uint8_t *b_row = task->b_data + (size_t)r * (size_t)task->groups * 2u;
        double acc = 0.0;
        for (int pc = 0; pc < task->packed_cols; pc++) {
            uint32_t packed = native_read_le32(w_row + (size_t)pc * 4u);
            for (int lane = 0; lane < pack; lane++) {
                int col = pc * pack + lane;
                int g = col / task->group_size;
                float scale = native_bf16_to_f32(native_read_le16(s_row + (size_t)g * 2u));
                float bias = native_bf16_to_f32(native_read_le16(b_row + (size_t)g * 2u));
                uint32_t q = (packed >> (lane * task->bits)) & mask;
                acc += (double)task->x[col] * ((double)q * (double)scale + (double)bias);
            }
        }
        task->out[r] = (float)acc;
    }
}

static void *native_matvec_thread_main(void *arg) {
    native_quant_affine_u32_matvec_rows((const native_matvec_task *)arg);
    return NULL;
}

static float *native_quant_affine_u32_dequant_cached(ds4_drafter_tensor_meta *t,
                                                     int rows,
                                                     int packed_cols,
                                                     int cols,
                                                     int groups,
                                                     int bits,
                                                     int group_size,
                                                     const uint8_t *w_data,
                                                     const uint8_t *s_data,
                                                     const uint8_t *b_data) {
    if (!native_dequant_cache_enabled()) return NULL;
    if (t->dequant_f32 &&
        t->dequant_rows == rows &&
        t->dequant_cols == cols) {
        return t->dequant_f32;
    }

    free(t->dequant_f32);
    t->dequant_f32 = NULL;
    t->dequant_rows = 0;
    t->dequant_cols = 0;

    if (rows <= 0 || cols <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)cols ||
        (size_t)rows * (size_t)cols > SIZE_MAX / sizeof(float)) {
        return NULL;
    }

    float *cache = malloc((size_t)rows * (size_t)cols * sizeof(cache[0]));
    if (!cache) return NULL;

    const int pack = 32 / bits;
    const uint32_t mask = (1u << bits) - 1u;
    for (int r = 0; r < rows; r++) {
        const uint8_t *w_row = w_data + (size_t)r * (size_t)packed_cols * 4u;
        const uint8_t *s_row = s_data + (size_t)r * (size_t)groups * 2u;
        const uint8_t *b_row = b_data + (size_t)r * (size_t)groups * 2u;
        float *dst = cache + (size_t)r * (size_t)cols;
        for (int pc = 0; pc < packed_cols; pc++) {
            uint32_t packed = native_read_le32(w_row + (size_t)pc * 4u);
            for (int lane = 0; lane < pack; lane++) {
                int col = pc * pack + lane;
                int g = col / group_size;
                float scale = native_bf16_to_f32(native_read_le16(s_row + (size_t)g * 2u));
                float bias = native_bf16_to_f32(native_read_le16(b_row + (size_t)g * 2u));
                uint32_t q = (packed >> (lane * bits)) & mask;
                dst[col] = (float)q * scale + bias;
            }
        }
    }

    t->dequant_f32 = cache;
    t->dequant_rows = rows;
    t->dequant_cols = cols;
    return cache;
}

static void *native_matvec_pool_thread_main(void *arg) {
    native_matvec_pool_arg *parg = (native_matvec_pool_arg *)arg;
    native_matvec_pool *pool = parg->pool;
    int index = parg->index;
    int seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stop && pool->generation == seen_generation) {
            pthread_cond_wait(&pool->start_cond, &pool->mutex);
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        seen_generation = pool->generation;
        native_matvec_task task = pool->tasks[index];
        int should_run = index < pool->task_count && task.row_begin < task.row_end;
        pthread_mutex_unlock(&pool->mutex);

        if (should_run) native_quant_affine_u32_matvec_rows(&task);

        pthread_mutex_lock(&pool->mutex);
        pool->active--;
        if (pool->active == 0) pthread_cond_signal(&pool->done_cond);
        pthread_mutex_unlock(&pool->mutex);
    }
}

static int native_matvec_pool_ensure(int n_threads) {
    if (n_threads <= 1) return 0;
    if (n_threads > 64) n_threads = 64;
    native_matvec_pool *pool = &g_native_matvec_pool;
    pthread_mutex_lock(&pool->mutex);
    if (pool->initialized && pool->n_workers >= n_threads - 1) {
        pthread_mutex_unlock(&pool->mutex);
        return 0;
    }
    if (pool->initialized) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }
    pool->n_workers = n_threads - 1;
    pool->stop = 0;
    pool->generation = 0;
    pool->task_count = 0;
    pool->active = 0;
    pool->busy = 0;
    pool->tasks = NULL;
    for (int i = 0; i < pool->n_workers; i++) {
        g_native_matvec_pool_args[i].pool = pool;
        g_native_matvec_pool_args[i].index = i + 1;
        if (pthread_create(&pool->threads[i], NULL,
                           native_matvec_pool_thread_main,
                           &g_native_matvec_pool_args[i]) != 0) {
            pool->stop = 1;
            pthread_cond_broadcast(&pool->start_cond);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            pthread_mutex_lock(&pool->mutex);
            pool->stop = 0;
            pool->n_workers = 0;
            pthread_mutex_unlock(&pool->mutex);
            return -1;
        }
    }
    pool->initialized = 1;
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

static int native_matvec_pool_run(native_matvec_task *tasks, int n_threads) {
    if (n_threads <= 1) {
        native_quant_affine_u32_matvec_rows(&tasks[0]);
        return 0;
    }
    if (native_matvec_pool_ensure(n_threads) != 0) return -1;

    native_matvec_pool *pool = &g_native_matvec_pool;
    pthread_mutex_lock(&pool->mutex);
    if (pool->busy || pool->n_workers < n_threads - 1) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }
    pool->busy = 1;
    pool->tasks = tasks;
    pool->task_count = n_threads;
    pool->active = n_threads - 1;
    pool->generation++;
    pthread_cond_broadcast(&pool->start_cond);
    pthread_mutex_unlock(&pool->mutex);

    native_quant_affine_u32_matvec_rows(&tasks[0]);

    pthread_mutex_lock(&pool->mutex);
    while (pool->active > 0) pthread_cond_wait(&pool->done_cond, &pool->mutex);
    pool->tasks = NULL;
    pool->task_count = 0;
    pool->busy = 0;
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

static int native_quant_affine_u32_matvec(const ds4_drafter_native_model *m,
                                          const char *weight_name,
                                          const float *x,
                                          int in_len,
                                          float *out,
                                          int out_len,
                                          char *err,
                                          size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
    if (!t || strcmp(t->dtype, "U32") != 0 || t->n_dims != 2) {
        snprintf(err, errlen, "native drafter expected U32 rank-2 weight tensor: %s",
                 weight_name);
        return -1;
    }

    char scales_name[1024];
    char biases_name[1024];
    if (native_quant_sibling_name(scales_name, sizeof(scales_name),
                                  weight_name, ".scales") != 0 ||
        native_quant_sibling_name(biases_name, sizeof(biases_name),
                                  weight_name, ".biases") != 0) {
        snprintf(err, errlen, "native drafter bad quantized weight name: %s", weight_name);
        return -1;
    }
    const ds4_drafter_tensor_meta *scales = native_find_tensor(m, scales_name);
    const ds4_drafter_tensor_meta *biases = native_find_tensor(m, biases_name);
    if (!scales || !biases ||
        strcmp(scales->dtype, "BF16") != 0 ||
        strcmp(biases->dtype, "BF16") != 0 ||
        scales->n_dims != 2 ||
        biases->n_dims != 2) {
        snprintf(err, errlen, "native drafter missing BF16 scales/biases for %s",
                 weight_name);
        return -1;
    }

    const int bits = m->cfg.quant_bits;
    const int group_size = m->cfg.quant_group_size;
    if (bits <= 0 || 32 % bits != 0 || group_size <= 0) {
        snprintf(err, errlen, "native drafter invalid affine quantization bits/group");
        return -1;
    }
    const int pack = 32 / bits;
    const int rows = (int)t->shape[0];
    const int packed_cols = (int)t->shape[1];
    const int cols = packed_cols * pack;
    const int groups = cols / group_size;
    if (rows != out_len ||
        cols != in_len ||
        cols % group_size != 0 ||
        scales->shape[0] != rows ||
        biases->shape[0] != rows ||
        scales->shape[1] != groups ||
        biases->shape[1] != groups) {
        snprintf(err, errlen,
                 "native drafter quantized matvec shape mismatch for %s "
                 "(rows=%d cols=%d groups=%d in=%d out=%d)",
                 weight_name, rows, cols, groups, in_len, out_len);
        return -1;
    }

    const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
    const uint8_t *s_data = native_tensor_data_ptr(m, scales, err, errlen);
    const uint8_t *b_data = native_tensor_data_ptr(m, biases, err, errlen);
    if (!w_data || !s_data || !b_data) return -1;

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled() && native_metal_argmax_enabled()) {
        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_affine_u32_matvec(
            w_data, t->end - t->begin,
            s_data, scales->end - scales->begin,
            b_data, biases->end - biases->begin,
            x, rows, packed_cols, cols, groups, bits, group_size, out,
            metal_err, sizeof(metal_err));
        if (metal_rc == 0) return 0;
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter matvec failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: Metal matvec failed, falling back to CPU: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    const float *f32_data = native_quant_affine_u32_dequant_cached(
        (ds4_drafter_tensor_meta *)t, rows, packed_cols, cols, groups,
        bits, group_size, w_data, s_data, b_data);

#if defined(__APPLE__)
    if (f32_data && native_accelerate_enabled()) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)rows, (int)cols,
                    1.0f, f32_data, (int)cols,
                    x, 1,
                    0.0f, out, 1);
        return 0;
    }
#endif

    int n_threads = native_drafter_threads();
    if (rows < 512) n_threads = 1;
    if (n_threads > rows) n_threads = rows;
    native_matvec_task stack_tasks[16];
    native_matvec_task *tasks = stack_tasks;
    if (n_threads > (int)(sizeof(stack_tasks) / sizeof(stack_tasks[0]))) {
        tasks = calloc((size_t)n_threads, sizeof(tasks[0]));
        if (!tasks) {
            free(tasks);
            snprintf(err, errlen, "native drafter out of memory creating matvec tasks");
            return -1;
        }
    }
    int rows_per = (rows + n_threads - 1) / n_threads;
    for (int tix = 0; tix < n_threads; tix++) {
        int begin = tix * rows_per;
        int end = begin + rows_per;
        if (end > rows) end = rows;
        if (begin >= end) break;
        tasks[tix] = (native_matvec_task){
            .w_data = w_data,
            .s_data = s_data,
            .b_data = b_data,
            .f32_data = f32_data,
            .x = x,
            .out = out,
            .row_begin = begin,
            .row_end = end,
            .packed_cols = packed_cols,
            .groups = groups,
            .bits = bits,
            .group_size = group_size,
        };
    }
    if (native_matvec_pool_run(tasks, n_threads) != 0) {
        pthread_t stack_threads[16];
        unsigned char stack_started[16];
        pthread_t *threads = stack_threads;
        unsigned char *started_flags = stack_started;
        memset(stack_started, 0, sizeof(stack_started));
        if (n_threads > (int)(sizeof(stack_threads) / sizeof(stack_threads[0]))) {
            threads = calloc((size_t)n_threads, sizeof(threads[0]));
            started_flags = calloc((size_t)n_threads, sizeof(started_flags[0]));
            if (!threads || !started_flags) {
                free(threads);
                free(started_flags);
                if (tasks != stack_tasks) free(tasks);
                snprintf(err, errlen, "native drafter out of memory creating matvec threads");
                return -1;
            }
        }
        for (int tix = 0; tix < n_threads; tix++) {
            if (tix == 0) {
                native_quant_affine_u32_matvec_rows(&tasks[tix]);
            } else if (pthread_create(&threads[tix], NULL, native_matvec_thread_main,
                                      &tasks[tix]) == 0) {
                started_flags[tix] = 1;
            } else {
                native_quant_affine_u32_matvec_rows(&tasks[tix]);
            }
        }
        for (int tix = 1; tix < n_threads; tix++) {
            if (started_flags[tix]) pthread_join(threads[tix], NULL);
        }
        if (threads != stack_threads) {
            free(threads);
            free(started_flags);
        }
    }
    if (tasks != stack_tasks) {
        free(tasks);
    }
    return 0;
}

static int native_quant_affine_u32_matvec_many(const ds4_drafter_native_model *m,
                                               const char **weight_names,
                                               const float *x,
                                               int in_len,
                                               float **outs,
                                               const int *out_lens,
                                               int n_jobs,
                                               char *err,
                                               size_t errlen) {
    if (n_jobs <= 0) return 0;
    if (n_jobs == 1) {
        return native_quant_affine_u32_matvec(m, weight_names[0], x, in_len,
                                              outs[0], out_lens[0], err, errlen);
    }

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled() && n_jobs <= 8) {
        ds4_drafter_metal_affine_job jobs[8];
        memset(jobs, 0, sizeof(jobs));
        const int bits = m->cfg.quant_bits;
        const int group_size = m->cfg.quant_group_size;
        int cols_expected = -1;
        for (int j = 0; j < n_jobs; j++) {
            const char *weight_name = weight_names[j];
            const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
            if (!t || strcmp(t->dtype, "U32") != 0 || t->n_dims != 2) {
                snprintf(err, errlen, "native drafter expected U32 rank-2 weight tensor: %s",
                         weight_name);
                return -1;
            }

            char scales_name[1024];
            char biases_name[1024];
            if (native_quant_sibling_name(scales_name, sizeof(scales_name),
                                          weight_name, ".scales") != 0 ||
                native_quant_sibling_name(biases_name, sizeof(biases_name),
                                          weight_name, ".biases") != 0) {
                snprintf(err, errlen, "native drafter bad quantized weight name: %s", weight_name);
                return -1;
            }
            const ds4_drafter_tensor_meta *scales = native_find_tensor(m, scales_name);
            const ds4_drafter_tensor_meta *biases = native_find_tensor(m, biases_name);
            if (!scales || !biases ||
                strcmp(scales->dtype, "BF16") != 0 ||
                strcmp(biases->dtype, "BF16") != 0 ||
                scales->n_dims != 2 ||
                biases->n_dims != 2) {
                snprintf(err, errlen, "native drafter missing BF16 scales/biases for %s",
                         weight_name);
                return -1;
            }

            if (bits <= 0 || 32 % bits != 0 || group_size <= 0) {
                snprintf(err, errlen, "native drafter invalid affine quantization bits/group");
                return -1;
            }
            const int pack = 32 / bits;
            const int rows = (int)t->shape[0];
            const int packed_cols = (int)t->shape[1];
            const int cols = packed_cols * pack;
            const int groups = cols / group_size;
            if (cols_expected < 0) cols_expected = cols;
            if (cols_expected != cols ||
                rows != out_lens[j] ||
                cols != in_len ||
                cols % group_size != 0 ||
                scales->shape[0] != rows ||
                biases->shape[0] != rows ||
                scales->shape[1] != groups ||
                biases->shape[1] != groups) {
                snprintf(err, errlen,
                         "native drafter batched matvec shape mismatch for %s "
                         "(rows=%d cols=%d groups=%d in=%d out=%d)",
                         weight_name, rows, cols, groups, in_len, out_lens[j]);
                return -1;
            }

            const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
            const uint8_t *s_data = native_tensor_data_ptr(m, scales, err, errlen);
            const uint8_t *b_data = native_tensor_data_ptr(m, biases, err, errlen);
            if (!w_data || !s_data || !b_data) return -1;

            jobs[j] = (ds4_drafter_metal_affine_job){
                .w_data = w_data,
                .w_bytes = t->end - t->begin,
                .scales_data = s_data,
                .scales_bytes = scales->end - scales->begin,
                .biases_data = b_data,
                .biases_bytes = biases->end - biases->begin,
                .rows = rows,
                .packed_cols = packed_cols,
                .groups = groups,
                .out = outs[j],
            };
        }

        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_affine_u32_matvec_many(
            jobs, n_jobs, x, in_len, bits, group_size, metal_err, sizeof(metal_err));
        if (metal_rc == 0) return 0;
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter batched matvec failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: batched Metal matvec failed, falling back to CPU: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    for (int j = 0; j < n_jobs; j++) {
        if (native_quant_affine_u32_matvec(m, weight_names[j], x, in_len,
                                           outs[j], out_lens[j], err, errlen) != 0) {
            return -1;
        }
    }
    return 0;
}

#if defined(DS4_DRAFTER_HAS_METAL)
static int native_metal_prepare_affine_job(const ds4_drafter_native_model *m,
                                           const char *weight_name,
                                           int expected_in,
                                           int expected_out,
                                           ds4_drafter_metal_affine_job *job,
                                           char *err,
                                           size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
    if (!t || strcmp(t->dtype, "U32") != 0 || t->n_dims != 2) {
        snprintf(err, errlen, "native drafter expected U32 rank-2 weight tensor: %s",
                 weight_name);
        return -1;
    }
    char scales_name[1024];
    char biases_name[1024];
    if (native_quant_sibling_name(scales_name, sizeof(scales_name),
                                  weight_name, ".scales") != 0 ||
        native_quant_sibling_name(biases_name, sizeof(biases_name),
                                  weight_name, ".biases") != 0) {
        snprintf(err, errlen, "native drafter bad quantized weight name: %s", weight_name);
        return -1;
    }
    const ds4_drafter_tensor_meta *scales = native_find_tensor(m, scales_name);
    const ds4_drafter_tensor_meta *biases = native_find_tensor(m, biases_name);
    if (!scales || !biases ||
        strcmp(scales->dtype, "BF16") != 0 ||
        strcmp(biases->dtype, "BF16") != 0 ||
        scales->n_dims != 2 ||
        biases->n_dims != 2) {
        snprintf(err, errlen, "native drafter missing BF16 scales/biases for %s",
                 weight_name);
        return -1;
    }
    const int bits = m->cfg.quant_bits;
    const int group_size = m->cfg.quant_group_size;
    if (bits <= 0 || 32 % bits != 0 || group_size <= 0) {
        snprintf(err, errlen, "native drafter invalid affine quantization bits/group");
        return -1;
    }
    const int pack = 32 / bits;
    const int rows = (int)t->shape[0];
    const int packed_cols = (int)t->shape[1];
    const int cols = packed_cols * pack;
    const int groups = cols / group_size;
    if (rows != expected_out ||
        cols != expected_in ||
        cols % group_size != 0 ||
        scales->shape[0] != rows ||
        biases->shape[0] != rows ||
        scales->shape[1] != groups ||
        biases->shape[1] != groups) {
        snprintf(err, errlen,
                 "native drafter Metal affine shape mismatch for %s "
                 "(rows=%d cols=%d groups=%d expected_in=%d expected_out=%d)",
                 weight_name, rows, cols, groups, expected_in, expected_out);
        return -1;
    }
    const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
    const uint8_t *s_data = native_tensor_data_ptr(m, scales, err, errlen);
    const uint8_t *b_data = native_tensor_data_ptr(m, biases, err, errlen);
    if (!w_data || !s_data || !b_data) return -1;
    *job = (ds4_drafter_metal_affine_job){
        .w_data = w_data,
        .w_bytes = t->end - t->begin,
        .scales_data = s_data,
        .scales_bytes = scales->end - scales->begin,
        .biases_data = b_data,
        .biases_bytes = biases->end - biases->begin,
        .rows = rows,
        .packed_cols = packed_cols,
        .groups = groups,
        .out = NULL,
    };
    if (native_metal_enabled()) {
        char metal_err[256] = {0};
        if (ds4_drafter_metal_prepare_affine_u32(job, cols, bits, group_size,
                                                 metal_err,
                                                 sizeof(metal_err)) != 0) {
            snprintf(err, errlen, "%s",
                     metal_err[0] ? metal_err : "native Metal drafter affine prepare failed");
            return -1;
        }
    }
    return 0;
}
#endif

static int native_embedding_lookup(const ds4_drafter_native_model *m,
                                   int token_id,
                                   float *out,
                                   int out_len,
                                   char *err,
                                   size_t errlen) {
    int rc = native_dequant_affine_u32_row(m,
                                           "language_model.model.embed_tokens.weight",
                                           token_id,
                                           out,
                                           out_len,
                                           err,
                                           errlen);
    if (rc != 0) return rc;
    for (int i = 0; i < out_len; i++) {
        out[i] = native_bf16_to_f32(native_f32_to_bf16(out[i]));
    }
    return 0;
}

static void native_round_bf16_vector(float *x, int len) {
    for (int i = 0; i < len; i++) {
        x[i] = native_bf16_to_f32(native_f32_to_bf16(x[i]));
    }
}

static int native_rms_norm(const ds4_drafter_native_model *m,
                           const char *weight_name,
                           const float *x,
                           int len,
                           float eps,
                           float *out,
                           char *err,
                           size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
    if (!t || strcmp(t->dtype, "BF16") != 0 || t->n_dims != 1 ||
        t->shape[0] != len) {
        snprintf(err, errlen, "native drafter expected BF16 RMSNorm vector: %s",
                 weight_name);
        return -1;
    }
    const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
    if (!w_data) return -1;
    double ss = 0.0;
    for (int i = 0; i < len; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)len) + eps);
    for (int i = 0; i < len; i++) {
        float w = native_bf16_to_f32(native_read_le16(w_data + (size_t)i * 2u));
        out[i] = x[i] * inv * w;
    }
    return 0;
}

static int native_qwen_full_attention_project(const ds4_drafter_native_model *m,
                                              int layer,
                                              const float *x_norm,
                                              float *queries_normed,
                                              float *gate,
                                              float *keys_normed,
                                              float *values,
                                              char *err,
                                              size_t errlen) {
    char q_name[256];
    char k_name[256];
    char v_name[256];
    char q_norm_name[256];
    char k_norm_name[256];
    int rc = snprintf(q_name, sizeof(q_name),
                      "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
    rc |= snprintf(k_name, sizeof(k_name),
                   "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
    rc |= snprintf(v_name, sizeof(v_name),
                   "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
    rc |= snprintf(q_norm_name, sizeof(q_norm_name),
                   "language_model.model.layers.%d.self_attn.q_norm.weight", layer);
    rc |= snprintf(k_norm_name, sizeof(k_norm_name),
                   "language_model.model.layers.%d.self_attn.k_norm.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad attention layer name");
        return -1;
    }

    float q_proj[4096];
    float k_proj[512];
    const char *proj_names[3] = { q_name, k_name, v_name };
    float *proj_outs[3] = { q_proj, k_proj, values };
    const int proj_lens[3] = { 4096, 512, 512 };
    if (native_quant_affine_u32_matvec_many(m, proj_names, x_norm, 1024,
                                            proj_outs, proj_lens, 3,
                                            err, errlen) != 0) {
        return -1;
    }

    for (int h = 0; h < 8; h++) {
        const float *q_head = q_proj + h * 512;
        memcpy(gate + h * 256, q_head + 256, 256u * sizeof(gate[0]));
        if (native_rms_norm(m, q_norm_name, q_head, 256, 1.0e-6f,
                            queries_normed + h * 256, err, errlen) != 0) {
            return -1;
        }
    }
    for (int h = 0; h < 2; h++) {
        if (native_rms_norm(m, k_norm_name, k_proj + h * 256, 256, 1.0e-6f,
                            keys_normed + h * 256, err, errlen) != 0) {
            return -1;
        }
    }
    return 0;
}

static void native_qwen_rope_apply_head(float *head,
                                        int head_dim,
                                        int rotary_dim,
                                        int position) {
    const int half = rotary_dim / 2;
    const double base = 10000000.0;
    for (int i = 0; i < half; i++) {
        double freq = pow(base, (double)(2 * i) / (double)rotary_dim);
        double angle = (double)position / freq;
        float c = (float)cos(angle);
        float s = (float)sin(angle);
        float a = head[i];
        float b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
    (void)head_dim;
}

static void native_qwen_full_attention_apply_rope(float *queries,
                                                  float *keys,
                                                  int position) {
    for (int h = 0; h < 8; h++) {
        native_qwen_rope_apply_head(queries + h * 256, 256, 64, position);
    }
    for (int h = 0; h < 2; h++) {
        native_qwen_rope_apply_head(keys + h * 256, 256, 64, position);
    }
}

static float native_sigmoidf(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static float native_siluf(float x) {
    return x * native_sigmoidf(x);
}

static float native_softplusf(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void native_rms_norm_weightless(const float *x,
                                       int len,
                                       float eps,
                                       float scale,
                                       float *out) {
    double ss = 0.0;
    for (int i = 0; i < len; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)len) + eps);
    for (int i = 0; i < len; i++) out[i] = x[i] * inv * scale;
}

static int native_qwen_full_attention_output(const ds4_drafter_native_model *m,
                                             int layer,
                                             const float *queries_rope,
                                             const float *gate,
                                             const float *keys_rope_cache,
                                             const float *values_cache,
                                             int n_ctx,
                                             float *out,
                                             char *err,
                                             size_t errlen) {
    if (n_ctx <= 0) {
        snprintf(err, errlen, "native drafter attention requires non-empty cache");
        return -1;
    }
    char o_name[256];
    int rc = snprintf(o_name, sizeof(o_name),
                      "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
    if (rc <= 0 || (size_t)rc >= sizeof(o_name)) {
        snprintf(err, errlen, "native drafter bad attention o_proj name");
        return -1;
    }

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled() && native_metal_argmax_enabled()) {
        ds4_drafter_metal_affine_job out_job;
        if (native_metal_prepare_affine_job(m, o_name, 2048, 1024,
                                            &out_job, err, errlen) != 0) {
            return -1;
        }
        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_full_attention_output_u32(
            &out_job, queries_rope, gate, keys_rope_cache, values_cache,
            n_ctx, m->cfg.quant_bits, m->cfg.quant_group_size,
            out, metal_err, sizeof(metal_err));
        if (metal_rc == 0) return 0;
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused attention output failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: fused Metal attention output failed, falling back to CPU: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    float attn[2048];
    const float scale = 1.0f / 16.0f; /* head_dim 256 */
    for (int qh = 0; qh < 8; qh++) {
        int kvh = qh / 4;
        const float *q = queries_rope + qh * 256;
        float logits_stack[512];
        float *logits = logits_stack;
        if (n_ctx > (int)(sizeof(logits_stack) / sizeof(logits_stack[0]))) {
            logits = malloc((size_t)n_ctx * sizeof(logits[0]));
            if (!logits) {
                snprintf(err, errlen, "native drafter out of memory allocating attention logits");
                return -1;
            }
        }

#if defined(__APPLE__)
        const int use_accelerate = native_attention_accelerate_enabled() && n_ctx >= 64;
#else
        const int use_accelerate = 0;
#endif
        double max_logit = -1.0e300;
        if (use_accelerate) {
#if defined(__APPLE__)
            const float *k0 = keys_rope_cache + (size_t)kvh * 256u;
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        n_ctx, 256,
                        scale, k0, 512,
                        q, 1,
                        0.0f, logits, 1);
            for (int t = 0; t < n_ctx; t++) {
                if ((double)logits[t] > max_logit) max_logit = (double)logits[t];
            }
#endif
        } else {
            for (int t = 0; t < n_ctx; t++) {
                const float *k = keys_rope_cache + ((size_t)t * 2u + (size_t)kvh) * 256u;
                float dot = 0.0f;
                for (int i = 0; i < 256; i++) dot += q[i] * k[i];
                float v = dot * scale;
                logits[t] = v;
                if ((double)v > max_logit) max_logit = (double)v;
            }
        }
        double denom = 0.0;
        for (int t = 0; t < n_ctx; t++) {
            double p = exp((double)logits[t] - max_logit);
            logits[t] = (float)p;
            denom += p;
        }
        if (denom <= 0.0 || !isfinite(denom)) {
            if (logits != logits_stack) free(logits);
            snprintf(err, errlen, "native drafter attention softmax produced invalid denominator");
            return -1;
        }
        const float inv_denom = (float)(1.0 / denom);
        for (int t = 0; t < n_ctx; t++) logits[t] *= inv_denom;
        if (use_accelerate) {
#if defined(__APPLE__)
            float ctx[256];
            const float *v0 = values_cache + (size_t)kvh * 256u;
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        n_ctx, 256,
                        1.0f, v0, 512,
                        logits, 1,
                        0.0f, ctx, 1);
            for (int i = 0; i < 256; i++) {
                attn[qh * 256 + i] = ctx[i] * native_sigmoidf(gate[qh * 256 + i]);
            }
#endif
        } else {
            for (int i = 0; i < 256; i++) {
                float acc = 0.0f;
                for (int t = 0; t < n_ctx; t++) {
                    const float *v = values_cache + ((size_t)t * 2u + (size_t)kvh) * 256u;
                    acc += logits[t] * v[i];
                }
                attn[qh * 256 + i] = acc * native_sigmoidf(gate[qh * 256 + i]);
            }
        }
        if (logits != logits_stack) free(logits);
    }
    return native_quant_affine_u32_matvec(m, o_name, attn, 2048,
                                          out, 1024, err, errlen);
}

static int native_qwen_mlp(const ds4_drafter_native_model *m,
                           int layer,
                           const float *x_norm,
                           float *out,
                           char *err,
                           size_t errlen) {
    char gate_name[256];
    char up_name[256];
    char down_name[256];
    int rc = snprintf(gate_name, sizeof(gate_name),
                      "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
    rc |= snprintf(up_name, sizeof(up_name),
                   "language_model.model.layers.%d.mlp.up_proj.weight", layer);
    rc |= snprintf(down_name, sizeof(down_name),
                   "language_model.model.layers.%d.mlp.down_proj.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad MLP layer name");
        return -1;
    }

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled()) {
        ds4_drafter_metal_affine_job gate_job;
        ds4_drafter_metal_affine_job up_job;
        ds4_drafter_metal_affine_job down_job;
        if (native_metal_prepare_affine_job(m, gate_name, 1024, 3584,
                                            &gate_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, up_name, 1024, 3584,
                                            &up_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, down_name, 3584, 1024,
                                            &down_job, err, errlen) != 0) {
            return -1;
        }
        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_mlp_u32(
            &gate_job, &up_job, &down_job, x_norm,
            1024, 3584, 1024,
            m->cfg.quant_bits, m->cfg.quant_group_size,
            out, metal_err, sizeof(metal_err));
        if (metal_rc == 0) return 0;
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused MLP failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: fused Metal MLP failed, falling back to CPU: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    float gate[3584];
    float up[3584];
    float hidden[3584];
    const char *proj_names[2] = { gate_name, up_name };
    float *proj_outs[2] = { gate, up };
    const int proj_lens[2] = { 3584, 3584 };
    if (native_quant_affine_u32_matvec_many(m, proj_names, x_norm, 1024,
                                            proj_outs, proj_lens, 2,
                                            err, errlen) != 0) {
        return -1;
    }
    for (int i = 0; i < 3584; i++) {
        hidden[i] = native_siluf(gate[i]) * up[i];
    }
    return native_quant_affine_u32_matvec(m, down_name, hidden, 3584,
                                          out, 1024, err, errlen);
}

static void native_full_attention_state_free(native_full_attention_state *s) {
    if (!s) return;
    free(s->keys);
    free(s->values);
    memset(s, 0, sizeof(*s));
}

static int native_full_attention_state_reserve(native_full_attention_state *s,
                                               int need,
                                               char *err,
                                               size_t errlen) {
    if (need <= s->cap) return 0;
    int cap = s->cap ? s->cap : 16;
    while (cap < need) {
        if (cap > INT_MAX / 2) {
            snprintf(err, errlen, "native drafter attention cache too large");
            return -1;
        }
        cap *= 2;
    }
    float *keys = realloc(s->keys, (size_t)cap * 2u * 256u * sizeof(keys[0]));
    if (!keys) {
        snprintf(err, errlen, "native drafter out of memory growing attention keys");
        return -1;
    }
    s->keys = keys;
    float *values = realloc(s->values, (size_t)cap * 2u * 256u * sizeof(values[0]));
    if (!values) {
        snprintf(err, errlen, "native drafter out of memory growing attention values");
        return -1;
    }
    s->values = values;
    s->cap = cap;
    s->metal_cache_valid = 0;
    s->metal_cache_cap = 0;
    return 0;
}

#if defined(DS4_DRAFTER_HAS_METAL)
static int native_full_attention_state_reserve_metal_only(native_full_attention_state *s,
                                                          int need,
                                                          char *err,
                                                          size_t errlen) {
    if (need <= s->cap) return 0;
    int cap = s->cap ? s->cap : 16;
    while (cap < need) {
        if (cap > INT_MAX / 2) {
            snprintf(err, errlen, "native drafter attention cache too large");
            return -1;
        }
        cap *= 2;
    }
    free(s->keys);
    free(s->values);
    s->keys = NULL;
    s->values = NULL;
    s->cap = cap;
    s->metal_cache_valid = 0;
    s->metal_cache_cap = 0;
    return 0;
}
#endif

static int native_qwen_full_attention_step(const ds4_drafter_native_model *m,
                                           int layer,
                                           const float *x_norm,
                                           int position,
                                           native_full_attention_state *state,
                                           float *queries_capture,
                                           float *out,
                                           char *err,
                                           size_t errlen) {
    if (!state) {
        snprintf(err, errlen, "native drafter missing full-attention state");
        return -1;
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled()) {
        if (native_full_attention_state_reserve(state, state->len + 1, err, errlen) != 0) {
            return -1;
        }
        if (state->len == 0 ||
            (state->metal_cache_valid && state->metal_cache_cap == state->cap)) {
            native_metal_full_layer_cache *full_cache =
                &((ds4_drafter_native_model *)m)->metal_full_cache[layer];
            if (full_cache->initialized) {
                char metal_err[256] = {0};
                int metal_rc = ds4_drafter_metal_full_attention_layer_u32(
                    &full_cache->q_job, &full_cache->k_job,
                    &full_cache->v_job, &full_cache->o_job,
                    full_cache->q_norm_data, full_cache->q_norm_bytes,
                    full_cache->k_norm_data, full_cache->k_norm_bytes,
                    x_norm, position, layer,
                    state->len, state->len + 1, state->cap,
                    m->cfg.quant_bits, m->cfg.quant_group_size,
                    queries_capture,
                    state->keys + (size_t)state->len * 2u * 256u,
                    state->values + (size_t)state->len * 2u * 256u,
                    out, metal_err, sizeof(metal_err));
                if (metal_rc == 0) {
                    state->len++;
                    state->metal_cache_valid = 1;
                    state->metal_cache_cap = state->cap;
                    return 0;
                }
                state->metal_cache_valid = 0;
                state->metal_cache_cap = 0;
                if (native_metal_strict()) {
                    snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter cached fused full attention failed");
                    return -1;
                }
                static int warned_cached = 0;
                if (!warned_cached) {
                    fprintf(stderr,
                            "native drafter: cached fused Metal full attention failed, falling back to setup path: %s\n",
                            metal_err[0] ? metal_err : "unknown error");
                    warned_cached = 1;
                }
            }
            char q_name[256];
            char k_name[256];
            char v_name[256];
            char o_name[256];
            char q_norm_name[256];
            char k_norm_name[256];
            int rc = snprintf(q_name, sizeof(q_name),
                              "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
            rc |= snprintf(k_name, sizeof(k_name),
                           "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
            rc |= snprintf(v_name, sizeof(v_name),
                           "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
            rc |= snprintf(o_name, sizeof(o_name),
                           "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
            rc |= snprintf(q_norm_name, sizeof(q_norm_name),
                           "language_model.model.layers.%d.self_attn.q_norm.weight", layer);
            rc |= snprintf(k_norm_name, sizeof(k_norm_name),
                           "language_model.model.layers.%d.self_attn.k_norm.weight", layer);
            if (rc <= 0) {
                snprintf(err, errlen, "native drafter bad fused attention layer name");
                return -1;
            }
            const ds4_drafter_tensor_meta *q_norm = native_find_tensor(m, q_norm_name);
            const ds4_drafter_tensor_meta *k_norm = native_find_tensor(m, k_norm_name);
            if (!q_norm || strcmp(q_norm->dtype, "BF16") != 0 ||
                q_norm->n_dims != 1 || q_norm->shape[0] != 256 ||
                !k_norm || strcmp(k_norm->dtype, "BF16") != 0 ||
                k_norm->n_dims != 1 || k_norm->shape[0] != 256) {
                snprintf(err, errlen, "native drafter invalid fused attention norm metadata");
                return -1;
            }
            const uint8_t *q_norm_data = native_tensor_data_ptr(m, q_norm, err, errlen);
            const uint8_t *k_norm_data = native_tensor_data_ptr(m, k_norm, err, errlen);
            if (!q_norm_data || !k_norm_data) return -1;
            ds4_drafter_metal_affine_job q_job;
            ds4_drafter_metal_affine_job k_job;
            ds4_drafter_metal_affine_job v_job;
            ds4_drafter_metal_affine_job o_job;
            if (native_metal_prepare_affine_job(m, q_name, 1024, 4096,
                                                &q_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, k_name, 1024, 512,
                                                &k_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, v_name, 1024, 512,
                                                &v_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, o_name, 2048, 1024,
                                                &o_job, err, errlen) != 0) {
                return -1;
            }
            if (!full_cache->initialized) {
                full_cache->q_norm_data = q_norm_data;
                full_cache->q_norm_bytes = q_norm->end - q_norm->begin;
                full_cache->k_norm_data = k_norm_data;
                full_cache->k_norm_bytes = k_norm->end - k_norm->begin;
                full_cache->q_job = q_job;
                full_cache->k_job = k_job;
                full_cache->v_job = v_job;
                full_cache->o_job = o_job;
                full_cache->initialized = 1;
            }
            char metal_err[256] = {0};
            int metal_rc = ds4_drafter_metal_full_attention_layer_u32(
                &q_job, &k_job, &v_job, &o_job,
                q_norm_data, q_norm->end - q_norm->begin,
                k_norm_data, k_norm->end - k_norm->begin,
                x_norm, position, layer,
                state->len, state->len + 1, state->cap,
                m->cfg.quant_bits, m->cfg.quant_group_size,
                queries_capture,
                state->keys + (size_t)state->len * 2u * 256u,
                state->values + (size_t)state->len * 2u * 256u,
                out, metal_err, sizeof(metal_err));
            if (metal_rc == 0) {
                state->len++;
                state->metal_cache_valid = 1;
                state->metal_cache_cap = state->cap;
                return 0;
            }
            state->metal_cache_valid = 0;
            state->metal_cache_cap = 0;
            if (native_metal_strict()) {
                snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused full attention failed");
                return -1;
            }
            static int warned = 0;
            if (!warned) {
                fprintf(stderr,
                        "native drafter: fused Metal full attention failed, falling back to CPU project path: %s\n",
                        metal_err[0] ? metal_err : "unknown error");
                warned = 1;
            }
        }
    }
#endif
    float queries[2048];
    float gate[2048];
    float keys[512];
    float values[512];
    if (native_qwen_full_attention_project(m, layer, x_norm,
                                           queries, gate, keys, values,
                                           err, errlen) != 0) {
        return -1;
    }
    native_qwen_full_attention_apply_rope(queries, keys, position);
    if (queries_capture) {
        memcpy(queries_capture, queries, 2048u * sizeof(queries[0]));
    }
    if (native_full_attention_state_reserve(state, state->len + 1, err, errlen) != 0) {
        return -1;
    }
    memcpy(state->keys + (size_t)state->len * 2u * 256u,
           keys,
           512u * sizeof(keys[0]));
    memcpy(state->values + (size_t)state->len * 2u * 256u,
           values,
           512u * sizeof(values[0]));
    state->len++;

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled() &&
        (state->len == 1 ||
         (state->metal_cache_valid && state->metal_cache_cap == state->cap))) {
        char o_name[256];
        int rc = snprintf(o_name, sizeof(o_name),
                          "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
        if (rc <= 0 || (size_t)rc >= sizeof(o_name)) {
            snprintf(err, errlen, "native drafter bad attention o_proj name");
            return -1;
        }
        ds4_drafter_metal_affine_job out_job;
        if (native_metal_prepare_affine_job(m, o_name, 2048, 1024,
                                            &out_job, err, errlen) != 0) {
            return -1;
        }
        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_full_attention_step_u32(
            &out_job, queries, gate, keys, values,
            layer, state->len - 1, state->len, state->cap,
            m->cfg.quant_bits, m->cfg.quant_group_size,
            out, metal_err, sizeof(metal_err));
        if (metal_rc == 0) {
            state->metal_cache_valid = 1;
            state->metal_cache_cap = state->cap;
            return 0;
        }
        state->metal_cache_valid = 0;
        state->metal_cache_cap = 0;
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter resident attention failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: resident Metal attention failed, falling back to full-cache path: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    return native_qwen_full_attention_output(m, layer, queries, gate,
                                             state->keys, state->values,
                                             state->len, out, err, errlen);
}

static int native_qwen_linear_attention_step(const ds4_drafter_native_model *m,
                                             int layer,
                                             const float *x_norm,
                                             float *conv_state,
                                             float *delta_state,
                                             float *out,
                                             char *err,
                                             size_t errlen) {
    char qkv_name[256];
    char z_name[256];
    char b_name[256];
    char a_name[256];
    char conv_name[256];
    char norm_name[256];
    char a_log_name[256];
    char dt_bias_name[256];
    char out_name[256];
    int rc = snprintf(qkv_name, sizeof(qkv_name),
                      "language_model.model.layers.%d.linear_attn.in_proj_qkv.weight", layer);
    rc |= snprintf(z_name, sizeof(z_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_z.weight", layer);
    rc |= snprintf(b_name, sizeof(b_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_b.weight", layer);
    rc |= snprintf(a_name, sizeof(a_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_a.weight", layer);
    rc |= snprintf(conv_name, sizeof(conv_name),
                   "language_model.model.layers.%d.linear_attn.conv1d.weight", layer);
    rc |= snprintf(norm_name, sizeof(norm_name),
                   "language_model.model.layers.%d.linear_attn.norm.weight", layer);
    rc |= snprintf(a_log_name, sizeof(a_log_name),
                   "language_model.model.layers.%d.linear_attn.A_log", layer);
    rc |= snprintf(dt_bias_name, sizeof(dt_bias_name),
                   "language_model.model.layers.%d.linear_attn.dt_bias", layer);
    rc |= snprintf(out_name, sizeof(out_name),
                   "language_model.model.layers.%d.linear_attn.out_proj.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad linear-attention layer name");
        return -1;
    }

    const ds4_drafter_tensor_meta *conv = native_find_tensor(m, conv_name);
    if (!conv || strcmp(conv->dtype, "BF16") != 0 || conv->n_dims != 3 ||
        conv->shape[0] != 6144 || conv->shape[1] != 4 || conv->shape[2] != 1) {
        snprintf(err, errlen, "native drafter expected BF16 depthwise conv tensor: %s",
                 conv_name);
        return -1;
    }
    const uint8_t *conv_data = native_tensor_data_ptr(m, conv, err, errlen);
    if (!conv_data) return -1;

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled()) {
        const ds4_drafter_tensor_meta *norm = native_find_tensor(m, norm_name);
        const ds4_drafter_tensor_meta *a_log_t = native_find_tensor(m, a_log_name);
        const ds4_drafter_tensor_meta *dt_bias_t = native_find_tensor(m, dt_bias_name);
        if (!norm || strcmp(norm->dtype, "BF16") != 0 || norm->n_dims != 1 ||
            norm->shape[0] != 128 ||
            !a_log_t || strcmp(a_log_t->dtype, "F32") != 0 || a_log_t->n_dims != 1 ||
            a_log_t->shape[0] != 16 ||
            !dt_bias_t || strcmp(dt_bias_t->dtype, "BF16") != 0 || dt_bias_t->n_dims != 1 ||
            dt_bias_t->shape[0] != 16) {
            snprintf(err, errlen, "native drafter invalid linear-attention Metal tensor metadata");
            return -1;
        }
        const uint8_t *norm_data = native_tensor_data_ptr(m, norm, err, errlen);
        const uint8_t *a_log_data = native_tensor_data_ptr(m, a_log_t, err, errlen);
        const uint8_t *dt_bias_data = native_tensor_data_ptr(m, dt_bias_t, err, errlen);
        if (!norm_data || !a_log_data || !dt_bias_data) return -1;
        ds4_drafter_metal_affine_job qkv_job;
        ds4_drafter_metal_affine_job z_job;
        ds4_drafter_metal_affine_job b_job;
        ds4_drafter_metal_affine_job a_job;
        ds4_drafter_metal_affine_job out_job;
        if (native_metal_prepare_affine_job(m, qkv_name, 1024, 6144,
                                            &qkv_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, z_name, 1024, 2048,
                                            &z_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, b_name, 1024, 16,
                                            &b_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, a_name, 1024, 16,
                                            &a_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, out_name, 2048, 1024,
                                            &out_job, err, errlen) != 0) {
            return -1;
        }
        char metal_err[256] = {0};
        int metal_rc = ds4_drafter_metal_linear_attention_u32(
            &qkv_job, &z_job, &b_job, &a_job, &out_job,
            conv_data, conv->end - conv->begin,
            norm_data, norm->end - norm->begin,
            a_log_data, a_log_t->end - a_log_t->begin,
            dt_bias_data, dt_bias_t->end - dt_bias_t->begin,
            x_norm, layer,
            m->cfg.quant_bits, m->cfg.quant_group_size,
            out, metal_err, sizeof(metal_err));
        if (metal_rc == 0) return 0;
        snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused linear attention failed");
        return -1;
    }
#endif

    float qkv[6144];
    float z[2048];
    float b[16];
    float a[16];
    const char *proj_names[4] = { qkv_name, z_name, b_name, a_name };
    float *proj_outs[4] = { qkv, z, b, a };
    const int proj_lens[4] = { 6144, 2048, 16, 16 };
    if (native_quant_affine_u32_matvec_many(m, proj_names, x_norm, 1024,
                                            proj_outs, proj_lens, 4,
                                            err, errlen) != 0) {
        return -1;
    }

    float conv_out[6144];
    for (int c = 0; c < 6144; c++) {
        double acc = 0.0;
        for (int k = 0; k < 3; k++) {
            float w = native_bf16_to_f32(native_read_le16(
                conv_data + ((size_t)c * 4u + (size_t)k) * 2u));
            acc += (double)conv_state[(size_t)k * 6144u + (size_t)c] * (double)w;
        }
        float w = native_bf16_to_f32(native_read_le16(
            conv_data + ((size_t)c * 4u + 3u) * 2u));
        acc += (double)qkv[c] * (double)w;
        conv_out[c] = native_siluf((float)acc);
    }
    memmove(conv_state, conv_state + 6144, (size_t)2u * 6144u * sizeof(conv_state[0]));
    memcpy(conv_state + (size_t)2u * 6144u, qkv, 6144u * sizeof(qkv[0]));

    float q_norm[2048];
    float k_norm[2048];
    const float inv_scale = 1.0f / sqrtf(128.0f);
    for (int h = 0; h < 16; h++) {
        native_rms_norm_weightless(conv_out + h * 128, 128, 1.0e-6f,
                                   inv_scale * inv_scale,
                                   q_norm + h * 128);
        native_rms_norm_weightless(conv_out + 2048 + h * 128, 128, 1.0e-6f,
                                   inv_scale,
                                   k_norm + h * 128);
    }

    float a_log[16];
    float dt_bias[16];
    float norm_w[128];
    if (native_load_f32_vector(m, a_log_name, a_log, 16, err, errlen) != 0 ||
        native_load_bf16_vector(m, dt_bias_name, dt_bias, 16, err, errlen) != 0 ||
        native_load_bf16_vector(m, norm_name, norm_w, 128, err, errlen) != 0) {
        return -1;
    }

    float gated[2048];
    const float *v = conv_out + 4096;
    for (int h = 0; h < 16; h++) {
        float beta = native_sigmoidf(b[h]);
        float g = expf(-expf(a_log[h]) * native_softplusf(a[h] + dt_bias[h]));
        float y[128];
        for (int dv = 0; dv < 128; dv++) {
            float *state = delta_state + ((size_t)h * 128u + (size_t)dv) * 128u;
            double kv_mem = 0.0;
            for (int dk = 0; dk < 128; dk++) {
                state[dk] *= g;
                kv_mem += (double)state[dk] * (double)k_norm[h * 128 + dk];
            }
            float delta = (v[h * 128 + dv] - (float)kv_mem) * beta;
            double out_acc = 0.0;
            for (int dk = 0; dk < 128; dk++) {
                state[dk] += k_norm[h * 128 + dk] * delta;
                out_acc += (double)state[dk] * (double)q_norm[h * 128 + dk];
            }
            y[dv] = (float)out_acc;
        }
        double ss = 0.0;
        for (int dv = 0; dv < 128; dv++) ss += (double)y[dv] * (double)y[dv];
        float inv = 1.0f / sqrtf((float)(ss / 128.0) + 1.0e-6f);
        for (int dv = 0; dv < 128; dv++) {
            int idx = h * 128 + dv;
            float normed = y[dv] * inv * norm_w[dv];
            gated[idx] = native_siluf(z[idx]) * normed;
        }
    }

    return native_quant_affine_u32_matvec(m, out_name, gated, 2048,
                                          out, 1024, err, errlen);
}

static int native_qwen_layer_is_linear(int layer) {
    return ((layer + 1) % 4) != 0;
}

#if defined(DS4_DRAFTER_HAS_METAL)
static int native_qwen_full_decoder_layer_metal_step(
        const ds4_drafter_native_model *m,
        int layer,
        const char *input_norm_name,
        const char *post_norm_name,
        const float *x,
        int position,
        native_full_attention_state *state,
        float *queries_capture,
        float *out,
        char *err,
        size_t errlen) {
    if (!state) {
        snprintf(err, errlen, "native drafter missing full-attention state");
        return -1;
    }
    if (native_full_attention_state_reserve(state, state->len + 1, err, errlen) != 0) {
        return -1;
    }
    if (!(state->len == 0 ||
          (state->metal_cache_valid && state->metal_cache_cap == state->cap))) {
        return 1;
    }
    native_metal_full_layer_cache *full_cache =
        &((ds4_drafter_native_model *)m)->metal_full_cache[layer];
    if (!full_cache->decoder_initialized) {
        char q_name[256];
        char k_name[256];
        char v_name[256];
        char o_name[256];
        char q_norm_name[256];
        char k_norm_name[256];
        char mlp_gate_name[256];
        char mlp_up_name[256];
        char mlp_down_name[256];
        int rc = snprintf(q_name, sizeof(q_name),
                          "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
        rc |= snprintf(k_name, sizeof(k_name),
                       "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
        rc |= snprintf(v_name, sizeof(v_name),
                       "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
        rc |= snprintf(o_name, sizeof(o_name),
                       "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
        rc |= snprintf(q_norm_name, sizeof(q_norm_name),
                       "language_model.model.layers.%d.self_attn.q_norm.weight", layer);
        rc |= snprintf(k_norm_name, sizeof(k_norm_name),
                       "language_model.model.layers.%d.self_attn.k_norm.weight", layer);
        rc |= snprintf(mlp_gate_name, sizeof(mlp_gate_name),
                       "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
        rc |= snprintf(mlp_up_name, sizeof(mlp_up_name),
                       "language_model.model.layers.%d.mlp.up_proj.weight", layer);
        rc |= snprintf(mlp_down_name, sizeof(mlp_down_name),
                       "language_model.model.layers.%d.mlp.down_proj.weight", layer);
        if (rc <= 0) {
            snprintf(err, errlen, "native drafter bad fused full decoder layer name");
            return -1;
        }
        const ds4_drafter_tensor_meta *input_norm_t = native_find_tensor(m, input_norm_name);
        const ds4_drafter_tensor_meta *post_norm_t = native_find_tensor(m, post_norm_name);
        const ds4_drafter_tensor_meta *q_norm = native_find_tensor(m, q_norm_name);
        const ds4_drafter_tensor_meta *k_norm = native_find_tensor(m, k_norm_name);
        if (!input_norm_t || strcmp(input_norm_t->dtype, "BF16") != 0 ||
            input_norm_t->n_dims != 1 || input_norm_t->shape[0] != 1024 ||
            !post_norm_t || strcmp(post_norm_t->dtype, "BF16") != 0 ||
            post_norm_t->n_dims != 1 || post_norm_t->shape[0] != 1024 ||
            !q_norm || strcmp(q_norm->dtype, "BF16") != 0 ||
            q_norm->n_dims != 1 || q_norm->shape[0] != 256 ||
            !k_norm || strcmp(k_norm->dtype, "BF16") != 0 ||
            k_norm->n_dims != 1 || k_norm->shape[0] != 256) {
            snprintf(err, errlen, "native drafter invalid fused full decoder layer norm metadata");
            return -1;
        }
        const uint8_t *input_norm_data = native_tensor_data_ptr(m, input_norm_t, err, errlen);
        const uint8_t *post_norm_data = native_tensor_data_ptr(m, post_norm_t, err, errlen);
        const uint8_t *q_norm_data = native_tensor_data_ptr(m, q_norm, err, errlen);
        const uint8_t *k_norm_data = native_tensor_data_ptr(m, k_norm, err, errlen);
        if (!input_norm_data || !post_norm_data || !q_norm_data || !k_norm_data) {
            return -1;
        }
        ds4_drafter_metal_affine_job q_job;
        ds4_drafter_metal_affine_job k_job;
        ds4_drafter_metal_affine_job v_job;
        ds4_drafter_metal_affine_job o_job;
        ds4_drafter_metal_affine_job mlp_gate_job;
        ds4_drafter_metal_affine_job mlp_up_job;
        ds4_drafter_metal_affine_job mlp_down_job;
        if (native_metal_prepare_affine_job(m, q_name, 1024, 4096,
                                            &q_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, k_name, 1024, 512,
                                            &k_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, v_name, 1024, 512,
                                            &v_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, o_name, 2048, 1024,
                                            &o_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, mlp_gate_name, 1024, 3584,
                                            &mlp_gate_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, mlp_up_name, 1024, 3584,
                                            &mlp_up_job, err, errlen) != 0 ||
            native_metal_prepare_affine_job(m, mlp_down_name, 3584, 1024,
                                            &mlp_down_job, err, errlen) != 0) {
            return -1;
        }
        full_cache->input_norm_data = input_norm_data;
        full_cache->input_norm_bytes = input_norm_t->end - input_norm_t->begin;
        full_cache->post_norm_data = post_norm_data;
        full_cache->post_norm_bytes = post_norm_t->end - post_norm_t->begin;
        full_cache->q_norm_data = q_norm_data;
        full_cache->q_norm_bytes = q_norm->end - q_norm->begin;
        full_cache->k_norm_data = k_norm_data;
        full_cache->k_norm_bytes = k_norm->end - k_norm->begin;
        full_cache->q_job = q_job;
        full_cache->k_job = k_job;
        full_cache->v_job = v_job;
        full_cache->o_job = o_job;
        full_cache->mlp_gate_job = mlp_gate_job;
        full_cache->mlp_up_job = mlp_up_job;
        full_cache->mlp_down_job = mlp_down_job;
        full_cache->initialized = 1;
        full_cache->decoder_initialized = 1;
    }
    char metal_err[256] = {0};
    int metal_rc = ds4_drafter_metal_full_decoder_layer_u32(
        full_cache->input_norm_data, full_cache->input_norm_bytes,
        full_cache->post_norm_data, full_cache->post_norm_bytes,
        &full_cache->q_job, &full_cache->k_job,
        &full_cache->v_job, &full_cache->o_job,
        full_cache->q_norm_data, full_cache->q_norm_bytes,
        full_cache->k_norm_data, full_cache->k_norm_bytes,
        &full_cache->mlp_gate_job, &full_cache->mlp_up_job,
        &full_cache->mlp_down_job,
        x, position, layer,
        state->len, state->len + 1, state->cap,
        m->cfg.quant_bits, m->cfg.quant_group_size,
        queries_capture,
        state->keys + (size_t)state->len * 2u * 256u,
        state->values + (size_t)state->len * 2u * 256u,
        out, metal_err, sizeof(metal_err));
    if (metal_rc == 0) {
        state->len++;
        state->metal_cache_valid = 1;
        state->metal_cache_cap = state->cap;
        return 0;
    }
    state->metal_cache_valid = 0;
    state->metal_cache_cap = 0;
    if (native_metal_strict()) {
        snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused full decoder layer failed");
        return -1;
    }
    static int warned = 0;
    if (!warned) {
        fprintf(stderr,
                "native drafter: fused Metal full decoder layer failed, falling back to split path: %s\n",
                metal_err[0] ? metal_err : "unknown error");
        warned = 1;
    }
    return 1;
}
#endif

static int native_qwen_decoder_layer_step(const ds4_drafter_native_model *m,
                                          int layer,
                                          const float *x,
                                          int position,
                                          float *linear_conv_state,
                                          float *linear_delta_state,
                                          native_full_attention_state *full_state,
                                          int bf16_path,
                                          float *full_query_capture,
                                          float *out,
                                          char *err,
                                          size_t errlen) {
    char input_norm_name[256];
    char post_norm_name[256];
    int rc = snprintf(input_norm_name, sizeof(input_norm_name),
                      "language_model.model.layers.%d.input_layernorm.weight", layer);
    rc |= snprintf(post_norm_name, sizeof(post_norm_name),
                   "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad decoder layer norm name");
        return -1;
    }

    float x_norm[1024];
    float residual[1024];
    float h[1024];
    float post_norm[1024];
    float mlp_out[1024];
    int x_norm_ready = 0;
    int try_fused_linear = 0;
    int try_fused_full = 0;
#if defined(DS4_DRAFTER_HAS_METAL)
    try_fused_linear = native_qwen_layer_is_linear(layer) &&
                       native_metal_enabled() &&
                       native_metal_fused_linear_enabled() &&
                       bf16_path;
    try_fused_full = !native_qwen_layer_is_linear(layer) &&
                     native_metal_enabled() &&
                     native_metal_fused_full_enabled() &&
                     bf16_path;
#endif
    if (!try_fused_linear && !try_fused_full) {
        if (native_rms_norm(m, input_norm_name, x, 1024, 1.0e-6f,
                            x_norm, err, errlen) != 0) {
            return -1;
        }
        if (bf16_path) native_round_bf16_vector(x_norm, 1024);
        x_norm_ready = 1;
    }
    if (native_qwen_layer_is_linear(layer)) {
#if defined(DS4_DRAFTER_HAS_METAL)
        if (try_fused_linear) {
            native_metal_linear_layer_cache *linear_cache =
                &((ds4_drafter_native_model *)m)->metal_linear_cache[layer];
            if (linear_cache->initialized) {
                char metal_err[256] = {0};
                int metal_rc = ds4_drafter_metal_linear_decoder_layer_u32(
                    linear_cache->input_norm_data, linear_cache->input_norm_bytes,
                    linear_cache->post_norm_data, linear_cache->post_norm_bytes,
                    &linear_cache->qkv_job, &linear_cache->z_job,
                    &linear_cache->b_job, &linear_cache->a_job,
                    &linear_cache->linear_out_job,
                    linear_cache->conv_data, linear_cache->conv_bytes,
                    linear_cache->linear_norm_data, linear_cache->linear_norm_bytes,
                    linear_cache->a_log_data, linear_cache->a_log_bytes,
                    linear_cache->dt_bias_data, linear_cache->dt_bias_bytes,
                    &linear_cache->mlp_gate_job, &linear_cache->mlp_up_job,
                    &linear_cache->mlp_down_job,
                    x, layer,
                    m->cfg.quant_bits, m->cfg.quant_group_size,
                    out, metal_err, sizeof(metal_err));
                if (metal_rc == 0) return 0;
                if (native_metal_strict()) {
                    snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused linear decoder layer failed");
                    return -1;
                }
                static int warned_cached = 0;
                if (!warned_cached) {
                    fprintf(stderr,
                            "native drafter: cached fused Metal linear decoder layer failed, falling back to split path: %s\n",
                            metal_err[0] ? metal_err : "unknown error");
                    warned_cached = 1;
                }
            }
            char qkv_name[256];
            char z_name[256];
            char b_name[256];
            char a_name[256];
            char conv_name[256];
            char linear_norm_name[256];
            char a_log_name[256];
            char dt_bias_name[256];
            char linear_out_name[256];
            char mlp_gate_name[256];
            char mlp_up_name[256];
            char mlp_down_name[256];
            int rc2 = snprintf(qkv_name, sizeof(qkv_name),
                               "language_model.model.layers.%d.linear_attn.in_proj_qkv.weight", layer);
            rc2 |= snprintf(z_name, sizeof(z_name),
                            "language_model.model.layers.%d.linear_attn.in_proj_z.weight", layer);
            rc2 |= snprintf(b_name, sizeof(b_name),
                            "language_model.model.layers.%d.linear_attn.in_proj_b.weight", layer);
            rc2 |= snprintf(a_name, sizeof(a_name),
                            "language_model.model.layers.%d.linear_attn.in_proj_a.weight", layer);
            rc2 |= snprintf(conv_name, sizeof(conv_name),
                            "language_model.model.layers.%d.linear_attn.conv1d.weight", layer);
            rc2 |= snprintf(linear_norm_name, sizeof(linear_norm_name),
                            "language_model.model.layers.%d.linear_attn.norm.weight", layer);
            rc2 |= snprintf(a_log_name, sizeof(a_log_name),
                            "language_model.model.layers.%d.linear_attn.A_log", layer);
            rc2 |= snprintf(dt_bias_name, sizeof(dt_bias_name),
                            "language_model.model.layers.%d.linear_attn.dt_bias", layer);
            rc2 |= snprintf(linear_out_name, sizeof(linear_out_name),
                            "language_model.model.layers.%d.linear_attn.out_proj.weight", layer);
            rc2 |= snprintf(mlp_gate_name, sizeof(mlp_gate_name),
                            "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
            rc2 |= snprintf(mlp_up_name, sizeof(mlp_up_name),
                            "language_model.model.layers.%d.mlp.up_proj.weight", layer);
            rc2 |= snprintf(mlp_down_name, sizeof(mlp_down_name),
                            "language_model.model.layers.%d.mlp.down_proj.weight", layer);
            if (rc2 <= 0) {
                snprintf(err, errlen, "native drafter bad fused linear layer name");
                return -1;
            }

            const ds4_drafter_tensor_meta *input_norm_t = native_find_tensor(m, input_norm_name);
            const ds4_drafter_tensor_meta *post_norm_t = native_find_tensor(m, post_norm_name);
            const ds4_drafter_tensor_meta *conv_t = native_find_tensor(m, conv_name);
            const ds4_drafter_tensor_meta *linear_norm_t = native_find_tensor(m, linear_norm_name);
            const ds4_drafter_tensor_meta *a_log_t = native_find_tensor(m, a_log_name);
            const ds4_drafter_tensor_meta *dt_bias_t = native_find_tensor(m, dt_bias_name);
            if (!input_norm_t || strcmp(input_norm_t->dtype, "BF16") != 0 ||
                input_norm_t->n_dims != 1 || input_norm_t->shape[0] != 1024 ||
                !post_norm_t || strcmp(post_norm_t->dtype, "BF16") != 0 ||
                post_norm_t->n_dims != 1 || post_norm_t->shape[0] != 1024 ||
                !conv_t || strcmp(conv_t->dtype, "BF16") != 0 ||
                conv_t->n_dims != 3 || conv_t->shape[0] != 6144 ||
                conv_t->shape[1] != 4 || conv_t->shape[2] != 1 ||
                !linear_norm_t || strcmp(linear_norm_t->dtype, "BF16") != 0 ||
                linear_norm_t->n_dims != 1 || linear_norm_t->shape[0] != 128 ||
                !a_log_t || strcmp(a_log_t->dtype, "F32") != 0 ||
                a_log_t->n_dims != 1 || a_log_t->shape[0] != 16 ||
                !dt_bias_t || strcmp(dt_bias_t->dtype, "BF16") != 0 ||
                dt_bias_t->n_dims != 1 || dt_bias_t->shape[0] != 16) {
                snprintf(err, errlen, "native drafter invalid fused linear layer tensor metadata");
                return -1;
            }
            const uint8_t *input_norm_data = native_tensor_data_ptr(m, input_norm_t, err, errlen);
            const uint8_t *post_norm_data = native_tensor_data_ptr(m, post_norm_t, err, errlen);
            const uint8_t *conv_data = native_tensor_data_ptr(m, conv_t, err, errlen);
            const uint8_t *linear_norm_data = native_tensor_data_ptr(m, linear_norm_t, err, errlen);
            const uint8_t *a_log_data = native_tensor_data_ptr(m, a_log_t, err, errlen);
            const uint8_t *dt_bias_data = native_tensor_data_ptr(m, dt_bias_t, err, errlen);
            if (!input_norm_data || !post_norm_data || !conv_data ||
                !linear_norm_data || !a_log_data || !dt_bias_data) {
                return -1;
            }
            ds4_drafter_metal_affine_job qkv_job;
            ds4_drafter_metal_affine_job z_job;
            ds4_drafter_metal_affine_job b_job;
            ds4_drafter_metal_affine_job a_job;
            ds4_drafter_metal_affine_job linear_out_job;
            ds4_drafter_metal_affine_job mlp_gate_job;
            ds4_drafter_metal_affine_job mlp_up_job;
            ds4_drafter_metal_affine_job mlp_down_job;
            if (native_metal_prepare_affine_job(m, qkv_name, 1024, 6144,
                                                &qkv_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, z_name, 1024, 2048,
                                                &z_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, b_name, 1024, 16,
                                                &b_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, a_name, 1024, 16,
                                                &a_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, linear_out_name, 2048, 1024,
                                                &linear_out_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, mlp_gate_name, 1024, 3584,
                                                &mlp_gate_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, mlp_up_name, 1024, 3584,
                                                &mlp_up_job, err, errlen) != 0 ||
                native_metal_prepare_affine_job(m, mlp_down_name, 3584, 1024,
                                                &mlp_down_job, err, errlen) != 0) {
                return -1;
            }
            if (!linear_cache->initialized) {
                linear_cache->input_norm_data = input_norm_data;
                linear_cache->input_norm_bytes = input_norm_t->end - input_norm_t->begin;
                linear_cache->post_norm_data = post_norm_data;
                linear_cache->post_norm_bytes = post_norm_t->end - post_norm_t->begin;
                linear_cache->conv_data = conv_data;
                linear_cache->conv_bytes = conv_t->end - conv_t->begin;
                linear_cache->linear_norm_data = linear_norm_data;
                linear_cache->linear_norm_bytes = linear_norm_t->end - linear_norm_t->begin;
                linear_cache->a_log_data = a_log_data;
                linear_cache->a_log_bytes = a_log_t->end - a_log_t->begin;
                linear_cache->dt_bias_data = dt_bias_data;
                linear_cache->dt_bias_bytes = dt_bias_t->end - dt_bias_t->begin;
                linear_cache->qkv_job = qkv_job;
                linear_cache->z_job = z_job;
                linear_cache->b_job = b_job;
                linear_cache->a_job = a_job;
                linear_cache->linear_out_job = linear_out_job;
                linear_cache->mlp_gate_job = mlp_gate_job;
                linear_cache->mlp_up_job = mlp_up_job;
                linear_cache->mlp_down_job = mlp_down_job;
                linear_cache->initialized = 1;
            }
            char metal_err[256] = {0};
            int metal_rc = ds4_drafter_metal_linear_decoder_layer_u32(
                input_norm_data, input_norm_t->end - input_norm_t->begin,
                post_norm_data, post_norm_t->end - post_norm_t->begin,
                &qkv_job, &z_job, &b_job, &a_job, &linear_out_job,
                conv_data, conv_t->end - conv_t->begin,
                linear_norm_data, linear_norm_t->end - linear_norm_t->begin,
                a_log_data, a_log_t->end - a_log_t->begin,
                dt_bias_data, dt_bias_t->end - dt_bias_t->begin,
                &mlp_gate_job, &mlp_up_job, &mlp_down_job,
                x, layer,
                m->cfg.quant_bits, m->cfg.quant_group_size,
                out, metal_err, sizeof(metal_err));
            if (metal_rc == 0) return 0;
            if (native_metal_strict()) {
                snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused linear decoder layer failed");
                return -1;
            }
            static int warned = 0;
            if (!warned) {
                fprintf(stderr,
                        "native drafter: fused Metal linear decoder layer failed, falling back to split path: %s\n",
                        metal_err[0] ? metal_err : "unknown error");
                warned = 1;
            }
        }
#endif
        if (!linear_conv_state || !linear_delta_state) {
            snprintf(err, errlen, "native drafter missing linear-attention state");
            return -1;
        }
        if (!x_norm_ready) {
            if (native_rms_norm(m, input_norm_name, x, 1024, 1.0e-6f,
                                x_norm, err, errlen) != 0) {
                return -1;
            }
            if (bf16_path) native_round_bf16_vector(x_norm, 1024);
            x_norm_ready = 1;
        }
        if (native_qwen_linear_attention_step(m, layer, x_norm,
                                              linear_conv_state,
                                              linear_delta_state,
                                              residual,
                                              err,
                                              errlen) != 0) {
            return -1;
        }
    } else {
#if defined(DS4_DRAFTER_HAS_METAL)
        if (try_fused_full) {
            int fused_rc = native_qwen_full_decoder_layer_metal_step(
                m, layer, input_norm_name, post_norm_name, x, position,
                full_state, full_query_capture, out, err, errlen);
            if (fused_rc == 0) return 0;
            if (fused_rc < 0) return -1;
        }
#endif
        if (!x_norm_ready) {
            if (native_rms_norm(m, input_norm_name, x, 1024, 1.0e-6f,
                                x_norm, err, errlen) != 0) {
                return -1;
            }
            if (bf16_path) native_round_bf16_vector(x_norm, 1024);
            x_norm_ready = 1;
        }
        if (native_qwen_full_attention_step(m, layer, x_norm, position,
                                            full_state, full_query_capture, residual,
                                            err, errlen) != 0) {
            return -1;
        }
    }
    if (bf16_path) native_round_bf16_vector(residual, 1024);
    for (int i = 0; i < 1024; i++) h[i] = x[i] + residual[i];
    if (bf16_path) native_round_bf16_vector(h, 1024);
    if (native_rms_norm(m, post_norm_name, h, 1024, 1.0e-6f,
                        post_norm, err, errlen) != 0) {
        return -1;
    }
    if (bf16_path) native_round_bf16_vector(post_norm, 1024);
    if (native_qwen_mlp(m, layer, post_norm, mlp_out, err, errlen) != 0) {
        return -1;
    }
    if (bf16_path) native_round_bf16_vector(mlp_out, 1024);
    for (int i = 0; i < 1024; i++) out[i] = h[i] + mlp_out[i];
    if (bf16_path) native_round_bf16_vector(out, 1024);
    return 0;
}

static void native_qwen_runtime_free(native_qwen_runtime *rt) {
    if (!rt) return;
    for (int i = 0; i < 24; i++) {
        free(rt->linear_conv[i]);
        free(rt->linear_delta[i]);
        native_full_attention_state_free(&rt->full[i]);
    }
    memset(rt, 0, sizeof(*rt));
}

static int native_qwen_runtime_linear_state(native_qwen_runtime *rt,
                                            int layer,
                                            float **conv_out,
                                            float **delta_out,
                                            char *err,
                                            size_t errlen) {
    if (layer < 0 || layer >= 24) {
        snprintf(err, errlen, "native drafter invalid linear layer index");
        return -1;
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    int allocated = 0;
#endif
    if (!rt->linear_conv[layer]) {
        rt->linear_conv[layer] = calloc((size_t)3u * 6144u,
                                        sizeof(rt->linear_conv[layer][0]));
        if (!rt->linear_conv[layer]) {
            snprintf(err, errlen, "native drafter out of memory allocating conv state");
            return -1;
        }
#if defined(DS4_DRAFTER_HAS_METAL)
        allocated = 1;
#endif
    }
    if (!rt->linear_delta[layer]) {
        rt->linear_delta[layer] = calloc((size_t)16u * 128u * 128u,
                                         sizeof(rt->linear_delta[layer][0]));
        if (!rt->linear_delta[layer]) {
            snprintf(err, errlen, "native drafter out of memory allocating delta state");
            return -1;
        }
#if defined(DS4_DRAFTER_HAS_METAL)
        allocated = 1;
#endif
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    if (allocated && native_metal_enabled()) {
        char metal_err[256] = {0};
        if (ds4_drafter_metal_linear_attention_reset(layer,
                                                     metal_err,
                                                     sizeof(metal_err)) != 0) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter linear state reset failed");
            return -1;
        }
    }
#endif
    *conv_out = rt->linear_conv[layer];
    *delta_out = rt->linear_delta[layer];
    return 0;
}

static double native_now_ms(void);

#if defined(DS4_DRAFTER_HAS_METAL)
static int native_qwen_resident_metal_hidden_step(
        const ds4_drafter_native_model *m,
        native_qwen_runtime *rt,
        const float *x,
        float **query_capture_by_layer,
        native_qwen_step_profile *profile,
        int need_output,
        float *out,
        char *err,
        size_t errlen) {
    if (!native_metal_enabled() || !native_metal_resident_token_enabled() ||
        rt->position <= 0) {
        return 1;
    }
    ds4_drafter_metal_decoder_layer_job jobs[24];
    memset(jobs, 0, sizeof(jobs));
    float *key_out_by_layer[24] = {0};
    float *value_out_by_layer[24] = {0};
    const bool mirror_cpu_kv = native_metal_resident_cpu_kv_enabled();
    int full_cache_index = -1;
    int full_n_ctx = -1;
    int full_cap = -1;
    for (int layer = 0; layer < 24; layer++) {
        jobs[layer].is_linear = native_qwen_layer_is_linear(layer);
        if (jobs[layer].is_linear) {
            native_metal_linear_layer_cache *lc =
                &((ds4_drafter_native_model *)m)->metal_linear_cache[layer];
            if (!lc->initialized) return 1;
            jobs[layer].input_norm_data = lc->input_norm_data;
            jobs[layer].input_norm_bytes = lc->input_norm_bytes;
            jobs[layer].post_norm_data = lc->post_norm_data;
            jobs[layer].post_norm_bytes = lc->post_norm_bytes;
            jobs[layer].conv_data = lc->conv_data;
            jobs[layer].conv_bytes = lc->conv_bytes;
            jobs[layer].linear_norm_data = lc->linear_norm_data;
            jobs[layer].linear_norm_bytes = lc->linear_norm_bytes;
            jobs[layer].a_log_data = lc->a_log_data;
            jobs[layer].a_log_bytes = lc->a_log_bytes;
            jobs[layer].dt_bias_data = lc->dt_bias_data;
            jobs[layer].dt_bias_bytes = lc->dt_bias_bytes;
            jobs[layer].qkv_job = lc->qkv_job;
            jobs[layer].z_job = lc->z_job;
            jobs[layer].b_job = lc->b_job;
            jobs[layer].a_job = lc->a_job;
            jobs[layer].linear_out_job = lc->linear_out_job;
            jobs[layer].mlp_gate_job = lc->mlp_gate_job;
            jobs[layer].mlp_up_job = lc->mlp_up_job;
            jobs[layer].mlp_down_job = lc->mlp_down_job;
        } else {
            native_metal_full_layer_cache *fc =
                &((ds4_drafter_native_model *)m)->metal_full_cache[layer];
            native_full_attention_state *st = &rt->full[layer];
            if (!fc->decoder_initialized ||
                (mirror_cpu_kv && (!st->keys || !st->values)) ||
                !st->metal_cache_valid || st->metal_cache_cap != st->cap ||
                st->len != rt->position) {
                return 1;
            }
            if (full_cache_index < 0) {
                full_cache_index = st->len;
                full_n_ctx = st->len + 1;
                full_cap = st->cap;
            } else if (full_cache_index != st->len ||
                       full_n_ctx != st->len + 1 ||
                       full_cap != st->cap) {
                return 1;
            }
            jobs[layer].input_norm_data = fc->input_norm_data;
            jobs[layer].input_norm_bytes = fc->input_norm_bytes;
            jobs[layer].post_norm_data = fc->post_norm_data;
            jobs[layer].post_norm_bytes = fc->post_norm_bytes;
            jobs[layer].q_norm_data = fc->q_norm_data;
            jobs[layer].q_norm_bytes = fc->q_norm_bytes;
            jobs[layer].k_norm_data = fc->k_norm_data;
            jobs[layer].k_norm_bytes = fc->k_norm_bytes;
            jobs[layer].q_job = fc->q_job;
            jobs[layer].k_job = fc->k_job;
            jobs[layer].v_job = fc->v_job;
            jobs[layer].o_job = fc->o_job;
            jobs[layer].mlp_gate_job = fc->mlp_gate_job;
            jobs[layer].mlp_up_job = fc->mlp_up_job;
            jobs[layer].mlp_down_job = fc->mlp_down_job;
            if (mirror_cpu_kv) {
                key_out_by_layer[layer] =
                    st->keys + (size_t)st->len * 2u * 256u;
                value_out_by_layer[layer] =
                    st->values + (size_t)st->len * 2u * 256u;
            }
        }
    }
    if (full_cache_index < 0 || full_n_ctx <= 0 || full_cap < full_n_ctx) {
        return 1;
    }
    const ds4_drafter_tensor_meta *final_norm =
        native_find_tensor(m, "language_model.model.norm.weight");
    if (!final_norm || strcmp(final_norm->dtype, "BF16") != 0 ||
        final_norm->n_dims != 1 || final_norm->shape[0] != 1024) {
        return 1;
    }
    const uint8_t *final_norm_data = native_tensor_data_ptr(m, final_norm, err, errlen);
    if (!final_norm_data) return -1;

    double t0 = profile ? native_now_ms() : 0.0;
    char metal_err[256] = {0};
    int rc = ds4_drafter_metal_qwen_hidden_step_u32(
        jobs, 24, x, rt->position,
        full_cache_index, full_n_ctx, full_cap,
        m->cfg.quant_bits, m->cfg.quant_group_size,
        query_capture_by_layer,
        mirror_cpu_kv ? key_out_by_layer : NULL,
        mirror_cpu_kv ? value_out_by_layer : NULL,
        need_output,
        final_norm_data, final_norm->end - final_norm->begin,
        out, metal_err, sizeof(metal_err));
    if (rc != 0) {
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal resident hidden step failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: resident Metal hidden step failed, falling back to layer path: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
        return 1;
    }
    for (int layer = 0; layer < 24; layer++) {
        if (native_qwen_layer_is_linear(layer)) continue;
        native_full_attention_state *st = &rt->full[layer];
        st->len++;
        st->metal_cache_valid = 1;
        st->metal_cache_cap = st->cap;
    }
    rt->position++;
    if (profile) {
        profile->resident_ms += native_now_ms() - t0;
        profile->resident_tokens++;
    }
    return 0;
}

static int native_qwen_sync_resident_full_caches(native_qwen_runtime *rt,
                                                 char *err,
                                                 size_t errlen) {
    if (!native_metal_enabled() || native_metal_resident_cpu_kv_enabled()) {
        return 0;
    }
    for (int layer = 0; layer < 24; layer++) {
        if (native_qwen_layer_is_linear(layer)) continue;
        native_full_attention_state *st = &rt->full[layer];
        if (!st->keys || !st->values || st->len <= 0 ||
            !st->metal_cache_valid || st->metal_cache_cap != st->cap) {
            continue;
        }
        char metal_err[256] = {0};
        if (ds4_drafter_metal_copy_full_attention_cache(layer, st->len,
                                                        st->keys, st->values,
                                                        metal_err,
                                                        sizeof(metal_err)) != 0) {
            snprintf(err, errlen, "%s",
                     metal_err[0] ? metal_err : "native Metal drafter full cache sync failed");
            return -1;
        }
    }
    return 0;
}

static bool native_metal_batch_prefill_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_BATCH_PREFILL");
    if (!env || !*env) return true;
    return strcmp(env, "0") != 0 && strcasecmp(env, "false") != 0;
}

static bool native_metal_resident_batch_prefill_enabled(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_RESIDENT_BATCH_PREFILL");
    if (!env || !*env) return true;
    return strcmp(env, "0") != 0 && strcasecmp(env, "false") != 0;
}

static int native_metal_prefill_chunk_size(void) {
    const char *env = getenv("DS4_DRAFTER_METAL_PREFILL_CHUNK");
    if (!env || !*env) return 0;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || v <= 0 || v > INT_MAX) return 0;
    return (int)v;
}

static int native_metal_bf16_vec_data(const ds4_drafter_native_model *m,
                                      const char *name,
                                      int len,
                                      const uint8_t **data_out,
                                      uint64_t *bytes_out,
                                      char *err,
                                      size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, name);
    if (!t || strcmp(t->dtype, "BF16") != 0 || t->n_dims != 1 ||
        t->shape[0] != len) {
        snprintf(err, errlen, "native drafter expected BF16 vector tensor: %s", name);
        return -1;
    }
    const uint8_t *data = native_tensor_data_ptr(m, t, err, errlen);
    if (!data) return -1;
    *data_out = data;
    *bytes_out = t->end - t->begin;
    return 0;
}

static int native_metal_f32_vec_data(const ds4_drafter_native_model *m,
                                     const char *name,
                                     int len,
                                     const uint8_t **data_out,
                                     uint64_t *bytes_out,
                                     char *err,
                                     size_t errlen) {
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, name);
    if (!t || strcmp(t->dtype, "F32") != 0 || t->n_dims != 1 ||
        t->shape[0] != len) {
        snprintf(err, errlen, "native drafter expected F32 vector tensor: %s", name);
        return -1;
    }
    const uint8_t *data = native_tensor_data_ptr(m, t, err, errlen);
    if (!data) return -1;
    *data_out = data;
    *bytes_out = t->end - t->begin;
    return 0;
}

static int native_qwen_prepare_linear_metal_cache(
        const ds4_drafter_native_model *m,
        int layer,
        native_metal_linear_layer_cache *cache,
        char *err,
        size_t errlen) {
    if (cache->initialized &&
        cache->input_norm_data && cache->input_norm_bytes >= 1024u * sizeof(uint16_t) &&
        cache->post_norm_data && cache->post_norm_bytes >= 1024u * sizeof(uint16_t) &&
        cache->conv_data && cache->conv_bytes >= 6144u * 4u * sizeof(uint16_t) &&
        cache->linear_norm_data && cache->linear_norm_bytes >= 128u * sizeof(uint16_t) &&
        cache->a_log_data && cache->a_log_bytes >= 16u * sizeof(float) &&
        cache->dt_bias_data && cache->dt_bias_bytes >= 16u * sizeof(uint16_t)) {
        return 0;
    }
    memset(cache, 0, sizeof(*cache));
    char input_norm_name[256];
    char post_norm_name[256];
    char qkv_name[256];
    char z_name[256];
    char b_name[256];
    char a_name[256];
    char conv_name[256];
    char linear_norm_name[256];
    char a_log_name[256];
    char dt_bias_name[256];
    char linear_out_name[256];
    char mlp_gate_name[256];
    char mlp_up_name[256];
    char mlp_down_name[256];
    int rc = snprintf(input_norm_name, sizeof(input_norm_name),
                      "language_model.model.layers.%d.input_layernorm.weight", layer);
    rc |= snprintf(post_norm_name, sizeof(post_norm_name),
                   "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
    rc |= snprintf(qkv_name, sizeof(qkv_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_qkv.weight", layer);
    rc |= snprintf(z_name, sizeof(z_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_z.weight", layer);
    rc |= snprintf(b_name, sizeof(b_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_b.weight", layer);
    rc |= snprintf(a_name, sizeof(a_name),
                   "language_model.model.layers.%d.linear_attn.in_proj_a.weight", layer);
    rc |= snprintf(conv_name, sizeof(conv_name),
                   "language_model.model.layers.%d.linear_attn.conv1d.weight", layer);
    rc |= snprintf(linear_norm_name, sizeof(linear_norm_name),
                   "language_model.model.layers.%d.linear_attn.norm.weight", layer);
    rc |= snprintf(a_log_name, sizeof(a_log_name),
                   "language_model.model.layers.%d.linear_attn.A_log", layer);
    rc |= snprintf(dt_bias_name, sizeof(dt_bias_name),
                   "language_model.model.layers.%d.linear_attn.dt_bias", layer);
    rc |= snprintf(linear_out_name, sizeof(linear_out_name),
                   "language_model.model.layers.%d.linear_attn.out_proj.weight", layer);
    rc |= snprintf(mlp_gate_name, sizeof(mlp_gate_name),
                   "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
    rc |= snprintf(mlp_up_name, sizeof(mlp_up_name),
                   "language_model.model.layers.%d.mlp.up_proj.weight", layer);
    rc |= snprintf(mlp_down_name, sizeof(mlp_down_name),
                   "language_model.model.layers.%d.mlp.down_proj.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad batched linear layer name");
        return -1;
    }
    if (native_metal_bf16_vec_data(m, input_norm_name, 1024,
                                   &cache->input_norm_data,
                                   &cache->input_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, post_norm_name, 1024,
                                   &cache->post_norm_data,
                                   &cache->post_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, linear_norm_name, 128,
                                   &cache->linear_norm_data,
                                   &cache->linear_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_f32_vec_data(m, a_log_name, 16,
                                  &cache->a_log_data,
                                  &cache->a_log_bytes,
                                  err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, dt_bias_name, 16,
                                   &cache->dt_bias_data,
                                   &cache->dt_bias_bytes,
                                   err, errlen) != 0) {
        return -1;
    }
    const ds4_drafter_tensor_meta *conv_t = native_find_tensor(m, conv_name);
    if (!conv_t || strcmp(conv_t->dtype, "BF16") != 0 ||
        conv_t->n_dims != 3 || conv_t->shape[0] != 6144 ||
        conv_t->shape[1] != 4 || conv_t->shape[2] != 1) {
        snprintf(err, errlen, "native drafter expected BF16 depthwise conv tensor: %s",
                 conv_name);
        return -1;
    }
    cache->conv_data = native_tensor_data_ptr(m, conv_t, err, errlen);
    if (!cache->conv_data) return -1;
    cache->conv_bytes = conv_t->end - conv_t->begin;
    if (native_metal_prepare_affine_job(m, qkv_name, 1024, 6144,
                                        &cache->qkv_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, z_name, 1024, 2048,
                                        &cache->z_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, b_name, 1024, 16,
                                        &cache->b_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, a_name, 1024, 16,
                                        &cache->a_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, linear_out_name, 2048, 1024,
                                        &cache->linear_out_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_gate_name, 1024, 3584,
                                        &cache->mlp_gate_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_up_name, 1024, 3584,
                                        &cache->mlp_up_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_down_name, 3584, 1024,
                                        &cache->mlp_down_job, err, errlen) != 0) {
        return -1;
    }
    cache->initialized = 1;
    return 0;
}

static int native_qwen_prepare_full_metal_cache(
        const ds4_drafter_native_model *m,
        int layer,
        native_metal_full_layer_cache *cache,
        char *err,
        size_t errlen) {
    if (cache->initialized &&
        cache->input_norm_data && cache->input_norm_bytes >= 1024u * sizeof(uint16_t) &&
        cache->post_norm_data && cache->post_norm_bytes >= 1024u * sizeof(uint16_t) &&
        cache->q_norm_data && cache->q_norm_bytes >= 256u * sizeof(uint16_t) &&
        cache->k_norm_data && cache->k_norm_bytes >= 256u * sizeof(uint16_t)) {
        return 0;
    }
    memset(cache, 0, sizeof(*cache));
    char input_norm_name[256];
    char post_norm_name[256];
    char q_name[256];
    char k_name[256];
    char v_name[256];
    char o_name[256];
    char q_norm_name[256];
    char k_norm_name[256];
    char mlp_gate_name[256];
    char mlp_up_name[256];
    char mlp_down_name[256];
    int rc = snprintf(input_norm_name, sizeof(input_norm_name),
                      "language_model.model.layers.%d.input_layernorm.weight", layer);
    rc |= snprintf(post_norm_name, sizeof(post_norm_name),
                   "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
    rc |= snprintf(q_name, sizeof(q_name),
                   "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
    rc |= snprintf(k_name, sizeof(k_name),
                   "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
    rc |= snprintf(v_name, sizeof(v_name),
                   "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
    rc |= snprintf(o_name, sizeof(o_name),
                   "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
    rc |= snprintf(q_norm_name, sizeof(q_norm_name),
                   "language_model.model.layers.%d.self_attn.q_norm.weight", layer);
    rc |= snprintf(k_norm_name, sizeof(k_norm_name),
                   "language_model.model.layers.%d.self_attn.k_norm.weight", layer);
    rc |= snprintf(mlp_gate_name, sizeof(mlp_gate_name),
                   "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
    rc |= snprintf(mlp_up_name, sizeof(mlp_up_name),
                   "language_model.model.layers.%d.mlp.up_proj.weight", layer);
    rc |= snprintf(mlp_down_name, sizeof(mlp_down_name),
                   "language_model.model.layers.%d.mlp.down_proj.weight", layer);
    if (rc <= 0) {
        snprintf(err, errlen, "native drafter bad batched full layer name");
        return -1;
    }
    if (native_metal_bf16_vec_data(m, input_norm_name, 1024,
                                   &cache->input_norm_data,
                                   &cache->input_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, post_norm_name, 1024,
                                   &cache->post_norm_data,
                                   &cache->post_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, q_norm_name, 256,
                                   &cache->q_norm_data,
                                   &cache->q_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_bf16_vec_data(m, k_norm_name, 256,
                                   &cache->k_norm_data,
                                   &cache->k_norm_bytes,
                                   err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, q_name, 1024, 4096,
                                        &cache->q_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, k_name, 1024, 512,
                                        &cache->k_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, v_name, 1024, 512,
                                        &cache->v_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, o_name, 2048, 1024,
                                        &cache->o_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_gate_name, 1024, 3584,
                                        &cache->mlp_gate_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_up_name, 1024, 3584,
                                        &cache->mlp_up_job, err, errlen) != 0 ||
        native_metal_prepare_affine_job(m, mlp_down_name, 3584, 1024,
                                        &cache->mlp_down_job, err, errlen) != 0) {
        return -1;
    }
    cache->initialized = 1;
    cache->decoder_initialized = 1;
    return 0;
}

static int native_qwen_prepare_all_metal_caches(ds4_drafter_native_model *m,
                                                char *err,
                                                size_t errlen) {
    if (!native_metal_enabled() || !native_metal_batch_prefill_enabled()) {
        return 0;
    }
    for (int layer = 0; layer < 24; layer++) {
        if (native_qwen_layer_is_linear(layer)) {
            if (native_qwen_prepare_linear_metal_cache(
                    m, layer, &m->metal_linear_cache[layer], err, errlen) != 0) {
                return -1;
            }
        } else {
            if (native_qwen_prepare_full_metal_cache(
                    m, layer, &m->metal_full_cache[layer], err, errlen) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int native_qwen_metal_batch_prefill(
        const ds4_drafter_native_model *m,
        native_qwen_runtime *rt,
        const native_token_vec *tokens,
        native_qwen_step_profile *profile,
        float *hidden_out,
        char *err,
        size_t errlen) {
    const int M = tokens->len;
    if (!native_metal_enabled() || !native_metal_batch_prefill_enabled() || M <= 1) {
        return 1;
    }
    double t0 = profile ? native_now_ms() : 0.0;
    float *hidden = malloc((size_t)M * 1024u * sizeof(hidden[0]));
    float *next = NULL;
    if (!hidden) {
        snprintf(err, errlen, "native drafter out of memory in batched Metal prefill");
        goto fail;
    }
    for (int i = 0; i < M; i++) {
        if (native_embedding_lookup(m, tokens->ids[i],
                                    hidden + (size_t)i * 1024u,
                                    1024, err, errlen) != 0) {
            goto fail;
        }
    }
    if (profile) {
        profile->embed_ms += native_now_ms() - t0;
        profile->tokens += M;
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_resident_batch_prefill_enabled()) {
        ds4_drafter_metal_decoder_layer_job jobs[24];
        memset(jobs, 0, sizeof(jobs));
        float *key_out_by_layer[24] = {0};
        float *value_out_by_layer[24] = {0};
        const bool mirror_cpu_kv = native_metal_resident_cpu_kv_enabled();
        const bool skip_cpu_kv = native_metal_skip_cpu_kv_enabled() && !mirror_cpu_kv;
        int prepared = 1;
        int full_cache_capacity = -1;
        for (int layer = 0; layer < 24; layer++) {
            jobs[layer].is_linear = native_qwen_layer_is_linear(layer);
            if (jobs[layer].is_linear) {
                native_metal_linear_layer_cache *cache =
                    &((ds4_drafter_native_model *)m)->metal_linear_cache[layer];
                if (native_qwen_prepare_linear_metal_cache(m, layer, cache,
                                                           err, errlen) != 0) {
                    prepared = 0;
                    break;
                }
                jobs[layer].input_norm_data = cache->input_norm_data;
                jobs[layer].input_norm_bytes = cache->input_norm_bytes;
                jobs[layer].post_norm_data = cache->post_norm_data;
                jobs[layer].post_norm_bytes = cache->post_norm_bytes;
                jobs[layer].conv_data = cache->conv_data;
                jobs[layer].conv_bytes = cache->conv_bytes;
                jobs[layer].linear_norm_data = cache->linear_norm_data;
                jobs[layer].linear_norm_bytes = cache->linear_norm_bytes;
                jobs[layer].a_log_data = cache->a_log_data;
                jobs[layer].a_log_bytes = cache->a_log_bytes;
                jobs[layer].dt_bias_data = cache->dt_bias_data;
                jobs[layer].dt_bias_bytes = cache->dt_bias_bytes;
                jobs[layer].qkv_job = cache->qkv_job;
                jobs[layer].z_job = cache->z_job;
                jobs[layer].b_job = cache->b_job;
                jobs[layer].a_job = cache->a_job;
                jobs[layer].linear_out_job = cache->linear_out_job;
                jobs[layer].mlp_gate_job = cache->mlp_gate_job;
                jobs[layer].mlp_up_job = cache->mlp_up_job;
                jobs[layer].mlp_down_job = cache->mlp_down_job;
            } else {
                native_metal_full_layer_cache *cache =
                    &((ds4_drafter_native_model *)m)->metal_full_cache[layer];
                native_full_attention_state *st = &rt->full[layer];
                int reserve_rc = skip_cpu_kv ?
                    native_full_attention_state_reserve_metal_only(st, M, err, errlen) :
                    native_full_attention_state_reserve(st, M, err, errlen);
                if (reserve_rc != 0 ||
                    native_qwen_prepare_full_metal_cache(m, layer, cache,
                                                         err, errlen) != 0) {
                    prepared = 0;
                    break;
                }
                if (full_cache_capacity < 0) {
                    full_cache_capacity = st->cap;
                } else if (full_cache_capacity != st->cap) {
                    snprintf(err, errlen, "native drafter mismatched full cache capacity in resident batch prefill");
                    prepared = 0;
                    break;
                }
                jobs[layer].input_norm_data = cache->input_norm_data;
                jobs[layer].input_norm_bytes = cache->input_norm_bytes;
                jobs[layer].post_norm_data = cache->post_norm_data;
                jobs[layer].post_norm_bytes = cache->post_norm_bytes;
                jobs[layer].q_norm_data = cache->q_norm_data;
                jobs[layer].q_norm_bytes = cache->q_norm_bytes;
                jobs[layer].k_norm_data = cache->k_norm_data;
                jobs[layer].k_norm_bytes = cache->k_norm_bytes;
                jobs[layer].q_job = cache->q_job;
                jobs[layer].k_job = cache->k_job;
                jobs[layer].v_job = cache->v_job;
                jobs[layer].o_job = cache->o_job;
                jobs[layer].mlp_gate_job = cache->mlp_gate_job;
                jobs[layer].mlp_up_job = cache->mlp_up_job;
                jobs[layer].mlp_down_job = cache->mlp_down_job;
                if (mirror_cpu_kv) {
                    key_out_by_layer[layer] = st->keys;
                    value_out_by_layer[layer] = st->values;
                }
            }
        }
        if (prepared && full_cache_capacity >= M) {
            double resident_t0 = profile ? native_now_ms() : 0.0;
            float last_hidden[1024];
            char metal_err[512] = {0};
            int chunk_size = native_metal_prefill_chunk_size();
            int rc = 0;
            if (chunk_size > 0 && chunk_size < M) {
                for (int base = 0; base < M; base += chunk_size) {
                    int chunk_n = M - base;
                    if (chunk_n > chunk_size) chunk_n = chunk_size;
                    rc = ds4_drafter_metal_qwen_batch_prefill_chunk_u32(
                        jobs, 24, hidden + (size_t)base * 1024u, chunk_n,
                        base, base + chunk_n, full_cache_capacity,
                        m->cfg.quant_bits, m->cfg.quant_group_size,
                        mirror_cpu_kv ? key_out_by_layer : NULL,
                        mirror_cpu_kv ? value_out_by_layer : NULL,
                        last_hidden, metal_err, sizeof(metal_err));
                    if (rc != 0) break;
                }
            } else {
                rc = ds4_drafter_metal_qwen_batch_prefill_u32(
                    jobs, 24, hidden, M,
                    full_cache_capacity,
                    m->cfg.quant_bits, m->cfg.quant_group_size,
                    mirror_cpu_kv ? key_out_by_layer : NULL,
                    mirror_cpu_kv ? value_out_by_layer : NULL,
                    last_hidden, metal_err, sizeof(metal_err));
            }
            if (rc == 0) {
                double final_norm_t0 = profile ? native_now_ms() : 0.0;
                const char *debug_hidden = getenv("DS4_DRAFTER_DEBUG_HIDDEN");
                if (debug_hidden && strcmp(debug_hidden, "1") == 0) {
                    double sum = 0.0;
                    float min_v = last_hidden[0];
                    float max_v = last_hidden[0];
                    int finite = 0;
                    int nan_count = 0;
                    for (int i = 0; i < 1024; i++) {
                        float v = last_hidden[i];
                        if (isfinite(v)) {
                            if (v < min_v) min_v = v;
                            if (v > max_v) max_v = v;
                            sum += (double)v;
                            finite++;
                        } else {
                            nan_count++;
                        }
                    }
                    fprintf(stderr,
                            "NATIVE_LAST_HIDDEN finite=%d nan=%d min=%.9g max=%.9g sum=%.9g first=%.9g\n",
                            finite, nan_count, min_v, max_v, sum, last_hidden[0]);
                }
                if (native_rms_norm(m, "language_model.model.norm.weight",
                                    last_hidden, 1024, 1.0e-6f,
                                    hidden_out, err, errlen) != 0) {
                    goto fail;
                }
                native_round_bf16_vector(hidden_out, 1024);
                if (debug_hidden && strcmp(debug_hidden, "1") == 0) {
                    double sum = 0.0;
                    float min_v = hidden_out[0];
                    float max_v = hidden_out[0];
                    int finite = 0;
                    int nan_count = 0;
                    for (int i = 0; i < 1024; i++) {
                        float v = hidden_out[i];
                        if (isfinite(v)) {
                            if (v < min_v) min_v = v;
                            if (v > max_v) max_v = v;
                            sum += (double)v;
                            finite++;
                        } else {
                            nan_count++;
                        }
                    }
                    fprintf(stderr,
                            "NATIVE_PREFILL_HIDDEN finite=%d nan=%d min=%.9g max=%.9g sum=%.9g first=%.9g\n",
                            finite, nan_count, min_v, max_v, sum, hidden_out[0]);
                }
                for (int layer = 0; layer < 24; layer++) {
                    if (native_qwen_layer_is_linear(layer)) continue;
                    native_full_attention_state *st = &rt->full[layer];
                    st->len = M;
                    st->metal_cache_valid = 1;
                    st->metal_cache_cap = st->cap;
                }
                rt->position = M;
                if (profile) {
                    profile->resident_ms += native_now_ms() - resident_t0;
                    profile->resident_tokens += M;
                    profile->linear_layers += 18;
                    profile->full_layers += 6;
                    profile->final_norm_ms += native_now_ms() - final_norm_t0;
                }
                free(hidden);
                return 0;
            }
            if (native_metal_strict()) {
                snprintf(err, errlen, "%s",
                         metal_err[0] ? metal_err : "native Metal resident batch prefill failed");
                goto fail;
            }
            static int warned = 0;
            if (!warned) {
                fprintf(stderr,
                        "native drafter: resident Metal batch prefill failed, falling back to per-layer path: %s\n",
                        metal_err[0] ? metal_err : "unknown error");
                warned = 1;
            }
        } else if (native_metal_strict()) {
            goto fail;
        }
    }
#endif
    next = malloc((size_t)M * 1024u * sizeof(next[0]));
    if (!next) {
        snprintf(err, errlen, "native drafter out of memory in batched Metal prefill fallback");
        goto fail;
    }
    for (int layer = 0; layer < 24; layer++) {
        double layer_t0 = profile ? native_now_ms() : 0.0;
        if (native_qwen_layer_is_linear(layer)) {
            native_metal_linear_layer_cache *cache =
                &((ds4_drafter_native_model *)m)->metal_linear_cache[layer];
            if (native_qwen_prepare_linear_metal_cache(m, layer, cache,
                                                       err, errlen) != 0) {
                goto fail;
            }
            char metal_err[256] = {0};
            if (ds4_drafter_metal_linear_decoder_layer_u32_mat(
                    cache->input_norm_data, cache->input_norm_bytes,
                    cache->post_norm_data, cache->post_norm_bytes,
                    &cache->qkv_job, &cache->z_job, &cache->b_job, &cache->a_job,
                    &cache->linear_out_job,
                    cache->conv_data, cache->conv_bytes,
                    cache->linear_norm_data, cache->linear_norm_bytes,
                    cache->a_log_data, cache->a_log_bytes,
                    cache->dt_bias_data, cache->dt_bias_bytes,
                    &cache->mlp_gate_job, &cache->mlp_up_job, &cache->mlp_down_job,
                    hidden, M, layer,
                    m->cfg.quant_bits, m->cfg.quant_group_size,
                    next, metal_err, sizeof(metal_err)) != 0) {
                snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter batched linear layer failed");
                goto fail;
            }
            if (profile) {
                profile->linear_ms += native_now_ms() - layer_t0;
                profile->linear_layers++;
            }
            float *swap = hidden;
            hidden = next;
            next = swap;
            continue;
        } else {
            native_metal_full_layer_cache *cache =
                &((ds4_drafter_native_model *)m)->metal_full_cache[layer];
            native_full_attention_state *st = &rt->full[layer];
            if (native_full_attention_state_reserve(st, M, err, errlen) != 0 ||
                native_qwen_prepare_full_metal_cache(m, layer, cache,
                                                     err, errlen) != 0) {
                goto fail;
            }
            char metal_err[256] = {0};
            if (ds4_drafter_metal_full_decoder_layer_u32_mat(
                    cache->input_norm_data, cache->input_norm_bytes,
                    cache->post_norm_data, cache->post_norm_bytes,
                    &cache->q_job, &cache->k_job, &cache->v_job, &cache->o_job,
                    cache->q_norm_data, cache->q_norm_bytes,
                    cache->k_norm_data, cache->k_norm_bytes,
                    &cache->mlp_gate_job, &cache->mlp_up_job, &cache->mlp_down_job,
                    hidden, M, m->cfg.quant_bits, m->cfg.quant_group_size,
                    next, st->keys, st->values, metal_err, sizeof(metal_err)) != 0) {
                snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter fused full layer failed");
                goto fail;
            }
            st->len = M;
            st->metal_cache_valid = 0;
            st->metal_cache_cap = 0;
            if (profile) {
                profile->full_ms += native_now_ms() - layer_t0;
                profile->full_layers++;
            }
            float *swap = hidden;
            hidden = next;
            next = swap;
            continue;
        }
    }
    double final_norm_t0 = profile ? native_now_ms() : 0.0;
    if (native_rms_norm(m, "language_model.model.norm.weight",
                        hidden + (size_t)(M - 1) * 1024u, 1024, 1.0e-6f,
                        hidden_out, err, errlen) != 0) {
        goto fail;
    }
    native_round_bf16_vector(hidden_out, 1024);
    rt->position = M;
    if (profile) profile->final_norm_ms += native_now_ms() - final_norm_t0;
    free(hidden);
    free(next);
    return 0;

fail:
    free(hidden);
    free(next);
    return -1;
}
#endif

static int native_qwen_model_hidden_step(const ds4_drafter_native_model *m,
                                         native_qwen_runtime *rt,
                                         int token_id,
                                         float **query_capture_by_layer,
                                         native_qwen_step_profile *profile,
                                         int need_output,
                                         float *out,
                                         char *err,
                                         size_t errlen) {
    float x_a[1024];
    float x_b[1024];
    double t0 = profile ? native_now_ms() : 0.0;
    if (native_embedding_lookup(m, token_id, x_a, 1024, err, errlen) != 0) {
        return -1;
    }
    if (profile) {
        profile->embed_ms += native_now_ms() - t0;
        profile->tokens++;
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    {
        int resident_rc = native_qwen_resident_metal_hidden_step(
            m, rt, x_a, query_capture_by_layer, profile, need_output,
            out, err, errlen);
        if (resident_rc == 0) return 0;
        if (resident_rc < 0) return -1;
    }
#endif
    const float *src = x_a;
    float *dst = x_b;
    for (int layer = 0; layer < 24; layer++) {
        float *conv = NULL;
        float *delta = NULL;
        native_full_attention_state *full = NULL;
        if (native_qwen_layer_is_linear(layer)) {
            int skip_cpu_linear_state = 0;
#if defined(DS4_DRAFTER_HAS_METAL)
            skip_cpu_linear_state = native_metal_enabled() &&
                                    native_metal_fused_linear_enabled();
            if (skip_cpu_linear_state && rt->position == 0) {
                char metal_err[256] = {0};
                if (ds4_drafter_metal_linear_attention_reset(layer,
                                                             metal_err,
                                                             sizeof(metal_err)) != 0) {
                    snprintf(err, errlen, "%s",
                             metal_err[0] ? metal_err : "native Metal drafter linear state reset failed");
                    return -1;
                }
            }
#endif
            if (!skip_cpu_linear_state &&
                native_qwen_runtime_linear_state(rt, layer, &conv, &delta,
                                                 err, errlen) != 0) {
                return -1;
            }
        } else {
            full = &rt->full[layer];
        }
        float *query_capture = query_capture_by_layer ? query_capture_by_layer[layer] : NULL;
        double layer_t0 = profile ? native_now_ms() : 0.0;
        if (native_qwen_decoder_layer_step(m, layer, src, rt->position,
                                           conv, delta, full, 1, query_capture, dst,
                                           err, errlen) != 0) {
            return -1;
        }
        if (profile) {
            double layer_ms = native_now_ms() - layer_t0;
            if (native_qwen_layer_is_linear(layer)) {
                profile->linear_ms += layer_ms;
                profile->linear_layers++;
            } else {
                profile->full_ms += layer_ms;
                profile->full_layers++;
            }
        }
        const float *next_src = dst;
        dst = (dst == x_b) ? x_a : x_b;
        src = next_src;
    }
    if (need_output) {
        double norm_t0 = profile ? native_now_ms() : 0.0;
        if (native_rms_norm(m, "language_model.model.norm.weight", src, 1024, 1.0e-6f,
                            out, err, errlen) != 0) {
            return -1;
        }
        native_round_bf16_vector(out, 1024);
        if (profile) profile->final_norm_ms += native_now_ms() - norm_t0;
    }
    rt->position++;
    return 0;
}

static double native_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

typedef struct {
    const uint8_t *w_data;
    const uint8_t *s_data;
    const uint8_t *b_data;
    const float *hidden;
    int row_begin;
    int row_end;
    int packed_cols;
    int groups;
    int bits;
    int group_size;
    int best_token;
    double best_score;
} native_argmax_task;

static void native_qwen_logits_argmax_rows(native_argmax_task *task) {
    const int pack = 32 / task->bits;
    const uint32_t mask = (1u << task->bits) - 1u;
    task->best_token = task->row_begin;
    task->best_score = -1.0e300;
    for (int r = task->row_begin; r < task->row_end; r++) {
        const uint8_t *w_row = task->w_data + (size_t)r * (size_t)task->packed_cols * 4u;
        const uint8_t *s_row = task->s_data + (size_t)r * (size_t)task->groups * 2u;
        const uint8_t *b_row = task->b_data + (size_t)r * (size_t)task->groups * 2u;
        double acc = 0.0;
        for (int pc = 0; pc < task->packed_cols; pc++) {
            uint32_t packed = native_read_le32(w_row + (size_t)pc * 4u);
            for (int lane = 0; lane < pack; lane++) {
                int col = pc * pack + lane;
                int g = col / task->group_size;
                float scale = native_bf16_to_f32(native_read_le16(s_row + (size_t)g * 2u));
                float bias = native_bf16_to_f32(native_read_le16(b_row + (size_t)g * 2u));
                uint32_t q = (packed >> (lane * task->bits)) & mask;
                acc += (double)task->hidden[col] * ((double)q * (double)scale + (double)bias);
            }
        }
        if (r == task->row_begin || acc > task->best_score) {
            task->best_score = acc;
            task->best_token = r;
        }
    }
}

static void *native_argmax_thread_main(void *arg) {
    native_qwen_logits_argmax_rows((native_argmax_task *)arg);
    return NULL;
}

static int native_qwen_logits_argmax(const ds4_drafter_native_model *m,
                                     const float *hidden,
                                     int *token_out,
                                     char *err,
                                     size_t errlen) {
    const char *weight_name = "language_model.model.embed_tokens.weight";
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, weight_name);
    const ds4_drafter_tensor_meta *scales =
        native_find_tensor(m, "language_model.model.embed_tokens.scales");
    const ds4_drafter_tensor_meta *biases =
        native_find_tensor(m, "language_model.model.embed_tokens.biases");
    if (!t || !scales || !biases ||
        strcmp(t->dtype, "U32") != 0 ||
        strcmp(scales->dtype, "BF16") != 0 ||
        strcmp(biases->dtype, "BF16") != 0 ||
        t->n_dims != 2 ||
        scales->n_dims != 2 ||
        biases->n_dims != 2) {
        snprintf(err, errlen, "native drafter expected quantized tied embedding tensors");
        return -1;
    }
    const int bits = m->cfg.quant_bits;
    const int group_size = m->cfg.quant_group_size;
    const int pack = 32 / bits;
    const int rows = (int)t->shape[0];
    const int packed_cols = (int)t->shape[1];
    const int cols = packed_cols * pack;
    const int groups = cols / group_size;
    if (cols != 1024 ||
        scales->shape[0] != rows ||
        biases->shape[0] != rows ||
        scales->shape[1] != groups ||
        biases->shape[1] != groups) {
        snprintf(err, errlen, "native drafter tied embedding shape mismatch");
        return -1;
    }
    const uint8_t *w_data = native_tensor_data_ptr(m, t, err, errlen);
    const uint8_t *s_data = native_tensor_data_ptr(m, scales, err, errlen);
    const uint8_t *b_data = native_tensor_data_ptr(m, biases, err, errlen);
    if (!w_data || !s_data || !b_data) return -1;

#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_enabled()) {
        char metal_err[256] = {0};
        int metal_token = -1;
        int metal_rc = ds4_drafter_metal_logits_argmax_u32(
            w_data, t->end - t->begin,
            s_data, scales->end - scales->begin,
            b_data, biases->end - biases->begin,
            hidden, rows, packed_cols, cols, groups, bits, group_size,
            &metal_token, metal_err, sizeof(metal_err));
        if (metal_rc == 0) {
            *token_out = metal_token;
            return 0;
        }
        if (native_metal_strict()) {
            snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter logits argmax failed");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "native drafter: Metal logits argmax failed, falling back to CPU: %s\n",
                    metal_err[0] ? metal_err : "unknown error");
            warned = 1;
        }
    }
#endif

    int n_threads = native_drafter_threads();
    if (n_threads > rows) n_threads = rows;
    native_argmax_task stack_tasks[16];
    pthread_t stack_threads[16];
    unsigned char stack_started[16];
    native_argmax_task *tasks = stack_tasks;
    pthread_t *threads = stack_threads;
    unsigned char *started_flags = stack_started;
    memset(stack_started, 0, sizeof(stack_started));
    if (n_threads > (int)(sizeof(stack_tasks) / sizeof(stack_tasks[0]))) {
        tasks = calloc((size_t)n_threads, sizeof(tasks[0]));
        threads = calloc((size_t)n_threads, sizeof(threads[0]));
        started_flags = calloc((size_t)n_threads, sizeof(started_flags[0]));
        if (!tasks || !threads || !started_flags) {
            free(tasks);
            free(threads);
            free(started_flags);
            snprintf(err, errlen, "native drafter out of memory creating argmax tasks");
            return -1;
        }
    }
    int rows_per = (rows + n_threads - 1) / n_threads;
    for (int tix = 0; tix < n_threads; tix++) {
        int begin = tix * rows_per;
        int end = begin + rows_per;
        if (end > rows) end = rows;
        if (begin >= end) break;
        tasks[tix] = (native_argmax_task){
            .w_data = w_data,
            .s_data = s_data,
            .b_data = b_data,
            .hidden = hidden,
            .row_begin = begin,
            .row_end = end,
            .packed_cols = packed_cols,
            .groups = groups,
            .bits = bits,
            .group_size = group_size,
        };
        if (tix == 0) {
            native_qwen_logits_argmax_rows(&tasks[tix]);
        } else if (pthread_create(&threads[tix], NULL, native_argmax_thread_main,
                                  &tasks[tix]) == 0) {
            started_flags[tix] = 1;
        } else {
            native_qwen_logits_argmax_rows(&tasks[tix]);
        }
    }
    for (int tix = 1; tix < n_threads; tix++) {
        if (started_flags[tix]) pthread_join(threads[tix], NULL);
    }
    int best_token = tasks[0].best_token;
    double best = tasks[0].best_score;
    for (int tix = 1; tix < n_threads; tix++) {
        if (tasks[tix].row_begin >= tasks[tix].row_end) continue;
        if (tasks[tix].best_score > best ||
            (tasks[tix].best_score == best && tasks[tix].best_token < best_token)) {
            best = tasks[tix].best_score;
            best_token = tasks[tix].best_token;
        }
    }
    if (tasks != stack_tasks) {
        free(tasks);
        free(threads);
        free(started_flags);
    }
    *token_out = best_token;
    return 0;
}

typedef struct {
    int index;
    float score;
} native_block_score;

static double g_native_block_recency_bias = 0.0;

static int native_block_score_cmp_desc(const void *a, const void *b) {
    const native_block_score *aa = (const native_block_score *)a;
    const native_block_score *bb = (const native_block_score *)b;
    double as = (double)aa->score + g_native_block_recency_bias * (double)aa->index;
    double bs = (double)bb->score + g_native_block_recency_bias * (double)bb->index;
    if (as < bs) return 1;
    if (as > bs) return -1;
    return aa->index - bb->index;
}

static int native_select_keep_scores(const float *importance,
                                     int n_tokens,
                                     int block_size,
                                     float keep_fraction,
                                     int sink_size,
                                     int tail_keep,
                                     double default_score_quantum,
                                     float *scores,
                                     char *err,
                                     size_t errlen) {
    if (n_tokens <= 0) return 0;
    for (int i = 0; i < n_tokens; i++) scores[i] = 0.0f;
    if (keep_fraction >= 1.0f) {
        for (int i = 0; i < n_tokens; i++) scores[i] = 1.0f;
        return 0;
    }
    if (block_size <= 0) block_size = 32;
    int n_blocks = (n_tokens + block_size - 1) / block_size;
    native_block_score *blocks = calloc((size_t)n_blocks, sizeof(blocks[0]));
    if (!blocks) {
        snprintf(err, errlen, "native drafter out of memory selecting blocks");
        return -1;
    }
    double score_quantum = default_score_quantum;
    const char *quantum_env = getenv("DS4_DRAFTER_BLOCK_SCORE_QUANTUM");
    if (quantum_env && quantum_env[0]) {
        char *end = NULL;
        double v = strtod(quantum_env, &end);
        if (end != quantum_env && isfinite(v) && v >= 0.0) {
            score_quantum = v;
        }
    }
    for (int b = 0; b < n_blocks; b++) {
        int start = b * block_size;
        double sum = 0.0;
        for (int j = 0; j < block_size; j++) {
            int idx = start + j;
            sum += idx < n_tokens ? (double)importance[idx] : -1.0e30;
        }
        blocks[b].index = b;
        double block_score = sum / (double)block_size;
        if (score_quantum > 0.0) {
            block_score = round(block_score / score_quantum) * score_quantum;
        }
        blocks[b].score = (float)block_score;
    }
    g_native_block_recency_bias = 0.0;
    const char *bias_env = getenv("DS4_DRAFTER_BLOCK_RECENCY_BIAS");
    if (bias_env && bias_env[0]) {
        char *end = NULL;
        double v = strtod(bias_env, &end);
        if (end != bias_env && isfinite(v) && v >= 0.0) {
            g_native_block_recency_bias = v;
        }
    }
    qsort(blocks, (size_t)n_blocks, sizeof(blocks[0]), native_block_score_cmp_desc);
    double keep_blocks = (double)keep_fraction * (double)n_blocks;
    double nearest = round(keep_blocks);
    if (fabs(keep_blocks - nearest) < 1.0e-4) keep_blocks = nearest;
    int k = (int)ceil(keep_blocks);
    if (k < 1) k = 1;
    if (k > n_blocks) k = n_blocks;
    const char *debug_blocks = getenv("DS4_DRAFTER_DEBUG_BLOCKS");
    if (debug_blocks && strcmp(debug_blocks, "1") == 0) {
        fprintf(stderr, "NATIVE_BLOCKS %d %d %d %d", n_tokens, block_size, n_blocks, k);
        for (int i = 0; i < n_blocks; i++) {
            fprintf(stderr, " %d:%.9g", blocks[i].index, blocks[i].score);
        }
        fprintf(stderr, "\n");
    }
    for (int i = 0; i < k; i++) {
        int start = blocks[i].index * block_size;
        int end = start + block_size;
        if (end > n_tokens) end = n_tokens;
        for (int j = start; j < end; j++) scores[j] = 1.0f;
    }
    free(blocks);
    if (sink_size > n_tokens) sink_size = n_tokens;
    for (int i = 0; i < sink_size; i++) scores[i] = 1.0f;
    int tail_start = n_tokens - tail_keep;
    if (tail_start < 0) tail_start = 0;
    for (int i = tail_start; i < n_tokens; i++) scores[i] = 1.0f;
    scores[n_tokens - 1] = 1.0f;
    return 0;
}

static int native_score_prompt_tokens(const ds4_drafter_native_model *m,
                                      const ds4_drafter_options *opt,
                                      const native_token_vec *tokens,
                                      float **q_scores_out,
                                      char *err,
                                      size_t errlen) {
    const int M = tokens->len;
    const bool profile = native_drafter_profile_enabled();
    double prof_start = 0.0;
    double prof_prefill = 0.0;
    double prof_argmax = 0.0;
    double prof_lookahead = 0.0;
    double prof_importance = 0.0;
    double prof_select = 0.0;
    if (profile) prof_start = native_now_ms();
    *q_scores_out = NULL;
    if (M <= 0) {
        *q_scores_out = calloc(1, sizeof(float));
        return *q_scores_out ? 0 : -1;
    }
    const int lookahead = opt->score_lookahead > 0 ? opt->score_lookahead : 4;
    int pool_kernel = opt->score_pool_kernel > 0 ? opt->score_pool_kernel : 13;
    if (pool_kernel < 1) pool_kernel = 1;
    if ((pool_kernel & 1) == 0) pool_kernel++;
    int block_size = opt->chunk > 0 ? opt->chunk : 32;
    float keep_pct = opt->keep_pct > 0.0f ? opt->keep_pct : 0.3f;
    int sink = opt->sink >= 0 ? opt->sink : 16;
    int tail = opt->tail >= 0 ? opt->tail : 256;
    if (keep_pct >= 1.0f || M <= sink + tail) {
        float *q_scores = malloc((size_t)M * sizeof(q_scores[0]));
        if (!q_scores) {
            snprintf(err, errlen, "native drafter out of memory selecting scores");
            return -1;
        }
        for (int i = 0; i < M; i++) q_scores[i] = 1.0f;
        *q_scores_out = q_scores;
        return 0;
    }

    native_qwen_runtime rt = {0};
    native_qwen_step_profile prefill_step_profile = {0};
    native_qwen_step_profile lookahead_step_profile = {0};
    for (int layer = 0; layer < 24; layer++) {
        if (native_qwen_layer_is_linear(layer)) continue;
#if defined(DS4_DRAFTER_HAS_METAL)
        if (native_metal_enabled() &&
            native_metal_skip_cpu_kv_enabled() &&
            !native_metal_resident_cpu_kv_enabled() &&
            native_metal_resident_batch_prefill_enabled() &&
            native_metal_resident_token_enabled() &&
            native_metal_importance_enabled()) {
            if (native_full_attention_state_reserve_metal_only(&rt.full[layer],
                                                               M + lookahead,
                                                               err,
                                                               errlen) != 0) {
                native_qwen_runtime_free(&rt);
                return -1;
            }
            continue;
        }
#endif
        if (native_full_attention_state_reserve(&rt.full[layer],
                                                M + lookahead,
                                                err,
                                                errlen) != 0) {
            native_qwen_runtime_free(&rt);
            return -1;
        }
    }
	    float hidden[1024];
	#if defined(DS4_DRAFTER_HAS_METAL)
	    int batch_prefill_done = 0;
	    if (native_metal_enabled() && native_metal_batch_prefill_enabled()) {
	        char metal_err[512] = {0};
	        int batch_rc = native_qwen_metal_batch_prefill(
	            m, &rt, tokens, profile ? &prefill_step_profile : NULL,
	            hidden, metal_err, sizeof(metal_err));
	        if (batch_rc == 0) {
	            batch_prefill_done = 1;
	        } else if (native_metal_strict()) {
	            snprintf(err, errlen, "%s",
	                     metal_err[0] ? metal_err : "native Metal drafter batched prefill failed");
	            native_qwen_runtime_free(&rt);
	            return -1;
	        } else {
	            static int warned = 0;
	            if (!warned) {
	                fprintf(stderr,
	                        "native drafter: batched Metal prefill failed, falling back to token loop: %s\n",
	                        metal_err[0] ? metal_err : "unknown error");
	                warned = 1;
	            }
	        }
	    }
	    if (!batch_prefill_done)
	#endif
    {
        for (int i = 0; i < M; i++) {
            int need_output = (i == M - 1);
            if (native_qwen_model_hidden_step(m, &rt, tokens->ids[i], NULL,
                                              profile ? &prefill_step_profile : NULL,
	                                              need_output,
	                                              hidden, err, errlen) != 0) {
	                native_qwen_runtime_free(&rt);
	                return -1;
            }
        }
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_metal_prefill_cache_sync_enabled() &&
        native_qwen_sync_resident_full_caches(&rt, err, errlen) != 0) {
        native_qwen_runtime_free(&rt);
        return -1;
    }
#endif
    if (profile) prof_prefill = native_now_ms() - prof_start;

    double prof_argmax_start = profile ? native_now_ms() : 0.0;
    int y = 0;
    if (native_qwen_logits_argmax(m, hidden, &y, err, errlen) != 0) {
        native_qwen_runtime_free(&rt);
        return -1;
    }
    const char *debug_argmax = getenv("DS4_DRAFTER_DEBUG_ARGMAX");
    if (debug_argmax && strcmp(debug_argmax, "1") == 0) {
        fprintf(stderr, "NATIVE_ARGMAX %d", y);
    }
    if (profile) prof_argmax = native_now_ms() - prof_argmax_start;

    float *queries = calloc((size_t)lookahead * 24u * 2048u, sizeof(queries[0]));
    float *max_per_step = calloc((size_t)lookahead * (size_t)M, sizeof(max_per_step[0]));
    float *importance = calloc((size_t)M, sizeof(importance[0]));
    float *probs = malloc((size_t)M * sizeof(probs[0]));
    float *smooth = malloc((size_t)M * sizeof(smooth[0]));
    if (!queries || !max_per_step || !importance || !probs || !smooth) {
        free(queries);
        free(max_per_step);
        free(importance);
        free(probs);
        free(smooth);
        native_qwen_runtime_free(&rt);
        snprintf(err, errlen, "native drafter out of memory scoring prompt");
        return -1;
    }

    double prof_lookahead_start = profile ? native_now_ms() : 0.0;
    for (int s = 0; s < lookahead; s++) {
        float *capture_by_layer[24] = {0};
        for (int layer = 0; layer < 24; layer++) {
            if (!native_qwen_layer_is_linear(layer)) {
                capture_by_layer[layer] =
                    queries + ((size_t)s * 24u + (size_t)layer) * 2048u;
            }
        }
        const int need_output = (s + 1 < lookahead);
        if (native_qwen_model_hidden_step(m, &rt, y, capture_by_layer,
                                          profile ? &lookahead_step_profile : NULL,
                                          need_output,
                                          hidden, err, errlen) != 0) {
            free(queries);
            free(max_per_step);
            free(importance);
            free(probs);
            free(smooth);
            native_qwen_runtime_free(&rt);
            return -1;
        }
        if (s + 1 < lookahead &&
            native_qwen_logits_argmax(m, hidden, &y, err, errlen) != 0) {
            free(queries);
            free(max_per_step);
            free(importance);
            free(probs);
            free(smooth);
            native_qwen_runtime_free(&rt);
            return -1;
        }
        if (debug_argmax && strcmp(debug_argmax, "1") == 0) {
            fprintf(stderr, " %d", y);
        }
    }
    if (debug_argmax && strcmp(debug_argmax, "1") == 0) {
        fprintf(stderr, "\n");
    }
    if (profile) prof_lookahead = native_now_ms() - prof_lookahead_start;

    double prof_importance_start = profile ? native_now_ms() : 0.0;
    const float scale = 1.0f / 16.0f;
    const int half = (pool_kernel - 1) / 2;
#if defined(DS4_DRAFTER_HAS_METAL)
    bool resident_caches_synced = native_metal_resident_cpu_kv_enabled();
#endif
    for (int layer = 0; layer < 24; layer++) {
        if (native_qwen_layer_is_linear(layer)) continue;
        native_full_attention_state *st = &rt.full[layer];
#if defined(DS4_DRAFTER_HAS_METAL)
        if (native_metal_enabled() &&
            native_metal_importance_enabled() &&
            st->len >= M &&
            st->metal_cache_valid &&
            st->metal_cache_cap == st->cap) {
            int metal_ok = 1;
            char metal_err[256] = {0};
            float q_batch_stack[4u * 2048u];
            float *q_batch = lookahead <= 4 ? q_batch_stack :
                malloc((size_t)lookahead * 2048u * sizeof(q_batch[0]));
            if (q_batch) {
                for (int s = 0; s < lookahead; s++) {
                    const float *q_layer =
                        queries + ((size_t)s * 24u + (size_t)layer) * 2048u;
                    memcpy(q_batch + (size_t)s * 2048u,
                           q_layer,
                           2048u * sizeof(q_batch[0]));
                }
                if (ds4_drafter_metal_attention_importance_batch(layer,
                                                                 q_batch,
                                                                 lookahead,
                                                                 M,
                                                                 pool_kernel,
                                                                 max_per_step,
                                                                 metal_err,
                                                                 sizeof(metal_err)) != 0) {
                    metal_ok = 0;
                }
                if (q_batch != q_batch_stack) free(q_batch);
            } else {
                for (int s = 0; s < lookahead; s++) {
                    const float *q_layer =
                        queries + ((size_t)s * 24u + (size_t)layer) * 2048u;
                    float *max_row = max_per_step + (size_t)s * (size_t)M;
                    if (ds4_drafter_metal_attention_importance(layer,
                                                               q_layer,
                                                               M,
                                                               pool_kernel,
                                                               max_row,
                                                               metal_err,
                                                               sizeof(metal_err)) != 0) {
                        metal_ok = 0;
                        break;
                    }
                }
            }
            if (metal_ok) continue;
            if (native_metal_strict()) {
                snprintf(err, errlen, "%s", metal_err[0] ? metal_err : "native Metal drafter attention importance failed");
                free(queries);
                free(max_per_step);
                free(importance);
                free(probs);
                free(smooth);
                native_qwen_runtime_free(&rt);
                return -1;
            }
            static int warned = 0;
            if (!warned) {
                fprintf(stderr,
                        "native drafter: Metal attention importance failed, falling back to CPU: %s\n",
                        metal_err[0] ? metal_err : "unknown error");
                warned = 1;
            }
            if (!resident_caches_synced) {
                if (native_qwen_sync_resident_full_caches(&rt, err, errlen) != 0) {
                    free(queries);
                    free(max_per_step);
                    free(importance);
                    free(probs);
                    free(smooth);
                    native_qwen_runtime_free(&rt);
                    return -1;
                }
                resident_caches_synced = true;
            }
        }
#endif
        if (st->len < M || !st->keys) continue;
        for (int s = 0; s < lookahead; s++) {
            const float *q_layer =
                queries + ((size_t)s * 24u + (size_t)layer) * 2048u;
            for (int qh = 0; qh < 8; qh++) {
                int kvh = qh / 4;
                const float *q = q_layer + qh * 256;
                double max_logit = -1.0e300;
                for (int t = 0; t < M; t++) {
                    const float *k = st->keys + ((size_t)t * 2u + (size_t)kvh) * 256u;
                    double dot = 0.0;
                    for (int d = 0; d < 256; d++) dot += (double)q[d] * (double)k[d];
                    double v = dot * (double)scale;
                    probs[t] = (float)v;
                    if (v > max_logit) max_logit = v;
                }
                double denom = 0.0;
                for (int t = 0; t < M; t++) {
                    double e = exp((double)probs[t] - max_logit);
                    probs[t] = (float)e;
                    denom += e;
                }
                if (denom <= 0.0 || !isfinite(denom)) continue;
                for (int t = 0; t < M; t++) probs[t] = (float)((double)probs[t] / denom);
                for (int t = 0; t < M; t++) {
                    double sum = 0.0;
                    for (int u = -half; u <= half; u++) {
                        int idx = t + u;
                        if (idx < 0) idx = 0;
                        if (idx >= M) idx = M - 1;
                        sum += probs[idx];
                    }
                    smooth[t] = (float)(sum / (double)pool_kernel);
                }
                float *max_row = max_per_step + (size_t)s * (size_t)M;
                for (int t = 0; t < M; t++) {
                    if (smooth[t] > max_row[t]) max_row[t] = smooth[t];
                }
            }
        }
    }
    for (int t = 0; t < M; t++) {
        double sum = 0.0;
        for (int s = 0; s < lookahead; s++) {
            sum += max_per_step[(size_t)s * (size_t)M + (size_t)t];
        }
        importance[t] = (float)(sum / (double)lookahead);
    }
    if (profile) prof_importance = native_now_ms() - prof_importance_start;

    double prof_select_start = profile ? native_now_ms() : 0.0;
    float *q_scores = malloc((size_t)M * sizeof(q_scores[0]));
    if (!q_scores) {
        free(queries);
        free(max_per_step);
        free(importance);
        free(probs);
        free(smooth);
        native_qwen_runtime_free(&rt);
        snprintf(err, errlen, "native drafter out of memory selecting scores");
        return -1;
    }
    double default_score_quantum = lookahead == 1 ? 2.0e-5 : 0.0;
    if (native_select_keep_scores(importance, M, block_size, keep_pct, sink, tail,
                                  default_score_quantum,
                                  q_scores, err, errlen) != 0) {
        free(q_scores);
        q_scores = NULL;
    }
    if (profile) prof_select = native_now_ms() - prof_select_start;
    if (profile) {
        double total = native_now_ms() - prof_start;
        fprintf(stderr,
                "native drafter profile: tokens=%d lookahead=%d prefill=%.3fms argmax=%.3fms lookahead=%.3fms importance=%.3fms select=%.3fms total=%.3fms\n",
                M, lookahead, prof_prefill, prof_argmax, prof_lookahead,
                prof_importance, prof_select, total);
        fprintf(stderr,
                "native drafter profile detail: prefill_tokens=%d embed=%.3fms linear=%.3fms/%d full=%.3fms/%d final_norm=%.3fms resident=%.3fms/%d lookahead_tokens=%d embed=%.3fms linear=%.3fms/%d full=%.3fms/%d final_norm=%.3fms resident=%.3fms/%d\n",
                prefill_step_profile.tokens,
                prefill_step_profile.embed_ms,
                prefill_step_profile.linear_ms,
                prefill_step_profile.linear_layers,
                prefill_step_profile.full_ms,
                prefill_step_profile.full_layers,
                prefill_step_profile.final_norm_ms,
                prefill_step_profile.resident_ms,
                prefill_step_profile.resident_tokens,
                lookahead_step_profile.tokens,
                lookahead_step_profile.embed_ms,
                lookahead_step_profile.linear_ms,
                lookahead_step_profile.linear_layers,
                lookahead_step_profile.full_ms,
                lookahead_step_profile.full_layers,
                lookahead_step_profile.final_norm_ms,
                lookahead_step_profile.resident_ms,
                lookahead_step_profile.resident_tokens);
    }
    free(queries);
    free(max_per_step);
    free(importance);
    free(probs);
    free(smooth);
    native_qwen_runtime_free(&rt);
    if (!q_scores) return -1;
    *q_scores_out = q_scores;
    return 0;
}

static int native_realign_scores_to_ds4(const native_token_vec *q_tokens,
                                        const float *q_scores,
                                        const uint32_t *ds4_spans,
                                        int ds4_len,
                                        float **scores_out,
                                        char *err,
                                        size_t errlen) {
    float *out = malloc((size_t)ds4_len * sizeof(out[0]));
    if (!out) {
        snprintf(err, errlen, "native drafter out of memory aligning scores");
        return -1;
    }
    double fallback = 0.0;
    for (int i = 0; i < q_tokens->len; i++) fallback += q_scores[i];
    fallback /= q_tokens->len > 0 ? (double)q_tokens->len : 1.0;
    int j = 0;
    for (int i = 0; i < ds4_len; i++) {
        uint32_t a = ds4_spans[(size_t)i * 2u + 0u];
        uint32_t b = ds4_spans[(size_t)i * 2u + 1u];
        if (a == b || q_tokens->len <= 0) {
            out[i] = (float)fallback;
            continue;
        }
        while (j < q_tokens->len &&
               q_tokens->offsets[(size_t)j * 2u + 1u] <= a) {
            j++;
        }
        int k = j > 0 ? j - 1 : 0;
        double num = 0.0;
        double den = 0.0;
        while (k < q_tokens->len &&
               q_tokens->offsets[(size_t)k * 2u + 0u] < b) {
            uint32_t s = q_tokens->offsets[(size_t)k * 2u + 0u];
            uint32_t e = q_tokens->offsets[(size_t)k * 2u + 1u];
            uint32_t lo = s > a ? s : a;
            uint32_t hi = e < b ? e : b;
            if (hi > lo) {
                double ov = (double)(hi - lo);
                num += (double)q_scores[k] * ov;
                den += ov;
            }
            k++;
        }
        out[i] = den > 0.0 ? (float)(num / den) : (float)fallback;
    }
    float smin = out[0];
    float smax = out[0];
    for (int i = 1; i < ds4_len; i++) {
        if (out[i] < smin) smin = out[i];
        if (out[i] > smax) smax = out[i];
    }
    float span = smax - smin;
    if (span > 0.0f) {
        for (int i = 0; i < ds4_len; i++) out[i] = (out[i] - smin) / span;
    } else {
        for (int i = 0; i < ds4_len; i++) out[i] = 0.0f;
    }
    *scores_out = out;
    return 0;
}

static int native_drafter_score_text(ds4_drafter *d,
                                     const ds4_drafter_options *opt,
                                     const char *text,
                                     size_t text_len,
                                     const uint32_t *ds4_spans,
                                     int ds4_len,
                                     float **scores_out,
                                     ds4_drafter_score_stats *stats_out,
                                     char *err,
                                     size_t errlen) {
    ds4_drafter_native_model *m = (ds4_drafter_native_model *)d->native;
    if (!m) {
        snprintf(err, errlen, "native drafter is not loaded");
        return -1;
    }
    double t0 = native_now_ms();
    native_token_vec q_tokens = {0};
    if (native_tokenizer_encode_text(&m->tokenizer, text, text_len,
                                     &q_tokens, err, errlen) != 0) {
        return -1;
    }
    double t1 = native_now_ms();
    float *q_scores = NULL;
    if (native_score_prompt_tokens(m, opt, &q_tokens, &q_scores, err, errlen) != 0) {
        native_token_vec_free(&q_tokens);
        return -1;
    }
    double t2 = native_now_ms();
    int rc = native_realign_scores_to_ds4(&q_tokens, q_scores, ds4_spans, ds4_len,
                                          scores_out, err, errlen);
    double t3 = native_now_ms();
    free(q_scores);
    native_token_vec_free(&q_tokens);
    if (rc != 0) return -1;
    if (stats_out) {
        stats_out->tokenize_ms = t1 - t0;
        stats_out->score_ms = t2 - t1;
        stats_out->align_ms = t3 - t2;
        stats_out->total_ms = t3 - t0;
    }
    return 0;
}

static int native_validate_dequant_sample(ds4_drafter_native_model *m,
                                          char *err,
                                          size_t errlen) {
    const char *name = "language_model.model.layers.3.self_attn.q_proj.weight";
    const ds4_drafter_tensor_meta *t = native_find_tensor(m, name);
    if (!t) {
        snprintf(err, errlen, "native drafter missing sample tensor %s", name);
        return -1;
    }
    int cols = (int)t->shape[1] * (32 / m->cfg.quant_bits);
    float *row = malloc((size_t)cols * sizeof(row[0]));
    if (!row) {
        snprintf(err, errlen, "native drafter out of memory dequantizing sample row");
        return -1;
    }
    int rc = native_dequant_affine_u32_row(m, name, 0, row, cols, err, errlen);
    if (rc == 0) {
        float checksum = 0.0f;
        for (int i = 0; i < cols; i += 67) checksum += row[i];
        if (!isfinite(checksum)) {
            snprintf(err, errlen, "native drafter dequantized sample produced non-finite values");
            rc = -1;
        }
    }
    free(row);
    if (rc != 0) return -1;

    float norm[1024];
    if (native_load_bf16_vector(m,
                                "language_model.model.layers.3.input_layernorm.weight",
                                norm,
                                1024,
                                err,
                                errlen) != 0) {
        return -1;
    }
    float emb[1024];
    if (native_embedding_lookup(m, 14556, emb, 1024, err, errlen) != 0) {
        return -1;
    }
    float emb_checksum = 0.0f;
    for (int i = 0; i < 1024; i += 79) emb_checksum += emb[i];
    if (!isfinite(emb_checksum)) {
        snprintf(err, errlen, "native drafter embedding sample produced non-finite values");
        return -1;
    }
    float x[1024];
    for (int i = 0; i < 1024; i++) {
        x[i] = emb[i] + ((float)((i * 17) % 29) - 14.0f) * 0.001f + norm[i] * 0.001f;
    }
    float xn[1024];
    if (native_rms_norm(m,
                        "language_model.model.layers.3.input_layernorm.weight",
                        x,
                        1024,
                        1.0e-6f,
                        xn,
                        err,
                        errlen) != 0) {
        return -1;
    }
    float y[4096];
    if (native_quant_affine_u32_matvec(m, name, xn, 1024, y, 4096, err, errlen) != 0) {
        return -1;
    }
    float qn[2048];
    float gate[2048];
    float kn[512];
    float vv[512];
    if (native_qwen_full_attention_project(m, 3, xn, qn, gate, kn, vv, err, errlen) != 0) {
        return -1;
    }
    native_qwen_full_attention_apply_rope(qn, kn, 17);
    float attn_out[1024];
    if (native_qwen_full_attention_output(m, 3, qn, gate, kn, vv, 1,
                                          attn_out, err, errlen) != 0) {
        return -1;
    }
    float post_h[1024];
    for (int i = 0; i < 1024; i++) post_h[i] = x[i] + attn_out[i];
    float post_norm[1024];
    if (native_rms_norm(m,
                        "language_model.model.layers.3.post_attention_layernorm.weight",
                        post_h,
                        1024,
                        1.0e-6f,
                        post_norm,
                        err,
                        errlen) != 0) {
        return -1;
    }
    float mlp_out[1024];
    if (native_qwen_mlp(m, 3, post_norm, mlp_out, err, errlen) != 0) {
        return -1;
    }
    float *linear_conv_state = calloc((size_t)3u * 6144u, sizeof(linear_conv_state[0]));
    float *linear_delta_state = calloc((size_t)16u * 128u * 128u,
                                       sizeof(linear_delta_state[0]));
    if (!linear_conv_state || !linear_delta_state) {
        free(linear_conv_state);
        free(linear_delta_state);
        snprintf(err, errlen, "native drafter out of memory validating linear attention");
        return -1;
    }
    float x0n[1024];
    float linear_out[1024];
    if (native_rms_norm(m,
                        "language_model.model.layers.0.input_layernorm.weight",
                        x,
                        1024,
                        1.0e-6f,
                        x0n,
                        err,
                        errlen) != 0 ||
        native_qwen_linear_attention_step(m, 0, x0n,
                                          linear_conv_state,
                                          linear_delta_state,
                                          linear_out,
                                          err,
                                          errlen) != 0) {
        free(linear_conv_state);
        free(linear_delta_state);
        return -1;
    }
    memset(linear_conv_state, 0, (size_t)3u * 6144u * sizeof(linear_conv_state[0]));
    memset(linear_delta_state, 0,
           (size_t)16u * 128u * 128u * sizeof(linear_delta_state[0]));
    float layer0_out[1024];
    if (native_qwen_decoder_layer_step(m, 0, x, 0,
                                       linear_conv_state,
                                       linear_delta_state,
                                       NULL,
                                       0,
                                       NULL,
                                       layer0_out,
                                       err,
                                       errlen) != 0) {
        free(linear_conv_state);
        free(linear_delta_state);
        return -1;
    }
    free(linear_conv_state);
    free(linear_delta_state);
    native_full_attention_state layer3_state = {0};
    float layer3_out[1024];
    if (native_qwen_decoder_layer_step(m, 3, x, 0,
                                       NULL,
                                       NULL,
                                       &layer3_state,
                                       0,
                                       NULL,
                                       layer3_out,
                                       err,
                                       errlen) != 0) {
        native_full_attention_state_free(&layer3_state);
        return -1;
    }
    native_full_attention_state_free(&layer3_state);
    float y_checksum = 0.0f;
    for (int i = 0; i < 4096; i += 113) y_checksum += y[i];
    for (int i = 0; i < 2048; i += 97) y_checksum += qn[i] + gate[i];
    for (int i = 0; i < 512; i += 41) y_checksum += kn[i] + vv[i];
    for (int i = 0; i < 1024; i += 53) y_checksum += attn_out[i];
    for (int i = 0; i < 1024; i += 47) y_checksum += mlp_out[i];
    for (int i = 0; i < 1024; i += 59) y_checksum += linear_out[i];
    for (int i = 0; i < 1024; i += 61) y_checksum += layer0_out[i] + layer3_out[i];
    if (!isfinite(y_checksum)) {
        snprintf(err, errlen, "native drafter attention projection sample produced non-finite values");
        return -1;
    }
    return 0;
}

static int native_read_config(const char *model_dir,
                              ds4_drafter_native_config *cfg,
                              char *err,
                              size_t errlen) {
    char path[4096];
    if (path_join(path, sizeof(path), model_dir, "config.json") != 0) {
        snprintf(err, errlen, "native drafter model path is too long");
        return -1;
    }
    size_t json_len = 0;
    char *json = read_file_text(path, &json_len);
    if (!json) {
        snprintf(err, errlen, "native drafter failed to read %s: %s",
                 path, strerror(errno));
        return -1;
    }
    (void)json_len;
    int ok = 0;
    ok |= strstr(json, "\"model_type\": \"qwen3_5\"") ? 0 : -1;
    ok |= json_int_after_key(json, "\"hidden_size\"", &cfg->hidden_size);
    ok |= json_int_after_key(json, "\"num_hidden_layers\"", &cfg->num_hidden_layers);
    ok |= json_int_after_key(json, "\"num_attention_heads\"", &cfg->num_attention_heads);
    ok |= json_int_after_key(json, "\"num_key_value_heads\"", &cfg->num_key_value_heads);
    ok |= json_int_after_key(json, "\"head_dim\"", &cfg->head_dim);
    ok |= json_int_after_key(json, "\"vocab_size\"", &cfg->vocab_size);
    ok |= json_int_after_key(json, "\"bits\"", &cfg->quant_bits);
    ok |= json_int_after_key(json, "\"group_size\"", &cfg->quant_group_size);
    ok |= count_layer_types(json, &cfg->full_attention_layers, &cfg->linear_attention_layers);
    free(json);
    if (ok != 0) {
        snprintf(err, errlen,
                 "native drafter supports Qwen3.5 config.json with text_config and layer_types; "
                 "failed to parse %s",
                 path);
        return -1;
    }
    if (cfg->hidden_size != 1024 ||
        cfg->num_hidden_layers != 24 ||
        cfg->num_attention_heads != 8 ||
        cfg->num_key_value_heads != 2 ||
        cfg->head_dim != 256 ||
        cfg->quant_bits != 4 ||
        cfg->quant_group_size != 64) {
        snprintf(err, errlen,
                 "native drafter recognized Qwen3.5 but not this shape "
                 "(hidden=%d layers=%d heads=%d kv=%d head_dim=%d qbits=%d group=%d)",
                 cfg->hidden_size,
                 cfg->num_hidden_layers,
                 cfg->num_attention_heads,
                 cfg->num_key_value_heads,
                 cfg->head_dim,
                 cfg->quant_bits,
                 cfg->quant_group_size);
        return -1;
    }
    return 0;
}

static int native_parse_tokenizer_added(native_tokenizer *tok,
                                        const char *p,
                                        char *err,
                                        size_t errlen) {
    if (json_expect_char(&p, '[') != 0) return -1;
    p = json_ws(p);
    while (*p && *p != ']') {
        if (json_expect_char(&p, '{') != 0) return -1;
        int id = -1;
        char *content = NULL;
        p = json_ws(p);
        while (*p && *p != '}') {
            char *key = json_parse_string_dup(&p);
            if (!key) {
                free(content);
                return -1;
            }
            if (json_expect_char(&p, ':') != 0) {
                free(key);
                free(content);
                return -1;
            }
            if (strcmp(key, "id") == 0) {
                int64_t v = 0;
                if (json_parse_i64_value(&p, &v) != 0 || v < 0 || v > INT_MAX) {
                    free(key);
                    free(content);
                    return -1;
                }
                id = (int)v;
            } else if (strcmp(key, "content") == 0) {
                free(content);
                content = json_parse_string_dup(&p);
                if (!content) {
                    free(key);
                    return -1;
                }
            } else if (json_skip_value(&p) != 0) {
                free(key);
                free(content);
                return -1;
            }
            free(key);
            p = json_ws(p);
            if (*p == ',') p++;
            p = json_ws(p);
        }
        if (*p != '}') {
            free(content);
            return -1;
        }
        p++;
        if (id >= 0 && content) {
            if (native_tokenizer_add_token(tok, content, id, true) != 0) {
                free(content);
                snprintf(err, errlen, "native drafter tokenizer out of memory");
                return -1;
            }
            content = NULL;
        }
        free(content);
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    return *p == ']' ? 0 : -1;
}

static int native_parse_tokenizer_vocab(native_tokenizer *tok,
                                        const char *p,
                                        int *count_out,
                                        char *err,
                                        size_t errlen) {
    if (json_expect_char(&p, '{') != 0) return -1;
    int count = 0;
    p = json_ws(p);
    while (*p && *p != '}') {
        char *token = json_parse_string_dup(&p);
        if (!token) return -1;
        if (json_expect_char(&p, ':') != 0) {
            free(token);
            return -1;
        }
        int64_t id64 = 0;
        if (json_parse_i64_value(&p, &id64) != 0 || id64 < 0 || id64 > INT_MAX) {
            free(token);
            return -1;
        }
        if (native_tokenizer_add_token(tok, token, (int)id64, false) != 0) {
            free(token);
            snprintf(err, errlen, "native drafter tokenizer out of memory");
            return -1;
        }
        count++;
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    if (*p != '}') return -1;
    *count_out = count;
    return 0;
}

static int native_parse_tokenizer_merges(native_tokenizer *tok,
                                         const char *p,
                                         int *count_out,
                                         char *err,
                                         size_t errlen) {
    if (json_expect_char(&p, '[') != 0) return -1;
    int rank = 0;
    p = json_ws(p);
    while (*p && *p != ']') {
        char *a = NULL;
        char *b = NULL;
        if (json_expect_char(&p, '[') != 0) return -1;
        a = json_parse_string_dup(&p);
        if (!a || json_expect_char(&p, ',') != 0) {
            free(a);
            return -1;
        }
        b = json_parse_string_dup(&p);
        if (!b || json_expect_char(&p, ']') != 0) {
            free(a);
            free(b);
            return -1;
        }
        size_t an = strlen(a);
        size_t bn = strlen(b);
        char *merge = malloc(an + 1u + bn + 1u);
        if (!merge) {
            free(a);
            free(b);
            snprintf(err, errlen, "native drafter tokenizer out of memory");
            return -1;
        }
        memcpy(merge, a, an);
        merge[an] = ' ';
        memcpy(merge + an + 1u, b, bn + 1u);
        if (tok->n_merge_storage == tok->cap_merge_storage) {
            int next = tok->cap_merge_storage ? tok->cap_merge_storage * 2 : 1024;
            char **v = realloc(tok->merge_storage, (size_t)next * sizeof(v[0]));
            if (!v) {
                free(merge);
                free(a);
                free(b);
                snprintf(err, errlen, "native drafter tokenizer out of memory");
                return -1;
            }
            tok->merge_storage = v;
            tok->cap_merge_storage = next;
        }
        tok->merge_storage[tok->n_merge_storage++] = merge;
        native_table_put(&tok->merge_rank,
                         (native_str){merge, (uint64_t)(an + 1u + bn)},
                         rank++);
        free(a);
        free(b);
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    if (*p != ']') return -1;
    *count_out = rank;
    return 0;
}

static int native_read_tokenizer(const char *model_dir,
                                 ds4_drafter_native_model *m,
                                 char *err,
                                 size_t errlen) {
    char path[4096];
    if (path_join(path, sizeof(path), model_dir, "tokenizer.json") != 0) {
        snprintf(err, errlen, "native drafter tokenizer path is too long");
        return -1;
    }
    size_t json_len = 0;
    char *json = read_file_text(path, &json_len);
    if (!json) {
        snprintf(err, errlen, "native drafter failed to read %s: %s",
                 path, strerror(errno));
        return -1;
    }
    (void)json_len;
    if (!strstr(json, "\"model\"") ||
        !strstr(json, "\"BPE\"") ||
        !strstr(json, "\"vocab\"") ||
        !strstr(json, "\"merges\"") ||
        !strstr(json, "\"ByteLevel\"")) {
        free(json);
        snprintf(err, errlen,
                 "native drafter tokenizer must be a Hugging Face BPE tokenizer with ByteLevel decoder");
        return -1;
    }

    native_tokenizer *tok = &m->tokenizer;
    if (native_table_init(&tok->token_to_id, 262144) != 0 ||
        native_table_init(&tok->merge_rank, 262144) != 0 ||
        native_tokenizer_ensure_slot(tok, 262143) != 0) {
        free(json);
        snprintf(err, errlen, "native drafter tokenizer out of memory");
        return -1;
    }

    const char *added_key = strstr(json, "\"added_tokens\"");
    if (added_key) {
        const char *p = strchr(added_key, ':');
        if (!p || native_parse_tokenizer_added(tok, p + 1, err, errlen) != 0) {
            free(json);
            snprintf(err, errlen, "native drafter failed to parse tokenizer added_tokens");
            return -1;
        }
    }

    int vocab_count = 0;
    int merge_count = 0;
    const char *vocab_key = strstr(json, "\"vocab\"");
    const char *merges_key = strstr(json, "\"merges\"");
    if (!vocab_key || !merges_key) {
        free(json);
        snprintf(err, errlen, "native drafter tokenizer missing vocab/merges");
        return -1;
    }
    const char *vocab_p = strchr(vocab_key, ':');
    const char *merges_p = strchr(merges_key, ':');
    if (!vocab_p ||
        native_parse_tokenizer_vocab(tok, vocab_p + 1, &vocab_count, err, errlen) != 0) {
        free(json);
        snprintf(err, errlen, "native drafter failed to parse tokenizer vocab");
        return -1;
    }
    if (!merges_p ||
        native_parse_tokenizer_merges(tok, merges_p + 1, &merge_count, err, errlen) != 0) {
        free(json);
        snprintf(err, errlen, "native drafter failed to parse tokenizer merges");
        return -1;
    }
    m->cfg.tokenizer_vocab_size = vocab_count;
    m->cfg.tokenizer_merges = merge_count;
    native_token_vec sample = {0};
    if (native_tokenizer_encode_text(tok, "hello world", strlen("hello world"),
                                     &sample, err, errlen) != 0) {
        native_token_vec_free(&sample);
        free(json);
        return -1;
    }
    if (sample.len != 2 || sample.ids[0] != 14556 || sample.ids[1] != 1814) {
        native_token_vec_free(&sample);
        free(json);
        snprintf(err, errlen, "native drafter tokenizer sample mismatch");
        return -1;
    }
    native_token_vec_free(&sample);
    free(json);
    return 0;
}

static int native_read_safetensors_shape(const char *model_dir,
                                         ds4_drafter_native_model *m,
                                         char *err,
                                         size_t errlen) {
    char path[4096];
    if (path_join(path, sizeof(path), model_dir, "model.safetensors") != 0) {
        snprintf(err, errlen, "native drafter safetensors path is too long");
        return -1;
    }
    size_t header_len = 0;
    char *header = read_safetensors_header(path, &header_len);
    if (!header) {
        snprintf(err, errlen, "native drafter failed to read safetensors header from %s: %s",
                 path, strerror(errno));
        return -1;
    }
    m->safetensors_path = malloc(strlen(path) + 1u);
    if (!m->safetensors_path) {
        free(header);
        snprintf(err, errlen, "native drafter out of memory");
        return -1;
    }
    strcpy(m->safetensors_path, path);
    m->safetensors_data_base = 8u + (uint64_t)header_len;

    const char *p = header;
    if (json_expect_char(&p, '{') != 0) {
        free(header);
        snprintf(err, errlen, "native drafter bad safetensors header object");
        return -1;
    }
    p = json_ws(p);
    while (*p && *p != '}') {
        char *name = json_parse_string_dup(&p);
        if (!name) {
            free(header);
            snprintf(err, errlen, "native drafter bad safetensors tensor name");
            return -1;
        }
        if (json_expect_char(&p, ':') != 0) {
            free(name);
            free(header);
            snprintf(err, errlen, "native drafter bad safetensors tensor separator");
            return -1;
        }
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            if (json_skip_value(&p) != 0) {
                free(header);
                snprintf(err, errlen, "native drafter bad safetensors metadata");
                return -1;
            }
        } else {
            ds4_drafter_tensor_meta meta = {0};
            meta.name = name;
            if (parse_safetensors_tensor_object(&p, &meta) != 0) {
                free(meta.name);
                free(header);
                snprintf(err, errlen, "native drafter bad safetensors tensor entry");
                return -1;
            }
            uint64_t elem_size = native_dtype_size(meta.dtype);
            uint64_t expected = native_tensor_elems(&meta) * elem_size;
            uint64_t actual = meta.end - meta.begin;
            if (elem_size == 0 || expected != actual) {
                snprintf(err, errlen,
                         "native drafter bad tensor byte size for %s dtype=%s expected=%llu actual=%llu",
                         meta.name,
                         meta.dtype,
                         (unsigned long long)expected,
                         (unsigned long long)actual);
                free(meta.name);
                free(header);
                return -1;
            }
    if (native_tensor_push(m, meta) != 0) {
                free(meta.name);
                free(header);
                snprintf(err, errlen, "native drafter out of memory parsing safetensors");
                return -1;
            }
        }
        p = json_ws(p);
        if (*p == ',') p++;
        p = json_ws(p);
    }
    if (*p != '}') {
        free(header);
        snprintf(err, errlen, "native drafter truncated safetensors header");
        return -1;
    }
    free(header);
    if (native_build_tensor_index(m, err, errlen) != 0) return -1;

    const char *required[] = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.embed_tokens.scales",
        "language_model.model.embed_tokens.biases",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.k_proj.weight",
        "language_model.model.layers.3.self_attn.v_proj.weight",
        "language_model.model.layers.23.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        if (!native_find_tensor(m, required[i])) {
            snprintf(err, errlen, "native drafter safetensors missing required tensor %s",
                     required[i]);
            return -1;
        }
    }
    m->cfg.safetensors_tensors = m->n_tensors;
    if (native_map_safetensors(m, err, errlen) != 0) return -1;
    if (native_validate_dequant_sample(m, err, errlen) != 0) return -1;
    return 0;
}

static int native_start(ds4_drafter *d,
                        const ds4_drafter_options *opt,
                        char *err,
                        size_t errlen) {
    if (d->native) return 0;
    ds4_drafter_native_model *m = calloc(1, sizeof(*m));
    if (!m) {
        snprintf(err, errlen, "native drafter out of memory");
        return -1;
    }
    m->safetensors_fd = -1;
    if (native_read_config(opt->model, &m->cfg, err, errlen) != 0) {
        native_model_free(m);
        return -1;
    }
    if (native_read_tokenizer(opt->model, m, err, errlen) != 0 ||
        native_read_safetensors_shape(opt->model, m, err, errlen) != 0) {
        native_model_free(m);
        return -1;
    }
#if defined(DS4_DRAFTER_HAS_METAL)
    if (native_qwen_prepare_all_metal_caches(m, err, errlen) != 0) {
        native_model_free(m);
        return -1;
    }
    ds4_drafter_metal_affine_job logits_job;
    if (native_metal_prepare_affine_job(m,
                                        "language_model.model.embed_tokens.weight",
                                        1024, m->cfg.vocab_size,
                                        &logits_job, err, errlen) != 0) {
        native_model_free(m);
        return -1;
    }
    float logits_warmup[1024] = {0};
    int warmup_token = 0;
    if (ds4_drafter_metal_logits_argmax_u32(
            logits_job.w_data, logits_job.w_bytes,
            logits_job.scales_data, logits_job.scales_bytes,
            logits_job.biases_data, logits_job.biases_bytes,
            logits_warmup,
            logits_job.rows,
            logits_job.packed_cols,
            1024,
            logits_job.groups,
            m->cfg.quant_bits,
            m->cfg.quant_group_size,
            &warmup_token,
            err,
            errlen) != 0) {
        native_model_free(m);
        return -1;
    }
#endif
    d->native = m;
    d->active_backend = DS4_DRAFTER_BACKEND_NATIVE;
    d->ready_detail[0] = '\0';
    if (err && errlen > 0) {
        snprintf(err, errlen,
                 "native Qwen drafter ready "
                 "(layers=%d full_attention=%d linear_attention=%d tensors=%d metal=%s)",
                 m->cfg.num_hidden_layers,
                 m->cfg.full_attention_layers,
                 m->cfg.linear_attention_layers,
                 m->cfg.safetensors_tensors,
#if defined(DS4_DRAFTER_HAS_METAL)
                 native_metal_enabled() ? "on" : "off"
#else
                 "unavailable"
#endif
        );
        snprintf(d->ready_detail, sizeof(d->ready_detail), "%s", err);
    }
    return 0;
}

int ds4_drafter_start(ds4_drafter *d,
                      const ds4_drafter_options *opt,
                      char *err,
                      size_t errlen) {
    if (!d || !opt || !opt->model) {
        snprintf(err, errlen, "live drafter is not configured");
        return -1;
    }
    if (opt->backend == DS4_DRAFTER_BACKEND_NATIVE) {
        return native_start(d, opt, err, errlen);
    }
    if (d->active_backend == DS4_DRAFTER_BACKEND_NATIVE || d->native) {
        ds4_drafter_stop(d);
    }
    if (d->pid > 0) return 0;
    d->active_backend = DS4_DRAFTER_BACKEND_PYTHON;

    const char *python = opt->python ? opt->python : "python3";
    const char *script = opt->script ? opt->script : "speed-bench/ds4_live_drafter.py";
    const char *tokenizer = opt->tokenizer ? opt->tokenizer : "./gguf/dsv4-tokenizer";
    const char *process_name = opt->process_name ? opt->process_name : "ds4";
    char lookahead_arg[32];
    char pool_arg[32];
    char keep_arg[32];
    char block_arg[32];
    char sink_arg[32];
    char tail_arg[32];
    snprintf(lookahead_arg, sizeof(lookahead_arg), "%d", opt->score_lookahead);
    snprintf(pool_arg, sizeof(pool_arg), "%d", opt->score_pool_kernel);
    snprintf(keep_arg, sizeof(keep_arg), "%.6f",
             opt->keep_pct > 0.0f ? opt->keep_pct : 0.3f);
    snprintf(block_arg, sizeof(block_arg), "%d",
             opt->chunk > 0 ? opt->chunk : 32);
    snprintf(sink_arg, sizeof(sink_arg), "%d",
             opt->sink >= 0 ? opt->sink : 16);
    snprintf(tail_arg, sizeof(tail_arg), "%d",
             opt->tail >= 0 ? opt->tail : 256);

    int to_child[2];
    int from_child[2];
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        snprintf(err, errlen, "live drafter pipe failed: %s", strerror(errno));
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, errlen, "live drafter fork failed: %s", strerror(errno));
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        execlp(python, python, "-u", script,
               "--scorer-model", opt->model,
               "--dsv4-tokenizer", tokenizer,
               "--n-lookahead", lookahead_arg,
               "--pool-kernel", pool_arg,
               "--keep-fraction", keep_arg,
               "--block-size", block_arg,
               "--sink-size", sink_arg,
               "--tail-keep", tail_arg,
               (char *)NULL);
        fprintf(stderr, "%s: exec live drafter failed: %s\n",
                process_name, strerror(errno));
        _exit(127);
    }

    close(to_child[0]);
    close(from_child[1]);
    FILE *out = fdopen(from_child[0], "rb");
    if (!out) {
        snprintf(err, errlen, "live drafter fdopen failed: %s", strerror(errno));
        close(to_child[1]);
        close(from_child[0]);
        kill(pid, SIGTERM);
        waitpid(pid, NULL, 0);
        return -1;
    }
    d->pid = pid;
    d->in_fd = to_child[1];
    d->out_fp = out;
    d->ready_detail[0] = '\0';

    char ready[128];
    if (!fgets(ready, sizeof(ready), d->out_fp)) {
        snprintf(err, errlen, "live drafter closed before ready");
        ds4_drafter_stop(d);
        return -1;
    }
    if (strcmp(ready, "READY\n") != 0) {
        snprintf(err, errlen, "live drafter startup error: %.96s", ready);
        ds4_drafter_stop(d);
        return -1;
    }
    snprintf(d->ready_detail, sizeof(d->ready_detail),
             "Python drafter ready pid=%ld", (long)d->pid);
    return 0;
}

int ds4_drafter_score(ds4_drafter *d,
                      const ds4_drafter_options *opt,
                      ds4_engine *engine,
                      const ds4_tokens *prompt,
                      float **scores_out,
                      int *scores_len_out,
                      ds4_drafter_score_stats *stats_out,
                      char *err,
                      size_t errlen) {
    if (scores_out) *scores_out = NULL;
    if (scores_len_out) *scores_len_out = 0;
    if (stats_out) memset(stats_out, 0, sizeof(*stats_out));
    if (!scores_out || !scores_len_out) {
        snprintf(err, errlen, "live drafter score output is not configured");
        return -1;
    }
    if (ds4_drafter_start(d, opt, err, errlen) != 0) return -1;

    size_t text_len = 0;
    uint32_t *spans = NULL;
    char *text = render_tokens_text_with_spans(engine, prompt, &text_len, &spans);
    if (!text) {
        snprintf(err, errlen, "live drafter failed to render transcript");
        return -1;
    }

    if (d->active_backend == DS4_DRAFTER_BACKEND_NATIVE) {
        ds4_drafter_score_stats stats = {0};
        int rc = native_drafter_score_text(d, opt, text, text_len, spans, prompt->len,
                                           scores_out, &stats, err, errlen);
        free(text);
        free(spans);
        if (rc != 0) return -1;
        *scores_len_out = prompt->len;
        if (stats_out) *stats_out = stats;
        return 0;
    }

    char header[128];
    int header_len = snprintf(header, sizeof(header), "SCORE2 %d %zu\n",
                              prompt->len, text_len);
    if (header_len <= 0 || (size_t)header_len >= sizeof(header) ||
        write_all_fd(d->in_fd, header, (size_t)header_len) != 0 ||
        write_all_fd(d->in_fd, text, text_len) != 0 ||
        write_all_fd(d->in_fd, spans,
                     (size_t)prompt->len * 2u * sizeof(spans[0])) != 0) {
        free(text);
        free(spans);
        snprintf(err, errlen, "live drafter request write failed: %s", strerror(errno));
        return -1;
    }
    free(text);
    free(spans);

    char line[256];
    if (!fgets(line, sizeof(line), d->out_fp)) {
        snprintf(err, errlen, "live drafter closed before response");
        return -1;
    }
    int n = 0;
    ds4_drafter_score_stats stats = {0};
    if (sscanf(line, "OK %d %lf %lf %lf %lf",
               &n,
               &stats.total_ms,
               &stats.tokenize_ms,
               &stats.score_ms,
               &stats.align_ms) != 5) {
        snprintf(err, errlen, "live drafter error: %.220s", line);
        return -1;
    }
    if (n != prompt->len) {
        snprintf(err, errlen, "live drafter returned %d scores for %d tokens",
                 n, prompt->len);
        return -1;
    }
    float *scores = malloc((size_t)n * sizeof(scores[0]));
    if (!scores) {
        snprintf(err, errlen, "out of memory reading live drafter scores");
        return -1;
    }
    size_t got = fread(scores, sizeof(scores[0]), (size_t)n, d->out_fp);
    if (got != (size_t)n) {
        free(scores);
        snprintf(err, errlen, "live drafter score payload truncated (%zu/%d)", got, n);
        return -1;
    }
    *scores_out = scores;
    *scores_len_out = n;
    if (stats_out) *stats_out = stats;
    return 0;
}
