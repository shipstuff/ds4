#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    bool warm_weights;
    bool quality;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Speculative prefill (SpecPrefill, dense-decode variant).
 *
 * Experimental, opt-in prompt preprocessor.  Given a prompt and a per-token
 * importance score vector, picks the most important fixed-size chunks of the
 * history, keeps the last `tail_size` tokens unconditionally, concatenates
 * them in original order, and writes the compressed list to `out`.  The
 * caller then feeds the compressed prompt to the normal prefill path
 * (ds4_session_sync, ds4_engine_generate_argmax); the target sees the
 * shortened prompt at contiguous positions, so the regular RoPE / KV cache
 * path is unchanged and decode runs at baseline speed.
 *
 * Scoring can come from three places:
 *   1. Caller-provided `scores` vector (use when an external draft like
 *      anemll-project's score_tokens(), or the bundled
 *      misc/tools/dump_specprefill_scores.py, has already produced scores).
 *   2. ds4's own native scorer when `self_score = true` (no `scores` vector
 *      needed).  Calls ds4_engine_score_prompt() internally — see that
 *      function for the attention-aggregation algorithm.  Available on all
 *      backends: the CPU path runs the projections via ds4's CPU reference
 *      helpers, the Metal/CUDA path runs them via the same ds4_gpu_*
 *      primitives the real prefill graph uses (Q and K are read back to
 *      host for the per-layer score aggregation).
 *   3. Recency heuristic fallback when neither is provided.
 *
 * Background: this is a port of the dense-decode SpecPrefill variant from
 * carl/anemll-project (vllm-mlx lineage).  The Python harness uses a small
 * draft model (Qwen-Next 0.8B) to score importance for a much larger target.
 * ds4 is DSV4-Flash-specific and has no second-model runtime; ports therefore
 * source scores externally (anemll harness writes them to a file) or fall
 * back to a recency+stride heuristic.  A real on-DSV4 self-score path needs
 * shallow-prefill Metal kernels and is intentionally out of scope here.
 *
 * AGENT.md note: this is a diagnostic / experimental switch, not a permanent
 * release-path semantic variant.  It exists to validate the dense-prefill
 * compression flow end-to-end against the anemll reference harness, and to
 * let local agent sessions opt into smaller prefills on long prompts. */
typedef struct {
    /* Fraction of history chunks to keep, in (0, 1].  1.0 disables selection
     * (returns prompt unchanged).  Default 0.3. */
    float keep_pct;

    /* Number of leading prompt tokens to always include in the compressed
     * output regardless of score.  These "sink" tokens consistently receive
     * disproportionate attention in autoregressive transformers; keeping
     * them stabilises the compressed prefix and is additive to the
     * chunk-selected middle section (never replaces a selected chunk).
     * Default 16.  Clamped so sink_size + tail_size <= prompt->len. */
    int sink_size;

    /* Number of trailing prompt tokens to always include in the compressed
     * output regardless of score.  Preserves chat-template tail (im_end /
     * assistant prefix etc).  Default 256.  Clamped to prompt length. */
    int tail_size;

    /* Tokens per chunk for chunk-wise selection.  Default 32. */
    int chunk_size;

    /* Optional per-prompt-token importance scores.  When non-NULL, must have
     * exactly `prompt->len` entries.  Higher = more important. */
    const float *scores;
    int scores_len;

    /* When true and `scores` is NULL, ds4_spec_prefill_compress calls
     * ds4_engine_score_prompt() to compute attention-based scores natively.
     * Requires CPU backend in v1 (the engine returns an error otherwise).
     * When both `self_score = false` and `scores = NULL`, the function
     * falls back to a recency-only heuristic. */
    bool self_score;

    /* Scoring knobs, only consulted when self_score = true.  Zero values
     * mean "use defaults": score_layers=2, score_lookahead=4, pool_kernel=13. */
    int  score_layers;
    int  score_lookahead;
    int  score_pool_kernel;
} ds4_spec_prefill_options;

/* Diagnostic / test helper.  Runs both the CPU and the graph-backend
 * (Metal/CUDA) scorers over the same model + prompt and reports the
 * max-abs-diff between their outputs.  Only the active engine backend's
 * graph runtime is required to be initialized -- the CPU scorer is
 * always available regardless of which backend the engine was opened on
 * because it reads model->weights directly.
 *
 * Useful for validating the Metal composition matches the CPU reference
 * (the underlying ds4_gpu_* primitives have their own unit tests; this
 * checks that the scorer puts them together in the right order with the
 * right shapes/offsets).  Each output buffer must have at least
 * `prompt->len` floats.  Returns 0 on success. */
int ds4_engine_score_prompt_validate(
        ds4_engine *e,
        const ds4_tokens *prompt,
        int score_layers,
        int score_lookahead,
        int pool_kernel,
        float *scores_cpu_out,
        float *scores_graph_out,
        float *max_abs_diff_out,
        char *err, size_t errlen);

/* Per-prompt-token attention-aggregation scoring against the loaded DSV4
 * model itself, on the CPU reference backend.  Algorithm mirrors the
 * SpecPrefill paper (and the mlx-lm port at the Python layer):
 *
 *   for layer in first `score_layers` attention layers:
 *       compute Q for every prompt position via the LoRA-projected MLA Q
 *           path (q_a -> q_a_norm -> q_b -> head_rms_norm -> RoPE)
 *       compute K_latent for every prompt position via the MLA K path
 *           (attn_kv -> kv_a_norm -> RoPE)
 *       use the last `score_lookahead` prompt rows as lookahead queries
 *           and compute softmax(Q[m] dot K[<=m]) / sqrt(head_dim)
 *           per (layer, head)
 *   smooth each row with a centered avg-pool of width pool_kernel
 *   take max across (layer, head)
 *   take mean across lookahead rows
 *
 * The CPU scorer reuses the existing CPU helpers (embed_token_f16,
 * hc_pre_norm_batch, matmul_q8_0_batch, head_rms_norm_inplace,
 * rope_tail_layer_inplace).  The Metal/CUDA scorer reuses the equivalent
 * ds4_gpu_* primitives the real prefill graph uses (embed_tokens,
 * matmul_q8_0_tensor, dsv4_qkv_rms_norm_rows, head_rms_norm,
 * rope_tail_tensor), so the math stays in lockstep with prefill on every
 * backend.  Both scorers deliberately skip the indexer, the KV
 * compressor, FFN, MoE routing, and any layer past `score_layers` --
 * scoring is a cheap diagnostic, not a full prefill.
 *
 * `scores_out` must point to at least `prompt->len` floats.  Returns 0
 * on success.  Backend dispatch is automatic: CPU engines run the CPU
 * path, Metal/CUDA engines run the GPU path. */
int ds4_engine_score_prompt(
        ds4_engine *e,
        const ds4_tokens *prompt,
        int score_layers,
        int score_lookahead,
        int pool_kernel,
        float *scores_out,
        char *err, size_t errlen);

/* Compute a sensible default-options block.  Caller can then override any
 * field before passing it to ds4_spec_prefill_compress. */
ds4_spec_prefill_options ds4_spec_prefill_options_default(void);

/* Compress a prompt down to (selected history chunks ∪ recent tail).  Token
 * order is preserved (sorted by original index).  Caller owns `out` and must
 * ds4_tokens_free it.  Returns 0 on success. */
int ds4_spec_prefill_compress(
        ds4_engine *e,
        const ds4_tokens *prompt,
        const ds4_spec_prefill_options *opt,
        ds4_tokens *out,
        char *err, size_t errlen);

/* Convenience: parse a whitespace-separated float-per-line scores file into a
 * heap-allocated float buffer.  `*out_scores` is malloc'd and the caller
 * must free it.  Returns 0 on success.  Errors include unreadable file,
 * non-numeric content, and short/long token counts when `expected_len > 0`. */
int ds4_spec_prefill_load_scores_file(
        const char *path,
        int expected_len,
        float **out_scores,
        int *out_len,
        char *err, size_t errlen);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
