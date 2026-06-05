#ifndef DS4_DRAFTER_METAL_H
#define DS4_DRAFTER_METAL_H

#include <stdint.h>
#include <stddef.h>

int ds4_drafter_metal_available(void);
void ds4_drafter_metal_shutdown(void);

typedef struct {
        const void *w_data;
        uint64_t w_bytes;
        const void *scales_data;
        uint64_t scales_bytes;
        const void *biases_data;
        uint64_t biases_bytes;
        int rows;
        int packed_cols;
        int groups;
        float *out;
} ds4_drafter_metal_affine_job;

typedef struct {
        int is_linear;
        const void *input_norm_data;
        uint64_t input_norm_bytes;
        const void *post_norm_data;
        uint64_t post_norm_bytes;
        const void *q_norm_data;
        uint64_t q_norm_bytes;
        const void *k_norm_data;
        uint64_t k_norm_bytes;
        const void *conv_data;
        uint64_t conv_bytes;
        const void *linear_norm_data;
        uint64_t linear_norm_bytes;
        const void *a_log_data;
        uint64_t a_log_bytes;
        const void *dt_bias_data;
        uint64_t dt_bias_bytes;
        ds4_drafter_metal_affine_job q_job;
        ds4_drafter_metal_affine_job k_job;
        ds4_drafter_metal_affine_job v_job;
        ds4_drafter_metal_affine_job o_job;
        ds4_drafter_metal_affine_job qkv_job;
        ds4_drafter_metal_affine_job z_job;
        ds4_drafter_metal_affine_job b_job;
        ds4_drafter_metal_affine_job a_job;
        ds4_drafter_metal_affine_job linear_out_job;
        ds4_drafter_metal_affine_job mlp_gate_job;
        ds4_drafter_metal_affine_job mlp_up_job;
        ds4_drafter_metal_affine_job mlp_down_job;
} ds4_drafter_metal_decoder_layer_job;

int ds4_drafter_metal_affine_u32_matvec(
        const void *w_data,
        uint64_t w_bytes,
        const void *scales_data,
        uint64_t scales_bytes,
        const void *biases_data,
        uint64_t biases_bytes,
        const float *x,
        int rows,
        int packed_cols,
        int cols,
        int groups,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_affine_u32_matvec_many(
        const ds4_drafter_metal_affine_job *jobs,
        int n_jobs,
        const float *x,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen);

int ds4_drafter_metal_affine_u32_matmat(
        const ds4_drafter_metal_affine_job *job,
        const float *x,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_prepare_affine_u32(
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen);

int ds4_drafter_metal_logits_argmax_u32(
        const void *w_data,
        uint64_t w_bytes,
        const void *scales_data,
        uint64_t scales_bytes,
        const void *biases_data,
        uint64_t biases_bytes,
        const float *x,
        int rows,
        int packed_cols,
        int cols,
        int groups,
        int bits,
        int group_size,
        int *token_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_dequant_u32_row(
        const void *w_data,
        uint64_t w_bytes,
        const void *scales_data,
        uint64_t scales_bytes,
        const void *biases_data,
        uint64_t biases_bytes,
        int row,
        int rows,
        int packed_cols,
        int cols,
        int groups,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_rms_norm_bf16(
        const void *weight_data,
        uint64_t weight_bytes,
        const float *x,
        int len,
        float eps,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_rms_norm_bf16_mat(
        const void *weight_data,
        uint64_t weight_bytes,
        const float *x,
        int n_vec,
        int len,
        float eps,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_mlp_u32(
        const ds4_drafter_metal_affine_job *gate,
        const ds4_drafter_metal_affine_job *up,
        const ds4_drafter_metal_affine_job *down,
        const float *x,
        int in_cols,
        int hidden_cols,
        int out_cols,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_mlp_u32_mat(
        const ds4_drafter_metal_affine_job *gate,
        const ds4_drafter_metal_affine_job *up,
        const ds4_drafter_metal_affine_job *down,
        const float *x,
        int n_vec,
        int in_cols,
        int hidden_cols,
        int out_cols,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_linear_attention_reset(int cache_id,
                                             char *err,
                                             size_t errlen);

int ds4_drafter_metal_linear_attention_u32(
        const ds4_drafter_metal_affine_job *qkv,
        const ds4_drafter_metal_affine_job *z,
        const ds4_drafter_metal_affine_job *b,
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *out_proj,
        const void *conv_data,
        uint64_t conv_bytes,
        const void *norm_data,
        uint64_t norm_bytes,
        const void *a_log_data,
        uint64_t a_log_bytes,
        const void *dt_bias_data,
        uint64_t dt_bias_bytes,
        const float *x,
        int cache_id,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_linear_project_conv_u32_mat(
        const ds4_drafter_metal_affine_job *qkv,
        const ds4_drafter_metal_affine_job *z,
        const ds4_drafter_metal_affine_job *b,
        const ds4_drafter_metal_affine_job *a,
        const void *conv_data,
        uint64_t conv_bytes,
        const float *x,
        int n_vec,
        int bits,
	        int group_size,
	        float *conv_out,
	        float *qkv_out,
	        float *z_out,
	        float *b_out,
	        float *a_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_linear_scan_u32_mat(
        const void *norm_data,
        uint64_t norm_bytes,
        const void *a_log_data,
        uint64_t a_log_bytes,
        const void *dt_bias_data,
        uint64_t dt_bias_bytes,
        const float *conv_out,
	        const float *z,
	        const float *b,
	        const float *a,
	        const float *qkv,
	        int n_vec,
	        int cache_id,
	        float *gated_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_linear_decoder_layer_u32_mat(
        const void *input_norm_data,
        uint64_t input_norm_bytes,
        const void *post_norm_data,
        uint64_t post_norm_bytes,
        const ds4_drafter_metal_affine_job *qkv,
        const ds4_drafter_metal_affine_job *z,
        const ds4_drafter_metal_affine_job *b,
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *linear_out,
        const void *conv_data,
        uint64_t conv_bytes,
        const void *linear_norm_data,
        uint64_t linear_norm_bytes,
        const void *a_log_data,
        uint64_t a_log_bytes,
        const void *dt_bias_data,
        uint64_t dt_bias_bytes,
        const ds4_drafter_metal_affine_job *mlp_gate,
        const ds4_drafter_metal_affine_job *mlp_up,
        const ds4_drafter_metal_affine_job *mlp_down,
        const float *x,
        int n_vec,
        int cache_id,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_decoder_layer_u32_mat(
        const void *input_norm_data,
        uint64_t input_norm_bytes,
        const void *post_norm_data,
        uint64_t post_norm_bytes,
        const ds4_drafter_metal_affine_job *q_proj,
        const ds4_drafter_metal_affine_job *k_proj,
        const ds4_drafter_metal_affine_job *v_proj,
        const ds4_drafter_metal_affine_job *out_proj,
        const void *q_norm_data,
        uint64_t q_norm_bytes,
        const void *k_norm_data,
        uint64_t k_norm_bytes,
        const ds4_drafter_metal_affine_job *mlp_gate,
        const ds4_drafter_metal_affine_job *mlp_up,
        const ds4_drafter_metal_affine_job *mlp_down,
        const float *x,
        int n_vec,
        int bits,
        int group_size,
        float *out,
        float *keys_rope_out,
        float *values_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_linear_decoder_layer_u32(
        const void *input_norm_data,
        uint64_t input_norm_bytes,
        const void *post_norm_data,
        uint64_t post_norm_bytes,
        const ds4_drafter_metal_affine_job *qkv,
        const ds4_drafter_metal_affine_job *z,
        const ds4_drafter_metal_affine_job *b,
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *linear_out,
        const void *conv_data,
        uint64_t conv_bytes,
        const void *linear_norm_data,
        uint64_t linear_norm_bytes,
        const void *a_log_data,
        uint64_t a_log_bytes,
        const void *dt_bias_data,
        uint64_t dt_bias_bytes,
        const ds4_drafter_metal_affine_job *mlp_gate,
        const ds4_drafter_metal_affine_job *mlp_up,
        const ds4_drafter_metal_affine_job *mlp_down,
        const float *x,
        int cache_id,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_attention_output_u32(
        const ds4_drafter_metal_affine_job *out_proj,
        const float *queries_rope,
        const float *gate,
        const float *keys_rope_cache,
        const float *values_cache,
        int n_ctx,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_attention_output_u32_mat(
        const ds4_drafter_metal_affine_job *out_proj,
        const float *queries_rope,
        const float *gate,
        const float *keys_rope_cache,
        const float *values_cache,
        int n_ctx,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_attention_project_u32_mat(
        const ds4_drafter_metal_affine_job *q_proj,
        const ds4_drafter_metal_affine_job *k_proj,
        const ds4_drafter_metal_affine_job *v_proj,
        const void *q_norm_data,
        uint64_t q_norm_bytes,
        const void *k_norm_data,
        uint64_t k_norm_bytes,
        const float *x,
        int n_vec,
        int base_position,
        int bits,
        int group_size,
        float *queries_rope_out,
        float *gate_out,
        float *keys_rope_out,
        float *values_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_attention_step_u32(
        const ds4_drafter_metal_affine_job *out_proj,
        const float *queries_rope,
        const float *gate,
        const float *key_rope,
        const float *value,
        int cache_id,
        int cache_index,
        int n_ctx,
        int cache_capacity,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_attention_layer_u32(
        const ds4_drafter_metal_affine_job *q_proj,
        const ds4_drafter_metal_affine_job *k_proj,
        const ds4_drafter_metal_affine_job *v_proj,
        const ds4_drafter_metal_affine_job *out_proj,
        const void *q_norm_data,
        uint64_t q_norm_bytes,
        const void *k_norm_data,
        uint64_t k_norm_bytes,
        const float *x,
        int position,
        int cache_id,
        int cache_index,
        int n_ctx,
        int cache_capacity,
        int bits,
        int group_size,
        float *query_capture,
        float *key_out,
        float *value_out,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_full_decoder_layer_u32(
        const void *input_norm_data,
        uint64_t input_norm_bytes,
        const void *post_norm_data,
        uint64_t post_norm_bytes,
        const ds4_drafter_metal_affine_job *q_proj,
        const ds4_drafter_metal_affine_job *k_proj,
        const ds4_drafter_metal_affine_job *v_proj,
        const ds4_drafter_metal_affine_job *out_proj,
        const void *q_norm_data,
        uint64_t q_norm_bytes,
        const void *k_norm_data,
        uint64_t k_norm_bytes,
        const ds4_drafter_metal_affine_job *mlp_gate,
        const ds4_drafter_metal_affine_job *mlp_up,
        const ds4_drafter_metal_affine_job *mlp_down,
        const float *x,
        int position,
        int cache_id,
        int cache_index,
        int n_ctx,
        int cache_capacity,
        int bits,
        int group_size,
        float *query_capture,
        float *key_out,
        float *value_out,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_qwen_hidden_step_u32(
        const ds4_drafter_metal_decoder_layer_job *layers,
        int n_layers,
        const float *x,
        int position,
        int full_cache_index,
        int full_n_ctx,
        int full_cache_capacity,
        int bits,
        int group_size,
        float **query_capture_by_layer,
        float **key_out_by_layer,
        float **value_out_by_layer,
        int need_output,
        const void *final_norm_data,
        uint64_t final_norm_bytes,
        float *out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_qwen_batch_prefill_u32(
        const ds4_drafter_metal_decoder_layer_job *layers,
        int n_layers,
        const float *x,
        int n_vec,
        int full_cache_capacity,
        int bits,
        int group_size,
        float **key_out_by_layer,
        float **value_out_by_layer,
        float *last_hidden_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_qwen_batch_prefill_chunk_u32(
        const ds4_drafter_metal_decoder_layer_job *layers,
        int n_layers,
        const float *x,
        int n_vec,
        int base_position,
        int full_n_ctx,
        int full_cache_capacity,
        int bits,
        int group_size,
        float **key_out_by_layer,
        float **value_out_by_layer,
        float *last_hidden_out,
        char *err,
        size_t errlen);

int ds4_drafter_metal_attention_importance(
        int cache_id,
        const float *queries_rope,
        int n_ctx,
        int pool_kernel,
        float *max_row,
        char *err,
        size_t errlen);

int ds4_drafter_metal_attention_importance_batch(
        int cache_id,
        const float *queries_rope,
        int n_queries,
        int n_ctx,
        int pool_kernel,
        float *max_rows,
        char *err,
        size_t errlen);

int ds4_drafter_metal_copy_full_attention_cache(
        int cache_id,
        int n_ctx,
        float *keys_out,
        float *values_out,
        char *err,
        size_t errlen);

#endif
