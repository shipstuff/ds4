#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "ds4_drafter_metal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int rows;
    int packed_cols;
    int cols;
    int groups;
    int bits;
    int group_size;
} ds4_drafter_metal_affine_args;

typedef struct {
    int row;
    int rows;
    int packed_cols;
    int cols;
    int groups;
    int bits;
    int group_size;
} ds4_drafter_metal_dequant_row_args;

typedef struct {
    int len;
    float eps;
} ds4_drafter_metal_rms_norm_args;

typedef struct {
    int n_ctx;
    float scale;
} ds4_drafter_metal_attention_args;

typedef struct {
    int n_ctx;
    int pool_kernel;
} ds4_drafter_metal_importance_args;

typedef struct {
    int position;
    int n_vec;
} ds4_drafter_metal_rope_args;

typedef struct {
    uint32_t n_vec;
    uint32_t qh;
    uint32_t kvh;
} ds4_drafter_metal_head_args;

typedef struct {
    uint32_t n_ctx;
    uint32_t q_start;
    uint32_t q_count;
    uint32_t qh;
    uint32_t kvh;
    uint32_t q_input_start;
    uint32_t q_output_start;
} ds4_drafter_metal_head_block_args;

static id<MTLDevice> g_drafter_device;
static id<MTLCommandQueue> g_drafter_queue;
static id<MTLComputePipelineState> g_affine_u32_matvec_pipeline;
static id<MTLComputePipelineState> g_affine_u32_matmat_pipeline;
static id<MTLComputePipelineState> g_affine_u32_matmat4_pipeline;
static id<MTLComputePipelineState> g_affine_u32_matmat4x2_pipeline;
static id<MTLComputePipelineState> g_affine_u32_matmat4x4_pipeline;
static id<MTLComputePipelineState> g_affine_u32_matmat8x2_pipeline;
static id<MTLComputePipelineState> g_affine_u32_q_only_matvec_pipeline;
static id<MTLComputePipelineState> g_dequant_u32_row_pipeline;
static id<MTLComputePipelineState> g_rms_norm_bf16_pipeline;
static id<MTLComputePipelineState> g_rms_norm_bf16_mat_pipeline;
static id<MTLComputePipelineState> g_rms_norm_bf16_mat_round_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_swiglu_x4_pipeline;
static id<MTLComputePipelineState> g_swiglu_packed_pair_pipeline;
static id<MTLComputePipelineState> g_attention_logits_pipeline;
static id<MTLComputePipelineState> g_attention_logits4_pipeline;
static id<MTLComputePipelineState> g_attention_context_pipeline;
static id<MTLComputePipelineState> g_attention_softmax_inplace_pipeline;
static id<MTLComputePipelineState> g_attention_context_parallel_pipeline;
static id<MTLComputePipelineState> g_attention_context4_pipeline;
static id<MTLComputePipelineState> g_attention_context8_pipeline;
static id<MTLComputePipelineState> g_attention_context16_pipeline;
static id<MTLComputePipelineState> g_attention_logits_causal_mat_pipeline;
static id<MTLComputePipelineState> g_attention_context_causal_mat_pipeline;
static id<MTLComputePipelineState> g_attention_context_causal_fused_mat_pipeline;
static id<MTLComputePipelineState> g_attention_context_causal_fused_mat_twopass_pipeline;
static id<MTLComputePipelineState> g_attention_pack_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_pack_q_block_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_pack_kv_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_pack_q_group_block_head_f16_pipeline;
static id<MTLComputePipelineState> g_attention_pack_q_group_block_head_f16x4_pipeline;
static id<MTLComputePipelineState> g_attention_pack_kv_head_f16_pipeline;
static id<MTLComputePipelineState> g_attention_pack_kv_head_f16x4_pipeline;
static id<MTLComputePipelineState> g_attention_pack_kv_all_head_f16x4_pipeline;
static id<MTLComputePipelineState> g_attention_pack_v_head_f16x4_pipeline;
static id<MTLComputePipelineState> g_attention_pack_q_group_block_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_causal_softmax_mat_pipeline;
static id<MTLComputePipelineState> g_attention_causal_softmax_block_mat_pipeline;
static id<MTLComputePipelineState> g_attention_causal_softmax_group_block_mat_pipeline;
static id<MTLComputePipelineState> g_attention_causal_softmax_group_block_f16_pipeline;
static id<MTLComputePipelineState> g_attention_causal_mask_group_block_mat_pipeline;
static id<MTLComputePipelineState> g_attention_unpack_gate_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_unpack_gate_block_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_unpack_gate_group_block_head_mat_pipeline;
static id<MTLComputePipelineState> g_attention_unpack_gate_group_block_head_f16_pipeline;
static id<MTLComputePipelineState> g_attention_unpack_gate_group_block_head_f16x4_pipeline;
static id<MTLComputePipelineState> g_attention_importance_pipeline;
static id<MTLComputePipelineState> g_attention_logits_batch_pipeline;
static id<MTLComputePipelineState> g_attention_logits_batch4_pipeline;
static id<MTLComputePipelineState> g_attention_importance_batch_pipeline;
static id<MTLComputePipelineState> g_attention_importance_reduce_heads_pipeline;
static id<MTLComputePipelineState> g_full_q_norm_rope_pipeline;
static id<MTLComputePipelineState> g_full_q_only_norm_rope_pipeline;
static id<MTLComputePipelineState> g_full_k_norm_rope_pipeline;
static id<MTLComputePipelineState> g_full_q_norm_rope_mat_pipeline;
static id<MTLComputePipelineState> g_full_k_norm_rope_mat_pipeline;
static id<MTLComputePipelineState> g_full_k_norm_rope_mat_pack_f16_pipeline;
static id<MTLComputePipelineState> g_argmax_pipeline;
static id<MTLComputePipelineState> g_linear_conv_pipeline;
static id<MTLComputePipelineState> g_linear_conv_mat_pipeline;
static id<MTLComputePipelineState> g_linear_conv_qkvz_mat_pipeline;
static id<MTLComputePipelineState> g_linear_conv_stateful_mat_pipeline;
static id<MTLComputePipelineState> g_linear_conv_stateful_qkvz_mat_pipeline;
static id<MTLComputePipelineState> g_linear_qk_norm_pipeline;
static id<MTLComputePipelineState> g_linear_qk_norm_mat_pipeline;
static id<MTLComputePipelineState> g_linear_qk_norm_kq_mat_pipeline;
static id<MTLComputePipelineState> g_linear_delta_pipeline;
static id<MTLComputePipelineState> g_linear_scan_params_pipeline;
static id<MTLComputePipelineState> g_linear_delta_scan_pipeline;
static id<MTLComputePipelineState> g_linear_delta_scan2_pipeline;
static id<MTLComputePipelineState> g_linear_delta_scan4_pipeline;
static id<MTLComputePipelineState> g_linear_conv_state_mat_pipeline;
static id<MTLComputePipelineState> g_linear_conv_state_qkvz_mat_pipeline;
static id<MTLComputePipelineState> g_linear_gate_pipeline;
static id<MTLComputePipelineState> g_linear_gate_mat_pipeline;
static id<MTLComputePipelineState> g_linear_gate_qkvz_mat_pipeline;
static id<MTLComputePipelineState> g_round_bf16_pipeline;
static id<MTLComputePipelineState> g_residual_add_round_pipeline;
static id<MTLComputePipelineState> g_residual_add_round_pre_b_round_pipeline;
static id<MTLComputePipelineState> g_residual_add_round_pre_b_round_norm_pipeline;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *g_buffer_cache;
static NSMutableDictionary<id<NSCopying>, id<MTLBuffer>> *g_dense_affine_cache;
static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *g_mps_matmul_cache;
static NSMutableDictionary<NSString *, NSNumber *> *g_mps_matmul_shape_warmup_cache;
static NSMutableDictionary<NSString *, MPSMatrix *> *g_mps_matrix_cache;
static MPSMatrixSoftMax *g_mps_matrix_softmax;

enum {
    DS4_DRAFTER_MPS_MATMUL_FAST_CACHE_SIZE = 2048,
    DS4_DRAFTER_MPS_MATRIX_FAST_CACHE_SIZE = 8192,
};

typedef struct {
    int used;
    int result_rows;
    int result_cols;
    int interior_cols;
    int transpose_left;
    int transpose_right;
    uint64_t alpha_key;
    __unsafe_unretained MPSMatrixMultiplication *matrix_multiplication;
} ds4_drafter_mps_matmul_fast_cache_entry;

typedef struct {
    int used;
    void *buffer;
    NSUInteger offset;
    int rows;
    int cols;
    NSUInteger row_bytes;
    MPSDataType data_type;
    __unsafe_unretained MPSMatrix *matrix;
} ds4_drafter_mps_matrix_fast_cache_entry;

static ds4_drafter_mps_matmul_fast_cache_entry
    g_mps_matmul_fast_cache[DS4_DRAFTER_MPS_MATMUL_FAST_CACHE_SIZE];
static ds4_drafter_mps_matrix_fast_cache_entry
    g_mps_matrix_fast_cache[DS4_DRAFTER_MPS_MATRIX_FAST_CACHE_SIZE];

static id<MTLBuffer> g_x_buffer;
static id<MTLBuffer> g_out_buffer;
static id<MTLBuffer> g_many_out_buffers[8];
static id<MTLBuffer> g_attention_q_buffer;
static id<MTLBuffer> g_attention_gate_buffer;
static id<MTLBuffer> g_attention_keys_buffer;
static id<MTLBuffer> g_attention_values_buffer;
static id<MTLBuffer> g_attention_logits_buffer;
static id<MTLBuffer> g_attention_out_buffer;
static id<MTLBuffer> g_attention_q_head_buffer;
static id<MTLBuffer> g_attention_k_head_buffer;
static id<MTLBuffer> g_attention_v_head_buffer;
static id<MTLBuffer> g_attention_ctx_head_buffer;
static id<MTLBuffer> g_attention_q_head_f16_buffer;
static id<MTLBuffer> g_attention_k_head_f16_buffer;
static id<MTLBuffer> g_attention_v_head_f16_buffer;
static id<MTLBuffer> g_attention_logits_f16_buffer;
static id<MTLBuffer> g_attention_ctx_head_f16_buffer;
static id<MTLBuffer> g_attention_key_cache_buffers[24];
static id<MTLBuffer> g_attention_value_cache_buffers[24];
static id<MTLBuffer> g_query_capture_buffers[24];
static id<MTLBuffer> g_argmax_buffer;
static id<MTLBuffer> g_importance_row_buffer;
static id<MTLBuffer> g_importance_max_buffer;
static id<MTLBuffer> g_linear_conv_state_buffers[24];
static id<MTLBuffer> g_linear_delta_state_buffers[24];
static id<MTLBuffer> g_linear_y_buffer;
static id<MTLBuffer> g_linear_kq_buffer;
static id<MTLBuffer> g_linear_scan_params_buffer;
static id<MTLBuffer> g_linear_scan_state_buffer;
static id<MTLBuffer> g_linear_scan_debug_buffer;
static id<MTLBuffer> g_token_hidden_buffers[2];
static NSUInteger g_x_bytes;
static NSUInteger g_out_bytes;
static NSUInteger g_many_out_bytes[8];
static NSUInteger g_attention_q_bytes;
static NSUInteger g_attention_gate_bytes;
static NSUInteger g_attention_keys_bytes;
static NSUInteger g_attention_values_bytes;
static NSUInteger g_attention_logits_bytes;
static NSUInteger g_attention_out_bytes;
static NSUInteger g_attention_q_head_bytes;
static NSUInteger g_attention_k_head_bytes;
static NSUInteger g_attention_v_head_bytes;
static NSUInteger g_attention_ctx_head_bytes;
static NSUInteger g_attention_q_head_f16_bytes;
static NSUInteger g_attention_k_head_f16_bytes;
static NSUInteger g_attention_v_head_f16_bytes;
static NSUInteger g_attention_logits_f16_bytes;
static NSUInteger g_attention_ctx_head_f16_bytes;
static int g_attention_prepacked_k_f16_n_ctx;
static NSUInteger g_attention_key_cache_bytes[24];
static NSUInteger g_attention_value_cache_bytes[24];
static NSUInteger g_query_capture_bytes[24];
static NSUInteger g_argmax_bytes;
static NSUInteger g_importance_row_bytes;
static NSUInteger g_importance_max_bytes;
static NSUInteger g_linear_conv_state_bytes[24];
static NSUInteger g_linear_delta_state_bytes[24];
static NSUInteger g_linear_y_bytes;
static NSUInteger g_linear_kq_bytes;
static NSUInteger g_linear_scan_params_bytes;
static NSUInteger g_linear_scan_state_bytes;
static NSUInteger g_linear_scan_debug_bytes;
static NSUInteger g_token_hidden_bytes[2];
static int g_init_attempted;
static int g_init_ok;

static int drafter_ensure_buffer(__strong id<MTLBuffer> *buf,
                                 NSUInteger *cap,
                                 NSUInteger need,
                                 const char *label,
                                 char *err,
                                 size_t errlen);
static int drafter_ensure_private_buffer(__strong id<MTLBuffer> *buf,
                                         NSUInteger *cap,
                                         NSUInteger need,
                                         const char *label,
                                         char *err,
                                         size_t errlen);

static const char *g_drafter_metal_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct ds4_drafter_metal_affine_args {\n"
"    int rows;\n"
"    int packed_cols;\n"
"    int cols;\n"
"    int groups;\n"
"    int bits;\n"
"    int group_size;\n"
"};\n"
"\n"
"static inline float bf16_to_f32(ushort v) {\n"
"    return as_type<float>(uint(v) << 16);\n"
"}\n"
"\n"
"static inline float round_f32_to_bf16_f32(float v) {\n"
"    uint raw = as_type<uint>(v);\n"
"    uint lsb = (raw >> 16) & 1u;\n"
"    raw += 0x7fffu + lsb;\n"
"    raw &= 0xffff0000u;\n"
"    return as_type<float>(raw);\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matvec(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (row >= uint(args.rows)) return;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const bool group_aligned = (uint(args.group_size) % pack) == 0u;\n"
"    float acc = 0.0f;\n"
"    const uint w_base = row * uint(args.packed_cols);\n"
"    const uint sb_base = row * uint(args.groups);\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint packed = w[w_base + pc];\n"
"        uint g_pack = (pc * pack) / uint(args.group_size);\n"
"        float scale_pack = bf16_to_f32(scales[sb_base + g_pack]);\n"
"        float bias_pack = bf16_to_f32(biases[sb_base + g_pack]);\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            float scale = scale_pack;\n"
"            float bias = bias_pack;\n"
"            if (!group_aligned) {\n"
"                uint g = col / uint(args.group_size);\n"
"                scale = bf16_to_f32(scales[sb_base + g]);\n"
"                bias = bf16_to_f32(biases[sb_base + g]);\n"
"            }\n"
"            uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"            acc += x[col] * (float(q) * scale + bias);\n"
"        }\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[row] = scratch[0];\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_q_only_matvec(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (row >= uint(args.rows)) return;\n"
"    const uint head = row / 256u;\n"
"    const uint dim = row - head * 256u;\n"
"    const uint src_row = head * 512u + dim;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const bool group_aligned = (uint(args.group_size) % pack) == 0u;\n"
"    float acc = 0.0f;\n"
"    const uint w_base = src_row * uint(args.packed_cols);\n"
"    const uint sb_base = src_row * uint(args.groups);\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint packed = w[w_base + pc];\n"
"        uint g_pack = (pc * pack) / uint(args.group_size);\n"
"        float scale_pack = bf16_to_f32(scales[sb_base + g_pack]);\n"
"        float bias_pack = bf16_to_f32(biases[sb_base + g_pack]);\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            float scale = scale_pack;\n"
"            float bias = bias_pack;\n"
"            if (!group_aligned) {\n"
"                uint g = col / uint(args.group_size);\n"
"                scale = bf16_to_f32(scales[sb_base + g]);\n"
"                bias = bf16_to_f32(biases[sb_base + g]);\n"
"            }\n"
"            uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"            acc += x[col] * (float(q) * scale + bias);\n"
"        }\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[row] = scratch[0];\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matmat(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint row = gid.x;\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (row >= uint(args.rows)) return;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const bool group_aligned = (uint(args.group_size) % pack) == 0u;\n"
"    float acc = 0.0f;\n"
"    const uint w_base = row * uint(args.packed_cols);\n"
"    const uint sb_base = row * uint(args.groups);\n"
"    const uint x_base = vec * uint(args.cols);\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint packed = w[w_base + pc];\n"
"        uint g_pack = (pc * pack) / uint(args.group_size);\n"
"        float scale_pack = bf16_to_f32(scales[sb_base + g_pack]);\n"
"        float bias_pack = bf16_to_f32(biases[sb_base + g_pack]);\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            float scale = scale_pack;\n"
"            float bias = bias_pack;\n"
"            if (!group_aligned) {\n"
"                uint g = col / uint(args.group_size);\n"
"                scale = bf16_to_f32(scales[sb_base + g]);\n"
"                bias = bf16_to_f32(biases[sb_base + g]);\n"
"            }\n"
"            uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"            acc += x[x_base + col] * (float(q) * scale + bias);\n"
"        }\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[vec * uint(args.rows) + row] = scratch[0];\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matmat4(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        constant uint &n_vec [[buffer(6)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint row = gid.x;\n"
"    const uint vec0 = gid.y * 4u;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (row >= uint(args.rows) || vec0 >= n_vec) return;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    float acc0 = 0.0f;\n"
"    float acc1 = 0.0f;\n"
"    float acc2 = 0.0f;\n"
"    float acc3 = 0.0f;\n"
"    const bool v1 = vec0 + 1u < n_vec;\n"
"    const bool v2 = vec0 + 2u < n_vec;\n"
"    const bool v3 = vec0 + 3u < n_vec;\n"
"    const uint w_base = row * uint(args.packed_cols);\n"
"    const uint sb_base = row * uint(args.groups);\n"
"    const uint x_base0 = vec0 * uint(args.cols);\n"
"    const uint x_base1 = x_base0 + uint(args.cols);\n"
"    const uint x_base2 = x_base1 + uint(args.cols);\n"
"    const uint x_base3 = x_base2 + uint(args.cols);\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint packed = w[w_base + pc];\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            uint g = col / uint(args.group_size);\n"
"            float scale = bf16_to_f32(scales[sb_base + g]);\n"
"            float bias = bf16_to_f32(biases[sb_base + g]);\n"
"            uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"            float wv = float(q) * scale + bias;\n"
"            acc0 += x[x_base0 + col] * wv;\n"
"            if (v1) acc1 += x[x_base1 + col] * wv;\n"
"            if (v2) acc2 += x[x_base2 + col] * wv;\n"
"            if (v3) acc3 += x[x_base3 + col] * wv;\n"
"        }\n"
"    }\n"
"    scratch[tid] = acc0;\n"
"    scratch[nt + tid] = acc1;\n"
"    scratch[2u * nt + tid] = acc2;\n"
"    scratch[3u * nt + tid] = acc3;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[tid] += scratch[tid + stride];\n"
"            scratch[nt + tid] += scratch[nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        out[vec0 * uint(args.rows) + row] = scratch[0];\n"
"        if (v1) out[(vec0 + 1u) * uint(args.rows) + row] = scratch[nt];\n"
"        if (v2) out[(vec0 + 2u) * uint(args.rows) + row] = scratch[2u * nt];\n"
"        if (v3) out[(vec0 + 3u) * uint(args.rows) + row] = scratch[3u * nt];\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matmat4x2(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        constant uint &n_vec [[buffer(6)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint row0 = gid.x * 2u;\n"
"    const uint row1 = row0 + 1u;\n"
"    const uint vec0 = gid.y * 4u;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (row0 >= uint(args.rows) || vec0 >= n_vec) return;\n"
"    const bool r1 = row1 < uint(args.rows);\n"
"    const bool v1 = vec0 + 1u < n_vec;\n"
"    const bool v2 = vec0 + 2u < n_vec;\n"
"    const bool v3 = vec0 + 3u < n_vec;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const bool group_aligned = (uint(args.group_size) % pack) == 0u;\n"
"    float a00 = 0.0f;\n"
"    float a01 = 0.0f;\n"
"    float a02 = 0.0f;\n"
"    float a03 = 0.0f;\n"
"    float a10 = 0.0f;\n"
"    float a11 = 0.0f;\n"
"    float a12 = 0.0f;\n"
"    float a13 = 0.0f;\n"
"    const uint w_base0 = row0 * uint(args.packed_cols);\n"
"    const uint w_base1 = row1 * uint(args.packed_cols);\n"
"    const uint sb_base0 = row0 * uint(args.groups);\n"
"    const uint sb_base1 = row1 * uint(args.groups);\n"
"    const uint x_base0 = vec0 * uint(args.cols);\n"
"    const uint x_base1 = x_base0 + uint(args.cols);\n"
"    const uint x_base2 = x_base1 + uint(args.cols);\n"
"    const uint x_base3 = x_base2 + uint(args.cols);\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint packed0 = w[w_base0 + pc];\n"
"        uint packed1 = r1 ? w[w_base1 + pc] : 0u;\n"
"        uint g_pack = (pc * pack) / uint(args.group_size);\n"
"        float scale0_pack = bf16_to_f32(scales[sb_base0 + g_pack]);\n"
"        float bias0_pack = bf16_to_f32(biases[sb_base0 + g_pack]);\n"
"        float scale1_pack = r1 ? bf16_to_f32(scales[sb_base1 + g_pack]) : 0.0f;\n"
"        float bias1_pack = r1 ? bf16_to_f32(biases[sb_base1 + g_pack]) : 0.0f;\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            float x0 = x[x_base0 + col];\n"
"            float x1 = v1 ? x[x_base1 + col] : 0.0f;\n"
"            float x2 = v2 ? x[x_base2 + col] : 0.0f;\n"
"            float x3 = v3 ? x[x_base3 + col] : 0.0f;\n"
"            float scale0 = scale0_pack;\n"
"            float bias0 = bias0_pack;\n"
"            float scale1 = scale1_pack;\n"
"            float bias1 = bias1_pack;\n"
"            if (!group_aligned) {\n"
"                uint g = col / uint(args.group_size);\n"
"                scale0 = bf16_to_f32(scales[sb_base0 + g]);\n"
"                bias0 = bf16_to_f32(biases[sb_base0 + g]);\n"
"                if (r1) {\n"
"                    scale1 = bf16_to_f32(scales[sb_base1 + g]);\n"
"                    bias1 = bf16_to_f32(biases[sb_base1 + g]);\n"
"                }\n"
"            }\n"
"            uint q0 = (packed0 >> (lane * uint(args.bits))) & mask;\n"
"            float wv0 = float(q0) * scale0 + bias0;\n"
"            a00 += x0 * wv0;\n"
"            a01 += x1 * wv0;\n"
"            a02 += x2 * wv0;\n"
"            a03 += x3 * wv0;\n"
"            if (r1) {\n"
"                uint q1 = (packed1 >> (lane * uint(args.bits))) & mask;\n"
"                float wv1 = float(q1) * scale1 + bias1;\n"
"                a10 += x0 * wv1;\n"
"                a11 += x1 * wv1;\n"
"                a12 += x2 * wv1;\n"
"                a13 += x3 * wv1;\n"
"            }\n"
"        }\n"
"    }\n"
"    scratch[tid] = a00;\n"
"    scratch[nt + tid] = a01;\n"
"    scratch[2u * nt + tid] = a02;\n"
"    scratch[3u * nt + tid] = a03;\n"
"    scratch[4u * nt + tid] = a10;\n"
"    scratch[5u * nt + tid] = a11;\n"
"    scratch[6u * nt + tid] = a12;\n"
"    scratch[7u * nt + tid] = a13;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[tid] += scratch[tid + stride];\n"
"            scratch[nt + tid] += scratch[nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"            scratch[4u * nt + tid] += scratch[4u * nt + tid + stride];\n"
"            scratch[5u * nt + tid] += scratch[5u * nt + tid + stride];\n"
"            scratch[6u * nt + tid] += scratch[6u * nt + tid + stride];\n"
"            scratch[7u * nt + tid] += scratch[7u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        out[vec0 * uint(args.rows) + row0] = scratch[0];\n"
"        if (v1) out[(vec0 + 1u) * uint(args.rows) + row0] = scratch[nt];\n"
"        if (v2) out[(vec0 + 2u) * uint(args.rows) + row0] = scratch[2u * nt];\n"
"        if (v3) out[(vec0 + 3u) * uint(args.rows) + row0] = scratch[3u * nt];\n"
"        if (r1) {\n"
"            out[vec0 * uint(args.rows) + row1] = scratch[4u * nt];\n"
"            if (v1) out[(vec0 + 1u) * uint(args.rows) + row1] = scratch[5u * nt];\n"
"            if (v2) out[(vec0 + 2u) * uint(args.rows) + row1] = scratch[6u * nt];\n"
"            if (v3) out[(vec0 + 3u) * uint(args.rows) + row1] = scratch[7u * nt];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matmat4x4(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        constant uint &n_vec [[buffer(6)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint row_base = gid.x * 4u;\n"
"    const uint vec_base = gid.y * 4u;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (row_base >= uint(args.rows) || vec_base >= n_vec) return;\n"
"    const bool rv[4] = { row_base < uint(args.rows), row_base + 1u < uint(args.rows), row_base + 2u < uint(args.rows), row_base + 3u < uint(args.rows) };\n"
"    const bool vv[4] = { vec_base < n_vec, vec_base + 1u < n_vec, vec_base + 2u < n_vec, vec_base + 3u < n_vec };\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const bool group_aligned = (uint(args.group_size) % pack) == 0u;\n"
"    float acc[16];\n"
"    for (uint i = 0; i < 16u; i++) acc[i] = 0.0f;\n"
"    const uint xb[4] = { vec_base * uint(args.cols), (vec_base + 1u) * uint(args.cols), (vec_base + 2u) * uint(args.cols), (vec_base + 3u) * uint(args.cols) };\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        uint g_pack = (pc * pack) / uint(args.group_size);\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            uint g = group_aligned ? g_pack : col / uint(args.group_size);\n"
"            float xv[4];\n"
"            for (uint v = 0; v < 4u; v++) xv[v] = vv[v] ? x[xb[v] + col] : 0.0f;\n"
"            for (uint r = 0; r < 4u; r++) {\n"
"                if (!rv[r]) continue;\n"
"                uint row = row_base + r;\n"
"                uint packed = w[row * uint(args.packed_cols) + pc];\n"
"                float scale = bf16_to_f32(scales[row * uint(args.groups) + g]);\n"
"                float bias = bf16_to_f32(biases[row * uint(args.groups) + g]);\n"
"                uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"                float wv = float(q) * scale + bias;\n"
"                for (uint v = 0; v < 4u; v++) acc[r * 4u + v] += xv[v] * wv;\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint i = 0; i < 16u; i++) scratch[i * nt + tid] = acc[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            for (uint i = 0; i < 16u; i++) scratch[i * nt + tid] += scratch[i * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        for (uint r = 0; r < 4u; r++) {\n"
"            if (!rv[r]) continue;\n"
"            uint row = row_base + r;\n"
"            for (uint v = 0; v < 4u; v++) {\n"
"                if (vv[v]) out[(vec_base + v) * uint(args.rows) + row] = scratch[(r * 4u + v) * nt];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_affine_u32_matmat8x2(\n"
"        constant ds4_drafter_metal_affine_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device const float *x [[buffer(4)]],\n"
"        device float *out [[buffer(5)]],\n"
"        constant uint &n_vec [[buffer(6)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint row_base = gid.x * 2u;\n"
"    const uint vec_base = gid.y * 8u;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (row_base >= uint(args.rows) || vec_base >= n_vec) return;\n"
"    const bool rv[2] = { row_base < uint(args.rows), row_base + 1u < uint(args.rows) };\n"
"    const bool vv[8] = { vec_base < n_vec, vec_base + 1u < n_vec, vec_base + 2u < n_vec, vec_base + 3u < n_vec, vec_base + 4u < n_vec, vec_base + 5u < n_vec, vec_base + 6u < n_vec, vec_base + 7u < n_vec };\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    float acc[16];\n"
"    for (uint i = 0; i < 16u; i++) acc[i] = 0.0f;\n"
"    const uint xb[8] = { vec_base * uint(args.cols), (vec_base + 1u) * uint(args.cols), (vec_base + 2u) * uint(args.cols), (vec_base + 3u) * uint(args.cols), (vec_base + 4u) * uint(args.cols), (vec_base + 5u) * uint(args.cols), (vec_base + 6u) * uint(args.cols), (vec_base + 7u) * uint(args.cols) };\n"
"    for (uint pc = tid; pc < uint(args.packed_cols); pc += nt) {\n"
"        for (uint lane = 0; lane < pack; lane++) {\n"
"            uint col = pc * pack + lane;\n"
"            if (col >= uint(args.cols)) continue;\n"
"            uint g = col / uint(args.group_size);\n"
"            float xv[8];\n"
"            for (uint v = 0; v < 8u; v++) xv[v] = vv[v] ? x[xb[v] + col] : 0.0f;\n"
"            for (uint r = 0; r < 2u; r++) {\n"
"                if (!rv[r]) continue;\n"
"                uint row = row_base + r;\n"
"                uint packed = w[row * uint(args.packed_cols) + pc];\n"
"                float scale = bf16_to_f32(scales[row * uint(args.groups) + g]);\n"
"                float bias = bf16_to_f32(biases[row * uint(args.groups) + g]);\n"
"                uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"                float wv = float(q) * scale + bias;\n"
"                for (uint v = 0; v < 8u; v++) acc[r * 8u + v] += xv[v] * wv;\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint i = 0; i < 16u; i++) scratch[i * nt + tid] = acc[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            for (uint i = 0; i < 16u; i++) scratch[i * nt + tid] += scratch[i * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        for (uint r = 0; r < 2u; r++) {\n"
"            if (!rv[r]) continue;\n"
"            uint row = row_base + r;\n"
"            for (uint v = 0; v < 8u; v++) {\n"
"                if (vv[v]) out[(vec_base + v) * uint(args.rows) + row] = scratch[(r * 8u + v) * nt];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"struct ds4_drafter_metal_dequant_row_args {\n"
"    int row;\n"
"    int rows;\n"
"    int packed_cols;\n"
"    int cols;\n"
"    int groups;\n"
"    int bits;\n"
"    int group_size;\n"
"};\n"
"\n"
"kernel void ds4_drafter_dequant_u32_row(\n"
"        constant ds4_drafter_metal_dequant_row_args &args [[buffer(0)]],\n"
"        device const uint *w [[buffer(1)]],\n"
"        device const ushort *scales [[buffer(2)]],\n"
"        device const ushort *biases [[buffer(3)]],\n"
"        device float *out [[buffer(4)]],\n"
"        uint col [[thread_position_in_grid]]) {\n"
"    if (args.row < 0 || args.row >= args.rows || col >= uint(args.cols)) return;\n"
"    const uint pack = uint(32 / args.bits);\n"
"    const uint mask = (uint(1) << uint(args.bits)) - uint(1);\n"
"    const uint pc = col / pack;\n"
"    const uint lane = col - pc * pack;\n"
"    const uint packed = w[uint(args.row) * uint(args.packed_cols) + pc];\n"
"    const uint g = col / uint(args.group_size);\n"
"    const float scale = bf16_to_f32(scales[uint(args.row) * uint(args.groups) + g]);\n"
"    const float bias = bf16_to_f32(biases[uint(args.row) * uint(args.groups) + g]);\n"
"    const uint q = (packed >> (lane * uint(args.bits))) & mask;\n"
"    out[col] = round_f32_to_bf16_f32(float(q) * scale + bias);\n"
"}\n"
"\n"
"kernel void ds4_drafter_argmax(\n"
"        constant uint &rows [[buffer(0)]],\n"
"        device const float *logits [[buffer(1)]],\n"
"        device uint *out_token [[buffer(2)]],\n"
"        threadgroup float *score_scratch [[threadgroup(0)]],\n"
"        threadgroup uint *idx_scratch [[threadgroup(1)]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    float best = -INFINITY;\n"
"    uint best_idx = 0u;\n"
"    for (uint r = tid; r < rows; r += nt) {\n"
"        float v = logits[r];\n"
"        if (v > best || (v == best && r < best_idx)) {\n"
"            best = v;\n"
"            best_idx = r;\n"
"        }\n"
"    }\n"
"    score_scratch[tid] = best;\n"
"    idx_scratch[tid] = best_idx;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            float other = score_scratch[tid + stride];\n"
"            uint other_idx = idx_scratch[tid + stride];\n"
"            if (other > score_scratch[tid] ||\n"
"                (other == score_scratch[tid] && other_idx < idx_scratch[tid])) {\n"
"                score_scratch[tid] = other;\n"
"                idx_scratch[tid] = other_idx;\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out_token[0] = idx_scratch[0];\n"
"}\n"
"\n"
"struct ds4_drafter_metal_rms_norm_args {\n"
"    int len;\n"
"    float eps;\n"
"};\n"
"\n"
"kernel void ds4_drafter_rms_norm_bf16(\n"
"        constant ds4_drafter_metal_rms_norm_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *x [[buffer(2)]],\n"
"        device float *out [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        float v = x[i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / float(args.len)) + args.eps);\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        out[i] = x[i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_rms_norm_bf16_mat(\n"
"        constant ds4_drafter_metal_rms_norm_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *x [[buffer(2)]],\n"
"        device float *out [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint base = vec * uint(args.len);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        float v = x[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / float(args.len)) + args.eps);\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        out[base + i] = x[base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_rms_norm_bf16_mat_round(\n"
"        constant ds4_drafter_metal_rms_norm_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *x [[buffer(2)]],\n"
"        device float *out [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint base = vec * uint(args.len);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        float v = x[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / float(args.len)) + args.eps);\n"
"    for (uint i = tid; i < uint(args.len); i += nt) {\n"
"        out[base + i] = round_f32_to_bf16_f32(x[base + i] * inv * bf16_to_f32(weight[i]));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_swiglu(\n"
"        device const float *gate [[buffer(0)]],\n"
"        device const float *up [[buffer(1)]],\n"
"        device float *out [[buffer(2)]],\n"
"        constant uint &len [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len) return;\n"
"    float g = gate[tid];\n"
"    out[tid] = (g / (1.0f + fast::exp(-g))) * up[tid];\n"
"}\n"
"\n"
"kernel void ds4_drafter_swiglu_x4(\n"
"        device const float4 *gate [[buffer(0)]],\n"
"        device const float4 *up [[buffer(1)]],\n"
"        device float4 *out [[buffer(2)]],\n"
"        constant uint &len4 [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len4) return;\n"
"    float4 g = gate[tid];\n"
"    out[tid] = (g / (1.0f + fast::exp(-g))) * up[tid];\n"
"}\n"
"\n"
"kernel void ds4_drafter_swiglu_packed_pair(\n"
"        device const float *gate_up [[buffer(0)]],\n"
"        device float *out [[buffer(1)]],\n"
"        constant uint &n_vec [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint hidden = 3584u;\n"
"    const uint total = n_vec * hidden;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / hidden;\n"
"    const uint i = tid - t * hidden;\n"
"    const uint base = t * (2u * hidden);\n"
"    float g = gate_up[base + i];\n"
"    float u = gate_up[base + hidden + i];\n"
"    out[tid] = (g / (1.0f + fast::exp(-g))) * u;\n"
"}\n"
"\n"
"kernel void ds4_drafter_round_bf16(\n"
"        device float *x [[buffer(0)]],\n"
"        constant uint &len [[buffer(1)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len) return;\n"
"    x[tid] = round_f32_to_bf16_f32(x[tid]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_residual_add_round(\n"
"        device const float *a [[buffer(0)]],\n"
"        device const float *b [[buffer(1)]],\n"
"        device float *out [[buffer(2)]],\n"
"        constant uint &len [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len) return;\n"
"    out[tid] = round_f32_to_bf16_f32(a[tid] + b[tid]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_residual_add_round_pre_b_round(\n"
"        device const float *a [[buffer(0)]],\n"
"        device const float *b [[buffer(1)]],\n"
"        device float *out [[buffer(2)]],\n"
"        constant uint &len [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len) return;\n"
"    out[tid] = round_f32_to_bf16_f32(a[tid] + round_f32_to_bf16_f32(b[tid]));\n"
"}\n"
"\n"
"kernel void ds4_drafter_residual_add_round_pre_b_round_norm(\n"
"        constant ds4_drafter_metal_rms_norm_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *a [[buffer(2)]],\n"
"        device const float *b [[buffer(3)]],\n"
"        device float *residual_out [[buffer(4)]],\n"
"        device float *norm_out [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint len = uint(args.len);\n"
"    const uint base = vec * len;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < len; i += nt) {\n"
"        float y = round_f32_to_bf16_f32(a[base + i] + round_f32_to_bf16_f32(b[base + i]));\n"
"        ss += y * y;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / float(len)) + args.eps);\n"
"    for (uint i = tid; i < len; i += nt) {\n"
"        float y = round_f32_to_bf16_f32(a[base + i] + round_f32_to_bf16_f32(b[base + i]));\n"
"        residual_out[base + i] = y;\n"
"        norm_out[base + i] = round_f32_to_bf16_f32(y * inv * bf16_to_f32(weight[i]));\n"
"    }\n"
"}\n"
"\n"
"struct ds4_drafter_metal_head_args {\n"
"    uint n_vec;\n"
"    uint qh;\n"
"    uint kvh;\n"
"};\n"
"\n"
"struct ds4_drafter_metal_head_block_args {\n"
"    uint n_ctx;\n"
"    uint q_start;\n"
"    uint q_count;\n"
"    uint qh;\n"
"    uint kvh;\n"
"    uint q_input_start;\n"
"    uint q_output_start;\n"
"};\n"
"\n"
"kernel void ds4_drafter_attention_pack_head_mat(\n"
"        constant ds4_drafter_metal_head_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *q_head [[buffer(4)]],\n"
"        device float *k_head [[buffer(5)]],\n"
"        device float *v_head [[buffer(6)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_vec * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 256u;\n"
"    const uint d = tid - t * 256u;\n"
"    q_head[tid] = queries[t * 2048u + args.qh * 256u + d];\n"
"    k_head[tid] = keys[t * 512u + args.kvh * 256u + d];\n"
"    v_head[tid] = values[t * 512u + args.kvh * 256u + d];\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_q_block_head_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device float *q_head [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint local_t = tid / 256u;\n"
"    const uint d = tid - local_t * 256u;\n"
"    const uint t = args.q_input_start + local_t;\n"
"    q_head[tid] = queries[t * 2048u + args.qh * 256u + d];\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_q_group_block_head_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device float *q_head [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 256u;\n"
"    const uint d = tid - row * 256u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint t = args.q_input_start + local_t;\n"
"    q_head[tid] = queries[t * 2048u + qh * 256u + d];\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_kv_head_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *keys [[buffer(1)]],\n"
"        device const float *values [[buffer(2)]],\n"
"        device float *k_head [[buffer(3)]],\n"
"        device float *v_head [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_ctx * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 256u;\n"
"    const uint d = tid - t * 256u;\n"
"    k_head[tid] = keys[t * 512u + args.kvh * 256u + d];\n"
"    v_head[tid] = values[t * 512u + args.kvh * 256u + d];\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_q_group_block_head_f16(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device half *q_head [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 256u;\n"
"    const uint d = tid - row * 256u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint t = args.q_input_start + local_t;\n"
"    q_head[tid] = half(queries[t * 2048u + qh * 256u + d]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_q_group_block_head_f16x4(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device half *q_head [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 64u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 64u;\n"
"    const uint d4 = tid - row * 64u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint t = args.q_input_start + local_t;\n"
"    const device float4 *src = (const device float4 *)(queries + t * 2048u + qh * 256u);\n"
"    device half4 *dst = (device half4 *)q_head;\n"
"    dst[tid] = half4(src[d4]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_kv_head_f16(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *keys [[buffer(1)]],\n"
"        device const float *values [[buffer(2)]],\n"
"        device half *k_head [[buffer(3)]],\n"
"        device half *v_head [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_ctx * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 256u;\n"
"    const uint d = tid - t * 256u;\n"
"    k_head[tid] = half(keys[t * 512u + args.kvh * 256u + d]);\n"
"    v_head[tid] = half(values[t * 512u + args.kvh * 256u + d]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_kv_head_f16x4(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *keys [[buffer(1)]],\n"
"        device const float *values [[buffer(2)]],\n"
"        device half *k_head [[buffer(3)]],\n"
"        device half *v_head [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_ctx * 64u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 64u;\n"
"    const uint d4 = tid - t * 64u;\n"
"    const device float4 *ksrc = (const device float4 *)(keys + t * 512u + args.kvh * 256u);\n"
"    const device float4 *vsrc = (const device float4 *)(values + t * 512u + args.kvh * 256u);\n"
"    device half4 *kdst = (device half4 *)k_head;\n"
"    device half4 *vdst = (device half4 *)v_head;\n"
"    kdst[tid] = half4(ksrc[d4]);\n"
"    vdst[tid] = half4(vsrc[d4]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_v_head_f16x4(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *values [[buffer(1)]],\n"
"        device half *v_head [[buffer(2)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_ctx * 64u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 64u;\n"
"    const uint d4 = tid - t * 64u;\n"
"    const device float4 *vsrc = (const device float4 *)(values + t * 512u + args.kvh * 256u);\n"
"    device half4 *vdst = (device half4 *)v_head;\n"
"    vdst[tid] = half4(vsrc[d4]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_pack_kv_all_head_f16x4(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *keys [[buffer(1)]],\n"
"        device const float *values [[buffer(2)]],\n"
"        device half *k_head [[buffer(3)]],\n"
"        device half *v_head [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 2u * args.n_ctx * 64u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 64u;\n"
"    const uint d4 = tid - row * 64u;\n"
"    const uint kvh = row / args.n_ctx;\n"
"    const uint t = row - kvh * args.n_ctx;\n"
"    const device float4 *ksrc = (const device float4 *)(keys + t * 512u + kvh * 256u);\n"
"    const device float4 *vsrc = (const device float4 *)(values + t * 512u + kvh * 256u);\n"
"    device half4 *kdst = (device half4 *)k_head;\n"
"    device half4 *vdst = (device half4 *)v_head;\n"
"    kdst[tid] = half4(ksrc[d4]);\n"
"    vdst[tid] = half4(vsrc[d4]);\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_causal_softmax_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (row >= n_vec) return;\n"
"    const uint base = row * n_vec;\n"
"    float local_max = -INFINITY;\n"
"    for (uint col = tid; col <= row; col += nt) {\n"
"        local_max = max(local_max, logits[base + col]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint col = tid; col <= row; col += nt) {\n"
"        local_denom += exp(logits[base + col] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    for (uint col = tid; col < n_vec; col += nt) {\n"
"        logits[base + col] = col <= row ? exp(logits[base + col] - max_logit) * inv_denom : 0.0f;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_causal_softmax_block_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (row >= args.q_count) return;\n"
"    const uint n_ctx = args.n_ctx;\n"
"    const uint global_row = args.q_start + row;\n"
"    const uint base = row * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_max = max(local_max, logits[base + col]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_denom += exp(logits[base + col] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    for (uint col = tid; col < n_ctx; col += nt) {\n"
"        logits[base + col] = col <= global_row ? exp(logits[base + col] - max_logit) * inv_denom : 0.0f;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_causal_softmax_group_block_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint sgitg [[simdgroup_index_in_threadgroup]],\n"
"        uint tiisg [[thread_index_in_simdgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    const uint row_count = 4u * args.q_count;\n"
"    if (row >= row_count) return;\n"
"    const uint n_ctx = args.n_ctx;\n"
"    const uint local_t = row - (row / args.q_count) * args.q_count;\n"
"    const uint global_row = args.q_start + local_t;\n"
"    const uint base = row * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_max = max(local_max, logits[base + col]);\n"
"    }\n"
"    float max_logit = simd_max(local_max);\n"
"    if (nt > 32u) {\n"
"        if (tiisg == 0u) scratch[sgitg] = max_logit;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint ng = (nt + 31u) >> 5u;\n"
"        max_logit = simd_max(tiisg < ng ? scratch[tiisg] : -INFINITY);\n"
"    }\n"
"    float local_denom = 0.0f;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_denom += fast::exp(logits[base + col] - max_logit);\n"
"    }\n"
"    float denom = simd_sum(local_denom);\n"
"    if (nt > 32u) {\n"
"        if (tiisg == 0u) scratch[sgitg] = denom;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint ng = (nt + 31u) >> 5u;\n"
"        denom = simd_sum(tiisg < ng ? scratch[tiisg] : 0.0f);\n"
"    }\n"
"    const float inv_denom = 1.0f / denom;\n"
"    for (uint col = tid; col < n_ctx; col += nt) {\n"
"        logits[base + col] = col <= global_row ? fast::exp(logits[base + col] - max_logit) * inv_denom : 0.0f;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_causal_softmax_group_block_f16(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device half *logits [[buffer(1)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint row [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint sgitg [[simdgroup_index_in_threadgroup]],\n"
"        uint tiisg [[thread_index_in_simdgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    const uint row_count = 4u * args.q_count;\n"
"    if (row >= row_count) return;\n"
"    const uint n_ctx = args.n_ctx;\n"
"    const uint local_t = row - (row / args.q_count) * args.q_count;\n"
"    const uint global_row = args.q_start + local_t;\n"
"    const uint base = row * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_max = max(local_max, float(logits[base + col]));\n"
"    }\n"
"    float max_logit = simd_max(local_max);\n"
"    if (nt > 32u) {\n"
"        if (tiisg == 0u) scratch[sgitg] = max_logit;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint ng = (nt + 31u) >> 5u;\n"
"        max_logit = simd_max(tiisg < ng ? scratch[tiisg] : -INFINITY);\n"
"    }\n"
"    float local_denom = 0.0f;\n"
"    for (uint col = tid; col <= global_row && col < n_ctx; col += nt) {\n"
"        local_denom += fast::exp(float(logits[base + col]) - max_logit);\n"
"    }\n"
"    float denom = simd_sum(local_denom);\n"
"    if (nt > 32u) {\n"
"        if (tiisg == 0u) scratch[sgitg] = denom;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint ng = (nt + 31u) >> 5u;\n"
"        denom = simd_sum(tiisg < ng ? scratch[tiisg] : 0.0f);\n"
"    }\n"
"    const float inv_denom = 1.0f / denom;\n"
"    for (uint col = tid; col < n_ctx; col += nt) {\n"
"        logits[base + col] = half(col <= global_row ? fast::exp(float(logits[base + col]) - max_logit) * inv_denom : 0.0f);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_causal_mask_group_block_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        uint2 tid2 [[thread_position_in_grid]]) {\n"
"    const uint row = tid2.y;\n"
"    const uint col = tid2.x;\n"
"    const uint row_count = 4u * args.q_count;\n"
"    const uint n_ctx = args.n_ctx;\n"
"    if (row >= row_count || col >= n_ctx) return;\n"
"    const uint local_t = row - (row / args.q_count) * args.q_count;\n"
"    const uint global_row = args.q_start + local_t;\n"
"    if (col > global_row) logits[row * n_ctx + col] = -INFINITY;\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_unpack_gate_head_mat(\n"
"        constant ds4_drafter_metal_head_args &args [[buffer(0)]],\n"
"        device const float *ctx_head [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device float *attn [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.n_vec * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 256u;\n"
"    const uint d = tid - t * 256u;\n"
"    const uint out_idx = t * 2048u + args.qh * 256u + d;\n"
"    float g = gate[out_idx];\n"
"    attn[out_idx] = ctx_head[tid] / (1.0f + fast::exp(-g));\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_unpack_gate_block_head_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *ctx_head [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device float *attn [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint local_t = tid / 256u;\n"
"    const uint d = tid - local_t * 256u;\n"
"    const uint in_t = args.q_input_start + local_t;\n"
"    const uint out_t = args.q_output_start + local_t;\n"
"    const uint gate_idx = in_t * 2048u + args.qh * 256u + d;\n"
"    const uint out_idx = out_t * 2048u + args.qh * 256u + d;\n"
"    float g = gate[gate_idx];\n"
"    attn[out_idx] = ctx_head[tid] / (1.0f + fast::exp(-g));\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_unpack_gate_group_block_head_mat(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const float *ctx_head [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device float *attn [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 256u;\n"
"    const uint d = tid - row * 256u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint in_t = args.q_input_start + local_t;\n"
"    const uint out_t = args.q_output_start + local_t;\n"
"    const uint gate_idx = in_t * 2048u + qh * 256u + d;\n"
"    const uint out_idx = out_t * 2048u + qh * 256u + d;\n"
"    float g = gate[gate_idx];\n"
"    attn[out_idx] = ctx_head[tid] / (1.0f + fast::exp(-g));\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_unpack_gate_group_block_head_f16(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const half *ctx_head [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device float *attn [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 256u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 256u;\n"
"    const uint d = tid - row * 256u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint in_t = args.q_input_start + local_t;\n"
"    const uint out_t = args.q_output_start + local_t;\n"
"    const uint gate_idx = in_t * 2048u + qh * 256u + d;\n"
"    const uint out_idx = out_t * 2048u + qh * 256u + d;\n"
"    float g = gate[gate_idx];\n"
"    attn[out_idx] = float(ctx_head[tid]) / (1.0f + fast::exp(-g));\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_unpack_gate_group_block_head_f16x4(\n"
"        constant ds4_drafter_metal_head_block_args &args [[buffer(0)]],\n"
"        device const half *ctx_head [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device float *attn [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = 4u * args.q_count * 64u;\n"
"    if (tid >= total) return;\n"
"    const uint row = tid / 64u;\n"
"    const uint d4 = tid - row * 64u;\n"
"    const uint local_qh = row / args.q_count;\n"
"    const uint local_t = row - local_qh * args.q_count;\n"
"    const uint qh = args.kvh * 4u + local_qh;\n"
"    const uint in_t = args.q_input_start + local_t;\n"
"    const uint out_t = args.q_output_start + local_t;\n"
"    const uint gate_base = in_t * 2048u + qh * 256u;\n"
"    const uint out_base = out_t * 2048u + qh * 256u;\n"
"    const device half4 *ctx4 = (const device half4 *)ctx_head;\n"
"    const device float4 *gate4 = (const device float4 *)(gate + gate_base);\n"
"    device float4 *out4 = (device float4 *)(attn + out_base);\n"
"    float4 g = gate4[d4];\n"
"    out4[d4] = float4(ctx4[tid]) / (1.0f + fast::exp(-g));\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv(\n"
"        device const ushort *conv_w [[buffer(0)]],\n"
"        device const float *qkv [[buffer(1)]],\n"
"        device float *conv_state [[buffer(2)]],\n"
"        device float *conv_out [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= 6144u) return;\n"
"    float acc = conv_state[tid] * bf16_to_f32(conv_w[tid * 4u + 0u]);\n"
"    acc += conv_state[6144u + tid] * bf16_to_f32(conv_w[tid * 4u + 1u]);\n"
"    acc += conv_state[12288u + tid] * bf16_to_f32(conv_w[tid * 4u + 2u]);\n"
"    acc += qkv[tid] * bf16_to_f32(conv_w[tid * 4u + 3u]);\n"
"    conv_out[tid] = acc / (1.0f + fast::exp(-acc));\n"
"    conv_state[tid] = conv_state[6144u + tid];\n"
"    conv_state[6144u + tid] = conv_state[12288u + tid];\n"
"    conv_state[12288u + tid] = qkv[tid];\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *conv_w [[buffer(1)]],\n"
"        device const float *qkv [[buffer(2)]],\n"
"        device float *conv_out [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = n_vec * 6144u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 6144u;\n"
"    const uint c = tid - t * 6144u;\n"
"    float acc = 0.0f;\n"
"    if (t >= 3u) acc += qkv[(t - 3u) * 6144u + c] * bf16_to_f32(conv_w[c * 4u + 0u]);\n"
"    if (t >= 2u) acc += qkv[(t - 2u) * 6144u + c] * bf16_to_f32(conv_w[c * 4u + 1u]);\n"
"    if (t >= 1u) acc += qkv[(t - 1u) * 6144u + c] * bf16_to_f32(conv_w[c * 4u + 2u]);\n"
"    acc += qkv[t * 6144u + c] * bf16_to_f32(conv_w[c * 4u + 3u]);\n"
"    conv_out[tid] = acc / (1.0f + fast::exp(-acc));\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_stateful_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *conv_w [[buffer(1)]],\n"
"        device const float *qkv [[buffer(2)]],\n"
"        device float *conv_out [[buffer(3)]],\n"
"        device const float *conv_state [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = n_vec * 6144u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 6144u;\n"
"    const uint c = tid - t * 6144u;\n"
"    float acc = 0.0f;\n"
"    acc += (t >= 3u ? qkv[(t - 3u) * 6144u + c] : conv_state[t * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 0u]);\n"
"    acc += (t >= 2u ? qkv[(t - 2u) * 6144u + c] : conv_state[(t + 1u) * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 1u]);\n"
"    acc += (t >= 1u ? qkv[(t - 1u) * 6144u + c] : conv_state[(t + 2u) * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 2u]);\n"
"    acc += qkv[t * 6144u + c] * bf16_to_f32(conv_w[c * 4u + 3u]);\n"
"    conv_out[tid] = acc / (1.0f + fast::exp(-acc));\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_qkvz_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *conv_w [[buffer(1)]],\n"
"        device const float *qkvz [[buffer(2)]],\n"
"        device float *conv_out [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = n_vec * 6144u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 6144u;\n"
"    const uint c = tid - t * 6144u;\n"
"    float acc = 0.0f;\n"
"    if (t >= 3u) acc += qkvz[(t - 3u) * 8192u + c] * bf16_to_f32(conv_w[c * 4u + 0u]);\n"
"    if (t >= 2u) acc += qkvz[(t - 2u) * 8192u + c] * bf16_to_f32(conv_w[c * 4u + 1u]);\n"
"    if (t >= 1u) acc += qkvz[(t - 1u) * 8192u + c] * bf16_to_f32(conv_w[c * 4u + 2u]);\n"
"    acc += qkvz[t * 8192u + c] * bf16_to_f32(conv_w[c * 4u + 3u]);\n"
"    conv_out[tid] = acc / (1.0f + fast::exp(-acc));\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_stateful_qkvz_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *conv_w [[buffer(1)]],\n"
"        device const float *qkvz [[buffer(2)]],\n"
"        device float *conv_out [[buffer(3)]],\n"
"        device const float *conv_state [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    const uint total = n_vec * 6144u;\n"
"    if (tid >= total) return;\n"
"    const uint t = tid / 6144u;\n"
"    const uint c = tid - t * 6144u;\n"
"    float acc = 0.0f;\n"
"    acc += (t >= 3u ? qkvz[(t - 3u) * 8192u + c] : conv_state[t * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 0u]);\n"
"    acc += (t >= 2u ? qkvz[(t - 2u) * 8192u + c] : conv_state[(t + 1u) * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 1u]);\n"
"    acc += (t >= 1u ? qkvz[(t - 1u) * 8192u + c] : conv_state[(t + 2u) * 6144u + c]) * bf16_to_f32(conv_w[c * 4u + 2u]);\n"
"    acc += qkvz[t * 8192u + c] * bf16_to_f32(conv_w[c * 4u + 3u]);\n"
"    conv_out[tid] = acc / (1.0f + fast::exp(-acc));\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_qk_norm(\n"
"        device const float *conv_out [[buffer(0)]],\n"
"        device float *q_norm [[buffer(1)]],\n"
"        device float *k_norm [[buffer(2)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint h = tg.x;\n"
"    const uint section = tg.y;\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    if (h >= 16u || section >= 2u) return;\n"
"    const uint base = (section == 0u ? h * 128u : 2048u + h * 128u);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = conv_out[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f);\n"
"    float scale = section == 0u ? 0.0078125f : 0.08838834764831845f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = conv_out[base + i] * inv * scale;\n"
"        if (section == 0u) q_norm[h * 128u + i] = v;\n"
"        else k_norm[h * 128u + i] = v;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_qk_norm_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *conv_out [[buffer(1)]],\n"
"        device float *q_norm [[buffer(2)]],\n"
"        device float *k_norm [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint h = gid.x;\n"
"    const uint section = gid.y;\n"
"    const uint t = gid.z;\n"
"    if (h >= 16u || section >= 2u || t >= n_vec) return;\n"
"    const uint in_base = t * 6144u + (section == 0u ? h * 128u : 2048u + h * 128u);\n"
"    const uint out_base = t * 2048u + h * 128u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = conv_out[in_base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f);\n"
"    float scale = section == 0u ? 0.0078125f : 0.08838834764831845f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = conv_out[in_base + i] * inv * scale;\n"
"        if (section == 0u) q_norm[out_base + i] = v;\n"
"        else k_norm[out_base + i] = v;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_qk_norm_kq_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *conv_out [[buffer(1)]],\n"
"        device float *q_norm [[buffer(2)]],\n"
"        device float *k_norm [[buffer(3)]],\n"
"        device float *kq [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 gid [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = gid.x;\n"
"    const uint t = gid.y;\n"
"    if (h >= 16u || t >= n_vec) return;\n"
"    const uint q_in_base = t * 6144u + h * 128u;\n"
"    const uint k_in_base = t * 6144u + 2048u + h * 128u;\n"
"    const uint out_base = t * 2048u + h * 128u;\n"
"    float qss = 0.0f;\n"
"    float kss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float qv = conv_out[q_in_base + i];\n"
"        float kv = conv_out[k_in_base + i];\n"
"        qss += qv * qv;\n"
"        kss += kv * kv;\n"
"    }\n"
"    scratch[tid] = qss;\n"
"    scratch[nt + tid] = kss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[tid] += scratch[tid + stride];\n"
"            scratch[nt + tid] += scratch[nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float q_inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f) * 0.0078125f;\n"
"    float k_inv = rsqrt((scratch[nt] / 128.0f) + 1.0e-6f) * 0.08838834764831845f;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float dot = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float qv = conv_out[q_in_base + i] * q_inv;\n"
"        float kv = conv_out[k_in_base + i] * k_inv;\n"
"        q_norm[out_base + i] = qv;\n"
"        k_norm[out_base + i] = kv;\n"
"        dot += qv * kv;\n"
"    }\n"
"    scratch[tid] = dot;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) kq[t * 16u + h] = scratch[0];\n"
"}\n"
"\n"
"static inline float ds4_softplus(float x) {\n"
"    if (x > 20.0f) return x;\n"
"    if (x < -20.0f) return exp(x);\n"
"    return log(1.0f + exp(x));\n"
"}\n"
"\n"
"static inline float ds4_decay_from_params(float a_log, float a, float dt_bias) {\n"
"    float sp = ds4_softplus(a + dt_bias);\n"
"    if (sp <= 0.0f) return 1.0f;\n"
"    float log_rate = clamp(a_log, -20.0f, 20.0f);\n"
"    return exp(-exp(log_rate) * sp);\n"
"}\n"
"\n"
"static inline float ds4_drafter_reduce_sum(float v, threadgroup float *scratch, uint tid, uint nt) {\n"
"    if (nt == 32u) return simd_sum(v);\n"
"    scratch[tid] = v;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float out = scratch[0];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    return out;\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_scan_params(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *a_log [[buffer(1)]],\n"
"        device const ushort *dt_bias [[buffer(2)]],\n"
"        device const float *b [[buffer(3)]],\n"
"        device const float *a [[buffer(4)]],\n"
"        device float2 *params [[buffer(5)]],\n"
"        uint idx [[thread_position_in_grid]]) {\n"
"    if (idx >= n_vec * 16u) return;\n"
"    const uint h = idx % 16u;\n"
"    float beta = 1.0f / (1.0f + fast::exp(-b[idx]));\n"
"    float decay = ds4_decay_from_params(a_log[h], a[idx], bf16_to_f32(dt_bias[h]));\n"
"    params[idx] = float2(beta, decay);\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_delta(\n"
"        device const float *a_log [[buffer(0)]],\n"
"        device const ushort *dt_bias [[buffer(1)]],\n"
"        device const float *b [[buffer(2)]],\n"
"        device const float *a [[buffer(3)]],\n"
"        device const float *conv_out [[buffer(4)]],\n"
"        device const float *q_norm [[buffer(5)]],\n"
"        device const float *k_norm [[buffer(6)]],\n"
"        device float *delta_state [[buffer(7)]],\n"
"        device float *y [[buffer(8)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint h = tg.x;\n"
"    const uint dv = tg.y;\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    if (h >= 16u || dv >= 128u) return;\n"
"    float beta = 1.0f / (1.0f + fast::exp(-b[h]));\n"
"    float decay = ds4_decay_from_params(a_log[h], a[h], bf16_to_f32(dt_bias[h]));\n"
"    const uint hk = h * 128u;\n"
"    const uint state_base = (h * 128u + dv) * 128u;\n"
"    float kv = 0.0f;\n"
"    for (uint dk = tid; dk < 128u; dk += nt) {\n"
"        float st = delta_state[state_base + dk] * decay;\n"
"        delta_state[state_base + dk] = st;\n"
"        kv += st * k_norm[hk + dk];\n"
"    }\n"
"    float kv_sum = ds4_drafter_reduce_sum(kv, scratch, tid, nt);\n"
"    float delta = (conv_out[4096u + hk + dv] - kv_sum) * beta;\n"
"    float out_acc = 0.0f;\n"
"    for (uint dk = tid; dk < 128u; dk += nt) {\n"
"        float st = delta_state[state_base + dk] + k_norm[hk + dk] * delta;\n"
"        delta_state[state_base + dk] = st;\n"
"        out_acc += st * q_norm[hk + dk];\n"
"    }\n"
"    float out_sum = ds4_drafter_reduce_sum(out_acc, scratch, tid, nt);\n"
"    if (tid == 0) y[hk + dv] = out_sum;\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_delta_scan(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *a_log [[buffer(1)]],\n"
"        device const ushort *dt_bias [[buffer(2)]],\n"
"        device const float *b [[buffer(3)]],\n"
"        device const float *a [[buffer(4)]],\n"
"        device const float *conv_out [[buffer(5)]],\n"
"        device const float *q_norm [[buffer(6)]],\n"
"        device const float *k_norm [[buffer(7)]],\n"
"        device float *delta_state [[buffer(8)]],\n"
"        device float *y [[buffer(9)]],\n"
"        device const float2 *scan_params [[buffer(10)]],\n"
"        device const float *kq [[buffer(11)]],\n"
"        constant uint &debug_enabled [[buffer(12)]],\n"
"        device atomic_uint *debug_flags [[buffer(13)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = tg.x;\n"
"    const uint dv = tg.y;\n"
"    if (h >= 16u || dv >= 128u) return;\n"
"    const uint state_base = (h * 128u + dv) * 128u;\n"
"    const uint h128 = h * 128u;\n"
"    if (nt == 32u) {\n"
"        uint dk0 = tid;\n"
"        uint dk1 = tid + 32u;\n"
"        uint dk2 = tid + 64u;\n"
"        uint dk3 = tid + 96u;\n"
"        float st0 = delta_state[state_base + dk0];\n"
"        float st1 = delta_state[state_base + dk1];\n"
"        float st2 = delta_state[state_base + dk2];\n"
"        float st3 = delta_state[state_base + dk3];\n"
"        for (uint t = 0; t < n_vec; ++t) {\n"
"            const uint ba_base = t * 16u + h;\n"
"            const uint qk_base = t * 2048u + h128;\n"
"            float2 beta_decay = scan_params[ba_base];\n"
"            float beta = beta_decay.x;\n"
"            float decay = beta_decay.y;\n"
"            float k0 = k_norm[qk_base + dk0];\n"
"            float k1 = k_norm[qk_base + dk1];\n"
"            float k2 = k_norm[qk_base + dk2];\n"
"            float k3 = k_norm[qk_base + dk3];\n"
"            float q0 = q_norm[qk_base + dk0];\n"
"            float q1 = q_norm[qk_base + dk1];\n"
"            float q2 = q_norm[qk_base + dk2];\n"
"            float q3 = q_norm[qk_base + dk3];\n"
"            st0 *= decay;\n"
"            st1 *= decay;\n"
"            st2 *= decay;\n"
"            st3 *= decay;\n"
"            float kv = st0 * k0 + st1 * k1 + st2 * k2 + st3 * k3;\n"
"            float state_q = st0 * q0 + st1 * q1 + st2 * q2 + st3 * q3;\n"
"            float kv_sum = simd_sum(kv);\n"
"            float state_q_sum = simd_sum(state_q);\n"
"            float delta = (conv_out[t * 6144u + 4096u + h128 + dv] - kv_sum) * beta;\n"
"            st0 += k0 * delta;\n"
"            st1 += k1 * delta;\n"
"            st2 += k2 * delta;\n"
"            st3 += k3 * delta;\n"
"            float out_sum = state_q_sum + delta * kq[ba_base];\n"
"            if (tid == 0) y[t * 2048u + h128 + dv] = out_sum;\n"
"        }\n"
"        delta_state[state_base + dk0] = st0;\n"
"        delta_state[state_base + dk1] = st1;\n"
"        delta_state[state_base + dk2] = st2;\n"
"        delta_state[state_base + dk3] = st3;\n"
"        return;\n"
"    }\n"
"    for (uint t = 0; t < n_vec; ++t) {\n"
"        const uint ba_base = t * 16u + h;\n"
"        const uint qk_base = t * 2048u + h128;\n"
"        float2 beta_decay = scan_params[ba_base];\n"
"        float beta = beta_decay.x;\n"
"        float decay = beta_decay.y;\n"
"        if (debug_enabled != 0u) {\n"
"            uint flags = (!isfinite(beta) ? 1u : 0u) | (!isfinite(decay) ? 2u : 0u);\n"
"            if (flags != 0u) atomic_fetch_or_explicit(&debug_flags[0], flags, memory_order_relaxed);\n"
"        }\n"
"        float kv = 0.0f;\n"
"        float state_q = 0.0f;\n"
"        for (uint dk = tid; dk < 128u; dk += nt) {\n"
"            float k = k_norm[qk_base + dk];\n"
"            float q = q_norm[qk_base + dk];\n"
"            float st = delta_state[state_base + dk] * decay;\n"
"            if (debug_enabled != 0u) {\n"
"                uint flags = (!isfinite(k) ? 4u : 0u) | (!isfinite(q) ? 8u : 0u) | (!isfinite(st) ? 16u : 0u);\n"
"                if (flags != 0u) atomic_fetch_or_explicit(&debug_flags[0], flags, memory_order_relaxed);\n"
"            }\n"
"            delta_state[state_base + dk] = st;\n"
"            kv += st * k;\n"
"            state_q += st * q;\n"
"        }\n"
"        float kv_sum = ds4_drafter_reduce_sum(kv, scratch, tid, nt);\n"
"        float state_q_sum = ds4_drafter_reduce_sum(state_q, scratch, tid, nt);\n"
"        float kq_sum = kq[ba_base];\n"
"        float conv_v = conv_out[t * 6144u + 4096u + h128 + dv];\n"
"        float delta = (conv_v - kv_sum) * beta;\n"
"        if (debug_enabled != 0u) {\n"
"            uint flags = (!isfinite(kv_sum) ? 32u : 0u) | (!isfinite(state_q_sum) ? 64u : 0u) |\n"
"                         (!isfinite(kq_sum) ? 128u : 0u) | (!isfinite(conv_v) ? 256u : 0u) |\n"
"                         (!isfinite(delta) ? 512u : 0u);\n"
"            if (flags != 0u) atomic_fetch_or_explicit(&debug_flags[0], flags, memory_order_relaxed);\n"
"        }\n"
"        for (uint dk = tid; dk < 128u; dk += nt) {\n"
"            float k = k_norm[qk_base + dk];\n"
"            float st = delta_state[state_base + dk] + k * delta;\n"
"            if (debug_enabled != 0u) {\n"
"                uint flags = (!isfinite(k) ? 4u : 0u) | (!isfinite(st) ? 1024u : 0u);\n"
"                if (flags != 0u) atomic_fetch_or_explicit(&debug_flags[0], flags, memory_order_relaxed);\n"
"            }\n"
"            delta_state[state_base + dk] = st;\n"
"        }\n"
"        float out_sum = state_q_sum + delta * kq_sum;\n"
"        if (debug_enabled != 0u && !isfinite(out_sum)) {\n"
"            atomic_fetch_or_explicit(&debug_flags[0], 2048u, memory_order_relaxed);\n"
"        }\n"
"        if (tid == 0) y[t * 2048u + h128 + dv] = out_sum;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_delta_scan2(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *a_log [[buffer(1)]],\n"
"        device const ushort *dt_bias [[buffer(2)]],\n"
"        device const float *b [[buffer(3)]],\n"
"        device const float *a [[buffer(4)]],\n"
"        device const float *conv_out [[buffer(5)]],\n"
"        device const float *q_norm [[buffer(6)]],\n"
"        device const float *k_norm [[buffer(7)]],\n"
"        device float *delta_state [[buffer(8)]],\n"
"        device float *y [[buffer(9)]],\n"
"        device const float2 *scan_params [[buffer(10)]],\n"
"        device const float *kq [[buffer(11)]],\n"
"        constant uint &debug_enabled [[buffer(12)]],\n"
"        device atomic_uint *debug_flags [[buffer(13)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = tg.x;\n"
"    const uint dv0 = tg.y * 2u;\n"
"    if (h >= 16u || dv0 >= 128u || nt != 32u || debug_enabled != 0u) return;\n"
"    const uint dv1 = dv0 + 1u;\n"
"    const uint h128 = h * 128u;\n"
"    const uint dk0 = tid;\n"
"    const uint dk1 = tid + 32u;\n"
"    const uint dk2 = tid + 64u;\n"
"    const uint dk3 = tid + 96u;\n"
"    const uint state_base0 = (h128 + dv0) * 128u;\n"
"    const uint state_base1 = (h128 + dv1) * 128u;\n"
"    float s00 = delta_state[state_base0 + dk0];\n"
"    float s01 = delta_state[state_base0 + dk1];\n"
"    float s02 = delta_state[state_base0 + dk2];\n"
"    float s03 = delta_state[state_base0 + dk3];\n"
"    float s10 = delta_state[state_base1 + dk0];\n"
"    float s11 = delta_state[state_base1 + dk1];\n"
"    float s12 = delta_state[state_base1 + dk2];\n"
"    float s13 = delta_state[state_base1 + dk3];\n"
"    for (uint t = 0; t < n_vec; ++t) {\n"
"        const uint ba_base = t * 16u + h;\n"
"        const uint qk_base = t * 2048u + h128;\n"
"        const uint conv_base = t * 6144u + 4096u + h128 + dv0;\n"
"        float2 beta_decay = scan_params[ba_base];\n"
"        float beta = beta_decay.x;\n"
"        float decay = beta_decay.y;\n"
"        float k0 = k_norm[qk_base + dk0];\n"
"        float k1 = k_norm[qk_base + dk1];\n"
"        float k2 = k_norm[qk_base + dk2];\n"
"        float k3 = k_norm[qk_base + dk3];\n"
"        float q0 = q_norm[qk_base + dk0];\n"
"        float q1 = q_norm[qk_base + dk1];\n"
"        float q2 = q_norm[qk_base + dk2];\n"
"        float q3 = q_norm[qk_base + dk3];\n"
"        float kq_sum = kq[ba_base];\n"
"        s00 *= decay;\n"
"        s01 *= decay;\n"
"        s02 *= decay;\n"
"        s03 *= decay;\n"
"        float kv0 = s00 * k0 + s01 * k1 + s02 * k2 + s03 * k3;\n"
"        float sq0 = s00 * q0 + s01 * q1 + s02 * q2 + s03 * q3;\n"
"        float kv_sum0 = simd_sum(kv0);\n"
"        float sq_sum0 = simd_sum(sq0);\n"
"        float delta0 = (conv_out[conv_base] - kv_sum0) * beta;\n"
"        s00 += k0 * delta0;\n"
"        s01 += k1 * delta0;\n"
"        s02 += k2 * delta0;\n"
"        s03 += k3 * delta0;\n"
"        s10 *= decay;\n"
"        s11 *= decay;\n"
"        s12 *= decay;\n"
"        s13 *= decay;\n"
"        float kv1 = s10 * k0 + s11 * k1 + s12 * k2 + s13 * k3;\n"
"        float sq1 = s10 * q0 + s11 * q1 + s12 * q2 + s13 * q3;\n"
"        float kv_sum1 = simd_sum(kv1);\n"
"        float sq_sum1 = simd_sum(sq1);\n"
"        float delta1 = (conv_out[conv_base + 1u] - kv_sum1) * beta;\n"
"        s10 += k0 * delta1;\n"
"        s11 += k1 * delta1;\n"
"        s12 += k2 * delta1;\n"
"        s13 += k3 * delta1;\n"
"        if (tid == 0) {\n"
"            y[t * 2048u + h128 + dv0] = sq_sum0 + delta0 * kq_sum;\n"
"            y[t * 2048u + h128 + dv1] = sq_sum1 + delta1 * kq_sum;\n"
"        }\n"
"    }\n"
"    delta_state[state_base0 + dk0] = s00;\n"
"    delta_state[state_base0 + dk1] = s01;\n"
"    delta_state[state_base0 + dk2] = s02;\n"
"    delta_state[state_base0 + dk3] = s03;\n"
"    delta_state[state_base1 + dk0] = s10;\n"
"    delta_state[state_base1 + dk1] = s11;\n"
"    delta_state[state_base1 + dk2] = s12;\n"
"    delta_state[state_base1 + dk3] = s13;\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_delta_scan4(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *a_log [[buffer(1)]],\n"
"        device const ushort *dt_bias [[buffer(2)]],\n"
"        device const float *b [[buffer(3)]],\n"
"        device const float *a [[buffer(4)]],\n"
"        device const float *conv_out [[buffer(5)]],\n"
"        device const float *q_norm [[buffer(6)]],\n"
"        device const float *k_norm [[buffer(7)]],\n"
"        device float *delta_state [[buffer(8)]],\n"
"        device float *y [[buffer(9)]],\n"
"        device const float2 *scan_params [[buffer(10)]],\n"
"        device const float *kq [[buffer(11)]],\n"
"        constant uint &debug_enabled [[buffer(12)]],\n"
"        device atomic_uint *debug_flags [[buffer(13)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = tg.x;\n"
"    const uint dv_base = tg.y * 4u;\n"
"    if (h >= 16u || dv_base >= 128u || nt != 32u || debug_enabled != 0u) return;\n"
"    const uint h128 = h * 128u;\n"
"    const uint dk0 = tid;\n"
"    const uint dk1 = tid + 32u;\n"
"    const uint dk2 = tid + 64u;\n"
"    const uint dk3 = tid + 96u;\n"
"    float s0[4];\n"
"    float s1[4];\n"
"    float s2[4];\n"
"    float s3[4];\n"
"    for (uint j = 0; j < 4u; ++j) {\n"
"        const uint state_base = (h128 + dv_base + j) * 128u;\n"
"        s0[j] = delta_state[state_base + dk0];\n"
"        s1[j] = delta_state[state_base + dk1];\n"
"        s2[j] = delta_state[state_base + dk2];\n"
"        s3[j] = delta_state[state_base + dk3];\n"
"    }\n"
"    for (uint t = 0; t < n_vec; ++t) {\n"
"        const uint ba_base = t * 16u + h;\n"
"        const uint qk_base = t * 2048u + h128;\n"
"        const uint conv_base = t * 6144u + 4096u + h128 + dv_base;\n"
"        float2 beta_decay = scan_params[ba_base];\n"
"        float beta = beta_decay.x;\n"
"        float decay = beta_decay.y;\n"
"        float k0 = k_norm[qk_base + dk0];\n"
"        float k1 = k_norm[qk_base + dk1];\n"
"        float k2 = k_norm[qk_base + dk2];\n"
"        float k3 = k_norm[qk_base + dk3];\n"
"        float q0 = q_norm[qk_base + dk0];\n"
"        float q1 = q_norm[qk_base + dk1];\n"
"        float q2 = q_norm[qk_base + dk2];\n"
"        float q3 = q_norm[qk_base + dk3];\n"
"        float kq_sum = kq[ba_base];\n"
"        for (uint j = 0; j < 4u; ++j) {\n"
"            s0[j] *= decay;\n"
"            s1[j] *= decay;\n"
"            s2[j] *= decay;\n"
"            s3[j] *= decay;\n"
"            float kv = s0[j] * k0 + s1[j] * k1 + s2[j] * k2 + s3[j] * k3;\n"
"            float sq = s0[j] * q0 + s1[j] * q1 + s2[j] * q2 + s3[j] * q3;\n"
"            float kv_sum = simd_sum(kv);\n"
"            float sq_sum = simd_sum(sq);\n"
"            float delta = (conv_out[conv_base + j] - kv_sum) * beta;\n"
"            s0[j] += k0 * delta;\n"
"            s1[j] += k1 * delta;\n"
"            s2[j] += k2 * delta;\n"
"            s3[j] += k3 * delta;\n"
"            if (tid == 0) y[t * 2048u + h128 + dv_base + j] = sq_sum + delta * kq_sum;\n"
"        }\n"
"    }\n"
"    for (uint j = 0; j < 4u; ++j) {\n"
"        const uint state_base = (h128 + dv_base + j) * 128u;\n"
"        delta_state[state_base + dk0] = s0[j];\n"
"        delta_state[state_base + dk1] = s1[j];\n"
"        delta_state[state_base + dk2] = s2[j];\n"
"        delta_state[state_base + dk3] = s3[j];\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_state_from_qkv_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *qkv [[buffer(1)]],\n"
"        device float *conv_state [[buffer(2)]],\n"
"        uint idx [[thread_position_in_grid]]) {\n"
"    if (idx >= 3u * 6144u) return;\n"
"    const uint slot = idx / 6144u;\n"
"    const uint col = idx - slot * 6144u;\n"
"    const uint keep = n_vec < 3u ? n_vec : 3u;\n"
"    if (slot < 3u - keep || n_vec == 0u) {\n"
"        conv_state[idx] = 0.0f;\n"
"        return;\n"
"    }\n"
"    const uint src_t = n_vec - keep + (slot - (3u - keep));\n"
"    conv_state[idx] = qkv[src_t * 6144u + col];\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_conv_state_from_qkvz_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const float *qkvz [[buffer(1)]],\n"
"        device float *conv_state [[buffer(2)]],\n"
"        uint idx [[thread_position_in_grid]]) {\n"
"    if (idx >= 3u * 6144u) return;\n"
"    const uint slot = idx / 6144u;\n"
"    const uint col = idx - slot * 6144u;\n"
"    const uint keep = n_vec < 3u ? n_vec : 3u;\n"
"    if (slot < 3u - keep || n_vec == 0u) {\n"
"        conv_state[idx] = 0.0f;\n"
"        return;\n"
"    }\n"
"    const uint src_t = n_vec - keep + (slot - (3u - keep));\n"
"    conv_state[idx] = qkvz[src_t * 8192u + col];\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_gate(\n"
"        device const ushort *norm_w [[buffer(0)]],\n"
"        device const float *z [[buffer(1)]],\n"
"        device const float *y [[buffer(2)]],\n"
"        device float *gated [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint h [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (h >= 16u) return;\n"
"    const uint base = h * 128u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = y[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f);\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        uint idx = base + i;\n"
"        float zv = z[idx];\n"
"        float silu = zv / (1.0f + fast::exp(-zv));\n"
"        gated[idx] = silu * y[idx] * inv * bf16_to_f32(norm_w[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_gate_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *norm_w [[buffer(1)]],\n"
"        device const float *z [[buffer(2)]],\n"
"        device const float *y [[buffer(3)]],\n"
"        device float *gated [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = tg.x;\n"
"    const uint t = tg.y;\n"
"    if (h >= 16u || t >= n_vec) return;\n"
"    const uint base = t * 2048u + h * 128u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = y[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f);\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        uint idx = base + i;\n"
"        float zv = z[idx];\n"
"        float silu = zv / (1.0f + fast::exp(-zv));\n"
"        gated[idx] = silu * y[idx] * inv * bf16_to_f32(norm_w[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_linear_gate_qkvz_mat(\n"
"        constant uint &n_vec [[buffer(0)]],\n"
"        device const ushort *norm_w [[buffer(1)]],\n"
"        device const float *qkvz [[buffer(2)]],\n"
"        device const float *y [[buffer(3)]],\n"
"        device float *gated [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint2 tg [[threadgroup_position_in_grid]],\n"
"        uint2 tid2 [[thread_position_in_threadgroup]],\n"
"        uint2 nt2 [[threads_per_threadgroup]]) {\n"
"    const uint tid = tid2.x;\n"
"    const uint nt = nt2.x;\n"
"    const uint h = tg.x;\n"
"    const uint t = tg.y;\n"
"    if (h >= 16u || t >= n_vec) return;\n"
"    const uint base = t * 2048u + h * 128u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        float v = y[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = rsqrt((scratch[0] / 128.0f) + 1.0e-6f);\n"
"    const uint z_base = t * 8192u + 6144u + h * 128u;\n"
"    for (uint i = tid; i < 128u; i += nt) {\n"
"        uint idx = base + i;\n"
"        float zv = qkvz[z_base + i];\n"
"        float silu = zv / (1.0f + fast::exp(-zv));\n"
"        gated[idx] = silu * y[idx] * inv * bf16_to_f32(norm_w[i]);\n"
"    }\n"
"}\n"
"\n"
"struct ds4_drafter_metal_rope_args {\n"
"    int position;\n"
"    int n_vec;\n"
"};\n"
"\n"
"static inline void qwen_rope_pair(float a, float b, uint i, int position,\n"
"                                 thread float &out_a, thread float &out_b) {\n"
"    float freq = pow(10000000.0f, float(2u * i) / 64.0f);\n"
"    float angle = float(position) / freq;\n"
"    float c = cos(angle);\n"
"    float s = sin(angle);\n"
"    out_a = a * c - b * s;\n"
"    out_b = b * c + a * s;\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_q_norm_rope(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *q_proj [[buffer(2)]],\n"
"        device float *queries [[buffer(3)]],\n"
"        device float *gate [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint qh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (qh >= 8u) return;\n"
"    const uint q_base = qh * 512u;\n"
"    const uint out_base = qh * 256u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = q_proj[q_base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        gate[out_base + i] = q_proj[q_base + 256u + i];\n"
"    }\n"
"    if (tid < 32u) {\n"
"        float a = q_proj[q_base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = q_proj[q_base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, args.position, ra, rb);\n"
"        queries[out_base + tid] = ra;\n"
"        queries[out_base + tid + 32u] = rb;\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        queries[out_base + i] = q_proj[q_base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_q_only_norm_rope(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *q_proj [[buffer(2)]],\n"
"        device float *queries [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint qh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (qh >= 8u) return;\n"
"    const uint q_base = qh * 256u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = q_proj[q_base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    if (tid < 32u) {\n"
"        float a = q_proj[q_base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = q_proj[q_base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, args.position, ra, rb);\n"
"        queries[q_base + tid] = ra;\n"
"        queries[q_base + tid + 32u] = rb;\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        queries[q_base + i] = q_proj[q_base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_q_norm_rope_mat(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *q_proj [[buffer(2)]],\n"
"        device float *queries [[buffer(3)]],\n"
"        device float *gate [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = gid.x;\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (qh >= 8u) return;\n"
"    const uint q_base = vec * 4096u + qh * 512u;\n"
"    const uint out_base = vec * 2048u + qh * 256u;\n"
"    const int position = args.position + int(vec);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = q_proj[q_base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        gate[out_base + i] = q_proj[q_base + 256u + i];\n"
"    }\n"
"    if (tid < 32u) {\n"
"        float a = q_proj[q_base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = q_proj[q_base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, position, ra, rb);\n"
"        queries[out_base + tid] = ra;\n"
"        queries[out_base + tid + 32u] = rb;\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        queries[out_base + i] = q_proj[q_base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_k_norm_rope(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *k_proj [[buffer(2)]],\n"
"        device float *keys [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint kh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (kh >= 2u) return;\n"
"    const uint base = kh * 256u;\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = k_proj[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    if (tid < 32u) {\n"
"        float a = k_proj[base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = k_proj[base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, args.position, ra, rb);\n"
"        keys[base + tid] = ra;\n"
"        keys[base + tid + 32u] = rb;\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        keys[base + i] = k_proj[base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_k_norm_rope_mat(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *k_proj [[buffer(2)]],\n"
"        device float *keys [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint kh = gid.x;\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (kh >= 2u) return;\n"
"    const uint base = vec * 512u + kh * 256u;\n"
"    const int position = args.position + int(vec);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = k_proj[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    if (tid < 32u) {\n"
"        float a = k_proj[base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = k_proj[base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, position, ra, rb);\n"
"        keys[base + tid] = ra;\n"
"        keys[base + tid + 32u] = rb;\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        keys[base + i] = k_proj[base + i] * inv * bf16_to_f32(weight[i]);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_full_k_norm_rope_mat_pack_f16(\n"
"        constant ds4_drafter_metal_rope_args &args [[buffer(0)]],\n"
"        device const ushort *weight [[buffer(1)]],\n"
"        device const float *k_proj [[buffer(2)]],\n"
"        device float *keys [[buffer(3)]],\n"
"        device half *k_head [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint kh = gid.x;\n"
"    const uint vec = gid.y;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    if (kh >= 2u) return;\n"
"    const uint base = vec * 512u + kh * 256u;\n"
"    const uint packed_base = (kh * uint(args.n_vec) + vec) * 256u;\n"
"    const int position = args.position + int(vec);\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        float v = k_proj[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    scratch[tid] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv = rsqrt((scratch[0] / 256.0f) + 1.0e-6f);\n"
"    if (tid < 32u) {\n"
"        float a = k_proj[base + tid] * inv * bf16_to_f32(weight[tid]);\n"
"        float b = k_proj[base + tid + 32u] * inv * bf16_to_f32(weight[tid + 32u]);\n"
"        float ra;\n"
"        float rb;\n"
"        qwen_rope_pair(a, b, tid, position, ra, rb);\n"
"        keys[base + tid] = ra;\n"
"        keys[base + tid + 32u] = rb;\n"
"        k_head[packed_base + tid] = half(ra);\n"
"        k_head[packed_base + tid + 32u] = half(rb);\n"
"    }\n"
"    for (uint i = tid + 64u; i < 256u; i += nt) {\n"
"        float y = k_proj[base + i] * inv * bf16_to_f32(weight[i]);\n"
"        keys[base + i] = y;\n"
"        k_head[packed_base + i] = half(y);\n"
"    }\n"
"}\n"
"\n"
"struct ds4_drafter_metal_attention_args {\n"
"    int n_ctx;\n"
"    float scale;\n"
"};\n"
"\n"
"kernel void ds4_drafter_attention_logits(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device float *logits [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint t = tg.y;\n"
"    if (qh >= 8u || t >= uint(args.n_ctx)) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = qh * 256u;\n"
"    const uint k_base = (t * 2u + kvh) * 256u;\n"
"    float acc = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        acc += queries[q_base + i] * keys[k_base + i];\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) logits[qh * uint(args.n_ctx) + t] = scratch[0] * args.scale;\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_logits4(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device float *logits [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint t_base = tg.y * 4u;\n"
"    if (qh >= 8u || t_base >= uint(args.n_ctx)) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = qh * 256u;\n"
"    float acc0 = 0.0f;\n"
"    float acc1 = 0.0f;\n"
"    float acc2 = 0.0f;\n"
"    float acc3 = 0.0f;\n"
"    const bool v0 = t_base < n_ctx;\n"
"    const bool v1 = t_base + 1u < n_ctx;\n"
"    const bool v2 = t_base + 2u < n_ctx;\n"
"    const bool v3 = t_base + 3u < n_ctx;\n"
"    const uint k0 = ((t_base + 0u) * 2u + kvh) * 256u;\n"
"    const uint k1 = ((t_base + 1u) * 2u + kvh) * 256u;\n"
"    const uint k2 = ((t_base + 2u) * 2u + kvh) * 256u;\n"
"    const uint k3 = ((t_base + 3u) * 2u + kvh) * 256u;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        const float q = queries[q_base + i];\n"
"        if (v0) acc0 += q * keys[k0 + i];\n"
"        if (v1) acc1 += q * keys[k1 + i];\n"
"        if (v2) acc2 += q * keys[k2 + i];\n"
"        if (v3) acc3 += q * keys[k3 + i];\n"
"    }\n"
"    scratch[0u * nt + tid] = acc0;\n"
"    scratch[1u * nt + tid] = acc1;\n"
"    scratch[2u * nt + tid] = acc2;\n"
"    scratch[3u * nt + tid] = acc3;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[0u * nt + tid] += scratch[0u * nt + tid + stride];\n"
"            scratch[1u * nt + tid] += scratch[1u * nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        const uint out_base = qh * n_ctx + t_base;\n"
"        if (v0) logits[out_base + 0u] = scratch[0u] * args.scale;\n"
"        if (v1) logits[out_base + 1u] = scratch[1u * nt] * args.scale;\n"
"        if (v2) logits[out_base + 2u] = scratch[2u * nt] * args.scale;\n"
"        if (v3) logits[out_base + 3u] = scratch[3u * nt] * args.scale;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *logits [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint qh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (qh >= 8u) return;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_max = max(local_max, logits[logit_base + t]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_denom += fast::exp(logits[logit_base + t] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    if (tid < 256u) {\n"
"        const uint kvh = qh / 4u;\n"
"        float acc = 0.0f;\n"
"        for (uint t = 0; t < n_ctx; t++) {\n"
"            float p = exp(logits[logit_base + t] - max_logit) * inv_denom;\n"
"            acc += p * values[(t * 2u + kvh) * 256u + tid];\n"
"        }\n"
"        float g = gate[qh * 256u + tid];\n"
"        attn[qh * 256u + tid] = acc / (1.0f + fast::exp(-g));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_softmax_inplace(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint qh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (qh >= 8u) return;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_max = max(local_max, logits[logit_base + t]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_denom += fast::exp(logits[logit_base + t] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        logits[logit_base + t] = exp(logits[logit_base + t] - max_logit) * inv_denom;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context_parallel(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *probs [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint dim = tg.y;\n"
"    if (qh >= 8u || dim >= 256u) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    const uint kvh = qh / 4u;\n"
"    float acc = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        acc += probs[logit_base + t] * values[(t * 2u + kvh) * 256u + dim];\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        float g = gate[qh * 256u + dim];\n"
"        attn[qh * 256u + dim] = scratch[0] / (1.0f + fast::exp(-g));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context4(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *probs [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint dim_base = tg.y * 4u;\n"
"    if (qh >= 8u || dim_base >= 256u) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    const uint kvh = qh / 4u;\n"
"    float acc0 = 0.0f;\n"
"    float acc1 = 0.0f;\n"
"    float acc2 = 0.0f;\n"
"    float acc3 = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        const float p = probs[logit_base + t];\n"
"        const uint v_base = (t * 2u + kvh) * 256u + dim_base;\n"
"        acc0 += p * values[v_base + 0u];\n"
"        acc1 += p * values[v_base + 1u];\n"
"        acc2 += p * values[v_base + 2u];\n"
"        acc3 += p * values[v_base + 3u];\n"
"    }\n"
"    scratch[0u * nt + tid] = acc0;\n"
"    scratch[1u * nt + tid] = acc1;\n"
"    scratch[2u * nt + tid] = acc2;\n"
"    scratch[3u * nt + tid] = acc3;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[0u * nt + tid] += scratch[0u * nt + tid + stride];\n"
"            scratch[1u * nt + tid] += scratch[1u * nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        const uint out_base = qh * 256u + dim_base;\n"
"        attn[out_base + 0u] = scratch[0u] / (1.0f + fast::exp(-gate[out_base + 0u]));\n"
"        attn[out_base + 1u] = scratch[1u * nt] / (1.0f + fast::exp(-gate[out_base + 1u]));\n"
"        attn[out_base + 2u] = scratch[2u * nt] / (1.0f + fast::exp(-gate[out_base + 2u]));\n"
"        attn[out_base + 3u] = scratch[3u * nt] / (1.0f + fast::exp(-gate[out_base + 3u]));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context8(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *probs [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint dim_base = tg.y * 8u;\n"
"    if (qh >= 8u || dim_base >= 256u) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    const uint kvh = qh / 4u;\n"
"    float acc0 = 0.0f;\n"
"    float acc1 = 0.0f;\n"
"    float acc2 = 0.0f;\n"
"    float acc3 = 0.0f;\n"
"    float acc4 = 0.0f;\n"
"    float acc5 = 0.0f;\n"
"    float acc6 = 0.0f;\n"
"    float acc7 = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        const float p = probs[logit_base + t];\n"
"        const uint v_base = (t * 2u + kvh) * 256u + dim_base;\n"
"        acc0 += p * values[v_base + 0u];\n"
"        acc1 += p * values[v_base + 1u];\n"
"        acc2 += p * values[v_base + 2u];\n"
"        acc3 += p * values[v_base + 3u];\n"
"        acc4 += p * values[v_base + 4u];\n"
"        acc5 += p * values[v_base + 5u];\n"
"        acc6 += p * values[v_base + 6u];\n"
"        acc7 += p * values[v_base + 7u];\n"
"    }\n"
"    scratch[0u * nt + tid] = acc0;\n"
"    scratch[1u * nt + tid] = acc1;\n"
"    scratch[2u * nt + tid] = acc2;\n"
"    scratch[3u * nt + tid] = acc3;\n"
"    scratch[4u * nt + tid] = acc4;\n"
"    scratch[5u * nt + tid] = acc5;\n"
"    scratch[6u * nt + tid] = acc6;\n"
"    scratch[7u * nt + tid] = acc7;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[0u * nt + tid] += scratch[0u * nt + tid + stride];\n"
"            scratch[1u * nt + tid] += scratch[1u * nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"            scratch[4u * nt + tid] += scratch[4u * nt + tid + stride];\n"
"            scratch[5u * nt + tid] += scratch[5u * nt + tid + stride];\n"
"            scratch[6u * nt + tid] += scratch[6u * nt + tid + stride];\n"
"            scratch[7u * nt + tid] += scratch[7u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        const uint out_base = qh * 256u + dim_base;\n"
"        attn[out_base + 0u] = scratch[0u] / (1.0f + fast::exp(-gate[out_base + 0u]));\n"
"        attn[out_base + 1u] = scratch[1u * nt] / (1.0f + fast::exp(-gate[out_base + 1u]));\n"
"        attn[out_base + 2u] = scratch[2u * nt] / (1.0f + fast::exp(-gate[out_base + 2u]));\n"
"        attn[out_base + 3u] = scratch[3u * nt] / (1.0f + fast::exp(-gate[out_base + 3u]));\n"
"        attn[out_base + 4u] = scratch[4u * nt] / (1.0f + fast::exp(-gate[out_base + 4u]));\n"
"        attn[out_base + 5u] = scratch[5u * nt] / (1.0f + fast::exp(-gate[out_base + 5u]));\n"
"        attn[out_base + 6u] = scratch[6u * nt] / (1.0f + fast::exp(-gate[out_base + 6u]));\n"
"        attn[out_base + 7u] = scratch[7u * nt] / (1.0f + fast::exp(-gate[out_base + 7u]));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context16(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *probs [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint dim_base = tg.y * 16u;\n"
"    if (qh >= 8u || dim_base >= 256u) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    const uint kvh = qh / 4u;\n"
"    float acc[16];\n"
"    for (uint i = 0u; i < 16u; i++) acc[i] = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        const float p = probs[logit_base + t];\n"
"        const uint v_base = (t * 2u + kvh) * 256u + dim_base;\n"
"        for (uint i = 0u; i < 16u; i++) acc[i] += p * values[v_base + i];\n"
"    }\n"
"    for (uint i = 0u; i < 16u; i++) scratch[i * nt + tid] = acc[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            for (uint i = 0u; i < 16u; i++) {\n"
"                scratch[i * nt + tid] += scratch[i * nt + tid + stride];\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        const uint out_base = qh * 256u + dim_base;\n"
"        for (uint i = 0u; i < 16u; i++) {\n"
"            attn[out_base + i] = scratch[i * nt] / (1.0f + fast::exp(-gate[out_base + i]));\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_logits_causal_mat(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device float *logits [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = gid.x;\n"
"    const uint q = gid.y;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    if (qh >= 8u || q >= n_ctx) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = q * 2048u + qh * 256u;\n"
"    const uint logit_base = (q * 8u + qh) * n_ctx;\n"
"    for (uint k = 0; k < n_ctx; k++) {\n"
"        float acc = 0.0f;\n"
"        if (k <= q) {\n"
"            const uint k_base = (k * 2u + kvh) * 256u;\n"
"            for (uint i = tid; i < 256u; i += nt) {\n"
"                acc += queries[q_base + i] * keys[k_base + i];\n"
"            }\n"
"        } else {\n"
"            acc = -INFINITY;\n"
"        }\n"
"        scratch[tid] = acc;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"            if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"        if (tid == 0) logits[logit_base + k] = k <= q ? scratch[0] * args.scale : -INFINITY;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context_causal_mat(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *logits [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *values [[buffer(3)]],\n"
"        device float *attn [[buffer(4)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = gid.x;\n"
"    const uint q = gid.y;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    if (qh >= 8u || q >= n_ctx) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint logit_base = (q * 8u + qh) * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint k = tid; k <= q; k += nt) {\n"
"        local_max = max(local_max, logits[logit_base + k]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint k = tid; k <= q; k += nt) {\n"
"        local_denom += exp(logits[logit_base + k] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    if (tid < 256u) {\n"
"        const uint kvh = qh / 4u;\n"
"        float acc = 0.0f;\n"
"        for (uint k = 0; k <= q; k++) {\n"
"            float p = exp(logits[logit_base + k] - max_logit) * inv_denom;\n"
"            acc += p * values[(k * 2u + kvh) * 256u + tid];\n"
"        }\n"
"        float g = gate[q * 2048u + qh * 256u + tid];\n"
"        attn[q * 2048u + qh * 256u + tid] = acc / (1.0f + fast::exp(-g));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context_causal_fused_mat(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *keys [[buffer(3)]],\n"
"        device const float *values [[buffer(4)]],\n"
"        device float *attn [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = gid.x;\n"
"    const uint q = gid.y;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    if (qh >= 8u || q >= n_ctx) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = q * 2048u + qh * 256u;\n"
"    float max_logit = -INFINITY;\n"
"    float denom = 0.0f;\n"
"    float acc = 0.0f;\n"
"    for (uint k = 0; k <= q; ++k) {\n"
"        const uint k_base = (k * 2u + kvh) * 256u;\n"
"        float dot = 0.0f;\n"
"        for (uint i = tid; i < 256u; i += nt) {\n"
"            dot += queries[q_base + i] * keys[k_base + i];\n"
"        }\n"
"        scratch[tid] = dot;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"            if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"        const float score = scratch[0] * args.scale;\n"
"        const float next_max = max(max_logit, score);\n"
"        const float old_scale = isinf(max_logit) ? 0.0f : exp(max_logit - next_max);\n"
"        const float add_scale = exp(score - next_max);\n"
"        if (tid < 256u) {\n"
"            acc = acc * old_scale + add_scale * values[k_base + tid];\n"
"        }\n"
"        denom = denom * old_scale + add_scale;\n"
"        max_logit = next_max;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid < 256u) {\n"
"        float g = gate[q_base + tid];\n"
"        attn[q_base + tid] = (acc / denom) / (1.0f + fast::exp(-g));\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_context_causal_fused_mat_twopass(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *gate [[buffer(2)]],\n"
"        device const float *keys [[buffer(3)]],\n"
"        device const float *values [[buffer(4)]],\n"
"        device float *attn [[buffer(5)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 gid [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = gid.x;\n"
"    const uint q = gid.y;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    if (qh >= 8u || q >= n_ctx) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = q * 2048u + qh * 256u;\n"
"    float max_logit = -INFINITY;\n"
"    for (uint k = 0; k <= q; ++k) {\n"
"        const uint k_base = (k * 2u + kvh) * 256u;\n"
"        float dot = 0.0f;\n"
"        for (uint i = tid; i < 256u; i += nt) {\n"
"            dot += queries[q_base + i] * keys[k_base + i];\n"
"        }\n"
"        scratch[tid] = dot;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"            if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"        max_logit = max(max_logit, scratch[0] * args.scale);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float denom = 0.0f;\n"
"    float acc = 0.0f;\n"
"    for (uint k = 0; k <= q; ++k) {\n"
"        const uint k_base = (k * 2u + kvh) * 256u;\n"
"        float dot = 0.0f;\n"
"        for (uint i = tid; i < 256u; i += nt) {\n"
"            dot += queries[q_base + i] * keys[k_base + i];\n"
"        }\n"
"        scratch[tid] = dot;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"            if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"        float p = exp(scratch[0] * args.scale - max_logit);\n"
"        denom += p;\n"
"        if (tid < 256u) acc += p * values[k_base + tid];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid < 256u) {\n"
"        float g = gate[q_base + tid];\n"
"        attn[q_base + tid] = (acc / denom) / (1.0f + fast::exp(-g));\n"
"    }\n"
"}\n"
"\n"
"struct ds4_drafter_metal_importance_args {\n"
"    int n_ctx;\n"
"    int pool_kernel;\n"
"};\n"
"\n"
"kernel void ds4_drafter_attention_importance(\n"
"        constant ds4_drafter_metal_importance_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        device float *head_rows [[buffer(2)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint qh [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]],\n"
"        uint nt [[threads_per_threadgroup]]) {\n"
"    if (qh >= 8u) return;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = qh * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_max = max(local_max, logits[logit_base + t]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_denom += exp(logits[logit_base + t] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    const int radius = (args.pool_kernel - 1) / 2;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        float sum = 0.0f;\n"
"        for (int u = -radius; u <= radius; u++) {\n"
"            int idx = int(t) + u;\n"
"            if (idx < 0) idx = 0;\n"
"            if (idx >= args.n_ctx) idx = args.n_ctx - 1;\n"
"            sum += fast::exp(logits[logit_base + uint(idx)] - max_logit) * inv_denom;\n"
"        }\n"
"        float smooth = sum / float(args.pool_kernel);\n"
"        head_rows[logit_base + t] = smooth;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_logits_batch(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device float *logits [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint step = tg.y;\n"
"    const uint t = tg.z;\n"
"    if (qh >= 8u || t >= uint(args.n_ctx)) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = step * 2048u + qh * 256u;\n"
"    const uint k_base = (t * 2u + kvh) * 256u;\n"
"    float acc = 0.0f;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        acc += queries[q_base + i] * keys[k_base + i];\n"
"    }\n"
"    scratch[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) logits[(step * 8u + qh) * n_ctx + t] = scratch[0] * args.scale;\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_logits_batch4(\n"
"        constant ds4_drafter_metal_attention_args &args [[buffer(0)]],\n"
"        device const float *queries [[buffer(1)]],\n"
"        device const float *keys [[buffer(2)]],\n"
"        device float *logits [[buffer(3)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint step = tg.y;\n"
"    const uint t_base = tg.z * 4u;\n"
"    if (qh >= 8u || t_base >= uint(args.n_ctx)) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint kvh = qh / 4u;\n"
"    const uint q_base = step * 2048u + qh * 256u;\n"
"    float acc0 = 0.0f;\n"
"    float acc1 = 0.0f;\n"
"    float acc2 = 0.0f;\n"
"    float acc3 = 0.0f;\n"
"    const bool v0 = t_base < n_ctx;\n"
"    const bool v1 = t_base + 1u < n_ctx;\n"
"    const bool v2 = t_base + 2u < n_ctx;\n"
"    const bool v3 = t_base + 3u < n_ctx;\n"
"    const uint k0 = ((t_base + 0u) * 2u + kvh) * 256u;\n"
"    const uint k1 = ((t_base + 1u) * 2u + kvh) * 256u;\n"
"    const uint k2 = ((t_base + 2u) * 2u + kvh) * 256u;\n"
"    const uint k3 = ((t_base + 3u) * 2u + kvh) * 256u;\n"
"    for (uint i = tid; i < 256u; i += nt) {\n"
"        const float q = queries[q_base + i];\n"
"        if (v0) acc0 += q * keys[k0 + i];\n"
"        if (v1) acc1 += q * keys[k1 + i];\n"
"        if (v2) acc2 += q * keys[k2 + i];\n"
"        if (v3) acc3 += q * keys[k3 + i];\n"
"    }\n"
"    scratch[0u * nt + tid] = acc0;\n"
"    scratch[1u * nt + tid] = acc1;\n"
"    scratch[2u * nt + tid] = acc2;\n"
"    scratch[3u * nt + tid] = acc3;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            scratch[0u * nt + tid] += scratch[0u * nt + tid + stride];\n"
"            scratch[1u * nt + tid] += scratch[1u * nt + tid + stride];\n"
"            scratch[2u * nt + tid] += scratch[2u * nt + tid + stride];\n"
"            scratch[3u * nt + tid] += scratch[3u * nt + tid + stride];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        const uint out_base = (step * 8u + qh) * n_ctx + t_base;\n"
"        if (v0) logits[out_base + 0u] = scratch[0u] * args.scale;\n"
"        if (v1) logits[out_base + 1u] = scratch[1u * nt] * args.scale;\n"
"        if (v2) logits[out_base + 2u] = scratch[2u * nt] * args.scale;\n"
"        if (v3) logits[out_base + 3u] = scratch[3u * nt] * args.scale;\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_importance_batch(\n"
"        constant ds4_drafter_metal_importance_args &args [[buffer(0)]],\n"
"        device float *logits [[buffer(1)]],\n"
"        device float *head_rows [[buffer(2)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint3 tg [[threadgroup_position_in_grid]],\n"
"        uint3 tid3 [[thread_position_in_threadgroup]],\n"
"        uint3 nt3 [[threads_per_threadgroup]]) {\n"
"    const uint qh = tg.x;\n"
"    const uint step = tg.y;\n"
"    if (qh >= 8u) return;\n"
"    const uint tid = tid3.x;\n"
"    const uint nt = nt3.x;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    const uint logit_base = (step * 8u + qh) * n_ctx;\n"
"    float local_max = -INFINITY;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_max = max(local_max, logits[logit_base + t]);\n"
"    }\n"
"    scratch[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float max_logit = scratch[0];\n"
"    float local_denom = 0.0f;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        local_denom += fast::exp(logits[logit_base + t] - max_logit);\n"
"    }\n"
"    scratch[tid] = local_denom;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_denom = 1.0f / scratch[0];\n"
"    const int radius = (args.pool_kernel - 1) / 2;\n"
"    for (uint t = tid; t < n_ctx; t += nt) {\n"
"        float sum = 0.0f;\n"
"        for (int u = -radius; u <= radius; u++) {\n"
"            int idx = int(t) + u;\n"
"            if (idx < 0) idx = 0;\n"
"            if (idx >= args.n_ctx) idx = args.n_ctx - 1;\n"
"            sum += fast::exp(logits[logit_base + uint(idx)] - max_logit) * inv_denom;\n"
"        }\n"
"        head_rows[logit_base + t] = sum / float(args.pool_kernel);\n"
"    }\n"
"}\n"
"\n"
"kernel void ds4_drafter_attention_importance_reduce_heads(\n"
"        constant ds4_drafter_metal_importance_args &args [[buffer(0)]],\n"
"        device const float *head_rows [[buffer(1)]],\n"
"        device float *max_rows [[buffer(2)]],\n"
"        uint3 gid [[thread_position_in_grid]]) {\n"
"    const uint t = gid.x;\n"
"    const uint step = gid.y;\n"
"    if (t >= uint(args.n_ctx)) return;\n"
"    const uint n_ctx = uint(args.n_ctx);\n"
"    float m = head_rows[(step * 8u + 0u) * n_ctx + t];\n"
"    for (uint qh = 1u; qh < 8u; qh++) {\n"
"        m = max(m, head_rows[(step * 8u + qh) * n_ctx + t]);\n"
"    }\n"
"    max_rows[step * n_ctx + t] = m;\n"
"}\n"
"\n";

static int drafter_metal_fail(char *err, size_t errlen, NSString *msg) {
    if (err && errlen) {
        snprintf(err, errlen, "%s", msg ? [msg UTF8String] : "native Metal drafter failed");
    }
    return -1;
}

static int drafter_metal_init(char *err, size_t errlen) {
    if (g_init_attempted) {
        if (g_init_ok) return 0;
        if (err && errlen) snprintf(err, errlen, "native Metal drafter initialization previously failed");
        return -1;
    }
    g_init_attempted = 1;
    @autoreleasepool {
        g_drafter_device = MTLCreateSystemDefaultDevice();
        if (!g_drafter_device) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter could not create default Metal device");
            return -1;
        }
        g_drafter_queue = [g_drafter_device newCommandQueue];
        if (!g_drafter_queue) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter could not create command queue");
            return -1;
        }
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:g_drafter_metal_source];
        id<MTLLibrary> library = [g_drafter_device newLibraryWithSource:src
                                                                 options:nil
                                                                   error:&error];
        if (!library) return drafter_metal_fail(err, errlen, error.localizedDescription);
        id<MTLFunction> fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matvec"];
        if (!fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matvec kernel");
            return -1;
        }
        g_affine_u32_matvec_pipeline = [g_drafter_device newComputePipelineStateWithFunction:fn
                                                                                       error:&error];
        if (!g_affine_u32_matvec_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> q_only_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_q_only_matvec"];
        if (!q_only_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing q-only affine matvec kernel");
            return -1;
        }
        g_affine_u32_q_only_matvec_pipeline = [g_drafter_device newComputePipelineStateWithFunction:q_only_fn
                                                                                              error:&error];
        if (!g_affine_u32_q_only_matvec_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> matmat_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matmat"];
        if (!matmat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matmat kernel");
            return -1;
        }
        g_affine_u32_matmat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:matmat_fn
                                                                                       error:&error];
        if (!g_affine_u32_matmat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> matmat4_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matmat4"];
        if (!matmat4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matmat4 kernel");
            return -1;
        }
        g_affine_u32_matmat4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:matmat4_fn
                                                                                        error:&error];
        if (!g_affine_u32_matmat4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> matmat4x2_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matmat4x2"];
        if (!matmat4x2_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matmat4x2 kernel");
            return -1;
        }
        g_affine_u32_matmat4x2_pipeline = [g_drafter_device newComputePipelineStateWithFunction:matmat4x2_fn
                                                                                          error:&error];
        if (!g_affine_u32_matmat4x2_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> matmat4x4_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matmat4x4"];
        if (!matmat4x4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matmat4x4 kernel");
            return -1;
        }
        g_affine_u32_matmat4x4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:matmat4x4_fn
                                                                                          error:&error];
        if (!g_affine_u32_matmat4x4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> matmat8x2_fn = [library newFunctionWithName:@"ds4_drafter_affine_u32_matmat8x2"];
        if (!matmat8x2_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing affine matmat8x2 kernel");
            return -1;
        }
        g_affine_u32_matmat8x2_pipeline = [g_drafter_device newComputePipelineStateWithFunction:matmat8x2_fn
                                                                                          error:&error];
        if (!g_affine_u32_matmat8x2_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> dequant_row_fn = [library newFunctionWithName:@"ds4_drafter_dequant_u32_row"];
        if (!dequant_row_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing dequant row kernel");
            return -1;
        }
        g_dequant_u32_row_pipeline = [g_drafter_device newComputePipelineStateWithFunction:dequant_row_fn
                                                                                     error:&error];
        if (!g_dequant_u32_row_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> argmax_fn = [library newFunctionWithName:@"ds4_drafter_argmax"];
        if (!argmax_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing argmax kernel");
            return -1;
        }
        g_argmax_pipeline = [g_drafter_device newComputePipelineStateWithFunction:argmax_fn
                                                                            error:&error];
        if (!g_argmax_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> rms_fn = [library newFunctionWithName:@"ds4_drafter_rms_norm_bf16"];
        if (!rms_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing RMSNorm kernel");
            return -1;
        }
        g_rms_norm_bf16_pipeline = [g_drafter_device newComputePipelineStateWithFunction:rms_fn
                                                                                   error:&error];
        if (!g_rms_norm_bf16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> rms_mat_fn = [library newFunctionWithName:@"ds4_drafter_rms_norm_bf16_mat"];
        if (!rms_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing RMSNorm matrix kernel");
            return -1;
        }
        g_rms_norm_bf16_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:rms_mat_fn
                                                                                       error:&error];
        if (!g_rms_norm_bf16_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> rms_mat_round_fn = [library newFunctionWithName:@"ds4_drafter_rms_norm_bf16_mat_round"];
        if (!rms_mat_round_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing rounded RMSNorm matrix kernel");
            return -1;
        }
        g_rms_norm_bf16_mat_round_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:rms_mat_round_fn
                                                            error:&error];
        if (!g_rms_norm_bf16_mat_round_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> swiglu_fn = [library newFunctionWithName:@"ds4_drafter_swiglu"];
        if (!swiglu_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing SwiGLU kernel");
            return -1;
        }
        g_swiglu_pipeline = [g_drafter_device newComputePipelineStateWithFunction:swiglu_fn
                                                                            error:&error];
        if (!g_swiglu_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> swiglu_x4_fn = [library newFunctionWithName:@"ds4_drafter_swiglu_x4"];
        if (swiglu_x4_fn) {
            g_swiglu_x4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:swiglu_x4_fn
                                                                                   error:&error];
            if (!g_swiglu_x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> swiglu_packed_pair_fn = [library newFunctionWithName:@"ds4_drafter_swiglu_packed_pair"];
        if (!swiglu_packed_pair_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing packed SwiGLU kernel");
            return -1;
        }
        g_swiglu_packed_pair_pipeline = [g_drafter_device newComputePipelineStateWithFunction:swiglu_packed_pair_fn
                                                                                    error:&error];
        if (!g_swiglu_packed_pair_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> round_fn = [library newFunctionWithName:@"ds4_drafter_round_bf16"];
        if (!round_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing BF16 round kernel");
            return -1;
        }
        g_round_bf16_pipeline = [g_drafter_device newComputePipelineStateWithFunction:round_fn
                                                                                error:&error];
        if (!g_round_bf16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> residual_add_fn = [library newFunctionWithName:@"ds4_drafter_residual_add_round"];
        if (!residual_add_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing residual add kernel");
            return -1;
        }
        g_residual_add_round_pipeline = [g_drafter_device newComputePipelineStateWithFunction:residual_add_fn
                                                                                        error:&error];
        if (!g_residual_add_round_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> residual_add_pre_b_round_fn = [library newFunctionWithName:@"ds4_drafter_residual_add_round_pre_b_round"];
        if (!residual_add_pre_b_round_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing residual add pre-round kernel");
            return -1;
        }
        g_residual_add_round_pre_b_round_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:residual_add_pre_b_round_fn
                                                            error:&error];
        if (!g_residual_add_round_pre_b_round_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> residual_add_pre_b_round_norm_fn = [library newFunctionWithName:@"ds4_drafter_residual_add_round_pre_b_round_norm"];
        if (!residual_add_pre_b_round_norm_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing residual add pre-round norm kernel");
            return -1;
        }
        g_residual_add_round_pre_b_round_norm_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:residual_add_pre_b_round_norm_fn
                                                            error:&error];
        if (!g_residual_add_round_pre_b_round_norm_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv"];
        if (!linear_conv_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear conv kernel");
            return -1;
        }
        g_linear_conv_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_conv_fn
                                                                                 error:&error];
        if (!g_linear_conv_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_mat"];
        if (!linear_conv_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear conv matrix kernel");
            return -1;
        }
        g_linear_conv_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_conv_mat_fn
                                                                                     error:&error];
        if (!g_linear_conv_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_stateful_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_stateful_mat"];
        if (!linear_conv_stateful_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing stateful linear conv matrix kernel");
            return -1;
        }
        g_linear_conv_stateful_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:linear_conv_stateful_mat_fn
                                                            error:&error];
        if (!g_linear_conv_stateful_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_qkvz_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_qkvz_mat"];
        if (!linear_conv_qkvz_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear conv qkvz matrix kernel");
            return -1;
        }
        g_linear_conv_qkvz_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_conv_qkvz_mat_fn
                                                                                          error:&error];
        if (!g_linear_conv_qkvz_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_stateful_qkvz_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_stateful_qkvz_mat"];
        if (!linear_conv_stateful_qkvz_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing stateful linear conv qkvz matrix kernel");
            return -1;
        }
        g_linear_conv_stateful_qkvz_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:linear_conv_stateful_qkvz_mat_fn
                                                            error:&error];
        if (!g_linear_conv_stateful_qkvz_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_state_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_state_from_qkv_mat"];
        if (!linear_conv_state_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear conv state matrix kernel");
            return -1;
        }
        g_linear_conv_state_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_conv_state_mat_fn
                                                                                           error:&error];
        if (!g_linear_conv_state_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_conv_state_qkvz_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_conv_state_from_qkvz_mat"];
        if (!linear_conv_state_qkvz_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear conv qkvz state matrix kernel");
            return -1;
        }
        g_linear_conv_state_qkvz_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_conv_state_qkvz_mat_fn
                                                                                                error:&error];
        if (!g_linear_conv_state_qkvz_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_qk_norm_fn = [library newFunctionWithName:@"ds4_drafter_linear_qk_norm"];
        if (!linear_qk_norm_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear qk norm kernel");
            return -1;
        }
        g_linear_qk_norm_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_qk_norm_fn
                                                                                    error:&error];
        if (!g_linear_qk_norm_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_qk_norm_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_qk_norm_mat"];
        if (!linear_qk_norm_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear qk norm matrix kernel");
            return -1;
        }
        g_linear_qk_norm_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_qk_norm_mat_fn
                                                                                    error:&error];
        if (!g_linear_qk_norm_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_qk_norm_kq_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_qk_norm_kq_mat"];
        if (!linear_qk_norm_kq_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear qk norm kq matrix kernel");
            return -1;
        }
        g_linear_qk_norm_kq_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_qk_norm_kq_mat_fn
                                                                                       error:&error];
        if (!g_linear_qk_norm_kq_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_delta_fn = [library newFunctionWithName:@"ds4_drafter_linear_delta"];
        if (!linear_delta_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear delta kernel");
            return -1;
        }
        g_linear_delta_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_delta_fn
                                                                                  error:&error];
        if (!g_linear_delta_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_scan_params_fn = [library newFunctionWithName:@"ds4_drafter_linear_scan_params"];
        if (!linear_scan_params_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear scan params kernel");
            return -1;
        }
        g_linear_scan_params_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_scan_params_fn
                                                                                    error:&error];
        if (!g_linear_scan_params_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_delta_scan_fn = [library newFunctionWithName:@"ds4_drafter_linear_delta_scan"];
        if (!linear_delta_scan_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear delta scan kernel");
            return -1;
        }
        g_linear_delta_scan_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_delta_scan_fn
                                                                                      error:&error];
        if (!g_linear_delta_scan_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_delta_scan2_fn = [library newFunctionWithName:@"ds4_drafter_linear_delta_scan2"];
        if (!linear_delta_scan2_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear delta scan2 kernel");
            return -1;
        }
        g_linear_delta_scan2_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_delta_scan2_fn
                                                                                       error:&error];
        if (!g_linear_delta_scan2_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_delta_scan4_fn = [library newFunctionWithName:@"ds4_drafter_linear_delta_scan4"];
        if (!linear_delta_scan4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear delta scan4 kernel");
            return -1;
        }
        g_linear_delta_scan4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_delta_scan4_fn
                                                                                       error:&error];
        if (!g_linear_delta_scan4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_gate_fn = [library newFunctionWithName:@"ds4_drafter_linear_gate"];
        if (!linear_gate_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear gate kernel");
            return -1;
        }
        g_linear_gate_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_gate_fn
                                                                                 error:&error];
        if (!g_linear_gate_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_gate_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_gate_mat"];
        if (!linear_gate_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear gate matrix kernel");
            return -1;
        }
        g_linear_gate_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_gate_mat_fn
                                                                                    error:&error];
        if (!g_linear_gate_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> linear_gate_qkvz_mat_fn = [library newFunctionWithName:@"ds4_drafter_linear_gate_qkvz_mat"];
        if (!linear_gate_qkvz_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing linear gate qkvz matrix kernel");
            return -1;
        }
        g_linear_gate_qkvz_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:linear_gate_qkvz_mat_fn
                                                                                          error:&error];
        if (!g_linear_gate_qkvz_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_q_fn = [library newFunctionWithName:@"ds4_drafter_full_q_norm_rope"];
        if (!full_q_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing full q norm rope kernel");
            return -1;
        }
        g_full_q_norm_rope_pipeline = [g_drafter_device newComputePipelineStateWithFunction:full_q_fn
                                                                                      error:&error];
        if (!g_full_q_norm_rope_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_q_only_fn = [library newFunctionWithName:@"ds4_drafter_full_q_only_norm_rope"];
        if (!full_q_only_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing full q-only norm rope kernel");
            return -1;
        }
        g_full_q_only_norm_rope_pipeline = [g_drafter_device newComputePipelineStateWithFunction:full_q_only_fn
                                                                                          error:&error];
        if (!g_full_q_only_norm_rope_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_q_mat_fn = [library newFunctionWithName:@"ds4_drafter_full_q_norm_rope_mat"];
        if (!full_q_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing full q norm rope matrix kernel");
            return -1;
        }
        g_full_q_norm_rope_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:full_q_mat_fn
                                                                                          error:&error];
        if (!g_full_q_norm_rope_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_k_fn = [library newFunctionWithName:@"ds4_drafter_full_k_norm_rope"];
        if (!full_k_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing full k norm rope kernel");
            return -1;
        }
        g_full_k_norm_rope_pipeline = [g_drafter_device newComputePipelineStateWithFunction:full_k_fn
                                                                                      error:&error];
        if (!g_full_k_norm_rope_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_k_mat_fn = [library newFunctionWithName:@"ds4_drafter_full_k_norm_rope_mat"];
        if (!full_k_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing full k norm rope matrix kernel");
            return -1;
        }
        g_full_k_norm_rope_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:full_k_mat_fn
                                                                                          error:&error];
        if (!g_full_k_norm_rope_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> full_k_mat_pack_f16_fn = [library newFunctionWithName:@"ds4_drafter_full_k_norm_rope_mat_pack_f16"];
        if (full_k_mat_pack_f16_fn) {
            g_full_k_norm_rope_mat_pack_f16_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:full_k_mat_pack_f16_fn
                                                                error:&error];
            if (!g_full_k_norm_rope_mat_pack_f16_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_logits_fn = [library newFunctionWithName:@"ds4_drafter_attention_logits"];
        if (!attn_logits_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention logits kernel");
            return -1;
        }
        g_attention_logits_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_logits_fn
                                                                                      error:&error];
        if (!g_attention_logits_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_logits4_fn = [library newFunctionWithName:@"ds4_drafter_attention_logits4"];
        if (!attn_logits4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention logits4 kernel");
            return -1;
        }
        g_attention_logits4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_logits4_fn
                                                                                   error:&error];
        if (!g_attention_logits4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_logits_causal_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_logits_causal_mat"];
        if (!attn_logits_causal_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing causal attention logits matrix kernel");
            return -1;
        }
        g_attention_logits_causal_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_logits_causal_mat_fn
                                                                                                  error:&error];
        if (!g_attention_logits_causal_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context_fn = [library newFunctionWithName:@"ds4_drafter_attention_context"];
        if (!attn_context_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention context kernel");
            return -1;
        }
        g_attention_context_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context_fn
                                                                                       error:&error];
        if (!g_attention_context_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_softmax_inplace_fn = [library newFunctionWithName:@"ds4_drafter_attention_softmax_inplace"];
        if (!attn_softmax_inplace_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention softmax inplace kernel");
            return -1;
        }
        g_attention_softmax_inplace_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_softmax_inplace_fn
                                                                                            error:&error];
        if (!g_attention_softmax_inplace_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context_parallel_fn = [library newFunctionWithName:@"ds4_drafter_attention_context_parallel"];
        if (!attn_context_parallel_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing parallel attention context kernel");
            return -1;
        }
        g_attention_context_parallel_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context_parallel_fn
                                                                                             error:&error];
        if (!g_attention_context_parallel_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context4_fn = [library newFunctionWithName:@"ds4_drafter_attention_context4"];
        if (!attn_context4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention context4 kernel");
            return -1;
        }
        g_attention_context4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context4_fn
                                                                                    error:&error];
        if (!g_attention_context4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context8_fn = [library newFunctionWithName:@"ds4_drafter_attention_context8"];
        if (!attn_context8_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention context8 kernel");
            return -1;
        }
        g_attention_context8_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context8_fn
                                                                                    error:&error];
        if (!g_attention_context8_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context16_fn = [library newFunctionWithName:@"ds4_drafter_attention_context16"];
        if (!attn_context16_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention context16 kernel");
            return -1;
        }
        g_attention_context16_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context16_fn
                                                                                     error:&error];
        if (!g_attention_context16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context_causal_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_context_causal_mat"];
        if (!attn_context_causal_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing causal attention context matrix kernel");
            return -1;
        }
        g_attention_context_causal_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context_causal_mat_fn
                                                                                                   error:&error];
        if (!g_attention_context_causal_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context_causal_fused_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_context_causal_fused_mat"];
        if (!attn_context_causal_fused_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing causal fused attention matrix context kernel");
            return -1;
        }
        g_attention_context_causal_fused_mat_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_context_causal_fused_mat_fn
                                                                                                        error:&error];
        if (!g_attention_context_causal_fused_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_context_causal_fused_mat_twopass_fn = [library newFunctionWithName:@"ds4_drafter_attention_context_causal_fused_mat_twopass"];
        if (!attn_context_causal_fused_mat_twopass_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing two-pass causal fused attention matrix context kernel");
            return -1;
        }
        g_attention_context_causal_fused_mat_twopass_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_context_causal_fused_mat_twopass_fn
                                                            error:&error];
        if (!g_attention_context_causal_fused_mat_twopass_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_head_mat"];
        if (!attn_pack_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention head pack matrix kernel");
            return -1;
        }
        g_attention_pack_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_head_mat_fn
                                                            error:&error];
        if (!g_attention_pack_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_q_block_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_q_block_head_mat"];
        if (!attn_pack_q_block_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention q block pack matrix kernel");
            return -1;
        }
        g_attention_pack_q_block_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_q_block_head_mat_fn
                                                            error:&error];
        if (!g_attention_pack_q_block_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_q_group_block_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_q_group_block_head_mat"];
        if (!attn_pack_q_group_block_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention grouped q block pack matrix kernel");
            return -1;
        }
        g_attention_pack_q_group_block_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_q_group_block_head_mat_fn
                                                            error:&error];
        if (!g_attention_pack_q_group_block_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_kv_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_kv_head_mat"];
        if (!attn_pack_kv_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention kv head pack matrix kernel");
            return -1;
        }
        g_attention_pack_kv_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_kv_head_mat_fn
                                                            error:&error];
        if (!g_attention_pack_kv_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_q_group_block_head_f16_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_q_group_block_head_f16"];
        if (!attn_pack_q_group_block_head_f16_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing grouped q block f16 pack kernel");
            return -1;
        }
        g_attention_pack_q_group_block_head_f16_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_q_group_block_head_f16_fn
                                                            error:&error];
        if (!g_attention_pack_q_group_block_head_f16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_q_group_block_head_f16x4_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_q_group_block_head_f16x4"];
        if (attn_pack_q_group_block_head_f16x4_fn) {
            g_attention_pack_q_group_block_head_f16x4_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:attn_pack_q_group_block_head_f16x4_fn
                                                                error:&error];
            if (!g_attention_pack_q_group_block_head_f16x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_pack_kv_head_f16_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_kv_head_f16"];
        if (!attn_pack_kv_head_f16_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention kv f16 pack kernel");
            return -1;
        }
        g_attention_pack_kv_head_f16_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_pack_kv_head_f16_fn
                                                            error:&error];
        if (!g_attention_pack_kv_head_f16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_pack_kv_head_f16x4_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_kv_head_f16x4"];
        if (attn_pack_kv_head_f16x4_fn) {
            g_attention_pack_kv_head_f16x4_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:attn_pack_kv_head_f16x4_fn
                                                                error:&error];
            if (!g_attention_pack_kv_head_f16x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_pack_v_head_f16x4_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_v_head_f16x4"];
        if (attn_pack_v_head_f16x4_fn) {
            g_attention_pack_v_head_f16x4_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:attn_pack_v_head_f16x4_fn
                                                                error:&error];
            if (!g_attention_pack_v_head_f16x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_pack_kv_all_head_f16x4_fn = [library newFunctionWithName:@"ds4_drafter_attention_pack_kv_all_head_f16x4"];
        if (attn_pack_kv_all_head_f16x4_fn) {
            g_attention_pack_kv_all_head_f16x4_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:attn_pack_kv_all_head_f16x4_fn
                                                                error:&error];
            if (!g_attention_pack_kv_all_head_f16x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_causal_softmax_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_causal_softmax_mat"];
        if (!attn_causal_softmax_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing causal attention softmax matrix kernel");
            return -1;
        }
        g_attention_causal_softmax_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_causal_softmax_mat_fn
                                                            error:&error];
        if (!g_attention_causal_softmax_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_causal_softmax_block_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_causal_softmax_block_mat"];
        if (!attn_causal_softmax_block_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing causal attention softmax block matrix kernel");
            return -1;
        }
        g_attention_causal_softmax_block_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_causal_softmax_block_mat_fn
                                                            error:&error];
        if (!g_attention_causal_softmax_block_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_causal_softmax_group_block_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_causal_softmax_group_block_mat"];
        if (!attn_causal_softmax_group_block_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing grouped causal attention softmax block matrix kernel");
            return -1;
        }
        g_attention_causal_softmax_group_block_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_causal_softmax_group_block_mat_fn
                                                             error:&error];
        if (!g_attention_causal_softmax_group_block_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_causal_softmax_group_block_f16_fn = [library newFunctionWithName:@"ds4_drafter_attention_causal_softmax_group_block_f16"];
        if (!attn_causal_softmax_group_block_f16_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing grouped causal attention f16 softmax block kernel");
            return -1;
        }
        g_attention_causal_softmax_group_block_f16_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_causal_softmax_group_block_f16_fn
                                                            error:&error];
        if (!g_attention_causal_softmax_group_block_f16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_causal_mask_group_block_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_causal_mask_group_block_mat"];
        if (!attn_causal_mask_group_block_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing grouped causal attention mask block matrix kernel");
            return -1;
        }
        g_attention_causal_mask_group_block_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_causal_mask_group_block_mat_fn
                                                            error:&error];
        if (!g_attention_causal_mask_group_block_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_unpack_gate_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_unpack_gate_head_mat"];
        if (!attn_unpack_gate_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention head unpack/gate matrix kernel");
            return -1;
        }
        g_attention_unpack_gate_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_unpack_gate_head_mat_fn
                                                            error:&error];
        if (!g_attention_unpack_gate_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_unpack_gate_block_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_unpack_gate_block_head_mat"];
        if (!attn_unpack_gate_block_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention head block unpack/gate matrix kernel");
            return -1;
        }
        g_attention_unpack_gate_block_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_unpack_gate_block_head_mat_fn
                                                            error:&error];
        if (!g_attention_unpack_gate_block_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_unpack_gate_group_block_head_mat_fn = [library newFunctionWithName:@"ds4_drafter_attention_unpack_gate_group_block_head_mat"];
        if (!attn_unpack_gate_group_block_head_mat_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention grouped head block unpack/gate matrix kernel");
            return -1;
        }
        g_attention_unpack_gate_group_block_head_mat_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_unpack_gate_group_block_head_mat_fn
                                                            error:&error];
        if (!g_attention_unpack_gate_group_block_head_mat_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_unpack_gate_group_block_head_f16_fn = [library newFunctionWithName:@"ds4_drafter_attention_unpack_gate_group_block_head_f16"];
        if (!attn_unpack_gate_group_block_head_f16_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention grouped f16 head block unpack/gate kernel");
            return -1;
        }
        g_attention_unpack_gate_group_block_head_f16_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_unpack_gate_group_block_head_f16_fn
                                                            error:&error];
        if (!g_attention_unpack_gate_group_block_head_f16_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_unpack_gate_group_block_head_f16x4_fn = [library newFunctionWithName:@"ds4_drafter_attention_unpack_gate_group_block_head_f16x4"];
        if (attn_unpack_gate_group_block_head_f16x4_fn) {
            g_attention_unpack_gate_group_block_head_f16x4_pipeline =
                [g_drafter_device newComputePipelineStateWithFunction:attn_unpack_gate_group_block_head_f16x4_fn
                                                                error:&error];
            if (!g_attention_unpack_gate_group_block_head_f16x4_pipeline) {
                return drafter_metal_fail(err, errlen, error.localizedDescription);
            }
        }
        id<MTLFunction> attn_importance_fn = [library newFunctionWithName:@"ds4_drafter_attention_importance"];
        if (!attn_importance_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention importance kernel");
            return -1;
        }
        g_attention_importance_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_importance_fn
                                                                                          error:&error];
        if (!g_attention_importance_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_logits_batch_fn = [library newFunctionWithName:@"ds4_drafter_attention_logits_batch"];
        if (!attn_logits_batch_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing batched attention logits kernel");
            return -1;
        }
        g_attention_logits_batch_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_logits_batch_fn
                                                                                            error:&error];
        if (!g_attention_logits_batch_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_logits_batch4_fn = [library newFunctionWithName:@"ds4_drafter_attention_logits_batch4"];
        if (!attn_logits_batch4_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing batched attention logits4 kernel");
            return -1;
        }
        g_attention_logits_batch4_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_logits_batch4_fn
                                                                                             error:&error];
        if (!g_attention_logits_batch4_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_importance_batch_fn = [library newFunctionWithName:@"ds4_drafter_attention_importance_batch"];
        if (!attn_importance_batch_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing batched attention importance kernel");
            return -1;
        }
        g_attention_importance_batch_pipeline = [g_drafter_device newComputePipelineStateWithFunction:attn_importance_batch_fn
                                                                                                error:&error];
        if (!g_attention_importance_batch_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        id<MTLFunction> attn_importance_reduce_heads_fn =
            [library newFunctionWithName:@"ds4_drafter_attention_importance_reduce_heads"];
        if (!attn_importance_reduce_heads_fn) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter missing attention importance reduce-heads kernel");
            return -1;
        }
        g_attention_importance_reduce_heads_pipeline =
            [g_drafter_device newComputePipelineStateWithFunction:attn_importance_reduce_heads_fn
                                                            error:&error];
        if (!g_attention_importance_reduce_heads_pipeline) {
            return drafter_metal_fail(err, errlen, error.localizedDescription);
        }
        g_buffer_cache = [NSMutableDictionary dictionary];
        g_dense_affine_cache = [NSMutableDictionary dictionary];
        g_mps_matmul_cache = [NSMutableDictionary dictionary];
        g_mps_matmul_shape_warmup_cache = [NSMutableDictionary dictionary];
        g_mps_matrix_cache = [NSMutableDictionary dictionary];
        g_mps_matrix_softmax = [[MPSMatrixSoftMax alloc] initWithDevice:g_drafter_device];
        g_init_ok = 1;
        return 0;
    }
}

int ds4_drafter_metal_available(void) {
    char err[256];
    return drafter_metal_init(err, sizeof(err)) == 0;
}

void ds4_drafter_metal_shutdown(void) {
    @autoreleasepool {
        [g_buffer_cache removeAllObjects];
        g_buffer_cache = nil;
        [g_dense_affine_cache removeAllObjects];
        g_dense_affine_cache = nil;
        [g_mps_matmul_cache removeAllObjects];
        g_mps_matmul_cache = nil;
        memset(g_mps_matmul_fast_cache, 0, sizeof(g_mps_matmul_fast_cache));
        [g_mps_matmul_shape_warmup_cache removeAllObjects];
        g_mps_matmul_shape_warmup_cache = nil;
        [g_mps_matrix_cache removeAllObjects];
        g_mps_matrix_cache = nil;
        memset(g_mps_matrix_fast_cache, 0, sizeof(g_mps_matrix_fast_cache));
        g_mps_matrix_softmax = nil;
        g_x_buffer = nil;
        g_out_buffer = nil;
        for (int i = 0; i < 8; i++) {
            g_many_out_buffers[i] = nil;
            g_many_out_bytes[i] = 0;
        }
        for (int i = 0; i < 2; i++) {
            g_token_hidden_buffers[i] = nil;
            g_token_hidden_bytes[i] = 0;
        }
        for (int i = 0; i < 24; i++) {
            g_attention_key_cache_buffers[i] = nil;
            g_attention_value_cache_buffers[i] = nil;
            g_query_capture_buffers[i] = nil;
            g_attention_key_cache_bytes[i] = 0;
            g_attention_value_cache_bytes[i] = 0;
            g_query_capture_bytes[i] = 0;
            g_linear_conv_state_buffers[i] = nil;
            g_linear_delta_state_buffers[i] = nil;
            g_linear_conv_state_bytes[i] = 0;
            g_linear_delta_state_bytes[i] = 0;
        }
        g_attention_q_buffer = nil;
        g_attention_gate_buffer = nil;
        g_attention_keys_buffer = nil;
        g_attention_values_buffer = nil;
        g_attention_logits_buffer = nil;
        g_attention_out_buffer = nil;
        g_attention_q_head_buffer = nil;
        g_attention_k_head_buffer = nil;
        g_attention_v_head_buffer = nil;
        g_attention_ctx_head_buffer = nil;
        g_attention_q_head_f16_buffer = nil;
        g_attention_k_head_f16_buffer = nil;
        g_attention_v_head_f16_buffer = nil;
        g_attention_logits_f16_buffer = nil;
        g_attention_ctx_head_f16_buffer = nil;
        g_argmax_buffer = nil;
        g_importance_row_buffer = nil;
        g_importance_max_buffer = nil;
        g_linear_y_buffer = nil;
        g_linear_kq_buffer = nil;
        g_linear_scan_params_buffer = nil;
        g_linear_scan_state_buffer = nil;
        g_linear_scan_debug_buffer = nil;
        g_x_bytes = 0;
        g_out_bytes = 0;
        g_attention_q_bytes = 0;
        g_attention_gate_bytes = 0;
        g_attention_keys_bytes = 0;
        g_attention_values_bytes = 0;
        g_attention_logits_bytes = 0;
        g_attention_out_bytes = 0;
        g_attention_q_head_bytes = 0;
        g_attention_k_head_bytes = 0;
        g_attention_v_head_bytes = 0;
        g_attention_ctx_head_bytes = 0;
        g_attention_q_head_f16_bytes = 0;
        g_attention_k_head_f16_bytes = 0;
        g_attention_v_head_f16_bytes = 0;
        g_attention_prepacked_k_f16_n_ctx = 0;
        g_argmax_bytes = 0;
        g_importance_row_bytes = 0;
        g_importance_max_bytes = 0;
        g_linear_y_bytes = 0;
        g_linear_kq_bytes = 0;
        g_linear_scan_params_bytes = 0;
        g_linear_scan_state_bytes = 0;
        g_linear_scan_debug_bytes = 0;
        g_attention_context_causal_fused_mat_pipeline = nil;
        g_attention_context_causal_fused_mat_twopass_pipeline = nil;
        g_attention_pack_head_mat_pipeline = nil;
        g_attention_pack_q_block_head_mat_pipeline = nil;
        g_attention_pack_kv_head_mat_pipeline = nil;
        g_attention_pack_q_group_block_head_f16_pipeline = nil;
        g_attention_pack_q_group_block_head_f16x4_pipeline = nil;
        g_attention_pack_kv_head_f16_pipeline = nil;
        g_attention_pack_kv_head_f16x4_pipeline = nil;
        g_attention_pack_kv_all_head_f16x4_pipeline = nil;
        g_attention_pack_v_head_f16x4_pipeline = nil;
        g_attention_pack_q_group_block_head_mat_pipeline = nil;
        g_attention_causal_softmax_mat_pipeline = nil;
        g_attention_causal_softmax_block_mat_pipeline = nil;
        g_attention_causal_softmax_group_block_mat_pipeline = nil;
        g_attention_causal_softmax_group_block_f16_pipeline = nil;
        g_attention_causal_mask_group_block_mat_pipeline = nil;
        g_attention_unpack_gate_head_mat_pipeline = nil;
        g_attention_unpack_gate_block_head_mat_pipeline = nil;
        g_attention_unpack_gate_group_block_head_mat_pipeline = nil;
        g_attention_unpack_gate_group_block_head_f16_pipeline = nil;
        g_attention_unpack_gate_group_block_head_f16x4_pipeline = nil;
        g_attention_context_causal_mat_pipeline = nil;
        g_attention_logits_causal_mat_pipeline = nil;
        g_attention_context_pipeline = nil;
        g_attention_softmax_inplace_pipeline = nil;
        g_attention_context_parallel_pipeline = nil;
        g_attention_context4_pipeline = nil;
        g_attention_context8_pipeline = nil;
        g_attention_context16_pipeline = nil;
        g_attention_logits_pipeline = nil;
        g_attention_logits4_pipeline = nil;
        g_attention_importance_pipeline = nil;
        g_attention_logits_batch_pipeline = nil;
        g_attention_logits_batch4_pipeline = nil;
        g_attention_importance_batch_pipeline = nil;
        g_attention_importance_reduce_heads_pipeline = nil;
        g_affine_u32_q_only_matvec_pipeline = nil;
        g_full_q_only_norm_rope_pipeline = nil;
        g_argmax_pipeline = nil;
        g_residual_add_round_pre_b_round_norm_pipeline = nil;
        g_residual_add_round_pre_b_round_pipeline = nil;
        g_residual_add_round_pipeline = nil;
        g_round_bf16_pipeline = nil;
        g_linear_gate_qkvz_mat_pipeline = nil;
        g_linear_gate_mat_pipeline = nil;
        g_linear_gate_pipeline = nil;
        g_linear_conv_state_qkvz_mat_pipeline = nil;
        g_linear_conv_state_mat_pipeline = nil;
        g_linear_delta_scan4_pipeline = nil;
        g_linear_delta_scan2_pipeline = nil;
        g_linear_delta_scan_pipeline = nil;
        g_linear_scan_params_pipeline = nil;
        g_linear_delta_pipeline = nil;
        g_linear_qk_norm_kq_mat_pipeline = nil;
        g_linear_qk_norm_mat_pipeline = nil;
        g_linear_qk_norm_pipeline = nil;
        g_linear_conv_stateful_qkvz_mat_pipeline = nil;
        g_linear_conv_stateful_mat_pipeline = nil;
        g_linear_conv_qkvz_mat_pipeline = nil;
        g_linear_conv_mat_pipeline = nil;
        g_linear_conv_pipeline = nil;
        g_full_k_norm_rope_mat_pipeline = nil;
        g_full_k_norm_rope_mat_pack_f16_pipeline = nil;
        g_full_q_norm_rope_mat_pipeline = nil;
        g_full_k_norm_rope_pipeline = nil;
        g_full_q_norm_rope_pipeline = nil;
        g_attention_pack_kv_head_f16x4_pipeline = nil;
        g_attention_pack_kv_all_head_f16x4_pipeline = nil;
        g_attention_pack_v_head_f16x4_pipeline = nil;
        g_attention_pack_q_group_block_head_f16x4_pipeline = nil;
        g_swiglu_packed_pair_pipeline = nil;
        g_swiglu_x4_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_rms_norm_bf16_mat_round_pipeline = nil;
        g_rms_norm_bf16_mat_pipeline = nil;
        g_rms_norm_bf16_pipeline = nil;
        g_dequant_u32_row_pipeline = nil;
        g_affine_u32_matmat8x2_pipeline = nil;
        g_affine_u32_matmat4x4_pipeline = nil;
        g_affine_u32_matmat4x2_pipeline = nil;
        g_affine_u32_matmat4_pipeline = nil;
        g_affine_u32_matmat_pipeline = nil;
        g_affine_u32_matvec_pipeline = nil;
        g_drafter_queue = nil;
        g_drafter_device = nil;
        g_init_attempted = 0;
        g_init_ok = 0;
    }
}

static id<MTLBuffer> drafter_cached_buffer(const void *ptr,
                                           uint64_t bytes,
                                           char *err,
                                           size_t errlen) {
    if (!ptr || bytes == 0 || bytes > NSUIntegerMax) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid tensor buffer");
        return nil;
    }
    NSValue *key = [NSValue valueWithPointer:ptr];
    id<MTLBuffer> buf = [g_buffer_cache objectForKey:key];
    if (buf) return buf;
    buf = [g_drafter_device newBufferWithBytes:ptr
                                        length:(NSUInteger)bytes
                                       options:MTLResourceStorageModeShared];
    if (!buf) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to upload tensor buffer");
        return nil;
    }
    [g_buffer_cache setObject:buf forKey:key];
    return buf;
}

static int drafter_mps_matmul_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_MATMUL");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_sync_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_SYNC");
        enabled = env && strcmp(env, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_sync_mps_command_buffer(id<MTLCommandBuffer> *cbp,
                                           const char *label,
                                           char *err,
                                           size_t errlen) {
    if (!cbp || !*cbp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter missing command buffer for MPS sync");
        return -1;
    }
    id<MTLCommandBuffer> cb = *cbp;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *fallback = [NSString stringWithFormat:@"native Metal drafter %s failed",
                              label ? label : "MPS sync"];
        NSString *msg = cb.error.localizedDescription ?: fallback;
        return drafter_metal_fail(err, errlen, msg);
    }
    cb = [g_drafter_queue commandBuffer];
    if (!cb) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create command buffer after MPS sync");
        return -1;
    }
    *cbp = cb;
    return 0;
}

static int drafter_mps_prepare_warmup_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_PREPARE_WARMUP");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_min_n_vec(void) {
    static int initialized;
    static int min_n_vec;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_MIN_N_VEC");
        min_n_vec = env ? atoi(env) : 1;
        if (min_n_vec < 1) min_n_vec = 1;
        initialized = 1;
    }
    return min_n_vec;
}

static int drafter_mps_min_rows(void) {
    static int initialized;
    static int min_rows;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_MIN_ROWS");
        min_rows = env ? atoi(env) : 1;
        if (min_rows < 1) min_rows = 1;
        initialized = 1;
    }
    return min_rows;
}

static int drafter_mps_shape_warmup_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_SHAPE_WARMUP");
        enabled = env && strcmp(env, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_mlp_pair_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_MLP_PAIR");
        enabled = env && strcmp(env, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_shape_warmup_count(void) {
    static int initialized;
    static int count;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_SHAPE_WARMUP_COUNT");
        count = env ? atoi(env) : 4;
        if (count < 1) count = 1;
        if (count > 8) count = 8;
        initialized = 1;
    }
    return count;
}

static int drafter_quant_tile_mode(void) {
    static int initialized;
    static int mode;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_QUANT_TILE");
        if (env && strcmp(env, "4x4") == 0) {
            mode = 44;
        } else if (env && strcmp(env, "8x2") == 0) {
            mode = 82;
        } else {
            mode = 42;
        }
        initialized = 1;
    }
    return mode;
}

static int drafter_mps_f16_weights_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_F16_WEIGHTS");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_linear_qkvz_pair_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_LINEAR_QKVZ_PAIR");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_logits_tiled_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_LOGITS_TILED");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static NSUInteger drafter_logits_tiled_threads(void) {
    static int initialized;
    static NSUInteger threads;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_LOGITS_TILED_THREADS");
        int value = env ? atoi(env) : 32;
        if (value != 16 && value != 32 && value != 64 &&
            value != 128 && value != 256) {
            value = 32;
        }
        threads = (NSUInteger)value;
        initialized = 1;
    }
    return threads;
}

static int drafter_mps_transposed_weights_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_TRANSPOSED_WEIGHTS");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_causal_attention_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_CAUSAL_ATTENTION");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_causal_attention_max_n_vec(void) {
    static int initialized;
    static int max_n_vec;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_CAUSAL_ATTENTION_MAX_N_VEC");
        if (env && env[0]) {
            max_n_vec = atoi(env);
        } else {
            max_n_vec = 8192;
        }
        if (max_n_vec < 0) max_n_vec = 0;
        initialized = 1;
    }
    return max_n_vec;
}

static int drafter_mps_causal_attention_block_rows(void) {
    static int initialized;
    static int block_rows;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_CAUSAL_ATTENTION_BLOCK_ROWS");
        block_rows = env ? atoi(env) : 1024;
        if (block_rows < 0) block_rows = 0;
        initialized = 1;
    }
    return block_rows;
}

static int drafter_mps_grouped_causal_attention_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_GROUPED_CAUSAL_ATTENTION");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_attention_f16_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_ATTENTION_F16");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_attention_prob_f16_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_ATTENTION_PROB_F16");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_attention_prepack_k_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_ATTENTION_PREPACK_K");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_private_scratch_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_PRIVATE_SCRATCH");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_mps_causal_attention_softmax_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_MPS_CAUSAL_SOFTMAX");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static NSUInteger drafter_attention_softmax_threads(void) {
    static int initialized;
    static NSUInteger threads;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_ATTENTION_SOFTMAX_THREADS");
        int v = env ? atoi(env) : 1024;
        if (v != 128 && v != 256 && v != 512 && v != 1024) v = 1024;
        threads = (NSUInteger)v;
        initialized = 1;
    }
    return threads;
}

static NSUInteger drafter_linear_delta_threads(void) {
    static int initialized;
    static NSUInteger threads;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_LINEAR_DELTA_THREADS");
        int value = env ? atoi(env) : 32;
        if (value != 8 && value != 16 && value != 32 &&
            value != 64 && value != 128 && value != 256) {
            value = 32;
        }
        threads = (NSUInteger)value;
        initialized = 1;
    }
    return threads;
}

static int drafter_linear_scan2_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_LINEAR_SCAN2");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_linear_scan4_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_LINEAR_SCAN4");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_batch_profile_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_BATCH_PROFILE");
        enabled = env && strcmp(env, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_attention_detail_profile_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_ATTENTION_PROFILE");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_token_profile_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_TOKEN_PROFILE");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_parallel_context_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_PARALLEL_CONTEXT");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_logits4_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_LOGITS4");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_importance_logits4_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_IMPORTANCE_LOGITS4");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_context4_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_CONTEXT4");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_context8_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_CONTEXT8");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_context16_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_METAL_CONTEXT16");
        enabled = !(env && strcmp(env, "0") == 0);
        initialized = 1;
    }
    return enabled;
}

static int drafter_batch_hidden_debug_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_DEBUG_BATCH_HIDDEN");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static int drafter_batch_hidden_debug_layer(void) {
    static int initialized;
    static int layer;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_DEBUG_BATCH_LAYER");
        layer = env && env[0] ? atoi(env) : -1;
        initialized = 1;
    }
    return layer;
}

static int drafter_scan_debug_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *env = getenv("DS4_DRAFTER_DEBUG_SCAN");
        enabled = env && strcmp(env, "1") == 0;
        initialized = 1;
    }
    return enabled;
}

static double drafter_now_ms(void) {
    return [[NSDate date] timeIntervalSince1970] * 1000.0;
}

static int drafter_profile_flush_when(int enabled,
                                      id<MTLCommandBuffer> *cbp,
                                      id<MTLComputeCommandEncoder> *encp,
                                      double *bucket_ms,
                                      const char *label,
                                      char *err,
                                      size_t errlen) {
    if (!enabled) return 0;
    if (!cbp || !*cbp || !encp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid batch profile state");
        return -1;
    }
    double t0 = drafter_now_ms();
    if (*encp) {
        [*encp endEncoding];
        *encp = nil;
    }
    id<MTLCommandBuffer> cb = *cbp;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *fallback = [NSString stringWithFormat:@"native Metal drafter batch profile %s failed",
                              label ? label : "phase"];
        NSString *msg = cb.error.localizedDescription ?: fallback;
        return drafter_metal_fail(err, errlen, msg);
    }
    if (bucket_ms) *bucket_ms += drafter_now_ms() - t0;
    cb = [g_drafter_queue commandBuffer];
    if (!cb) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create profiled command buffer");
        return -1;
    }
    *cbp = cb;
    *encp = [cb computeCommandEncoder];
    if (!*encp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create profiled compute encoder");
        return -1;
    }
    return 0;
}

static int drafter_profile_flush(id<MTLCommandBuffer> *cbp,
                                 id<MTLComputeCommandEncoder> *encp,
                                 double *bucket_ms,
                                 const char *label,
                                 char *err,
                                 size_t errlen) {
    return drafter_profile_flush_when(drafter_batch_profile_enabled(),
                                      cbp, encp, bucket_ms, label,
                                      err, errlen);
}

static int drafter_attention_detail_flush(id<MTLCommandBuffer> *cbp,
                                          id<MTLComputeCommandEncoder> *encp,
                                          double *bucket_ms,
                                          const char *label,
                                          char *err,
                                          size_t errlen) {
    if (!drafter_attention_detail_profile_enabled()) return 0;
    if (!cbp || !*cbp || !encp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid attention detail profile state");
        return -1;
    }
    double t0 = drafter_now_ms();
    if (*encp) {
        [*encp endEncoding];
        *encp = nil;
    }
    id<MTLCommandBuffer> cb = *cbp;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *fallback = [NSString stringWithFormat:@"native Metal drafter attention detail profile %s failed",
                              label ? label : "phase"];
        NSString *msg = cb.error.localizedDescription ?: fallback;
        return drafter_metal_fail(err, errlen, msg);
    }
    if (bucket_ms) *bucket_ms += drafter_now_ms() - t0;
    cb = [g_drafter_queue commandBuffer];
    if (!cb) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create attention detail profile command buffer");
        return -1;
    }
    *cbp = cb;
    *encp = nil;
    return 0;
}

static int drafter_debug_batch_hidden_flush(id<MTLCommandBuffer> *cbp,
                                            id<MTLComputeCommandEncoder> *encp,
                                            id<MTLBuffer> buf,
                                            const char *label,
                                            int layer,
                                            int n_vec,
                                            int width,
                                            char *err,
                                            size_t errlen) {
    if (!drafter_batch_hidden_debug_enabled()) return 0;
    if (!cbp || !*cbp || !encp || !buf || n_vec <= 0 || width <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid batch hidden debug state");
        return -1;
    }
    if (*encp) {
        [*encp endEncoding];
        *encp = nil;
    }
    id<MTLCommandBuffer> cb = *cbp;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter batch hidden debug command buffer failed";
        return drafter_metal_fail(err, errlen, msg);
    }
    const float *data = (const float *)[buf contents];
    const NSUInteger offset = ((NSUInteger)n_vec - 1u) * (NSUInteger)width;
    double sum = 0.0;
    float min_v = data[offset];
    float max_v = data[offset];
    int finite = 0;
    int nan_count = 0;
    for (int i = 0; i < width; i++) {
        float v = data[offset + (NSUInteger)i];
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
            "NATIVE_BATCH_LAYER layer=%d stage=%s finite=%d nan=%d min=%.9g max=%.9g sum=%.9g first=%.9g\n",
            layer, label ? label : "out", finite, nan_count, min_v, max_v,
            sum, data[offset]);
    *cbp = [g_drafter_queue commandBuffer];
    if (!*cbp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create command buffer after batch hidden debug");
        return -1;
    }
    *encp = [*cbp computeCommandEncoder];
    if (!*encp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create encoder after batch hidden debug");
        return -1;
    }
    return 0;
}

static id<MTLComputePipelineState> drafter_attention_context_causal_fused_pipeline(void) {
    return g_attention_context_causal_fused_mat_pipeline;
}

static inline float drafter_host_bf16_to_f32(uint16_t v) {
    uint32_t raw = (uint32_t)v << 16;
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

static uint16_t drafter_host_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static int drafter_fill_dense_affine_buffer(
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        float *dense,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        cols <= 0 || bits <= 0 || group_size <= 0 || !dense) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid dense affine job");
        return -1;
    }
    if (job->w_bytes < (uint64_t)job->rows * (uint64_t)job->packed_cols * sizeof(uint32_t) ||
        job->scales_bytes < (uint64_t)job->rows * (uint64_t)job->groups * sizeof(uint16_t) ||
        job->biases_bytes < (uint64_t)job->rows * (uint64_t)job->groups * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter dense affine buffer size overflow");
        return -1;
    }

    const uint32_t *w = (const uint32_t *)job->w_data;
    const uint16_t *scales = (const uint16_t *)job->scales_data;
    const uint16_t *biases = (const uint16_t *)job->biases_data;
    const uint32_t pack = (uint32_t)(32 / bits);
    const uint32_t mask = ((uint32_t)1 << (uint32_t)bits) - 1u;
    const int group_aligned = (group_size % (int)pack) == 0;
    for (int row = 0; row < job->rows; row++) {
        const uint32_t *w_row = w + (size_t)row * (size_t)job->packed_cols;
        const uint16_t *s_row = scales + (size_t)row * (size_t)job->groups;
        const uint16_t *b_row = biases + (size_t)row * (size_t)job->groups;
        float *d_row = dense + (size_t)row * (size_t)cols;
        for (int pc = 0; pc < job->packed_cols; pc++) {
            const uint32_t packed = w_row[pc];
            const uint32_t g_pack = (uint32_t)((uint64_t)pc * pack / (uint32_t)group_size);
            const float scale_pack = drafter_host_bf16_to_f32(s_row[g_pack]);
            const float bias_pack = drafter_host_bf16_to_f32(b_row[g_pack]);
            for (uint32_t lane = 0; lane < pack; lane++) {
                const uint32_t col = (uint32_t)pc * pack + lane;
                if (col >= (uint32_t)cols) continue;
                float scale = scale_pack;
                float bias = bias_pack;
                if (!group_aligned) {
                    const uint32_t g = col / (uint32_t)group_size;
                    scale = drafter_host_bf16_to_f32(s_row[g]);
                    bias = drafter_host_bf16_to_f32(b_row[g]);
                }
                const uint32_t q = (packed >> (lane * (uint32_t)bits)) & mask;
                d_row[col] = (float)q * scale + bias;
            }
        }
    }
    return 0;
}

static id<MTLBuffer> drafter_upload_dense_buffer(id<MTLBuffer> staging,
                                                 NSUInteger dense_bytes,
                                                 char *err,
                                                 size_t errlen) {
    if (!staging || dense_bytes == 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid dense upload buffer");
        return nil;
    }
    id<MTLBuffer> buf = [g_drafter_device newBufferWithLength:dense_bytes
                                                      options:MTLResourceStorageModePrivate];
    if (!buf) return staging;
    id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging
            sourceOffset:0
                toBuffer:buf
       destinationOffset:0
                    size:dense_bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter dense affine upload failed";
        drafter_metal_fail(err, errlen, msg);
        return nil;
    }
    return buf;
}

static id<MTLBuffer> drafter_dense_affine_buffer(
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        cols <= 0 || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid dense affine job");
        return nil;
    }
    const uint64_t dense_elems = (uint64_t)job->rows * (uint64_t)cols;
    if (dense_elems > NSUIntegerMax / sizeof(float)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter dense affine buffer size overflow");
        return nil;
    }

    NSValue *key = [NSValue valueWithPointer:job->w_data];
    id<MTLBuffer> buf = [g_dense_affine_cache objectForKey:key];
    if (buf) return buf;

    const int f16_weights = drafter_mps_f16_weights_enabled();
    const NSUInteger dense_bytes = (NSUInteger)dense_elems * (f16_weights ? sizeof(uint16_t) : sizeof(float));
    id<MTLBuffer> staging = [g_drafter_device newBufferWithLength:dense_bytes
                                                          options:MTLResourceStorageModeShared];
    if (!staging) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate dense affine buffer");
        return nil;
    }
    if (f16_weights) {
        float *tmp = (float *)malloc((size_t)dense_elems * sizeof(float));
        if (!tmp) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate dense fp16 staging");
            return nil;
        }
        if (drafter_fill_dense_affine_buffer(job, cols, bits, group_size,
                                             tmp, err, errlen) != 0) {
            free(tmp);
            return nil;
        }
        uint16_t *half = (uint16_t *)[staging contents];
        for (uint64_t i = 0; i < dense_elems; i++) {
            half[i] = drafter_host_f32_to_f16(tmp[i]);
        }
        free(tmp);
    } else {
        if (drafter_fill_dense_affine_buffer(job, cols, bits, group_size,
                                             (float *)[staging contents], err,
                                             errlen) != 0) {
            return nil;
        }
    }
    buf = drafter_upload_dense_buffer(staging, dense_bytes, err, errlen);
    if (!buf) return nil;
    [g_dense_affine_cache setObject:buf forKey:key];
    return buf;
}

static id<MTLBuffer> drafter_dense_affine_transposed_buffer(
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        cols <= 0 || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid transposed dense affine job");
        return nil;
    }
    const uint64_t dense_elems = (uint64_t)job->rows * (uint64_t)cols;
    if (dense_elems > NSUIntegerMax / sizeof(float)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter transposed dense affine size overflow");
        return nil;
    }

    NSString *key = [NSString stringWithFormat:@"trans:%p:%d:%d:%d:%d",
                     job->w_data, job->rows, cols, bits, group_size];
    id<MTLBuffer> buf = [g_dense_affine_cache objectForKey:key];
    if (buf) return buf;

    float *tmp = (float *)malloc((size_t)dense_elems * sizeof(float));
    if (!tmp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate transposed dense staging");
        return nil;
    }
    if (drafter_fill_dense_affine_buffer(job, cols, bits, group_size,
                                         tmp, err, errlen) != 0) {
        free(tmp);
        return nil;
    }

    const int f16_weights = drafter_mps_f16_weights_enabled();
    const NSUInteger elem_bytes = f16_weights ? sizeof(uint16_t) : sizeof(float);
    const NSUInteger dense_bytes = (NSUInteger)dense_elems * elem_bytes;
    id<MTLBuffer> staging = [g_drafter_device newBufferWithLength:dense_bytes
                                                          options:MTLResourceStorageModeShared];
    if (!staging) {
        free(tmp);
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate transposed dense affine buffer");
        return nil;
    }
    if (f16_weights) {
        uint16_t *half = (uint16_t *)[staging contents];
        for (int row = 0; row < job->rows; row++) {
            const float *src = tmp + (size_t)row * (size_t)cols;
            for (int col = 0; col < cols; col++) {
                half[(size_t)col * (size_t)job->rows + (size_t)row] =
                    drafter_host_f32_to_f16(src[col]);
            }
        }
    } else {
        float *dst = (float *)[staging contents];
        for (int row = 0; row < job->rows; row++) {
            const float *src = tmp + (size_t)row * (size_t)cols;
            for (int col = 0; col < cols; col++) {
                dst[(size_t)col * (size_t)job->rows + (size_t)row] = src[col];
            }
        }
    }
    free(tmp);

    buf = drafter_upload_dense_buffer(staging, dense_bytes, err, errlen);
    if (!buf) return nil;
    [g_dense_affine_cache setObject:buf forKey:key];
    return buf;
}

static id<MTLBuffer> drafter_dense_affine_pair_buffer(
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *b,
        int cols,
        int bits,
        int group_size,
        int *rows_out,
        char *err,
        size_t errlen) {
    if (!a || !b || !a->w_data || !b->w_data || cols <= 0 ||
        a->rows <= 0 || b->rows <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid dense affine pair job");
        return nil;
    }
    const int rows = a->rows + b->rows;
    if (rows_out) *rows_out = rows;
    const uint64_t dense_elems = (uint64_t)rows * (uint64_t)cols;
    if (dense_elems > NSUIntegerMax / sizeof(float)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter dense affine pair size overflow");
        return nil;
    }
    NSString *key = [NSString stringWithFormat:@"pair:%p:%p:%d:%d:%d",
                     a->w_data, b->w_data, cols, bits, group_size];
    id<MTLBuffer> buf = [g_dense_affine_cache objectForKey:key];
    if (buf) return buf;

    const int f16_weights = drafter_mps_f16_weights_enabled();
    const NSUInteger dense_bytes = (NSUInteger)dense_elems * (f16_weights ? sizeof(uint16_t) : sizeof(float));
    id<MTLBuffer> staging = [g_drafter_device newBufferWithLength:dense_bytes
                                                          options:MTLResourceStorageModeShared];
    if (!staging) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate dense affine pair buffer");
        return nil;
    }
    if (f16_weights) {
        float *tmp = (float *)malloc((size_t)dense_elems * sizeof(float));
        if (!tmp) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate dense affine pair staging");
            return nil;
        }
        if (drafter_fill_dense_affine_buffer(a, cols, bits, group_size,
                                             tmp, err, errlen) != 0 ||
            drafter_fill_dense_affine_buffer(b, cols, bits, group_size,
                                             tmp + (size_t)a->rows * (size_t)cols,
                                             err, errlen) != 0) {
            free(tmp);
            return nil;
        }
        uint16_t *half = (uint16_t *)[staging contents];
        for (uint64_t i = 0; i < dense_elems; i++) {
            half[i] = drafter_host_f32_to_f16(tmp[i]);
        }
        free(tmp);
    } else {
        float *dense = (float *)[staging contents];
        if (drafter_fill_dense_affine_buffer(a, cols, bits, group_size,
                                             dense, err, errlen) != 0 ||
            drafter_fill_dense_affine_buffer(b, cols, bits, group_size,
                                             dense + (size_t)a->rows * (size_t)cols,
                                             err, errlen) != 0) {
            return nil;
        }
    }
    buf = drafter_upload_dense_buffer(staging, dense_bytes, err, errlen);
    if (!buf) return nil;
    [g_dense_affine_cache setObject:buf forKey:key];
    return buf;
}

static id<MTLBuffer> drafter_dense_affine_pair_transposed_buffer(
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *b,
        int cols,
        int bits,
        int group_size,
        int *rows_out,
        char *err,
        size_t errlen) {
    if (!a || !b || !a->w_data || !b->w_data || cols <= 0 ||
        a->rows <= 0 || b->rows <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid transposed dense affine pair job");
        return nil;
    }
    const int rows = a->rows + b->rows;
    if (rows_out) *rows_out = rows;
    const uint64_t dense_elems = (uint64_t)rows * (uint64_t)cols;
    if (dense_elems > NSUIntegerMax / sizeof(float)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter transposed dense affine pair size overflow");
        return nil;
    }
    NSString *key = [NSString stringWithFormat:@"trans-pair:%p:%p:%d:%d:%d:%d",
                     a->w_data, b->w_data, rows, cols, bits, group_size];
    id<MTLBuffer> buf = [g_dense_affine_cache objectForKey:key];
    if (buf) return buf;

    float *tmp = (float *)malloc((size_t)dense_elems * sizeof(float));
    if (!tmp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate transposed pair staging");
        return nil;
    }
    if (drafter_fill_dense_affine_buffer(a, cols, bits, group_size,
                                         tmp, err, errlen) != 0 ||
        drafter_fill_dense_affine_buffer(b, cols, bits, group_size,
                                         tmp + (size_t)a->rows * (size_t)cols,
                                         err, errlen) != 0) {
        free(tmp);
        return nil;
    }

    const int f16_weights = drafter_mps_f16_weights_enabled();
    const NSUInteger elem_bytes = f16_weights ? sizeof(uint16_t) : sizeof(float);
    const NSUInteger dense_bytes = (NSUInteger)dense_elems * elem_bytes;
    id<MTLBuffer> staging = [g_drafter_device newBufferWithLength:dense_bytes
                                                          options:MTLResourceStorageModeShared];
    if (!staging) {
        free(tmp);
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate transposed pair buffer");
        return nil;
    }
    if (f16_weights) {
        uint16_t *half = (uint16_t *)[staging contents];
        for (int row = 0; row < rows; row++) {
            const float *src = tmp + (size_t)row * (size_t)cols;
            for (int col = 0; col < cols; col++) {
                half[(size_t)col * (size_t)rows + (size_t)row] =
                    drafter_host_f32_to_f16(src[col]);
            }
        }
    } else {
        float *dst = (float *)[staging contents];
        for (int row = 0; row < rows; row++) {
            const float *src = tmp + (size_t)row * (size_t)cols;
            for (int col = 0; col < cols; col++) {
                dst[(size_t)col * (size_t)rows + (size_t)row] = src[col];
            }
        }
    }
    free(tmp);

    buf = drafter_upload_dense_buffer(staging, dense_bytes, err, errlen);
    if (!buf) return nil;
    [g_dense_affine_cache setObject:buf forKey:key];
    return buf;
}

static MPSMatrixMultiplication *drafter_mps_matmul_kernel_ex(int result_rows,
                                                             int result_cols,
                                                             int interior_cols,
                                                             BOOL transpose_left,
                                                             BOOL transpose_right,
                                                             double alpha) {
    uint64_t alpha_key = 0;
    memcpy(&alpha_key, &alpha, sizeof(alpha_key));
    NSUInteger h = (NSUInteger)result_rows * 1000003u ^
                   (NSUInteger)result_cols * 9176u ^
                   (NSUInteger)interior_cols * 131u ^
                   (transpose_left ? 0x9e37u : 0u) ^
                   (transpose_right ? 0x85ebu : 0u) ^
                   (NSUInteger)(alpha_key ^ (alpha_key >> 32));
    ds4_drafter_mps_matmul_fast_cache_entry *entry =
        &g_mps_matmul_fast_cache[h & (DS4_DRAFTER_MPS_MATMUL_FAST_CACHE_SIZE - 1)];
    if (entry->used &&
        entry->result_rows == result_rows &&
        entry->result_cols == result_cols &&
        entry->interior_cols == interior_cols &&
        entry->transpose_left == (transpose_left ? 1 : 0) &&
        entry->transpose_right == (transpose_right ? 1 : 0) &&
        entry->alpha_key == alpha_key &&
        entry->matrix_multiplication) {
        return entry->matrix_multiplication;
    }
    NSString *key = [NSString stringWithFormat:@"%d:%d:%d:%d:%d:%.9g",
                     result_rows, result_cols, interior_cols,
                     transpose_left ? 1 : 0, transpose_right ? 1 : 0,
                     alpha];
    MPSMatrixMultiplication *mm = [g_mps_matmul_cache objectForKey:key];
    if (mm) {
        entry->used = 1;
        entry->result_rows = result_rows;
        entry->result_cols = result_cols;
        entry->interior_cols = interior_cols;
        entry->transpose_left = transpose_left ? 1 : 0;
        entry->transpose_right = transpose_right ? 1 : 0;
        entry->alpha_key = alpha_key;
        entry->matrix_multiplication = mm;
        return mm;
    }
    mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_drafter_device
                                           transposeLeft:transpose_left
                                          transposeRight:transpose_right
                                              resultRows:(NSUInteger)result_rows
                                           resultColumns:(NSUInteger)result_cols
                                         interiorColumns:(NSUInteger)interior_cols
                                                   alpha:alpha
                                                    beta:0.0];
    if (mm) {
        [g_mps_matmul_cache setObject:mm forKey:key];
        entry->used = 1;
        entry->result_rows = result_rows;
        entry->result_cols = result_cols;
        entry->interior_cols = interior_cols;
        entry->transpose_left = transpose_left ? 1 : 0;
        entry->transpose_right = transpose_right ? 1 : 0;
        entry->alpha_key = alpha_key;
        entry->matrix_multiplication = mm;
    }
    return mm;
}

static MPSMatrix *drafter_mps_cached_matrix_offset(id<MTLBuffer> buf,
                                                   NSUInteger offset,
                                                   int rows,
                                                   int cols,
                                                   NSUInteger row_bytes,
                                                   MPSDataType data_type) {
    if (!buf || rows <= 0 || cols <= 0 || row_bytes == 0) return nil;
    void *buffer_key = (__bridge void *)buf;
    NSUInteger h = ((NSUInteger)(uintptr_t)buffer_key >> 4) ^
                   (offset >> 4) ^
                   (NSUInteger)rows * 1000003u ^
                   (NSUInteger)cols * 9176u ^
                   (row_bytes >> 2) ^
                   (NSUInteger)data_type * 131u;
    ds4_drafter_mps_matrix_fast_cache_entry *entry =
        &g_mps_matrix_fast_cache[h & (DS4_DRAFTER_MPS_MATRIX_FAST_CACHE_SIZE - 1)];
    if (entry->used &&
        entry->buffer == buffer_key &&
        entry->offset == offset &&
        entry->rows == rows &&
        entry->cols == cols &&
        entry->row_bytes == row_bytes &&
        entry->data_type == data_type &&
        entry->matrix) {
        return entry->matrix;
    }
    NSString *key = [NSString stringWithFormat:@"%p:%llu:%d:%d:%llu:%u",
                     buffer_key,
                     (unsigned long long)offset, rows, cols,
                     (unsigned long long)row_bytes,
                     (unsigned)data_type];
    MPSMatrix *matrix = [g_mps_matrix_cache objectForKey:key];
    if (matrix) {
        entry->used = 1;
        entry->buffer = buffer_key;
        entry->offset = offset;
        entry->rows = rows;
        entry->cols = cols;
        entry->row_bytes = row_bytes;
        entry->data_type = data_type;
        entry->matrix = matrix;
        return matrix;
    }
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                              columns:(NSUInteger)cols
                                             rowBytes:row_bytes
                                             dataType:data_type];
    matrix = [[MPSMatrix alloc] initWithBuffer:buf offset:offset descriptor:desc];
    if (matrix) {
        [g_mps_matrix_cache setObject:matrix forKey:key];
        entry->used = 1;
        entry->buffer = buffer_key;
        entry->offset = offset;
        entry->rows = rows;
        entry->cols = cols;
        entry->row_bytes = row_bytes;
        entry->data_type = data_type;
        entry->matrix = matrix;
    }
    return matrix;
}

static MPSMatrix *drafter_mps_cached_matrix(id<MTLBuffer> buf,
                                            int rows,
                                            int cols,
                                            NSUInteger row_bytes,
                                            MPSDataType data_type) {
    return drafter_mps_cached_matrix_offset(buf, 0, rows, cols, row_bytes,
                                            data_type);
}

static int drafter_mps_warmup_matmul_shape(MPSMatrixMultiplication *mm,
                                           id<MTLBuffer> dense,
                                           int n_vec,
                                           int cols,
                                           int rows,
                                           int transposed_weights,
                                           char *err,
                                           size_t errlen) {
    if (!drafter_mps_shape_warmup_enabled()) return 0;
    if (!mm || !dense || n_vec <= 0 || cols <= 0 || rows <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid MPS shape warmup");
        return -1;
    }
    NSString *key = [NSString stringWithFormat:@"%d:%d:%d:%d", n_vec, cols,
                     rows, transposed_weights ? 1 : 0];
    if ([g_mps_matmul_shape_warmup_cache objectForKey:key]) return 0;
    const NSUInteger x_bytes = (NSUInteger)n_vec * (NSUInteger)cols * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)n_vec * (NSUInteger)rows * sizeof(float);
    id<MTLBuffer> xbuf = [g_drafter_device newBufferWithLength:x_bytes
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> outbuf = [g_drafter_device newBufferWithLength:out_bytes
                                                          options:MTLResourceStorageModeShared];
    if (!xbuf || !outbuf) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate MPS shape warmup buffers");
        return -1;
    }
    float *x = (float *)[xbuf contents];
    const NSUInteger n = (NSUInteger)n_vec * (NSUInteger)cols;
    for (NSUInteger i = 0; i < n; i++) {
        x[i] = (float)((int)(i % 17u) - 8) * 0.01f;
    }

    MPSMatrixDescriptor *left_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n_vec
                                              columns:(NSUInteger)cols
                                             rowBytes:(NSUInteger)cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    const int f16_weights = drafter_mps_f16_weights_enabled();
    MPSMatrixDescriptor *right_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)(transposed_weights ? cols : rows)
                                              columns:(NSUInteger)(transposed_weights ? rows : cols)
                                             rowBytes:(NSUInteger)(transposed_weights ? rows : cols) * (f16_weights ? sizeof(uint16_t) : sizeof(float))
                                             dataType:f16_weights ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
    MPSMatrixDescriptor *out_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n_vec
                                              columns:(NSUInteger)rows
                                             rowBytes:(NSUInteger)rows * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix *left = [[MPSMatrix alloc] initWithBuffer:xbuf descriptor:left_desc];
    MPSMatrix *right = [[MPSMatrix alloc] initWithBuffer:dense descriptor:right_desc];
    MPSMatrix *result = [[MPSMatrix alloc] initWithBuffer:outbuf descriptor:out_desc];
    if (!left || !right || !result) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS shape warmup matrices");
        return -1;
    }
    const int warmup_count = drafter_mps_shape_warmup_count();
    for (int i = 0; i < warmup_count; i++) {
        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:left rightMatrix:right resultMatrix:result];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter MPS shape warmup command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
    }
    [g_mps_matmul_shape_warmup_cache setObject:@YES forKey:key];
    return 0;
}

static int drafter_encode_affine_mps_matmat(
        id<MTLCommandBuffer> cb,
        const ds4_drafter_metal_affine_job *job,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    if (!drafter_mps_matmul_enabled()) return 1;
    if (!cb || !job || !xbuf || !outbuf || n_vec <= 0 || cols <= 0 ||
        job->rows <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid MPS matmul job");
        return -1;
    }
    const int transposed_weights = drafter_mps_transposed_weights_enabled();
    id<MTLBuffer> dense = transposed_weights ?
        drafter_dense_affine_transposed_buffer(job, cols, bits, group_size, err, errlen) :
        drafter_dense_affine_buffer(job, cols, bits, group_size, err, errlen);
    if (!dense) return -1;

    const int f16_weights = drafter_mps_f16_weights_enabled();
    MPSMatrix *left = drafter_mps_cached_matrix(xbuf, n_vec, cols,
                                                (NSUInteger)cols * sizeof(float),
                                                MPSDataTypeFloat32);
    MPSMatrix *right = drafter_mps_cached_matrix(dense,
                                                 transposed_weights ? cols : job->rows,
                                                 transposed_weights ? job->rows : cols,
                                                 (NSUInteger)(transposed_weights ? job->rows : cols) * (f16_weights ? sizeof(uint16_t) : sizeof(float)),
                                                 f16_weights ? MPSDataTypeFloat16 : MPSDataTypeFloat32);
    MPSMatrix *result = drafter_mps_cached_matrix(outbuf, n_vec, job->rows,
                                                  (NSUInteger)job->rows * sizeof(float),
                                                  MPSDataTypeFloat32);
    MPSMatrixMultiplication *mm =
        drafter_mps_matmul_kernel_ex(n_vec, job->rows, cols, NO,
                                     transposed_weights ? NO : YES, 1.0);
    if (!mm || !left || !right || !result) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS matrix objects");
        return -1;
    }
    if (drafter_mps_warmup_matmul_shape(mm, dense, n_vec, cols, job->rows,
                                        transposed_weights, err, errlen) != 0) {
        return -1;
    }
    [mm encodeToCommandBuffer:cb leftMatrix:left rightMatrix:right resultMatrix:result];
    return 0;
}

static int drafter_can_encode_affine_mps_matmat(
        const ds4_drafter_metal_affine_job *job,
        int n_vec) {
    return drafter_mps_matmul_enabled() &&
           n_vec >= drafter_mps_min_n_vec() &&
           job && job->rows >= drafter_mps_min_rows();
}

static int drafter_encode_affine_mps_matmat_batch(
        id<MTLCommandBuffer> *cbp,
        id<MTLComputeCommandEncoder> *enc,
        const ds4_drafter_metal_affine_job **jobs,
        const int *cols,
        id<MTLBuffer> const *xbufs,
        id<MTLBuffer> const *outbufs,
        int count,
        int n_vec,
        int bits,
        int group_size,
        char *err,
        size_t errlen) {
    if (!cbp || !*cbp || !enc || !jobs || !cols || !xbufs || !outbufs ||
        count <= 0) {
        return 1;
    }
    for (int i = 0; i < count; i++) {
        if (!drafter_can_encode_affine_mps_matmat(jobs[i], n_vec)) return 1;
    }
    if (*enc) {
        [*enc endEncoding];
        *enc = nil;
    }
    for (int i = 0; i < count; i++) {
        int rc = drafter_encode_affine_mps_matmat(*cbp, jobs[i], n_vec,
                                                  cols[i], bits, group_size,
                                                  xbufs[i], outbufs[i],
                                                  err, errlen);
        if (rc != 0) return rc < 0 ? -1 : 1;
    }
    if (drafter_mps_sync_enabled()) {
        if (drafter_sync_mps_command_buffer(cbp, "MPS affine batch sync", err, errlen) != 0) return -1;
    }
    *enc = [*cbp computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after MPS affine batch");
        return -1;
    }
    return 0;
}

static int drafter_encode_affine_pair_mps_matmat(
        id<MTLCommandBuffer> *cbp,
        id<MTLComputeCommandEncoder> *enc,
        const ds4_drafter_metal_affine_job *a,
        const ds4_drafter_metal_affine_job *b,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    if (!drafter_mps_matmul_enabled()) return 1;
    if (!cbp || !*cbp || !enc || !a || !b || !xbuf || !outbuf ||
        n_vec <= 0 || cols <= 0 || a->rows <= 0 || b->rows <= 0) {
        return 1;
    }
    int rows = 0;
    const int transposed_weights = drafter_mps_transposed_weights_enabled();
    id<MTLBuffer> dense = transposed_weights ?
        drafter_dense_affine_pair_transposed_buffer(a, b, cols, bits,
                                                    group_size, &rows,
                                                    err, errlen) :
        drafter_dense_affine_pair_buffer(a, b, cols, bits, group_size,
                                         &rows, err, errlen);
    if (!dense) return -1;
    if (*enc) {
        [*enc endEncoding];
        *enc = nil;
    }
    const int f16_weights = drafter_mps_f16_weights_enabled();
    MPSMatrix *left = drafter_mps_cached_matrix(xbuf, n_vec, cols,
                                                (NSUInteger)cols * sizeof(float),
                                                MPSDataTypeFloat32);
    MPSMatrix *right = drafter_mps_cached_matrix(dense,
                                                 transposed_weights ? cols : rows,
                                                 transposed_weights ? rows : cols,
                                                 (NSUInteger)(transposed_weights ? rows : cols) * (f16_weights ? sizeof(uint16_t) : sizeof(float)),
                                                 f16_weights ? MPSDataTypeFloat16 : MPSDataTypeFloat32);
    MPSMatrix *result = drafter_mps_cached_matrix(outbuf, n_vec, rows,
                                                  (NSUInteger)rows * sizeof(float),
                                                  MPSDataTypeFloat32);
    MPSMatrixMultiplication *mm =
        drafter_mps_matmul_kernel_ex(n_vec, rows, cols, NO,
                                     transposed_weights ? NO : YES, 1.0);
    if (!mm || !left || !right || !result) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS pair matrix objects");
        return -1;
    }
    if (drafter_mps_warmup_matmul_shape(mm, dense, n_vec, cols, rows,
                                        transposed_weights, err, errlen) != 0) {
        return -1;
    }
    [mm encodeToCommandBuffer:*cbp leftMatrix:left rightMatrix:right resultMatrix:result];
    if (drafter_mps_sync_enabled()) {
        if (drafter_sync_mps_command_buffer(cbp, "MPS affine pair sync", err, errlen) != 0) return -1;
    }
    *enc = [*cbp computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after MPS pair");
        return -1;
    }
    return 0;
}

static int drafter_encode_mps_causal_attention_block_mat(
        id<MTLCommandBuffer> cb,
        id<MTLComputeCommandEncoder> *enc,
        int n_vec,
        int block_rows,
        char *err,
        size_t errlen) {
    if (!cb || !enc || n_vec <= 0 || block_rows <= 0) return 1;
    if (block_rows > n_vec) block_rows = n_vec;
    const int direct_kv = 0;
    const NSUInteger q_head_bytes = (NSUInteger)block_rows * 256u * sizeof(float);
    const NSUInteger kv_head_bytes = (NSUInteger)n_vec * 256u * sizeof(float);
    const NSUInteger logits_bytes = (NSUInteger)block_rows * (NSUInteger)n_vec * sizeof(float);
    if (drafter_ensure_private_buffer(&g_attention_q_head_buffer, &g_attention_q_head_bytes,
                              q_head_bytes, "attention query block head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_ctx_head_buffer, &g_attention_ctx_head_bytes,
                              q_head_bytes, "attention context block head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                              logits_bytes, "attention probability block matrix", err, errlen) != 0) {
        return -1;
    }
    if (!direct_kv &&
        (drafter_ensure_private_buffer(&g_attention_k_head_buffer, &g_attention_k_head_bytes,
                               kv_head_bytes, "attention key head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_v_head_buffer, &g_attention_v_head_bytes,
                               kv_head_bytes, "attention value head", err, errlen) != 0)) {
        return -1;
    }

    const uint32_t n_vec_u = (uint32_t)n_vec;
    for (uint32_t qh = 0; qh < 8u; qh++) {
        ds4_drafter_metal_head_block_args kv_args = {
            .n_ctx = n_vec_u,
            .q_start = 0,
            .q_count = 0,
            .qh = qh,
            .kvh = qh / 4u,
        };
        if (!direct_kv) {
            if (!*enc) {
                *enc = [cb computeCommandEncoder];
                if (!*enc) {
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create blocked attention KV pack encoder");
                    return -1;
                }
            }
            [*enc setComputePipelineState:g_attention_pack_kv_head_mat_pipeline];
            [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
            [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:1];
            [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:2];
            [*enc setBuffer:g_attention_k_head_buffer offset:0 atIndex:3];
            [*enc setBuffer:g_attention_v_head_buffer offset:0 atIndex:4];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 256u, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
        } else if (*enc) {
            [*enc endEncoding];
            *enc = nil;
        }

        for (int q_start = 0; q_start < n_vec; q_start += block_rows) {
            const int q_count = (q_start + block_rows <= n_vec) ? block_rows : (n_vec - q_start);
            const int kv_count = q_start + q_count;
            ds4_drafter_metal_head_block_args block_args = {
                .n_ctx = (uint32_t)kv_count,
                .q_start = (uint32_t)q_start,
                .q_count = (uint32_t)q_count,
                .qh = qh,
                .kvh = qh / 4u,
                .q_input_start = (uint32_t)q_start,
                .q_output_start = (uint32_t)q_start,
            };

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create blocked attention Q pack encoder");
                return -1;
            }
            [*enc setComputePipelineState:g_attention_pack_q_block_head_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [*enc setBuffer:g_attention_q_head_buffer offset:0 atIndex:2];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_count * 256u, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;

            MPSMatrix *q_mat = drafter_mps_cached_matrix(g_attention_q_head_buffer, q_count, 256,
                                                         256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *k_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_keys_buffer,
                                                 (NSUInteger)(qh / 4u) * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                drafter_mps_cached_matrix(g_attention_k_head_buffer, kv_count, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *v_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_values_buffer,
                                                 (NSUInteger)(qh / 4u) * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                drafter_mps_cached_matrix(g_attention_v_head_buffer, kv_count, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *prob_mat = drafter_mps_cached_matrix(g_attention_logits_buffer, q_count, kv_count,
                                                            (NSUInteger)kv_count * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *ctx_mat = drafter_mps_cached_matrix(g_attention_ctx_head_buffer, q_count, 256,
                                                           256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrixMultiplication *qk_mm =
                drafter_mps_matmul_kernel_ex(q_count, kv_count, 256, NO, YES, 1.0 / 16.0);
            MPSMatrixMultiplication *pv_mm =
                drafter_mps_matmul_kernel_ex(q_count, 256, kv_count, NO, NO, 1.0);
            if (!q_mat || !k_prefix_mat || !v_prefix_mat || !prob_mat || !ctx_mat || !qk_mm || !pv_mm) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create blocked MPS attention matrices");
                return -1;
            }

            if (*enc) {
                [*enc endEncoding];
                *enc = nil;
            }
            [qk_mm encodeToCommandBuffer:cb leftMatrix:q_mat rightMatrix:k_prefix_mat resultMatrix:prob_mat];

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create blocked attention softmax encoder");
                return -1;
            }
            [*enc setComputePipelineState:g_attention_causal_softmax_block_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
            const NSUInteger softmax_threads = drafter_attention_softmax_threads();
            [*enc setThreadgroupMemoryLength:softmax_threads * sizeof(float) atIndex:0];
            [*enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(softmax_threads, 1, 1)];
            [*enc endEncoding];
            *enc = nil;

            if (*enc) {
                [*enc endEncoding];
                *enc = nil;
            }
            [pv_mm encodeToCommandBuffer:cb leftMatrix:prob_mat rightMatrix:v_prefix_mat resultMatrix:ctx_mat];

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create blocked attention gate encoder");
                return -1;
            }
            [*enc setComputePipelineState:g_attention_unpack_gate_block_head_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:g_attention_ctx_head_buffer offset:0 atIndex:1];
            [*enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
            [*enc setBuffer:g_attention_out_buffer offset:0 atIndex:3];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_count * 256u, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
        }
    }

    *enc = [cb computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after blocked MPS attention");
        return -1;
    }
    return 0;
}

static int drafter_encode_mps_causal_attention_group_block_mat(
        id<MTLCommandBuffer> *cbp,
        id<MTLComputeCommandEncoder> *enc,
        int n_vec,
        int block_rows,
        char *err,
        size_t errlen) {
    if (!drafter_mps_grouped_causal_attention_enabled()) return 1;
    if (!cbp || !*cbp || !enc || n_vec <= 0 || block_rows <= 0) return 1;
    if (!g_attention_pack_q_group_block_head_mat_pipeline ||
        !g_attention_causal_softmax_group_block_mat_pipeline ||
        !g_attention_unpack_gate_group_block_head_mat_pipeline) {
        return 1;
    }
    id<MTLCommandBuffer> cb = *cbp;
    double prof_kv_pack_ms = 0.0;
    double prof_q_pack_ms = 0.0;
    double prof_qk_ms = 0.0;
    double prof_softmax_ms = 0.0;
    double prof_pv_ms = 0.0;
    double prof_gate_ms = 0.0;
    if (block_rows > n_vec) block_rows = n_vec;
    const int direct_kv = 0;
    const int use_f16_attention = !direct_kv && drafter_mps_attention_f16_enabled();
    const int use_f16_all_kv_pack = 0;
    const int use_f16_prepacked_k = use_f16_attention &&
        !use_f16_all_kv_pack &&
        g_attention_prepacked_k_f16_n_ctx == n_vec &&
        g_attention_pack_v_head_f16x4_pipeline;
    const int use_f16_prob = use_f16_attention &&
        drafter_mps_attention_prob_f16_enabled() &&
        g_attention_causal_softmax_group_block_f16_pipeline &&
        g_attention_unpack_gate_group_block_head_f16_pipeline;
    const NSUInteger q_group_head_bytes = 4u * (NSUInteger)block_rows * 256u * sizeof(float);
    const NSUInteger kv_head_bytes = (NSUInteger)n_vec * 256u * sizeof(float);
    const NSUInteger q_group_head_f16_bytes = 4u * (NSUInteger)block_rows * 256u * sizeof(uint16_t);
    const NSUInteger kv_head_f16_bytes = (use_f16_all_kv_pack || use_f16_prepacked_k ? 2u : 1u) * (NSUInteger)n_vec * 256u * sizeof(uint16_t);
    const NSUInteger logits_bytes = 4u * (NSUInteger)block_rows * (NSUInteger)n_vec * sizeof(float);
    const NSUInteger logits_f16_bytes = 4u * (NSUInteger)block_rows * (NSUInteger)n_vec * sizeof(uint16_t);
    if (!use_f16_attention &&
        drafter_ensure_private_buffer(&g_attention_q_head_buffer, &g_attention_q_head_bytes,
                                      q_group_head_bytes, "attention grouped query block head", err, errlen) != 0) {
        return -1;
    }
    if (!use_f16_prob &&
        (drafter_ensure_private_buffer(&g_attention_ctx_head_buffer, &g_attention_ctx_head_bytes,
                                       q_group_head_bytes, "attention grouped context block head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                       logits_bytes, "attention grouped probability block matrix", err, errlen) != 0)) {
        return -1;
    }
    if (!direct_kv && !use_f16_attention &&
        (drafter_ensure_private_buffer(&g_attention_k_head_buffer, &g_attention_k_head_bytes,
                                       kv_head_bytes, "attention key head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_v_head_buffer, &g_attention_v_head_bytes,
                                       kv_head_bytes, "attention value head", err, errlen) != 0)) {
        return -1;
    }
    if (use_f16_attention &&
        (drafter_ensure_private_buffer(&g_attention_q_head_f16_buffer,
                                       &g_attention_q_head_f16_bytes,
                                       q_group_head_f16_bytes, "attention grouped query f16 block head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_k_head_f16_buffer,
                                       &g_attention_k_head_f16_bytes,
                                       kv_head_f16_bytes, "attention key f16 head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_v_head_f16_buffer,
                                       &g_attention_v_head_f16_bytes,
                                       kv_head_f16_bytes, "attention value f16 head", err, errlen) != 0)) {
        return -1;
    }
    if (use_f16_prob &&
        (drafter_ensure_private_buffer(&g_attention_logits_f16_buffer,
                                       &g_attention_logits_f16_bytes,
                                       logits_f16_bytes, "attention grouped probability f16 block matrix", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_ctx_head_f16_buffer,
                                       &g_attention_ctx_head_f16_bytes,
                                       q_group_head_f16_bytes, "attention grouped context f16 block head", err, errlen) != 0)) {
        return -1;
    }

    const uint32_t n_vec_u = (uint32_t)n_vec;
    if (use_f16_all_kv_pack) {
        ds4_drafter_metal_head_block_args kv_args = {
            .n_ctx = n_vec_u,
            .q_start = 0,
            .q_count = 0,
            .qh = 0,
            .kvh = 0,
        };
        if (!*enc) {
            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped attention all-KV pack encoder");
                return -1;
            }
        }
        [*enc setComputePipelineState:g_attention_pack_kv_all_head_f16x4_pipeline];
        [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
        [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:1];
        [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:2];
        [*enc setBuffer:g_attention_k_head_f16_buffer offset:0 atIndex:3];
        [*enc setBuffer:g_attention_v_head_f16_buffer offset:0 atIndex:4];
        [*enc dispatchThreads:MTLSizeMake(2u * (NSUInteger)n_vec * 64u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [*enc endEncoding];
        *enc = nil;
        if (drafter_attention_detail_flush(&cb, enc, &prof_kv_pack_ms,
                                           "grouped attention all-kv pack",
                                           err, errlen) != 0) {
            return -1;
        }
    }

    for (uint32_t kvh = 0; kvh < 2u; kvh++) {
        ds4_drafter_metal_head_block_args kv_args = {
            .n_ctx = n_vec_u,
            .q_start = 0,
            .q_count = 0,
            .qh = kvh * 4u,
            .kvh = kvh,
        };
        if (!direct_kv && !use_f16_all_kv_pack) {
            if (!*enc) {
                *enc = [cb computeCommandEncoder];
                if (!*enc) {
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped attention KV pack encoder");
                    return -1;
                }
            }
            if (use_f16_prepacked_k) {
                [*enc setComputePipelineState:g_attention_pack_v_head_f16x4_pipeline];
                [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
                [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:1];
                [*enc setBuffer:g_attention_v_head_f16_buffer offset:0 atIndex:2];
                [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 64u, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                id<MTLComputePipelineState> kv_pack_pipeline = use_f16_attention ?
                (g_attention_pack_kv_head_f16x4_pipeline ? g_attention_pack_kv_head_f16x4_pipeline : g_attention_pack_kv_head_f16_pipeline) :
                g_attention_pack_kv_head_mat_pipeline;
                [*enc setComputePipelineState:kv_pack_pipeline];
                [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
                [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:1];
                [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:2];
                [*enc setBuffer:use_f16_attention ? g_attention_k_head_f16_buffer : g_attention_k_head_buffer
                            offset:0 atIndex:3];
                [*enc setBuffer:use_f16_attention ? g_attention_v_head_f16_buffer : g_attention_v_head_buffer
                            offset:0 atIndex:4];
                [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * (use_f16_attention && g_attention_pack_kv_head_f16x4_pipeline ? 64u : 256u), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [*enc endEncoding];
            *enc = nil;
            if (drafter_attention_detail_flush(&cb, enc, &prof_kv_pack_ms,
                                               "grouped attention kv pack",
                                               err, errlen) != 0) {
                return -1;
            }
        } else if (*enc) {
            [*enc endEncoding];
            *enc = nil;
            if (drafter_attention_detail_flush(&cb, enc, &prof_kv_pack_ms,
                                               "grouped attention direct kv sync",
                                               err, errlen) != 0) {
                return -1;
            }
        }

        for (int q_start = 0; q_start < n_vec; q_start += block_rows) {
            const int q_count = (q_start + block_rows <= n_vec) ? block_rows : (n_vec - q_start);
            const int kv_count = q_start + q_count;
            const int q_rows = 4 * q_count;
            ds4_drafter_metal_head_block_args block_args = {
                .n_ctx = (uint32_t)kv_count,
                .q_start = (uint32_t)q_start,
                .q_count = (uint32_t)q_count,
                .qh = kvh * 4u,
                .kvh = kvh,
                .q_input_start = (uint32_t)q_start,
                .q_output_start = (uint32_t)q_start,
            };

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped attention Q pack encoder");
                return -1;
            }
            id<MTLComputePipelineState> q_pack_pipeline = use_f16_attention ?
                (g_attention_pack_q_group_block_head_f16x4_pipeline ? g_attention_pack_q_group_block_head_f16x4_pipeline : g_attention_pack_q_group_block_head_f16_pipeline) :
                g_attention_pack_q_group_block_head_mat_pipeline;
            [*enc setComputePipelineState:q_pack_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [*enc setBuffer:use_f16_attention ? g_attention_q_head_f16_buffer : g_attention_q_head_buffer
                        offset:0 atIndex:2];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_rows * (use_f16_attention && g_attention_pack_q_group_block_head_f16x4_pipeline ? 64u : 256u), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
            if (drafter_attention_detail_flush(&cb, enc, &prof_q_pack_ms,
                                               "grouped attention q pack",
                                               err, errlen) != 0) {
                return -1;
            }

            MPSMatrix *q_mat = use_f16_attention ?
                drafter_mps_cached_matrix(g_attention_q_head_f16_buffer, q_rows, 256,
                                          256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_q_head_buffer, q_rows, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *k_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_keys_buffer,
                                                 (NSUInteger)kvh * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                (use_f16_attention ?
                 ((use_f16_all_kv_pack || use_f16_prepacked_k) ?
                  drafter_mps_cached_matrix_offset(g_attention_k_head_f16_buffer,
                                                   (NSUInteger)kvh * (NSUInteger)n_vec * 256u * sizeof(uint16_t),
                                                   kv_count, 256,
                                                   256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                  drafter_mps_cached_matrix(g_attention_k_head_f16_buffer, kv_count, 256,
                                            256u * sizeof(uint16_t), MPSDataTypeFloat16)) :
                 drafter_mps_cached_matrix(g_attention_k_head_buffer, kv_count, 256,
                                           256u * sizeof(float), MPSDataTypeFloat32));
            MPSMatrix *prob_mat = use_f16_prob ?
                drafter_mps_cached_matrix(g_attention_logits_f16_buffer, q_rows, kv_count,
                                          (NSUInteger)kv_count * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_logits_buffer, q_rows, kv_count,
                                          (NSUInteger)kv_count * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *v_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_values_buffer,
                                                 (NSUInteger)kvh * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                (use_f16_attention ?
                 (use_f16_all_kv_pack ?
                  drafter_mps_cached_matrix_offset(g_attention_v_head_f16_buffer,
                                                   (NSUInteger)kvh * (NSUInteger)n_vec * 256u * sizeof(uint16_t),
                                                   kv_count, 256,
                                                   256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                  drafter_mps_cached_matrix(g_attention_v_head_f16_buffer, kv_count, 256,
                                            256u * sizeof(uint16_t), MPSDataTypeFloat16)) :
                 drafter_mps_cached_matrix(g_attention_v_head_buffer, kv_count, 256,
                                           256u * sizeof(float), MPSDataTypeFloat32));
            MPSMatrix *ctx_mat = use_f16_prob ?
                drafter_mps_cached_matrix(g_attention_ctx_head_f16_buffer, q_rows, 256,
                                          256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_ctx_head_buffer, q_rows, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrixMultiplication *qk_mm =
                drafter_mps_matmul_kernel_ex(q_rows, kv_count, 256, NO, YES, 1.0 / 16.0);
            MPSMatrixMultiplication *pv_mm =
                drafter_mps_matmul_kernel_ex(q_rows, 256, kv_count, NO, NO, 1.0);
            if (!q_mat || !k_prefix_mat || !v_prefix_mat || !prob_mat || !ctx_mat || !qk_mm || !pv_mm) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped blocked MPS attention matrices");
                return -1;
            }

            if (*enc) {
                [*enc endEncoding];
                *enc = nil;
            }
            [qk_mm encodeToCommandBuffer:cb leftMatrix:q_mat rightMatrix:k_prefix_mat resultMatrix:prob_mat];
            if (drafter_attention_detail_flush(&cb, enc, &prof_qk_ms,
                                               "grouped attention qk",
                                               err, errlen) != 0) {
                return -1;
            }

            if (!use_f16_prob &&
                drafter_mps_causal_attention_softmax_enabled() &&
                g_mps_matrix_softmax &&
                g_attention_causal_mask_group_block_mat_pipeline) {
                *enc = [cb computeCommandEncoder];
                if (!*enc) {
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped blocked attention mask encoder");
                    return -1;
                }
                [*enc setComputePipelineState:g_attention_causal_mask_group_block_mat_pipeline];
                [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
                [*enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                [*enc dispatchThreads:MTLSizeMake((NSUInteger)kv_count, (NSUInteger)q_rows, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [*enc endEncoding];
                *enc = nil;
                g_mps_matrix_softmax.sourceRows = (NSUInteger)q_rows;
                g_mps_matrix_softmax.sourceColumns = (NSUInteger)kv_count;
                [g_mps_matrix_softmax encodeToCommandBuffer:cb
                                                 inputMatrix:prob_mat
                                                resultMatrix:prob_mat];
                if (drafter_attention_detail_flush(&cb, enc, &prof_softmax_ms,
                                                   "grouped attention mps softmax",
                                                   err, errlen) != 0) {
                    return -1;
                }
            } else {
                *enc = [cb computeCommandEncoder];
                if (!*enc) {
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped blocked attention softmax encoder");
                    return -1;
                }
                [*enc setComputePipelineState:use_f16_prob ?
                    g_attention_causal_softmax_group_block_f16_pipeline :
                    g_attention_causal_softmax_group_block_mat_pipeline];
                [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
                [*enc setBuffer:use_f16_prob ? g_attention_logits_f16_buffer : g_attention_logits_buffer
                            offset:0 atIndex:1];
                const NSUInteger softmax_threads = drafter_attention_softmax_threads();
                [*enc setThreadgroupMemoryLength:softmax_threads * sizeof(float) atIndex:0];
                [*enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_rows, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(softmax_threads, 1, 1)];
                [*enc endEncoding];
                *enc = nil;
                if (drafter_attention_detail_flush(&cb, enc, &prof_softmax_ms,
                                                   "grouped attention softmax",
                                                   err, errlen) != 0) {
                    return -1;
                }
            }

            if (*enc) {
                [*enc endEncoding];
                *enc = nil;
            }
            [pv_mm encodeToCommandBuffer:cb leftMatrix:prob_mat rightMatrix:v_prefix_mat resultMatrix:ctx_mat];
            if (drafter_attention_detail_flush(&cb, enc, &prof_pv_ms,
                                               "grouped attention pv",
                                               err, errlen) != 0) {
                return -1;
            }

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped blocked attention gate encoder");
                return -1;
            }
            const int use_f16_gate_x4 = use_f16_prob &&
                g_attention_unpack_gate_group_block_head_f16x4_pipeline;
            [*enc setComputePipelineState:use_f16_prob ?
                (use_f16_gate_x4 ? g_attention_unpack_gate_group_block_head_f16x4_pipeline :
                 g_attention_unpack_gate_group_block_head_f16_pipeline) :
                g_attention_unpack_gate_group_block_head_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:use_f16_prob ? g_attention_ctx_head_f16_buffer : g_attention_ctx_head_buffer
                        offset:0 atIndex:1];
            [*enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
            [*enc setBuffer:g_attention_out_buffer offset:0 atIndex:3];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_rows * (use_f16_gate_x4 ? 64u : 256u), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
            if (drafter_attention_detail_flush(&cb, enc, &prof_gate_ms,
                                               "grouped attention gate",
                                               err, errlen) != 0) {
                return -1;
            }
        }
    }

    if (drafter_attention_detail_profile_enabled()) {
        fprintf(stderr,
                "native Metal attention detail: kv_pack=%.3fms q_pack=%.3fms qk=%.3fms softmax=%.3fms pv=%.3fms gate=%.3fms\n",
                prof_kv_pack_ms,
                prof_q_pack_ms,
                prof_qk_ms,
                prof_softmax_ms,
                prof_pv_ms,
                prof_gate_ms);
    }
    *cbp = cb;
    *enc = [cb computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after grouped blocked MPS attention");
        return -1;
    }
    return 0;
}

static int drafter_encode_mps_causal_attention_group_chunk_mat(
        id<MTLCommandBuffer> cb,
        id<MTLComputeCommandEncoder> *enc,
        int n_ctx,
        int q_start_global,
        int q_count_total,
        int block_rows,
        char *err,
        size_t errlen) {
    if (!drafter_mps_grouped_causal_attention_enabled()) return 1;
    if (!cb || !enc || n_ctx <= 0 || q_start_global < 0 ||
        q_count_total <= 0 || q_start_global + q_count_total > n_ctx ||
        block_rows <= 0) {
        return 1;
    }
    if (!g_attention_pack_q_group_block_head_mat_pipeline ||
        !g_attention_causal_softmax_group_block_mat_pipeline ||
        !g_attention_unpack_gate_group_block_head_mat_pipeline) {
        return 1;
    }
    if (block_rows > q_count_total) block_rows = q_count_total;
    const int direct_kv = 0;
    const int use_f16_attention = !direct_kv && drafter_mps_attention_f16_enabled();
    const int use_f16_all_kv_pack = 0;
    const int use_f16_prob = use_f16_attention &&
        drafter_mps_attention_prob_f16_enabled() &&
        g_attention_causal_softmax_group_block_f16_pipeline &&
        g_attention_unpack_gate_group_block_head_f16_pipeline;
    const NSUInteger q_group_head_bytes = 4u * (NSUInteger)block_rows * 256u * sizeof(float);
    const NSUInteger kv_head_bytes = (NSUInteger)n_ctx * 256u * sizeof(float);
    const NSUInteger q_group_head_f16_bytes = 4u * (NSUInteger)block_rows * 256u * sizeof(uint16_t);
    const NSUInteger kv_head_f16_bytes = (use_f16_all_kv_pack ? 2u : 1u) * (NSUInteger)n_ctx * 256u * sizeof(uint16_t);
    const NSUInteger logits_bytes = 4u * (NSUInteger)block_rows * (NSUInteger)n_ctx * sizeof(float);
    const NSUInteger logits_f16_bytes = 4u * (NSUInteger)block_rows * (NSUInteger)n_ctx * sizeof(uint16_t);
    if (!use_f16_attention &&
        drafter_ensure_private_buffer(&g_attention_q_head_buffer, &g_attention_q_head_bytes,
                                      q_group_head_bytes, "attention grouped chunk query block head", err, errlen) != 0) {
        return -1;
    }
    if (!use_f16_prob &&
        (drafter_ensure_private_buffer(&g_attention_ctx_head_buffer, &g_attention_ctx_head_bytes,
                                       q_group_head_bytes, "attention grouped chunk context block head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                       logits_bytes, "attention grouped chunk probability block matrix", err, errlen) != 0)) {
        return -1;
    }
    if (!direct_kv && !use_f16_attention &&
        (drafter_ensure_private_buffer(&g_attention_k_head_buffer, &g_attention_k_head_bytes,
                                       kv_head_bytes, "attention chunk key head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_v_head_buffer, &g_attention_v_head_bytes,
                                       kv_head_bytes, "attention chunk value head", err, errlen) != 0)) {
        return -1;
    }
    if (use_f16_attention &&
        (drafter_ensure_private_buffer(&g_attention_q_head_f16_buffer,
                                       &g_attention_q_head_f16_bytes,
                                       q_group_head_f16_bytes, "attention grouped chunk query f16 block head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_k_head_f16_buffer,
                                       &g_attention_k_head_f16_bytes,
                                       kv_head_f16_bytes, "attention chunk key f16 head", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_v_head_f16_buffer,
                                       &g_attention_v_head_f16_bytes,
                                       kv_head_f16_bytes, "attention chunk value f16 head", err, errlen) != 0)) {
        return -1;
    }
    if (use_f16_prob &&
        (drafter_ensure_private_buffer(&g_attention_logits_f16_buffer,
                                       &g_attention_logits_f16_bytes,
                                       logits_f16_bytes, "attention grouped chunk probability f16 block matrix", err, errlen) != 0 ||
         drafter_ensure_private_buffer(&g_attention_ctx_head_f16_buffer,
                                       &g_attention_ctx_head_f16_bytes,
                                       q_group_head_f16_bytes, "attention grouped chunk context f16 block head", err, errlen) != 0)) {
        return -1;
    }

    if (use_f16_all_kv_pack) {
        ds4_drafter_metal_head_block_args kv_args = {
            .n_ctx = (uint32_t)n_ctx,
            .q_start = 0,
            .q_count = 0,
            .qh = 0,
            .kvh = 0,
            .q_input_start = 0,
            .q_output_start = 0,
        };
        if (!*enc) {
            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention all-KV pack encoder");
                return -1;
            }
        }
        [*enc setComputePipelineState:g_attention_pack_kv_all_head_f16x4_pipeline];
        [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
        [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:1];
        [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:2];
        [*enc setBuffer:g_attention_k_head_f16_buffer offset:0 atIndex:3];
        [*enc setBuffer:g_attention_v_head_f16_buffer offset:0 atIndex:4];
        [*enc dispatchThreads:MTLSizeMake(2u * (NSUInteger)n_ctx * 64u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [*enc endEncoding];
        *enc = nil;
    }

    for (uint32_t kvh = 0; kvh < 2u; kvh++) {
        ds4_drafter_metal_head_block_args kv_args = {
            .n_ctx = (uint32_t)n_ctx,
            .q_start = 0,
            .q_count = 0,
            .qh = kvh * 4u,
            .kvh = kvh,
            .q_input_start = 0,
            .q_output_start = 0,
        };
        if (!direct_kv && !use_f16_all_kv_pack) {
            if (!*enc) {
                *enc = [cb computeCommandEncoder];
                if (!*enc) {
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention KV pack encoder");
                    return -1;
                }
            }
            id<MTLComputePipelineState> kv_pack_pipeline = use_f16_attention ?
                (g_attention_pack_kv_head_f16x4_pipeline ? g_attention_pack_kv_head_f16x4_pipeline : g_attention_pack_kv_head_f16_pipeline) :
                g_attention_pack_kv_head_mat_pipeline;
            [*enc setComputePipelineState:kv_pack_pipeline];
            [*enc setBytes:&kv_args length:sizeof(kv_args) atIndex:0];
            [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:1];
            [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:2];
            [*enc setBuffer:use_f16_attention ? g_attention_k_head_f16_buffer : g_attention_k_head_buffer
                        offset:0 atIndex:3];
            [*enc setBuffer:use_f16_attention ? g_attention_v_head_f16_buffer : g_attention_v_head_buffer
                        offset:0 atIndex:4];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_ctx * (use_f16_attention && g_attention_pack_kv_head_f16x4_pipeline ? 64u : 256u), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
        } else if (*enc) {
            [*enc endEncoding];
            *enc = nil;
        }

        for (int q_local_start = 0; q_local_start < q_count_total; q_local_start += block_rows) {
            const int q_count = (q_local_start + block_rows <= q_count_total) ?
                block_rows : (q_count_total - q_local_start);
            const int q_rows = 4 * q_count;
            const int q_global_start = q_start_global + q_local_start;
            const int kv_count = q_global_start + q_count;
            ds4_drafter_metal_head_block_args block_args = {
                .n_ctx = (uint32_t)kv_count,
                .q_start = (uint32_t)q_global_start,
                .q_count = (uint32_t)q_count,
                .qh = kvh * 4u,
                .kvh = kvh,
                .q_input_start = (uint32_t)q_local_start,
                .q_output_start = (uint32_t)q_local_start,
            };

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention Q pack encoder");
                return -1;
            }
            id<MTLComputePipelineState> q_pack_pipeline = use_f16_attention ?
                (g_attention_pack_q_group_block_head_f16x4_pipeline ? g_attention_pack_q_group_block_head_f16x4_pipeline : g_attention_pack_q_group_block_head_f16_pipeline) :
                g_attention_pack_q_group_block_head_mat_pipeline;
            [*enc setComputePipelineState:q_pack_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [*enc setBuffer:use_f16_attention ? g_attention_q_head_f16_buffer : g_attention_q_head_buffer
                        offset:0 atIndex:2];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_rows * (use_f16_attention && g_attention_pack_q_group_block_head_f16x4_pipeline ? 64u : 256u), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;

            MPSMatrix *q_mat = use_f16_attention ?
                drafter_mps_cached_matrix(g_attention_q_head_f16_buffer, q_rows, 256,
                                          256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_q_head_buffer, q_rows, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *k_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_keys_buffer,
                                                 (NSUInteger)kvh * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                (use_f16_attention ?
                 (use_f16_all_kv_pack ?
                  drafter_mps_cached_matrix_offset(g_attention_k_head_f16_buffer,
                                                   (NSUInteger)kvh * (NSUInteger)n_ctx * 256u * sizeof(uint16_t),
                                                   kv_count, 256,
                                                   256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                  drafter_mps_cached_matrix(g_attention_k_head_f16_buffer, kv_count, 256,
                                            256u * sizeof(uint16_t), MPSDataTypeFloat16)) :
                 drafter_mps_cached_matrix(g_attention_k_head_buffer, kv_count, 256,
                                           256u * sizeof(float), MPSDataTypeFloat32));
            MPSMatrix *v_prefix_mat = direct_kv ?
                drafter_mps_cached_matrix_offset(g_attention_values_buffer,
                                                 (NSUInteger)kvh * 256u * sizeof(float),
                                                 kv_count, 256,
                                                 512u * sizeof(float),
                                                 MPSDataTypeFloat32) :
                (use_f16_attention ?
                 (use_f16_all_kv_pack ?
                  drafter_mps_cached_matrix_offset(g_attention_v_head_f16_buffer,
                                                   (NSUInteger)kvh * (NSUInteger)n_ctx * 256u * sizeof(uint16_t),
                                                   kv_count, 256,
                                                   256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                  drafter_mps_cached_matrix(g_attention_v_head_f16_buffer, kv_count, 256,
                                            256u * sizeof(uint16_t), MPSDataTypeFloat16)) :
                 drafter_mps_cached_matrix(g_attention_v_head_buffer, kv_count, 256,
                                           256u * sizeof(float), MPSDataTypeFloat32));
            MPSMatrix *prob_mat = use_f16_prob ?
                drafter_mps_cached_matrix(g_attention_logits_f16_buffer, q_rows, kv_count,
                                          (NSUInteger)kv_count * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_logits_buffer, q_rows, kv_count,
                                          (NSUInteger)kv_count * sizeof(float), MPSDataTypeFloat32);
            MPSMatrix *ctx_mat = use_f16_prob ?
                drafter_mps_cached_matrix(g_attention_ctx_head_f16_buffer, q_rows, 256,
                                          256u * sizeof(uint16_t), MPSDataTypeFloat16) :
                drafter_mps_cached_matrix(g_attention_ctx_head_buffer, q_rows, 256,
                                          256u * sizeof(float), MPSDataTypeFloat32);
            MPSMatrixMultiplication *qk_mm =
                drafter_mps_matmul_kernel_ex(q_rows, kv_count, 256, NO, YES, 1.0 / 16.0);
            MPSMatrixMultiplication *pv_mm =
                drafter_mps_matmul_kernel_ex(q_rows, 256, kv_count, NO, NO, 1.0);
            if (!q_mat || !k_prefix_mat || !v_prefix_mat || !prob_mat || !ctx_mat || !qk_mm || !pv_mm) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention matrices");
                return -1;
            }

            [qk_mm encodeToCommandBuffer:cb leftMatrix:q_mat rightMatrix:k_prefix_mat resultMatrix:prob_mat];

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention softmax encoder");
                return -1;
            }
            [*enc setComputePipelineState:use_f16_prob ?
                g_attention_causal_softmax_group_block_f16_pipeline :
                g_attention_causal_softmax_group_block_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:use_f16_prob ? g_attention_logits_f16_buffer : g_attention_logits_buffer
                        offset:0 atIndex:1];
            const NSUInteger softmax_threads = drafter_attention_softmax_threads();
            [*enc setThreadgroupMemoryLength:softmax_threads * sizeof(float) atIndex:0];
            [*enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(softmax_threads, 1, 1)];
            [*enc endEncoding];
            *enc = nil;

            [pv_mm encodeToCommandBuffer:cb leftMatrix:prob_mat rightMatrix:v_prefix_mat resultMatrix:ctx_mat];

            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create grouped chunk attention gate encoder");
                return -1;
            }
            const int use_f16_gate_x4 = use_f16_prob &&
                g_attention_unpack_gate_group_block_head_f16x4_pipeline;
            [*enc setComputePipelineState:use_f16_prob ?
                (use_f16_gate_x4 ? g_attention_unpack_gate_group_block_head_f16x4_pipeline :
                 g_attention_unpack_gate_group_block_head_f16_pipeline) :
                g_attention_unpack_gate_group_block_head_mat_pipeline];
            [*enc setBytes:&block_args length:sizeof(block_args) atIndex:0];
            [*enc setBuffer:use_f16_prob ? g_attention_ctx_head_f16_buffer : g_attention_ctx_head_buffer
                        offset:0 atIndex:1];
            [*enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
            [*enc setBuffer:g_attention_out_buffer offset:0 atIndex:3];
            [*enc dispatchThreads:MTLSizeMake((NSUInteger)q_rows * (use_f16_gate_x4 ? 64u : 256u), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [*enc endEncoding];
            *enc = nil;
        }
    }

    *enc = [cb computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after grouped chunk attention");
        return -1;
    }
    return 0;
}

static int drafter_encode_mps_causal_attention_mat(
        id<MTLCommandBuffer> *cbp,
        id<MTLComputeCommandEncoder> *enc,
        int n_vec,
        char *err,
        size_t errlen) {
    if (!drafter_mps_matmul_enabled() || !drafter_mps_causal_attention_enabled()) return 1;
    const int max_n_vec = drafter_mps_causal_attention_max_n_vec();
    if (!cbp || !*cbp || !enc || n_vec <= 0) return 1;
    id<MTLCommandBuffer> cb = *cbp;
    if (max_n_vec <= 0 || n_vec > max_n_vec) {
        int grouped_rc = drafter_encode_mps_causal_attention_group_block_mat(
            cbp, enc, n_vec, drafter_mps_causal_attention_block_rows(),
            err, errlen);
        if (grouped_rc <= 0) return grouped_rc;
        return drafter_encode_mps_causal_attention_block_mat(
            cb, enc, n_vec, drafter_mps_causal_attention_block_rows(),
            err, errlen);
    }

    const NSUInteger head_bytes = (NSUInteger)n_vec * 256u * sizeof(float);
    const NSUInteger logits_bytes = (NSUInteger)n_vec * (NSUInteger)n_vec * sizeof(float);
    if (drafter_ensure_private_buffer(&g_attention_q_head_buffer, &g_attention_q_head_bytes,
                              head_bytes, "attention query head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_k_head_buffer, &g_attention_k_head_bytes,
                              head_bytes, "attention key head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_v_head_buffer, &g_attention_v_head_bytes,
                              head_bytes, "attention value head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_ctx_head_buffer, &g_attention_ctx_head_bytes,
                              head_bytes, "attention context head", err, errlen) != 0 ||
        drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                              logits_bytes, "attention probability matrix", err, errlen) != 0) {
        return -1;
    }

    MPSMatrix *q_mat = drafter_mps_cached_matrix(g_attention_q_head_buffer, n_vec, 256,
                                                 256u * sizeof(float), MPSDataTypeFloat32);
    MPSMatrix *k_mat = drafter_mps_cached_matrix(g_attention_k_head_buffer, n_vec, 256,
                                                 256u * sizeof(float), MPSDataTypeFloat32);
    MPSMatrix *v_mat = drafter_mps_cached_matrix(g_attention_v_head_buffer, n_vec, 256,
                                                 256u * sizeof(float), MPSDataTypeFloat32);
    MPSMatrix *prob_mat = drafter_mps_cached_matrix(g_attention_logits_buffer, n_vec, n_vec,
                                                    (NSUInteger)n_vec * sizeof(float), MPSDataTypeFloat32);
    MPSMatrix *ctx_mat = drafter_mps_cached_matrix(g_attention_ctx_head_buffer, n_vec, 256,
                                                   256u * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixMultiplication *qk_mm =
        drafter_mps_matmul_kernel_ex(n_vec, n_vec, 256, NO, YES, 1.0 / 16.0);
    MPSMatrixMultiplication *pv_mm =
        drafter_mps_matmul_kernel_ex(n_vec, 256, n_vec, NO, NO, 1.0);
    if (!q_mat || !k_mat || !v_mat || !prob_mat || !ctx_mat || !qk_mm || !pv_mm) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS causal attention matrices");
        return -1;
    }

    const uint32_t n_vec_u = (uint32_t)n_vec;
    for (uint32_t qh = 0; qh < 8u; qh++) {
        ds4_drafter_metal_head_args head_args = {
            .n_vec = n_vec_u,
            .qh = qh,
            .kvh = qh / 4u,
        };
        if (!*enc) {
            *enc = [cb computeCommandEncoder];
            if (!*enc) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS attention pack encoder");
                return -1;
            }
        }
        [*enc setComputePipelineState:g_attention_pack_head_mat_pipeline];
        [*enc setBytes:&head_args length:sizeof(head_args) atIndex:0];
        [*enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
        [*enc setBuffer:g_attention_keys_buffer offset:0 atIndex:2];
        [*enc setBuffer:g_attention_values_buffer offset:0 atIndex:3];
        [*enc setBuffer:g_attention_q_head_buffer offset:0 atIndex:4];
        [*enc setBuffer:g_attention_k_head_buffer offset:0 atIndex:5];
        [*enc setBuffer:g_attention_v_head_buffer offset:0 atIndex:6];
        [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 256u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [*enc endEncoding];
        *enc = nil;

        [qk_mm encodeToCommandBuffer:cb leftMatrix:q_mat rightMatrix:k_mat resultMatrix:prob_mat];
        if (drafter_mps_sync_enabled()) {
            if (drafter_sync_mps_command_buffer(&cb, "MPS attention qk sync", err, errlen) != 0) return -1;
        }

        *enc = [cb computeCommandEncoder];
        if (!*enc) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS attention softmax encoder");
            return -1;
        }
        [*enc setComputePipelineState:g_attention_causal_softmax_mat_pipeline];
        [*enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [*enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [*enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [*enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_vec, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [*enc endEncoding];
        *enc = nil;

        [pv_mm encodeToCommandBuffer:cb leftMatrix:prob_mat rightMatrix:v_mat resultMatrix:ctx_mat];
        if (drafter_mps_sync_enabled()) {
            if (drafter_sync_mps_command_buffer(&cb, "MPS attention pv sync", err, errlen) != 0) return -1;
        }

        *enc = [cb computeCommandEncoder];
        if (!*enc) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to create MPS attention gate encoder");
            return -1;
        }
        [*enc setComputePipelineState:g_attention_unpack_gate_head_mat_pipeline];
        [*enc setBytes:&head_args length:sizeof(head_args) atIndex:0];
        [*enc setBuffer:g_attention_ctx_head_buffer offset:0 atIndex:1];
        [*enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
        [*enc setBuffer:g_attention_out_buffer offset:0 atIndex:3];
        [*enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 256u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [*enc endEncoding];
        *enc = nil;
    }

    *cbp = cb;
    *enc = [cb computeCommandEncoder];
    if (!*enc) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder after MPS attention");
        return -1;
    }
    return 0;
}

int ds4_drafter_metal_prepare_affine_u32(
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen) {
    if (!drafter_mps_matmul_enabled()) return 0;
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        cols <= 0 || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid affine prepare job");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        id<MTLBuffer> dense = drafter_mps_transposed_weights_enabled() ?
            drafter_dense_affine_transposed_buffer(job, cols, bits, group_size,
                                                   err, errlen) :
            drafter_dense_affine_buffer(job, cols, bits, group_size, err,
                                        errlen);
        if (!dense) return -1;
        if (drafter_mps_prepare_warmup_enabled()) {
            id<MTLBuffer> xbuf = [g_drafter_device newBufferWithLength:(NSUInteger)cols * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> outbuf = [g_drafter_device newBufferWithLength:(NSUInteger)job->rows * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
            if (!xbuf || !outbuf) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate affine warmup buffers");
                return -1;
            }
            float *xwarm = (float *)[xbuf contents];
            for (int i = 0; i < cols; i++) {
                xwarm[i] = (float)((i % 17) - 8) * 0.01f;
            }
            id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
            int rc = drafter_encode_affine_mps_matmat(cb, job, 1, cols, bits,
                                                      group_size, xbuf, outbuf,
                                                      err, errlen);
            if (rc != 0) return -1;
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter affine warmup command buffer failed";
                return drafter_metal_fail(err, errlen, msg);
            }
        }
        return 0;
    }
}

static int drafter_ensure_io_buffers(NSUInteger x_bytes,
                                     NSUInteger out_bytes,
                                     char *err,
                                     size_t errlen) {
    if (!g_x_buffer || g_x_bytes < x_bytes) {
        g_x_buffer = [g_drafter_device newBufferWithLength:x_bytes
                                                   options:MTLResourceStorageModeShared];
        g_x_bytes = g_x_buffer ? x_bytes : 0;
        if (!g_x_buffer) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate x buffer");
            return -1;
        }
    }
    if (!g_out_buffer || g_out_bytes < out_bytes) {
        g_out_buffer = [g_drafter_device newBufferWithLength:out_bytes
                                                     options:MTLResourceStorageModeShared];
        g_out_bytes = g_out_buffer ? out_bytes : 0;
        if (!g_out_buffer) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate output buffer");
            return -1;
        }
    }
    return 0;
}

static int drafter_ensure_many_out_buffer(int index,
                                          NSUInteger bytes,
                                          char *err,
                                          size_t errlen) {
    if (index < 0 || index >= 8) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter too many batched matvec jobs");
        return -1;
    }
    if (!g_many_out_buffers[index] || g_many_out_bytes[index] < bytes ||
        [g_many_out_buffers[index] storageMode] != MTLStorageModeShared) {
        g_many_out_buffers[index] = [g_drafter_device newBufferWithLength:bytes
                                                                  options:MTLResourceStorageModeShared];
        g_many_out_bytes[index] = g_many_out_buffers[index] ? bytes : 0;
        if (!g_many_out_buffers[index]) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate batched output buffer");
            return -1;
        }
    }
    return 0;
}

static int drafter_ensure_buffer(__strong id<MTLBuffer> *buf,
                                 NSUInteger *have,
                                 NSUInteger need,
                                 const char *name,
                                 char *err,
                                 size_t errlen) {
    if (!buf || !have || need == 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid %s buffer request",
                                    name ? name : "dynamic");
        return -1;
    }
    if (!*buf || *have < need || [*buf storageMode] != MTLStorageModeShared) {
        *buf = [g_drafter_device newBufferWithLength:need
                                             options:MTLResourceStorageModeShared];
        *have = *buf ? need : 0;
        if (!*buf) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate %s buffer",
                                        name ? name : "dynamic");
            return -1;
        }
    }
    return 0;
}

static int drafter_ensure_private_buffer(__strong id<MTLBuffer> *buf,
                                         NSUInteger *have,
                                         NSUInteger need,
                                         const char *name,
                                         char *err,
                                         size_t errlen) {
    if (!drafter_mps_private_scratch_enabled()) {
        return drafter_ensure_buffer(buf, have, need, name, err, errlen);
    }
    if (!buf || !have || need == 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid %s buffer request",
                                    name ? name : "dynamic");
        return -1;
    }
    if (!*buf || *have < need || [*buf storageMode] != MTLStorageModePrivate) {
        *buf = [g_drafter_device newBufferWithLength:need
                                             options:MTLResourceStorageModePrivate];
        *have = *buf ? need : 0;
        if (!*buf) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate private %s buffer",
                                        name ? name : "dynamic");
            return -1;
        }
    }
    return 0;
}

int ds4_drafter_metal_linear_attention_reset(int cache_id,
                                             char *err,
                                             size_t errlen) {
    if (cache_id < 0 || cache_id >= 24) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid linear cache id");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger conv_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_bytes = 16u * 128u * 128u * sizeof(float);
        if (drafter_ensure_buffer(&g_linear_conv_state_buffers[cache_id],
                                  &g_linear_conv_state_bytes[cache_id],
                                  conv_bytes, "linear conv state", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_delta_state_buffers[cache_id],
                                  &g_linear_delta_state_bytes[cache_id],
                                  delta_bytes, "linear delta state", err, errlen) != 0) {
            return -1;
        }
        memset([g_linear_conv_state_buffers[cache_id] contents], 0, conv_bytes);
        memset([g_linear_delta_state_buffers[cache_id] contents], 0, delta_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!w_data || !scales_data || !biases_data || !out ||
        row < 0 || rows <= 0 || row >= rows || packed_cols <= 0 ||
        cols <= 0 || groups <= 0 || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid dequant row shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger out_bytes = (NSUInteger)cols * sizeof(float);
        if (drafter_ensure_io_buffers(1, out_bytes, err, errlen) != 0) return -1;
        id<MTLBuffer> wbuf = drafter_cached_buffer(w_data, w_bytes, err, errlen);
        id<MTLBuffer> sbuf = drafter_cached_buffer(scales_data, scales_bytes, err, errlen);
        id<MTLBuffer> bbuf = drafter_cached_buffer(biases_data, biases_bytes, err, errlen);
        if (!wbuf || !sbuf || !bbuf) return -1;
        ds4_drafter_metal_dequant_row_args args = {
            .row = row,
            .rows = rows,
            .packed_cols = packed_cols,
            .cols = cols,
            .groups = groups,
            .bits = bits,
            .group_size = group_size,
        };
        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_dequant_u32_row_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:0 atIndex:1];
        [enc setBuffer:sbuf offset:0 atIndex:2];
        [enc setBuffer:bbuf offset:0 atIndex:3];
        [enc setBuffer:g_out_buffer offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter dequant row command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    ds4_drafter_metal_affine_job job = {
        .w_data = w_data,
        .w_bytes = w_bytes,
        .scales_data = scales_data,
        .scales_bytes = scales_bytes,
        .biases_data = biases_data,
        .biases_bytes = biases_bytes,
        .rows = rows,
        .packed_cols = packed_cols,
        .groups = groups,
        .out = out,
    };
    return ds4_drafter_metal_affine_u32_matvec_many(&job, 1, x, cols, bits,
                                                    group_size, err, errlen);
}

int ds4_drafter_metal_affine_u32_matvec_many(
        const ds4_drafter_metal_affine_job *jobs,
        int n_jobs,
        const float *x,
        int cols,
        int bits,
        int group_size,
        char *err,
        size_t errlen) {
    if (!jobs || n_jobs <= 0 || n_jobs > 8 || cols <= 0 ||
        bits <= 0 || group_size <= 0 || !x) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid batched affine matvec shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, 1, err, errlen) != 0) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLBuffer> wbufs[8] = { nil };
        id<MTLBuffer> sbufs[8] = { nil };
        id<MTLBuffer> bbufs[8] = { nil };
        for (int j = 0; j < n_jobs; j++) {
            const ds4_drafter_metal_affine_job *job = jobs + j;
            if (!job->w_data || !job->scales_data || !job->biases_data ||
                !job->out || job->rows <= 0 || job->packed_cols <= 0 ||
                job->groups <= 0) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid batched matvec job");
                return -1;
            }
            wbufs[j] = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
            sbufs[j] = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
            bbufs[j] = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
            if (!wbufs[j] || !sbufs[j] || !bbufs[j]) return -1;
            NSUInteger out_bytes = (NSUInteger)job->rows * sizeof(float);
            if (drafter_ensure_many_out_buffer(j, out_bytes, err, errlen) != 0) return -1;
        }

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_affine_u32_matvec_pipeline];
        [enc setBuffer:g_x_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        for (int j = 0; j < n_jobs; j++) {
            const ds4_drafter_metal_affine_job *job = jobs + j;
            ds4_drafter_metal_affine_args args = {
                .rows = job->rows,
                .packed_cols = job->packed_cols,
                .cols = cols,
                .groups = job->groups,
                .bits = bits,
                .group_size = group_size,
            };
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbufs[j] offset:0 atIndex:1];
            [enc setBuffer:sbufs[j] offset:0 atIndex:2];
            [enc setBuffer:bbufs[j] offset:0 atIndex:3];
            [enc setBuffer:g_many_out_buffers[j] offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)job->rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        for (int j = 0; j < n_jobs; j++) {
            const ds4_drafter_metal_affine_job *job = jobs + j;
            memcpy(job->out, [g_many_out_buffers[j] contents],
                   (NSUInteger)job->rows * sizeof(float));
        }
        return 0;
    }
}

int ds4_drafter_metal_affine_u32_matmat(
        const ds4_drafter_metal_affine_job *job,
        const float *x,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        float *out,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        !x || !out || n_vec <= 0 || cols <= 0 || bits <= 0 ||
        group_size <= 0 || job->rows <= 0 || job->packed_cols <= 0 ||
        job->groups <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid affine matmat shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * (NSUInteger)cols * sizeof(float);
        const NSUInteger out_bytes = (NSUInteger)n_vec * (NSUInteger)job->rows * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);
        id<MTLBuffer> wbuf = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
        id<MTLBuffer> sbuf = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
        id<MTLBuffer> bbuf = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
        if (!wbuf || !sbuf || !bbuf) return -1;
        ds4_drafter_metal_affine_args args = {
            .rows = job->rows,
            .packed_cols = job->packed_cols,
            .cols = cols,
            .groups = job->groups,
            .bits = bits,
            .group_size = group_size,
        };
        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_affine_u32_matmat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:0 atIndex:1];
        [enc setBuffer:sbuf offset:0 atIndex:2];
        [enc setBuffer:bbuf offset:0 atIndex:3];
        [enc setBuffer:g_x_buffer offset:0 atIndex:4];
        [enc setBuffer:g_out_buffer offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)job->rows,
                                              (NSUInteger)n_vec,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter affine matmat command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

int ds4_drafter_metal_rms_norm_bf16(
        const void *weight_data,
        uint64_t weight_bytes,
        const float *x,
        int len,
        float eps,
        float *out,
        char *err,
        size_t errlen) {
    if (!weight_data || !x || !out || len <= 0 ||
        weight_bytes < (uint64_t)len * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid RMSNorm shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        NSUInteger bytes = (NSUInteger)len * sizeof(float);
        if (drafter_ensure_io_buffers(bytes, bytes, err, errlen) != 0) return -1;
        id<MTLBuffer> wbuf = drafter_cached_buffer(weight_data, weight_bytes, err, errlen);
        if (!wbuf) return -1;
        memcpy([g_x_buffer contents], x, bytes);
        ds4_drafter_metal_rms_norm_args args = {
            .len = len,
            .eps = eps,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_rms_norm_bf16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:0 atIndex:1];
        [enc setBuffer:g_x_buffer offset:0 atIndex:2];
        [enc setBuffer:g_out_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter RMSNorm command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], bytes);
        return 0;
    }
}

int ds4_drafter_metal_rms_norm_bf16_mat(
        const void *weight_data,
        uint64_t weight_bytes,
        const float *x,
        int n_vec,
        int len,
        float eps,
        float *out,
        char *err,
        size_t errlen) {
    if (!weight_data || !x || !out || n_vec <= 0 || len <= 0 ||
        weight_bytes < (uint64_t)len * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid RMSNorm matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        NSUInteger bytes = (NSUInteger)n_vec * (NSUInteger)len * sizeof(float);
        if (drafter_ensure_io_buffers(bytes, bytes, err, errlen) != 0) return -1;
        id<MTLBuffer> wbuf = drafter_cached_buffer(weight_data, weight_bytes, err, errlen);
        if (!wbuf) return -1;
        memcpy([g_x_buffer contents], x, bytes);
        ds4_drafter_metal_rms_norm_args args = {
            .len = len,
            .eps = eps,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_rms_norm_bf16_mat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:0 atIndex:1];
        [enc setBuffer:g_x_buffer offset:0 atIndex:2];
        [enc setBuffer:g_out_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter RMSNorm matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], bytes);
        return 0;
    }
}

static int drafter_encode_affine_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused affine job");
        return -1;
    }
    id<MTLBuffer> wbuf = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
    id<MTLBuffer> sbuf = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
    id<MTLBuffer> bbuf = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
    if (!wbuf || !sbuf || !bbuf) return -1;
    ds4_drafter_metal_affine_args args = {
        .rows = job->rows,
        .packed_cols = job->packed_cols,
        .cols = cols,
        .groups = job->groups,
        .bits = bits,
        .group_size = group_size,
    };
    [enc setComputePipelineState:g_affine_u32_matvec_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:sbuf offset:0 atIndex:2];
    [enc setBuffer:bbuf offset:0 atIndex:3];
    [enc setBuffer:xbuf offset:0 atIndex:4];
    [enc setBuffer:outbuf offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)job->rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 0;
}

static int drafter_encode_affine_q_only_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_affine_job *job,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows != 4096 || job->packed_cols <= 0 || job->groups <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid q-only affine job");
        return -1;
    }
    id<MTLBuffer> wbuf = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
    id<MTLBuffer> sbuf = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
    id<MTLBuffer> bbuf = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
    if (!wbuf || !sbuf || !bbuf) return -1;
    ds4_drafter_metal_affine_args args = {
        .rows = 2048,
        .packed_cols = job->packed_cols,
        .cols = cols,
        .groups = job->groups,
        .bits = bits,
        .group_size = group_size,
    };
    [enc setComputePipelineState:g_affine_u32_q_only_matvec_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:sbuf offset:0 atIndex:2];
    [enc setBuffer:bbuf offset:0 atIndex:3];
    [enc setBuffer:xbuf offset:0 atIndex:4];
    [enc setBuffer:outbuf offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(2048, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 0;
}

static int drafter_encode_affine_matmat_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_affine_job *job,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        n_vec <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused affine matmat job");
        return -1;
    }
    id<MTLBuffer> wbuf = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
    id<MTLBuffer> sbuf = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
    id<MTLBuffer> bbuf = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
    if (!wbuf || !sbuf || !bbuf) return -1;
    ds4_drafter_metal_affine_args args = {
        .rows = job->rows,
        .packed_cols = job->packed_cols,
        .cols = cols,
        .groups = job->groups,
        .bits = bits,
        .group_size = group_size,
    };
    [enc setComputePipelineState:g_affine_u32_matmat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:sbuf offset:0 atIndex:2];
    [enc setBuffer:bbuf offset:0 atIndex:3];
    [enc setBuffer:xbuf offset:0 atIndex:4];
    [enc setBuffer:outbuf offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)job->rows,
                                          (NSUInteger)n_vec,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 0;
}

static int drafter_encode_affine_matmat4_threads_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_affine_job *job,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        NSUInteger threads_per_group,
        char *err,
        size_t errlen) {
    if (!job || !job->w_data || !job->scales_data || !job->biases_data ||
        job->rows <= 0 || job->packed_cols <= 0 || job->groups <= 0 ||
        n_vec <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused affine matmat4 job");
        return -1;
    }
    id<MTLBuffer> wbuf = drafter_cached_buffer(job->w_data, job->w_bytes, err, errlen);
    id<MTLBuffer> sbuf = drafter_cached_buffer(job->scales_data, job->scales_bytes, err, errlen);
    id<MTLBuffer> bbuf = drafter_cached_buffer(job->biases_data, job->biases_bytes, err, errlen);
    if (!wbuf || !sbuf || !bbuf) return -1;
    ds4_drafter_metal_affine_args args = {
        .rows = job->rows,
        .packed_cols = job->packed_cols,
        .cols = cols,
        .groups = job->groups,
        .bits = bits,
        .group_size = group_size,
    };
    uint32_t n_vec_u = (uint32_t)n_vec;
    const int tile_mode = drafter_quant_tile_mode();
    if (tile_mode == 44) {
        [enc setComputePipelineState:g_affine_u32_matmat4x4_pipeline];
    } else if (tile_mode == 82) {
        [enc setComputePipelineState:g_affine_u32_matmat8x2_pipeline];
    } else {
        [enc setComputePipelineState:g_affine_u32_matmat4x2_pipeline];
    }
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:sbuf offset:0 atIndex:2];
    [enc setBuffer:bbuf offset:0 atIndex:3];
    [enc setBuffer:xbuf offset:0 atIndex:4];
    [enc setBuffer:outbuf offset:0 atIndex:5];
    [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:6];
    if (tile_mode == 44) {
        [enc setThreadgroupMemoryLength:16u * threads_per_group * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)job->rows + 3u) / 4u,
                                              ((NSUInteger)n_vec + 3u) / 4u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    } else if (tile_mode == 82) {
        [enc setThreadgroupMemoryLength:16u * threads_per_group * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)job->rows + 1u) / 2u,
                                              ((NSUInteger)n_vec + 7u) / 8u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    } else {
        [enc setThreadgroupMemoryLength:8u * threads_per_group * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)job->rows + 1u) / 2u,
                                              ((NSUInteger)n_vec + 3u) / 4u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    }
    return 0;
}

static int drafter_encode_affine_matmat4_64_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_affine_job *job,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        char *err,
        size_t errlen) {
    return drafter_encode_affine_matmat4_threads_dispatch(enc, job, n_vec,
                                                          cols, bits,
                                                          group_size, xbuf,
                                                          outbuf, 64u, err,
                                                          errlen);
}

static int drafter_encode_affine_matmat_best_dispatch(
        id<MTLCommandBuffer> *cbp,
        id<MTLComputeCommandEncoder> *enc,
        const ds4_drafter_metal_affine_job *job,
        int n_vec,
        int cols,
        int bits,
        int group_size,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        NSUInteger fallback_threads,
        char *err,
        size_t errlen) {
    if (!cbp || !*cbp) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter missing command buffer");
        return -1;
    }
    id<MTLCommandBuffer> cb = *cbp;
    if (drafter_mps_matmul_enabled() &&
        n_vec >= drafter_mps_min_n_vec() &&
        job && job->rows >= drafter_mps_min_rows()) {
        if (*enc) {
            [*enc endEncoding];
            *enc = nil;
        }
        int rc = drafter_encode_affine_mps_matmat(cb, job, n_vec, cols,
                                                  bits, group_size, xbuf,
                                                  outbuf, err, errlen);
        if (rc != 0) return rc < 0 ? -1 : rc;
        if (drafter_mps_sync_enabled()) {
            if (drafter_sync_mps_command_buffer(cbp, "MPS affine sync", err, errlen) != 0) return -1;
            cb = *cbp;
        }
        *enc = [cb computeCommandEncoder];
        if (!*enc) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume compute encoder");
            return -1;
        }
        return 0;
    }
    if (fallback_threads == 0) {
        return drafter_encode_affine_matmat_dispatch(*enc, job, n_vec, cols,
                                                     bits, group_size, xbuf,
                                                     outbuf, err, errlen);
    }
    return drafter_encode_affine_matmat4_threads_dispatch(*enc, job, n_vec,
                                                          cols, bits,
                                                          group_size, xbuf,
                                                          outbuf,
                                                          fallback_threads,
                                                          err, errlen);
}

static int drafter_encode_rms_norm_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const void *weight_data,
        uint64_t weight_bytes,
        id<MTLBuffer> xbuf,
        id<MTLBuffer> outbuf,
        int len,
        float eps,
        char *err,
        size_t errlen) {
    if (!weight_data || !xbuf || !outbuf || len <= 0 ||
        weight_bytes < (uint64_t)len * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused RMSNorm shape");
        return -1;
    }
    id<MTLBuffer> wbuf = drafter_cached_buffer(weight_data, weight_bytes, err, errlen);
    if (!wbuf) return -1;
    ds4_drafter_metal_rms_norm_args args = {
        .len = len,
        .eps = eps,
    };
    [enc setComputePipelineState:g_rms_norm_bf16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:xbuf offset:0 atIndex:2];
    [enc setBuffer:outbuf offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 0;
}

static void drafter_encode_round_bf16_dispatch(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> buf,
        uint32_t len) {
    [enc setComputePipelineState:g_round_bf16_pipeline];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&len length:sizeof(len) atIndex:1];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)len, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void drafter_encode_residual_add_round_dispatch(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> out,
        uint32_t len) {
    [enc setComputePipelineState:g_residual_add_round_pipeline];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&len length:sizeof(len) atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)len, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void drafter_encode_residual_add_round_pre_b_round_dispatch(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> out,
        uint32_t len) {
    [enc setComputePipelineState:g_residual_add_round_pre_b_round_pipeline];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&len length:sizeof(len) atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)len, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void drafter_encode_residual_add_round_pre_b_round_norm_dispatch(
        id<MTLComputeCommandEncoder> enc,
        const ds4_drafter_metal_rms_norm_args *args,
        id<MTLBuffer> weight,
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> residual_out,
        id<MTLBuffer> norm_out,
        int n_vec) {
    [enc setComputePipelineState:g_residual_add_round_pre_b_round_norm_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:a offset:0 atIndex:2];
    [enc setBuffer:b offset:0 atIndex:3];
    [enc setBuffer:residual_out offset:0 atIndex:4];
    [enc setBuffer:norm_out offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

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
        size_t errlen) {
    if (!w_data || !scales_data || !biases_data || !x || !token_out ||
        rows <= 0 || packed_cols <= 0 || cols <= 0 || groups <= 0 ||
        bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid logits argmax shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)rows * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, logits_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_argmax_buffer, &g_argmax_bytes,
                                  sizeof(uint32_t), "argmax", err, errlen) != 0) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);

        ds4_drafter_metal_affine_job job = {
            .w_data = w_data,
            .w_bytes = w_bytes,
            .scales_data = scales_data,
            .scales_bytes = scales_bytes,
            .biases_data = biases_data,
            .biases_bytes = biases_bytes,
            .rows = rows,
            .packed_cols = packed_cols,
            .groups = groups,
            .out = NULL,
        };
        uint32_t rows_u = (uint32_t)rows;

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        int affine_rc = drafter_logits_tiled_enabled() ?
            drafter_encode_affine_matmat4_threads_dispatch(enc, &job, 1, cols,
                                                           bits, group_size,
                                                           g_x_buffer,
                                                           g_out_buffer,
                                                           drafter_logits_tiled_threads(),
                                                           err, errlen) :
            drafter_encode_affine_dispatch(enc, &job, cols, bits, group_size,
                                           g_x_buffer, g_out_buffer,
                                           err, errlen);
        if (affine_rc != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc setComputePipelineState:g_argmax_pipeline];
        [enc setBytes:&rows_u length:sizeof(rows_u) atIndex:0];
        [enc setBuffer:g_out_buffer offset:0 atIndex:1];
        [enc setBuffer:g_argmax_buffer offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc setThreadgroupMemoryLength:256u * sizeof(uint32_t) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter logits argmax command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        uint32_t token = ((uint32_t *)[g_argmax_buffer contents])[0];
        if (token >= (uint32_t)rows) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter logits argmax produced invalid token");
            return -1;
        }
        *token_out = (int)token;
        return 0;
    }
}

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
        size_t errlen) {
    if (!gate || !up || !down || !x || !out ||
        in_cols <= 0 || hidden_cols <= 0 || out_cols <= 0 ||
        gate->rows != hidden_cols || up->rows != hidden_cols ||
        down->rows != out_cols || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused MLP shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)in_cols * sizeof(float);
        NSUInteger out_bytes = (NSUInteger)out_cols * sizeof(float);
        NSUInteger hidden_bytes = (NSUInteger)hidden_cols * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0) return -1;
        if (drafter_ensure_many_out_buffer(0, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, hidden_bytes, err, errlen) != 0) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_dispatch(enc, gate, in_cols, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[0],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, up, in_cols, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[1],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        uint32_t hidden_len = (uint32_t)hidden_cols;
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, down, hidden_cols, bits, group_size,
                                           g_many_out_buffers[2], g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused MLP command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!gate || !up || !down || !x || !out ||
        n_vec <= 0 || in_cols <= 0 || hidden_cols <= 0 || out_cols <= 0 ||
        gate->rows != hidden_cols || up->rows != hidden_cols ||
        down->rows != out_cols || bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused MLP matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)n_vec * (NSUInteger)in_cols * sizeof(float);
        NSUInteger out_bytes = (NSUInteger)n_vec * (NSUInteger)out_cols * sizeof(float);
        NSUInteger hidden_bytes = (NSUInteger)n_vec * (NSUInteger)hidden_cols * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0) return -1;
        if (drafter_ensure_many_out_buffer(0, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, hidden_bytes, err, errlen) != 0) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_matmat_dispatch(enc, gate, n_vec,
                                                  in_cols, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[0],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, up, n_vec,
                                                  in_cols, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[1],
                                                  err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        uint32_t hidden_len = (uint32_t)((NSUInteger)n_vec * (NSUInteger)hidden_cols);
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_matmat_dispatch(enc, down, n_vec,
                                                  hidden_cols, bits, group_size,
                                                  g_many_out_buffers[2],
                                                  g_out_buffer,
                                                  err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused MLP matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!qkv || !z || !b || !a || !conv_data || !x ||
        !conv_out || !z_out || !b_out || !a_out ||
        n_vec <= 0 || bits <= 0 || group_size <= 0 ||
        qkv->rows != 6144 || z->rows != 2048 ||
        b->rows != 16 || a->rows != 16 ||
        conv_bytes < 6144u * 4u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid linear projection matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger qkv_bytes = (NSUInteger)n_vec * 6144u * sizeof(float);
        const NSUInteger z_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger ba_bytes = (NSUInteger)n_vec * 16u * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, 1, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, qkv_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, z_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, ba_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, ba_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, qkv_bytes, err, errlen) != 0) {
            return -1;
        }
        id<MTLBuffer> conv_w_buf = drafter_cached_buffer(conv_data, conv_bytes, err, errlen);
        if (!conv_w_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);
        uint32_t n_vec_u = (uint32_t)n_vec;

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_matmat_dispatch(enc, qkv, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[0],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, z, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[1],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, b, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[2],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, a, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[3],
                                                  err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_linear_conv_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:conv_w_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 6144u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter linear projection matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(conv_out, [g_many_out_buffers[4] contents], qkv_bytes);
        if (qkv_out) memcpy(qkv_out, [g_many_out_buffers[0] contents], qkv_bytes);
        memcpy(z_out, [g_many_out_buffers[1] contents], z_bytes);
        memcpy(b_out, [g_many_out_buffers[2] contents], ba_bytes);
        memcpy(a_out, [g_many_out_buffers[3] contents], ba_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!norm_data || !a_log_data || !dt_bias_data || !conv_out || !z ||
        !b || !a || !gated_out || n_vec <= 0 ||
        cache_id < -1 || cache_id >= 24 ||
        norm_bytes < 128u * sizeof(uint16_t) ||
        a_log_bytes < 16u * sizeof(float) ||
        dt_bias_bytes < 16u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid linear scan matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger conv_bytes = (NSUInteger)n_vec * 6144u * sizeof(float);
        const NSUInteger z_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger ba_bytes = (NSUInteger)n_vec * 16u * sizeof(float);
        const NSUInteger qk_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger kq_bytes = (NSUInteger)n_vec * 16u * sizeof(float);
        const NSUInteger y_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger gated_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
        const NSUInteger scan_params_bytes = (NSUInteger)n_vec * 16u * 2u * sizeof(float);
        if (drafter_ensure_io_buffers(conv_bytes, gated_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, z_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, ba_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, ba_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, qk_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, qk_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, y_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_kq_buffer,
                                  &g_linear_kq_bytes,
                                  kq_bytes,
                                  "linear kq",
                                  err,
                                  errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_state_buffer,
                                  &g_linear_scan_state_bytes,
                                  delta_state_bytes,
                                  "linear scan state",
                                  err,
                                  errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_params_buffer,
                                  &g_linear_scan_params_bytes,
                                  scan_params_bytes,
                                  "linear scan params",
                                  err,
                                  errlen) != 0 ||
            (cache_id >= 0 &&
             (drafter_ensure_buffer(&g_linear_delta_state_buffers[cache_id],
                                    &g_linear_delta_state_bytes[cache_id],
                                    delta_state_bytes,
                                    "linear delta state",
                                    err,
                                    errlen) != 0 ||
              drafter_ensure_buffer(&g_linear_conv_state_buffers[cache_id],
                                    &g_linear_conv_state_bytes[cache_id],
                                    conv_state_bytes,
                                    "linear conv state",
                                    err,
                                    errlen) != 0))) {
            return -1;
        }
        id<MTLBuffer> norm_buf = drafter_cached_buffer(norm_data, norm_bytes, err, errlen);
        id<MTLBuffer> a_log_buf = drafter_cached_buffer(a_log_data, a_log_bytes, err, errlen);
        id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(dt_bias_data, dt_bias_bytes, err, errlen);
        if (!norm_buf || !a_log_buf || !dt_bias_buf) return -1;
        memcpy([g_x_buffer contents], conv_out, conv_bytes);
        memcpy([g_many_out_buffers[0] contents], z, z_bytes);
        memcpy([g_many_out_buffers[1] contents], b, ba_bytes);
        memcpy([g_many_out_buffers[2] contents], a, ba_bytes);
	        memset([g_linear_scan_state_buffer contents], 0, delta_state_bytes);
	        uint32_t n_vec_u = (uint32_t)n_vec;
	        NSUInteger delta_threads = drafter_linear_delta_threads();
	
	        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
	        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:g_linear_qk_norm_kq_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:g_x_buffer offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:3];
        [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:2u * 128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_scan_params_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:a_log_buf offset:0 atIndex:1];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:4];
        [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 16u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_linear_delta_scan_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:a_log_buf offset:0 atIndex:1];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:4];
        [enc setBuffer:g_x_buffer offset:0 atIndex:5];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:6];
	        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:7];
	        [enc setBuffer:g_linear_scan_state_buffer offset:0 atIndex:8];
	        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:9];
        [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:10];
        [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:11];
	        [enc setThreadgroupMemoryLength:delta_threads * sizeof(float) atIndex:0];
	        [enc dispatchThreadgroups:MTLSizeMake(16, 128, 1)
	             threadsPerThreadgroup:MTLSizeMake(delta_threads, 1, 1)];

        [enc setComputePipelineState:g_linear_gate_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
        [enc setBuffer:g_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter linear scan matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        if (cache_id >= 0) {
            memcpy([g_linear_delta_state_buffers[cache_id] contents],
                   [g_linear_scan_state_buffer contents],
                   delta_state_bytes);
            memset([g_linear_conv_state_buffers[cache_id] contents], 0, conv_state_bytes);
            if (qkv) {
                const int keep = n_vec < 3 ? n_vec : 3;
                float *conv_state = (float *)[g_linear_conv_state_buffers[cache_id] contents];
                for (int i = 0; i < keep; i++) {
                    const int src_t = n_vec - keep + i;
                    const int dst_t = 3 - keep + i;
                    memcpy(conv_state + (size_t)dst_t * 6144u,
                           qkv + (size_t)src_t * 6144u,
                           6144u * sizeof(float));
                }
            }
        }
        memcpy(gated_out, [g_out_buffer contents], gated_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!input_norm_data || !post_norm_data || !qkv || !z || !b || !a ||
        !linear_out || !conv_data || !linear_norm_data || !a_log_data ||
        !dt_bias_data || !mlp_gate || !mlp_up || !mlp_down || !x || !out ||
        n_vec <= 0 || cache_id < -1 || cache_id >= 24 ||
        bits <= 0 || group_size <= 0 ||
        input_norm_bytes < 1024u * sizeof(uint16_t) ||
        post_norm_bytes < 1024u * sizeof(uint16_t) ||
        linear_norm_bytes < 128u * sizeof(uint16_t) ||
        conv_bytes < 6144u * 4u * sizeof(uint16_t) ||
        a_log_bytes < 16u * sizeof(float) ||
        dt_bias_bytes < 16u * sizeof(uint16_t) ||
        qkv->rows != 6144 || z->rows != 2048 ||
        b->rows != 16 || a->rows != 16 || linear_out->rows != 1024 ||
        mlp_gate->rows != 3584 || mlp_up->rows != 3584 ||
        mlp_down->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused linear decoder matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger out_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger qkv_bytes = (NSUInteger)n_vec * 6144u * sizeof(float);
        const NSUInteger z_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger qk_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger kq_bytes = (NSUInteger)n_vec * 16u * sizeof(float);
        const NSUInteger hidden_bytes = (NSUInteger)n_vec * 3584u * sizeof(float);
        const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
        const NSUInteger gated_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger y_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger scan_params_bytes = (NSUInteger)n_vec * 16u * 2u * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, qkv_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, z_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, qkv_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, qk_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, qk_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  y_bytes, "linear matrix y", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  gated_bytes, "linear matrix gated", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_kq_buffer,
                                  &g_linear_kq_bytes,
                                  kq_bytes,
                                  "linear kq",
                                  err,
                                  errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_state_buffer,
                                  &g_linear_scan_state_bytes,
                                  delta_state_bytes,
                                  "linear scan state",
                                  err,
                                  errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_params_buffer,
                                  &g_linear_scan_params_bytes,
                                  scan_params_bytes,
                                  "linear scan params",
                                  err,
                                  errlen) != 0 ||
            (cache_id >= 0 &&
             (drafter_ensure_buffer(&g_linear_delta_state_buffers[cache_id],
                                    &g_linear_delta_state_bytes[cache_id],
                                    delta_state_bytes,
                                    "linear delta state",
                                    err,
                                    errlen) != 0 ||
              drafter_ensure_buffer(&g_linear_conv_state_buffers[cache_id],
                                    &g_linear_conv_state_bytes[cache_id],
                                    conv_state_bytes,
                                    "linear conv state",
                                    err,
                                    errlen) != 0))) {
            return -1;
        }
        id<MTLBuffer> input_norm_buf = drafter_cached_buffer(input_norm_data, input_norm_bytes, err, errlen);
        id<MTLBuffer> post_norm_buf = drafter_cached_buffer(post_norm_data, post_norm_bytes, err, errlen);
        id<MTLBuffer> conv_w_buf = drafter_cached_buffer(conv_data, conv_bytes, err, errlen);
        id<MTLBuffer> linear_norm_buf = drafter_cached_buffer(linear_norm_data, linear_norm_bytes, err, errlen);
        id<MTLBuffer> a_log_buf = drafter_cached_buffer(a_log_data, a_log_bytes, err, errlen);
        id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(dt_bias_data, dt_bias_bytes, err, errlen);
        if (!input_norm_buf || !post_norm_buf || !conv_w_buf ||
            !linear_norm_buf || !a_log_buf || !dt_bias_buf) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);
        memset([g_linear_scan_state_buffer contents], 0, delta_state_bytes);
        uint32_t n_vec_u = (uint32_t)n_vec;
        uint32_t hidden_len = (uint32_t)((NSUInteger)n_vec * 3584u);
        NSUInteger delta_threads = drafter_linear_delta_threads();
        ds4_drafter_metal_rms_norm_args rms_args = {
            .len = 1024,
            .eps = 1.0e-6f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:g_rms_norm_bf16_mat_pipeline];
        [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
        [enc setBuffer:input_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_x_buffer offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0],
                                           (NSUInteger)n_vec * 1024u);

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, qkv, n_vec,
                                                       1024, bits, group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[1],
                                                       32u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, z, n_vec,
                                                       1024, bits, group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[2],
                                                       32u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, b, n_vec,
                                                       1024, bits, group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[3],
                                                       32u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, a, n_vec,
                                                       1024, bits, group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[4],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_linear_conv_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:conv_w_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 6144u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_linear_qk_norm_kq_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:3];
        [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:2u * 128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_scan_params_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:a_log_buf offset:0 atIndex:1];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 16u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_linear_delta_scan_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:a_log_buf offset:0 atIndex:1];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:5];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:6];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:7];
        [enc setBuffer:g_linear_scan_state_buffer offset:0 atIndex:8];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:9];
        [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:10];
        [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:11];
        [enc setThreadgroupMemoryLength:delta_threads * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 128, 1)
             threadsPerThreadgroup:MTLSizeMake(delta_threads, 1, 1)];

        [enc setComputePipelineState:g_linear_gate_mat_pipeline];
        [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
        [enc setBuffer:linear_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:3];
        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, linear_out,
                                                       n_vec, 2048, bits,
                                                       group_size,
                                                       g_attention_out_buffer,
                                                       g_many_out_buffers[6],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[6],
                                           (NSUInteger)n_vec * 1024u);
        drafter_encode_residual_add_round_dispatch(enc, g_x_buffer,
                                                   g_many_out_buffers[6],
                                                   g_many_out_buffers[2],
                                                   (NSUInteger)n_vec * 1024u);

        [enc setComputePipelineState:g_rms_norm_bf16_mat_pipeline];
        [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
        [enc setBuffer:post_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0],
                                           (NSUInteger)n_vec * 1024u);

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_gate,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[3],
                                                       32u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_up,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[4],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_down,
                                                       n_vec, 3584, bits,
                                                       group_size,
                                                       g_many_out_buffers[5],
                                                       g_many_out_buffers[7],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7],
                                           (NSUInteger)n_vec * 1024u);
        drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[2],
                                                   g_many_out_buffers[7],
                                                   g_out_buffer,
                                                   (NSUInteger)n_vec * 1024u);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused linear decoder matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        if (cache_id >= 0) {
            memcpy([g_linear_delta_state_buffers[cache_id] contents],
                   [g_linear_scan_state_buffer contents],
                   delta_state_bytes);
            memset([g_linear_conv_state_buffers[cache_id] contents], 0, conv_state_bytes);
            float *conv_state = (float *)[g_linear_conv_state_buffers[cache_id] contents];
            const float *qkv_data = (const float *)[g_many_out_buffers[1] contents];
            const int keep = n_vec < 3 ? n_vec : 3;
            for (int i = 0; i < keep; i++) {
                const int src_t = n_vec - keep + i;
                const int dst_t = 3 - keep + i;
                memcpy(conv_state + (size_t)dst_t * 6144u,
                       qkv_data + (size_t)src_t * 6144u,
                       6144u * sizeof(float));
            }
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!input_norm_data || !post_norm_data || !q_proj || !k_proj ||
        !v_proj || !out_proj || !q_norm_data || !k_norm_data ||
        !mlp_gate || !mlp_up || !mlp_down || !x || !out ||
        !keys_rope_out || !values_out || n_vec <= 0 ||
        bits <= 0 || group_size <= 0 ||
        input_norm_bytes < 1024u * sizeof(uint16_t) ||
        post_norm_bytes < 1024u * sizeof(uint16_t) ||
        q_norm_bytes < 256u * sizeof(uint16_t) ||
        k_norm_bytes < 256u * sizeof(uint16_t) ||
        q_proj->rows != 4096 || k_proj->rows != 512 ||
        v_proj->rows != 512 || out_proj->rows != 1024 ||
        mlp_gate->rows != 3584 || mlp_up->rows != 3584 ||
        mlp_down->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused full decoder matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = (NSUInteger)n_vec * 4096u * sizeof(float);
        const NSUInteger q_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_vec * 512u * sizeof(float);
        const NSUInteger hidden_bytes = (NSUInteger)n_vec * 3584u * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "full decoder query matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_gate_buffer, &g_attention_gate_bytes,
                                  q_bytes, "full decoder gate matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_keys_buffer, &g_attention_keys_bytes,
                                  kv_bytes, "full decoder key matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_values_buffer, &g_attention_values_bytes,
                                  kv_bytes, "full decoder value matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "full decoder attention output", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  hidden_bytes, "full decoder MLP scratch", err, errlen) != 0) {
            return -1;
        }
        id<MTLBuffer> input_norm_buf = drafter_cached_buffer(input_norm_data, input_norm_bytes, err, errlen);
        id<MTLBuffer> post_norm_buf = drafter_cached_buffer(post_norm_data, post_norm_bytes, err, errlen);
        id<MTLBuffer> q_norm_buf = drafter_cached_buffer(q_norm_data, q_norm_bytes, err, errlen);
        id<MTLBuffer> k_norm_buf = drafter_cached_buffer(k_norm_data, k_norm_bytes, err, errlen);
        if (!input_norm_buf || !post_norm_buf || !q_norm_buf || !k_norm_buf) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);
        uint32_t hidden_len = (uint32_t)((NSUInteger)n_vec * 3584u);
        ds4_drafter_metal_rms_norm_args rms_args = {
            .len = 1024,
            .eps = 1.0e-6f,
        };
        ds4_drafter_metal_rope_args rope_args = { .position = 0 };
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_vec,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:g_rms_norm_bf16_mat_pipeline];
        [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
        [enc setBuffer:input_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_x_buffer offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0],
                                           (NSUInteger)n_vec * 1024u);

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, q_proj,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[1],
                                                       64u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, k_proj,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[2],
                                                       64u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, v_proj,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_attention_values_buffer,
                                                       0u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_full_q_norm_rope_mat_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:q_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
        [enc setBuffer:g_attention_q_buffer offset:0 atIndex:3];
        [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_full_k_norm_rope_mat_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:k_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBuffer:g_attention_keys_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(2, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        int mps_attn_rc =
            drafter_encode_mps_causal_attention_mat(&cb, &enc, n_vec,
                                                    err, errlen);
        if (mps_attn_rc < 0) {
            [enc endEncoding];
            return -1;
        }
        if (mps_attn_rc > 0) {
            [enc setComputePipelineState:drafter_attention_context_causal_fused_pipeline()];
            [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
            [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
            [enc setBuffer:g_attention_keys_buffer offset:0 atIndex:3];
            [enc setBuffer:g_attention_values_buffer offset:0 atIndex:4];
            [enc setBuffer:g_attention_out_buffer offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_vec, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, out_proj,
                                                       n_vec, 2048, bits,
                                                       group_size,
                                                       g_attention_out_buffer,
                                                       g_many_out_buffers[6],
                                                       64u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[6],
                                           (NSUInteger)n_vec * 1024u);
        drafter_encode_residual_add_round_dispatch(enc, g_x_buffer,
                                                   g_many_out_buffers[6],
                                                   g_many_out_buffers[2],
                                                   (NSUInteger)n_vec * 1024u);

        [enc setComputePipelineState:g_rms_norm_bf16_mat_pipeline];
        [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
        [enc setBuffer:post_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0],
                                           (NSUInteger)n_vec * 1024u);

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_gate,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[3],
                                                       32u, err, errlen) != 0 ||
            drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_up,
                                                       n_vec, 1024, bits,
                                                       group_size,
                                                       g_many_out_buffers[0],
                                                       g_many_out_buffers[4],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:1];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, mlp_down,
                                                       n_vec, 3584, bits,
                                                       group_size,
                                                       g_linear_y_buffer,
                                                       g_many_out_buffers[6],
                                                       32u, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[6],
                                           (NSUInteger)n_vec * 1024u);
        drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[2],
                                                   g_many_out_buffers[6],
                                                   g_out_buffer,
                                                   (NSUInteger)n_vec * 1024u);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused full decoder matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], x_bytes);
        memcpy(keys_rope_out, [g_attention_keys_buffer contents], kv_bytes);
        memcpy(values_out, [g_attention_values_buffer contents], kv_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!qkv || !z || !b || !a || !out_proj || !conv_data || !norm_data ||
        !a_log_data || !dt_bias_data || !x || !out ||
        cache_id < 0 || cache_id >= 24 || bits <= 0 || group_size <= 0 ||
        qkv->rows != 6144 || z->rows != 2048 || b->rows != 16 ||
        a->rows != 16 || out_proj->rows != 1024 ||
        conv_bytes < 6144u * 4u * sizeof(uint16_t) ||
        norm_bytes < 128u * sizeof(uint16_t) ||
        a_log_bytes < 16u * sizeof(float) ||
        dt_bias_bytes < 16u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused linear-attention shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = 1024u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
        const NSUInteger y_bytes = 2048u * sizeof(float);
        const int state_was_missing = !g_linear_conv_state_buffers[cache_id] ||
                                      !g_linear_delta_state_buffers[cache_id];
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, 16u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, 16u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  y_bytes, "linear y", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_conv_state_buffers[cache_id],
                                  &g_linear_conv_state_bytes[cache_id],
                                  conv_state_bytes, "linear conv state", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_delta_state_buffers[cache_id],
                                  &g_linear_delta_state_bytes[cache_id],
                                  delta_state_bytes, "linear delta state", err, errlen) != 0) {
            return -1;
        }
        if (state_was_missing) {
            memset([g_linear_conv_state_buffers[cache_id] contents], 0, conv_state_bytes);
            memset([g_linear_delta_state_buffers[cache_id] contents], 0, delta_state_bytes);
        }
        id<MTLBuffer> conv_w_buf = drafter_cached_buffer(conv_data, conv_bytes, err, errlen);
        id<MTLBuffer> norm_buf = drafter_cached_buffer(norm_data, norm_bytes, err, errlen);
        id<MTLBuffer> a_log_buf = drafter_cached_buffer(a_log_data, a_log_bytes, err, errlen);
        id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(dt_bias_data, dt_bias_bytes, err, errlen);
        if (!conv_w_buf || !norm_buf || !a_log_buf || !dt_bias_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_dispatch(enc, qkv, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[0],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, z, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[1],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, b, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[2],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, a, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[3],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_linear_conv_pipeline];
        [enc setBuffer:conv_w_buf offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:1];
        [enc setBuffer:g_linear_conv_state_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(6144, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_linear_qk_norm_pipeline];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_delta_pipeline];
        [enc setBuffer:a_log_buf offset:0 atIndex:0];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:5];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:6];
        [enc setBuffer:g_linear_delta_state_buffers[cache_id] offset:0 atIndex:7];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:8];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 128, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_gate_pipeline];
        [enc setBuffer:norm_buf offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, out_proj, 2048, bits, group_size,
                                           g_many_out_buffers[7], g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused linear-attention command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!input_norm_data || !post_norm_data || !qkv || !z || !b || !a ||
        !linear_out || !conv_data || !linear_norm_data || !a_log_data ||
        !dt_bias_data || !mlp_gate || !mlp_up || !mlp_down || !x || !out ||
        cache_id < 0 || cache_id >= 24 || bits <= 0 || group_size <= 0 ||
        input_norm_bytes < 1024u * sizeof(uint16_t) ||
        post_norm_bytes < 1024u * sizeof(uint16_t) ||
        linear_norm_bytes < 128u * sizeof(uint16_t) ||
        conv_bytes < 6144u * 4u * sizeof(uint16_t) ||
        a_log_bytes < 16u * sizeof(float) ||
        dt_bias_bytes < 16u * sizeof(uint16_t) ||
        qkv->rows != 6144 || z->rows != 2048 ||
        b->rows != 16 || a->rows != 16 || linear_out->rows != 1024 ||
        mlp_gate->rows != 3584 || mlp_up->rows != 3584 ||
        mlp_down->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused linear decoder layer shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = 1024u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
        const int state_was_missing = !g_linear_conv_state_buffers[cache_id] ||
                                      !g_linear_delta_state_buffers[cache_id];
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, 1024u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, 16u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  3584u * sizeof(float), "linear fused scratch", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_conv_state_buffers[cache_id],
                                  &g_linear_conv_state_bytes[cache_id],
                                  conv_state_bytes, "linear conv state", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_delta_state_buffers[cache_id],
                                  &g_linear_delta_state_bytes[cache_id],
                                  delta_state_bytes, "linear delta state", err, errlen) != 0) {
            return -1;
        }
        if (state_was_missing) {
            memset([g_linear_conv_state_buffers[cache_id] contents], 0, conv_state_bytes);
            memset([g_linear_delta_state_buffers[cache_id] contents], 0, delta_state_bytes);
        }
        id<MTLBuffer> conv_w_buf = drafter_cached_buffer(conv_data, conv_bytes, err, errlen);
        id<MTLBuffer> linear_norm_buf = drafter_cached_buffer(linear_norm_data, linear_norm_bytes, err, errlen);
        id<MTLBuffer> a_log_buf = drafter_cached_buffer(a_log_data, a_log_bytes, err, errlen);
        id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(dt_bias_data, dt_bias_bytes, err, errlen);
        if (!conv_w_buf || !linear_norm_buf || !a_log_buf || !dt_bias_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        if (drafter_encode_rms_norm_dispatch(enc, input_norm_data, input_norm_bytes,
                                             g_x_buffer, g_many_out_buffers[0],
                                             1024, 1.0e-6f, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);

        if (drafter_encode_affine_dispatch(enc, qkv, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[1],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, z, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[2],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, b, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[3],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, a, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_linear_conv_pipeline];
        [enc setBuffer:conv_w_buf offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
        [enc setBuffer:g_linear_conv_state_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(6144, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_linear_qk_norm_pipeline];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_delta_pipeline];
        [enc setBuffer:a_log_buf offset:0 atIndex:0];
        [enc setBuffer:dt_bias_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:2];
        [enc setBuffer:g_out_buffer offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:5];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:6];
        [enc setBuffer:g_linear_delta_state_buffers[cache_id] offset:0 atIndex:7];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:8];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 128, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_linear_gate_pipeline];
        [enc setBuffer:linear_norm_buf offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:1];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, linear_out, 2048, bits, group_size,
                                           g_many_out_buffers[6], g_many_out_buffers[1],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[1], 1024);
        drafter_encode_residual_add_round_dispatch(enc, g_x_buffer, g_many_out_buffers[1],
                                                   g_many_out_buffers[2], 1024);

        if (drafter_encode_rms_norm_dispatch(enc, post_norm_data, post_norm_bytes,
                                             g_many_out_buffers[2], g_many_out_buffers[0],
                                             1024, 1.0e-6f, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);

        if (drafter_encode_affine_dispatch(enc, mlp_gate, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[5],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, mlp_up, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[7],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        uint32_t hidden_len = 3584u;
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:1];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(3584, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, mlp_down, 3584, bits, group_size,
                                           g_linear_y_buffer, g_many_out_buffers[0],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);
        drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[2],
                                                   g_many_out_buffers[0],
                                                   g_out_buffer, 1024);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused linear decoder layer command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!out_proj || !queries_rope || !gate || !keys_rope_cache ||
        !values_cache || !out || n_ctx <= 0 || bits <= 0 || group_size <= 0 ||
        out_proj->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused attention shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger gate_bytes = 2048u * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_ctx * 512u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        if (drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "attention query", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_gate_buffer, &g_attention_gate_bytes,
                                  gate_bytes, "attention gate", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_keys_buffer, &g_attention_keys_bytes,
                                  kv_bytes, "attention keys", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_values_buffer, &g_attention_values_bytes,
                                  kv_bytes, "attention values", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "attention output", err, errlen) != 0 ||
            drafter_ensure_io_buffers(1, out_bytes, err, errlen) != 0) {
            return -1;
        }
        memcpy([g_attention_q_buffer contents], queries_rope, q_bytes);
        memcpy([g_attention_gate_buffer contents], gate, gate_bytes);
        memcpy([g_attention_keys_buffer contents], keys_rope_cache, kv_bytes);
        memcpy([g_attention_values_buffer contents], values_cache, kv_bytes);

        ds4_drafter_metal_attention_args args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_logits_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
        [enc setBuffer:g_attention_keys_buffer offset:0 atIndex:2];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_context_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
        [enc setBuffer:g_attention_values_buffer offset:0 atIndex:3];
        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, out_proj, 2048, bits, group_size,
                                           g_attention_out_buffer, g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused attention command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!out_proj || !queries_rope || !gate || !keys_rope_cache ||
        !values_cache || !out || n_ctx <= 0 || bits <= 0 ||
        group_size <= 0 || out_proj->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused attention matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger q_bytes = (NSUInteger)n_ctx * 2048u * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_ctx * 512u * sizeof(float);
	        const NSUInteger out_bytes = (NSUInteger)n_ctx * 1024u * sizeof(float);
        if (drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "attention query matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_gate_buffer, &g_attention_gate_bytes,
                                  q_bytes, "attention gate matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_keys_buffer, &g_attention_keys_bytes,
                                  kv_bytes, "attention key matrix", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_values_buffer, &g_attention_values_bytes,
                                  kv_bytes, "attention value matrix", err, errlen) != 0 ||
	            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
	                                  q_bytes, "attention output matrix", err, errlen) != 0 ||
            drafter_ensure_io_buffers(1, out_bytes, err, errlen) != 0) {
            return -1;
        }
        memcpy([g_attention_q_buffer contents], queries_rope, q_bytes);
        memcpy([g_attention_gate_buffer contents], gate, q_bytes);
        memcpy([g_attention_keys_buffer contents], keys_rope_cache, kv_bytes);
        memcpy([g_attention_values_buffer contents], values_cache, kv_bytes);

        ds4_drafter_metal_attention_args args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        int mps_attn_rc =
            drafter_encode_mps_causal_attention_mat(&cb, &enc, n_ctx,
                                                    err, errlen);
        if (mps_attn_rc < 0) {
            [enc endEncoding];
            return -1;
        }
        if (mps_attn_rc > 0) {
            [enc setComputePipelineState:drafter_attention_context_causal_fused_pipeline()];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
            [enc setBuffer:g_attention_keys_buffer offset:0 atIndex:3];
            [enc setBuffer:g_attention_values_buffer offset:0 atIndex:4];
            [enc setBuffer:g_attention_out_buffer offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (drafter_encode_affine_matmat4_64_dispatch(enc, out_proj, n_ctx,
                                                      2048, bits, group_size,
                                                      g_attention_out_buffer,
                                                      g_out_buffer,
                                                      err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused attention matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!q_proj || !k_proj || !v_proj || !q_norm_data || !k_norm_data ||
        !x || !queries_rope_out || !gate_out || !keys_rope_out ||
        !values_out || n_vec <= 0 || bits <= 0 || group_size <= 0 ||
        q_proj->rows != 4096 || k_proj->rows != 512 || v_proj->rows != 512 ||
        q_norm_bytes < 256u * sizeof(uint16_t) ||
        k_norm_bytes < 256u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid full-attention projection matrix shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = (NSUInteger)n_vec * 4096u * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_vec * 512u * sizeof(float);
        const NSUInteger q_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        if (drafter_ensure_io_buffers(x_bytes, 1, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, kv_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, kv_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, kv_bytes, err, errlen) != 0) {
            return -1;
        }
        id<MTLBuffer> q_norm_buf = drafter_cached_buffer(q_norm_data, q_norm_bytes, err, errlen);
        id<MTLBuffer> k_norm_buf = drafter_cached_buffer(k_norm_data, k_norm_bytes, err, errlen);
        if (!q_norm_buf || !k_norm_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);
        ds4_drafter_metal_rope_args rope_args = { .position = base_position, .n_vec = n_vec };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_matmat_dispatch(enc, q_proj, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[0],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, k_proj, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[1],
                                                  err, errlen) != 0 ||
            drafter_encode_affine_matmat_dispatch(enc, v_proj, n_vec,
                                                  1024, bits, group_size,
                                                  g_x_buffer,
                                                  g_many_out_buffers[2],
                                                  err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_full_q_norm_rope_mat_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:q_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_full_k_norm_rope_mat_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:k_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(2, (NSUInteger)n_vec, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter full-attention projection matrix command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(queries_rope_out, [g_many_out_buffers[3] contents], q_bytes);
        memcpy(gate_out, [g_many_out_buffers[4] contents], q_bytes);
        memcpy(keys_rope_out, [g_many_out_buffers[5] contents], kv_bytes);
        memcpy(values_out, [g_many_out_buffers[2] contents], kv_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!out_proj || !queries_rope || !gate || !key_rope || !value || !out ||
        cache_id < 0 || cache_id >= 24 || cache_index < 0 ||
        n_ctx <= 0 || cache_capacity < n_ctx || cache_index >= cache_capacity ||
        cache_index != n_ctx - 1 || bits <= 0 || group_size <= 0 ||
        out_proj->rows != 1024) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident attention shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger gate_bytes = 2048u * sizeof(float);
        const NSUInteger kv_slice_bytes = 512u * sizeof(float);
        const NSUInteger kv_cache_bytes = (NSUInteger)cache_capacity * 512u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        if (drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "attention query", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_gate_buffer, &g_attention_gate_bytes,
                                  gate_bytes, "attention gate", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_key_cache_buffers[cache_id],
                                  &g_attention_key_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention key cache", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_value_cache_buffers[cache_id],
                                  &g_attention_value_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention value cache", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "attention output", err, errlen) != 0 ||
            drafter_ensure_io_buffers(1, out_bytes, err, errlen) != 0) {
            return -1;
        }
        memcpy([g_attention_q_buffer contents], queries_rope, q_bytes);
        memcpy([g_attention_gate_buffer contents], gate, gate_bytes);
        memcpy((uint8_t *)[g_attention_key_cache_buffers[cache_id] contents] +
                   (NSUInteger)cache_index * kv_slice_bytes,
               key_rope, kv_slice_bytes);
        memcpy((uint8_t *)[g_attention_value_cache_buffers[cache_id] contents] +
                   (NSUInteger)cache_index * kv_slice_bytes,
               value, kv_slice_bytes);

        ds4_drafter_metal_attention_args args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_logits_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
        [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_context_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
        [enc setBuffer:g_attention_value_cache_buffers[cache_id] offset:0 atIndex:3];
        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, out_proj, 2048, bits, group_size,
                                           g_attention_out_buffer, g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter resident attention command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!q_proj || !k_proj || !v_proj || !out_proj || !q_norm_data ||
        !k_norm_data || !x || !key_out || !value_out || !out ||
        cache_id < 0 || cache_id >= 24 || cache_index < 0 ||
        n_ctx <= 0 || cache_capacity < n_ctx || cache_index >= cache_capacity ||
        cache_index != n_ctx - 1 || bits <= 0 || group_size <= 0 ||
        q_proj->rows != 4096 || k_proj->rows != 512 || v_proj->rows != 512 ||
        out_proj->rows != 1024 ||
        q_norm_bytes < 256u * sizeof(uint16_t) ||
        k_norm_bytes < 256u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused full-attention layer shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = 4096u * sizeof(float);
        const NSUInteger kv_slice_bytes = 512u * sizeof(float);
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger kv_cache_bytes = (NSUInteger)cache_capacity * 512u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        const int cache_was_missing = !g_attention_key_cache_buffers[cache_id] ||
                                      !g_attention_value_cache_buffers[cache_id];
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, kv_slice_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, kv_slice_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, kv_slice_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_key_cache_buffers[cache_id],
                                  &g_attention_key_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention key cache", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_value_cache_buffers[cache_id],
                                  &g_attention_value_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention value cache", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "attention output", err, errlen) != 0) {
            return -1;
        }
        if (cache_was_missing) {
            memset([g_attention_key_cache_buffers[cache_id] contents], 0, kv_cache_bytes);
            memset([g_attention_value_cache_buffers[cache_id] contents], 0, kv_cache_bytes);
        }
        id<MTLBuffer> q_norm_buf = drafter_cached_buffer(q_norm_data, q_norm_bytes, err, errlen);
        id<MTLBuffer> k_norm_buf = drafter_cached_buffer(k_norm_data, k_norm_bytes, err, errlen);
        if (!q_norm_buf || !k_norm_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);
        ds4_drafter_metal_rope_args rope_args = { .position = position };
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_affine_dispatch(enc, q_proj, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[0],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, k_proj, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[1],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, v_proj, 1024, bits, group_size,
                                           g_x_buffer, g_many_out_buffers[2],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_full_q_norm_rope_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:q_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_full_k_norm_rope_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:k_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:g_many_out_buffers[5]
                sourceOffset:0
                    toBuffer:g_attention_key_cache_buffers[cache_id]
           destinationOffset:(NSUInteger)cache_index * kv_slice_bytes
                        size:kv_slice_bytes];
        [blit copyFromBuffer:g_many_out_buffers[2]
                sourceOffset:0
                    toBuffer:g_attention_value_cache_buffers[cache_id]
           destinationOffset:(NSUInteger)cache_index * kv_slice_bytes
                        size:kv_slice_bytes];
        [blit endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_logits_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:1];
        [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_context_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
        [enc setBuffer:g_attention_value_cache_buffers[cache_id] offset:0 atIndex:3];
        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, out_proj, 2048, bits, group_size,
                                           g_attention_out_buffer, g_out_buffer,
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused full-attention layer command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        if (query_capture) {
            memcpy(query_capture, [g_many_out_buffers[3] contents], q_bytes);
        }
        memcpy(key_out, [g_many_out_buffers[5] contents], kv_slice_bytes);
        memcpy(value_out, [g_many_out_buffers[2] contents], kv_slice_bytes);
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!input_norm_data || !post_norm_data || !q_proj || !k_proj || !v_proj ||
        !out_proj || !q_norm_data || !k_norm_data || !mlp_gate || !mlp_up ||
        !mlp_down || !x || !key_out || !value_out || !out ||
        cache_id < 0 || cache_id >= 24 || cache_index < 0 ||
        n_ctx <= 0 || cache_capacity < n_ctx || cache_index >= cache_capacity ||
        cache_index != n_ctx - 1 || bits <= 0 || group_size <= 0 ||
        input_norm_bytes < 1024u * sizeof(uint16_t) ||
        post_norm_bytes < 1024u * sizeof(uint16_t) ||
        q_proj->rows != 4096 || k_proj->rows != 512 || v_proj->rows != 512 ||
        out_proj->rows != 1024 || mlp_gate->rows != 3584 ||
        mlp_up->rows != 3584 || mlp_down->rows != 1024 ||
        q_norm_bytes < 256u * sizeof(uint16_t) ||
        k_norm_bytes < 256u * sizeof(uint16_t)) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid fused full decoder layer shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = 1024u * sizeof(float);
        const NSUInteger out_bytes = 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = 4096u * sizeof(float);
        const NSUInteger kv_slice_bytes = 512u * sizeof(float);
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger kv_cache_bytes = (NSUInteger)cache_capacity * 512u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        const int cache_was_missing = !g_attention_key_cache_buffers[cache_id] ||
                                      !g_attention_value_cache_buffers[cache_id];
        if (drafter_ensure_io_buffers(x_bytes, out_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, kv_slice_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, kv_slice_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, 1024u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  3584u * sizeof(float), "full decoder scratch", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_key_cache_buffers[cache_id],
                                  &g_attention_key_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention key cache", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_value_cache_buffers[cache_id],
                                  &g_attention_value_cache_bytes[cache_id],
                                  kv_cache_bytes, "attention value cache", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "attention output", err, errlen) != 0) {
            return -1;
        }
        if (cache_was_missing) {
            memset([g_attention_key_cache_buffers[cache_id] contents], 0, kv_cache_bytes);
            memset([g_attention_value_cache_buffers[cache_id] contents], 0, kv_cache_bytes);
        }
        id<MTLBuffer> q_norm_buf = drafter_cached_buffer(q_norm_data, q_norm_bytes, err, errlen);
        id<MTLBuffer> k_norm_buf = drafter_cached_buffer(k_norm_data, k_norm_bytes, err, errlen);
        if (!q_norm_buf || !k_norm_buf) return -1;
        memcpy([g_x_buffer contents], x, x_bytes);
        ds4_drafter_metal_rope_args rope_args = { .position = position };
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_encode_rms_norm_dispatch(enc, input_norm_data, input_norm_bytes,
                                             g_x_buffer, g_many_out_buffers[7],
                                             1024, 1.0e-6f, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7], 1024);

        if (drafter_encode_affine_dispatch(enc, q_proj, 1024, bits, group_size,
                                           g_many_out_buffers[7], g_many_out_buffers[0],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, k_proj, 1024, bits, group_size,
                                           g_many_out_buffers[7], g_many_out_buffers[1],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, v_proj, 1024, bits, group_size,
                                           g_many_out_buffers[7], g_many_out_buffers[2],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }

        [enc setComputePipelineState:g_full_q_norm_rope_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:q_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_full_k_norm_rope_pipeline];
        [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [enc setBuffer:k_norm_buf offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
        [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:g_many_out_buffers[5]
                sourceOffset:0
                    toBuffer:g_attention_key_cache_buffers[cache_id]
           destinationOffset:(NSUInteger)cache_index * kv_slice_bytes
                        size:kv_slice_bytes];
        [blit copyFromBuffer:g_many_out_buffers[2]
                sourceOffset:0
                    toBuffer:g_attention_value_cache_buffers[cache_id]
           destinationOffset:(NSUInteger)cache_index * kv_slice_bytes
                        size:kv_slice_bytes];
        [blit endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_logits_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:1];
        [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_context_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
        [enc setBuffer:g_attention_value_cache_buffers[cache_id] offset:0 atIndex:3];
        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, out_proj, 2048, bits, group_size,
                                           g_attention_out_buffer, g_many_out_buffers[7],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7], 1024);
        drafter_encode_residual_add_round_dispatch(enc, g_x_buffer, g_many_out_buffers[7],
                                                   g_many_out_buffers[1], 1024);

        if (drafter_encode_rms_norm_dispatch(enc, post_norm_data, post_norm_bytes,
                                             g_many_out_buffers[1], g_many_out_buffers[0],
                                             1024, 1.0e-6f, err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);

        if (drafter_encode_affine_dispatch(enc, mlp_gate, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[4],
                                           err, errlen) != 0 ||
            drafter_encode_affine_dispatch(enc, mlp_up, 1024, bits, group_size,
                                           g_many_out_buffers[0], g_many_out_buffers[6],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        uint32_t hidden_len = 3584u;
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:0];
        [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:1];
        [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
        [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(3584, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (drafter_encode_affine_dispatch(enc, mlp_down, 3584, bits, group_size,
                                           g_linear_y_buffer, g_many_out_buffers[0],
                                           err, errlen) != 0) {
            [enc endEncoding];
            return -1;
        }
        drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);
        drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[1],
                                                   g_many_out_buffers[0],
                                                   g_out_buffer, 1024);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter fused full decoder layer command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        if (query_capture) {
            memcpy(query_capture, [g_many_out_buffers[3] contents], q_bytes);
        }
        memcpy(key_out, [g_many_out_buffers[5] contents], kv_slice_bytes);
        memcpy(value_out, [g_many_out_buffers[2] contents], kv_slice_bytes);
        memcpy(out, [g_out_buffer contents], out_bytes);
        return 0;
    }
}

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
        size_t errlen) {
    if (!layers || n_layers != 24 || !x || !last_hidden_out ||
        n_vec <= 0 || base_position < 0 || full_n_ctx != base_position + n_vec ||
        full_cache_capacity < full_n_ctx ||
        bits <= 0 || group_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident batch-prefill shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger x_bytes = (NSUInteger)n_vec * 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = (NSUInteger)n_vec * 4096u * sizeof(float);
        const NSUInteger qkv_bytes = (NSUInteger)n_vec * 6144u * sizeof(float);
        const NSUInteger qkvz_bytes = (NSUInteger)n_vec * 8192u * sizeof(float);
        const NSUInteger q_bytes = (NSUInteger)n_vec * 2048u * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_vec * 512u * sizeof(float);
        const NSUInteger kv_cache_bytes = (NSUInteger)full_cache_capacity * 512u * sizeof(float);
        const NSUInteger hidden_bytes = (NSUInteger)n_vec * 3584u * sizeof(float);
        const NSUInteger mlp_pair_bytes = hidden_bytes * 2u;
        const NSUInteger kq_bytes = (NSUInteger)n_vec * 16u * sizeof(float);
        const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
        const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
        const NSUInteger scan_params_bytes = (NSUInteger)n_vec * 16u * 2u * sizeof(float);
        const NSUInteger scan_debug_bytes = 16u * sizeof(uint32_t);
        if (drafter_ensure_io_buffers(x_bytes, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, x_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, qkvz_bytes > q_proj_bytes ? qkvz_bytes : q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, mlp_pair_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, qkv_bytes > hidden_bytes ? qkv_bytes : hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "resident batch query", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_gate_buffer, &g_attention_gate_bytes,
                                  q_bytes, "resident batch gate", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "resident batch attention", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  hidden_bytes, "resident batch MLP scratch", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_kq_buffer, &g_linear_kq_bytes,
                                  kq_bytes, "resident batch kq", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_params_buffer,
                                  &g_linear_scan_params_bytes,
                                  scan_params_bytes, "resident batch scan params", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_scan_debug_buffer,
                                  &g_linear_scan_debug_bytes,
                                  scan_debug_bytes, "resident batch scan debug", err, errlen) != 0) {
            return -1;
        }
        memcpy([g_x_buffer contents], x, x_bytes);

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLBuffer> src = g_x_buffer;
        id<MTLBuffer> dst = g_out_buffer;
	        uint32_t n_vec_u = (uint32_t)n_vec;
	        uint32_t hidden_len = (uint32_t)((NSUInteger)n_vec * 3584u);
	        NSUInteger delta_threads = drafter_linear_delta_threads();
        uint32_t scan_debug_enabled = drafter_scan_debug_enabled() ? 1u : 0u;
	        double prof_input_norm_ms = 0.0;
        double prof_linear_proj_ms = 0.0;
        double prof_linear_conv_ms = 0.0;
        double prof_linear_scan_ms = 0.0;
        double prof_linear_out_ms = 0.0;
        double prof_full_proj_ms = 0.0;
        double prof_full_rope_ms = 0.0;
        double prof_full_attn_ms = 0.0;
        double prof_full_out_ms = 0.0;
        double prof_post_norm_ms = 0.0;
        double prof_mlp_proj_ms = 0.0;
        double prof_mlp_down_ms = 0.0;
        ds4_drafter_metal_rms_norm_args rms_args = {
            .len = 1024,
            .eps = 1.0e-6f,
        };
        ds4_drafter_metal_rope_args rope_args = { .position = base_position, .n_vec = n_vec };
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_vec,
            .scale = 1.0f / 16.0f,
        };
        int input_norm_ready = 0;

        for (int layer = 0; layer < n_layers; layer++) {
            const ds4_drafter_metal_decoder_layer_job *job = layers + layer;
            if (!job->input_norm_data || !job->post_norm_data ||
                job->input_norm_bytes < 1024u * sizeof(uint16_t) ||
                job->post_norm_bytes < 1024u * sizeof(uint16_t) ||
                !job->mlp_gate_job.w_data || !job->mlp_up_job.w_data ||
                !job->mlp_down_job.w_data ||
                job->mlp_gate_job.rows != 3584 ||
                job->mlp_up_job.rows != 3584 ||
                job->mlp_down_job.rows != 1024) {
                [enc endEncoding];
                if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident batch layer metadata");
                return -1;
            }

            id<MTLBuffer> input_norm_buf = drafter_cached_buffer(job->input_norm_data, job->input_norm_bytes, err, errlen);
            id<MTLBuffer> post_norm_buf = drafter_cached_buffer(job->post_norm_data, job->post_norm_bytes, err, errlen);
            if (!input_norm_buf || !post_norm_buf) {
                [enc endEncoding];
                return -1;
            }
            int post_norm_done = 0;

            if (!input_norm_ready) {
                [enc setComputePipelineState:g_rms_norm_bf16_mat_round_pipeline];
                [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
                [enc setBuffer:input_norm_buf offset:0 atIndex:1];
                [enc setBuffer:src offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                if (drafter_profile_flush(&cb, &enc, &prof_input_norm_ms,
                                          "input norm", err, errlen) != 0) {
                    return -1;
                }
            }
            input_norm_ready = 0;

            if (job->is_linear) {
                if (!job->conv_data || !job->linear_norm_data ||
                    !job->a_log_data || !job->dt_bias_data ||
                    job->conv_bytes < 6144u * 4u * sizeof(uint16_t) ||
                    job->linear_norm_bytes < 128u * sizeof(uint16_t) ||
                    job->a_log_bytes < 16u * sizeof(float) ||
                    job->dt_bias_bytes < 16u * sizeof(uint16_t) ||
                    job->qkv_job.rows != 6144 || job->z_job.rows != 2048 ||
                    job->b_job.rows != 16 || job->a_job.rows != 16 ||
                    job->linear_out_job.rows != 1024) {
                    [enc endEncoding];
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident batch linear metadata");
                    return -1;
                }
                if (drafter_ensure_buffer(&g_linear_delta_state_buffers[layer],
                                          &g_linear_delta_state_bytes[layer],
                                          delta_state_bytes, "linear delta state", err, errlen) != 0 ||
                    drafter_ensure_buffer(&g_linear_conv_state_buffers[layer],
                                          &g_linear_conv_state_bytes[layer],
                                          conv_state_bytes, "linear conv state", err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                if (base_position == 0) {
                    memset([g_linear_conv_state_buffers[layer] contents], 0,
                           conv_state_bytes);
                    memset([g_linear_delta_state_buffers[layer] contents], 0,
                           delta_state_bytes);
                }
                id<MTLBuffer> conv_w_buf = drafter_cached_buffer(job->conv_data, job->conv_bytes, err, errlen);
                id<MTLBuffer> linear_norm_buf = drafter_cached_buffer(job->linear_norm_data, job->linear_norm_bytes, err, errlen);
                id<MTLBuffer> a_log_buf = drafter_cached_buffer(job->a_log_data, job->a_log_bytes, err, errlen);
                id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(job->dt_bias_data, job->dt_bias_bytes, err, errlen);
                if (!conv_w_buf || !linear_norm_buf || !a_log_buf || !dt_bias_buf) {
                    [enc endEncoding];
                    return -1;
                }
                int qkvz_pair_rc = 1;
                if (drafter_mps_linear_qkvz_pair_enabled()) {
                    qkvz_pair_rc = drafter_encode_affine_pair_mps_matmat(
                        &cb, &enc, &job->qkv_job, &job->z_job, n_vec, 1024,
                        bits, group_size, g_many_out_buffers[0],
                        g_many_out_buffers[1], err, errlen);
                    if (qkvz_pair_rc < 0) {
                        if (enc) [enc endEncoding];
                        return -1;
                    }
                }
                if (qkvz_pair_rc == 0) {
                    const ds4_drafter_metal_affine_job *ba_jobs[2] = {
                        &job->b_job, &job->a_job,
                    };
                    const int ba_cols[2] = {1024, 1024};
                    id<MTLBuffer> ba_x[2] = {
                        g_many_out_buffers[0], g_many_out_buffers[0],
                    };
                    id<MTLBuffer> ba_out[2] = {
                        g_many_out_buffers[3], g_many_out_buffers[4],
                    };
                    int ba_rc = drafter_encode_affine_mps_matmat_batch(
                        &cb, &enc, ba_jobs, ba_cols, ba_x, ba_out, 2,
                        n_vec, bits, group_size, err, errlen);
                    if (ba_rc < 0) {
                        if (enc) [enc endEncoding];
                        return -1;
                    }
                    if (ba_rc > 0 &&
                        (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->b_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[3],
                                                                    32u, err, errlen) != 0 ||
                         drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->a_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[4],
                                                                    32u, err, errlen) != 0)) {
                        if (enc) [enc endEncoding];
                        return -1;
                    }
                } else {
                    const ds4_drafter_metal_affine_job *linear_proj_jobs[4] = {
                        &job->qkv_job, &job->z_job, &job->b_job, &job->a_job,
                    };
                    const int linear_proj_cols[4] = {1024, 1024, 1024, 1024};
                    id<MTLBuffer> linear_proj_x[4] = {
                        g_many_out_buffers[0], g_many_out_buffers[0],
                        g_many_out_buffers[0], g_many_out_buffers[0],
                    };
                    id<MTLBuffer> linear_proj_out[4] = {
                        g_many_out_buffers[1], g_many_out_buffers[2],
                        g_many_out_buffers[3], g_many_out_buffers[4],
                    };
                    int linear_proj_rc = drafter_encode_affine_mps_matmat_batch(
                        &cb, &enc, linear_proj_jobs, linear_proj_cols,
                        linear_proj_x, linear_proj_out, 4, n_vec, bits,
                        group_size, err, errlen);
                    if (linear_proj_rc < 0) {
                        if (enc) [enc endEncoding];
                        return -1;
                    }
                    if (linear_proj_rc > 0 &&
                        (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->qkv_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[1],
                                                                    32u, err, errlen) != 0 ||
                         drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->z_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[2],
                                                                    32u, err, errlen) != 0 ||
                         drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->b_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[3],
                                                                    32u, err, errlen) != 0 ||
                         drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->a_job,
                                                                    n_vec, 1024, bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[4],
                                                                    32u, err, errlen) != 0)) {
                        if (enc) [enc endEncoding];
                        return -1;
                    }
                }
                if (drafter_profile_flush(&cb, &enc, &prof_linear_proj_ms,
                                          "linear projections", err, errlen) != 0) {
                    return -1;
                }
                if (layer == drafter_batch_hidden_debug_layer()) {
                    if (drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[1],
                                                         "qkv", layer, n_vec,
                                                         6144, err, errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         qkvz_pair_rc == 0 ? g_many_out_buffers[1] : g_many_out_buffers[2],
                                                         "z", layer, n_vec,
                                                         2048, err, errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[3],
                                                         "b", layer, n_vec,
                                                         16, err, errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[4],
                                                         "a", layer, n_vec,
                                                         16, err, errlen) != 0) {
                        return -1;
                    }
                }

                [enc setComputePipelineState:qkvz_pair_rc == 0 ?
                    (base_position > 0 ? g_linear_conv_stateful_qkvz_mat_pipeline : g_linear_conv_qkvz_mat_pipeline) :
                    (base_position > 0 ? g_linear_conv_stateful_mat_pipeline : g_linear_conv_mat_pipeline)];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:conv_w_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
                if (base_position > 0) {
                    [enc setBuffer:g_linear_conv_state_buffers[layer] offset:0 atIndex:4];
                }
                [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 6144u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:qkvz_pair_rc == 0 ?
                    g_linear_conv_state_qkvz_mat_pipeline : g_linear_conv_state_mat_pipeline];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
                [enc setBuffer:g_linear_conv_state_buffers[layer] offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(3u * 6144u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:g_linear_qk_norm_kq_mat_pipeline];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:3];
                [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:4];
                [enc setThreadgroupMemoryLength:2u * 128u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                if (drafter_profile_flush(&cb, &enc, &prof_linear_conv_ms,
                                          "linear conv/qk norm", err, errlen) != 0) {
                    return -1;
                }
                if (layer == drafter_batch_hidden_debug_layer()) {
                    if (drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[5],
                                                         "conv_out", layer,
                                                         n_vec, 6144, err,
                                                         errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[6],
                                                         "q_norm", layer,
                                                         n_vec, 2048, err,
                                                         errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[7],
                                                         "k_norm", layer,
                                                         n_vec, 2048, err,
                                                         errlen) != 0 ||
                        drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_linear_kq_buffer,
                                                         "kq", layer, n_vec,
                                                         16, err, errlen) != 0) {
                        return -1;
                    }
                }

                [enc setComputePipelineState:g_linear_scan_params_pipeline];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:a_log_buf offset:0 atIndex:1];
                [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
                [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)n_vec * 16u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                if (base_position == 0) {
                    memset([g_linear_delta_state_buffers[layer] contents], 0,
                           delta_state_bytes);
                }
                if (scan_debug_enabled) {
                    memset([g_linear_scan_debug_buffer contents], 0,
                           scan_debug_bytes);
                }

                int use_scan4 = drafter_linear_scan4_enabled() &&
                                delta_threads == 32u &&
                                scan_debug_enabled == 0u;
                int use_scan2 = !use_scan4 &&
                                drafter_linear_scan2_enabled() &&
                                delta_threads == 32u &&
                                scan_debug_enabled == 0u;
                [enc setComputePipelineState:use_scan4 ? g_linear_delta_scan4_pipeline :
                    (use_scan2 ? g_linear_delta_scan2_pipeline : g_linear_delta_scan_pipeline)];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:a_log_buf offset:0 atIndex:1];
                [enc setBuffer:dt_bias_buf offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
                [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:5];
                [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:6];
		                [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:7];
                [enc setBuffer:g_linear_delta_state_buffers[layer] offset:0 atIndex:8];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:9];
                [enc setBuffer:g_linear_scan_params_buffer offset:0 atIndex:10];
                [enc setBuffer:g_linear_kq_buffer offset:0 atIndex:11];
                [enc setBytes:&scan_debug_enabled length:sizeof(scan_debug_enabled) atIndex:12];
                [enc setBuffer:g_linear_scan_debug_buffer offset:0 atIndex:13];
			                [enc setThreadgroupMemoryLength:delta_threads * sizeof(float) atIndex:0];
			                [enc dispatchThreadgroups:MTLSizeMake(16, use_scan4 ? 32 : (use_scan2 ? 64 : 128), 1)
			                     threadsPerThreadgroup:MTLSizeMake(delta_threads, 1, 1)];

                if (scan_debug_enabled) {
                    double t0 = drafter_now_ms();
                    [enc endEncoding];
                    enc = nil;
                    [cb commit];
                    [cb waitUntilCompleted];
                    if (cb.status == MTLCommandBufferStatusError) {
                        NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter scan debug command buffer failed";
                        return drafter_metal_fail(err, errlen, msg);
                    }
                    prof_linear_scan_ms += drafter_now_ms() - t0;
                    const uint32_t *scan_debug = (const uint32_t *)[g_linear_scan_debug_buffer contents];
                    uint32_t flags = scan_debug ? scan_debug[0] : 0u;
                    if (flags != 0u) {
                        fprintf(stderr,
                                "NATIVE_SCAN_DEBUG layer=%d flags=0x%08x beta=%u decay=%u k=%u q=%u st_decay=%u kv_sum=%u state_q_sum=%u kq=%u conv=%u delta=%u st_update=%u out=%u\n",
                                layer, flags,
                                (flags & 1u) != 0u, (flags & 2u) != 0u,
                                (flags & 4u) != 0u, (flags & 8u) != 0u,
                                (flags & 16u) != 0u, (flags & 32u) != 0u,
                                (flags & 64u) != 0u, (flags & 128u) != 0u,
                                (flags & 256u) != 0u, (flags & 512u) != 0u,
                                (flags & 1024u) != 0u, (flags & 2048u) != 0u);
                    }
                    cb = [g_drafter_queue commandBuffer];
                    enc = [cb computeCommandEncoder];
                    if (!cb || !enc) {
                        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to recreate command buffer after scan debug");
                        return -1;
                    }
                } else {
                    if (drafter_profile_flush(&cb, &enc, &prof_linear_scan_ms,
                                              "linear scan", err, errlen) != 0) {
                        return -1;
                    }
                }
                if (layer == drafter_batch_hidden_debug_layer() &&
                    drafter_debug_batch_hidden_flush(&cb, &enc,
                                                     g_linear_y_buffer,
                                                     "linear_scan_y", layer,
                                                     n_vec, 2048, err,
                                                     errlen) != 0) {
                    return -1;
                }

                [enc setComputePipelineState:qkvz_pair_rc == 0 ?
                    g_linear_gate_qkvz_mat_pipeline : g_linear_gate_mat_pipeline];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:0];
                [enc setBuffer:linear_norm_buf offset:0 atIndex:1];
                [enc setBuffer:qkvz_pair_rc == 0 ? g_many_out_buffers[1] : g_many_out_buffers[2]
                         offset:0 atIndex:2];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:3];
                [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(16, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                if (layer == drafter_batch_hidden_debug_layer() &&
                    drafter_debug_batch_hidden_flush(&cb, &enc,
                                                     g_attention_out_buffer,
                                                     "linear_gate", layer,
                                                     n_vec, 2048, err,
                                                     errlen) != 0) {
                    return -1;
                }

                if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->linear_out_job,
                                                               n_vec, 2048, bits, group_size,
                                                               g_attention_out_buffer,
                                                               g_many_out_buffers[6],
                                                               32u, err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                if (layer == drafter_batch_hidden_debug_layer()) {
                    drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[6],
                                                       (NSUInteger)n_vec * 1024u);
                    if (drafter_debug_batch_hidden_flush(&cb, &enc,
                                                         g_many_out_buffers[6],
                                                         "linear_out_proj", layer,
                                                         n_vec, 1024, err,
                                                         errlen) != 0) {
                        return -1;
                    }
                    drafter_encode_residual_add_round_dispatch(enc, src,
                                                               g_many_out_buffers[6],
                                                               g_many_out_buffers[2],
                                                               (NSUInteger)n_vec * 1024u);
                } else {
                    drafter_encode_residual_add_round_pre_b_round_norm_dispatch(
                        enc, &rms_args, post_norm_buf, src,
                        g_many_out_buffers[6], g_many_out_buffers[2],
                        g_many_out_buffers[0], n_vec);
                    post_norm_done = 1;
                }
                if (drafter_profile_flush(&cb, &enc, &prof_linear_out_ms,
                                          "linear output", err, errlen) != 0) {
                    return -1;
                }
                if (layer == drafter_batch_hidden_debug_layer() &&
                    drafter_debug_batch_hidden_flush(&cb, &enc,
                                                     g_many_out_buffers[2],
                                                     "linear_out", layer,
                                                     n_vec, 1024, err,
                                                     errlen) != 0) {
                    return -1;
                }
            } else {
                if (!job->q_norm_data || !job->k_norm_data ||
                    job->q_norm_bytes < 256u * sizeof(uint16_t) ||
                    job->k_norm_bytes < 256u * sizeof(uint16_t) ||
                    job->q_job.rows != 4096 || job->k_job.rows != 512 ||
                    job->v_job.rows != 512 || job->o_job.rows != 1024) {
                    [enc endEncoding];
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident batch full metadata");
                    return -1;
                }
                if (drafter_ensure_buffer(&g_attention_key_cache_buffers[layer],
                                          &g_attention_key_cache_bytes[layer],
                                          kv_cache_bytes, "attention key cache", err, errlen) != 0 ||
                    drafter_ensure_buffer(&g_attention_value_cache_buffers[layer],
                                          &g_attention_value_cache_bytes[layer],
                                          kv_cache_bytes, "attention value cache", err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                id<MTLBuffer> q_norm_buf = drafter_cached_buffer(job->q_norm_data, job->q_norm_bytes, err, errlen);
                id<MTLBuffer> k_norm_buf = drafter_cached_buffer(job->k_norm_data, job->k_norm_bytes, err, errlen);
                if (!q_norm_buf || !k_norm_buf) {
                    [enc endEncoding];
                    return -1;
                }
                const ds4_drafter_metal_affine_job *full_proj_jobs[3] = {
                    &job->q_job, &job->k_job, &job->v_job,
                };
                const int full_proj_cols[3] = {1024, 1024, 1024};
                id<MTLBuffer> full_proj_x[3] = {
                    g_many_out_buffers[0], g_many_out_buffers[0],
                    g_many_out_buffers[0],
                };
                id<MTLBuffer> value_proj_out =
                    base_position == 0 ? g_attention_value_cache_buffers[layer] : g_many_out_buffers[5];
                id<MTLBuffer> full_proj_out[3] = {
                    g_many_out_buffers[1], g_many_out_buffers[2],
                    value_proj_out,
                };
                int full_proj_rc = drafter_encode_affine_mps_matmat_batch(
                    &cb, &enc, full_proj_jobs, full_proj_cols, full_proj_x,
                    full_proj_out, 3, n_vec, bits, group_size, err, errlen);
                if (full_proj_rc < 0) {
                    if (enc) [enc endEncoding];
                    return -1;
                }
                if (full_proj_rc > 0 &&
                    (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->q_job,
                                                                n_vec, 1024, bits, group_size,
                                                                g_many_out_buffers[0],
                                                                g_many_out_buffers[1],
                                                                64u, err, errlen) != 0 ||
                     drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->k_job,
                                                                n_vec, 1024, bits, group_size,
                                                                g_many_out_buffers[0],
                                                                g_many_out_buffers[2],
                                                                64u, err, errlen) != 0 ||
                     drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->v_job,
                                                                n_vec, 1024, bits, group_size,
                                                                g_many_out_buffers[0],
                                                                value_proj_out,
                                                                0u, err, errlen) != 0)) {
                    if (enc) [enc endEncoding];
                    return -1;
                }
                if (drafter_profile_flush(&cb, &enc, &prof_full_proj_ms,
                                          "full projections", err, errlen) != 0) {
                    return -1;
                }
                if (base_position > 0) {
                    if (enc) {
                        [enc endEncoding];
                        enc = nil;
                    }
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:value_proj_out
                            sourceOffset:0
                                toBuffer:g_attention_value_cache_buffers[layer]
                       destinationOffset:(NSUInteger)base_position * 512u * sizeof(float)
                                    size:kv_bytes];
                    [blit endEncoding];
                    enc = [cb computeCommandEncoder];
                    if (!enc) {
                        if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to resume encoder after value cache blit");
                        return -1;
                    }
                }

                [enc setComputePipelineState:g_full_q_norm_rope_mat_pipeline];
                [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
                [enc setBuffer:q_norm_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
                [enc setBuffer:g_attention_q_buffer offset:0 atIndex:3];
                [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:4];
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                g_attention_prepacked_k_f16_n_ctx = 0;
                const int will_use_grouped_attention =
                    drafter_mps_grouped_causal_attention_enabled() &&
                    (drafter_mps_causal_attention_max_n_vec() <= 0 ||
                     n_vec > drafter_mps_causal_attention_max_n_vec());
                const int prepack_k_f16 =
                    base_position == 0 &&
                    drafter_mps_attention_prepack_k_enabled() &&
                    will_use_grouped_attention &&
                    drafter_mps_attention_f16_enabled() &&
                    g_full_k_norm_rope_mat_pack_f16_pipeline &&
                    g_attention_pack_v_head_f16x4_pipeline;
                if (prepack_k_f16 &&
                    drafter_ensure_private_buffer(&g_attention_k_head_f16_buffer,
                                                  &g_attention_k_head_f16_bytes,
                                                  2u * (NSUInteger)n_vec * 256u * sizeof(uint16_t),
                                                  "attention prepacked key f16 head", err, errlen) != 0) {
                    return -1;
                }

                [enc setComputePipelineState:prepack_k_f16 ?
                    g_full_k_norm_rope_mat_pack_f16_pipeline :
                    g_full_k_norm_rope_mat_pipeline];
                [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
                [enc setBuffer:k_norm_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
                [enc setBuffer:g_attention_key_cache_buffers[layer]
                        offset:(NSUInteger)base_position * 512u * sizeof(float)
                        atIndex:3];
                if (prepack_k_f16) {
                    [enc setBuffer:g_attention_k_head_f16_buffer offset:0 atIndex:4];
                }
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(2, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                g_attention_prepacked_k_f16_n_ctx = prepack_k_f16 ? n_vec : 0;

                if (drafter_profile_flush(&cb, &enc, &prof_full_rope_ms,
                                          "full rope/cache", err, errlen) != 0) {
                    return -1;
                }

                id<MTLBuffer> saved_attention_keys_buffer = g_attention_keys_buffer;
                id<MTLBuffer> saved_attention_values_buffer = g_attention_values_buffer;
                NSUInteger saved_attention_keys_bytes = g_attention_keys_bytes;
                NSUInteger saved_attention_values_bytes = g_attention_values_bytes;
                g_attention_keys_buffer = g_attention_key_cache_buffers[layer];
                g_attention_values_buffer = g_attention_value_cache_buffers[layer];
                g_attention_keys_bytes = g_attention_key_cache_bytes[layer];
                g_attention_values_bytes = g_attention_value_cache_bytes[layer];
                int mps_attn_rc = base_position == 0 ?
                    drafter_encode_mps_causal_attention_mat(&cb, &enc, n_vec,
                                                            err, errlen) :
                    drafter_encode_mps_causal_attention_group_chunk_mat(
                        cb, &enc, full_n_ctx, base_position, n_vec,
                        drafter_mps_causal_attention_block_rows(), err, errlen);
                if (mps_attn_rc < 0) {
                    g_attention_prepacked_k_f16_n_ctx = 0;
                    g_attention_keys_buffer = saved_attention_keys_buffer;
                    g_attention_values_buffer = saved_attention_values_buffer;
                    g_attention_keys_bytes = saved_attention_keys_bytes;
                    g_attention_values_bytes = saved_attention_values_bytes;
                    [enc endEncoding];
                    return -1;
                }
                if (mps_attn_rc > 0) {
                    g_attention_prepacked_k_f16_n_ctx = 0;
                    if (base_position > 0) {
                        g_attention_keys_buffer = saved_attention_keys_buffer;
                        g_attention_values_buffer = saved_attention_values_buffer;
                        g_attention_keys_bytes = saved_attention_keys_bytes;
                        g_attention_values_bytes = saved_attention_values_bytes;
                        [enc endEncoding];
                        if (err && errlen) snprintf(err, errlen, "native Metal drafter chunk attention requires grouped MPS attention");
                        return -1;
                    }
                    [enc setComputePipelineState:drafter_attention_context_causal_fused_pipeline()];
                    [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                    [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
                    [enc setBuffer:g_attention_gate_buffer offset:0 atIndex:2];
                    [enc setBuffer:g_attention_keys_buffer offset:0 atIndex:3];
                    [enc setBuffer:g_attention_values_buffer offset:0 atIndex:4];
                    [enc setBuffer:g_attention_out_buffer offset:0 atIndex:5];
                    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_vec, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                g_attention_prepacked_k_f16_n_ctx = 0;
                g_attention_keys_buffer = saved_attention_keys_buffer;
                g_attention_values_buffer = saved_attention_values_buffer;
                g_attention_keys_bytes = saved_attention_keys_bytes;
                g_attention_values_bytes = saved_attention_values_bytes;
                if (drafter_profile_flush(&cb, &enc, &prof_full_attn_ms,
                                          "full attention", err, errlen) != 0) {
                    return -1;
                }

                if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->o_job,
                                                               n_vec, 2048, bits, group_size,
                                                               g_attention_out_buffer,
                                                               g_many_out_buffers[6],
                                                               64u, err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                drafter_encode_residual_add_round_pre_b_round_norm_dispatch(
                    enc, &rms_args, post_norm_buf, src,
                    g_many_out_buffers[6], g_many_out_buffers[2],
                    g_many_out_buffers[0], n_vec);
                post_norm_done = 1;
                if (drafter_profile_flush(&cb, &enc, &prof_full_out_ms,
                                          "full output", err, errlen) != 0) {
                    return -1;
                }
                if (layer == drafter_batch_hidden_debug_layer() &&
                    drafter_debug_batch_hidden_flush(&cb, &enc,
                                                     g_many_out_buffers[2],
                                                     "full_out", layer,
                                                     n_vec, 1024, err,
                                                     errlen) != 0) {
                    return -1;
                }
            }

            if (!post_norm_done) {
                [enc setComputePipelineState:g_rms_norm_bf16_mat_round_pipeline];
                [enc setBytes:&rms_args length:sizeof(rms_args) atIndex:0];
                [enc setBuffer:post_norm_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)n_vec, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                if (drafter_profile_flush(&cb, &enc, &prof_post_norm_ms,
                                          "post norm", err, errlen) != 0) {
                    return -1;
                }
            }
            if (layer == drafter_batch_hidden_debug_layer() &&
                drafter_debug_batch_hidden_flush(&cb, &enc,
                                                 g_many_out_buffers[0],
                                                 "post_norm", layer, n_vec,
                                                 1024, err, errlen) != 0) {
                return -1;
            }

            int mlp_pair_rc = 1;
            if (drafter_mps_mlp_pair_enabled()) {
                mlp_pair_rc = drafter_encode_affine_pair_mps_matmat(&cb, &enc,
                                                                    &job->mlp_gate_job,
                                                                    &job->mlp_up_job,
                                                                    n_vec, 1024,
                                                                    bits, group_size,
                                                                    g_many_out_buffers[0],
                                                                    g_many_out_buffers[3],
                                                                    err, errlen);
                if (mlp_pair_rc < 0) {
                    [enc endEncoding];
                    return -1;
                }
            }
            int mlp_gate_up_rc = 1;
            if (mlp_pair_rc > 0 &&
                !drafter_mps_mlp_pair_enabled()) {
                const ds4_drafter_metal_affine_job *mlp_proj_jobs[2] = {
                    &job->mlp_gate_job, &job->mlp_up_job,
                };
                const int mlp_proj_cols[2] = {1024, 1024};
                id<MTLBuffer> mlp_proj_x[2] = {
                    g_many_out_buffers[0], g_many_out_buffers[0],
                };
                id<MTLBuffer> mlp_proj_out[2] = {
                    g_many_out_buffers[3], g_many_out_buffers[4],
                };
                mlp_gate_up_rc = drafter_encode_affine_mps_matmat_batch(
                    &cb, &enc, mlp_proj_jobs, mlp_proj_cols, mlp_proj_x,
                    mlp_proj_out, 2, n_vec, bits, group_size, err, errlen);
                if (mlp_gate_up_rc < 0) {
                    if (enc) [enc endEncoding];
                    return -1;
                }
            }
            if (mlp_pair_rc > 0 && mlp_gate_up_rc > 0 &&
                (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->mlp_gate_job,
                                                            n_vec, 1024, bits, group_size,
                                                            g_many_out_buffers[0],
                                                            g_many_out_buffers[3],
                                                            32u, err, errlen) != 0 ||
                 drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->mlp_up_job,
                                                            n_vec, 1024, bits, group_size,
                                                            g_many_out_buffers[0],
                                                            g_many_out_buffers[4],
                                                            32u, err, errlen) != 0)) {
                if (enc) [enc endEncoding];
                return -1;
            }
            if (drafter_profile_flush(&cb, &enc, &prof_mlp_proj_ms,
                                      "mlp gate/up", err, errlen) != 0) {
                return -1;
            }
            if (mlp_pair_rc == 0) {
                [enc setComputePipelineState:g_swiglu_packed_pair_pipeline];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:0];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:1];
                [enc setBytes:&n_vec_u length:sizeof(n_vec_u) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_len, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                const int use_swiglu_x4 = g_swiglu_x4_pipeline && ((hidden_len & 3u) == 0u);
                [enc setComputePipelineState:use_swiglu_x4 ? g_swiglu_x4_pipeline : g_swiglu_pipeline];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:0];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:1];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
                uint32_t swiglu_len = use_swiglu_x4 ? (hidden_len >> 2) : hidden_len;
                [enc setBytes:&swiglu_len length:sizeof(swiglu_len) atIndex:3];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)swiglu_len, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (layer == drafter_batch_hidden_debug_layer() &&
                drafter_debug_batch_hidden_flush(&cb, &enc, g_linear_y_buffer,
                                                 "swiglu", layer, n_vec,
                                                 3584, err, errlen) != 0) {
                return -1;
            }

            if (drafter_encode_affine_matmat_best_dispatch(&cb, &enc, &job->mlp_down_job,
                                                           n_vec, 3584, bits, group_size,
                                                           g_linear_y_buffer,
                                                           g_many_out_buffers[7],
                                                           32u, err, errlen) != 0) {
                [enc endEncoding];
                return -1;
            }
            if (layer == drafter_batch_hidden_debug_layer()) {
                drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7],
                                                   (NSUInteger)n_vec * 1024u);
                if (drafter_debug_batch_hidden_flush(&cb, &enc,
                                                     g_many_out_buffers[7],
                                                     "mlp_down", layer, n_vec,
                                                     1024, err, errlen) != 0) {
                    return -1;
                }
                drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[2],
                                                           g_many_out_buffers[7],
                                                           dst,
                                                           (NSUInteger)n_vec * 1024u);
                input_norm_ready = 0;
            } else {
                input_norm_ready = 0;
                if (layer + 1 < n_layers &&
                    layers[layer + 1].input_norm_data &&
                    layers[layer + 1].input_norm_bytes >= 1024u * sizeof(uint16_t)) {
                    id<MTLBuffer> next_input_norm_buf =
                        drafter_cached_buffer(layers[layer + 1].input_norm_data,
                                              layers[layer + 1].input_norm_bytes,
                                              err, errlen);
                    if (!next_input_norm_buf) {
                        [enc endEncoding];
                        return -1;
                    }
                    drafter_encode_residual_add_round_pre_b_round_norm_dispatch(
                        enc, &rms_args, next_input_norm_buf,
                        g_many_out_buffers[2], g_many_out_buffers[7], dst,
                        g_many_out_buffers[0], n_vec);
                    input_norm_ready = 1;
                } else {
                    drafter_encode_residual_add_round_pre_b_round_dispatch(
                        enc, g_many_out_buffers[2], g_many_out_buffers[7], dst,
                        (NSUInteger)n_vec * 1024u);
                }
            }
            if (drafter_profile_flush(&cb, &enc, &prof_mlp_down_ms,
                                      "mlp down", err, errlen) != 0) {
                return -1;
            }
            if (drafter_debug_batch_hidden_flush(&cb, &enc, dst, "layer_out",
                                                 layer, n_vec, 1024, err,
                                                 errlen) != 0) {
                return -1;
            }
            id<MTLBuffer> tmp = src;
            src = dst;
            dst = tmp;
        }
        if (enc) [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter resident batch-prefill command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        if (drafter_batch_profile_enabled()) {
            fprintf(stderr,
                    "native Metal batch profile: input_norm=%.3fms linear_proj=%.3fms linear_conv=%.3fms linear_scan=%.3fms linear_out=%.3fms full_proj=%.3fms full_rope=%.3fms full_attn=%.3fms full_out=%.3fms post_norm=%.3fms mlp_proj=%.3fms mlp_down=%.3fms\n",
                    prof_input_norm_ms,
                    prof_linear_proj_ms,
                    prof_linear_conv_ms,
                    prof_linear_scan_ms,
                    prof_linear_out_ms,
                    prof_full_proj_ms,
                    prof_full_rope_ms,
                    prof_full_attn_ms,
                    prof_full_out_ms,
                    prof_post_norm_ms,
                    prof_mlp_proj_ms,
                    prof_mlp_down_ms);
        }
        memcpy(last_hidden_out,
               (const uint8_t *)[src contents] + ((NSUInteger)n_vec - 1u) * 1024u * sizeof(float),
               1024u * sizeof(float));
        if (key_out_by_layer || value_out_by_layer) {
            for (int layer = 0; layer < n_layers; layer++) {
                if (layers[layer].is_linear) continue;
                if (key_out_by_layer && key_out_by_layer[layer]) {
                    memcpy(key_out_by_layer[layer] + (NSUInteger)base_position * 512u,
                           (const uint8_t *)[g_attention_key_cache_buffers[layer] contents] +
                               (NSUInteger)base_position * 512u * sizeof(float),
                           kv_bytes);
                }
                if (value_out_by_layer && value_out_by_layer[layer]) {
                    memcpy(value_out_by_layer[layer] + (NSUInteger)base_position * 512u,
                           (const uint8_t *)[g_attention_value_cache_buffers[layer] contents] +
                               (NSUInteger)base_position * 512u * sizeof(float),
                           kv_bytes);
                }
            }
        }
        return 0;
    }
}

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
        size_t errlen) {
    return ds4_drafter_metal_qwen_batch_prefill_chunk_u32(
        layers, n_layers, x, n_vec, 0, n_vec, full_cache_capacity,
        bits, group_size, key_out_by_layer, value_out_by_layer,
        last_hidden_out, err, errlen);
}

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
        size_t errlen) {
    if (!layers || n_layers != 24 || !x ||
        (need_output && (!final_norm_data || !out)) ||
        full_cache_index < 0 || full_n_ctx <= 0 ||
        full_cache_capacity < full_n_ctx ||
        full_cache_index != full_n_ctx - 1 ||
        bits <= 0 || group_size <= 0 ||
        (need_output && final_norm_bytes < 1024u * sizeof(uint16_t))) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident hidden-step shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const int token_profile = drafter_token_profile_enabled();
        const double token_profile_start = token_profile ? drafter_now_ms() : 0.0;
        double prof_linear_ms = 0.0;
        double prof_full_qkv_ms = 0.0;
        double prof_full_logits_ms = 0.0;
        double prof_full_context_ms = 0.0;
        double prof_full_o_ms = 0.0;
        double prof_mlp_ms = 0.0;
        double prof_final_ms = 0.0;
        double prof_wait_ms = 0.0;
        double prof_copy_ms = 0.0;
        int prof_linear_layers = 0;
        int prof_full_layers = 0;
        const NSUInteger hidden_bytes = 1024u * sizeof(float);
        const NSUInteger q_proj_bytes = 4096u * sizeof(float);
        const NSUInteger kv_slice_bytes = 512u * sizeof(float);
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger kv_cache_bytes = (NSUInteger)full_cache_capacity * 512u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)full_cache_capacity * 8u * sizeof(float);
        const int wait_for_completion = need_output || key_out_by_layer ||
                                        value_out_by_layer || query_capture_by_layer;
        int last_full_layer = -1;
        for (int i = 0; i < n_layers; i++) {
            if (!layers[i].is_linear) last_full_layer = i;
        }
        if (drafter_ensure_buffer(&g_token_hidden_buffers[0],
                                  &g_token_hidden_bytes[0],
                                  hidden_bytes, "token hidden A", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_token_hidden_buffers[1],
                                  &g_token_hidden_bytes[1],
                                  hidden_bytes, "token hidden B", err, errlen) != 0 ||
            drafter_ensure_io_buffers(hidden_bytes, hidden_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(0, q_proj_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(1, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(2, 2048u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(3, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(4, 6144u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(5, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(6, q_bytes, err, errlen) != 0 ||
            drafter_ensure_many_out_buffer(7, 3584u * sizeof(float), err, errlen) != 0 ||
            drafter_ensure_buffer(&g_linear_y_buffer, &g_linear_y_bytes,
                                  3584u * sizeof(float), "resident token scratch", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_attention_out_buffer, &g_attention_out_bytes,
                                  q_bytes, "attention output", err, errlen) != 0) {
            return -1;
        }
        id<MTLBuffer> token_input_buffer = nil;
        if (wait_for_completion) {
            token_input_buffer = g_token_hidden_buffers[0];
            memcpy([token_input_buffer contents], x, hidden_bytes);
        } else {
            token_input_buffer = [g_drafter_device newBufferWithBytes:x
                                                                length:hidden_bytes
                                                               options:MTLResourceStorageModeShared];
            if (!token_input_buffer) {
                if (err && errlen) snprintf(err, errlen, "native Metal drafter failed to allocate resident token input buffer");
                return -1;
            }
        }

        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLBuffer> src = token_input_buffer;
        id<MTLBuffer> dst = g_token_hidden_buffers[1];
        uint32_t hidden_len = 3584u;
        for (int layer = 0; layer < n_layers; layer++) {
            const ds4_drafter_metal_decoder_layer_job *job = layers + layer;
            if (!job->input_norm_data || !job->post_norm_data ||
                job->input_norm_bytes < 1024u * sizeof(uint16_t) ||
                job->post_norm_bytes < 1024u * sizeof(uint16_t) ||
                !job->mlp_gate_job.w_data || !job->mlp_up_job.w_data ||
                !job->mlp_down_job.w_data ||
                job->mlp_gate_job.rows != 3584 ||
                job->mlp_up_job.rows != 3584 ||
                job->mlp_down_job.rows != 1024) {
                [enc endEncoding];
                if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident layer metadata");
                return -1;
            }

            if (job->is_linear) {
                if (!job->conv_data || !job->linear_norm_data ||
                    !job->a_log_data || !job->dt_bias_data ||
                    job->conv_bytes < 6144u * 4u * sizeof(uint16_t) ||
                    job->linear_norm_bytes < 128u * sizeof(uint16_t) ||
                    job->a_log_bytes < 16u * sizeof(float) ||
                    job->dt_bias_bytes < 16u * sizeof(uint16_t) ||
                    job->qkv_job.rows != 6144 || job->z_job.rows != 2048 ||
                    job->b_job.rows != 16 || job->a_job.rows != 16 ||
                    job->linear_out_job.rows != 1024) {
                    [enc endEncoding];
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident linear metadata");
                    return -1;
                }
                const NSUInteger conv_state_bytes = 3u * 6144u * sizeof(float);
                const NSUInteger delta_state_bytes = 16u * 128u * 128u * sizeof(float);
                const int state_was_missing = !g_linear_conv_state_buffers[layer] ||
                                              !g_linear_delta_state_buffers[layer];
                if (drafter_ensure_buffer(&g_linear_conv_state_buffers[layer],
                                          &g_linear_conv_state_bytes[layer],
                                          conv_state_bytes, "linear conv state", err, errlen) != 0 ||
                    drafter_ensure_buffer(&g_linear_delta_state_buffers[layer],
                                          &g_linear_delta_state_bytes[layer],
                                          delta_state_bytes, "linear delta state", err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                if (state_was_missing) {
                    memset([g_linear_conv_state_buffers[layer] contents], 0, conv_state_bytes);
                    memset([g_linear_delta_state_buffers[layer] contents], 0, delta_state_bytes);
                }
                id<MTLBuffer> conv_w_buf = drafter_cached_buffer(job->conv_data, job->conv_bytes, err, errlen);
                id<MTLBuffer> linear_norm_buf = drafter_cached_buffer(job->linear_norm_data, job->linear_norm_bytes, err, errlen);
                id<MTLBuffer> a_log_buf = drafter_cached_buffer(job->a_log_data, job->a_log_bytes, err, errlen);
                id<MTLBuffer> dt_bias_buf = drafter_cached_buffer(job->dt_bias_data, job->dt_bias_bytes, err, errlen);
                if (!conv_w_buf || !linear_norm_buf || !a_log_buf || !dt_bias_buf) {
                    [enc endEncoding];
                    return -1;
                }
                if (drafter_encode_rms_norm_dispatch(enc, job->input_norm_data, job->input_norm_bytes,
                                                     src, g_many_out_buffers[0],
                                                     1024, 1.0e-6f, err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);
                if (drafter_encode_affine_dispatch(enc, &job->qkv_job, 1024, bits, group_size,
                                                   g_many_out_buffers[0], g_many_out_buffers[1],
                                                   err, errlen) != 0 ||
                    drafter_encode_affine_dispatch(enc, &job->z_job, 1024, bits, group_size,
                                                   g_many_out_buffers[0], g_many_out_buffers[2],
                                                   err, errlen) != 0 ||
                    drafter_encode_affine_dispatch(enc, &job->b_job, 1024, bits, group_size,
                                                   g_many_out_buffers[0], g_many_out_buffers[3],
                                                   err, errlen) != 0 ||
                    drafter_encode_affine_dispatch(enc, &job->a_job, 1024, bits, group_size,
                                                   g_many_out_buffers[0], g_out_buffer,
                                                   err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                [enc setComputePipelineState:g_linear_conv_pipeline];
                [enc setBuffer:conv_w_buf offset:0 atIndex:0];
                [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:1];
                [enc setBuffer:g_linear_conv_state_buffers[layer] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(6144, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:g_linear_qk_norm_pipeline];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:0];
                [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:2];
                [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(16, 2, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:g_linear_delta_pipeline];
                [enc setBuffer:a_log_buf offset:0 atIndex:0];
                [enc setBuffer:dt_bias_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:2];
                [enc setBuffer:g_out_buffer offset:0 atIndex:3];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
                [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:5];
                [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:6];
                [enc setBuffer:g_linear_delta_state_buffers[layer] offset:0 atIndex:7];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:8];
                [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(16, 128, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:g_linear_gate_pipeline];
                [enc setBuffer:linear_norm_buf offset:0 atIndex:0];
                [enc setBuffer:g_many_out_buffers[2] offset:0 atIndex:1];
                [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[6] offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                if (drafter_encode_affine_dispatch(enc, &job->linear_out_job, 2048, bits, group_size,
                                                   g_many_out_buffers[6], g_many_out_buffers[1],
                                                   err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[1], 1024);
                drafter_encode_residual_add_round_dispatch(enc, src, g_many_out_buffers[1],
                                                           g_many_out_buffers[2], 1024);
                if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_linear_ms,
                                               "resident token linear core", err, errlen) != 0) {
                    return -1;
                }
                prof_linear_layers++;
            } else {
                if (!job->q_norm_data || !job->k_norm_data ||
                    job->q_norm_bytes < 256u * sizeof(uint16_t) ||
                    job->k_norm_bytes < 256u * sizeof(uint16_t) ||
                    job->q_job.rows != 4096 || job->k_job.rows != 512 ||
                    job->v_job.rows != 512 || job->o_job.rows != 1024) {
                    [enc endEncoding];
                    if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid resident full metadata");
                    return -1;
                }
                if (drafter_ensure_buffer(&g_attention_key_cache_buffers[layer],
                                          &g_attention_key_cache_bytes[layer],
                                          kv_cache_bytes, "attention key cache", err, errlen) != 0 ||
                    drafter_ensure_buffer(&g_attention_value_cache_buffers[layer],
                                          &g_attention_value_cache_bytes[layer],
                                          kv_cache_bytes, "attention value cache", err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                if (query_capture_by_layer && query_capture_by_layer[layer] &&
                    drafter_ensure_buffer(&g_query_capture_buffers[layer],
                                          &g_query_capture_bytes[layer],
                                          q_bytes, "query capture", err,
                                          errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                id<MTLBuffer> q_norm_buf = drafter_cached_buffer(job->q_norm_data, job->q_norm_bytes, err, errlen);
                id<MTLBuffer> k_norm_buf = drafter_cached_buffer(job->k_norm_data, job->k_norm_bytes, err, errlen);
                if (!q_norm_buf || !k_norm_buf) {
                    [enc endEncoding];
                    return -1;
                }
                ds4_drafter_metal_rope_args rope_args = { .position = position };
                ds4_drafter_metal_attention_args attn_args = {
                    .n_ctx = full_n_ctx,
                    .scale = 1.0f / 16.0f,
                };
                if (drafter_encode_rms_norm_dispatch(enc, job->input_norm_data, job->input_norm_bytes,
                                                     src, g_many_out_buffers[7],
                                                     1024, 1.0e-6f, err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7], 1024);
                const int capture_only_final_full =
                    !need_output && layer == last_full_layer &&
                    query_capture_by_layer && query_capture_by_layer[layer];
                if (capture_only_final_full) {
                    if (drafter_encode_affine_q_only_dispatch(enc, &job->q_job,
                                                              1024, bits, group_size,
                                                              g_many_out_buffers[7],
                                                              g_many_out_buffers[0],
                                                              err, errlen) != 0) {
                        [enc endEncoding];
                        return -1;
                    }
                    [enc setComputePipelineState:g_full_q_only_norm_rope_pipeline];
                    [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
                    [enc setBuffer:q_norm_buf offset:0 atIndex:1];
                    [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
                    [enc setBuffer:g_attention_q_buffer offset:0 atIndex:3];
                    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:g_attention_q_buffer
                            sourceOffset:0
                                toBuffer:g_query_capture_buffers[layer]
                       destinationOffset:0
                                    size:q_bytes];
                    [blit endEncoding];
                    enc = nil;
                    if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_full_qkv_ms,
                                                   "resident token full q-only", err, errlen) != 0) {
                        return -1;
                    }
                    prof_full_layers++;
                    break;
                }
                if (drafter_encode_affine_dispatch(enc, &job->q_job, 1024, bits, group_size,
                                                   g_many_out_buffers[7], g_many_out_buffers[0],
                                                   err, errlen) != 0 ||
                    drafter_encode_affine_dispatch(enc, &job->k_job, 1024, bits, group_size,
                                                   g_many_out_buffers[7], g_many_out_buffers[1],
                                                   err, errlen) != 0 ||
                    drafter_encode_affine_dispatch(enc, &job->v_job, 1024, bits, group_size,
                                                   g_many_out_buffers[7], g_many_out_buffers[2],
                                                   err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }

                [enc setComputePipelineState:g_full_q_norm_rope_pipeline];
                [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
                [enc setBuffer:q_norm_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[0] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:3];
                [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:4];
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:g_full_k_norm_rope_pipeline];
                [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
                [enc setBuffer:k_norm_buf offset:0 atIndex:1];
                [enc setBuffer:g_many_out_buffers[1] offset:0 atIndex:2];
                [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                if (query_capture_by_layer && query_capture_by_layer[layer]) {
                    [blit copyFromBuffer:g_many_out_buffers[3]
                            sourceOffset:0
                                toBuffer:g_query_capture_buffers[layer]
                       destinationOffset:0
                                    size:q_bytes];
                }
                [blit copyFromBuffer:g_many_out_buffers[5]
                        sourceOffset:0
                            toBuffer:g_attention_key_cache_buffers[layer]
                   destinationOffset:(NSUInteger)full_cache_index * kv_slice_bytes
                                size:kv_slice_bytes];
                [blit copyFromBuffer:g_many_out_buffers[2]
                        sourceOffset:0
                            toBuffer:g_attention_value_cache_buffers[layer]
                   destinationOffset:(NSUInteger)full_cache_index * kv_slice_bytes
                                size:kv_slice_bytes];
                [blit endEncoding];
                enc = nil;
                if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_full_qkv_ms,
                                               "resident token full qkv", err, errlen) != 0) {
                    return -1;
                }
                prof_full_layers++;
                if (!need_output && layer == last_full_layer &&
                    query_capture_by_layer && query_capture_by_layer[layer]) {
                    break;
                }

                if (!enc) enc = [cb computeCommandEncoder];
                if (drafter_logits4_enabled()) {
                    [enc setComputePipelineState:g_attention_logits4_pipeline];
                    [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                    [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:1];
                    [enc setBuffer:g_attention_key_cache_buffers[layer] offset:0 atIndex:2];
                    [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
                    [enc setThreadgroupMemoryLength:4u * 256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, ((NSUInteger)full_n_ctx + 3u) / 4u, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    [enc setComputePipelineState:g_attention_logits_pipeline];
                    [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                    [enc setBuffer:g_many_out_buffers[3] offset:0 atIndex:1];
                    [enc setBuffer:g_attention_key_cache_buffers[layer] offset:0 atIndex:2];
                    [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
                    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)full_n_ctx, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_full_logits_ms,
                                               "resident token full logits", err, errlen) != 0) {
                    return -1;
                }

                if (drafter_parallel_context_enabled()) {
                    [enc setComputePipelineState:g_attention_softmax_inplace_pipeline];
                    [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                    [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    if (drafter_context16_enabled()) {
                        [enc setComputePipelineState:g_attention_context16_pipeline];
                        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
                        [enc setBuffer:g_attention_value_cache_buffers[layer] offset:0 atIndex:3];
                        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                        [enc setThreadgroupMemoryLength:16u * 256u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(8, 16, 1)
                             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else if (drafter_context8_enabled()) {
                        [enc setComputePipelineState:g_attention_context8_pipeline];
                        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
                        [enc setBuffer:g_attention_value_cache_buffers[layer] offset:0 atIndex:3];
                        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                        [enc setThreadgroupMemoryLength:8u * 256u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(8, 32, 1)
                             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else if (drafter_context4_enabled()) {
                        [enc setComputePipelineState:g_attention_context4_pipeline];
                        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
                        [enc setBuffer:g_attention_value_cache_buffers[layer] offset:0 atIndex:3];
                        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                        [enc setThreadgroupMemoryLength:4u * 256u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(8, 64, 1)
                             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        [enc setComputePipelineState:g_attention_context_parallel_pipeline];
                        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                        [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
                        [enc setBuffer:g_attention_value_cache_buffers[layer] offset:0 atIndex:3];
                        [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(8, 256, 1)
                             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                } else {
                    [enc setComputePipelineState:g_attention_context_pipeline];
                    [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
                    [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
                    [enc setBuffer:g_many_out_buffers[4] offset:0 atIndex:2];
                    [enc setBuffer:g_attention_value_cache_buffers[layer] offset:0 atIndex:3];
                    [enc setBuffer:g_attention_out_buffer offset:0 atIndex:4];
                    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_full_context_ms,
                                               "resident token full context", err, errlen) != 0) {
                    return -1;
                }

                if (drafter_encode_affine_dispatch(enc, &job->o_job, 2048, bits, group_size,
                                                   g_attention_out_buffer, g_many_out_buffers[7],
                                                   err, errlen) != 0) {
                    [enc endEncoding];
                    return -1;
                }
                drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[7], 1024);
                drafter_encode_residual_add_round_dispatch(enc, src, g_many_out_buffers[7],
                                                           g_many_out_buffers[2], 1024);
                if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_full_o_ms,
                                               "resident token full output", err, errlen) != 0) {
                    return -1;
                }
            }

            if (drafter_encode_rms_norm_dispatch(enc, job->post_norm_data, job->post_norm_bytes,
                                                 g_many_out_buffers[2], g_many_out_buffers[0],
                                                 1024, 1.0e-6f, err, errlen) != 0) {
                [enc endEncoding];
                return -1;
            }
            drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);
            if (drafter_encode_affine_dispatch(enc, &job->mlp_gate_job, 1024, bits, group_size,
                                               g_many_out_buffers[0], g_many_out_buffers[5],
                                               err, errlen) != 0 ||
                drafter_encode_affine_dispatch(enc, &job->mlp_up_job, 1024, bits, group_size,
                                               g_many_out_buffers[0], g_many_out_buffers[7],
                                               err, errlen) != 0) {
                [enc endEncoding];
                return -1;
            }
            [enc setComputePipelineState:g_swiglu_pipeline];
            [enc setBuffer:g_many_out_buffers[5] offset:0 atIndex:0];
            [enc setBuffer:g_many_out_buffers[7] offset:0 atIndex:1];
            [enc setBuffer:g_linear_y_buffer offset:0 atIndex:2];
            [enc setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(3584, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            if (drafter_encode_affine_dispatch(enc, &job->mlp_down_job, 3584, bits, group_size,
                                               g_linear_y_buffer, g_many_out_buffers[0],
                                               err, errlen) != 0) {
                [enc endEncoding];
                return -1;
            }
            drafter_encode_round_bf16_dispatch(enc, g_many_out_buffers[0], 1024);
            drafter_encode_residual_add_round_dispatch(enc, g_many_out_buffers[2],
                                                       g_many_out_buffers[0],
                                                       dst, 1024);
            if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_mlp_ms,
                                           "resident token mlp", err, errlen) != 0) {
                return -1;
            }
            id<MTLBuffer> tmp = src;
            src = dst;
            dst = tmp;
        }

        if (need_output) {
            if (drafter_encode_rms_norm_dispatch(enc, final_norm_data, final_norm_bytes,
                                                 src, g_out_buffer,
                                                 1024, 1.0e-6f, err, errlen) != 0) {
                [enc endEncoding];
                return -1;
            }
            drafter_encode_round_bf16_dispatch(enc, g_out_buffer, 1024);
            if (drafter_profile_flush_when(token_profile, &cb, &enc, &prof_final_ms,
                                           "resident token final norm", err, errlen) != 0) {
                return -1;
            }
        }
        [enc endEncoding];
        [cb commit];
        if (!wait_for_completion) {
            return 0;
        }
        double wait_start = token_profile ? drafter_now_ms() : 0.0;
        [cb waitUntilCompleted];
        if (token_profile) prof_wait_ms += drafter_now_ms() - wait_start;
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter resident hidden-step command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        double copy_start = token_profile ? drafter_now_ms() : 0.0;
        if (query_capture_by_layer || key_out_by_layer || value_out_by_layer) {
            for (int layer = 0; layer < n_layers; layer++) {
                if (layers[layer].is_linear) continue;
                if (query_capture_by_layer && query_capture_by_layer[layer]) {
                    memcpy(query_capture_by_layer[layer],
                           [g_query_capture_buffers[layer] contents],
                           q_bytes);
                }
                if (key_out_by_layer && key_out_by_layer[layer]) {
                    memcpy(key_out_by_layer[layer],
                           (const uint8_t *)[g_attention_key_cache_buffers[layer] contents] +
                               (NSUInteger)full_cache_index * kv_slice_bytes,
                           kv_slice_bytes);
                }
                if (value_out_by_layer && value_out_by_layer[layer]) {
                    memcpy(value_out_by_layer[layer],
                           (const uint8_t *)[g_attention_value_cache_buffers[layer] contents] +
                               (NSUInteger)full_cache_index * kv_slice_bytes,
                           kv_slice_bytes);
                }
            }
        }
        if (need_output) {
            memcpy(out, [g_out_buffer contents], hidden_bytes);
        }
        if (token_profile) {
            prof_copy_ms += drafter_now_ms() - copy_start;
            const double total_ms = drafter_now_ms() - token_profile_start;
            fprintf(stderr,
                    "native drafter token profile: pos=%d need_output=%d wait=%d linear=%.3fms/%d full_qkv=%.3fms/%d full_logits=%.3fms full_context=%.3fms full_o=%.3fms mlp=%.3fms final=%.3fms final_wait=%.3fms copy=%.3fms total=%.3fms\n",
                    position, need_output, wait_for_completion,
                    prof_linear_ms, prof_linear_layers,
                    prof_full_qkv_ms, prof_full_layers,
                    prof_full_logits_ms, prof_full_context_ms, prof_full_o_ms,
                    prof_mlp_ms, prof_final_ms,
                    prof_wait_ms, prof_copy_ms, total_ms);
        }
        return 0;
    }
}

int ds4_drafter_metal_attention_importance(
        int cache_id,
        const float *queries_rope,
        int n_ctx,
        int pool_kernel,
        float *max_row,
        char *err,
        size_t errlen) {
    if (cache_id < 0 || cache_id >= 24 || !queries_rope || !max_row ||
        n_ctx <= 0 || pool_kernel <= 0 ||
        !g_attention_key_cache_buffers[cache_id]) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid attention importance shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger q_bytes = 2048u * sizeof(float);
        const NSUInteger row_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)n_ctx * 8u * sizeof(float);
        if (drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "attention query", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_importance_row_buffer, &g_importance_row_bytes,
                                  row_bytes, "attention importance row", err, errlen) != 0) {
            return -1;
        }
        memcpy([g_attention_q_buffer contents], queries_rope, q_bytes);
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };
        ds4_drafter_metal_importance_args imp_args = {
            .n_ctx = n_ctx,
            .pool_kernel = pool_kernel,
        };
        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_logits_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
        [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_ctx, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_importance_pipeline];
        [enc setBytes:&imp_args length:sizeof(imp_args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_importance_row_buffer offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter attention importance command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        const float *head_rows = (const float *)[g_importance_row_buffer contents];
        for (int qh = 0; qh < 8; qh++) {
            const float *row = head_rows + (NSUInteger)qh * (NSUInteger)n_ctx;
            for (int t = 0; t < n_ctx; t++) {
                if (row[t] > max_row[t]) max_row[t] = row[t];
            }
        }
        return 0;
    }
}

int ds4_drafter_metal_attention_importance_batch(
        int cache_id,
        const float *queries_rope,
        int n_queries,
        int n_ctx,
        int pool_kernel,
        float *max_rows,
        char *err,
        size_t errlen) {
    if (cache_id < 0 || cache_id >= 24 || !queries_rope || !max_rows ||
        n_queries <= 0 || n_ctx <= 0 || pool_kernel <= 0 ||
        !g_attention_key_cache_buffers[cache_id]) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid batched attention importance shape");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger q_bytes = (NSUInteger)n_queries * 2048u * sizeof(float);
        const NSUInteger row_bytes = (NSUInteger)n_queries * (NSUInteger)n_ctx * 8u * sizeof(float);
        const NSUInteger max_bytes = (NSUInteger)n_queries * (NSUInteger)n_ctx * sizeof(float);
        const NSUInteger logits_bytes = row_bytes;
        if (drafter_ensure_buffer(&g_attention_q_buffer, &g_attention_q_bytes,
                                  q_bytes, "batched attention query", err, errlen) != 0 ||
            drafter_ensure_private_buffer(&g_attention_logits_buffer, &g_attention_logits_bytes,
                                  logits_bytes, "batched attention logits", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_importance_row_buffer, &g_importance_row_bytes,
                                  row_bytes, "batched attention importance rows", err, errlen) != 0 ||
            drafter_ensure_buffer(&g_importance_max_buffer, &g_importance_max_bytes,
                                  max_bytes, "batched attention importance max rows", err, errlen) != 0) {
            return -1;
        }
        memcpy([g_attention_q_buffer contents], queries_rope, q_bytes);
        ds4_drafter_metal_attention_args attn_args = {
            .n_ctx = n_ctx,
            .scale = 1.0f / 16.0f,
        };
        ds4_drafter_metal_importance_args imp_args = {
            .n_ctx = n_ctx,
            .pool_kernel = pool_kernel,
        };
        id<MTLCommandBuffer> cb = [g_drafter_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (drafter_importance_logits4_enabled()) {
            [enc setComputePipelineState:g_attention_logits_batch4_pipeline];
            [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
            [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
            [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
            [enc setThreadgroupMemoryLength:4u * 256u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_queries, ((NSUInteger)n_ctx + 3u) / 4u)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc setComputePipelineState:g_attention_logits_batch_pipeline];
            [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
            [enc setBuffer:g_attention_q_buffer offset:0 atIndex:1];
            [enc setBuffer:g_attention_key_cache_buffers[cache_id] offset:0 atIndex:2];
            [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:3];
            [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_queries, (NSUInteger)n_ctx)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc setComputePipelineState:g_attention_importance_batch_pipeline];
        [enc setBytes:&imp_args length:sizeof(imp_args) atIndex:0];
        [enc setBuffer:g_attention_logits_buffer offset:0 atIndex:1];
        [enc setBuffer:g_importance_row_buffer offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)n_queries, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:g_attention_importance_reduce_heads_pipeline];
        [enc setBytes:&imp_args length:sizeof(imp_args) atIndex:0];
        [enc setBuffer:g_importance_row_buffer offset:0 atIndex:1];
        [enc setBuffer:g_importance_max_buffer offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_ctx, (NSUInteger)n_queries, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *msg = cb.error.localizedDescription ?: @"native Metal drafter batched attention importance command buffer failed";
            return drafter_metal_fail(err, errlen, msg);
        }
        const float *compact_rows = (const float *)[g_importance_max_buffer contents];
        for (int s = 0; s < n_queries; s++) {
            float *max_row = max_rows + (size_t)s * (size_t)n_ctx;
            const float *row = compact_rows + (size_t)s * (size_t)n_ctx;
            for (int t = 0; t < n_ctx; t++) {
                if (row[t] > max_row[t]) max_row[t] = row[t];
            }
        }
        return 0;
    }
}

int ds4_drafter_metal_copy_full_attention_cache(
        int cache_id,
        int n_ctx,
        float *keys_out,
        float *values_out,
        char *err,
        size_t errlen) {
    if (cache_id < 0 || cache_id >= 24 || n_ctx <= 0 ||
        !keys_out || !values_out ||
        !g_attention_key_cache_buffers[cache_id] ||
        !g_attention_value_cache_buffers[cache_id]) {
        if (err && errlen) snprintf(err, errlen, "native Metal drafter invalid full attention cache copy");
        return -1;
    }
    if (drafter_metal_init(err, errlen) != 0) return -1;
    @autoreleasepool {
        const NSUInteger bytes = (NSUInteger)n_ctx * 512u * sizeof(float);
        if (g_attention_key_cache_bytes[cache_id] < bytes ||
            g_attention_value_cache_bytes[cache_id] < bytes) {
            if (err && errlen) snprintf(err, errlen, "native Metal drafter full attention cache copy exceeds cache");
            return -1;
        }
        memcpy(keys_out, [g_attention_key_cache_buffers[cache_id] contents], bytes);
        memcpy(values_out, [g_attention_value_cache_buffers[cache_id] contents], bytes);
        return 0;
    }
}
