#include "math_functions.glsl"

uvec2 math_clip_offsets(uint index) {
    uint left = p.operation == 64u ? index : 0u, right = 0u;
    uint argument = p.operation == 63u ? 2u : 3u;
    for (uint axis = p.rank; axis > 0u; --axis) {
        uint coordinate = index % metadata[axis - 1u];
        index /= metadata[axis - 1u];
        if (p.operation == 63u) left += coordinate * metadata[p.rank + axis - 1u];
        right += coordinate * metadata[argument * p.rank + axis - 1u];
    }
    return uvec2(left, right);
}

bool math_before(uint a, uint ai, uint b, uint bi, bool largest) {
    bool an = reduction_nan(a), bn = reduction_nan(b);
    if (an != bn) return largest ? an : bn;
    if (!an && (reduction_less(a, b) || reduction_less(b, a)))
        return largest ? reduction_less(b, a) : reduction_less(a, b);
    return ai < bi;
}

void math_topk(uint group) {
    uint reduce = p.scalar_bits, inner = p.scalar_left, k = p.rank;
    uint base = (group / inner) * reduce * inner + group % inner;
    uint dest = (group / inner) * k * inner + group % inner;
    bool largest = (p.reserved & 1u) != 0u, sorted = (p.reserved & 2u) != 0u;
    uint previous = 0u, previous_index = 0xffffffffu;
    for (uint slot = 0u; slot < k; ++slot) {
        uint best = 0u, best_index = 0xffffffffu;
        for (uint j = 0u; j < reduce; ++j) {
            uint value = lhs[base + j * inner];
            if (slot != 0u && !math_before(previous, previous_index, value, j, largest)) continue;
            if (best_index == 0xffffffffu || math_before(value, j, best, best_index, largest)) {
                best = value; best_index = j;
            }
        }
        previous = best; previous_index = best_index;
        uint position = slot;
        if (!sorted) {
            while (position > 0u && rhs[dest + (position - 1u) * inner] > best_index) {
                rhs[dest + position * inner] = rhs[dest + (position - 1u) * inner];
                --position;
            }
        }
        rhs[dest + position * inner] = best_index;
    }
    for (uint slot = 0u; slot < k; ++slot)
        output_data[dest + slot * inner] = lhs[base + rhs[dest + slot * inner] * inner];
}

void math_operation(uint index) {
    if (p.operation == 65u) { math_topk(index); return; }
    if (p.operation == 63u || p.operation == 64u) {
        uvec2 offsets = math_clip_offsets(index);
        uint value = lhs[offsets.x], bound = rhs[offsets.y];
        if (!reduction_nan(value) && (reduction_nan(bound) || (p.operation == 63u
                ? !reduction_less(bound, value) : !reduction_less(value, bound)))) value = bound;
        output_data[index] = value;
        return;
    }
    uint value = lhs[index];
    if (p.operation == 60u) value = math_log(value);
    else if (p.operation == 61u) value = math_sqrt(value);
    else value = p.integer_dtype == 0u ? value & 0x7fffffffu
                    : (value & CX_F32_SIGN) != 0u ? 0u - value : value;
    output_data[index] = value;
}
