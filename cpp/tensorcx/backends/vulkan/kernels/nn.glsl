#include "nn_functions.glsl"

void nn_unary(uint index) {
    uint x = lhs[index], value;
    if (p.operation == 70u) value = nn_exp(x);
    else if (p.operation == 72u)
        value = cx_f32_div_bits(x, cx_f32_add_bits(NN_ONE, nn_exp(x ^ CX_F32_SIGN)));
    else {
        // Preserve the ordinary CPU GELU tanh approximation and evaluation
        // order, including the existing NaN result at negative infinity.
        uint cubic = cx_f32_mul_bits(cx_f32_mul_bits(cx_f32_mul_bits(0x3d372713u, x), x), x);
        uint inner = cx_f32_mul_bits(0x3f4c422au, cx_f32_add_bits(x, cubic));
        value = cx_f32_mul_bits(cx_f32_mul_bits(0x3f000000u, x),
                               cx_f32_add_bits(NN_ONE, nn_tanh(inner)));
    }
    output_data[index] = value;
}

void nn_axis(uint group) {
    uint reduce = p.scalar_bits, stride = p.scalar_left;
    uint base = (group / stride) * reduce * stride + group % stride;
    if (p.operation == 73u) {
        uint maximum = CX_F32_SIGN | CX_F32_INF;
        for (uint j = 0u; j < reduce; ++j) {
            uint value = lhs[base + j * stride];
            if (nn_nan(value)) { maximum = value; break; }
            if (nn_less(maximum, value)) maximum = value;
        }
        uint total = 0u;
        for (uint j = 0u; j < reduce; ++j) {
            uint index = base + j * stride;
            uint value = nn_exp(nn_sub(lhs[index], maximum));
            output_data[index] = value;
            total = cx_f32_add_bits(total, value);
        }
        for (uint j = 0u; j < reduce; ++j) {
            uint index = base + j * stride;
            output_data[index] = cx_f32_div_bits(output_data[index], total);
        }
        return;
    }
    bool layer = p.operation == 75u, equal = true;
    uint first = lhs[base], total = 0u;
    for (uint j = 0u; j < reduce; ++j) {
        uint value = lhs[base + j * stride];
        equal = equal && nn_equal(value, first);
        total = cx_f32_add_bits(total, layer ? value : cx_f32_mul_bits(value, value));
    }
    if (layer && equal && (first & 0x7fffffffu) < CX_F32_INF) {
        uint value = (p.reserved & 0x7fffffffu) == 0u ? CX_F32_NAN : 0u;
        for (uint j = 0u; j < reduce; ++j) output_data[base + j * stride] = value;
        return;
    }
    uint mean = cx_f32_div_bits(total, metadata[0]);
    if (layer) {
        total = 0u;
        for (uint j = 0u; j < reduce; ++j) {
            uint difference = nn_sub(lhs[base + j * stride], mean);
            total = cx_f32_add_bits(total, cx_f32_mul_bits(difference, difference));
        }
    }
    uint denominator = cx_f32_add_bits(layer ? cx_f32_div_bits(total, metadata[0]) : mean, p.reserved);
    uint scale = cx_f32_div_bits(NN_ONE, math_sqrt(denominator));
    for (uint j = 0u; j < reduce; ++j) {
        uint index = base + j * stride;
        uint value = layer ? nn_sub(lhs[index], mean) : lhs[index];
        output_data[index] = layer && (denominator & 0x7fffffffu) == 0u
            ? CX_F32_NAN : cx_f32_mul_bits(value, scale);
    }
}

uint nn_mapped(uint index, uint argument) {
    uint offset = 0u;
    for (uint axis = p.rank; axis > 0u; --axis) {
        offset += (index % metadata[axis - 1u]) * metadata[(argument + 1u) * p.rank + axis - 1u];
        index /= metadata[axis - 1u];
    }
    return offset;
}
uint nn_score(uint index, uint query, uint column) {
    uint kind = p.integer_dtype & 3u;
    if ((p.integer_dtype & 4u) != 0u && column > query) return CX_F32_SIGN | CX_F32_INF;
    uint offset = kind == 0u ? 0u : nn_mapped(index, 1u);
    if (kind == 1u && ((rhs[offset / 4u] >> (8u * (offset % 4u))) & 0xffu) == 0u)
        return CX_F32_SIGN | CX_F32_INF;
    uint bias = kind == 2u ? rhs[offset] : 0u;
    if (bias == (CX_F32_SIGN | CX_F32_INF)) return bias;
    return cx_f32_add_bits(cx_f32_mul_bits(lhs[nn_mapped(index, 0u)], p.scalar_left), bias);
}
void nn_attention(uint row) {
    uint columns = p.scalar_bits, base = row * columns;
    uint query = row % metadata[p.rank - 2u];
    uint maximum = CX_F32_SIGN | CX_F32_INF;
    for (uint col = 0u; col < columns; ++col) {
        uint value = nn_score(base + col, query, col);
        if (nn_nan(value)) { maximum = value; break; }
        if (nn_less(maximum, value)) maximum = value;
    }
    if (maximum == (CX_F32_SIGN | CX_F32_INF)) {
        for (uint col = 0u; col < columns; ++col) output_data[base + col] = 0u;
        return;
    }
    uint total = 0u;
    for (uint col = 0u; col < columns; ++col) {
        uint value = nn_exp(nn_sub(nn_score(base + col, query, col), maximum));
        output_data[base + col] = value;
        total = cx_f32_add_bits(total, value);
    }
    for (uint col = 0u; col < columns; ++col)
        output_data[base + col] = cx_f32_div_bits(output_data[base + col], total);
}

void nn_operation(uint index) {
    if (p.operation <= 72u) nn_unary(index);
    else if (p.operation <= 75u) nn_axis(index);
    else if (p.operation == 76u) {
        uint value = lhs[index];
        if ((value & CX_F32_SIGN) != 0u || value >= p.scalar_bits) atomicOr(rhs[0], 1u);
    } else if (p.operation == 77u) {
        uint width = p.scalar_bits;
        output_data[index] = rhs[lhs[index / width] * width + index % width];
    } else nn_attention(index);
}
