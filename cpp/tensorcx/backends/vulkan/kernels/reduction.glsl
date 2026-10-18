// Numeric reductions retain the CPU's row-major reduction order. One invocation
// owns an output; no atomics, host value readback or native float FTZ behavior.
uint reduction_offset(uint index, uint begin, uint end) {
    uint source = 0u;
    for (uint axis = end; axis > begin; --axis) {
        uint extent = metadata[2u * (axis - 1u)];
        source += (index % extent) * metadata[2u * (axis - 1u) + 1u];
        index /= extent;
    }
    return source;
}

bool reduction_nan(uint bits) {
    return p.integer_dtype == 0u && (bits & 0x7fffffffu) > CX_F32_INF;
}

bool reduction_less(uint a, uint b) {
    if (p.integer_dtype != 0u) return int(a) < int(b);
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) return false;
    uint ak = a ^ ((a & CX_F32_SIGN) != 0u ? 0xffffffffu : CX_F32_SIGN);
    uint bk = b ^ ((b & CX_F32_SIGN) != 0u ? 0xffffffffu : CX_F32_SIGN);
    return ak < bk;
}

void numeric_reduction(uint index) {
    if (p.operation == 45u) { output_data[index] = lhs[index]; return; }
    if (p.scalar_bits == 0u) {
        // max/min/argmax reject empty reduction spaces before dispatch.
        output_data[index] = p.operation == 42u ? CX_F32_NAN : 0u;
        return;
    }
    uint base = reduction_offset(index, 0u, p.scalar_left);
    bool sum = p.operation == 40u || p.operation == 42u;
    uint value = sum ? 0u : lhs[base];
    uint best_index = 0u;
    for (uint j = sum ? 0u : 1u; j < p.scalar_bits; ++j) {
        if (!sum && reduction_nan(value)) break;
        uint item = lhs[base + reduction_offset(j, p.scalar_left, p.rank)];
        if (sum) {
            value = p.integer_dtype != 0u ? value + item : cx_f32_add_bits(value, item);
        } else if (reduction_nan(item) || (p.operation == 43u
                   ? reduction_less(item, value) : reduction_less(value, item))) {
            value = item;
            best_index = j;
        }
    }
    if (p.operation == 42u) value = cx_f32_div_bits(value, p.reserved);
    output_data[index] = p.operation == 44u ? best_index : value;
}
