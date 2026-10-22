// Float32 neural-network primitives use the same uint32 RNE arithmetic as
// ordinary operations. No optional float controls or native transcendental FTZ.
const uint NN_ONE = 0x3f800000u;
bool nn_nan(uint value) { return (value & 0x7fffffffu) > CX_F32_INF; }
bool nn_less(uint a, uint b) {
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) return false;
    uint ak = a ^ ((a & CX_F32_SIGN) != 0u ? 0xffffffffu : CX_F32_SIGN);
    uint bk = b ^ ((b & CX_F32_SIGN) != 0u ? 0xffffffffu : CX_F32_SIGN);
    return ak < bk;
}
bool nn_equal(uint a, uint b) {
    return !nn_nan(a) && !nn_nan(b) &&
           (a == b || ((a | b) & 0x7fffffffu) == 0u);
}
uint nn_sub(uint a, uint b) { return cx_f32_add_bits(a, b ^ CX_F32_SIGN); }

// x = n*ln(2) + r, |r| <= ln(2)/2. A split ln(2) keeps range reduction
// accurate over the full finite output range. The degree-ten Taylor polynomial
// is evaluated with explicit separate RNE operations, then scaled by exponent
// bits with gradual underflow. This is a bounded-error exp, not a correctly
// rounded transcendental. The only native float work selects a small integer n.
uint nn_exp(uint bits) {
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > CX_F32_INF) return bits | 0x00400000u;
    if (magnitude >= 0x42d00000u) return (bits & CX_F32_SIGN) != 0u ? 0u : CX_F32_INF;
    if (magnitude < 0x33000000u) return NN_ONE;
    int n = int(roundEven(uintBitsToFloat(cx_f32_mul_bits(bits, 0x3fb8aa3bu))));
    uint nb = floatBitsToUint(float(n));
    uint r = nn_sub(nn_sub(bits, cx_f32_mul_bits(nb, 0x3f317200u)),
                     cx_f32_mul_bits(nb, 0x35bfbe8eu));
    uint polynomial = 0x3493f27eu; // 1/10!, ..., 1/2!.
    polynomial = cx_f32_add_bits(0x3638ef1du, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x37d00d01u, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x39500d01u, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x3ab60b61u, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x3c088889u, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x3d2aaaabu, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x3e2aaaabu, cx_f32_mul_bits(r, polynomial));
    polynomial = cx_f32_add_bits(0x3f000000u, cx_f32_mul_bits(r, polynomial));
    uint value = cx_f32_add_bits(NN_ONE, cx_f32_mul_bits(r,
        cx_f32_add_bits(NN_ONE, cx_f32_mul_bits(r, polynomial))));
    uint significand; int exponent;
    cx_f32_unpack(value, significand, exponent);
    return cx_f32_round_pack(0u, exponent + n, significand << 3u);
}

uint nn_tanh(uint bits) {
    uint magnitude = bits & 0x7fffffffu, sign = bits & CX_F32_SIGN;
    if (magnitude > CX_F32_INF) return bits | 0x00400000u;
    if (magnitude >= 0x41200000u) return sign | NN_ONE;
    if (magnitude < 0x39800000u) return bits; // |x| < 2^-12.
    uint value;
    if (magnitude < 0x3e000000u) {
        uint square = cx_f32_mul_bits(magnitude, magnitude);
        uint poly = 0x3cb327a4u; // 62/2835, -17/315, 2/15, -1/3.
        poly = cx_f32_add_bits(0xbd5d0dd1u, cx_f32_mul_bits(square, poly));
        poly = cx_f32_add_bits(0x3e088889u, cx_f32_mul_bits(square, poly));
        poly = cx_f32_add_bits(0xbeaaaaabu, cx_f32_mul_bits(square, poly));
        value = cx_f32_mul_bits(magnitude,
            cx_f32_add_bits(NN_ONE, cx_f32_mul_bits(square, poly)));
    } else {
        uint q = nn_exp(cx_f32_mul_bits(0xc0000000u, magnitude));
        value = nn_sub(NN_ONE, cx_f32_div_bits(cx_f32_mul_bits(0x40000000u, q),
                                             cx_f32_add_bits(NN_ONE, q)));
    }
    return value | sign;
}

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
