// Positive finite square root by restoring base-four digits. The normalized
// 24-bit significand is shifted into two uint32 words; no shaderInt64 needed.
// 27 root bits plus the exact remainder feed the existing single RNE pack.
uint math_sqrt(uint bits) {
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > CX_F32_INF) return bits | 0x00400000u;
    if (magnitude == 0u) return bits; // Preserve negative zero.
    if ((bits & CX_F32_SIGN) != 0u) return CX_F32_NAN;
    if (magnitude == CX_F32_INF) return bits;
    uint significand; int exponent;
    cx_f32_unpack(bits, significand, exponent);
    uint odd = uint(exponent & 1), shift = 29u + odd;
    uint high = significand >> (32u - shift), low = significand << shift;
    uint root = 0u, remainder = 0u;
    for (int digit = 26; digit >= 0; --digit) {
        uint pair = digit >= 16 ? high >> uint(2 * digit - 32) : low >> uint(2 * digit);
        remainder = (remainder << 2u) | (pair & 3u);
        root <<= 1u;
        uint trial = (root << 1u) | 1u;
        if (remainder >= trial) { remainder -= trial; root |= 1u; }
    }
    root |= uint(remainder != 0u);
    return cx_f32_round_pack(0u, (exponent - int(odd)) / 2, root);
}

// log(x) = e*ln(2) + 2*atanh((m-1)/(m+1)). Normalize m into
// [sqrt(1/2), sqrt(2)] by bits, including subnormal x, so no input is flushed
// and subtraction near x=1 retains its significant digits. An odd degree-13
// series is evaluated using the uint32 RNE helpers. Split ln(2) keeps the
// integer exponent's high product exact and avoids a large rounding residue.
uint math_log(uint bits) {
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > CX_F32_INF) return bits | 0x00400000u;
    if (magnitude == 0u) return CX_F32_SIGN | CX_F32_INF;
    if ((bits & CX_F32_SIGN) != 0u) return CX_F32_NAN;
    if (magnitude == CX_F32_INF) return bits;
    uint significand; int exponent;
    cx_f32_unpack(bits, significand, exponent);
    uint m = 0x3f800000u | (significand & 0x007fffffu);
    if (significand > 0x00b504f3u) { m -= 0x00800000u; ++exponent; }
    uint t = cx_f32_div_bits(cx_f32_add_bits(m, 0xbf800000u),
                             cx_f32_add_bits(m, 0x3f800000u));
    uint z = cx_f32_mul_bits(t, t);
    uint polynomial = 0x3d9d89d9u; // 1/13, then 1/11 ... 1/3, 1.
    polynomial = cx_f32_add_bits(0x3dba2e8cu, cx_f32_mul_bits(z, polynomial));
    polynomial = cx_f32_add_bits(0x3de38e39u, cx_f32_mul_bits(z, polynomial));
    polynomial = cx_f32_add_bits(0x3e124925u, cx_f32_mul_bits(z, polynomial));
    polynomial = cx_f32_add_bits(0x3e4ccccdu, cx_f32_mul_bits(z, polynomial));
    polynomial = cx_f32_add_bits(0x3eaaaaabu, cx_f32_mul_bits(z, polynomial));
    polynomial = cx_f32_add_bits(0x3f800000u, cx_f32_mul_bits(z, polynomial));
    uint log_m = cx_f32_mul_bits(0x40000000u, cx_f32_mul_bits(t, polynomial));
    uint e = floatBitsToUint(float(exponent)); // Small integer, exactly representable.
    uint high = cx_f32_mul_bits(e, 0x3f317200u);
    uint low = cx_f32_mul_bits(e, 0x35bfbe8eu);
    return cx_f32_add_bits(high, cx_f32_add_bits(log_m, low));
}

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
