// Binary32 round-to-nearest, ties-to-even arithmetic using core uint32 only.
// Keeping significands as integers avoids driver-dependent flushing of input
// and output subnormals. NaN arithmetic returns one canonical quiet NaN;
// copying, selection and negate have their separate bit-preserving contracts.

const uint CX_F32_SIGN = 0x80000000u;
const uint CX_F32_INF = 0x7f800000u;
const uint CX_F32_NAN = 0x7fc00000u;

uint cx_f32_shift_jam(uint value, uint distance) {
    if (distance == 0u) return value;
    if (distance >= 32u) return uint(value != 0u);
    return (value >> distance) | uint((value << (32u - distance)) != 0u);
}

// Nonzero finite values become a 24-bit significand with its leading bit set,
// and an unbiased exponent, including the original subnormal exponent range.
void cx_f32_unpack(uint bits, out uint significand, out int exponent) {
    uint field = (bits >> 23u) & 0xffu;
    significand = bits & 0x007fffffu;
    if (field != 0u) {
        significand |= 0x00800000u;
        exponent = int(field) - 127;
    } else {
        uint shift = uint(23 - findMSB(significand));
        significand <<= shift;
        exponent = -126 - int(shift);
    }
}

// significand has 24 significant bits plus guard/round/sticky bits. No rounding
// takes place before gradual-underflow shifting, preventing double rounding.
uint cx_f32_round_pack(uint sign, int exponent, uint significand) {
    if (exponent < -126) {
        significand = cx_f32_shift_jam(significand, uint(-126 - exponent));
        exponent = -126;
    }
    uint remainder = significand & 7u;
    uint mantissa = significand >> 3u;
    if (remainder > 4u || (remainder == 4u && (mantissa & 1u) != 0u)) ++mantissa;
    if (mantissa >= 0x01000000u) {
        mantissa >>= 1u;
        ++exponent;
    }
    if (exponent > 127) return sign | CX_F32_INF;
    uint field = mantissa >= 0x00800000u ? uint(exponent + 127) : 0u;
    return sign | (field << 23u) | (mantissa & 0x007fffffu);
}

uint cx_f32_add_bits(uint a, uint b) {
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if (am > CX_F32_INF || bm > CX_F32_INF) return CX_F32_NAN;
    if (am == CX_F32_INF || bm == CX_F32_INF) {
        if (am == CX_F32_INF && bm == CX_F32_INF && ((a ^ b) & CX_F32_SIGN) != 0u)
            return CX_F32_NAN;
        return am == CX_F32_INF ? a : b;
    }
    if (am == 0u && bm == 0u) return (a & b) & CX_F32_SIGN;
    if (am == 0u) return b;
    if (bm == 0u) return a;
    uint asig, bsig;
    int ae, be;
    cx_f32_unpack(a, asig, ae);
    cx_f32_unpack(b, bsig, be);
    // Sort by magnitude so subtraction is unsigned and the output sign comes
    // from the larger magnitude. The alignment difference is nonnegative.
    if (ae < be || (ae == be && asig < bsig)) {
        uint temp = a; a = b; b = temp;
        temp = asig; asig = bsig; bsig = temp;
        int exponent = ae; ae = be; be = exponent;
    }
    asig <<= 3u;
    bsig = cx_f32_shift_jam(bsig << 3u, uint(ae - be));
    uint result;
    if (((a ^ b) & CX_F32_SIGN) == 0u) {
        result = asig + bsig;
        if (result >= 0x08000000u) {
            result = cx_f32_shift_jam(result, 1u);
            ++ae;
        }
    } else {
        result = asig - bsig;
        if (result == 0u) return 0u;
        uint shift = uint(26 - findMSB(result));
        result <<= shift;
        ae -= int(shift);
    }
    return cx_f32_round_pack(a & CX_F32_SIGN, ae, result);
}

uint cx_f32_mul_bits(uint a, uint b) {
    uint sign = (a ^ b) & CX_F32_SIGN;
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if (am > CX_F32_INF || bm > CX_F32_INF) return CX_F32_NAN;
    if (am == CX_F32_INF || bm == CX_F32_INF)
        return am == 0u || bm == 0u ? CX_F32_NAN : sign | CX_F32_INF;
    if (am == 0u || bm == 0u) return sign;
    uint asig, bsig;
    int ae, be;
    cx_f32_unpack(a, asig, ae);
    cx_f32_unpack(b, bsig, be);
    uint high, low;
    umulExtended(asig, bsig, high, low);
    bool carry = (high & 0x00008000u) != 0u; // Product leading bit is 46 or 47.
    uint shift = carry ? 21u : 20u;
    uint result = (high << (32u - shift)) | (low >> shift);
    result |= uint((low & ((1u << shift) - 1u)) != 0u);
    return cx_f32_round_pack(sign, ae + be + int(carry), result);
}

uint cx_f32_div_bits(uint a, uint b) {
    uint sign = (a ^ b) & CX_F32_SIGN;
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if (am > CX_F32_INF || bm > CX_F32_INF ||
        (am == CX_F32_INF && bm == CX_F32_INF) || (am == 0u && bm == 0u)) return CX_F32_NAN;
    if (am == CX_F32_INF || bm == 0u) return sign | CX_F32_INF;
    if (am == 0u || bm == CX_F32_INF) return sign;
    uint asig, bsig;
    int ae, be;
    cx_f32_unpack(a, asig, ae);
    cx_f32_unpack(b, bsig, be);
    int exponent = ae - be;
    if (asig < bsig) { asig <<= 1u; --exponent; }
    uint remainder = asig - bsig;
    uint result = 0x04000000u;
    // Binary long division: the remainder stays below a 24-bit denominator;
    // doubling it fits uint32. Retain all GRS bits before any underflow shift.
    for (int bit = 25; bit >= 0; --bit) {
        remainder <<= 1u;
        if (remainder >= bsig) { remainder -= bsig; result |= 1u << uint(bit); }
    }
    result |= uint(remainder != 0u);
    return cx_f32_round_pack(sign, exponent, result);
}
