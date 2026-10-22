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
