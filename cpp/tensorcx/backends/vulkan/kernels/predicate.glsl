// Included by the validated elementwise compute shader. Operation codes match
// vulkan_predicate.cpp; metadata/indices are checked uint32 element offsets.
uint right_value(uint index, uint dtype) {
    if (dtype == 2u) return (rhs[index / 4u] >> (8u * (index % 4u))) & 0xffu;
    return rhs[index];
}

uvec2 predicate_offsets(uint index) {
    uint a = 0u, b = 0u, rest = index;
    for (uint axis = p.rank; axis > 0u; --axis) {
        uint coordinate = rest % metadata[axis];
        rest /= metadata[axis];
        a += coordinate * metadata[p.rank + axis];
        b += coordinate * metadata[2u * p.rank + axis];
    }
    return uvec2(a, b);
}

uint compare_bits(uint a, uint b) {
    bool equal, less, greater;
    if (p.integer_dtype == 0u) {
        uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
        bool ordered = am <= 0x7f800000u && bm <= 0x7f800000u;
        bool zeros = am == 0u && bm == 0u;
        uint ak = a ^ ((a & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u);
        uint bk = b ^ ((b & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u);
        equal = ordered && (a == b || zeros);
        less = ordered && !zeros && ak < bk;
        greater = ordered && !zeros && ak > bk;
    } else {
        if (p.integer_dtype == 2u) { a = uint(a != 0u); b = uint(b != 0u); }
        equal = a == b;
        less = int(a) < int(b);
        greater = int(a) > int(b);
    }
    if (p.operation == 20u) return uint(equal);
    if (p.operation == 21u) return uint(!equal);
    if (p.operation == 22u) return uint(less);
    if (p.operation == 23u) return uint(less || equal);
    if (p.operation == 24u) return uint(greater);
    return uint(greater || equal);
}

uint predicate_element(uint index) {
    if (p.operation == 32u || p.operation == 33u) {
        bool value = p.operation == 33u;
        for (uint j = 0u; j < p.scalar_bits; ++j) {
            uint rest = index * p.scalar_bits + j, source = 0u;
            for (uint axis = p.rank; axis > 0u; --axis) {
                uint extent = metadata[1u + 2u * (axis - 1u)];
                source += (rest % extent) * metadata[2u + 2u * (axis - 1u)];
                rest /= extent;
            }
            bool present = read_source(source) != 0u;
            value = p.operation == 33u ? value && present : value || present;
            if (value != (p.operation == 33u)) break;
        }
        return uint(value);
    }
    if (p.operation == 36u) {
        // Invert the inclusive mask scan. The first prefix >= selected row+1
        // identifies its source row; this preserves stable input order and
        // gives packed bool output exactly one writer per word.
        uint target = index / p.scalar_left + 1u;
        uint low = 0u, high = p.scalar_bits;
        while (low < high) {
            uint middle = low + (high - low) / 2u;
            if (rhs[middle] < target) low = middle + 1u;
            else high = middle;
        }
        return read_source(low * p.scalar_left + index % p.scalar_left);
    }
    uvec2 offsets = predicate_offsets(index);
    if (p.operation == 30u || p.operation == 31u) {
        bool condition = ((lhs[offsets.x / 4u] >> (8u * (offsets.x % 4u))) & 0xffu) != 0u;
        if (p.operation == 30u)
            return condition ? 0u : right_value(offsets.y, p.integer_dtype);
        return right_value(offsets.y, p.integer_dtype); // Called only for true lanes.
    }
    uint a = read_source(offsets.x);
    if (p.operation == 29u) return uint(a == 0u);
    uint b = right_value(offsets.y, p.integer_dtype);
    if (p.operation <= 25u) return compare_bits(a, b);
    bool left = a != 0u, right = b != 0u;
    if (p.operation == 26u) return uint(left && right);
    if (p.operation == 27u) return uint(left || right);
    return uint(left != right);
}

bool replace_where_lane(uint index) {
    if (p.operation != 31u) return true;
    uint source = predicate_offsets(index).x;
    return ((lhs[source / 4u] >> (8u * (source % 4u))) & 0xffu) != 0u;
}

void predicate(uint word) {
    if (p.operation == 34u) { output_data[word] = uint(read_source(word) != 0u); return; }
    if (p.operation == 35u) {
        uint value = lhs[word];
        if (word >= p.scalar_bits) value += lhs[word - p.scalar_bits];
        output_data[word] = value;
        return;
    }
    if (p.reserved != 2u) {
        if (replace_where_lane(word)) output_data[word] = predicate_element(word);
    } else {
        uint packed = p.operation == 31u ? output_data[word] : 0u;
        for (uint lane = 0u; lane < 4u; ++lane) {
            uint index = word * 4u + lane;
            if (index < metadata[0] && replace_where_lane(index)) {
                uint shift = 8u * lane;
                packed = (packed & ~(0xffu << shift)) | (predicate_element(index) << shift);
            }
        }
        output_data[word] = packed;
    }
}
