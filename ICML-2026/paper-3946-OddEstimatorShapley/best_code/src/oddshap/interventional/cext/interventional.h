#ifndef INTERVENTIONAL_TREE_SHAP_ITERATIVE_H
#define INTERVENTIONAL_TREE_SHAP_ITERATIVE_H


#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <math.h>
#include <vector>

typedef double tfloat;

struct Tree {
    tfloat *values;
    tfloat *thresholds;
    int *features;
    int *children_left;
    int *children_right;

    Tree(
        tfloat *values,
        tfloat *thresholds,
        int *features, int *children_left, int *children_right
    ):
        values(values),
        thresholds(thresholds),
        features(features),
        children_left(children_left),
        children_right(children_right){};

    bool is_internal(int pos)const {
        return children_left[pos] != children_right[pos] ;
}};

#if defined(_WIN32) || defined(WIN32)
#include <malloc.h>
#elif defined(__MVS__)
#include <stdlib.h>
#else
#include <alloca.h>
#endif

// Debug Information
#define DEBUG_PRINT 0

// Bit array utilities for unlimited features
#define BITS_PER_WORD 32
#define WORD_INDEX(bit) ((bit) / BITS_PER_WORD)
#define BIT_OFFSET(bit) ((bit) % BITS_PER_WORD)
#define NUM_WORDS(n_features) (((n_features) + BITS_PER_WORD - 1) / BITS_PER_WORD)

// BitSet: Represents a set of features as a bit array
class BitSet {
public:
    std::vector<unsigned int> words;
    int num_words;
    int n_features;
    
    BitSet() : num_words(0), n_features(0) {}

    BitSet(int n_feat) : num_words(NUM_WORDS(n_feat)), n_features(n_feat) {
        words.resize(num_words, 0);
    }
    
    // Default copy constructor, assignment operator, and destructor handle std::vector correctly
    
    void set(int feature) {
        if (feature < n_features) {
            words[WORD_INDEX(feature)] |= (1u << BIT_OFFSET(feature));
        }
    }
    
    void clear(int feature) {
        if (feature < n_features) {
            words[WORD_INDEX(feature)] &= ~(1u << BIT_OFFSET(feature));
        }
    }
    
    bool test(int feature) const {
        if (feature >= n_features) return false;
        return (words[WORD_INDEX(feature)] & (1u << BIT_OFFSET(feature))) != 0;
    }
    
    void set_all_zero() {
        std::fill(words.begin(), words.end(), 0);
    }
    
    void set_union(const BitSet& a, const BitSet& b) {
        for (int i = 0; i < num_words; i++) {
            words[i] = a.words[i] | b.words[i];
        }
    }
    
    void set_intersection(const BitSet& a, const BitSet& b) {
        for (int i = 0; i < num_words; i++) {
            words[i] = a.words[i] & b.words[i];
        }
    }
    
    void set_difference(const BitSet& a, const BitSet& b) {
        for (int i = 0; i < num_words; i++) {
            words[i] = a.words[i] & ~b.words[i];
        }
    }
    
    int popcount() const {
        int count = 0;
        for (int i = 0; i < num_words; i++) {
            count += __builtin_popcount(words[i]);
        }
        return count;
    }
    
    bool is_empty() const {
        for (int i = 0; i < num_words; i++) {
            if (words[i] != 0) return false;
        }
        return true;
    }
    
    bool equals(const BitSet& other) const {
        if (num_words != other.num_words) return false;
        for (int i = 0; i < num_words; i++) {
            if (words[i] != other.words[i]) return false;
        }
        return true;
    }
    
    // Hash function for bit arrays
    unsigned int hash() const {
        unsigned int h = 0;
        for (int i = 0; i < num_words; i++) {
            h ^= words[i] * 0x9e3779b9;
            h = (h << 13) | (h >> 19);  // Rotate left 13 bits
        }
        // Apply MurmurHash3 finalizer
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
    
    void print_debug() const {
        printf("[");
        bool first = true;
        for (int i = 0; i < n_features; i++) {
            if (test(i)) {
                if (!first) printf(",");
                printf("%d", i);
                first = false;
            }
        }
        printf("]");
    }
};

// Stack frame structure for iterative traversal (now using BitSets)
struct StackFrame
{
    int node;
    BitSet A;
    BitSet B;

    StackFrame(int n, const BitSet& a, const BitSet& b) 
        : node(n), A(a), B(b) { }
};

// Simple hash map for sparse subset storage using BitSets
struct SubsetEntry {
    BitSet* key;       // BitSet representing the subset
    tfloat value;      // contribution
    bool occupied;     // slot occupied flag
    
    SubsetEntry() : key(nullptr), value(0.0), occupied(false) {}
    
    ~SubsetEntry() {
        if (key != nullptr) {
            delete key;
        }
    }
};

class SparseSubsetMap {
private:
    SubsetEntry* table;
    int capacity;
    int size;
    int n_features;
    
public:
    SparseSubsetMap(int n_feat, int initial_capacity = 256) 
        : capacity(initial_capacity), size(0), n_features(n_feat) {
        table = new SubsetEntry[capacity];
    }
    
    ~SparseSubsetMap() {
        delete[] table;
    }
    
    // Add or update value
    void add(const BitSet& key, tfloat value) {
        unsigned int h = key.hash();
        int idx = h % capacity;
        int probe = 0;
        
        // Linear probing
        while (probe < capacity) {
            if (!table[idx].occupied) {
                // Empty slot, insert new
                table[idx].key = new BitSet(key);
                table[idx].value = value;
                table[idx].occupied = true;
                size++;
                return;
            } else if (table[idx].key->equals(key)) {
                // Key exists, update
                table[idx].value += value;
                return;
            }
            // Collision, probe next
            idx = (idx + 1) % capacity;
            probe++;
        }
        
        // Table full (should resize, but for now just warn)
        if (DEBUG_PRINT) {
            printf("WARNING: Hash table full!\n");
        }
    }
    
    // Convert BitSet to linear index for values array
    // This assumes values array uses a compact representation
    // For now, we'll use a simple mapping based on features present
    int bitset_to_index(const BitSet& bs) const {
        // Simple encoding: treat bitset as binary number
        // This only works if n_features is small enough
        // For larger n_features, Python should pass a mapping
        int index = 0;
        for (int i = 0; i < bs.n_features && i < 32; i++) {
            if (bs.test(i)) {
                index |= (1 << i);
            }
        }
        return index;
    }
    
    // Write back to values array
    void write_back(tfloat* values) const {
        for (int i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                int idx = bitset_to_index(*table[i].key);
                values[idx] += table[i].value;
            }
        }
    }
    
    // Extract all entries for Python
    // Returns parallel arrays of keys (as feature lists) and values
    void extract_entries(int** feature_lists_out, int* list_lengths_out, 
                        tfloat** values_out, int& num_entries_out) const {
        // Count actual entries
        num_entries_out = size;
        
        if (size == 0) {
            *feature_lists_out = nullptr;
            *values_out = nullptr;
            return;
        }
        
        // Allocate output arrays
        *values_out = new tfloat[size];
        
        // First pass: compute total features needed
        int total_features = 0;
        int entry_idx = 0;
        for (int i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                list_lengths_out[entry_idx] = table[i].key->popcount();
                total_features += list_lengths_out[entry_idx];
                entry_idx++;
            }
        }
        
        // Allocate flat feature list
        *feature_lists_out = new int[total_features];
        
        // Second pass: fill arrays
        entry_idx = 0;
        int feature_idx = 0;
        for (int i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                (*values_out)[entry_idx] = table[i].value;
                
                // Extract feature indices from BitSet
                for (int f = 0; f < table[i].key->n_features; f++) {
                    if (table[i].key->test(f)) {
                        (*feature_lists_out)[feature_idx++] = f;
                    }
                }
                
                entry_idx++;
            }
        }
    }
    
    // Get current size
    int get_size() const { return size; }
};

// Old powerset function for compatibility with int bitmasks
inline int *create_powerset(int n, int *valid_pos)
{
    /** Create a powerset of feature sets, only using the features in the valid_pos array.
        n: number of valid positions
        valid_pos: array of bitmasks for each valid feature position
        Returns: array of bitmasks representing all subsets
    */
    int size = 1 << n; // Number of subsets
    int *powerset = new int[size];
    for (int i = 0; i < size; i++) {
        int subset = 0;
        for (int j = 0; j < n; j++) {
            if (i & (1 << j)) {
                subset |= valid_pos[j];
            }
        }
        powerset[i] = subset;
    }
    return powerset;
}




inline int binomial_coefficient(int n, int k) {
    int* c = new int[k + 1];
    memset(c, 0, (k + 1) * sizeof(int));
    c[0] = 1; // nC0 is 1
    for (int i = 1; i <= n; i++) {
        for (int j = std::min(i, k); j > 0; j--) {
            c[j] = c[j] + c[j - 1];
        }
    }
    int result = c[k];
    delete[] c;
    return result;
}

// ============================================================================
// DIRECT ARRAY INDEXING FUNCTIONS (No Hash Map)
// ============================================================================

// Compute total array size needed for given max_order
inline int compute_values_array_size(int n_features, int max_order) {
    /**
     * Computes the total size needed for the values array.
     * Size = sum_{k=1}^{max_order} C(n_features, k)
     */
    int total = 0;
    int effective_max_order = (max_order < 0 || max_order > n_features) ? n_features : max_order;
    for (int k = 1; k <= effective_max_order; k++) {
        total += binomial_coefficient(n_features, k);
    }
    return total;
}

// Compute the starting index for subsets of size k
inline int subset_start_index(int n_features, int k) {
    /**
     * Returns the starting index in the values array for subsets of size k.
     * 
     * Layout:
     * - Size 1: indices [0, n)
     * - Size 2: indices [n, n + C(n,2))
     * - Size 3: indices [n + C(n,2), n + C(n,2) + C(n,3))
     */
    int start = 0;
    for (int i = 1; i < k; i++) {
        start += binomial_coefficient(n_features, i);
    }
    return start;
}

// Compute the lexicographic rank of a subset
inline int subset_rank(int* feature_list, int subset_size) {
    /**
     * Computes the lexicographic rank using combinatorial number system.
     * For subset {a, b, c} where a < b < c, rank = C(a,1) + C(b,2) + C(c,3)
     * 
     * feature_list: sorted array of feature indices
     * subset_size: number of features in subset
     */
    int rank = 0;
    for (int i = 0; i < subset_size; i++) {
        rank += binomial_coefficient(feature_list[i], i + 1);
    }
    return rank;
}

// Map BitSet to array index
inline int bitset_to_array_index(const BitSet& subset, int n_features) {
    /**
     * Maps a BitSet subset to its unique index in the values array.
     * 
     * Returns:
     * - Index in range [0, array_size)
     * - -1 if subset is empty
     * 
     * Example for n_features=4:
     * {0}    -> 0
     * {1}    -> 1  
     * {2}    -> 2
     * {3}    -> 3
     * {0,1}  -> 4
     * {0,2}  -> 5
     * etc.
     */
    
    // Extract features from BitSet
    int subset_size = subset.popcount();
    if (subset_size == 0) return -1;
    
    int* feature_list = new int[subset_size];
    int idx = 0;
    for (int i = 0; i < n_features; i++) {
        if (subset.test(i)) {
            feature_list[idx++] = i;
        }
    }
    
    // Compute index
    int start = subset_start_index(n_features, subset_size);
    int rank = subset_rank(feature_list, subset_size);
    
    delete[] feature_list;
    return start + rank;
}

// Version using direct array indexing with BitSets
inline void update_values_direct(const BitSet& A, const BitSet& B, const BitSet& N, tfloat const_value, 
                                tfloat* values, int n_features, int max_order) {
    /**
     * Updates values array directly using combinatorial indexing.
     * Much faster than hash map approach - O(1) array access vs O(k) hashing + probing.
     * 
     * Parameters:
     * - A, B, N: BitSet subsets for SHAP computation
     * - const_value: Weight to multiply with computed weight
     * - values: Pre-allocated array sized by compute_values_array_size(n_features, max_order)
     * - n_features: Total number of features
     * - max_order: Maximum subset size to consider
     */
    
    // Compute difference NB = N \ B
    BitSet NB(N.n_features);
    NB.set_difference(N, B);

    // Compute union: present_features = A | NB
    BitSet present_features(A.n_features);
    present_features.set_union(A, NB);
    
    // Skip if union is empty
    int len_union = present_features.popcount();
    if (len_union == 0) {
        return;  // No features to process
    }
    
    // Extract feature indices from union
    int* feature_list = new int[len_union];
    int idx = 0;
    for (int i = 0; i < present_features.n_features; i++) {
        if (present_features.test(i)) {
            feature_list[idx++] = i;
        }
    }
    
    // Pre-allocate reusable BitSets to avoid allocation in loop
    BitSet subset(present_features.n_features);
    BitSet intersection(A.n_features);
    BitSet B_union_subset(N.n_features);
    BitSet B_and_subset(N.n_features);
    BitSet N_set_diff(N.n_features);
    
    // Debug information
    if (DEBUG_PRINT) {
        printf("Updating values for A: ");
        A.print_debug();
        printf(", NB: ");
        NB.print_debug();
        printf("\nLength of union: %d\n", len_union);
    }
    
    // Process each subset
    for (int i = 1; i < (1 << len_union); i++) {
        int subset_size = __builtin_popcount(i);
        
        // Only process subsets up to max_order
        if (subset_size > max_order) continue;

        // Construct the subset BitSet
        subset.set_all_zero();
        for (int j = 0; j < len_union; j++) {
            if (i & (1 << j)) {
                subset.set(feature_list[j]);
            }
        }

        if (DEBUG_PRINT) {
            printf("Updating Subset: ");
            subset.print_debug();
            printf("\n");
        }

        // Compute the sets for the calculation of the weights
        intersection.set_intersection(subset, NB);
        B_union_subset.set_union(B, subset);
        B_and_subset.set_intersection(B, subset);
        N_set_diff.set_difference(N, B_union_subset);

        int parity = intersection.popcount() % 2;

        tfloat weight = (parity == 0) ? 1 : -1;

        int a = A.popcount() - (B_and_subset.popcount());
        int b = (N_set_diff).popcount();
       
        weight *= 1.0 / ((a + b + 1) * binomial_coefficient(a+b,a));

        // Debug
        if (DEBUG_PRINT) {
            printf("Parity: %d\n", parity);
            printf("Initial Weight: %f\n", weight);
            printf("a: %d, b: %d\n", a, b);
            printf("Binomial Coefficient(%d, %d) = %d\n", a + b, a, binomial_coefficient(a + b, a));
            printf("Weight: %f\n", weight);
        }

        // Direct array indexing instead of hash map
        int array_idx = bitset_to_array_index(subset, n_features);
        if (array_idx >= 0) {
            values[array_idx] += weight * const_value;
        }
    }
    
    if (DEBUG_PRINT) {
        printf("Const value: %f\n", const_value);
    }

    delete[] feature_list;
}

// Version using sparse map with BitSets
inline void update_values_sparse(const BitSet& A, const BitSet& B, const BitSet& N, tfloat const_value, 
                                 SparseSubsetMap& sparse_map, int max_order) {
    // Compute difference NB = N \ B
    BitSet NB(N.n_features);
    NB.set_difference(N, B);

    // Compute union: present_features = A | NB
    BitSet present_features(A.n_features);
    present_features.set_union(A, NB);
    
    // Skip if union is empty
    int len_union = present_features.popcount();
    if (len_union == 0) {
        return;  // No features to process
    }
    
    // Extract feature indices from union
    int* feature_list = new int[len_union];
    int idx = 0;
    for (int i = 0; i < present_features.n_features; i++) {
        if (present_features.test(i)) {
            feature_list[idx++] = i;
        }
    }
    
    // Pre-allocate reusable BitSets
    BitSet subset(present_features.n_features);
    BitSet intersection(A.n_features);
    BitSet B_union_subset(N.n_features);
    BitSet B_and_subset(N.n_features);
    BitSet N_set_diff(N.n_features);
    
    // Debug information
    if (DEBUG_PRINT) {
        printf("Updating values for A: ");
        A.print_debug();
        printf(", NB: ");
        NB.print_debug();
        printf("\nLength of union: %d\n", len_union);
    }
    
    // Process each subset
    for (int i = 1; i < (1 << len_union); i++) {
        int subset_size = __builtin_popcount(i);
        
        // Only process subsets up to max_order
        if (subset_size > max_order) continue;

        // Construct the subset BitSet
        subset.set_all_zero();
        for (int j = 0; j < len_union; j++) {
            if (i & (1 << j)) {
                subset.set(feature_list[j]);
            }
        }

        if (DEBUG_PRINT) {
            printf("Updating Subset: ");
            subset.print_debug();
            printf("\n");
        }

        // Compute the Ses for the calculation of the weights
        intersection.set_intersection(subset, NB);
        B_union_subset.set_union(B, subset);
        B_and_subset.set_intersection(B, subset);
        N_set_diff.set_difference(N, B_union_subset);

        int parity = intersection.popcount() % 2;

        tfloat weight = (parity == 0) ? 1 : -1;


        int a = A.popcount() - (B_and_subset.popcount());
        int b = (N_set_diff).popcount();
       
        weight *= 1.0 / ((a + b + 1) * binomial_coefficient(a+b,a));

        // Debug
        if (DEBUG_PRINT) {
            printf("Parity: %d\n", parity);
            printf("Initial Weight: %f\n", weight);
            printf("a: %d, b: %d\n", a, b);
            printf("Binomial Coefficient(%d, %d) = %d\n", a + b, a, binomial_coefficient(a + b, a));
            printf("Weight: %f\n", weight);
        }

        // Add to sparse map
        sparse_map.add(subset, weight * const_value);
    }
    
    if (DEBUG_PRINT) {
        printf("Const value: %f\n", const_value);
        printf("Sparse map size after update: %d\n", sparse_map.get_size());
    }

    delete[] feature_list;
}

// Original version (kept for compatibility)
inline void update_values(int A, int NB, tfloat const_value, tfloat* values) {
    int present_features = A | NB; // All the features in A and NB
    int len_union = __builtin_popcount(present_features); // Number of features in A and NB
    if (len_union == 0) {
        // No features to process
        return;
    }

    int* valid_pos = new int[len_union];
    int index = 0;
    int pf = present_features;
    while (pf)
    {
        int bit = pf & -pf;       // Isolate lowest set bit. -pf is two's complement of pf, flipped bits plus one.
        valid_pos[index++] = bit; // Store it
        pf ^= bit;                // Clear that bit
    }
    int* power_set = create_powerset(len_union,valid_pos); // Create powerset of union
    
    // Debug information
    if (DEBUG_PRINT) {
        printf("Updating values for A: %d, NB: %d\n", A, NB);
        printf("Present features bitmask: %d\n", present_features);
        printf("Length of union: %d\n", len_union);
        printf("Powerset subsets:\n");
        for (int i = 0; i < (1 << len_union); i++) {
            printf("Subset %d: bitmask %d\n", i, power_set[i]);
        }
    }

    for (int i = 0; i < (1 << len_union); i++) {
        int subset = present_features & power_set[i]; // Get subset S

        // Build subset S based on union and subset
        int weight = (__builtin_popcount(subset & NB) % 2 == 0) ? 1 : -1;
        
        // Update values for the entry subset
        values[subset] += weight * const_value;
    }

    // Output the updated values
    if (DEBUG_PRINT) {
        printf("Constv values: %f\n", const_value);
        printf("Updated values:\n");
        for (int i = 0; i < 2; i++) {
            printf("values[%d] = %f\n", i, values[i]);
        }
    }

    delete[] power_set;
    delete[] valid_pos;

}

// Stack frame for old bitmask-based approach (32-feature limit)
struct StackFrameBitmask {
    int node;
    int A;  // Bitmask for set A
    int B;  // Bitmask for set B
    
    StackFrameBitmask() : node(0), A(0), B(0) {}  // Default constructor
    StackFrameBitmask(int n, int a, int b) : node(n), A(a), B(b) {}
};

// Old bitmask-based implementation (32-feature limit, direct array writes)
// This function is for benchmarking to isolate BitSet overhead
void compute_values_interventional_bitmask(
    const Tree &tree,
    const tfloat *x,
    const tfloat *reference_points,
    tfloat *values,
    int n_ref,
    int n_features,
    int max_order = -1  // -1 means no limit (compute all subsets)
) {
    if (n_features > 32) {
        printf("ERROR: compute_values_interventional_bitmask only supports up to 32 features!\n");
        return;
    }
    
    if (DEBUG_PRINT) {
        printf("Starting compute_values_interventional_bitmask (old approach)\n");
        printf("Total features (n_features): %d\n", n_features);
        printf("Total reference points (n_ref): %d\n", n_ref);
    }
    
    int N = (1 << n_features) - 1;  // All features bitmask
    int MAX_STACK_SIZE = 1000;
    StackFrameBitmask* stack = new StackFrameBitmask[MAX_STACK_SIZE];
    
    for (int ref_idx = 0; ref_idx < n_ref; ref_idx++) {
        const tfloat* ref_point = reference_points + ref_idx * n_features;
        
        // Initialize stack
        int stack_top = 0;
        stack[stack_top++] = StackFrameBitmask(0, 0, N);
        
        while (stack_top > 0) {
            StackFrameBitmask current = stack[--stack_top];
            int node = current.node;
            int A = current.A;
            int B = current.B;
            
            // Inner node logic
            if (tree.is_internal(node)) {
                int feature = tree.features[node];
                tfloat threshold = tree.thresholds[node];
                
                int x_child = (x[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                int ref_child = (ref_point[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                
                int feature_mask = 1 << feature;
                
                if (x_child == ref_child) {
                    // Both go same way
                    stack[stack_top++] = StackFrameBitmask(x_child, A, B);
                } else {
                    // Split paths
                    if (B & feature_mask) {
                        // Feature is in B, add to A and go x path
                        stack[stack_top++] = StackFrameBitmask(x_child, A | feature_mask, B);
                    }
                    
                    if (!(A & feature_mask)) {
                        // Feature not in A, remove from B and go ref path
                        stack[stack_top++] = StackFrameBitmask(ref_child, A, B & ~feature_mask);
                    }
                }
            } else {
                // Leaf node
                tfloat const_value = tree.values[node] / n_ref;
                
                // Compute NB = N \ B
                int NB = N & ~B;
                
                // Update values using old approach
                update_values(A, NB, const_value, values);
            }
        }
    }
    
    delete[] stack;
    
    if (DEBUG_PRINT) {
        printf("Completed compute_values_interventional_bitmask\n");
    }
}

// Direct array indexing implementation with BitSets (unlimited features, O(1) access)
// This function uses pre-allocated array with combinatorial indexing
void compute_values_interventional_direct(
    const Tree &tree,
    const tfloat *x,
    const tfloat *reference_points,
    tfloat *values,
    int n_ref,
    int n_features,
    int max_order = -1  // -1 means no limit (compute all subsets)
) {
    if (DEBUG_PRINT) {
        printf("Starting compute_values_interventional_direct\n");
        printf("Total features (n_features): %d\n", n_features);
        printf("Total reference points (n_ref): %d\n", n_ref);
        printf("Max order: %d\n", max_order);
    }

    // Create BitSet N with all features
    BitSet N(n_features);
    for (int i = 0; i < n_features; i++) {
        N.set(i);
    }
    if (DEBUG_PRINT) {
        printf("BitSet N (all features): ");
        N.print_debug();
        printf("\n");
    }

    // Create a stack for iterative traversal
    int MAX_STACK_SIZE = 1000;
    std::vector<StackFrame> stack;
    stack.reserve(MAX_STACK_SIZE);
    
    for (int ref_idx = 0; ref_idx < n_ref; ref_idx++) {
        if (DEBUG_PRINT) {
            printf("Reference point %d/%d\n", ref_idx + 1, n_ref);
        }
        const tfloat* ref_point = reference_points + ref_idx * n_features;
        
        // Initialize stack for this reference point
        stack.clear();
        BitSet empty_A(n_features);  // A = {}
        BitSet full_B(N);            // B = N (all features)
        stack.push_back(StackFrame(0, empty_A, full_B));
        
        while (!stack.empty()) {
            StackFrame current = stack.back();
            stack.pop_back();
            int node = current.node;
            BitSet& A = current.A;
            BitSet& B = current.B;
            
            if (DEBUG_PRINT) {
                printf("Processing node %d, A: ", node);
                A.print_debug();
                printf(", B: ");
                B.print_debug();
                printf("\n");
            }
            
            // Inner node logic
            if (tree.is_internal(node)) {
                int feature = tree.features[node];
                tfloat threshold = tree.thresholds[node];
                
                int x_child = (x[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                int ref_child = (ref_point[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                
                if (DEBUG_PRINT) {
                    printf("Feature %d: x_child=%d, ref_child=%d\n", feature, x_child, ref_child);
                }
                
                if (x_child == ref_child) {
                    // Both go same way - no split, continue down
                    stack.push_back(StackFrame(x_child, A, B));
                } else {
                    // Paths split
                    if (B.test(feature)) {
                        // Feature is in B: add to A, go x path
                        BitSet new_A(A);
                        new_A.set(feature);
                        stack.push_back(StackFrame(x_child, new_A, B));
                    }
                    
                    if (!A.test(feature)) {
                        // Feature not in A: remove from B, go ref path
                        BitSet new_B(B);
                        new_B.clear(feature);
                        stack.push_back(StackFrame(ref_child, A, new_B));
                    }
                }
            } else {
                // Leaf node - update values array directly
                tfloat const_value = tree.values[node] / n_ref;
                
                if (DEBUG_PRINT) {
                    printf("Leaf value: %f\n", const_value);
                }
                
                // Use direct array indexing instead of hash map
                update_values_direct(A, B, N, const_value, values, n_features, max_order);
            }
        }
    }
    
    if (DEBUG_PRINT) {
        printf("Completed compute_values_interventional_direct\n");
    }
}

void compute_values_interventional(
    const Tree &tree,
    const tfloat *x,
    const tfloat *reference_points,
    tfloat *values,
    int n_ref,
    int n_features,
    int max_order = -1  // -1 means no limit (compute all subsets)
) {
    if (DEBUG_PRINT) {
        printf("Starting compute_values_interventional\n");
        printf("Total features (n_features): %d\n", n_features);
        printf("Total reference points (n_ref): %d\n", n_ref);
    }

    
    // Create BitSet N with all features
    BitSet N(n_features);
    for (int i = 0; i < n_features; i++) {
        N.set(i);
    }
    if (DEBUG_PRINT) {
        printf("BitSet N (all features): ");
        N.print_debug();
        printf("\n");
    }

    // Create sparse map for efficient subset storage
    int estimated_subsets = 256;
    SparseSubsetMap sparse_map(n_features, estimated_subsets);
    
// Create a stack for iterative traversal
// Note: We need to use dynamic allocation since StackFrame contains BitSet members
int MAX_STACK_SIZE = 1000;
std::vector<StackFrame> stack;
stack.reserve(MAX_STACK_SIZE);

for (int ref_idx = 0; ref_idx < n_ref; ref_idx++) {
    if (DEBUG_PRINT) {
        printf("Reference point %d/%d\n", ref_idx + 1, n_ref);
    }
    const tfloat* ref_point = reference_points + ref_idx * n_features;
    
    // Initialize stack for this reference point
    stack.clear();
    BitSet empty_A(n_features);
    stack.push_back(StackFrame(0, empty_A, N));
    
    while (!stack.empty()) {
        StackFrame current = stack.back();
        stack.pop_back();
        int node = current.node;
        const BitSet& A = current.A;
        const BitSet& B = current.B;
        
        if (DEBUG_PRINT) {
            printf("Visiting node %d with A: ", node);
            A.print_debug();
            printf(", B: ");
            B.print_debug();
            printf("\n");
        }
        
        // Inner node logic
        if (tree.is_internal(node)) {
            int feature = tree.features[node];
            tfloat threshold = tree.thresholds[node];
            
            int x_child = (x[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
            int ref_child = (ref_point[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
            
            if (DEBUG_PRINT) {
                printf("Feature %d, Threshold: %f\n", feature, threshold);
                printf("x[feature]: %f, ref_point[feature]: %f\n", x[feature], ref_point[feature]);
                printf("x_child: %d, ref_child: %d\n", x_child, ref_child);
            }
            
            if (x_child == ref_child) {
                // Both go same way
                stack.push_back(StackFrame(x_child, A, B));
            }
            else {
                // Split paths
                // Check if feature is in B
                if (B.test(feature)) {
                    // Feature is in B, branch both ways
                    BitSet new_A(A);
                    new_A.set(feature);
                    stack.push_back(StackFrame(x_child, new_A, B));

                    if (DEBUG_PRINT) {
                        printf("Feature %d is in B\n", feature);
                        printf("Adding to stack: node %d, A: ", x_child);
                        new_A.print_debug();
                        printf(", B: ");
                        B.print_debug();
                        printf("\n");
                    }
                }
                
                // Check if feature is not in A
                if (!A.test(feature)) {
                    BitSet new_B(B);
                    new_B.clear(feature);
                    stack.push_back(StackFrame(ref_child, A, new_B));
                    if (DEBUG_PRINT) {
                        printf("Feature %d is not in A\n", feature);
                        printf("Adding to stack: node %d, A: ", ref_child);
                        A.print_debug();
                        printf(", B: ");
                        new_B.print_debug();
                        printf("\n");
                    }
                }
            }
        } else {
            tfloat const_value = tree.values[node] / n_ref;
            if (DEBUG_PRINT) {
                printf("--- Leaf node %d reached. Const value: %f---\n", node, const_value);
                printf("N: ");
                N.print_debug();
                printf(", A: ");
                A.print_debug();
                printf("\n");
                printf("B: ");
                B.print_debug();
                printf("\n");
            }
            // Use sparse map with max_order
            int effective_max_order = (max_order < 0) ? n_features : max_order;
            update_values_sparse(A, B, N, const_value, sparse_map, effective_max_order);
            
            if (DEBUG_PRINT) {
                printf("--- Leaf node %d processing complete ---\n", node);
            }
        }
    }
    }
    
    // Write back accumulated values from sparse map to values array
    if (DEBUG_PRINT) {
        printf("Writing back %d entries from sparse map to values array\n", sparse_map.get_size());
    }
    sparse_map.write_back(values);
}

// New version that returns the sparse map instead of writing to array
SparseSubsetMap* compute_values_interventional_sparse(
    const Tree &tree,
    const tfloat *x,
    const tfloat *reference_points,
    int n_ref,
    int n_features,
    int max_order = -1  // -1 means no limit (compute all subsets)
) {
    if (DEBUG_PRINT) {
        printf("Starting compute_values_interventional_sparse\n");
        printf("Total features (n_features): %d\n", n_features);
        printf("Total reference points (n_ref): %d\n", n_ref);
    }

    // Create BitSet N with all features
    BitSet N(n_features);
    for (int i = 0; i < n_features; i++) {
        N.set(i);
    }
    if (DEBUG_PRINT) {
        printf("BitSet N (all features): ");
        N.print_debug();
        printf("\n");
    }

    // Create sparse map for efficient subset storage
    int estimated_subsets = 256;
    SparseSubsetMap* sparse_map = new SparseSubsetMap(n_features, estimated_subsets);
    
    // Create a stack for iterative traversal
    int MAX_STACK_SIZE = 1000;
    std::vector<StackFrame> stack;
    stack.reserve(MAX_STACK_SIZE);
    
    for (int ref_idx = 0; ref_idx < n_ref; ref_idx++) {
        if (DEBUG_PRINT) {
            printf("Reference point %d/%d\n", ref_idx + 1, n_ref);
        }
        const tfloat* ref_point = reference_points + ref_idx * n_features;
        
        // Initialize stack for this reference point
        stack.clear();
        BitSet empty_A(n_features);
        stack.push_back(StackFrame(0, empty_A, N));
        
        while (!stack.empty()) {
            StackFrame current = stack.back();
            stack.pop_back();
            int node = current.node;
            const BitSet& A = current.A;
            const BitSet& B = current.B;
            
            if (DEBUG_PRINT) {
                printf("Visiting node %d with A: ", node);
                A.print_debug();
                printf(", B: ");
                B.print_debug();
                printf("\n");
            }
            
            // Inner node logic
            if (tree.is_internal(node)) {
                int feature = tree.features[node];
                tfloat threshold = tree.thresholds[node];
                
                int x_child = (x[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                int ref_child = (ref_point[feature] <= threshold) ? tree.children_left[node] : tree.children_right[node];
                
                if (DEBUG_PRINT) {
                    printf("Feature %d, Threshold: %f\n", feature, threshold);
                    printf("x[feature]: %f, ref_point[feature]: %f\n", x[feature], ref_point[feature]);
                    printf("x_child: %d, ref_child: %d\n", x_child, ref_child);
                }

                if (x_child == ref_child) {
                    // Both go same way
                    stack.push_back(StackFrame(x_child, A, B));
                }
                else {
                    // Split paths
                    // Check if feature is in B
                    if (B.test(feature)) {
                        BitSet new_A(A);
                        new_A.set(feature);
                        stack.push_back(StackFrame(x_child, new_A, B));
                        if (DEBUG_PRINT)
                        {
                            printf("Feature %d is in B\n", feature);
                            printf("Adding to stack: node %d, A: ", ref_child);
                            new_A.print_debug();
                            printf(", B: ");
                            B.print_debug();
                            printf("\n");
                        }
                    }
                    
                    // Check if feature is not in A
                    if (!A.test(feature)) {
                        BitSet new_B(B);
                        new_B.clear(feature);
                        stack.push_back(StackFrame(ref_child, A, new_B));
                        if (DEBUG_PRINT)
                        {
                            printf("Feature %d is not in A\n", feature);
                            printf("Adding to stack: node %d, A: ", ref_child);
                            A.print_debug();
                            printf(", B: ");
                            new_B.print_debug();
                            printf("\n");
                        }
                    }
                }
            } else {
                                
                tfloat const_value = tree.values[node] / n_ref;
                if (DEBUG_PRINT) {
                    printf("--- Leaf node %d reached. Const value: %f---\n", node, const_value);
                    printf("N: ");
                    N.print_debug();
                    printf(", A: ");
                    A.print_debug();
                    printf("\n");
                    printf("B: ");
                    B.print_debug();
                    printf("\n");
                }
                
                // Use sparse map with max_order
                int effective_max_order = (max_order < 0) ? n_features : max_order;
                update_values_sparse(A, B, N, const_value, *sparse_map, effective_max_order);
                
                if (DEBUG_PRINT) {
                    printf("--- Leaf node %d processing complete ---\n", node);
                }
            }
        }
    }
    
    return sparse_map;  // Caller must delete this
}

#endif // INTERVENTIONAL_TREE_SHAP_ITERATIVE_H