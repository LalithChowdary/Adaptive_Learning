# List Indexing (`positive`, `negative`, and dynamic index expressions)

## Core Concepts and Mental Model
Python lists are **zero-based ordered sequences** supporting both forward (non-negative) and backward (negative) integer offsets:

1. **Zero-Based Positive Indexing**: For list `arr` of length `N = len(arr)`:
   - First item is `arr[0]`.
   - Last item is `arr[N - 1]`.
   - Valid non-negative indices satisfy `0 <= idx <= N - 1`. Accessing `arr[N]` always raises `IndexError`!
2. **Negative Indexing**: Counts backward from the end of the list (`arr[-k]` is equivalent to `arr[len(arr) - k]`):
   - Last item is `arr[-1]`.
   - Second-to-last is `arr[-2]`.
   - First item is `arr[-N]`.
   - Valid negative indices satisfy `-N <= idx <= -1`. Accessing `arr[-(N + 1)]` raises `IndexError`.
3. **Computed / Expression Indices**: Index expressions inside brackets (`arr[i + 1]`, `arr[arr[0]]`, `arr[len(arr) // 2]`) are fully evaluated to an integer first before lookup.

## Common Misconceptions and Pitfalls
- **One-based counting**: Treating `arr[1]` as the first item instead of the second (`arr[0]` is position 0).
- **Negative `-0` or off-by-one tail**: Expecting `arr[-1]` to exclude the final element (unlike `range` slices) or forgetting that `-1` is the tail while `len(arr) - 1` is also the tail.
- **Boundary overflow on `len(arr)`**: Writing `arr[len(arr)]` instead of `arr[len(arr) - 1]` triggers an immediate `IndexError`.
- **Nested indexing order**: In `vals[idx[k]]`, resolve `idx[k]` completely before indexing into `vals`.

## Difficulty Progression and Worked Examples
- **Easy**: Direct positive lookups (`nums[0]`, `nums[2]`) and tail lookup (`nums[-1]`) on small 4–5 element lists.
- **Medium**: Arithmetic index expressions (`nums[i + 1]`, `nums[len(nums) - 2]`, negative offsets in loops).
- **Hard**: Loop-driven index swaps/reads (`nums[i]` paired with `nums[-1 - i]`), indirect indexing (`a[b[i]]`), and guarding bounds cleanly when comparing neighbors (`i < len(nums) - 1`).

## Prerequisite Connections
- **Depends on**: `range_boundaries` (matching `range(len(arr))` indices strictly to valid offsets `0 .. len(arr) - 1` without stepping onto `len(arr)`).
