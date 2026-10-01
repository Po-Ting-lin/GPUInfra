# Payload identity

`GpuDataKey` now compares `(frameId, cameraId, variantId)` exactly.
The hash table uses all three fields and resolves hash collisions through exact
key equality. Variant differences are separate cache entries, not metadata errors.
They share the manager's fixed entry capacity, wait policy and inactive-entry LRU.

## Caller contract

- `variantId=0` preserves existing two-field initializers and the demo's
  raw-frame behavior. AOI should reserve zero for raw frames.
- All callers sharing a manager must agree on what each ID means. Do not use
  independently allocated task-local IDs, object addresses, or an unchecked hash
  of preprocessing parameters as the identity.
- Within a manager, the same complete key must always describe identical
  immutable bytes and the same layout until a successful quiescent reset.
- Moving or completing a lease carries the entire key. A fill is still published
  only by successful completion; adding a variant does not enable early sharing.
- Changing a variant ID does not modify data: the caller must actually generate
  the corresponding payload on CacheFill and TaskFallback.

## AOI mapping to implement in the adapter

The source review reports raw/full undistorted cache payloads. Smoothing is
performed into private buffers and is not part of cache identity. The review
cannot guarantee one distortion configuration per frame/camera within a cycle.

The graph owner should establish an exact mapping from this tuple to an ID:

```text
(payload kind, distortion table generation, distortion index, rotation)
```

Use exact structured equality in that mapping. Assign distinct nonzero IDs to
undistorted variants; look up the same ID from every task. A finite configuration
can be enumerated during cold initialization, with IDs retained across execution.
If variants cannot be enumerated in advance, the adapter needs a bounded shared
registration design before integration; do not add unbounded hot-path allocation.
The generic GPUInfra manager does not implement this AOI-specific registry.

Only normalize rotations that the actual AOI implementation guarantees produce
identical bytes. Until that equivalence is proven, keep distinct exact rotation
values. Distortion generation must change when table contents change, unless the
old cache is cleared before IDs are reused. Reject ID exhaustion instead of
wrapping and aliasing a live identity.

Example assignment (illustrative IDs, not a hashing formula):

| Payload | Variant ID |
| --- | --- |
| Raw frame | 0 |
| Undistorted, generation 5, index 2, rotation 0 | 1 |
| Undistorted, generation 5, index 2, rotation 90 | 2 |
| Undistorted, generation 6, index 2, rotation 0 | 3 |

```cpp
FrameMetadata metadata;
metadata.key.frameId = frameId;
metadata.key.cameraId = cameraId;
metadata.key.variantId = resolvedVariantId;
// Set bytes, width, height and dtype to the manager's fixed payload layout.
GpuDataAccess access = frameCache.getCacheData(metadata, request);
```

This key change provides the infrastructure capability. AOI still must wire the
shared mapping into both master and reference acquisition and reproduce the
same preprocessing for fallback. Filling in zero for all processed payloads
would preserve the original identity bug.

## Reset and layout boundaries

Variant IDs do not solve frame-ID reuse or mutable CPU source bytes. Continue to
reset once at a quiescent run boundary before reused frame IDs can be requested.
An ID mapping may be rebuilt/reused only after old requests and leases are drained
and old entries are cleared, or after the manager is destroyed and reinitialized.

Static distortion updates still require drained readers and cold reinitialization;
a generation field does not make concurrent static-buffer mutation safe.

Keep one fixed payload layout per manager. FrameMetadata currently describes
bytes, width, height and dtype, not a separate stride field. AOI must enforce its
fixed stride contract; incompatible layout changes require reinitialization.

## Verification

Tests cover colliding hashes for distinct variants, simultaneous variant fills,
same-variant Loading fallback, byte-correct hits, eviction while another variant
has a reader, fill rollback, and reset. The new GPU test uses K=2 and zero wait.
Diagnostics include `variant_id` (or `unknown` without a known key).
