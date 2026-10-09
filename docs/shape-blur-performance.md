# Shape Blur implementation and performance

The existing `filter.blur.shapeBlur` command, dialog and `FilterParams::ShapeBlur`
API use the same kernels. This optimization changes traversal and buffer reuse,
without introducing a new approximation, color conversion or dependency.

## Sampling contract

The command clamps radius to 5–1000 px. The algorithm also supports 0–5 px:
negative radii become zero and radii below 0.5 use the single origin sample.
Rasterization visits integer offsets from `-ceil(radius)` to `ceil(radius)`;
the shape predicate receives `u = dx / radius`, `v = dy / radius`. Coverage is
binary, with equal weight for every accepted sample, no subpixel coverage or
interpolation between radius levels. An empty kernel falls back to the origin.

| Shape | Existing predicate / orientation |
|---|---|
| Circle | `u² + v² ≤ 1` |
| Ring | `0.36 ≤ u² + v² ≤ 1` |
| Square | Every offset in the rasterization square; fractional radii still use `ceil(radius)` |
| Diamond | `abs(u) + abs(v) ≤ 1` |
| Triangle | `v ≤ 0.75`, `abs(u) ≤ (v + 1) / 1.75`; tip points upward |
| Hexagon | Existing regular hexagon polar predicate, circumradius 1, rotation 0 |
| Star | Even–odd polygon, ten alternating vertices of radius 1 and 0.42, first vertex upward |
| Heart | `(x² + y² - 1)³ - x²y³ ≤ 0`, with `x = 1.2u`, `y = -1.2v + 0.15` |
| Cross | `abs(u) ≤ 0.3` or `abs(v) ≤ 0.3`, within the rasterization square |

The horizontal runs, their order, and total sample count are unchanged. Ring
and concave shapes can have multiple runs per row. Samples are taken at
`(x + dx, y + dy)`, so an impulse
footprint mirrors the stored predicate for asymmetric shapes; that existing
orientation convention is retained. Normalization always divides by the full
kernel count, including when a direct `kernel` caller supplies an image without
enough halo; missing samples contribute zero. `apply` reads sparse
surface transparency outside the source. Document filtering (`apply_in` and
`apply_in_with`) repeats pixels beyond the layer's extent (canvas plus off-canvas
content) and clips output to that extent. The halo remains `ceil(radius) + 1`.
Selections restrict the output area and mix filtered samples with the original
by coverage; they do not change the convolution kernel. Cancellation and
progress still operate through tile dispatch, and the source remains immutable.

Samples stay in the source color model, with no sRGB assumption. Color channels
are multiplied by alpha in f32, accumulated and normalized in f64, converted
back to f32, and divided by the filtered alpha when alpha is greater than
`1e-7` (otherwise color is zero). Alpha itself is averaged. Without alpha every
channel is averaged directly. Integer surfaces quantize only at the existing
surface write boundary. Float/HDR values are not clipped by Shape Blur; the
existing NaN/infinity propagation from prefix arithmetic is preserved.

## Optimization and input bounds

Kernel rasterization happens once per operation rather than once per tile.
The premultiplied f32 image copy is eliminated: prefix construction performs
the same f32 multiplication while adding to f64 row sums. Convolution iterates
spans outside pixels, accumulating contiguous slices of an output row. Each
sample receives exactly the original additions in their original order;
Rust/LLVM can vectorize the independent samples without unsafe code, CPU flags
or architecture-specific intrinsics. Tiles retain the existing native Rayon
parallelism; wasm uses the same sequential implementation.

Direct algorithm radii are bounded to 0–1000; nonfinite radii use zero. These
previously unbounded inputs could overflow halo arithmetic or request absurd
kernel work. Valid command input and all supported radii retain their behavior.
Malformed `Image` channel counts/data lengths and absurd output rectangles
return an empty direct-kernel result. Checked sizes and fallible reservations
bound prefix/output buffers to 512 Mi samples each; this accommodates the
largest standard 2048 px tile with a 1001 px halo in every supported model,
and direct full-image 36 MP calls at all eight supported channels. Prefix
reservation can still fail on a 32-bit address space or under memory pressure.
An incomplete tile stops the cancellable operation before surface writes.
The legacy `apply` API returns the original surface if that operation fails;
it has no error return channel. This does not provide a general memory budget
for all other filters or for caller-created image buffers.

## Reproduction and evidence

Starting source: `578bc905a6320cbbe55d41601e4d1eaa72fdb2df` (2026-10-09).
The benchmark harness was built against that unchanged algorithm and the
resulting executable retained before rebuilding the optimized implementation.
`bench_shape_blur` times actual `apply_in` operations, including tile reads,
edge extension, filtering, output writes and pruning. Source generation and
result destruction are outside the timer. It does not measure engine history,
dialog previews or GPU refresh.

```powershell
$env:CARGO_TARGET_DIR = 'target/agent-shape-blur'
$env:RAYON_NUM_THREADS = '8'
cargo build --release -p photocraft-algo --example bench_shape_blur
# width height repetitions radius/all depth shape/all
target/agent-shape-blur/release/examples/bench_shape_blur.exe 6000 4000 3 all 8 all
```

The default matrix covers all nine shapes at radii 5, 25 and 100, with one
untimed warmup then three timed repetitions per case. Depths 8, 16 and 32 were
run separately; 32×24 overhead cases use seven repetitions. Output is CSV with
median milliseconds and the complete sorted timing samples. The synthetic
RGBA source has gradients/texture and five alpha levels, including zero.

## Windows x86_64 measurements (2026-10-09)

AMD Ryzen 7 9700X (8 cores / 16 logical processors), approximately 61.6 GiB
usable RAM, Windows 11 Pro 10.0.26200, Rust 1.97.1 MSVC. Ordinary release
profile, eight Rayon workers, no `target-cpu=native`. The SIMD and AVIF chats
were idle when checked before the timing runs; no known Cargo/rustc process
was active at final-run start. This task's compilation and tests did not overlap
measurements. OS background activity and CPU frequency were not controlled.

All 81 standard 24 MP cases (nine shapes × three radii × three depths) improved,
by **1.33–2.84×**. The mean of the case speedups is 1.96× (not a pooled
throughput measure). Across the 162 baseline/final sample groups, the mean
`(max - min) / median` is 3.81%, worst 24.79%; the full samples are retained
in the evidence report. Three repetitions are a practical development
measurement, not a precise cross-platform performance guarantee.

RGBA8 medians, milliseconds:

| Shape | r 5 baseline → final ms | r 25 baseline → final ms | r 100 baseline → final ms |
|---|---:|---:|---:|
| Circle | 330.4 → 211.8 (1.56×) | 792.5 → 342.5 (2.31×) | 2434.4 → 1080.3 (2.25×) |
| Ring | 368.9 → 237.1 (1.56×) | 919.0 → 464.0 (1.98×) | 3337.8 → 1391.7 (2.40×) |
| Square | 327.4 → 223.0 (1.47×) | 718.9 → 351.4 (2.05×) | 2239.6 → 990.2 (2.26×) |
| Diamond | 343.8 → 224.0 (1.53×) | 694.5 → 369.4 (1.88×) | 2210.5 → 1034.7 (2.14×) |
| Triangle | 306.4 → 208.6 (1.47×) | 628.7 → 332.9 (1.89×) | 2008.4 → 897.3 (2.24×) |
| Hexagon | 311.4 → 211.1 (1.47×) | 625.2 → 325.9 (1.92×) | 2084.8 → 904.4 (2.31×) |
| Star | 310.0 → 210.7 (1.47×) | 702.9 → 355.4 (1.98×) | 2329.0 → 1026.0 (2.27×) |
| Heart | 322.0 → 207.6 (1.55×) | 1010.6 → 355.9 (2.84×) | 2265.4 → 1015.9 (2.23×) |
| Cross | 319.7 → 213.4 (1.50×) | 682.6 → 348.9 (1.96×) | 2296.9 → 988.3 (2.32×) |

The 32×24 cases all improved too, by 1.43–2.02×. Their medians remain below 1.23 ms on this
machine, including radius 100. Setup/prefix construction dominates those cases.

Memory: the separate 24 MP RGBA8 Circle/r100 process peak working set fell
from 474,861,568 to 447,524,864 bytes (452.9 → 426.8 MiB, 5.8%). Each 256 px
output tile with a 101 px halo previously allocated a 3,356,224-byte extra f32
copy; the replacement row accumulator is 8,192 bytes. The largest halo has a
much larger copy. Process peaks include input setup, warmups and repetitions,
and are sampled through Windows `PeakWorkingSet64` every 50 ms; they are not
per-operation allocation counts or guarantees of a memory budget.

Additional stress cases (one warmup, three timed repetitions):

| Case | Baseline median | Final median | Speedup | Baseline → final process peak |
|---|---:|---:|---:|---:|
| 24 MP RGBA8 Circle, radius 1000 | 154.272 s | 50.448 s | 3.06× | 6.16 → 4.73 GiB |
| 36 MP RGBA32F Ring, radius 100 | 5.616 s | 2.018 s | 2.78× | 1238.2 → 1210.7 MiB |

Radius 1000 still takes tens of seconds and several GiB: the exact method
remains O(radius) per pixel and reads a wide halo. Its baseline timed samples
were 150.537–159.503 s; final 50.376–51.561 s. The 36 MP baseline samples were
5.600–5.620 s; final 1.981–2.022 s. The unchanged default automatic tile size
at radius 1000 is 2048 px, so this image has six tiles; eight Rayon workers
does not imply eight active convolution workers throughout that case.

All 111 comparisons, complete sorted samples, executable hashes and memory
records are in [`perf/shape-blur-2026-10-09.json`](../perf/shape-blur-2026-10-09.json).
The existing macOS nightly baseline and its budgets are unchanged: these
Windows development measurements are a separate report, not a replacement
baseline from a different machine class.

## Quality and validation

The original scalar prefix implementation is retained under `cfg(test)` in
`blur2.rs`; the optimized implementation must match its finite output bit for
bit. A separate point-by-point convolution tests sampling, alpha and fixed
normalization independently of prefix sums and span accumulation. The observed
maximum finite error is **0**, including exactly matching integer surface output.
NaN/infinity tests compare classification (NaN payload bits are not specified).

Tests cover all nine shapes; radii 0, 0.49, 0.5, 0.75, 1, 1.1, 1.25, 2.5, 5,
7.75, 25, 100, 999.25 and 1000; one through eight channels; alpha on/off;
empty/1×1/one-row/one-column images; negative origins, corners and absent halo;
output subrectangles; tile boundaries; repeated versus transparent edges;
feathered selections and untouched pixels; U8/U16/F32, RGB/Gray/CMYK/Lab and
Multichannel; HDR/negative/extreme values and alpha near the unpremultiply
threshold; nonfinite radii, malformed images, absurd rectangles and tile sizes;
and cancellation without modifying the input.

Native timing evidence is limited to this Windows x86_64 machine. No macOS,
Linux, ARM or wasm execution speedup is claimed. Rust 1.95 is the supported
minimum; the installed toolchain used here is 1.97.1. No newer language feature,
new dependency, unsafe block or architecture-specific intrinsic is introduced.
The shared `craftrules` checkout was absent both beside this managed worktree
and at the supplied original-repository location; the supplied AGENTS.md and
repository architecture/development/contributing/roadmap/scorecard rules were
used instead.

Completed checks:

- `cargo test -p photocraft-algo`: 299 passed; five ignored. The release run also
  passed 299 tests, including the final nonfinite-alpha cases. The ignored
  36 MP direct-image regression was run separately and passed; it reserves
  approximately 1.7 GiB, so routine test runs leave it opt-in.
- `cargo clippy -p photocraft-algo --all-targets -- -D warnings` and
  `cargo xtask layers` passed.
- Engine filter/depth/selection/undo tests: 16 passed. The ignored engine
  `panic_hunt` adversarial command test passed.
- `cargo xtask wasm` passed all 23 checks. Algo compilation also passed for
  `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin` and
  `aarch64-pc-windows-msvc`. These are compile checks, not execution tests.
- `cargo xtask perf --quick` completed successfully. Its baseline comparison
  was skipped because the stored macOS baseline is a different machine class;
  this run does not establish compliance with the full performance budgets.
- `cargo xtask scorecard` regenerated identical content. No Shape Blur budget
  or checklist measurement is registered there, so no unrelated floor or
  budget was changed.

All Cargo invocations used `CARGO_TARGET_DIR=target/agent-shape-blur`. No format,
compositor, UI or command registration changed, so corpus, visual snapshot and
menu-parity regeneration were not applicable.
