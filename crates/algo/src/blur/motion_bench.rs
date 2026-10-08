//! Native, opt-in benchmarks for the full tiled Motion Blur pipeline.
#![cfg(not(target_arch = "wasm32"))]

use super::*;
use photocraft_color::PixelFormat;
use photocraft_raster::Surface;
use rayon::prelude::*;
use std::time::Instant;

type MotionKernel = fn(&Image, Rect, &Ctx, f32, f32) -> Vec<f32>;

/// The original kernel with the production tile sizes, scheduling, halo reads, writes and
/// pruning. This deliberately measures the same pipeline as `apply_in`, rather than comparing
/// the optimized pipeline with a single-threaded or already-decoded kernel.
#[allow(clippy::panic)] // This helper is compiled only for the opt-in benchmark test.
fn apply_direct(surface: &Surface, params: &FilterParams, area: Rect) -> Surface {
    apply_reference(surface, params, area, motion_direct)
}

#[allow(clippy::panic)] // Benchmark-only, with fixed Motion Blur parameters.
fn apply_reference(surface: &Surface, params: &FilterParams, area: Rect, filter: MotionKernel) -> Surface {
    let FilterParams::MotionBlur { angle, distance } = params else { panic!("Motion Blur benchmark parameters required") };
    let crate::Halo::Radius(halo) = params.halo() else { panic!("Motion Blur must declare a radius halo") };
    let tile = crate::auto_tile(params);
    let fmt = surface.format();
    let ctx = Ctx { bounds: area, mode: fmt.mode, alpha: fmt.alpha };
    let mut out = surface.clone();
    let mut tiles = Vec::new();
    let mut y = area.y0;
    while y < area.y1 {
        let mut x = area.x0;
        while x < area.x1 {
            tiles.push(Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1)));
            x += tile;
        }
        y += tile;
    }
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let total = tiles.len().max(1);
    let ctl = Interrupt::NONE;
    let run = |t: &Rect| {
        let src = Image::read_clamped(surface, t.inflate(halo), area);
        let data = filter(&src, *t, &ctx, *angle, *distance);
        let n = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        ctl.progress(0.98 * n as f32 / total as f32);
        (*t, data)
    };
    let per_tile = (tile.max(1) as usize).pow(2) * fmt.channels() * std::mem::size_of::<f32>();
    let group = (crate::RESULT_BUDGET / per_tile.max(1)).max(rayon::current_num_threads()).max(1);
    for chunk in tiles.chunks(group) {
        let results: Vec<_> = chunk.par_iter().map(run).collect();
        for (t, data) in results {
            out.write_region(t, &data);
        }
    }
    out.prune();
    ctl.progress(1.0);
    out
}

fn compare_fft(surface: &Surface, area: Rect, angle: f32, distance: f32, label: &str) {
    compare_convolution(surface, area, angle, distance, label, Baseline::Precomputed);
}

enum Baseline {
    Original,
    Precomputed,
}

fn compare_convolution(surface: &Surface, area: Rect, angle: f32, distance: f32, label: &str, baseline: Baseline) {
    let (filter, prefix, baseline_field): (MotionKernel, &str, &str) = match baseline {
        Baseline::Original => (motion_direct, "motion original-final", "original_ms"),
        Baseline::Precomputed => (motion, "motion fft", "sampled_ms"),
    };
    let params = FilterParams::MotionBlur { angle, distance };
    let mut sampled_ms = [0.0; 3];
    let mut fft_ms = [0.0; 3];
    for run in 0..3 {
        let sampled = || {
            let start = Instant::now();
            let result = apply_reference(surface, &params, area, filter);
            (result, start.elapsed().as_secs_f64() * 1000.0)
        };
        let fft = || {
            let start = Instant::now();
            let result = crate::apply_in(surface, &params, area, area, None, area);
            (result, start.elapsed().as_secs_f64() * 1000.0)
        };
        let ((before, before_ms), (after, after_ms)) = if run % 2 == 0 {
            (sampled(), fft())
        } else {
            let after = fft();
            (sampled(), after)
        };
        sampled_ms[run] = before_ms;
        fft_ms[run] = after_ms;
        let before = before.to_interleaved(area);
        let after = after.to_interleaved(area);
        let mut max_error = 0u8;
        let mut changed = 0usize;
        let mut squared = 0u64;
        for (&a, &b) in before.iter().zip(&after) {
            let error = a.abs_diff(b);
            max_error = max_error.max(error);
            changed += usize::from(error != 0);
            squared += u64::from(error).pow(2);
        }
        assert_eq!(before.len(), after.len());
        assert!(max_error <= 1, "{label}, distance={distance}: U8 error {max_error}");
        println!(
            "{prefix} run: {label}, {}x{}, angle={angle}, distance={distance}, run={run}, {baseline_field}={before_ms:.3}, fft_ms={after_ms:.3}, max_u8_error={max_error}, changed_samples={changed}, rms_u8_error={:.6}",
            area.width(),
            area.height(),
            (squared as f64 / before.len() as f64).sqrt()
        );
    }
    println!(
        "{prefix} median: {label}, {}x{}, angle={angle}, distance={distance}, {baseline_field}={:.3}, fft_ms={:.3}, speedup={:.3}, sampled_runs={sampled_ms:?}, fft_runs={fft_ms:?}",
        area.width(),
        area.height(),
        median(&sampled_ms),
        median(&fft_ms),
        median(&sampled_ms) / median(&fft_ms)
    );
}

/// Direct comparison of the untouched original kernel with the final public
/// pipeline. Same 24 MP source, angle, tile IO and Rayon pool as earlier passes;
/// every timed pair is compared over the entire image outside the timed region.
#[test]
#[ignore = "slow original-versus-final 24 MP comparison; run explicitly in release"]
fn motion_original_final_performance() {
    println!("motion original-final machine: os={}, arch={}, rayon_workers={}", std::env::consts::OS, std::env::consts::ARCH, rayon::current_num_threads());
    let quick = std::env::var_os("PHOTOCRAFT_MOTION_BENCH_QUICK").is_some();
    let area = if quick { Rect::new(0, 0, 600, 400) } else { Rect::new(0, 0, 6000, 4000) };
    let surface = source(area, false);
    for distance in [64.0, 256.0] {
        compare_convolution(&surface, area, 30.0, distance, if quick { "small synthetic" } else { "24 MP opaque" }, Baseline::Original);
    }
}

/// Sparse legacy checks at the largest distances, without decoding a huge
/// square halo for every point. Each tap gets the same four clamped samples
/// and calls the original Image sampler with the original f32 expressions.
fn validate_wide_points(source: &Surface, actual: &Surface, area: Rect, angle: f32, distance: f32) {
    let (sin, cos) = angle.to_radians().sin_cos();
    let steps = distance.ceil() as i32;
    let points = [
        (0, 0),
        (area.x1 - 1, 0),
        (0, area.y1 - 1),
        (area.x1 - 1, area.y1 - 1),
        (area.x1 / 2, area.y1 / 2),
        (511, 511),
        (512, 512),
        (2047, 2047),
        (2048, 2048),
    ];
    let mut checked = 0;
    let mut maximum = 0;
    for (x, y) in points.into_iter().filter(|&(x, y)| area.contains(x, y)) {
        let mut acc = [0.0f32; 4];
        let mut sample = [0.0f32; 4];
        for i in 0..=steps {
            let t = i as f32 / steps as f32 - 0.5;
            let (sx, sy) = (x as f32 + 0.5 + cos * distance * t, y as f32 + 0.5 - sin * distance * t);
            let (ix, iy) = ((sx - 0.5).floor() as i32, (sy - 0.5).floor() as i32);
            let rect = Rect::new(ix, iy, ix + 2, iy + 2);
            let mut data = Vec::with_capacity(16);
            for dy in 0..2 {
                for dx in 0..2 {
                    data.extend(source.pixel((ix + dx).clamp(area.x0, area.x1 - 1), (iy + dy).clamp(area.y0, area.y1 - 1)));
                }
            }
            let src = Image { rect, ch: 4, data };
            src.sample(sx, sy, Edge::Transparent, rect, true, &mut sample);
            for c in 0..3 {
                acc[c] += sample[c] * sample[3];
            }
            acc[3] += sample[3];
        }
        let norm = 1.0 / (steps + 1) as f32;
        for value in &mut acc {
            *value *= norm;
        }
        let alpha = acc[3];
        for value in acc.iter_mut().take(3) {
            *value = if alpha > 1e-7 { *value / alpha } else { 0.0 };
        }
        let rect = Rect::new(x, y, x + 1, y + 1);
        let mut expected = Surface::new(source.format());
        expected.write_region(rect, &acc);
        for (&a, &b) in expected.to_interleaved(rect).iter().zip(actual.to_interleaved(rect).iter()) {
            maximum = maximum.max(a.abs_diff(b));
        }
        checked += 1;
    }
    assert!(maximum <= 1, "distance={distance}, sparse legacy U8 error={maximum}");
    println!("motion wide sample validation: distance={distance}, points={checked}, max_u8_error={maximum}");
}

/// Compare the new convolution with the previous precomputed sampling pipeline.
/// Includes decode/halo reads, transforms, writes and pruning; excludes input
/// construction, comparison, UI proxy construction, composition and GPU upload.
#[test]
#[ignore = "24 MP and large-distance FFT comparison; run explicitly in release"]
fn motion_fft_performance() {
    println!("motion fft machine: os={}, arch={}, rayon_workers={}", std::env::consts::OS, std::env::consts::ARCH, rayon::current_num_threads());
    let quick = std::env::var_os("PHOTOCRAFT_MOTION_BENCH_QUICK").is_some();
    let area = if quick { Rect::new(0, 0, 600, 400) } else { Rect::new(0, 0, 6000, 4000) };
    let large = source(area, false);
    for distance in [64.0, 256.0] {
        compare_fft(&large, area, 30.0, distance, if quick { "small synthetic" } else { "24 MP opaque" });
    }
    for distance in [1000.0, 2000.0] {
        let params = FilterParams::MotionBlur { angle: 30.0, distance };
        let mut times = [0.0; 3];
        for (run, time) in times.iter_mut().enumerate() {
            let start = Instant::now();
            let result = crate::apply_in(&large, &params, area, area, None, area);
            *time = start.elapsed().as_secs_f64() * 1000.0;
            validate_wide_points(&large, &result, area, 30.0, distance);
            println!("motion fft large-only: distance={distance}, run={run}, fft_ms={time:.3}");
        }
        println!(
            "motion fft large-only median: {}x{} opaque, angle=30, distance={distance}, fft_ms={:.3}, runs={times:?}",
            area.width(),
            area.height(),
            median(&times)
        );
    }
    drop(large);
    let area = if quick { Rect::new(0, 0, 600, 400) } else { Rect::new(0, 0, 1500, 1000) };
    let proxy = source(area, false);
    for distance in [64.0, 256.0, 1000.0] {
        compare_fft(&proxy, area, 30.0, distance, "1.5 MP synthetic");
    }
    let variable = source(area, true);
    compare_fft(&variable, area, 43.0, 256.0, "1.5 MP variable alpha");
}

fn source(area: Rect, variable_alpha: bool) -> Surface {
    let mut surface = Surface::new(PixelFormat::RGBA8);
    let bytes: Vec<u8> = (area.y0..area.y1)
        .flat_map(|y| {
            (area.x0..area.x1).flat_map(move |x| {
                [(x % 256) as u8, ((x * 3 + y) / 11 % 256) as u8, ((x ^ y) % 256) as u8, if variable_alpha { (x ^ (y * 7)) as u8 } else { 255 }]
            })
        })
        .collect();
    surface.write_interleaved(area, &bytes);
    surface
}

fn median(values: &[f64; 3]) -> f64 {
    let mut sorted = *values;
    sorted.sort_by(f64::total_cmp);
    sorted[1]
}

fn compare(surface: &Surface, area: Rect, angle: f32, distance: f32, label: &str) {
    let params = FilterParams::MotionBlur { angle, distance };
    let mut original_ms = [0.0; 3];
    let mut optimized_ms = [0.0; 3];
    for run in 0..3 {
        // Alternate which implementation runs first to reduce order-related timing bias.
        let (original, optimized) = if run % 2 == 0 {
            let started = Instant::now();
            let original = apply_direct(surface, &params, area);
            original_ms[run] = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let optimized = crate::apply_in(surface, &params, area, area, None, area);
            optimized_ms[run] = started.elapsed().as_secs_f64() * 1000.0;
            (original, optimized)
        } else {
            let started = Instant::now();
            let optimized = crate::apply_in(surface, &params, area, area, None, area);
            optimized_ms[run] = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let original = apply_direct(surface, &params, area);
            original_ms[run] = started.elapsed().as_secs_f64() * 1000.0;
            (original, optimized)
        };
        // Conversion and equality are intentionally outside the timed region.
        assert!(original.to_interleaved(area) == optimized.to_interleaved(area), "{label}: byte mismatch at angle {angle}, distance {distance}, run {run}");
        println!(
            "motion run: {label}, angle={angle}, distance={distance}, run={run}, original_ms={:.3}, optimized_ms={:.3}",
            original_ms[run], optimized_ms[run]
        );
    }
    println!(
        "motion median: {label}, {}x{} RGBA8, angle={angle}, distance={distance}, tile={}, halo={:?}, original_ms={:.3}, optimized_ms={:.3}, speedup={:.3}, original_runs={original_ms:?}, optimized_runs={optimized_ms:?}",
        area.width(),
        area.height(),
        crate::auto_tile(&params),
        params.halo(),
        median(&original_ms),
        median(&optimized_ms),
        median(&original_ms) / median(&optimized_ms),
    );
}

/// Run with `cargo test --release -p photocraft-algo motion_performance -- --ignored --nocapture`.
/// Timings include source halo reads, tile kernels, output writes and pruning, and exclude
/// synthetic source construction, byte extraction/equality, proxy generation, compositing and
/// GPU upload. Every timed result is compared byte-for-byte with the preserved original kernel.
#[test]
#[ignore = "24 MP Motion Blur performance comparison; run explicitly in release"]
fn motion_performance() {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines().find_map(|line| line.strip_prefix("model name").and_then(|name| name.split_once(':')).map(|(_, name)| name.trim().to_owned()))
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("motion machine: os={}, arch={}, cpu={cpu}, rayon_workers={}", std::env::consts::OS, std::env::consts::ARCH, rayon::current_num_threads());
    let large_area = Rect::new(0, 0, 6000, 4000);
    let large = source(large_area, false);
    for distance in [10.0, 50.0] {
        compare(&large, large_area, 30.0, distance, "24 MP opaque");
    }
    drop(large);
    let proxy_area = Rect::new(0, 0, 600, 400);
    let proxy = source(proxy_area, false);
    for distance in [2.5, 12.5, 50.0] {
        compare(&proxy, proxy_area, 30.0, distance, "600x400 opaque proxy");
    }
    compare(&proxy, proxy_area, 0.0, 50.0, "600x400 opaque cardinal proxy");
    let variable = source(proxy_area, true);
    compare(&variable, proxy_area, 30.0, 12.5, "600x400 variable-alpha proxy");
}
