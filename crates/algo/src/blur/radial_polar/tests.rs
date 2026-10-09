use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::ColorMode;
use crate::image::{Edge, premultiply};

fn context(bounds: Rect, alpha: bool) -> Ctx {
    Ctx { bounds, mode: ColorMode::Rgb, alpha }
}

fn fixture(rect: Rect, channels: usize, alpha: bool, sharp: bool) -> Image {
    let mut src = Image::new(rect, channels);
    let width = rect.width() as usize;
    for (i, pixel) in src.data.chunks_exact_mut(channels).enumerate() {
        let x = rect.x0 as f64 + (i % width) as f64 + 0.5;
        let y = rect.y0 as f64 + (i / width) as f64 + 0.5;
        let radius = x.hypot(y);
        let angle = y.atan2(x);
        for (c, value) in pixel.iter_mut().enumerate() {
            *value = if alpha && c + 1 == channels {
                if (x as i32 + y as i32).rem_euclid(19) == 0 { 0.0 } else { (0.55 + 0.4 * (x * 0.03 + y * 0.017).sin()) as f32 }
            } else if sharp {
                let checker = ((x.floor() as i32).div_euclid(3) + (y.floor() as i32).div_euclid(3)).rem_euclid(2);
                let spokes = (angle * 19.0).sin() > 0.0;
                let rings = (radius * 0.37).sin() > 0.0;
                match c % 3 {
                    0 => checker as f32,
                    1 => {
                        if spokes {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        if rings {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
            } else {
                // Float/HDR colour with distinct frequencies per channel, no RGB assumption.
                (1.1 + 0.9 * (x * (0.013 + c as f64 * 0.009)).sin() + 0.65 * (y * (0.021 + c as f64 * 0.005)).cos()) as f32
            };
        }
    }
    src
}

/// Independent dense Cartesian midpoint quadrature, using the public sampler and no polar grid
/// or prefix code. Eight samples per source-pixel path length approach the continuous target.
fn dense_reference(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, center: (f32, f32)) -> Vec<f32> {
    dense_reference_with_density(src, out, ctx, amount, method, center, 8.0)
}

fn dense_reference_with_density(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, center: (f32, f32), density: f64) -> Vec<f32> {
    let cx = ctx.bounds.x0 as f32 + ctx.bounds.width() as f32 * center.0;
    let cy = ctx.bounds.y0 as f32 + ctx.bounds.height() as f32 * center.1;
    let arc = f64::from(amount).to_radians();
    let span = f64::from(amount) / 200.0;
    let width = out.width() as usize;
    let mut result = vec![0.0; width * out.height() as usize * src.ch];
    par_rows(&mut result, width * ROWS_PER_TASK, src.ch, |task, rows| {
        let mut sampled = vec![0.0; src.ch];
        let mut accum = vec![0.0f64; src.ch];
        for (k, row) in rows.chunks_exact_mut(width * src.ch).enumerate() {
            let y = out.y0 + (task * ROWS_PER_TASK + k) as i32;
            for (k, pixel) in row.chunks_exact_mut(src.ch).enumerate() {
                let x = out.x0 + k as i32;
                let (dx, dy) = (f64::from(x) + 0.5 - f64::from(cx), f64::from(y) + 0.5 - f64::from(cy));
                let length = dx.hypot(dy)
                    * match method {
                        RadialMethod::Spin => arc,
                        RadialMethod::Zoom => span,
                    };
                let count = (length * density).ceil().max(16.0) as usize;
                accum.fill(0.0);
                for i in 0..count {
                    let t = (i as f64 + 0.5) / count as f64;
                    let (sx, sy) = match method {
                        RadialMethod::Spin => {
                            let (s, c) = ((t - 0.5) * arc).sin_cos();
                            (f64::from(cx) + dx * c - dy * s, f64::from(cy) + dx * s + dy * c)
                        }
                        RadialMethod::Zoom => (f64::from(cx) + dx * (1.0 - span * t), f64::from(cy) + dy * (1.0 - span * t)),
                    };
                    src.sample(sx as f32, sy as f32, Edge::Transparent, src.rect, ctx.alpha, &mut sampled);
                    let a = if ctx.alpha { f64::from(*sampled.last().unwrap()) } else { 1.0 };
                    for (c, (dst, value)) in accum.iter_mut().zip(&sampled).enumerate() {
                        *dst += f64::from(*value) * if ctx.alpha && c + 1 < src.ch { a } else { 1.0 };
                    }
                }
                for value in &mut accum {
                    *value /= count as f64;
                }
                finish(&accum, ctx.alpha, pixel);
            }
        }
    });
    result
}

fn error(mut a: Vec<f32>, mut b: Vec<f32>, channels: usize, alpha: bool) -> (f64, f32) {
    premultiply(&mut a, channels, alpha);
    premultiply(&mut b, channels, alpha);
    assert_eq!(a.len(), b.len());
    let mut square = 0.0;
    let mut max = 0.0f32;
    for (a, b) in a.iter().zip(&b) {
        let difference = (a - b).abs();
        square += f64::from(difference).powi(2);
        max = max.max(difference);
    }
    ((square / a.len() as f64).sqrt(), max)
}

#[test]
fn continuous_dense_quality_on_spokes_rings_hdr_and_transparency() {
    let rect = Rect::new(-40, -32, 88, 72);
    let out = Rect::new(-16, -8, 56, 48);
    for (channels, alpha, sharp) in [(1, false, true), (2, true, false), (4, true, true), (5, true, false)] {
        let src = fixture(rect, channels, alpha, sharp);
        let ctx = context(out, alpha);
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            for (amount, center) in [(10.0, (0.5, 0.5)), (100.0, (0.5, 0.5)), (50.0, (0.0, 0.0)), (100.0, (-0.2, 0.35))] {
                let fast = filter(&src, out, &ctx, amount, method, center, &Interrupt::NONE).expect("bounded geometry");
                let reference = dense_reference(&src, out, &ctx, amount, method, center);
                let (rmse, max) = error(fast, reference, channels, alpha);
                eprintln!(
                    "polar dense quality {method:?} amount={amount} center={center:?} channels={channels} alpha={alpha} sharp={sharp}: rmse={rmse:.7} max={max:.7}"
                );
                // This is an explicit experimental error budget, not a Photoshop parity claim.
                assert!(rmse < 0.018 && max < 0.12, "{method:?} {amount} {center:?}: rmse {rmse}, max {max}");
            }
        }
    }
}

#[test]
fn constant_hdr_cmyka_alpha_and_axis_seams() {
    let rect = Rect::new(-80, -80, 80, 80);
    let out = Rect::new(-17, -17, 18, 18);
    let mut src = Image::new(rect, 5);
    for pixel in src.data.as_chunks_mut::<5>().0 {
        pixel.copy_from_slice(&[-0.5, 2.0, 4.0, 0.7, 0.37]);
    }
    // Centre is exactly a pixel centre, covering both zero radius and horizontal axis ownership.
    let ctx = context(out, true);
    let center = (0.5, 0.5);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let got = filter(&src, out, &ctx, 100.0, method, center, &Interrupt::NONE).expect("constant bounded image");
        for (i, pixel) in got.as_chunks::<5>().0.iter().enumerate() {
            for (c, (value, want)) in pixel.iter().zip([-0.5, 2.0, 4.0, 0.7, 0.37]).enumerate() {
                assert!((value - want).abs() < 2e-6, "{method:?}: pixel {i}, channel {c}: {value} != {want}");
            }
        }
    }
}

#[test]
fn refuses_bad_samples_malformed_images_and_extreme_geometry() {
    let rect = Rect::new(0, 0, 16, 16);
    let ctx = context(rect, true);
    let mut src = fixture(rect, 4, true, false);
    let args = |src: &Image, amount, center| Plan::new(src, rect, &ctx, amount, RadialMethod::Spin, center, DEFAULT_PITCH);
    assert!(args(&src, f32::NAN, (0.5, 0.5)).is_none());
    assert!(args(&src, 100.0, (f32::INFINITY, 0.5)).is_none());
    assert!(args(&src, 100.0, (5000.0, 0.5)).is_none());
    src.data[3] = -1.0;
    assert!(args(&src, 100.0, (0.5, 0.5)).is_none());
    src.data[3] = 0.5;
    src.data[0] = f32::NAN;
    assert!(args(&src, 100.0, (0.5, 0.5)).is_none());
    src.data[0] = 0.5;
    src.data.pop();
    assert!(args(&src, 100.0, (0.5, 0.5)).is_none());
    let empty = Image { rect, ch: 0, data: Vec::new() };
    assert!(args(&empty, 100.0, (0.5, 0.5)).is_none());
}

#[test]
fn allocation_and_work_estimates_are_bounded_without_allocating_large_sources() {
    // Exercise the resource model on a real small source but a requested huge output.
    let rect = Rect::new(0, 0, 16, 16);
    let src = fixture(rect, 4, true, false);
    let ctx = context(rect, true);
    assert!(Plan::new(&src, Rect::new(0, 0, 60000, 40000), &ctx, 100.0, RadialMethod::Spin, (0.5, 0.5), DEFAULT_PITCH).is_none());
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let plan = Plan::new(&src, rect, &ctx, 100.0, method, (0.5, 0.5), DEFAULT_PITCH).expect("small plan");
        assert!(plan.working_bytes <= WORKING_LIMIT);
        assert!(plan.remap_cells <= CELL_LIMIT as usize);
    }
}

#[test]
fn cancels_during_remap_and_before_output_allocation() {
    let rect = Rect::new(-64, -64, 64, 64);
    let src = fixture(rect, 4, true, false);
    let ctx = context(rect, true);
    let cancel = || true;
    let ctl = Interrupt::cancel_only(&cancel);
    assert!(filter(&src, rect, &ctx, 100.0, RadialMethod::Spin, (0.5, 0.5), &ctl).is_none());
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let calls = AtomicUsize::new(0);
        let cancel = || calls.fetch_add(1, Ordering::Relaxed) > 5;
        let ctl = Interrupt::cancel_only(&cancel);
        assert!(filter(&src, rect, &ctx, 100.0, method, (0.5, 0.5), &ctl).is_none());
        assert!(calls.load(Ordering::Relaxed) < 512, "bounded interruption checks");
    }
}

#[test]
fn zero_amount_preserves_visible_hdr_and_transparent_black() {
    let rect = Rect::new(-11, 7, 20, 24);
    let src = fixture(rect, 4, true, false);
    let ctx = context(rect, true);
    let got = filter(&src, rect, &ctx, 0.0, RadialMethod::Spin, (0.5, 0.5), &Interrupt::NONE).expect("identity");
    let (rmse, max) = error(got, src.data.clone(), 4, true);
    assert!(rmse < 1e-7 && max < 2e-7, "identity rmse {rmse}, max {max}");
}

#[test]
fn prefix_fractional_endpoints_and_periodic_wrap_are_exact_for_linear_nodes() {
    let line = Line { values: vec![0.0, 2.0, 4.0, 0.0], prefix: vec![0.0, 1.0, 4.0, 6.0], channels: 1, steps: 3 };
    assert!((line.integral(0.5, 0) - 0.25).abs() < 1e-12);
    assert!((line.integral(1.5, 0) - 2.25).abs() < 1e-12);
    assert!((line.circular_integral(-0.5, 0) + 0.5).abs() < 1e-12);
    assert!((line.circular_integral(3.5, 0) - 6.25).abs() < 1e-12);
}

/// Run only on request in release, for example:
/// `cargo test -p photocraft-algo --release polar_24mp_pipeline_comparison -- --ignored --nocapture`
/// This is a one-repeat experiment; it deliberately does not publish a hardware-independent
/// performance claim or alter a tracked baseline. The output includes actual pipeline times and
/// quality against both legacy sampling and an independent dense Cartesian oracle.
#[test]
#[ignore = "24 MP release performance/quality comparison; explicitly requested only"]
fn polar_24mp_pipeline_comparison() {
    use std::time::Instant;

    use photocraft_color::PixelFormat;
    use photocraft_raster::Surface;

    let rect = Rect::new(0, 0, 6000, 4000);
    let image = fixture(rect, 4, true, true);
    let mut surface = Surface::new(PixelFormat::RGBA32F);
    surface.write_region(rect, &image.data);
    drop(image);
    let ctx = context(rect, true);
    let center = (0.5, 0.5);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let params = crate::FilterParams::RadialBlur { quality: crate::RadialQuality::Draft, amount: 100.0, method, center_x: center.0, center_y: center.1 };
        let prepared = Image::read(&surface, rect);
        let plan = Plan::new(&prepared, rect, &ctx, 100.0, method, center, DEFAULT_PITCH).expect("24 MP centered plan");
        eprintln!("polar 24MP plan {method:?}: pitch={DEFAULT_PITCH} remap_cells={} working_bytes={} band={}", plan.remap_cells, plan.working_bytes, plan.band);
        drop(prepared);
        let began = Instant::now();
        let direct = crate::apply_radial_direct(&surface, &params, rect, rect, None, rect, &Interrupt::NONE).expect("direct input");
        let direct_seconds = began.elapsed().as_secs_f64();
        let began = Instant::now();
        let polar = crate::apply_radial_polar_with(&surface, &params, rect, rect, None, rect, &Interrupt::NONE).expect("24 MP bounded polar pipeline");
        let polar_seconds = began.elapsed().as_secs_f64();
        let mut square = 0.0;
        let mut maximum = 0.0f32;
        for y in rect.y0..rect.y1 {
            let row = Rect::new(rect.x0, y, rect.x1, y + 1);
            let mut d = direct.read_region(row);
            let mut p = polar.read_region(row);
            premultiply(&mut d, 4, true);
            premultiply(&mut p, 4, true);
            for (d, p) in d.iter().zip(&p) {
                let delta = (d - p).abs();
                square += f64::from(delta).powi(2);
                maximum = maximum.max(delta);
            }
        }
        let rmse = (square / (6000.0 * 4000.0 * 4.0)).sqrt();
        eprintln!(
            "polar 24MP pipeline {method:?}: direct={direct_seconds:.6}s polar={polar_seconds:.6}s ratio={:.3}x full_sparse_vs_dense_rmse={rmse:.7} max={maximum:.7}",
            direct_seconds / polar_seconds
        );
        let reference_source = Image::read(&surface, rect);
        let positions = [
            (3000, 2000),
            (3001, 2000),
            (20, 20),
            (1000, 700),
            (5100, 3200),
            (5900, 2000),
            (3000, 3900),
            (1700, 2000),
            (3000, 950),
            (20, 2000),
            (5999, 0),
            (0, 3999),
            (4400, 2900),
            (875, 3275),
            (2950, 1950),
            (3050, 2050),
        ];
        let (mut dense_direct, mut dense_polar, mut dense_legacy) = (Vec::new(), Vec::new(), Vec::new());
        let mut reference = Vec::new();
        for (x, y) in positions {
            let out = Rect::new(x, y, x + 1, y + 1);
            reference.extend(dense_reference(&reference_source, out, &ctx, 100.0, method, center));
            dense_direct.extend(direct.pixel(x, y));
            dense_polar.extend(polar.pixel(x, y));
            dense_legacy.extend(crate::blur::radial::reference(&reference_source, out, &ctx, 100.0, method, center));
        }
        let direct_error = error(dense_direct, reference.clone(), 4, true);
        let polar_error = error(dense_polar, reference.clone(), 4, true);
        let legacy_error = error(dense_legacy, reference, 4, true);
        eprintln!(
            "polar 24MP sampled dense oracle {method:?}: legacy_rmse={:.7} legacy_max={:.7} direct_rmse={:.7} direct_max={:.7} polar_rmse={:.7} polar_max={:.7}",
            legacy_error.0, legacy_error.1, direct_error.0, direct_error.1, polar_error.0, polar_error.1
        );
        assert!(polar_error.0 < 0.018 && polar_error.1 < 0.12, "{method:?}: sampled dense quality {polar_error:?}");
    }
}

/// Creates inspectable, noncommittable synthetic comparison images only when explicitly run.
/// The environment destination should be an ignored target/ subdirectory in this checkout.
#[test]
#[ignore = "native release visual quality artifacts, explicitly requested only"]
#[cfg(not(target_arch = "wasm32"))]
fn quality_artifacts() {
    use std::path::Path;
    use std::time::Instant;

    use photocraft_color::PixelFormat;
    use photocraft_raster::Surface;

    let destination = std::env::var("PHOTOCRAFT_RADIAL_QUALITY_DIR").expect("set PHOTOCRAFT_RADIAL_QUALITY_DIR to an ignored target path");
    let destination = Path::new(&destination);
    std::fs::create_dir_all(destination).expect("create quality artifact directory");
    let rect = Rect::new(0, 0, 512, 384);
    let source = Image { rect, ch: 4, data: fixture(Rect::new(-256, -192, 256, 192), 4, true, true).data };
    let ctx = context(rect, true);
    let mut surface = Surface::new(PixelFormat::RGBA32F);
    surface.write_region(rect, &source.data);
    let center = (0.5, 0.5);
    write_ppm(&destination.join("source.ppm"), &source.data, 512, 384);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let label = match method {
            RadialMethod::Spin => "spin",
            RadialMethod::Zoom => "zoom",
        };
        let params = crate::FilterParams::RadialBlur { quality: crate::RadialQuality::Draft, amount: 100.0, method, center_x: center.0, center_y: center.1 };
        let before = Instant::now();
        let legacy = crate::apply_radial_reference(&surface, &params, rect, rect, None, rect, &Interrupt::NONE).expect("legacy image");
        let direct = crate::apply_radial_direct(&surface, &params, rect, rect, None, rect, &Interrupt::NONE).expect("direct input");
        let polar = crate::apply_radial_polar_with(&surface, &params, rect, rect, None, rect, &Interrupt::NONE).expect("polar image");
        let reference = dense_reference_with_density(&source, rect, &ctx, 100.0, method, center, 12.0);
        eprintln!("quality artifacts {method:?}: reference density=12 samples/source-pixel, elapsed={:.3}s", before.elapsed().as_secs_f64());
        for (kind, values) in
            [("legacy", legacy.read_region(rect)), ("direct", direct.read_region(rect)), ("polar", polar.read_region(rect)), ("dense", reference.clone())]
        {
            let (rmse, maximum) = error(values.clone(), reference.clone(), 4, true);
            eprintln!("quality artifacts {method:?} {kind}: premultiplied rmse={rmse:.7} max={maximum:.7}");
            write_ppm(&destination.join(format!("{label}_{kind}.ppm")), &values, 512, 384);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_ppm(path: &std::path::Path, values: &[f32], width: usize, height: usize) {
    // Composite every result by the same alpha over a neutral checkerboard. Pixels outside the
    // source remain transparent; the display makes the edge/alpha effect visible consistently.
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for (i, pixel) in values.as_chunks::<4>().0.iter().enumerate() {
        let (x, y) = (i % width, i / width);
        let background = if ((x / 16) + (y / 16)).is_multiple_of(2) { 0.35 } else { 0.55 };
        let alpha = pixel[3].clamp(0.0, 1.0);
        for value in pixel.iter().take(3) {
            ppm.push(((value * alpha + background * (1.0 - alpha)).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    std::fs::write(path, ppm).expect("write quality PPM");
}
