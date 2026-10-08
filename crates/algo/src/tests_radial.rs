use super::*;
use photocraft_color::PixelFormat;
use photocraft_raster::Interrupt;

#[test]
fn radial_public_calls_decline_huge_sources_before_allocating() {
    let source = Surface::new(PixelFormat::RGBA8);
    let bounds = Rect::new(0, 0, 100_000, 100_000);
    let area = Rect::new(0, 0, 1, 1);
    let params = FilterParams::RadialBlur { quality: crate::RadialQuality::Good, amount: 100.0, method: RadialMethod::Spin, center_x: 0.5, center_y: 0.5 };
    assert!(apply_in_with(&source, &params, area, bounds, None, area, &Interrupt::NONE).is_none());
    assert!(apply_radial_polar_with(&source, &params, area, bounds, None, area, &Interrupt::NONE).is_none());
    let cancelled = || true;
    assert!(apply_in_with(&source, &params, area, bounds, None, area, &Interrupt::cancel_only(&cancelled)).is_none());
}

#[test]
fn radial_polar_public_api_clips_output_and_preserves_selection_and_depth() {
    let extent = Rect::new(-12, -10, 31, 27);
    let area = Rect::new(0, 0, 20, 18);
    let params = FilterParams::RadialBlur { quality: crate::RadialQuality::Good, amount: 100.0, method: RadialMethod::Zoom, center_x: 0.5, center_y: 0.5 };
    let selection = Surface::new(PixelFormat::GRAY8);
    for depth in [photocraft_color::SampleType::U8, photocraft_color::SampleType::U16, photocraft_color::SampleType::F32] {
        let mut source = Surface::new(PixelFormat::new(ColorMode::Cmyk, depth, true));
        source.fill_rect(extent, &[0.2, 0.7, 0.1, 0.4, 0.6]);
        source.fill_rect(Rect::new(0, 0, 10, 18), &[0.9, 0.1, 0.6, 0.2, 0.3]);
        let selected = apply_radial_polar_with(&source, &params, area, area, Some(&selection), extent, &Interrupt::NONE).unwrap();
        assert_eq!(selected.format(), source.format());
        assert_eq!(selected.read_region(extent), source.read_region(extent));
        let filtered = apply_radial_polar_with(&source, &params, area, area, None, extent, &Interrupt::NONE).unwrap();
        assert_eq!(filtered.pixel(-3, -4), source.pixel(-3, -4));
        assert_eq!(filtered.pixel(25, 22), source.pixel(25, 22));
        let feather = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, photocraft_color::SampleType::F32, false), &[0.37]);
        let mixed = apply_radial_polar_with(&source, &params, area, area, Some(&feather), extent, &Interrupt::NONE).unwrap();
        let tolerance = match depth {
            photocraft_color::SampleType::U8 => 2.0 / 255.0,
            photocraft_color::SampleType::U16 => 2.0 / 65535.0,
            photocraft_color::SampleType::F32 => 1e-6,
        };
        let mut changed = false;
        for ((original, blurred), selected) in source.read_region(area).iter().zip(filtered.read_region(area)).zip(mixed.read_region(area)) {
            changed |= (original - blurred).abs() > 0.01;
            let expected = original + (blurred - original) * 0.37;
            assert!((selected - expected).abs() <= tolerance, "{depth:?}: feathered selection {selected} vs {expected}");
        }
        assert!(changed, "exercise a nonidentity blur");
    }
}

#[test]
fn radial_default_selects_polar_across_models_depths_selection_and_tile_sizes() {
    let extent = Rect::new(-8, -6, 56, 42);
    let area = Rect::new(0, 0, 40, 32);
    let selection = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, photocraft_color::SampleType::F32, false), &[0.37]);
    for depth in [photocraft_color::SampleType::U8, photocraft_color::SampleType::U16, photocraft_color::SampleType::F32] {
        for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
            let format = PixelFormat::new(mode, depth, true);
            let mut source = Surface::new(format);
            let mut values = Vec::new();
            for y in extent.y0..extent.y1 {
                for x in extent.x0..extent.x1 {
                    for channel in 0..format.channels() {
                        let value = (x * 11 + y * 17 + channel as i32 * 23).rem_euclid(97) as f32 / 96.0;
                        values.push(if channel + 1 == format.channels() { 0.25 + 0.75 * value } else { value });
                    }
                }
            }
            source.write_region(extent, &values);
            for method in [RadialMethod::Spin, RadialMethod::Zoom] {
                let params = FilterParams::RadialBlur { quality: crate::RadialQuality::Good, amount: 100.0, method, center_x: 0.5, center_y: 0.5 };
                let expected =
                    apply_radial_polar_with(&source, &params, area, area, Some(&selection), extent, &Interrupt::NONE).expect("supported polar input");
                let default = apply_in_with(&source, &params, area, area, Some(&selection), extent, &Interrupt::NONE).expect("default input");
                assert_eq!(default.format(), format);
                assert_eq!(default.read_region(extent), expected.read_region(extent), "{format:?}, {method:?}: select polar");
                let direct = apply_radial_direct(&source, &params, area, area, Some(&selection), extent, &Interrupt::NONE).expect("direct input");
                assert_ne!(default.read_region(area), direct.read_region(area), "exercise a different dense approximation");
                for tile in [1, 16, 256] {
                    let tiled = apply_tiled_with(&source, &params, area, area, Some(&selection), tile, None, &Interrupt::NONE).expect("explicit tile size");
                    assert_eq!(tiled.read_region(extent), expected.read_region(extent), "{format:?}, {method:?}: tile {tile}");
                }
            }
        }
    }
}

#[test]
fn radial_default_falls_back_for_unsupported_geometry_and_negative_alpha() {
    let area = Rect::new(-16, -12, 16, 12);
    let mut source = Surface::new(PixelFormat::RGBA32F);
    source.fill_rect(area, &[0.2, 0.7, 1.4, 0.6]);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        for (center_x, negative_alpha) in [(1000.0, false), (0.5, true)] {
            let mut input = source.clone();
            if negative_alpha {
                input.fill_rect(Rect::new(-2, -2, 2, 2), &[0.8, -0.3, 1.7, -0.4]);
            }
            let params = FilterParams::RadialBlur { quality: crate::RadialQuality::Good, amount: 100.0, method, center_x, center_y: 0.5 };
            assert!(apply_radial_polar_with(&input, &params, area, area, None, area, &Interrupt::NONE).is_none());
            let expected = apply_radial_direct(&input, &params, area, area, None, area, &Interrupt::NONE).expect("direct fallback");
            let actual = apply_in_with(&input, &params, area, area, None, area, &Interrupt::NONE).expect("automatic fallback");
            assert_eq!(actual.read_region(area), expected.read_region(area), "{method:?}, negative alpha {negative_alpha}");
        }
    }
}

#[test]
fn radial_default_stops_when_polar_is_cancelled() {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    let area = Rect::new(-64, -64, 64, 64);
    let mut source = Surface::new(PixelFormat::RGBA32F);
    source.fill_rect(area, &[0.2, 0.7, 1.4, 0.6]);
    let original = source.read_region(area);
    for method in [RadialMethod::Spin, RadialMethod::Zoom] {
        let flag = AtomicBool::new(false);
        let last = AtomicU32::new(0);
        let cancel = || flag.load(Ordering::Relaxed);
        let progress = |fraction: f32| {
            last.store(fraction.to_bits(), Ordering::Relaxed);
            flag.store(true, Ordering::Relaxed);
        };
        let params = FilterParams::RadialBlur { quality: crate::RadialQuality::Good, amount: 100.0, method, center_x: 0.5, center_y: 0.5 };
        assert!(apply_in_with(&source, &params, area, area, None, area, &Interrupt::new(&cancel, &progress)).is_none());
        assert!(flag.load(Ordering::Relaxed), "cancel during polar work");
        assert!(f32::from_bits(last.load(Ordering::Relaxed)) < 1.0, "cancel before writeback");
        assert_eq!(source.read_region(area), original);
    }
}

#[test]
fn draft_and_best_keep_their_direct_quality_paths() {
    let area = Rect::new(0, 0, 48, 40);
    let mut source = Surface::new(PixelFormat::RGBA32F);
    let values: Vec<f32> = (0..area.height())
        .flat_map(|y| (0..area.width()).flat_map(move |x| [(x % 3) as f32 / 2.0, (y % 5) as f32 / 4.0, ((x ^ y) % 7) as f32 / 6.0, 0.8]))
        .collect();
    source.write_region(area, &values);
    for quality in [RadialQuality::Draft, RadialQuality::Best] {
        for method in [RadialMethod::Spin, RadialMethod::Zoom] {
            let params = FilterParams::RadialBlur { amount: 100.0, method, quality, center_x: 0.5, center_y: 0.5 };
            let expected = apply_radial_direct(&source, &params, area, area, None, area, &Interrupt::NONE).expect("direct quality");
            let actual = apply_in_with(&source, &params, area, area, None, area, &Interrupt::NONE).expect("quality route");
            assert_eq!(actual.read_region(area), expected.read_region(area), "{quality:?}, {method:?}");
        }
    }
}
