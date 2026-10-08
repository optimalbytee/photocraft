use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

fn assert_samples_equal(actual: &[f32], expected: &[f32], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{case}, sample {i}: {a:?} != {b:?}");
    }
}

fn patterned_image(rect: Rect, ch: usize, alpha: bool, unusual: bool) -> Image {
    let mut data = Vec::new();
    for y in 0..rect.height() as usize {
        for x in 0..rect.width() as usize {
            for c in 0..ch {
                let k = x * 37 + y * 13 + c * 29;
                let value = if alpha && c + 1 == ch {
                    if unusual { [0.0, -0.0, 1.0, 0.125, 1e-8, -0.5, f32::NAN, f32::INFINITY][k % 8] } else { [0.0, 1.0, 0.125, 0.875][k % 4] }
                } else if unusual {
                    [-0.0, 0.0, -3.5, 8.25, 1e-12, f32::NAN, f32::INFINITY, f32::NEG_INFINITY][k % 8]
                } else {
                    (k % 256) as f32 / 255.0
                };
                data.push(value);
            }
        }
    }
    Image { rect, ch, data }
}

#[test]
fn motion_matches_direct_samples_and_absolute_coordinate_rounding() {
    let cases = [
        (0.0, 0.0),
        (33.0, 0.49),
        (0.0, 0.5),
        (0.0, 1.0),
        (90.0, 2.0),
        (180.0, 3.0),
        (-90.0, 6.0),
        (-41.25, -7.25),
        (29.5, 15.5),
        (45.0, 11.0),
        (360.0, 7.0),
        (-359.875, 4.75),
        (1e30, 5.0),
        (f32::NAN, 3.0),
        (f32::INFINITY, 3.0),
        (30.0, f32::NAN),
    ];
    for (x0, y0) in [(0, 0), (-17, 31), (65_534, -65_538), (1 << 20, -(1 << 20)), (8_388_604, -8_388_620)] {
        let rect = Rect::new(x0, y0, x0 + 29, y0 + 23);
        for ch in 1..=5 {
            for alpha in [false, true] {
                let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha };
                for unusual in [false, true] {
                    let src = patterned_image(rect, ch, alpha, unusual);
                    // Include both an interior rectangle and reads past the source edges.
                    for out in [Rect::new(x0 + 8, y0 + 9, x0 + 15, y0 + 14), Rect::new(x0 - 2, y0 - 1, x0 + 3, y0 + 3)] {
                        for (angle, distance) in cases {
                            let expected = motion_direct(&src, out, &ctx, angle, distance);
                            let actual = motion(&src, out, &ctx, angle, distance);
                            assert_samples_equal(
                                &actual,
                                &expected,
                                &format!("origin=({x0},{y0}), ch={ch}, alpha={alpha}, unusual={unusual}, out={out:?}, angle={angle}, distance={distance}"),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn motion_empty_rows_columns_and_long_distance_match_direct() {
    let rect = Rect::new(-3, -4, 6, 5);
    let src = patterned_image(rect, 4, true, false);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    for out in [Rect::EMPTY, Rect::new(-2, -1, 5, 0), Rect::new(0, -3, 1, 4), Rect::new(1, 1, 2, 2)] {
        for (angle, distance) in [(0.0, 5.5), (90.0, 5.5), (17.0, 128.0), (-63.0, 2000.0)] {
            assert_samples_equal(
                &motion(&src, out, &ctx, angle, distance),
                &motion_direct(&src, out, &ctx, angle, distance),
                &format!("out={out:?}, angle={angle}, distance={distance}"),
            );
        }
    }
}

#[test]
fn motion_identity_preserves_signed_zero_and_transparent_colours() {
    let rect = Rect::new(-1, -1, 2, 2);
    let src = Image { rect, ch: 4, data: [-0.0, 8.0, -2.0, 0.0].repeat(9) };
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    for distance in [-0.49, -0.0, 0.0, 0.49] {
        assert_samples_equal(&motion(&src, rect, &ctx, 17.0, distance), &src.data, &format!("distance={distance}"));
    }
    let blurred = motion(&src, rect, &ctx, 31.0, 2.0);
    assert!(blurred.iter().all(|value| *value == 0.0));
}

#[test]
fn motion_uncommon_channel_counts_retain_direct_fallback() {
    let rect = Rect::new(-3, 7, 5, 13);
    let out = Rect::new(-1, 9, 2, 11);
    for ch in [6, 8, 16] {
        for alpha in [false, true] {
            let src = patterned_image(rect, ch, alpha, false);
            let ctx = Ctx { bounds: rect, mode: ColorMode::Multichannel, alpha };
            assert_samples_equal(&motion(&src, out, &ctx, 23.0, 3.25), &motion_direct(&src, out, &ctx, 23.0, 3.25), &format!("ch={ch}, alpha={alpha}"));
        }
    }
}

fn direct_surface(surface: &Surface, params: &FilterParams, area: Rect, bounds: Rect, selection: Option<&Surface>, extent: Option<Rect>) -> Surface {
    let FilterParams::MotionBlur { angle, distance } = params else { panic!("motion-only oracle") };
    let crate::Halo::Radius(reach) = params.halo() else { panic!("motion uses a local halo") };
    let src = match extent {
        Some(e) => Image::read_clamped(surface, area.inflate(reach), e),
        None => Image::read(surface, area.inflate(reach)),
    };
    let format = surface.format();
    let ctx = Ctx { bounds, mode: format.mode, alpha: format.alpha };
    let mut data = motion_direct(&src, area, &ctx, *angle, *distance);
    if let Some(selection) = selection {
        crate::mix_selection(&mut data, area, selection, &src);
    }
    let mut result = surface.clone();
    result.write_region(area, &data);
    result.prune();
    result
}

#[test]
fn motion_pipeline_matches_direct_across_formats_edges_selections_and_tiles() {
    let rect = Rect::new(-13, 9, 16, 30);
    let mut selection = Surface::new(PixelFormat::GRAY8);
    let mut coverage = Vec::new();
    for y in 0..rect.height() as usize {
        for x in 0..rect.width() as usize {
            coverage.push([0.0, 0.25, 0.5, 0.75, 1.0][(x + y * 3) % 5]);
        }
    }
    selection.write_region(rect, &coverage);
    for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let format = PixelFormat::new(mode, sample, alpha);
                let mut source = Surface::new(format);
                let mut image = patterned_image(rect, source.channels(), alpha, false);
                if sample == SampleType::F32 {
                    for pixel in image.data.chunks_exact_mut(image.ch) {
                        for value in pixel.iter_mut().take(image.ch - usize::from(alpha)) {
                            *value = *value * 4.0 - 1.0;
                        }
                    }
                }
                source.write_region(rect, &image.data);
                for params in [
                    FilterParams::MotionBlur { angle: 0.0, distance: 2.5 },
                    FilterParams::MotionBlur { angle: 32.0, distance: 11.0 },
                    FilterParams::MotionBlur { angle: -90.0, distance: -7.25 },
                ] {
                    for extent in [None, Some(rect)] {
                        for mask in [None, Some(&selection)] {
                            let expected = direct_surface(&source, &params, rect, rect, mask, extent);
                            for tile in [5, 17, 256] {
                                let actual = crate::apply_tiled(&source, &params, rect, rect, mask, tile, extent);
                                assert_eq!(
                                    actual.to_interleaved(rect.inflate(2)),
                                    expected.to_interleaved(rect.inflate(2)),
                                    "mode={mode:?}, sample={sample:?}, alpha={alpha}, params={params:?}, extent={extent:?}, selection={}, tile={tile}",
                                    mask.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
