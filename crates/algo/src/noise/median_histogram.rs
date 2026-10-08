//! Exact sliding-disc Median for samples representable as `byte / 255.0`.

use photocraft_geom::Rect;

use crate::image::Image;

// Bound scratch allocations for callers of the public kernel as well as normal tiles.
const MAX_BUFFER_BYTES: usize = 128 << 20;
const MAX_RADIUS: i32 = 500;

fn sample_count(rect: Rect, channels: usize, bytes: usize) -> Option<usize> {
    let count = (rect.width() as usize).checked_mul(rect.height() as usize)?.checked_mul(channels)?;
    (channels > 0 && count.checked_mul(bytes)? <= MAX_BUFFER_BYTES).then_some(count)
}

struct ByteImage {
    rect: Rect,
    width: usize,
    channels: usize,
    data: Vec<u8>,
}

impl ByteImage {
    fn read(src: &Image) -> Option<Self> {
        let count = sample_count(src.rect, src.ch, 1)?;
        if count != src.data.len() {
            return None;
        }
        let mut data = Vec::new();
        data.try_reserve_exact(count).ok()?;
        for &value in &src.data {
            let level = (value * 255.0).round();
            let byte = level as u8;
            // Comparing bits rejects NaN, HDR, signed zero and values between levels.
            if !(0.0..=255.0).contains(&level) || (f32::from(byte) / 255.0).to_bits() != value.to_bits() {
                return None;
            }
            data.push(byte);
        }
        Some(Self { rect: src.rect, width: src.rect.width() as usize, channels: src.ch, data })
    }

    #[inline]
    fn get(&self, x: i64, y: i64, channel: usize) -> u8 {
        if x < i64::from(self.rect.x0) || x >= i64::from(self.rect.x1) || y < i64::from(self.rect.y0) || y >= i64::from(self.rect.y1) {
            return 0;
        }
        // Layout was checked and capped in read(); valid coordinates keep this below count.
        let row = (y - i64::from(self.rect.y0)) as usize;
        let column = (x - i64::from(self.rect.x0)) as usize;
        let index = (row * self.width + column) * self.channels + channel;
        self.data.get(index).copied().unwrap_or(0)
    }
}

struct Histogram {
    counts: [u32; 256],
    pivot: usize,
    below: u32,
}

impl Histogram {
    #[inline]
    fn change(&mut self, level: u8, add: bool) -> Option<()> {
        let count = self.counts.get_mut(usize::from(level))?;
        *count = if add { count.checked_add(1)? } else { count.checked_sub(1)? };
        if usize::from(level) < self.pivot {
            self.below = if add { self.below.checked_add(1)? } else { self.below.checked_sub(1)? };
        }
        Some(())
    }

    fn median(&mut self, rank: u32) -> Option<f32> {
        while self.below > rank {
            self.pivot = self.pivot.checked_sub(1)?;
            self.below = self.below.checked_sub(*self.counts.get(self.pivot)?)?;
        }
        while self.below.checked_add(*self.counts.get(self.pivot)?)? <= rank {
            self.below = self.below.checked_add(*self.counts.get(self.pivot)?)?;
            self.pivot = self.pivot.checked_add(1)?;
        }
        Some(self.pivot as f32 / 255.0)
    }
}

/// Reuse a disc histogram between neighbouring output pixels, including at row ends.
/// Returns None if the samples, dimensions or allocation budget require the direct kernel.
pub(super) fn filter(src: &Image, out: Rect, radius: i32, threshold: Option<f32>) -> Option<Vec<f32>> {
    if !(1..=MAX_RADIUS).contains(&radius) {
        return None;
    }
    let count = sample_count(out, src.ch, std::mem::size_of::<f32>())?;
    if out.is_empty() {
        return Some(Vec::new());
    }
    let samples = ByteImage::read(src)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count).ok()?;
    result.resize(count, 0.0);
    // radius <= 500 makes all footprint arithmetic exact and bounded. A span is the
    // same dx² + dy² <= r² + r footprint as the direct kernel, not a square window.
    let mut spans = Vec::new();
    spans.try_reserve_exact((2 * radius + 1) as usize).ok()?;
    let mut population = 0u32;
    for delta in -radius..=radius {
        let half = f64::from(radius * radius + radius - delta * delta).sqrt().floor() as i64;
        spans.push((i64::from(delta), half));
        population = population.checked_add((2 * half + 1) as u32)?;
    }
    let rank = population / 2;
    let width = out.width() as usize;
    let height = out.height() as usize;
    let threshold = threshold.map(|value| value / 255.0);
    for channel in 0..src.ch {
        let mut histogram = Histogram { counts: [0; 256], pivot: 0, below: 0 };
        let mut x = i64::from(out.x0);
        let mut y = i64::from(out.y0);
        for &(dy, half) in &spans {
            for dx in -half..=half {
                histogram.change(samples.get(x + dx, y + dy, channel), true)?;
            }
        }
        // Serpentine traversal keeps the histogram alive when moving down a row,
        // avoiding another O(radius²) initialization per row and channel.
        for row in 0..height {
            let direction = if row % 2 == 0 { 1 } else { -1 };
            for step in 0..width {
                let median = histogram.median(rank)?;
                let original = f32::from(samples.get(x, y, channel)) / 255.0;
                let column = (x - i64::from(out.x0)) as usize;
                let index = (row * width + column) * src.ch + channel;
                *result.get_mut(index)? = match threshold {
                    Some(t) if (original - median).abs() <= t => original,
                    _ => median,
                };
                if step + 1 < width {
                    for &(dy, half) in &spans {
                        histogram.change(samples.get(x - direction * half, y + dy, channel), false)?;
                        histogram.change(samples.get(x + direction * (half + 1), y + dy, channel), true)?;
                    }
                    x += direction;
                }
            }
            if row + 1 < height {
                // The footprint is symmetric, so the same spans describe columns.
                for &(dx, half) in &spans {
                    histogram.change(samples.get(x + dx, y - half, channel), false)?;
                    histogram.change(samples.get(x + dx, y + half + 1, channel), true)?;
                }
                y += 1;
            }
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{median, median_direct};
    use crate::{FilterParams, Halo, apply_tiled_with, mix_selection};
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    use photocraft_raster::{Interrupt, Surface};

    fn pattern(rect: Rect, channels: usize, sample: SampleType) -> Image {
        let mut seed = 0x072e_1905u32;
        let data = (0..rect.width() as usize * rect.height() as usize * channels)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                match sample {
                    SampleType::U8 => (seed >> 24) as f32 / 255.0,
                    SampleType::U16 => (seed >> 16) as f32 / 65535.0,
                    SampleType::F32 if i % 29 == 0 => {
                        [-f32::INFINITY, -3.0, -0.0, 0.0, 3.0, f32::INFINITY, f32::from_bits(0xffc0_0001), f32::from_bits(0x7fc0_0002)][i % 8]
                    }
                    SampleType::F32 => (seed >> 8) as f32 / 16777215.0 * 4.0 - 1.0,
                }
            })
            .collect();
        Image { rect, ch: channels, data }
    }

    fn assert_exact(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "sample {index}");
        }
    }

    #[test]
    fn median_matches_direct_at_every_depth_and_threshold() {
        for rect in [Rect::new(0, 0, 24, 19), Rect::new(-7, 4, 17, 23)] {
            for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
                for channels in [1, 4, 5] {
                    let src = pattern(rect, channels, sample);
                    for out in [rect, Rect::new(rect.x0 + 3, rect.y0 + 5, rect.x1 - 4, rect.y1 - 3), rect.inflate(2)] {
                        for radius in [0.0, 1.0, 2.0, 4.0, 8.0] {
                            for threshold in [None, Some(10.0), Some(255.0)] {
                                let expected = median_direct(&src, out, radius, threshold);
                                assert_exact(&median(&src, out, radius, threshold), &expected);
                                if sample == SampleType::U8 && radius > 0.0 {
                                    assert_exact(&filter(&src, out, radius as i32, threshold).unwrap(), &expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn histogram_rejects_non_byte_values_without_quantizing() {
        let rect = Rect::new(0, 0, 2, 2);
        for value in [-0.0, -0.1, 0.12345, 1.001, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let src = Image { rect, ch: 1, data: vec![0.0, 1.0, value, 51.0 / 255.0] };
            assert!(filter(&src, rect, 1, None).is_none());
            assert_exact(&median(&src, rect, 1.0, None), &median_direct(&src, rect, 1.0, None));
        }
        let signed_zero = Image { rect: Rect::new(0, 0, 3, 3), ch: 1, data: vec![-0.0; 9] };
        assert_exact(&median(&signed_zero, Rect::new(1, 1, 2, 2), 1.0, None), &[-0.0]);
        let src = Image { rect: Rect::new(0, 0, 256, 1), ch: 1, data: (0..256).map(|v| v as f32 / 255.0).collect() };
        assert_exact(&filter(&src, src.rect, 2, None).unwrap(), &median_direct(&src, src.rect, 2.0, None));
    }

    #[test]
    fn median_rounding_empty_and_single_dimension_outputs() {
        let src = pattern(Rect::new(-2, -3, 9, 8), 4, SampleType::U8);
        for out in [Rect::new(0, 0, 1, 5), Rect::new(0, 0, 5, 1), Rect::new(0, 0, 0, 0)] {
            for radius in [0.49, 0.5, 1.49, 1.5, 2.5, 40.0] {
                assert_exact(&median(&src, out, radius, Some(0.0)), &median_direct(&src, out, radius, Some(0.0)));
            }
        }
        let out = Rect::new(0, 0, 1, 1);
        assert_exact(&median(&src, out, 500.0, None), &median_direct(&src, out, 500.0, None));
    }

    #[test]
    fn histogram_keeps_disc_corners_alpha_and_threshold_equality() {
        let rect = Rect::new(-2, -2, 3, 3);
        let out = Rect::new(0, 0, 1, 1);
        let mut src = Image::new(rect, 2);
        // Exactly four edge-adjacent neighbours are dark; the four diagonal samples
        // must contribute to the radius-one disc's nine samples, making its median 1.
        src.data.fill(1.0);
        for (x, y) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
            let i = ((y + 2) * 5 + x + 2) as usize * 2;
            src.data[i] = 0.0;
        }
        src.data[12 * 2 + 1] = 0.0;
        assert_exact(&median(&src, out, 1.0, None), &[1.0, 1.0]);
        // Difference exactly equal to threshold keeps the original alpha sample.
        assert_exact(&median(&src, out, 1.0, Some(255.0)), &[1.0, 0.0]);
        assert_exact(&median(&src, out, 1.0, Some(254.0)), &[1.0, 1.0]);
        assert_exact(&median(&src, rect, 2.0, None), &median_direct(&src, rect, 2.0, None));
    }

    #[test]
    fn histogram_checks_layout_budget_and_extreme_coordinates() {
        let huge = Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX);
        assert!(sample_count(huge, 5, 4).is_none());
        assert!(sample_count(Rect::new(0, 0, 1, 1), usize::MAX, 4).is_none());
        let malformed = Image { rect: Rect::new(0, 0, 2, 2), ch: 1, data: vec![0.0] };
        assert!(filter(&malformed, malformed.rect, 1, None).is_none());
        assert!(filter(&malformed, huge, 1, None).is_none());
        assert!(filter(&malformed, malformed.rect, i32::MAX, None).is_none());
        for rect in [Rect::new(i32::MIN, i32::MIN, i32::MIN + 3, i32::MIN + 3), Rect::new(i32::MAX - 3, i32::MAX - 3, i32::MAX, i32::MAX)] {
            let src = Image { rect, ch: 1, data: vec![1.0; 9] };
            let actual = filter(&src, rect, 2, None).unwrap();
            let origin = Image { rect: Rect::new(0, 0, 3, 3), ch: 1, data: vec![1.0; 9] };
            assert_exact(&actual, &median_direct(&origin, origin.rect, 2.0, None));
        }
    }

    #[test]
    fn median_pipeline_matches_direct_with_tiles_edges_and_selections() {
        let area = Rect::new(-11, 7, 13, 26);
        let selection_data: Vec<f32> = (0..area.height()).flat_map(|y| (0..area.width()).map(move |x| ((x + y) % 9) as f32 / 8.0)).collect();
        let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
        selection.write_region(area, &selection_data);
        for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
            for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
                for alpha in [false, true] {
                    let fmt = PixelFormat::new(mode, sample, alpha);
                    let input = pattern(area, fmt.channels(), SampleType::U8);
                    let mut surface = Surface::new(fmt);
                    surface.write_region(area, &input.data);
                    for threshold in [None, Some(0.0), Some(10.0), Some(255.0)] {
                        let params = match threshold {
                            None => FilterParams::Median { radius: 2.5 },
                            Some(threshold) => FilterParams::DustAndScratches { radius: 2.5, threshold },
                        };
                        let Halo::Radius(halo) = params.halo() else { panic!("Median halo") };
                        for extent in [None, Some(area)] {
                            let src = match extent {
                                None => Image::read(&surface, area.inflate(halo)),
                                Some(e) => Image::read_clamped(&surface, area.inflate(halo), e),
                            };
                            for sel in [None, Some(&selection)] {
                                let mut expected_data = median_direct(&src, area, 2.5, threshold);
                                if let Some(sel) = sel {
                                    mix_selection(&mut expected_data, area, sel, &src);
                                }
                                let mut expected = surface.clone();
                                expected.write_region(area, &expected_data);
                                for tile in [5, 32] {
                                    let actual = apply_tiled_with(&surface, &params, area, area, sel, tile, extent, &Interrupt::NONE).unwrap();
                                    assert_eq!(actual.to_interleaved(area), expected.to_interleaved(area), "{mode:?}/{sample:?}/alpha{alpha}/tile{tile}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_direct(surface: &Surface, area: Rect, radius: f32) -> Surface {
        use rayon::prelude::*;

        let halo = radius.ceil() as i32 + 1;
        let tile = 256;
        let mut tiles = Vec::new();
        for y in (area.y0..area.y1).step_by(tile as usize) {
            for x in (area.x0..area.x1).step_by(tile as usize) {
                tiles.push(Rect::new(x, y, (x + tile).min(area.x1), (y + tile).min(area.y1)));
            }
        }
        let mut output = surface.clone();
        let per_tile = tile as usize * tile as usize * surface.channels() * std::mem::size_of::<f32>();
        let group = (crate::RESULT_BUDGET / per_tile).max(rayon::current_num_threads());
        for chunk in tiles.chunks(group) {
            let results: Vec<_> = chunk
                .par_iter()
                .map(|&t| {
                    let src = Image::read_clamped(surface, t.inflate(halo), area);
                    (t, median_direct(&src, t, radius, None))
                })
                .collect();
            for (t, data) in results {
                output.write_region(t, &data);
            }
        }
        output.prune();
        output
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "24MP release performance comparison; run with --release --ignored --nocapture"]
    fn median_24mp_performance() {
        use std::hint::black_box;
        use std::time::Instant;

        let rect = Rect::new(0, 0, 6000, 4000);
        let mut surface = Surface::new(PixelFormat::RGBA8);
        let data: Vec<u8> =
            (0..4000).flat_map(|y| (0..6000).flat_map(move |x| [(x % 256) as u8, ((x * 3 + y) / 11 % 256) as u8, ((x ^ y) % 256) as u8, 255])).collect();
        surface.write_interleaved(rect, &data);
        println!("Median comparison: 24MP RGBA8, median of 3, {} Rayon workers", rayon::current_num_threads());
        for radius in [1.0, 5.0] {
            let params = FilterParams::Median { radius };
            let mut medians = Vec::new();
            let mut expected = None;
            for direct in [true, false] {
                let mut times = Vec::new();
                for _ in 0..3 {
                    let start = Instant::now();
                    let output = if direct {
                        // Same tile IO, Rayon scheduling, writes and pruning as apply_in.
                        apply_direct(&surface, rect, radius)
                    } else {
                        crate::apply_in(&surface, &params, rect, rect, None, rect)
                    };
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                    let bytes = black_box(output.to_interleaved(rect));
                    if let Some(ref expected) = expected {
                        assert_eq!(&bytes, expected);
                    } else {
                        expected = Some(bytes);
                    }
                }
                times.sort_by(f64::total_cmp);
                medians.push(times[1]);
            }
            println!("radius{radius}: direct {:.3}ms, histogram {:.3}ms, {:.2}x; byte-identical", medians[0], medians[1], medians[0] / medians[1]);
        }
    }
}
