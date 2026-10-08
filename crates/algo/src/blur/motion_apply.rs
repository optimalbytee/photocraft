//! Bounded tile scheduling for wide Motion Blur convolution.

use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};

use super::motion_fft::{COORDINATE_LIMIT, Plan, ordinary_coordinates, worthwhile};
use crate::{Ctx, FilterParams, Image};

const MAX_RESIDENT_BYTES: usize = 512 << 20;
const MIN_TILE: i32 = 64;
const MAX_TILE: i32 = 512;
const FALLBACK_ROWS: i32 = 8;

fn valid_rect(rect: Rect) -> bool {
    !rect.is_empty() && rect.width() <= COORDINATE_LIMIT as u32 && rect.height() <= COORDINATE_LIMIT as u32 && ordinary_coordinates(rect)
}

fn bytes(rect: Rect, channels: usize) -> Option<usize> {
    usize::try_from(rect.width()).ok()?.checked_mul(usize::try_from(rect.height()).ok()?)?.checked_mul(channels)?.checked_mul(std::mem::size_of::<f32>())
}

fn group_size(resident: usize, metadata: usize, worker: usize, output: usize, tiles: usize, workers: usize) -> Option<usize> {
    if tiles == 0 || workers == 0 {
        return None;
    }
    let available = MAX_RESIDENT_BYTES.checked_sub(resident)?.checked_sub(metadata)?;
    let per_tile = worker.checked_add(output)?;
    let memory_workers = available.checked_div(per_tile.max(1))?;
    (memory_workers > 0).then_some(memory_workers.min(workers).min(tiles))
}

fn fallback(src: &Image, out: Rect, ctx: &Ctx, angle: f32, distance: f32, ctl: &Interrupt) -> Option<Vec<f32>> {
    let count = bytes(out, src.ch)?.checked_div(std::mem::size_of::<f32>())?;
    let mut result = Vec::new();
    result.try_reserve_exact(count).ok()?;
    let mut y = out.y0;
    while y < out.y1 {
        if ctl.cancelled() {
            return None;
        }
        let y1 = y.checked_add(FALLBACK_ROWS)?.min(out.y1);
        let row = Rect::new(out.x0, y, out.x1, y1);
        let data = super::motion(src, row, ctx, angle, distance);
        if ctl.cancelled() || data.len().checked_mul(std::mem::size_of::<f32>())? != bytes(row, src.ch)? {
            return None;
        }
        result.extend_from_slice(&data);
        y = y1;
    }
    Some(result)
}

/// `None` declines the specialization; `Some(None)` propagates cancellation. Input pixels are
/// never edited, including when cancellation discards a partially written output clone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    surface: &Surface,
    params: &FilterParams,
    area: Rect,
    bounds: Rect,
    selection: Option<&Surface>,
    extent: Option<Rect>,
    tile: i32,
    ctl: &Interrupt,
) -> Option<Option<Surface>> {
    let FilterParams::MotionBlur { angle, distance } = *params else { return None };
    if ctl.cancelled() {
        return Some(None);
    }
    if area.is_empty() {
        return Some(Some(surface.clone()));
    }
    let fmt = surface.format();
    let channels = fmt.channels();
    if !(1..=5).contains(&channels)
        || !valid_rect(area)
        || !valid_rect(bounds)
        || extent.is_some_and(|rect| !valid_rect(rect))
        // read_clamped falls back to raw reads when its window misses the
        // extent. A directional and square halo can differ in that case;
        // preserve the old boundary policy for out-of-canvas output areas.
        || extent.is_some_and(|rect| area.intersect(&rect) != area)
        || selection.is_some_and(|selection| selection.channels() == 0)
    {
        return None;
    }
    let pixels = u64::from(area.width()).checked_mul(u64::from(area.height()))?;
    if pixels < 4096 {
        return None;
    }
    let mut tile = tile.clamp(MIN_TILE, MAX_TILE);
    while tile > MIN_TILE && (area.width() < tile as u32 || area.height() < tile as u32) {
        tile = (tile / 2).max(MIN_TILE);
    }
    let tile_usize = usize::try_from(tile).ok()?;
    if !worthwhile(angle, distance, tile_usize, area)? {
        return None;
    }
    let plan = match Plan::new_with(angle, distance, tile_usize, ctl) {
        Some(plan) => plan,
        None if ctl.cancelled() => return Some(None),
        None => return None,
    };
    if ctl.cancelled() {
        return Some(None);
    }
    let columns = usize::try_from(area.width()).ok()?.div_ceil(tile_usize);
    let rows = usize::try_from(area.height()).ok()?.div_ceil(tile_usize);
    let count = columns.checked_mul(rows)?;
    let metadata_bytes = count.checked_mul(std::mem::size_of::<(Rect, Rect)>())?;
    let mut tiles = Vec::new();
    tiles.try_reserve_exact(count).ok()?;
    let mut largest_source_bytes = 0;
    let mut y = area.y0;
    while y < area.y1 {
        let y1 = y.checked_add(tile)?.min(area.y1);
        let mut x = area.x0;
        while x < area.x1 {
            let x1 = x.checked_add(tile)?.min(area.x1);
            let output = Rect::new(x, y, x1, y1);
            let source = plan.read_rect(output)?;
            if !valid_rect(source) {
                return None;
            }
            largest_source_bytes = largest_source_bytes.max(bytes(source, channels)?);
            tiles.push((output, source));
            x = x1;
        }
        y = y1;
    }
    // read_clamped temporarily holds both its inner read and expanded image. The convolution
    // estimate includes its input and scratch, but that decoding peak needs a separate check.
    let decode_bytes = largest_source_bytes.checked_mul(if extent.is_some() { 2 } else { 1 })?;
    let worker_bytes = plan.worker_bytes(channels, tile_usize)?.max(decode_bytes);
    let output_bytes = tile_usize.checked_mul(tile_usize)?.checked_mul(channels)?.checked_mul(std::mem::size_of::<f32>())?;
    #[cfg(not(target_arch = "wasm32"))]
    let workers = rayon::current_num_threads();
    #[cfg(target_arch = "wasm32")]
    let workers = 1;
    // A group's results remain live until its last worker finishes. Bounding the complete
    // group by each tile's scratch plus output is conservative and avoids forcing one per core.
    let group = group_size(plan.resident_bytes(), metadata_bytes, worker_bytes, output_bytes, tiles.len(), workers)?;
    let ctx = Ctx { bounds, mode: fmt.mode, alpha: fmt.alpha };
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let run = |&(output, source): &(Rect, Rect)| -> Option<(Rect, Vec<f32>)> {
        if ctl.cancelled() {
            return None;
        }
        let src = match extent {
            Some(extent) => Image::read_clamped(surface, source, extent),
            None => Image::read(surface, source),
        };
        if ctl.cancelled() {
            return None;
        }
        let mut data = match plan.filter(&src, output, fmt.alpha, ctl) {
            Some(data) => data,
            None if ctl.cancelled() => return None,
            // Unsupported samples retain the original sampling path. In particular, do not
            // mistake cancellation for a reason to start the uncancellable direct kernel.
            None => fallback(&src, output, &ctx, angle, distance, ctl)?,
        };
        if ctl.cancelled() || data.len().checked_mul(std::mem::size_of::<f32>())? != bytes(output, channels)? {
            return None;
        }
        if let Some(selection) = selection {
            crate::mix_selection(&mut data, output, selection, &src);
        }
        if ctl.cancelled() {
            return None;
        }
        let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        ctl.progress(0.98 * done as f32 / count as f32);
        Some((output, data))
    };
    let mut out = surface.clone();
    for chunk in tiles.chunks(group) {
        if ctl.cancelled() {
            return Some(None);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let results: Option<Vec<_>> = {
            use rayon::prelude::*;
            chunk.par_iter().map(run).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let results: Option<Vec<_>> = chunk.iter().map(run).collect();
        let Some(results) = results else { return if ctl.cancelled() { Some(None) } else { None } };
        for (output, data) in results {
            if ctl.cancelled() {
                return Some(None);
            }
            out.write_region(output, &data);
        }
    }
    if ctl.cancelled() {
        return Some(None);
    }
    out.prune();
    if ctl.cancelled() {
        return Some(None);
    }
    ctl.progress(1.0);
    Some(Some(out))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use photocraft_color::PixelFormat;

    use super::*;

    #[test]
    fn scheduler_bounds_complete_groups_including_sources_and_outputs() {
        let mib = 1 << 20;
        let group = group_size(100 * mib, mib, 150 * mib, 5 * mib, 100, 12).unwrap();
        assert_eq!(group, 2, "memory limits must take precedence over one tile per worker");
        assert!(100 * mib + mib + group * (150 * mib + 5 * mib) <= MAX_RESIDENT_BYTES);
        assert_eq!(group_size(100 * mib, 0, 450 * mib, 5 * mib, 100, 12), None);
        assert_eq!(group_size(MAX_RESIDENT_BYTES, 1, 1, 1, 1, 1), None);
        assert_eq!(group_size(0, 0, usize::MAX, 1, 1, 1), None);
        assert_eq!(group_size(0, 0, 1, 1, 0, 1), None);
    }

    #[test]
    fn scheduler_preflight_declines_unsupported_geometry_and_small_distances() {
        let surface = Surface::new(PixelFormat::GRAY8);
        let bounds = Rect::new(0, 0, 128, 128);
        let small = FilterParams::MotionBlur { angle: 30.0, distance: 10.0 };
        assert!(apply(&surface, &small, bounds, bounds, None, Some(bounds), 64, &Interrupt::NONE).is_none());
        let wide = FilterParams::MotionBlur { angle: 30.0, distance: 64.0 };
        let oversized = Rect::new(0, 0, i32::MAX, 128);
        assert!(apply(&surface, &wide, oversized, bounds, None, Some(bounds), 64, &Interrupt::NONE).is_none());
        assert!(apply(&surface, &wide, bounds, bounds, None, Some(Rect::EMPTY), 64, &Interrupt::NONE).is_none());
        assert!(apply(&surface, &wide, Rect::new(0, 0, 1, 1), bounds, None, Some(bounds), 512, &Interrupt::NONE).is_none());
        let huge = FilterParams::MotionBlur { angle: 30.0, distance: 2000.0 };
        assert!(apply(&surface, &huge, bounds, bounds, None, Some(bounds), 512, &Interrupt::NONE).is_none());
        assert!(!valid_rect(Rect::new(-COORDINATE_LIMIT, 0, COORDINATE_LIMIT, 1)));
        assert!(apply(&surface, &wide, Rect::EMPTY, bounds, None, Some(bounds), 64, &Interrupt::NONE).unwrap().unwrap().tiles().next().is_none());
    }

    #[test]
    fn scheduler_cancellation_discards_all_partial_output() {
        let bounds = Rect::new(0, 0, 128, 128);
        let mut surface = Surface::new(PixelFormat::GRAY8);
        surface.fill_rect(bounds, &[0.4]);
        let before = surface.to_interleaved(bounds);
        let params = FilterParams::MotionBlur { angle: 0.0, distance: 64.0 };
        let cancelled = AtomicBool::new(true);
        let cancel = || cancelled.load(Ordering::Relaxed);
        assert!(matches!(apply(&surface, &params, bounds, bounds, None, Some(bounds), 64, &Interrupt::cancel_only(&cancel)), Some(None)));
        cancelled.store(false, Ordering::Relaxed);
        let progress = |fraction: f32| {
            if fraction > 0.0 {
                cancelled.store(true, Ordering::Relaxed);
            }
        };
        let ctl = Interrupt::new(&cancel, &progress);
        assert!(matches!(apply(&surface, &params, bounds, bounds, None, Some(bounds), 64, &ctl), Some(None)));
        assert_eq!(surface.to_interleaved(bounds), before);
    }
}
