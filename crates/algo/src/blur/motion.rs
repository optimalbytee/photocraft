//! Motion sampling geometry shared by each output column and row. The original
//! tap order and floating-point expressions are retained, including the per-tap
//! unpremultiply / premultiply round trip.

use photocraft_geom::Rect;

use crate::Image;

const MAX_BUFFER_BYTES: usize = 128 << 20;
const MAX_TABLE_BYTES: usize = 64 << 20;
const MAX_DISTANCE: f32 = 2000.0;

#[derive(Clone, Copy)]
struct Axis {
    lo: u32,
    hi: u32,
    fraction: f32,
}

fn dimensions(rect: Rect) -> Option<(usize, usize)> {
    Some((usize::try_from(i64::from(rect.x1) - i64::from(rect.x0)).ok()?, usize::try_from(i64::from(rect.y1) - i64::from(rect.y0)).ok()?))
}

fn buffer<T: Clone>(count: usize, limit: usize, value: T) -> Option<Vec<T>> {
    if count.checked_mul(std::mem::size_of::<T>())? > limit {
        return None;
    }
    let mut data = Vec::new();
    data.try_reserve_exact(count).ok()?;
    data.resize(count, value);
    Some(data)
}

#[allow(clippy::too_many_arguments)]
fn axis(start: i32, end: i32, source_start: i32, source_end: i32, stride: usize, outside: usize, offsets: &[f32], subtract: bool) -> Option<Vec<Axis>> {
    let width = usize::try_from(i64::from(end) - i64::from(start)).ok()?;
    let count = width.checked_mul(offsets.len())?;
    let outside = u32::try_from(outside).ok()?;
    let mut axis = buffer(count, MAX_TABLE_BYTES, Axis { lo: outside, hi: outside, fraction: 0.0 })?;
    for (coordinate, samples) in (start..end).zip(axis.chunks_exact_mut(offsets.len())) {
        let center = coordinate as f32 + 0.5;
        for (&offset, sample) in offsets.iter().zip(samples) {
            // Keep these two operations separate, exactly as Image::sample does.
            let position = if subtract { center - offset } else { center + offset };
            let position = position - 0.5;
            let floor = position.floor();
            let lo = floor as i64;
            let index = |coordinate: i64| -> Option<u32> {
                if coordinate < i64::from(source_start) || coordinate >= i64::from(source_end) {
                    return Some(outside);
                }
                u32::try_from(usize::try_from(coordinate - i64::from(source_start)).ok()?.checked_mul(stride)?).ok()
            };
            *sample = Axis { lo: index(lo)?, hi: index(lo + 1)?, fraction: position - floor };
        }
    }
    Some(axis)
}

pub(super) fn filter(src: &Image, out: Rect, alpha: bool, angle: f32, distance: f32) -> Option<Vec<f32>> {
    if out.is_empty() {
        return Some(Vec::new());
    }
    let d = distance.abs();
    if !(0.5..=MAX_DISTANCE).contains(&d) || !angle.is_finite() || !(1..=5).contains(&src.ch) {
        return None;
    }
    // The reference sampler uses i32 neighbour coordinates. Within this range
    // neither its float-to-integer cast nor its neighbour addition can overflow.
    const COORDINATE_LIMIT: i32 = 1 << 29;
    if [src.rect.x0, src.rect.y0, src.rect.x1, src.rect.y1, out.x0, out.y0, out.x1, out.y1]
        .iter()
        .any(|&v| !(-COORDINATE_LIMIT..=COORDINATE_LIMIT).contains(&v))
    {
        return None;
    }
    let (sw, sh) = dimensions(src.rect)?;
    let pixels = sw.checked_mul(sh)?;
    // Invalid neighbours use `pixels` as their index on either axis. Their sum
    // must fit in u32 too; compact offsets keep the tables at 12 bytes per tap.
    if pixels.checked_mul(src.ch)? != src.data.len() || pixels > u32::MAX as usize / 2 || sw == 0 || sh == 0 {
        return None;
    }
    let (ow, oh) = dimensions(out)?;
    let mut result = buffer(ow.checked_mul(oh)?.checked_mul(src.ch)?, MAX_BUFFER_BYTES, 0.0)?;
    let steps = d.ceil() as i32;
    let (s, c) = angle.to_radians().sin_cos();
    let xs: Vec<f32> = (0..=steps).map(|i| c * d * (i as f32 / steps as f32 - 0.5)).collect();
    let ys: Vec<f32> = (0..=steps).map(|i| s * d * (i as f32 / steps as f32 - 0.5)).collect();
    let x = axis(out.x0, out.x1, src.rect.x0, src.rect.x1, 1, pixels, &xs, false)?;
    let y = axis(out.y0, out.y1, src.rect.y0, src.rect.y1, sw, pixels, &ys, true)?;
    macro_rules! run {
        ($channels:literal) => {
            if alpha {
                sample::<$channels, true>(&src.data, &mut result, &x, &y, xs.len(), ow);
            } else {
                sample::<$channels, false>(&src.data, &mut result, &x, &y, xs.len(), ow);
            }
        };
    }
    match src.ch {
        1 => run!(1),
        2 => run!(2),
        3 => run!(3),
        4 => run!(4),
        5 => run!(5),
        _ => return None,
    }
    Some(result)
}

fn sample<const N: usize, const ALPHA: bool>(src: &[f32], result: &mut [f32], x: &[Axis], y: &[Axis], taps: usize, width: usize) {
    let pixels = src.as_chunks::<N>().0;
    let colours = if ALPHA { N - 1 } else { N };
    let norm = 1.0 / taps as f32;
    for (row, ys) in result.chunks_exact_mut(width * N).zip(y.chunks_exact(taps)) {
        for (dst, xs) in row.as_chunks_mut::<N>().0.iter_mut().zip(x.chunks_exact(taps)) {
            let mut acc = [0.0; N];
            for (x, y) in xs.iter().zip(ys) {
                let (ax, ay) = (x.fraction, y.fraction);
                let mut value = [0.0; N];
                let mut acc_a = 0.0;
                for (index, weight) in
                    [(x.lo + y.lo, (1.0 - ax) * (1.0 - ay)), (x.hi + y.lo, ax * (1.0 - ay)), (x.lo + y.hi, (1.0 - ax) * ay), (x.hi + y.hi, ax * ay)]
                {
                    if weight <= 0.0 {
                        continue;
                    }
                    let pixel = pixels.get(index as usize).unwrap_or(&[0.0; N]);
                    let a = if ALPHA { pixel[N - 1] } else { 1.0 };
                    for c in 0..colours {
                        value[c] += pixel[c] * a * weight;
                    }
                    acc_a += a * weight;
                }
                if ALPHA {
                    if acc_a > 0.0 {
                        for v in value.iter_mut().take(colours) {
                            *v /= acc_a;
                        }
                    }
                    value[N - 1] = acc_a;
                }
                let a = if ALPHA { value[N - 1] } else { 1.0 };
                for c in 0..N {
                    acc[c] += if ALPHA && c < colours { value[c] * a } else { value[c] };
                }
            }
            for v in &mut acc {
                *v *= norm;
            }
            if ALPHA {
                let a = acc[N - 1];
                for v in acc.iter_mut().take(colours) {
                    *v = if a > 1e-7 { *v / a } else { 0.0 };
                }
            }
            *dst = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_fast_path_checks_layout_geometry_and_allocation_budgets() {
        let rect = Rect::new(0, 0, 2, 2);
        let src = Image::new(rect, 4);
        let malformed = Image { rect, ch: 4, data: vec![1.0] };
        assert!(filter(&malformed, rect, true, 30.0, 10.0).is_none());
        for (angle, distance) in [(f32::NAN, 10.0), (f32::INFINITY, 10.0), (30.0, f32::NAN), (30.0, f32::INFINITY), (30.0, 2001.0)] {
            assert!(filter(&src, rect, true, angle, distance).is_none());
        }
        assert!(filter(&src, Rect::new(0, 0, 9000, 9000), true, 30.0, 10.0).is_none());
        assert!(filter(&src, Rect::new(i32::MIN, 0, i32::MAX, 1), true, 30.0, 10.0).is_none());
        assert!(filter(&src, Rect::EMPTY, true, 30.0, 10.0).unwrap().is_empty());
        assert!(axis(0, 10_000_000, 0, 2, 1, 4, &[0.0], false).is_none());
        assert!(filter(&src, rect, true, 30.0, 10.0).is_some());
    }
}
