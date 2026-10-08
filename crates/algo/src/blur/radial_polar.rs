//! Default dense Radial Blur: bounded polar strips and reusable line integrals.
//!
//! This approximates a continuous, uniformly weighted arc/ray, **not** the existing inclusive
//! 65-sample kernel. The public filter pipeline falls back to Cartesian sampling for inputs
//! outside this backend's supported data, geometry or resource budgets.
//! The source remains straight; only the small polar strips are premultiplied. Interleaved
//! samples and line-local scans deliberately leave room for safe SIMD/blocked scans later.

use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

use photocraft_geom::Rect;
use photocraft_raster::Interrupt;

use crate::image::Image;
use crate::photo_util::{par_map, par_rows};
use crate::{Ctx, RadialMethod};

/// Subpixel remapping pitch. This is a quality parameter, not a requested blur sample ceiling.
pub(crate) const DEFAULT_PITCH: f64 = 0.75;
const SCRATCH_LIMIT: usize = 64 * 1024 * 1024;
const WORKING_LIMIT: usize = 1536 * 1024 * 1024;
const CELL_LIMIT: f64 = 400_000_000.0;
const MAX_BAND: usize = 16;
const ROWS_PER_TASK: usize = 8;
// For short paths, polar reconstruction error outweighs the benefit of reusable integrals.
// A small Cartesian quadrature has bounded work and preserves thin details near the centre.
const DIRECT_PATH: f64 = 16.0;

/// Validated geometry and upper bounds, prepared before allocating a polar strip.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    out: Rect,
    channels: usize,
    alpha: bool,
    center: (f64, f64),
    amount: f64,
    method: RadialMethod,
    pitch: f64,
    max_radius: f64,
    radial_steps: usize,
    angular_steps: usize,
    first_ring: usize,
    band: usize,
    /// Includes the caller's normalized source and this operation's output and scratch.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) working_bytes: usize,
    /// Upper bound on source samples used in the remap, including strip overlap.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) remap_cells: usize,
}

impl Plan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, center: (f32, f32), pitch: f64) -> Option<Self> {
        if src.ch == 0
            || out.is_empty()
            || src.rect.is_empty()
            || !amount.is_finite()
            || !center.0.is_finite()
            || !center.1.is_finite()
            || !pitch.is_finite()
            || !(0.25..=2.0).contains(&pitch)
        {
            return None;
        }
        let channels = src.ch;
        let source_len = (src.rect.width() as usize).checked_mul(src.rect.height() as usize)?.checked_mul(channels)?;
        let output_len = (out.width() as usize).checked_mul(out.height() as usize)?.checked_mul(channels)?;
        if src.data.len() != source_len {
            return None;
        }
        // This backend requires ordinary finite samples and nonnegative alpha. Negative/nonfinite
        // alpha has different semantics in the legacy sampler; decline rather than change it.
        if src.data.iter().any(|v| !v.is_finite())
            || (ctx.alpha
                && src.data.chunks_exact(channels).any(|p| p.last().is_none_or(|a| *a < 0.0 || p.iter().take(channels - 1).any(|v| !(v * a).is_finite()))))
        {
            return None;
        }
        // Retain the command's existing f32 centre calculation, including off-origin bounds.
        let b = ctx.bounds;
        let cx = b.x0 as f32 + b.width() as f32 * center.0;
        let cy = b.y0 as f32 + b.height() as f32 * center.1;
        if !cx.is_finite() || !cy.is_finite() {
            return None;
        }
        let center = (f64::from(cx), f64::from(cy));
        let (x0, x1, y0, y1) = (f64::from(out.x0) + 0.5, f64::from(out.x1) - 0.5, f64::from(out.y0) + 0.5, f64::from(out.y1) - 0.5);
        let max_radius = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)].into_iter().map(|(x, y)| (x - center.0).hypot(y - center.1)).fold(0.0, f64::max);
        if !max_radius.is_finite() || max_radius > 32768.0 {
            return None;
        }
        let min_radius = (center.0 - center.0.clamp(x0, x1)).hypot(center.1 - center.1.clamp(y0, y1));
        let radial_steps = (max_radius / pitch).floor() as usize + 1;
        // Multiples of four make axis-aligned wedge boundaries exact and avoid seam ambiguity.
        let angular_steps = (((TAU * radial_steps as f64).ceil() as usize).max(4).checked_add(3)? / 4).checked_mul(4)?;
        let first_ring = (min_radius / pitch).floor() as usize;
        let max_line = match method {
            RadialMethod::Spin => angular_steps.checked_add(1)?,
            RadialMethod::Zoom => radial_steps.checked_add(1)?,
        };
        // Each line has f32 remapped samples and f64 prefix sums. An extra line overlaps strips.
        // Also count line-local sampler/scan vectors, row accumulators and vector metadata.
        let line_bytes = max_line.checked_mul(channels)?.checked_mul(12)?.checked_add(channels.checked_mul(16)?)?.checked_add(128)?;
        let row_bytes = channels.checked_mul(16)?.checked_add(48)?.checked_mul(worker_count())?;
        let band = SCRATCH_LIMIT.checked_sub(row_bytes)?.checked_div(line_bytes)?.saturating_sub(1).min(MAX_BAND);
        if band == 0 {
            return None;
        }
        let base_cells = match method {
            RadialMethod::Spin => {
                let lo = first_ring as f64 * pitch;
                std::f64::consts::PI * ((max_radius + pitch).powi(2) - lo.powi(2)) / pitch.powi(2) + radial_steps as f64 * 5.0
            }
            RadialMethod::Zoom => angular_steps as f64 * (radial_steps + 1) as f64,
        };
        let cells = base_cells * (1.0 + 1.0 / band as f64);
        if !cells.is_finite() || cells > CELL_LIMIT {
            return None;
        }
        // Extreme off-canvas centres can turn a tiny output into a huge remap. The 65-tap
        // baseline is a conservative work reference, not the dense quality target.
        let pixels = output_len / channels;
        if cells > pixels as f64 * 65.0 + 4096.0 {
            return None;
        }
        let working_bytes = source_len.checked_add(output_len)?.checked_mul(4)?.checked_add(line_bytes.checked_mul(band + 1)?)?.checked_add(row_bytes)?;
        if working_bytes > WORKING_LIMIT {
            return None;
        }
        Some(Self {
            out,
            channels,
            alpha: ctx.alpha,
            center,
            amount: f64::from(amount.clamp(0.0, 100.0)),
            method,
            pitch,
            max_radius,
            radial_steps,
            angular_steps,
            first_ring,
            band,
            working_bytes,
            remap_cells: cells.ceil() as usize,
        })
    }

    pub(crate) fn run(&self, src: &Image, ctl: &Interrupt<'_>) -> Option<Vec<f32>> {
        if ctl.cancelled() {
            return None;
        }
        let len = (self.out.width() as usize).checked_mul(self.out.height() as usize)?.checked_mul(self.channels)?;
        let mut result = zeros::<f32>(len)?;
        if self.amount == 0.0 {
            let stride = (self.out.width() as usize).checked_mul(self.channels)?;
            let failed = AtomicBool::new(false);
            par_rows(&mut result, self.out.width() as usize * ROWS_PER_TASK, self.channels, |task, rows| {
                let Some(mut sample) = zeros::<f64>(self.channels) else {
                    failed.store(true, Ordering::Relaxed);
                    return;
                };
                for (k, row) in rows.chunks_exact_mut(stride).enumerate() {
                    if ctl.cancelled() {
                        return;
                    }
                    let y = i64::from(self.out.y0) + (task * ROWS_PER_TASK + k) as i64;
                    for (i, px) in row.chunks_exact_mut(self.channels).enumerate() {
                        let x = i64::from(self.out.x0) + i as i64;
                        sample_into(src, x as f64 + 0.5, y as f64 + 0.5, self.alpha, &mut sample);
                        finish(&sample, self.alpha, px);
                    }
                }
            });
            if failed.load(Ordering::Relaxed) {
                return None;
            }
        } else {
            match self.method {
                RadialMethod::Spin => self.spin(src, &mut result, ctl)?,
                RadialMethod::Zoom => self.zoom(src, &mut result, ctl)?,
            }
        }
        (!ctl.cancelled()).then_some(result)
    }

    fn spin(&self, src: &Image, result: &mut [f32], ctl: &Interrupt<'_>) -> Option<()> {
        let arc = self.amount.to_radians();
        let mut first = self.first_ring;
        while first < self.radial_steps {
            if ctl.cancelled() {
                return None;
            }
            let end = first.saturating_add(self.band).min(self.radial_steps);
            let lines: Option<Vec<Line>> = par_map(end - first + 1, |i| {
                let radius = (first + i) as f64 * self.pitch;
                let count = ((TAU * radius / self.pitch).ceil() as usize).max(4);
                Line::ring(src, self.channels, self.alpha, self.center, radius, count, ctl)
            })
            .into_iter()
            .collect();
            let lines = lines?;
            let low = first as f64 * self.pitch;
            let high = end as f64 * self.pitch;
            self.rows(result, ctl, |y, row, acc, sample| {
                let dy = f64::from(y) + 0.5 - self.center.1;
                if dy.abs() > high {
                    return;
                }
                let outer = (high * high - dy * dy).max(0.0).sqrt();
                let inner = (low * low - dy * dy).max(0.0).sqrt();
                let segments = if dy.abs() >= low {
                    [(self.center.0 - outer, self.center.0 + outer), (0.0, -1.0)]
                } else {
                    [(self.center.0 - outer, self.center.0 - inner), (self.center.0 + inner, self.center.0 + outer)]
                };
                for (left, right) in segments {
                    self.span(row, left, right, |x, px| {
                        let dx = f64::from(x) + 0.5 - self.center.0;
                        let radius = dx.hypot(dy);
                        let ring = (radius / self.pitch).floor() as usize;
                        if ring < first || ring >= end {
                            return;
                        }
                        if arc * radius <= DIRECT_PATH {
                            self.central(src, dx, dy, radius, acc, sample, px);
                            return;
                        }
                        let theta = dy.atan2(dx).rem_euclid(TAU);
                        let fraction = radius / self.pitch - ring as f64;
                        let (Some(a), Some(b)) = (lines.get(ring - first), lines.get(ring - first + 1)) else {
                            return;
                        };
                        for (c, v) in acc.iter_mut().enumerate() {
                            *v = a.arc_mean(theta, arc, c) * (1.0 - fraction) + b.arc_mean(theta, arc, c) * fraction;
                        }
                        finish(acc, self.alpha, px);
                    });
                }
            })?;
            first = end;
            ctl.progress(first as f32 / self.radial_steps.max(1) as f32);
        }
        Some(())
    }

    fn zoom(&self, src: &Image, result: &mut [f32], ctl: &Interrupt<'_>) -> Option<()> {
        let span = self.amount / 200.0;
        let quadrant = self.angular_steps / 4;
        let mut first = 0;
        while first < self.angular_steps {
            if ctl.cancelled() {
                return None;
            }
            // A strip never crosses an axis, allowing a simple row intersection of its rays.
            let end = first.saturating_add(self.band).min((first / quadrant + 1) * quadrant);
            let lines: Option<Vec<Line>> = par_map(end - first + 1, |i| {
                let (s, c) = direction(first + i, self.angular_steps);
                Line::ray(src, self.channels, self.alpha, self.center, (c, s), self.pitch, self.radial_steps, ctl)
            })
            .into_iter()
            .collect();
            let lines = lines?;
            let (s0, c0) = direction(first, self.angular_steps);
            let (s1, c1) = direction(end, self.angular_steps);
            self.rows(result, ctl, |y, row, acc, sample| {
                let dy = f64::from(y) + 0.5 - self.center.1;
                if dy.abs() > self.max_radius {
                    return;
                }
                let above = first < self.angular_steps / 2;
                if (dy > 0.0 && !above) || (dy < 0.0 && above) {
                    return;
                }
                let (left, right) = if dy == 0.0 {
                    if first != 0 && first != self.angular_steps / 2 {
                        return;
                    }
                    (self.center.0 - self.max_radius, self.center.0 + self.max_radius)
                } else {
                    let a = if s0 == 0.0 { c0.signum() * f64::INFINITY } else { dy * c0 / s0 };
                    let b = if s1 == 0.0 { c1.signum() * f64::INFINITY } else { dy * c1 / s1 };
                    (self.center.0 + a.min(b), self.center.0 + a.max(b))
                };
                self.span(row, left, right, |x, px| {
                    let dx = f64::from(x) + 0.5 - self.center.0;
                    let radius = dx.hypot(dy);
                    let angle_index = dy.atan2(dx).rem_euclid(TAU) * self.angular_steps as f64 / TAU;
                    let wedge = (angle_index.floor() as usize).min(self.angular_steps - 1);
                    if wedge < first || wedge >= end {
                        return;
                    }
                    if span * radius <= DIRECT_PATH {
                        self.central(src, dx, dy, radius, acc, sample, px);
                        return;
                    }
                    let fraction = angle_index - wedge as f64;
                    let (Some(a), Some(b)) = (lines.get(wedge - first), lines.get(wedge - first + 1)) else {
                        return;
                    };
                    let (lo, hi) = ((1.0 - span) * radius / self.pitch, radius / self.pitch);
                    for (c, v) in acc.iter_mut().enumerate() {
                        *v = a.mean(lo, hi, c) * (1.0 - fraction) + b.mean(lo, hi, c) * fraction;
                    }
                    finish(acc, self.alpha, px);
                });
            })?;
            first = end;
            ctl.progress(first as f32 / self.angular_steps.max(1) as f32);
        }
        Some(())
    }

    fn rows(&self, result: &mut [f32], ctl: &Interrupt<'_>, f: impl Fn(i32, &mut [f32], &mut [f64], &mut [f64]) + Send + Sync) -> Option<()> {
        let stride = self.out.width() as usize * self.channels;
        let failed = AtomicBool::new(false);
        par_rows(result, self.out.width() as usize * ROWS_PER_TASK, self.channels, |task, rows| {
            let (Some(mut acc), Some(mut sample)) = (zeros::<f64>(self.channels), zeros::<f64>(self.channels)) else {
                failed.store(true, Ordering::Relaxed);
                return;
            };
            for (k, row) in rows.chunks_exact_mut(stride).enumerate() {
                if ctl.cancelled() {
                    return;
                }
                let y = i64::from(self.out.y0) + (task * ROWS_PER_TASK + k) as i64;
                if let Ok(y) = i32::try_from(y) {
                    f(y, row, &mut acc, &mut sample);
                }
            }
        });
        (!failed.load(Ordering::Relaxed) && !ctl.cancelled()).then_some(())
    }

    /// Intersect a derived geometric row span; the two-pixel guard handles tangencies/rounding.
    /// Exact ring/wedge ownership is checked by the caller so the guard never double-writes.
    fn span(&self, row: &mut [f32], left: f64, right: f64, mut f: impl FnMut(i32, &mut [f32])) {
        if left > right {
            return;
        }
        let lo = ((left - 0.5).floor() - 2.0).max(f64::from(self.out.x0)).min(f64::from(self.out.x1)) as i64;
        let hi = ((right - 0.5).ceil() + 2.0).max(f64::from(self.out.x0)).min(f64::from(self.out.x1)) as i64;
        for x in lo..hi {
            let offset = (x - i64::from(self.out.x0)) as usize * self.channels;
            if let (Ok(x), Some(px)) = (i32::try_from(x), row.get_mut(offset..offset + self.channels)) {
                f(x, px);
            }
        }
    }

    /// Tiny polar paths are cheaper and more accurate sampled directly, avoiding the singularity.
    #[allow(clippy::too_many_arguments)]
    fn central(&self, src: &Image, dx: f64, dy: f64, radius: f64, acc: &mut [f64], sample: &mut [f64], px: &mut [f32]) {
        acc.fill(0.0);
        let length = match self.method {
            RadialMethod::Spin => self.amount.to_radians() * radius,
            RadialMethod::Zoom => self.amount / 200.0 * radius,
        };
        let steps = (length * 8.0).ceil().clamp(1.0, 128.0) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let (x, y) = match self.method {
                RadialMethod::Spin => {
                    let (s, c) = ((t - 0.5) * self.amount.to_radians()).sin_cos();
                    (self.center.0 + dx * c - dy * s, self.center.1 + dx * s + dy * c)
                }
                RadialMethod::Zoom => {
                    let scale = 1.0 - self.amount / 200.0 * t;
                    (self.center.0 + dx * scale, self.center.1 + dy * scale)
                }
            };
            sample_into(src, x, y, self.alpha, sample);
            let weight = if i == 0 || i == steps { 0.5 } else { 1.0 } / steps as f64;
            for (a, s) in acc.iter_mut().zip(sample.iter()) {
                *a += weight * s;
            }
        }
        finish(acc, self.alpha, px);
    }
}

/// Dense backend entry point. `None` means cancelled or declined by resource/data/geometry
/// checks. Production callers stop on cancellation and may fall back for declined inputs.
pub(crate) fn filter(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, center: (f32, f32), ctl: &Interrupt<'_>) -> Option<Vec<f32>> {
    Plan::new(src, out, ctx, amount, method, center, DEFAULT_PITCH)?.run(src, ctl)
}

fn worker_count() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

fn zeros<T: Default + Clone>(len: usize) -> Option<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).ok()?;
    data.resize(len, T::default());
    Some(data)
}

/// A piecewise-linear remapped line and its trapezoidal integral (grid-cell units). Prefix
/// sums use f64 so small differences of long, bright/HDR prefixes retain useful precision.
struct Line {
    values: Vec<f32>,
    prefix: Vec<f64>,
    channels: usize,
    steps: usize,
}

impl Line {
    #[allow(clippy::too_many_arguments)]
    fn ring(src: &Image, channels: usize, alpha: bool, center: (f64, f64), radius: f64, steps: usize, ctl: &Interrupt<'_>) -> Option<Self> {
        Self::build(channels, steps, ctl, |i, sample| {
            let (s, c) = if i == steps { (0.0, 1.0) } else { (i as f64 * TAU / steps as f64).sin_cos() };
            sample_into(src, center.0 + radius * c, center.1 + radius * s, alpha, sample);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ray(
        src: &Image,
        channels: usize,
        alpha: bool,
        center: (f64, f64),
        direction: (f64, f64),
        pitch: f64,
        steps: usize,
        ctl: &Interrupt<'_>,
    ) -> Option<Self> {
        Self::build(channels, steps, ctl, |i, sample| {
            let radius = i as f64 * pitch;
            sample_into(src, center.0 + radius * direction.0, center.1 + radius * direction.1, alpha, sample);
        })
    }

    fn build(channels: usize, steps: usize, ctl: &Interrupt<'_>, mut sample_at: impl FnMut(usize, &mut [f64])) -> Option<Self> {
        let len = steps.checked_add(1)?.checked_mul(channels)?;
        let mut values = zeros::<f32>(len)?;
        let mut prefix = zeros::<f64>(len)?;
        let mut sample = zeros::<f64>(channels)?;
        for (i, px) in values.chunks_exact_mut(channels).enumerate() {
            if i.is_multiple_of(256) && ctl.cancelled() {
                return None;
            }
            sample_at(i, &mut sample);
            for (v, s) in px.iter_mut().zip(&sample) {
                *v = *s as f32;
            }
        }
        let mut sum = zeros::<f64>(channels)?;
        for (i, pair) in values.chunks_exact(channels).zip(values.chunks_exact(channels).skip(1)).enumerate() {
            if i.is_multiple_of(256) && ctl.cancelled() {
                return None;
            }
            let dst = prefix.get_mut((i + 1) * channels..(i + 2) * channels)?;
            for (((d, s), a), b) in dst.iter_mut().zip(&mut sum).zip(pair.0).zip(pair.1) {
                *s += (f64::from(*a) + f64::from(*b)) * 0.5;
                *d = *s;
            }
        }
        Some(Self { values, prefix, channels, steps })
    }

    fn integral(&self, position: f64, channel: usize) -> f64 {
        let position = position.clamp(0.0, self.steps as f64);
        let index = position.floor() as usize;
        let offset = index * self.channels + channel;
        let base = self.prefix.get(offset).copied().unwrap_or(0.0);
        if index >= self.steps {
            return base;
        }
        let t = position - index as f64;
        let a = f64::from(self.values.get(offset).copied().unwrap_or(0.0));
        let b = f64::from(self.values.get(offset + self.channels).copied().unwrap_or(0.0));
        base + a * t + (b - a) * (0.5 * t * t)
    }

    fn circular_integral(&self, position: f64, channel: usize) -> f64 {
        let period = self.steps as f64;
        let turns = (position / period).floor();
        turns * self.integral(period, channel) + self.integral(position - turns * period, channel)
    }

    fn arc_mean(&self, theta: f64, arc: f64, channel: usize) -> f64 {
        let scale = self.steps as f64 / TAU;
        let mid = theta * scale;
        let span = arc * scale;
        (self.circular_integral(mid + span * 0.5, channel) - self.circular_integral(mid - span * 0.5, channel)) / span
    }

    fn mean(&self, lo: f64, hi: f64, channel: usize) -> f64 {
        if hi <= lo {
            return 0.0;
        }
        (self.integral(hi, channel) - self.integral(lo, channel)) / (hi - lo)
    }
}

/// Exact axis directions when a ray falls on a quadrant boundary.
fn direction(index: usize, steps: usize) -> (f64, f64) {
    match index.checked_mul(4).and_then(|v| v.checked_div(steps)) {
        Some(q) if index.is_multiple_of(steps / 4) => match q % 4 {
            0 => (0.0, 1.0),
            1 => (1.0, 0.0),
            2 => (0.0, -1.0),
            _ => (-1.0, 0.0),
        },
        _ => (index as f64 * TAU / steps as f64).sin_cos(),
    }
}

/// Fused transparent bilinear sampling into premultiplied f64 accumulators. Resolve tap
/// addresses once, then perform contiguous channel arithmetic. Never clamp HDR colour values.
fn sample_into(src: &Image, x: f64, y: f64, alpha: bool, out: &mut [f64]) {
    out.fill(0.0);
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (bx, by) = (fx.floor(), fy.floor());
    let (ax, ay) = (fx - bx, fy - by);
    if !bx.is_finite() || !by.is_finite() || bx < f64::from(i32::MIN) || bx >= f64::from(i32::MAX) || by < f64::from(i32::MIN) || by >= f64::from(i32::MAX) {
        return;
    }
    let (bx, by) = (bx as i32, by as i32);
    for (dx, dy, weight) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
        if weight <= 0.0 {
            continue;
        }
        let (Some(x), Some(y)) = (bx.checked_add(dx), by.checked_add(dy)) else {
            continue;
        };
        if !src.rect.contains(x, y) {
            continue;
        }
        let Some(offset) = ((i64::from(y) - i64::from(src.rect.y0)) as usize)
            .checked_mul(src.rect.width() as usize)
            .and_then(|v| v.checked_add((i64::from(x) - i64::from(src.rect.x0)) as usize))
            .and_then(|v| v.checked_mul(src.ch))
        else {
            continue;
        };
        let Some(end) = offset.checked_add(src.ch) else {
            continue;
        };
        let Some(pixel) = src.data.get(offset..end) else {
            continue;
        };
        let alpha_value = if alpha { pixel.last().copied().unwrap_or(0.0) } else { 1.0 };
        for (c, (dst, value)) in out.iter_mut().zip(pixel).enumerate() {
            let a = if alpha && c + 1 < src.ch { f64::from(alpha_value) } else { 1.0 };
            *dst += f64::from(*value) * a * weight;
        }
    }
}

fn finish(premultiplied: &[f64], alpha: bool, out: &mut [f32]) {
    let a = if alpha { premultiplied.last().copied().unwrap_or(0.0) } else { 1.0 };
    let channels = premultiplied.len();
    for (c, (dst, value)) in out.iter_mut().zip(premultiplied).enumerate() {
        *dst = if alpha && c + 1 < channels { if a > 1e-7 { (value / a) as f32 } else { 0.0 } } else { *value as f32 };
    }
}

#[cfg(test)]
mod tests;
