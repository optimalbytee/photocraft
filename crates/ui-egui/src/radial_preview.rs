//! Cancellable Radial Blur previews. Preparation, engine dispatch and compositing stay off the
//! native drawing thread. One request runs at a time; slider changes replace the pending request
//! and cancel the running one. Only the latest generation can replace the last complete image.
//!
//! Preview resolution and engine command semantics are unchanged. The proxy is cached by source
//! snapshot, revision and target; parameter changes reuse its copy-on-write tiles. wasm retains
//! the synchronous engine path, as the engine's background jobs do there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use photocraft_doc::{DocId, Document, LayerId};
use photocraft_engine::channel_cmds::{ChannelTarget, ChannelView};
use serde_json::Value;

use crate::PhotocraftApp;

pub(crate) const COMMAND: &str = "filter.blur.radialBlur";
const MAX_PREVIEW_PIXELS: u64 = crate::proxy::PREVIEW_PIXELS * 2;

#[derive(Clone)]
struct Source {
    doc: DocId,
    revision: u64,
    snapshot: Weak<Document>,
    active: Option<LayerId>,
    channel: ChannelTarget,
    target: Option<Value>,
    k: u32,
}

impl Source {
    fn same(&self, other: &Self) -> bool {
        self.doc == other.doc
            && self.revision == other.revision
            && self.snapshot.ptr_eq(&other.snapshot)
            && self.active == other.active
            && self.channel == other.channel
            && self.target == other.target
            && self.k == other.k
    }
}

#[derive(Clone)]
struct Request {
    source: Source,
    original: Arc<Document>,
    dialog: u64,
    params: Value,
    view: ChannelView,
    max_texture_side: u32,
    generation: u64,
}

impl Request {
    fn same(&self, other: &Self) -> bool {
        self.source.same(&other.source)
            && self.dialog == other.dialog
            && self.params == other.params
            && self.view == other.view
            && self.max_texture_side == other.max_texture_side
    }

    fn from_app(app: &PhotocraftApp, idx: usize) -> Option<Self> {
        if app.session.active_index() != Some(idx) {
            return None;
        }
        let dialog = app.ui.dialogs.iter().find(|d| d.fields.contains_key("__filter"))?;
        if dialog.fields.get("__command").and_then(Value::as_str) != Some(COMMAND) || dialog.fields.get("__preview").and_then(Value::as_bool) != Some(true) {
            return None;
        }
        let st = app.session.documents().get(idx)?;
        let params = app.with_mask_target(COMMAND, crate::filter_dialog::params_of(&dialog.fields));
        Some(Self {
            source: Source {
                doc: st.doc.id,
                revision: st.revision,
                snapshot: Arc::downgrade(&st.doc),
                active: st.active_layer,
                channel: st.channel_view.target,
                target: params.get("target").cloned(),
                k: crate::proxy::factor(&st.doc),
            },
            original: st.doc.clone(),
            dialog: dialog.id,
            params,
            view: st.channel_view.clone(),
            max_texture_side: app.radial_preview.max_texture_side.max(1),
            generation: 0,
        })
    }
}

#[derive(Clone)]
struct CachedProxy {
    source: Source,
    doc: Arc<Document>,
}

/// A complete frame. The buffer is already composited by the worker; only display conversion and
/// texture upload remain on the drawing thread.
#[derive(Clone)]
pub(crate) struct Frame {
    pub doc: DocId,
    pub revision: u64,
    pub generation: u64,
    pub k: u32,
    pub result: Arc<Document>,
    pub buffer: Arc<photocraft_compose::Buffer>,
    overlay: Option<Arc<egui::ColorImage>>,
    pub compute_ms: f64,
}

struct Completion {
    generation: u64,
    cache: Option<CachedProxy>,
    result: Result<Option<Frame>, String>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Running {
    cancel: Arc<AtomicBool>,
    rx: std::sync::mpsc::Receiver<Completion>,
    handle: std::thread::JoinHandle<()>,
}

/// Runtime scheduling state; it is neither document data nor persisted UI state.
#[derive(Default)]
pub(crate) struct RadialPreview {
    wanted: Option<Request>,
    generation: u64,
    started: Option<u64>,
    cache: Option<CachedProxy>,
    shown: Option<Frame>,
    max_texture_side: u32,
    overlay_texture: Option<(u64, egui::TextureHandle)>,
    #[cfg(not(target_arch = "wasm32"))]
    running: Option<Running>,
}

impl Drop for RadialPreview {
    fn drop(&mut self) {
        self.cancel_running();
    }
}

impl RadialPreview {
    fn cancel_running(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(running) = &self.running {
            running.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn request(&mut self, mut request: Request) {
        if self.wanted.as_ref().is_some_and(|old| old.same(&request)) {
            return;
        }
        self.cancel_running();
        // Completed images from another source/target must never be shown while this one runs.
        if self.wanted.as_ref().is_none_or(|old| {
            !old.source.same(&request.source) || old.dialog != request.dialog || old.view != request.view || old.max_texture_side != request.max_texture_side
        }) {
            self.shown = None;
            self.overlay_texture = None;
        }
        if self.cache.as_ref().is_some_and(|cache| !cache.source.same(&request.source)) {
            self.cache = None;
        }
        self.generation = self.generation.wrapping_add(1);
        request.generation = self.generation;
        self.wanted = Some(request);
    }

    fn clear(&mut self) {
        self.cancel_running();
        self.wanted = None;
        self.shown = None;
        self.overlay_texture = None;
        self.cache = None;
        self.started = None;
    }

    fn accept(&mut self, completion: Completion) -> Option<String> {
        if let Some(cache) = completion.cache
            && self.wanted.as_ref().is_some_and(|wanted| wanted.source.same(&cache.source))
        {
            self.cache = Some(cache);
        }
        if self.wanted.as_ref().is_none_or(|wanted| wanted.generation != completion.generation) {
            return None;
        }
        match completion.result {
            Ok(Some(frame)) => {
                self.shown = Some(frame);
                None
            }
            Ok(None) => None,
            Err(error) => Some(error),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll(&mut self) -> Option<String> {
        let result = self.running.as_ref().map(|running| running.rx.try_recv());
        match result {
            Some(Ok(completion)) => {
                // Sending is the worker's last operation, but join only once it has actually
                // exited: a drawing frame must never wait for a worker, including panic cleanup.
                if self.running.as_ref().is_some_and(|running| !running.handle.is_finished()) {
                    return self.accept(completion);
                }
                if let Some(running) = self.running.take()
                    && running.handle.join().is_err()
                {
                    return Some("the preview worker stopped unexpectedly".into());
                }
                self.accept(completion)
            }
            Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                if self.running.as_ref().is_some_and(|running| !running.handle.is_finished()) {
                    return None;
                }
                let failed = self.running.take().is_some_and(|running| running.handle.join().is_err());
                if failed { Some("the preview worker stopped unexpectedly".into()) } else { None }
            }
            _ => None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll(&mut self) -> Option<String> {
        None
    }

    fn start(&mut self, ctx: &egui::Context) -> Option<String> {
        #[cfg(not(target_arch = "wasm32"))]
        if self.running.is_some() {
            return None;
        }
        let request = self.wanted.as_ref()?;
        if self.started == Some(request.generation) {
            return None;
        }
        self.started = Some(request.generation);
        let request = request.clone();
        let cached = self.cache.clone().filter(|cached| cached.source.same(&request.source));
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_cancel = cancel.clone();
            let repaint = ctx.clone();
            match std::thread::Builder::new().name("photocraft-radial-preview".into()).spawn(move || {
                let completion = compute(request, cached, &worker_cancel);
                let _ = tx.send(completion);
                repaint.request_repaint();
            }) {
                Ok(handle) => {
                    self.running = Some(Running { cancel, rx, handle });
                    None
                }
                Err(error) => Some(format!("could not start the preview worker: {error}")),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let completion = compute(request, cached, &AtomicBool::new(false));
            let error = self.accept(completion);
            ctx.request_repaint();
            error
        }
    }

    fn pending(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        if self.running.is_some() {
            return true;
        }
        self.wanted.as_ref().is_some_and(|wanted| self.started != Some(wanted.generation))
    }
}

fn report(app: &mut PhotocraftApp, error: Option<String>) {
    if let Some(error) = error {
        log::warn!("Radial Blur preview: {error}");
        app.ui.status = format!("Radial Blur preview: {error}");
        app.ui.status_error = true;
    }
}

/// Cancel invalid generations every frame, including when the GPU canvas is not being drawn.
/// Call after dialog handling so closing or turning off Preview cancels in that same frame.
pub(crate) fn retain(app: &mut PhotocraftApp) {
    let current = app.session.active_index().and_then(|idx| Request::from_app(app, idx));
    let valid = match (&app.radial_preview.wanted, current.as_ref()) {
        (Some(wanted), Some(current)) => wanted.same(current),
        (None, _) => true,
        _ => false,
    };
    if !valid {
        let owns = app.filter_preview.as_ref().is_some_and(|preview| {
            app.radial_preview.shown.as_ref().is_some_and(|shown| preview.result.as_ref().is_some_and(|result| Arc::ptr_eq(result, &shown.result)))
        });
        if current.is_none() {
            app.radial_preview.clear();
        } else if let Some(current) = current {
            app.radial_preview.request(current);
        }
        if owns && app.radial_preview.shown.is_none() {
            app.filter_preview = None;
        }
    }
    let error = app.radial_preview.poll();
    report(app, error);
}

/// Schedule or reuse the latest request and return the last complete frame while it is pending.
pub(crate) fn update(app: &mut PhotocraftApp, idx: usize, ctx: &egui::Context) -> Option<Frame> {
    app.radial_preview.max_texture_side = u32::try_from(ctx.input(|input| input.max_texture_side)).unwrap_or(u32::MAX).clamp(1, 4096);
    let request = Request::from_app(app, idx)?;
    app.radial_preview.request(request);
    let error = app.radial_preview.poll();
    report(app, error);
    let error = app.radial_preview.start(ctx);
    report(app, error);
    if app.radial_preview.pending() {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
    app.radial_preview.shown.clone()
}

/// The overlay of the radial frame the canvas actually draws. Outer None means this is another
/// canvas image, so its normal channel overlay still applies; Some(None) is a plain composite.
pub(crate) fn overlay(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, key: u64) -> Option<Option<egui::TextureId>> {
    let frame = app.radial_preview.shown.as_ref()?;
    if key != (frame.doc.0 ^ (1u64 << 61))
        || app.session.documents().get(idx)?.doc.id != frame.doc
        || !app.filter_preview.as_ref().is_some_and(|preview| preview.result.as_ref().is_some_and(|result| Arc::ptr_eq(result, &frame.result)))
    {
        return None;
    }
    let Some(image) = &frame.overlay else { return Some(None) };
    if app.radial_preview.overlay_texture.as_ref().is_none_or(|(generation, _)| *generation != frame.generation) {
        let image = image.clone();
        match app.radial_preview.overlay_texture.as_mut() {
            Some((generation, texture)) if texture.size() == image.size => {
                texture.set(image, egui::TextureOptions::LINEAR);
                *generation = frame.generation;
            }
            _ => {
                let texture = ctx.load_texture(format!("radial-preview-overlay-{}", frame.doc.0), image, egui::TextureOptions::LINEAR);
                app.radial_preview.overlay_texture = Some((frame.generation, texture));
            }
        }
    }
    Some(app.radial_preview.overlay_texture.as_ref().map(|(_, texture)| texture.id()))
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("cancelled".into()) } else { Ok(()) }
}

fn compute(request: Request, cached: Option<CachedProxy>, cancel: &AtomicBool) -> Completion {
    let generation = request.generation;
    let mut cache = cached;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check(cancel)?;
        let started = crate::gpu_canvas::now_ms();
        if cache.is_none() {
            let k = request.source.k;
            if k == 0 {
                return Err("the preview scale is invalid".into());
            }
            let pixels = u64::from(request.original.size.width.div_ceil(k)).saturating_mul(u64::from(request.original.size.height.div_ceil(k)));
            if pixels > MAX_PREVIEW_PIXELS {
                return Err("the preview exceeds its pixel budget".into());
            }
            let proxy = prepare_proxy(&request.original, request.source.k);
            if proxy.size.area() > MAX_PREVIEW_PIXELS {
                return Err("the preview exceeds its pixel budget".into());
            }
            cache = Some(CachedProxy { source: request.source.clone(), doc: Arc::new(proxy) });
        }
        check(cancel)?;
        let Some(proxy) = cache.as_ref() else { return Err("the preview proxy is unavailable".into()) };
        let result = command_preview(&request, &proxy.doc, cancel)?;
        check(cancel)?;
        let buffer = flatten(&result, cancel)?;
        check(cancel)?;
        let overlay = channel_overlay(&result, &request.view, request.max_texture_side, cancel)?;
        check(cancel)?;
        Ok(Some(Frame {
            doc: request.source.doc,
            revision: request.source.revision,
            generation,
            k: request.source.k,
            result: Arc::new(result),
            buffer: Arc::new(buffer),
            overlay,
            compute_ms: crate::gpu_canvas::now_ms() - started,
        }))
    }))
    .unwrap_or_else(|_| Err("the preview failed with an internal error; the document is unchanged".into()));
    let result = if cancel.load(Ordering::Relaxed) { Ok(None) } else { result };
    Completion { generation, cache, result }
}

fn prepare_proxy(doc: &Document, k: u32) -> Document {
    let mut proxy = crate::proxy::proxy_document(doc, k);
    // The shared proxy helper scales alpha/spot channels but currently omits this temporary
    // channel. Filtering an unscaled Quick Mask on a reduced canvas would use the wrong geometry.
    if k > 1
        && let Some(quick) = &mut proxy.quick_mask
    {
        quick.surface = crate::proxy::downsample(&quick.surface, k);
    }
    proxy
}

fn channel_overlay(doc: &Document, view: &ChannelView, max_side: u32, cancel: &AtomicBool) -> Result<Option<Arc<egui::ColorImage>>, String> {
    check(cancel)?;
    let factor = doc.size.width.max(doc.size.height).div_ceil(max_side.max(1)).max(1);
    let Some(pixels) = crate::channel_view::render(doc, view, doc.bounds(), factor, false) else { return Ok(None) };
    check(cancel)?;
    let size = [doc.size.width.div_ceil(factor) as usize, doc.size.height.div_ceil(factor) as usize];
    Ok(Some(Arc::new(egui::ColorImage::new(size, pixels))))
}

/// The original engine command runs on an isolated proxy session, including selection, mask,
/// colour-channel restrictions, transparency locks and smart filters.
fn command_preview(request: &Request, proxy: &Document, cancel: &AtomicBool) -> Result<Document, String> {
    let mut session = photocraft_engine::Session::new();
    session.add_document(proxy.clone(), None);
    if let Some(id) = request.source.active {
        session.select_layer(id).map_err(|error| error.to_string())?;
    }
    if let Some(st) = session.active_mut() {
        st.channel_view = request.view.clone();
        st.channel_view.target = request.source.channel;
    }
    check(cancel)?;
    let params = crate::filter_dialog::preview_params(&request.params, request.source.k);
    let started = session.start(COMMAND, params).map_err(|error| error.to_string())?;
    #[cfg(not(target_arch = "wasm32"))]
    if let photocraft_engine::jobs::Started::Job(id) = started {
        loop {
            if cancel.load(Ordering::Relaxed) {
                session.cancel_job(id);
                // Wait on this scheduler thread, never on the UI. A newer request cannot start
                // a second filter until the cancelled worker has released its scratch buffers.
                session.join_cancelled_jobs();
                return Err("cancelled".into());
            }
            if let Some(event) = session.poll_jobs().into_iter().find(|event| event.id == id) {
                match event.outcome {
                    photocraft_engine::jobs::JobOutcome::Done(_) => break,
                    photocraft_engine::jobs::JobOutcome::Failed(error) => return Err(error),
                    photocraft_engine::jobs::JobOutcome::Cancelled => return Err("cancelled".into()),
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = started;
    check(cancel)?;
    session.active().map(|st| (*st.doc).clone()).ok_or_else(|| "the preview document is unavailable".into())
}

fn flatten(doc: &Document, cancel: &AtomicBool) -> Result<photocraft_compose::Buffer, String> {
    let count = usize::try_from(doc.size.area()).map_err(|_| "the preview is too large".to_string())?;
    if doc.size.area() > MAX_PREVIEW_PIXELS {
        return Err("the preview exceeds its pixel budget".into());
    }
    let mut px = Vec::new();
    px.try_reserve_exact(count).map_err(|error| format!("could not allocate the preview image: {error}"))?;
    check(cancel)?;
    photocraft_compose::render_bands(doc, doc.bounds(), 256, |band| {
        check(cancel)?;
        px.extend(band.px);
        Ok::<(), String>(())
    })?;
    Ok(photocraft_compose::Buffer { rect: doc.bounds(), px })
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, PixelFormat, SampleType, Size};
    use photocraft_geom::Rect;
    use serde_json::json;

    fn image(depth: SampleType) -> Arc<Document> {
        let mut doc = Document::with_background("Radial preview test", Size::new(32, 24), ColorMode::Rgb, depth, Color::TRANSPARENT);
        let mut values = Vec::new();
        for y in 0..24 {
            for x in 0..32 {
                values.extend([x as f32 / 31.0, y as f32 / 23.0, ((x * 7 + y * 11) % 19) as f32 / 18.0, (x % 7) as f32 / 6.0]);
            }
        }
        let bounds = doc.bounds();
        doc.layers[0].surface_mut().unwrap().write_region(bounds, &values);
        let mut selection = photocraft_raster::Surface::new(PixelFormat::GRAY8);
        selection.fill_rect(Rect::new(3, 2, 25, 18), &[0.5]);
        doc.selection = Some(selection);
        Arc::new(doc)
    }

    fn request(doc: &Arc<Document>, params: Value, k: u32) -> Request {
        Request {
            source: Source {
                doc: doc.id,
                revision: 1,
                snapshot: Arc::downgrade(doc),
                active: doc.top_layer(),
                channel: ChannelTarget::Composite,
                target: params.get("target").cloned(),
                k,
            },
            original: doc.clone(),
            dialog: 11,
            params,
            view: ChannelView::default(),
            max_texture_side: 4096,
            generation: 0,
        }
    }

    fn frame(request: &Request) -> Frame {
        Frame {
            doc: request.source.doc,
            revision: request.source.revision,
            generation: request.generation,
            k: request.source.k,
            result: request.original.clone(),
            buffer: Arc::new(photocraft_compose::Buffer::transparent(request.original.bounds())),
            overlay: None,
            compute_ms: 0.0,
        }
    }

    #[test]
    fn stale_results_cannot_replace_latest_but_their_proxy_can_be_reused() {
        let doc = image(SampleType::U8);
        let mut preview = RadialPreview::default();
        preview.request(request(&doc, json!({"amount": 10}), 1));
        let stale = preview.wanted.clone().unwrap();
        preview.request(request(&doc, json!({"amount": 30}), 1));
        let latest = preview.wanted.clone().unwrap();
        let cache = CachedProxy { source: stale.source.clone(), doc: doc.clone() };
        assert!(preview.accept(Completion { generation: stale.generation, cache: Some(cache), result: Ok(Some(frame(&stale))) }).is_none());
        assert!(preview.shown.is_none());
        assert!(Arc::ptr_eq(&preview.cache.as_ref().unwrap().doc, &doc));
        preview.accept(Completion { generation: latest.generation, cache: None, result: Ok(Some(frame(&latest))) });
        assert_eq!(preview.shown.as_ref().unwrap().generation, latest.generation);
        preview.accept(Completion { generation: stale.generation, cache: None, result: Ok(Some(frame(&stale))) });
        assert_eq!(preview.shown.as_ref().unwrap().generation, latest.generation);
    }

    #[test]
    fn amount_changes_keep_last_complete_frame_and_only_newest_request() {
        let doc = image(SampleType::U8);
        let mut preview = RadialPreview::default();
        preview.request(request(&doc, json!({"amount": 10}), 1));
        let first = preview.wanted.clone().unwrap();
        preview.accept(Completion { generation: first.generation, cache: None, result: Ok(Some(frame(&first))) });
        for amount in 11..100 {
            preview.request(request(&doc, json!({"amount": amount}), 1));
        }
        assert_eq!(preview.shown.as_ref().unwrap().generation, first.generation);
        assert_eq!(preview.wanted.as_ref().unwrap().params, json!({"amount": 99}));
        assert!(preview.pending());
    }

    #[test]
    fn source_target_revision_and_reopened_dialog_invalidate_previous_frame() {
        let doc = image(SampleType::U8);
        for changed in ["layer", "channel", "mask", "revision", "dialog", "snapshot", "visibility", "texture"] {
            let mut preview = RadialPreview::default();
            preview.request(request(&doc, json!({"amount": 10}), 1));
            let first = preview.wanted.clone().unwrap();
            preview.accept(Completion {
                generation: first.generation,
                cache: Some(CachedProxy { source: first.source.clone(), doc: doc.clone() }),
                result: Ok(Some(frame(&first))),
            });
            let ctx = egui::Context::default();
            preview.overlay_texture = Some((
                first.generation,
                ctx.load_texture("Previous overlay", egui::ColorImage::filled([3, 3], egui::Color32::WHITE), egui::TextureOptions::LINEAR),
            ));
            let mut next = first.clone();
            match changed {
                "layer" => next.source.active = Some(LayerId(u64::MAX)),
                "channel" => next.source.channel = ChannelTarget::Color(0),
                "mask" => next.source.target = Some(json!("mask")),
                "revision" => next.source.revision += 1,
                "dialog" => next.dialog += 1,
                "visibility" => next.view.color_hidden = vec![true, false, false],
                "texture" => next.max_texture_side = 2,
                _ => {
                    next.original = Arc::new((*doc).clone());
                    next.source.snapshot = Arc::downgrade(&next.original);
                }
            }
            preview.request(next);
            assert!(preview.shown.is_none(), "{changed}");
            assert!(preview.overlay_texture.is_none(), "{changed}");
            if !matches!(changed, "dialog" | "visibility" | "texture") {
                assert!(preview.cache.is_none(), "{changed}");
            } else {
                assert!(preview.cache.is_some(), "reuse the source proxy for {changed}");
            }
        }
    }

    #[test]
    fn background_command_and_composite_match_existing_proxy_preview() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let doc = image(depth);
            for method in ["spin", "zoom"] {
                for k in [1, 2] {
                    let request = request(&doc, json!({"amount": 50, "method": method, "centerX": 0.25, "centerY": 0.75}), k);
                    let expected = crate::filter_dialog::preview_document(&doc, request.source.active, COMMAND, &request.params, k).unwrap();
                    let completion = compute(request, None, &AtomicBool::new(false));
                    let actual = completion.result.unwrap().unwrap();
                    assert_eq!(&*actual.buffer, &photocraft_compose::flatten(&expected), "{depth:?}, {method}, k={k}");
                    let actual_surface = actual.result.layers[0].surface().unwrap();
                    let expected_surface = expected.layers[0].surface().unwrap();
                    assert_eq!(actual_surface.read_region(expected.bounds()), expected_surface.read_region(expected.bounds()));
                }
            }
        }
    }

    #[test]
    fn color_channel_preview_matches_committed_command_and_keeps_other_channels() {
        let doc = image(SampleType::F32);
        let mut request = request(&doc, json!({"amount": 100, "method": "spin"}), 1);
        request.source.channel = ChannelTarget::Color(0);
        let mut session = photocraft_engine::Session::new();
        session.add_document((*doc).clone(), None);
        session.active_mut().unwrap().channel_view.target = request.source.channel;
        session.execute(COMMAND, request.params.clone()).unwrap();
        let expected = session.active().unwrap().doc.clone();
        let actual = compute(request, None, &AtomicBool::new(false)).result.unwrap().unwrap();
        assert_eq!(&*actual.buffer, &photocraft_compose::flatten(&expected));
        for (before, after) in doc.layers[0]
            .surface()
            .unwrap()
            .read_region(doc.bounds())
            .as_chunks::<4>()
            .0
            .iter()
            .zip(actual.result.layers[0].surface().unwrap().read_region(doc.bounds()).as_chunks::<4>().0.iter())
        {
            assert_eq!(&before[1..], &after[1..]);
        }
    }

    #[test]
    fn filtered_channel_overlay_matches_alpha_quick_mask_and_gray_mask_commands() {
        use photocraft_doc::{AlphaChannel, LayerMask};
        use photocraft_engine::mask_view_cmds::{LayerMaskView, MaskViewMode};

        for kind in ["alpha", "quick", "mask"] {
            for k in [1, 2] {
                let mut doc = (*image(SampleType::F32)).clone();
                doc.selection = None;
                let mut plane = photocraft_raster::Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
                plane.fill_rect(doc.bounds(), &[0.0]);
                plane.fill_rect(Rect::new(0, 0, 16, 12), &[1.0]);
                let mut view = ChannelView::default();
                let mut params = json!({"amount": 100, "method": "spin"});
                match kind {
                    "alpha" => {
                        doc.channels.push(AlphaChannel::new("Radial alpha", plane));
                        view.target = ChannelTarget::Alpha(0);
                        view.color_hidden = vec![true; 3];
                        view.alpha_visible = vec![true];
                    }
                    "quick" => doc.quick_mask = Some(AlphaChannel::new("Quick Mask", plane)),
                    _ => {
                        let layer = doc.layers[0].id;
                        doc.layers[0].mask = Some(LayerMask { surface: plane, enabled: true, linked: true, density: 1.0, feather: 0.0 });
                        view.layer_mask = Some(LayerMaskView { layer, mode: MaskViewMode::Gray });
                        params["target"] = json!("mask");
                    }
                }
                let doc = Arc::new(doc);
                let mut request = request(&doc, params, k);
                request.source.channel = view.target;
                request.view = view.clone();
                request.max_texture_side = 7;
                let original_overlay = channel_overlay(&prepare_proxy(&doc, k), &view, 7, &AtomicBool::new(false)).unwrap().unwrap();

                // Build the independent command reference, explicitly scaling Quick Mask too.
                let mut expected_proxy = crate::proxy::proxy_document(&doc, k);
                if let Some(quick) = &mut expected_proxy.quick_mask {
                    quick.surface = crate::proxy::downsample(&doc.quick_mask.as_ref().unwrap().surface, k);
                }
                let expected = command_preview(&request, &expected_proxy, &AtomicBool::new(false)).unwrap();
                let expected_overlay = channel_overlay(&expected, &view, 7, &AtomicBool::new(false)).unwrap().unwrap();
                let actual = compute(request, None, &AtomicBool::new(false)).result.unwrap().unwrap();
                let actual_overlay = actual.overlay.as_ref().unwrap();
                assert_eq!(actual_overlay.size, expected_overlay.size, "{kind}, k={k}");
                assert_eq!(actual_overlay.pixels, expected_overlay.pixels, "{kind}, k={k}");
                assert_ne!(actual_overlay.pixels, original_overlay.pixels, "the filtered overlay must replace the original {kind}, k={k}");
                assert!(actual_overlay.size.iter().all(|side| *side <= 7), "honour the context's texture limit");
                if kind == "quick" {
                    let actual_plane = &actual.result.quick_mask.as_ref().unwrap().surface;
                    let expected_plane = &expected.quick_mask.as_ref().unwrap().surface;
                    assert_eq!(actual_plane.read_region(expected.bounds()), expected_plane.read_region(expected.bounds()), "reduced Quick Mask geometry");
                    assert!(actual_plane.content_bounds().x1 <= 32 / k as i32);
                    assert!(actual_plane.content_bounds().y1 <= 24 / k as i32);
                }
            }
        }
    }

    #[test]
    fn overlay_upload_only_uses_the_displayed_complete_radial_frame() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.add_document((*image(SampleType::U8)).clone(), None);
        crate::filter_dialog::open(&mut app, COMMAND).unwrap();
        let request = Request::from_app(&app, 0).unwrap();
        app.radial_preview.request(request);
        let request = app.radial_preview.wanted.clone().unwrap();
        let mut shown = frame(&request);
        let ctx = egui::Context::default();
        let key = shown.doc.0 ^ (1u64 << 61);
        app.radial_preview.shown = Some(shown.clone());
        assert!(overlay(&mut app, &ctx, 0, key).is_none(), "the radial frame has not been drawn");
        app.filter_preview = Some(crate::filter_dialog::FilterPreview {
            doc: shown.doc,
            revision: shown.revision,
            hash: shown.generation,
            k: shown.k,
            result: Some(shown.result.clone()),
        });
        assert_eq!(overlay(&mut app, &ctx, 0, key), Some(None), "plain composite has no overlay");
        shown.overlay = Some(Arc::new(egui::ColorImage::new([1, 1], vec![egui::Color32::WHITE])));
        app.radial_preview.shown = Some(shown);
        assert!(overlay(&mut app, &ctx, 0, key).unwrap().is_some());
        let texture = app.radial_preview.overlay_texture.as_ref().unwrap().1.id();
        assert_eq!(overlay(&mut app, &ctx, 0, key), Some(Some(texture)), "reuse completed-generation texture");
        assert!(overlay(&mut app, &ctx, 0, key ^ 1).is_none(), "another GPU image is being displayed");
        app.filter_preview.as_mut().unwrap().result = Some(Arc::new((*request.original).clone()));
        assert!(overlay(&mut app, &ctx, 0, key).is_none(), "another preview result is being displayed");
    }

    #[test]
    fn cancellation_skips_preparation_and_cannot_publish_a_frame() {
        let doc = image(SampleType::U8);
        let request = request(&doc, json!({"amount": 100}), 1);
        let completion = compute(request, None, &AtomicBool::new(true));
        assert!(completion.cache.is_none());
        assert!(completion.result.unwrap().is_none());
        assert!(flatten(&doc, &AtomicBool::new(true)).is_err());
    }

    #[test]
    fn very_thin_huge_document_is_rejected_before_proxy_allocation() {
        let doc = Arc::new(Document::new("Thin preview", Size::new(1, u32::MAX), ColorMode::Rgb, SampleType::F32));
        let k = crate::proxy::factor(&doc);
        let completion = compute(request(&doc, json!({"amount": 100}), k), None, &AtomicBool::new(false));
        assert!(completion.cache.is_none());
        assert!(matches!(completion.result, Err(ref error) if error.contains("pixel budget")));
    }

    #[test]
    fn closed_or_disabled_dialog_discards_completed_and_late_results() {
        for close in [true, false] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.session.add_document((*image(SampleType::U8)).clone(), None);
            let id = crate::filter_dialog::open(&mut app, COMMAND).unwrap();
            let request = Request::from_app(&app, 0).unwrap();
            app.radial_preview.request(request);
            let wanted = app.radial_preview.wanted.clone().unwrap();
            app.radial_preview.accept(Completion { generation: wanted.generation, cache: None, result: Ok(Some(frame(&wanted))) });
            #[cfg(not(target_arch = "wasm32"))]
            let cancel = {
                let cancel = Arc::new(AtomicBool::new(false));
                let worker_cancel = cancel.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                let handle = std::thread::spawn(move || {
                    while !worker_cancel.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    drop(tx);
                });
                app.radial_preview.running = Some(Running { cancel: cancel.clone(), rx, handle });
                cancel
            };
            if close {
                app.ui.close_dialog(id);
            } else {
                app.ui.dialog_mut(id).unwrap().fields.insert("__preview".into(), json!(false));
            }
            retain(&mut app);
            assert!(app.radial_preview.wanted.is_none());
            assert!(app.radial_preview.shown.is_none());
            #[cfg(not(target_arch = "wasm32"))]
            {
                assert!(cancel.load(Ordering::Relaxed));
                if let Some(running) = app.radial_preview.running.take() {
                    running.handle.join().unwrap();
                }
            }
            app.radial_preview.accept(Completion { generation: wanted.generation, cache: None, result: Ok(Some(frame(&wanted))) });
            assert!(app.radial_preview.shown.is_none());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn changes_cancel_the_worker_without_starting_a_second_one() {
        let doc = image(SampleType::U8);
        let mut preview = RadialPreview::default();
        preview.request(request(&doc, json!({"amount": 10}), 1));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            while !worker_cancel.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            drop(tx);
        });
        preview.running = Some(Running { cancel: cancel.clone(), rx, handle });
        preview.request(request(&doc, json!({"amount": 20}), 1));
        assert!(cancel.load(Ordering::Relaxed));
        assert!(preview.start(&egui::Context::default()).is_none());
        assert!(preview.started.is_none(), "the replacement waits until the first worker exits");
        preview.running.take().unwrap().handle.join().unwrap();
    }

    /// Compare the previous drawing-thread command+flatten work with scheduling the same image.
    /// This is opt-in because it runs a full 2.4 MP preview and is meaningful in release only.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "release-only 2.4 MP Radial Blur preview responsiveness measurement"]
    fn release_preview_responsiveness() {
        use std::time::{Duration, Instant};

        let mut doc = Document::with_background("Radial preview benchmark", Size::new(2000, 1200), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        for x in (0..2000).step_by(32) {
            doc.layers[0].surface_mut().unwrap().fill_rect(Rect::new(x, 0, (x + 16).min(2000), 1200), &[x as f32 / 2000.0, 0.125, 0.625, 1.0]);
        }
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.add_document(doc, None);
        let id = crate::filter_dialog::open(&mut app, COMMAND).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("amount".into(), json!(100));
        app.ui.dialog_mut(id).unwrap().fields.insert("method".into(), json!("spin"));
        let request = Request::from_app(&app, 0).unwrap();
        assert_eq!(request.source.k, 1, "use the unchanged 2.4 MP preview policy");
        let baseline_started = Instant::now();
        let synchronous = crate::filter_dialog::preview_document(&request.original, request.source.active, COMMAND, &request.params, request.source.k).unwrap();
        let baseline = photocraft_compose::flatten(&synchronous);
        let baseline_ms = baseline_started.elapsed().as_secs_f64() * 1000.0;
        drop(synchronous);

        let ctx = egui::Context::default();
        let background_started = Instant::now();
        let schedule_started = Instant::now();
        assert!(update(&mut app, 0, &ctx).is_none());
        let schedule_ms = schedule_started.elapsed().as_secs_f64() * 1000.0;
        let mut max_poll_ms = 0.0f64;
        let ready = loop {
            assert!(background_started.elapsed() < Duration::from_secs(120), "the preview worker did not complete");
            std::thread::sleep(Duration::from_millis(2));
            let poll_started = Instant::now();
            let ready = update(&mut app, 0, &ctx);
            max_poll_ms = max_poll_ms.max(poll_started.elapsed().as_secs_f64() * 1000.0);
            if let Some(ready) = ready {
                break ready;
            }
        };
        let background_ms = background_started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(&*ready.buffer, &baseline);
        println!(
            "{}",
            json!({
                "scenario": "radial-preview-responsive", "pixels": 2_400_000, "method": "spin", "amount": 100,
                "synchronous_command_flatten_ms": baseline_ms, "schedule_ms": schedule_ms,
                "max_completion_poll_ms": max_poll_ms, "background_command_flatten_ms": background_ms,
                "quality": "identical composite"
            })
        );
    }
}
