use std::sync::Arc;

use crate::{
    layer::{Layer, scene::DebugEvent},
    renderer::Graphics,
    types::{self, Timings},
};

pub struct GuiLayer {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    window: Arc<winit::window::Window>,
    view_model: ViewModel,
}

impl GuiLayer {
    pub fn try_new(
        graphics: &Graphics,
        window: Arc<winit::window::Window>,
    ) -> anyhow::Result<Self> {
        let ctx = egui::Context::default();
        let viewport_id = ctx.viewport_id();
        let state = egui_winit::State::new(ctx, viewport_id, &window, None, None, None);
        let renderer = egui_wgpu::Renderer::new(
            graphics.device(),
            graphics.surface_config().format,
            None,
            1u32,
            true,
        );

        Ok(Self {
            state,
            renderer,
            window,
            view_model: ViewModel {
                culling_stats_section: None,
                timing_stats_section: None,
                camera_position_toolbar: CameraPositionToolbar {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
        })
    }

    pub fn handle_debug_event(&mut self, debug_event: DebugEvent) {
        self.view_model.handle_debug_event(debug_event);
    }
}

impl Layer for GuiLayer {
    fn render(
        &mut self,
        camera: &types::Camera,
        timings: &types::Timings,
        graphics: &Graphics,
        view: &wgpu::TextureView,
    ) -> anyhow::Result<()> {
        let ctx = self.state.egui_ctx().clone();
        let width = graphics.surface_config().width;
        let height = graphics.surface_config().height;
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width as u32, height as u32],
            pixels_per_point: ctx.pixels_per_point(),
        };

        self.view_model.update_camera(camera);
        self.view_model.update_timings(timings);

        let input = self.state.take_egui_input(&self.window);
        let output = ctx.run(input, |ctx| {
            self.view_model.ui(ctx);
        });

        self.state
            .handle_platform_output(&self.window, output.platform_output);

        let texture_deltas = output.textures_delta;
        let paint_jobs = ctx.tessellate(output.shapes, ctx.pixels_per_point());

        for (id, image_delta) in &texture_deltas.set {
            self.renderer
                .update_texture(graphics.device(), graphics.queue(), *id, image_delta);
        }

        for id in &texture_deltas.free {
            self.renderer.free_texture(id);
        }

        let mut encoder =
            graphics
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GUI Render Encoder"),
                });

        {
            self.renderer.update_buffers(
                graphics.device(),
                graphics.queue(),
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            );

            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        // Submit commands
        graphics.queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}

struct ViewModel {
    camera_position_toolbar: CameraPositionToolbar,
    timing_stats_section: Option<TimingStatsSection>,
    culling_stats_section: Option<CullingStatsSection>,
}

impl ViewModel {
    fn update_camera(&mut self, camera: &types::Camera) {
        self.camera_position_toolbar.x = camera.transform.position.x;
        self.camera_position_toolbar.y = camera.transform.position.y;
        self.camera_position_toolbar.z = camera.transform.position.z;
    }

    fn update_timings(&mut self, timings: &types::Timings) {
        if let Some(timing_stats) = &mut self.timing_stats_section {
            timing_stats.timings.frame = timings.frame;
            timing_stats.timings.time_since_start = timings.time_since_start;
            if timings.frame % 10 == 0 {
                timing_stats.timings.fps = timings.fps;
                timing_stats.timings.time_since_last_frame = timings.time_since_last_frame;
            }
        } else {
            self.timing_stats_section = Some(TimingStatsSection {
                timings: timings.clone(),
            });
        }
    }

    fn handle_debug_event(&mut self, debug_event: DebugEvent) {
        match debug_event {
            DebugEvent::CullingStats {
                total_points,
                visible_points,
            } => {
                self.culling_stats_section = Some(CullingStatsSection {
                    total_points,
                    visible_points,
                });
            }
        }
    }

    fn ui(&mut self, ctx: &egui::Context) {
        ctx.set_zoom_factor(0.75);
        egui::Window::new(
            egui::RichText::new("Performance")
                .text_style(egui::TextStyle::Body)
                .strong(),
        )
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            if let Some(timing_stats) = &self.timing_stats_section {
                timing_stats.ui(ui);
            }

            if let Some(culling_stats) = &self.culling_stats_section {
                culling_stats.ui(ui);
            }
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |_| {
                egui::panel::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
                    self.camera_position_toolbar.ui(ui);
                });
            });
    }
}

#[derive(Clone)]
struct TimingStatsSection {
    timings: Timings,
}

impl TimingStatsSection {
    fn ui(&self, ui: &mut egui::Ui) {
        draw_section(ui, "Timing", |ui| {
            ui.label("FPS:");
            ui.label(egui::RichText::new(format!("{:.0}", self.timings.fps)).monospace());

            ui.end_row();

            ui.label("Frame:");
            ui.label(egui::RichText::new(format!("{}", self.timings.frame)).monospace());

            ui.end_row();

            ui.label("Frame Time:");
            ui.label(
                egui::RichText::new(format!(
                    "{}ms",
                    self.timings.time_since_last_frame.as_millis()
                ))
                .monospace(),
            );

            ui.end_row();

            ui.label("TotalTime:");
            ui.label(
                egui::RichText::new(format!("{:.3}s", self.timings.time_since_start.as_secs()))
                    .monospace(),
            );

            ui.end_row();
        });
    }
}

#[derive(Clone)]
struct CullingStatsSection {
    total_points: usize,
    visible_points: usize,
}

impl CullingStatsSection {
    fn ui(&self, ui: &mut egui::Ui) {
        draw_section(ui, "Culling", |ui| {
            ui.label("Total Points:");
            ui.label(egui::RichText::new(format!("{}", self.total_points)).monospace());

            ui.end_row();

            ui.label("Culled Points:");
            ui.label(
                egui::RichText::new(format!("{}", self.total_points - self.visible_points))
                    .monospace(),
            );

            ui.end_row();

            ui.label("Culling Rate:");
            ui.label(
                egui::RichText::new(format!(
                    "{:.1}%",
                    (self.total_points - self.visible_points) as f32 / self.total_points as f32
                        * 100.0
                ))
                .monospace(),
            );
        });
    }
}

struct CameraPositionToolbar {
    x: f32,
    y: f32,
    z: f32,
}

impl CameraPositionToolbar {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Pad in away from the window corner
            ui.add_space(16.0);
            ui.label(egui::RichText::new("Camera Position").monospace().strong());
            ui.separator();

            {
                ui.label(egui::RichText::new("x =").monospace());
                ui.label(egui::RichText::new(format!("{:.2}", self.x)).monospace());
            }

            ui.separator();

            {
                ui.label(egui::RichText::new("y =").monospace());
                ui.label(egui::RichText::new(format!("{:.2}", self.y)).monospace());
            }

            ui.separator();

            {
                ui.label(egui::RichText::new("z =").monospace());
                ui.label(egui::RichText::new(format!("{:.2}", self.z)).monospace());
            }

            ui.separator();
        });
    }
}

fn draw_section<F>(ui: &mut egui::Ui, name: &'static str, builder: F)
where
    F: FnOnce(&mut egui::Ui),
{
    egui::CollapsingHeader::new(name)
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new(name)
                .striped(true)
                .spacing([10.0, 10.0])
                .show(ui, |ui| {
                    builder(ui);
                });
        });
}
