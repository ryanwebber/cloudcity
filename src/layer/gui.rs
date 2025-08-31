use std::sync::Arc;

use crate::{layer::Layer, renderer::Graphics, types};

pub struct GuiLayer {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    window: Arc<winit::window::Window>,
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
        })
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

        // TODO: Implement in GUI
        _ = timings;

        let mut view_model = ViewModel {
            camera_position_toolbar: CameraPositionToolbar {
                x: camera.transform.position.x,
                y: camera.transform.position.y,
                z: camera.transform.position.z,
            },
        };

        let input = self.state.take_egui_input(&self.window);
        let output = ctx.run(input, |ctx| {
            view_model.ui(ctx);
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
}

impl ViewModel {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |_| {
                egui::panel::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
                    self.camera_position_toolbar.ui(ui);
                });
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
