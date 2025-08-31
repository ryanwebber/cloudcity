use crate::{
    renderer::Graphics,
    types::{Camera, Timings},
};

pub mod gui;
pub mod scene;

pub trait Layer {
    fn render(
        &mut self,
        camera: &Camera,
        timings: &Timings,
        graphics: &Graphics,
        view: &wgpu::TextureView,
    ) -> anyhow::Result<()>;

    fn resize(&mut self, _: &Graphics, _: winit::dpi::PhysicalSize<u32>) {
        // Do nothing by default
    }
}
