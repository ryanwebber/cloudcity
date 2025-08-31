use crate::{
    renderer::Graphics,
    types::{Camera, Timings},
};

pub mod scene;

pub trait Layer {
    fn render(
        &self,
        camera: &Camera,
        timings: &Timings,
        graphics: &Graphics,
        view: &wgpu::TextureView,
    ) -> anyhow::Result<()>;

    fn resize(&mut self, _new_size: winit::dpi::PhysicalSize<u32>) {
        // Do nothing by default
    }
}
