use pollster::FutureExt;
use std::sync::Arc;
use wgpu;
use winit::window::Window;

use crate::{layer::Layer, types};

pub struct Renderer {
    graphics: Graphics,
    layers: Vec<Box<dyn Layer>>,
}

impl Renderer {
    pub fn try_new(window: Arc<Window>) -> anyhow::Result<Self> {
        let graphics = Graphics::try_new(window.clone())?;
        Ok(Self {
            graphics,
            layers: vec![],
        })
    }

    pub fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.graphics.resize_surface(new_size);
        for layer in self.layers.iter_mut() {
            layer.resize(new_size);
        }
    }

    pub fn render(
        &mut self,
        camera: &types::Camera,
        timings: &types::Timings,
    ) -> anyhow::Result<()> {
        let surface = self.graphics.surface.get_current_texture()?;
        let view = surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        for layer in self.layers.iter() {
            layer.render(camera, timings, &self.graphics, &view)?;
        }

        surface.present();

        Ok(())
    }

    pub fn graphics(&self) -> &Graphics {
        &self.graphics
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}

pub struct Graphics {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layers: Vec<Box<dyn Layer>>,
}

impl Graphics {
    fn try_new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .block_on()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            })
            .block_on()?;

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code assumes an sRGB surface texture
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            desired_maximum_frame_latency: 2,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            layers: vec![],
        })
    }

    fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);

        for layer in self.layers.iter_mut() {
            layer.resize(new_size);
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_config(&self) -> &wgpu::SurfaceConfiguration {
        &self.surface_config
    }
}
