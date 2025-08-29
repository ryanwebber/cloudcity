use std::sync::Arc;

use pollster::FutureExt;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{pipeline::PointCloudPipeline, storage, types};

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    quad: Quad,
    point_cloud_pipeline: PointCloudPipeline,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    camera_uniforms: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub fn try_new(window: Arc<Window>) -> anyhow::Result<Self> {
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

        let config = wgpu::SurfaceConfiguration {
            desired_maximum_frame_latency: 2,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        let quad = Quad::create(&device);
        let point_cloud_pipeline = PointCloudPipeline::new(&device, surface_format);

        // Create instance data with random points
        let instances = Self::create_random_instances(5000); // Increased from 1000 to 5000
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create camera uniforms buffer
        let camera_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms"),
            size: std::mem::size_of::<storage::uniform::Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for camera uniforms
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &point_cloud_pipeline.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniforms.as_entire_binding(),
            }],
        });

        surface.configure(&device, &config);

        Ok(Self {
            surface,
            surface_config: config,
            device,
            queue,
            quad,
            point_cloud_pipeline,
            instance_buffer,
            instance_count: instances.len() as u32,
            camera_uniforms,
            camera_bind_group,
        })
    }

    fn create_random_instances(count: usize) -> Vec<storage::instance::Instance> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..count)
            .map(|_| {
                // Random unit-sphere point
                let unit_sphere = {
                    let x = rng.gen_range(-1.0..1.0);
                    let y = rng.gen_range(-1.0..1.0);
                    let z = rng.gen_range(-1.0..1.0);
                    let distance = f32::sqrt(x * x + y * y + z * z);
                    let x = x / distance;
                    let y = y / distance;
                    let z = z / distance;
                    glam::f32::vec3(x, y, z)
                };

                let position = unit_sphere * rng.gen_range(5.0..20.0);

                let r = rng.gen_range(100..255);
                let g = rng.gen_range(100..255);
                let b = rng.gen_range(100..255);

                storage::instance::Instance {
                    position,
                    color: storage::instance::Color::from_rgba(r, g, b, 255),
                }
            })
            .collect()
    }

    pub fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn render(
        &mut self,
        camera: &types::Camera,
        _timings: &types::Timings,
    ) -> anyhow::Result<()> {
        // Update camera uniforms
        self.update_camera_uniforms(camera);

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.1,
                                b: 0.1,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set the pipeline and bind group
            render_pass.set_pipeline(&self.point_cloud_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Set vertex and index buffers
            render_pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass
                .set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw instances
            render_pass.draw_indexed(0..6, 0, 0..self.instance_count);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update_camera_uniforms(&mut self, camera: &types::Camera) {
        // Convert camera transform to matrix (not used in current implementation)
        let _transform_matrix = glam::f32::Mat4::from_translation(camera.transform.position)
            * glam::f32::Mat4::from_rotation_x(camera.transform.rotation.x.to_radians())
            * glam::f32::Mat4::from_rotation_y(camera.transform.rotation.y.to_radians())
            * glam::f32::Mat4::from_rotation_z(camera.transform.rotation.z.to_radians())
            * glam::f32::Mat4::from_scale(camera.transform.scale);

        // Create view-projection matrix
        let aspect_ratio = self.surface_config.width as f32 / self.surface_config.height as f32;
        let fov = match camera.lens {
            types::Lens::Perspective { fov, .. } => fov.to_radians(),
        };

        let projection = glam::f32::Mat4::perspective_rh(
            fov,
            aspect_ratio,
            camera.clipping.near,
            camera.clipping.far,
        );

        // For a camera, we want to look from the camera position towards the negative Z direction
        // The view matrix should transform world coordinates to camera coordinates
        let view = glam::f32::Mat4::look_at_rh(
            camera.transform.position,
            camera.transform.position + glam::f32::vec3(0.0, 0.0, 1.0),
            glam::f32::vec3(0.0, 1.0, 0.0),
        );

        let _view_proj = projection * view;

        // Create camera uniforms struct
        let camera_uniforms = storage::uniform::Camera {
            focal_view: glam::f32::vec3(0.0, 0.0, -1.0), // Simplified
            world_space_position: camera.transform.position,
            view_matrix: view,
            projection_matrix: projection,
            near_clip: camera.clipping.near,
            far_clip: camera.clipping.far,
        };

        // Update the buffer
        self.queue.write_buffer(
            &self.camera_uniforms,
            0,
            bytemuck::cast_slice(&[camera_uniforms]),
        );
    }
}

struct Quad {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl Quad {
    const VERTICIES: &[storage::buffer::Vertex] = &[
        storage::buffer::Vertex {
            position: glam::f32::vec3(-1.0, 1.0, 0.0),
            uvs: glam::f32::vec2(0.0, 1.0),
        },
        storage::buffer::Vertex {
            position: glam::f32::vec3(-1.0, -1.0, 0.0),
            uvs: glam::f32::vec2(0.0, 0.0),
        },
        storage::buffer::Vertex {
            position: glam::f32::vec3(1.0, -1.0, 0.0),
            uvs: glam::f32::vec2(1.0, 0.0),
        },
        storage::buffer::Vertex {
            position: glam::f32::vec3(1.0, 1.0, 0.0),
            uvs: glam::f32::vec2(1.0, 1.0),
        },
    ];

    const INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];

    fn create(device: &wgpu::Device) -> Self {
        Self {
            vertex_buffer: {
                let bytes = bytemuck::cast_slice(Self::VERTICIES);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::VERTEX,
                })
            },
            index_buffer: {
                let bytes = bytemuck::cast_slice(Self::INDICES);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::INDEX,
                })
            },
        }
    }
}
