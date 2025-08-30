use std::sync::Arc;

use pollster::FutureExt;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{pipeline::PointCloudPipeline, spatial, storage, types};

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    hexagon: Hexagon,
    point_cloud_pipeline: PointCloudPipeline,
    camera_uniforms: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    spatial_index: spatial::SpatialIndex,
    all_instances: Vec<storage::instance::Instance>,
    visible_instance_buffer: wgpu::Buffer,
    visible_instance_count: u32,
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

        let hexagon = Hexagon::create(&device);
        let point_cloud_pipeline = PointCloudPipeline::new(&device, surface_format);

        // Create instance data with random points
        let instances = Self::create_random_instances(5000); // Increased from 1000 to 5000

        // Create spatial index for frustum culling
        let point_positions: Vec<glam::f32::Vec3> = instances.iter().map(|i| i.position).collect();
        let spatial_index = spatial::SpatialIndex::new(point_positions);

        // Create buffer for visible instances
        let visible_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Instance Buffer"),
            size: (std::mem::size_of::<storage::instance::Instance>() * instances.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            surface,
            surface_config: config,
            device,
            queue,
            hexagon,
            point_cloud_pipeline,
            camera_uniforms,
            camera_bind_group,
            depth_texture,
            depth_texture_view,
            spatial_index,
            all_instances: instances,
            visible_instance_buffer,
            visible_instance_count: 0,
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

                let color = if position.z < 0.0 {
                    storage::instance::Color::from_rgba(0, 12, 255, 255)
                } else {
                    // Create depth-based color gradient: red (near) to green (far)
                    let distance_from_camera = position.length();
                    let normalized_distance = (distance_from_camera - 5.0) / 15.0; // Normalize to 0.0-1.0
                    let normalized_distance = normalized_distance.clamp(0.0, 1.0);

                    // Red component decreases with distance, Green component increases with distance
                    let r = ((1.0 - normalized_distance) * 255.0) as u8;
                    let g = (normalized_distance * 255.0) as u8;
                    let b = 0; // No blue component

                    storage::instance::Color::from_rgba(r, g, b, 255)
                };

                storage::instance::Instance { position, color }
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

        // Recreate depth texture with new size
        self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: new_size.width,
                height: new_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        self.depth_texture_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub fn render(
        &mut self,
        camera: &types::Camera,
        _timings: &types::Timings,
    ) -> anyhow::Result<()> {
        // Update camera uniforms
        self.update_camera_uniforms(camera);

        // Perform frustum culling to get visible instances
        self.update_visible_instances(camera);

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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set the pipeline and bind group
            render_pass.set_pipeline(&self.point_cloud_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Set vertex and index buffers
            render_pass.set_vertex_buffer(0, self.hexagon.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.visible_instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.hexagon.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );

            // Draw instances
            render_pass.draw_indexed(0..12, 0, 0..self.visible_instance_count);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update_camera_uniforms(&mut self, camera: &types::Camera) {
        // Convert camera transform to matrix (not used in current implementation)
        let _transform_matrix = glam::f32::Mat4::from_translation(camera.transform.position)
            * glam::f32::Mat4::from_rotation_x(camera.transform.rotation.x)
            * glam::f32::Mat4::from_rotation_y(camera.transform.rotation.y)
            * glam::f32::Mat4::from_rotation_z(camera.transform.rotation.z)
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

        // Calculate forward direction based on camera rotation
        let rotation_matrix = glam::f32::Mat4::from_rotation_y(camera.transform.rotation.y)
            * glam::f32::Mat4::from_rotation_x(camera.transform.rotation.x);
        let forward = rotation_matrix.transform_point3(glam::f32::vec3(0.0, 0.0, -1.0));

        // Create view matrix using camera position and calculated forward direction
        let view = glam::f32::Mat4::look_at_rh(
            camera.transform.position,
            camera.transform.position + forward,
            glam::f32::vec3(0.0, 1.0, 0.0),
        );

        let _view_proj = projection * view;

        // Create camera uniforms struct
        let camera_uniforms = storage::uniform::Camera {
            focal_view: forward, // Use calculated forward direction
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

    fn update_visible_instances(&mut self, camera: &types::Camera) {
        // Calculate view-projection matrix for frustum culling
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

        let rotation_matrix = glam::f32::Mat4::from_rotation_y(camera.transform.rotation.y)
            * glam::f32::Mat4::from_rotation_x(camera.transform.rotation.x);
        let forward = rotation_matrix.transform_point3(glam::f32::vec3(0.0, 0.0, -1.0));

        let view = glam::f32::Mat4::look_at_rh(
            camera.transform.position,
            camera.transform.position + forward,
            glam::f32::vec3(0.0, 1.0, 0.0),
        );

        let view_projection = projection * view;

        // Get visible point indices using frustum culling
        let visible_indices = self.spatial_index.get_visible_points(view_projection);

        // Create visible instances buffer
        let mut visible_instances = Vec::with_capacity(visible_indices.len());
        for &index in &visible_indices {
            visible_instances.push(self.all_instances[index]);
        }

        // Update the visible instance buffer
        if !visible_instances.is_empty() {
            self.queue.write_buffer(
                &self.visible_instance_buffer,
                0,
                bytemuck::cast_slice(&visible_instances),
            );
            self.visible_instance_count = visible_instances.len() as u32;
        } else {
            self.visible_instance_count = 0;
        }

        // Log culling statistics occasionally
        if self.visible_instance_count % 1000 == 0 {
            log::info!(
                "Frustum culling: {}/{} points visible ({:.1}% culled)",
                self.visible_instance_count,
                self.all_instances.len(),
                (1.0 - self.visible_instance_count as f32 / self.all_instances.len() as f32)
                    * 100.0
            );
        }
    }

    /// Get current culling statistics
    pub fn get_culling_stats(&self) -> (u32, usize, f32) {
        let total = self.all_instances.len();
        let visible = self.visible_instance_count as usize;
        let culled_percentage = if total > 0 {
            (1.0 - visible as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        (self.visible_instance_count, total, culled_percentage)
    }
}

struct Hexagon {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl Hexagon {
    const VERTICIES: &[storage::buffer::Vertex] = &[
        // Vertex 0 (Top-left)
        storage::buffer::Vertex {
            position: glam::f32::vec3(-0.5, 0.866, 0.0),
            uvs: glam::f32::vec2(0.0, 0.866),
        },
        // Vertex 1 (Top-right)
        storage::buffer::Vertex {
            position: glam::f32::vec3(0.5, 0.866, 0.0),
            uvs: glam::f32::vec2(1.0, 0.866),
        },
        // Vertex 2 (Right)
        storage::buffer::Vertex {
            position: glam::f32::vec3(1.0, 0.0, 0.0),
            uvs: glam::f32::vec2(1.0, 0.5),
        },
        // Vertex 3 (Bottom-right)
        storage::buffer::Vertex {
            position: glam::f32::vec3(0.5, -0.866, 0.0),
            uvs: glam::f32::vec2(1.0, 0.134),
        },
        // Vertex 4 (Bottom-left)
        storage::buffer::Vertex {
            position: glam::f32::vec3(-0.5, -0.866, 0.0),
            uvs: glam::f32::vec2(0.0, 0.134),
        },
        // Vertex 5 (Left)
        storage::buffer::Vertex {
            position: glam::f32::vec3(-1.0, 0.0, 0.0),
            uvs: glam::f32::vec2(0.0, 0.5),
        },
    ];

    const INDICES: &[u32] = &[
        0, 5, 1, // Triangle 1: top-left, left, top-right
        1, 5, 2, // Triangle 2: top-right, left, right
        2, 5, 4, // Triangle 3: right, left, bottom-left
        2, 4, 3, // Triangle 4: right, bottom-left, bottom-right
    ];

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
