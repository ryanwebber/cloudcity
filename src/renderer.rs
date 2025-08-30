use crossbeam::channel::{Receiver, Sender};
use glam::Vec4;
use pollster::FutureExt;
use std::sync::Arc;
use wgpu;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{
    pipelines,
    storage::{self},
    types,
};

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    hexagon: Hexagon,
    instance_count: usize,
    // Textures
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    // Bind groups
    culling_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    // Pipelines
    render_pipeline: pipelines::RenderPipeline,
    culling_pipeline: pipelines::CullingPipeline,
    // Buffers
    camera_uniforms: wgpu::Buffer,
    point_positions_buffer: wgpu::Buffer,
    visibility_buffer: wgpu::Buffer,
    indirect_draw_args_buffer: wgpu::Buffer,
    compacted_indices_buffer: wgpu::Buffer,
    culling_stats_buffer: wgpu::Buffer,
    // Buffered readbacks
    debug_rx: Receiver<DebugEvent>,
    debug_tx: Sender<DebugEvent>,
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

        let hexagon = Hexagon::create(&device);
        let render_pipeline = pipelines::RenderPipeline::new(&device, surface_format);
        let culling_pipeline = pipelines::CullingPipeline::new(&device);

        // Create instance data with random points
        let instances = Self::create_random_instances(5000); // Increased from 1000 to 5000
        let instance_count = instances.len();

        log::debug!("Initial instance count: {}", instance_count);

        // Create camera uniforms buffer
        let camera_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms"),
            size: std::mem::size_of::<storage::uniform::Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create GPU culling buffers
        let point_positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Point Positions Buffer"),
            size: (std::mem::size_of::<glam::f32::Vec4>() * instance_count) as u64, // Vec4 for alignment
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visibility Buffer"),
            size: (std::mem::size_of::<u32>() * instance_count) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_draw_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Draw Args Buffer"),
            size: 32, // 5 u32 values for indirect draw, aligned to 4 bytes
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let culling_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Culling Stats Buffer"),
            size: std::mem::size_of::<[u32; 3]>() as u64, // total, visible, culled
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compacted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compacted Indices Buffer"),
            size: (std::mem::size_of::<u32>() * instance_count) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create instance buffer for rendering (contains all instances)
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<storage::instance::Instance>() * instance_count) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Populate the instance buffer with all instances
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));

        // Populate the point positions buffer
        let point_positions: Vec<glam::f32::Vec4> = instances
            .iter()
            .map(|i| Vec4::new(i.position.x, i.position.y, i.position.z, 0.0))
            .collect();

        queue.write_buffer(
            &point_positions_buffer,
            0,
            bytemuck::cast_slice(&point_positions[..]),
        );

        // Initialize indirect draw args buffer
        let initial_draw_args = [12u32, 0u32, 0u32, 0u32, 0u32]; // vertex_count, instance_count, first_index, base_vertex, first_instance
        queue.write_buffer(
            &indirect_draw_args_buffer,
            0,
            bytemuck::cast_slice(&initial_draw_args),
        );

        // Initialize culling stats buffer
        let initial_stats = [instance_count as u32, 0u32, 0u32]; // total, visible, culled
        queue.write_buffer(
            &culling_stats_buffer,
            0,
            bytemuck::cast_slice(&initial_stats),
        );

        // Initialize compacted indices buffer
        queue.write_buffer(
            &compacted_indices_buffer,
            0,
            bytemuck::cast_slice(&vec![0u32; instance_count]),
        );

        // Create bind group for rendering
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: compacted_indices_buffer.as_entire_binding(),
                },
            ],
        });

        // Create bind group for culling
        let culling_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Culling Bind Group"),
            layout: &culling_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: point_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirect_draw_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: compacted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: culling_stats_buffer.as_entire_binding(),
                },
            ],
        });

        surface.configure(&device, &surface_config);

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
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
        let (debug_tx, debug_rx) = crossbeam::channel::bounded(1);

        Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            hexagon,
            instance_count,
            depth_texture,
            depth_texture_view,
            culling_bind_group,
            render_bind_group,
            render_pipeline,
            culling_pipeline,
            camera_uniforms,
            point_positions_buffer,
            visibility_buffer,
            indirect_draw_args_buffer,
            compacted_indices_buffer,
            culling_stats_buffer,
            debug_rx,
            debug_tx,
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
        // Update camera uniforms for GPU culling
        self.update_camera_uniforms(camera);

        // Perform frustum culling to get visible instances
        self.update_visible_instances();

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
                                r: 0.01,
                                g: 0.02,
                                b: 0.03,
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
            render_pass.set_pipeline(&self.render_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);

            // Set vertex and index buffers
            render_pass.set_vertex_buffer(0, self.hexagon.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.hexagon.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );

            // Draw using indirect rendering with GPU-computed draw args
            // The compute shader has already updated indirect_draw_args with the correct visible count
            // and the vertex shader will use compacted_indices[gl_InstanceIndex] to get the right instance
            render_pass.draw_indexed_indirect(&self.indirect_draw_args_buffer, 0);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn create_debug_receiver(&self) -> Receiver<DebugEvent> {
        self.debug_rx.clone()
    }

    fn update_camera_uniforms(&mut self, camera: &types::Camera) {
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
        // FIXED: Add missing Z-axis rotation but keep original multiplication order for compatibility
        // Original order: Y * X (yaw * pitch) - this matches the existing camera controls
        let rotation_matrix = glam::f32::Mat4::from_rotation_y(camera.transform.rotation.y)
            * glam::f32::Mat4::from_rotation_x(camera.transform.rotation.x)
            * glam::f32::Mat4::from_rotation_z(camera.transform.rotation.z);

        // Use transform_vector3 for direction vectors, not transform_point3
        let forward = rotation_matrix.transform_vector3(glam::f32::vec3(0.0, 0.0, -1.0));

        // Create view matrix using camera position and calculated forward direction
        let view = glam::f32::Mat4::look_at_rh(
            camera.transform.position,
            camera.transform.position + forward,
            glam::f32::vec3(0.0, 1.0, 0.0),
        );

        // Create camera uniforms struct
        let camera_uniforms = storage::uniform::Camera {
            focal_view: forward, // Use calculated forward direction
            world_space_position: camera.transform.position,
            view_matrix: view,
            projection_matrix: projection,
            near_clip: camera.clipping.near,
            far_clip: camera.clipping.far,
            fov: fov, // Pass the FOV for accurate frustum calculation
        };

        // Update the buffer
        self.queue.write_buffer(
            &self.camera_uniforms,
            0,
            bytemuck::cast_slice(&[camera_uniforms]),
        );

        // Recreate the culling bind group with updated camera uniforms
        // The old bind group was pointing to stale camera data
        self.culling_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Culling Bind Group"),
            layout: &self.culling_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.point_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.indirect_draw_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.compacted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.culling_stats_buffer.as_entire_binding(),
                },
            ],
        });
    }

    fn update_visible_instances(&mut self) {
        // Reset culling stats buffer
        let reset_stats = [self.instance_count as u32, 0u32, 0u32];
        self.queue.write_buffer(
            &self.culling_stats_buffer,
            0,
            bytemuck::cast_slice(&reset_stats),
        );

        // Create command encoder for compute operations
        let mut compute_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Culling Compute Encoder"),
                });

        // Set up compute pass for culling
        {
            let mut compute_pass =
                compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Culling Compute Pass"),
                    timestamp_writes: None,
                });

            // Bind the culling pipeline and bind group
            compute_pass.set_pipeline(&self.culling_pipeline.cull_pipeline);
            compute_pass.set_bind_group(0, &self.culling_bind_group, &[]);

            // Dispatch culling compute shader
            let workgroup_count = (self.instance_count + 63) / 64; // 64 threads per workgroup
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Submit compute commands
        self.queue.submit(std::iter::once(compute_encoder.finish()));

        // Now run the compaction shader to update indirect draw args
        let mut compact_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compaction Compute Encoder"),
                });

        {
            let mut compute_pass =
                compact_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compaction Compute Pass"),
                    timestamp_writes: None,
                });

            // Bind the compaction pipeline and bind group
            compute_pass.set_pipeline(&self.culling_pipeline.compact_pipeline);
            compute_pass.set_bind_group(0, &self.culling_bind_group, &[]);

            // Dispatch compaction compute shader
            let workgroup_count = (self.instance_count + 63) / 64; // 64 threads per workgroup
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Submit compaction commands
        self.queue.submit(std::iter::once(compact_encoder.finish()));

        {
            // Processing culling stats readback
            let culling_stats_readback_buffer =
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Culling Stats Readback Buffer"),
                    size: std::mem::size_of::<storage::uniform::CullingStats>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

            let mut readback_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Readback Encoder"),
                    });

            // Copy the culling stats buffer to the host
            readback_encoder.copy_buffer_to_buffer(
                &self.culling_stats_buffer,
                0,
                &culling_stats_readback_buffer,
                0,
                std::mem::size_of::<storage::uniform::CullingStats>() as u64,
            );

            self.queue
                .submit(std::iter::once(readback_encoder.finish()));

            let stats_tx_clone = self.debug_tx.clone();
            let culling_stats_readback_buffer = Arc::new(culling_stats_readback_buffer);
            culling_stats_readback_buffer.clone().slice(..).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    if let Ok(..) = result {
                        let bytes = culling_stats_readback_buffer.slice(..).get_mapped_range();
                        let stats: &storage::uniform::CullingStats = bytemuck::from_bytes(&bytes);
                        _ = stats_tx_clone.try_send(DebugEvent::CullingStats {
                            total_points: stats.total_points as usize,
                            visible_points: stats.visible_points as usize,
                        });
                    }
                },
            );
        }
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

#[derive(Debug, Clone)]
pub enum DebugEvent {
    CullingStats {
        total_points: usize,
        visible_points: usize,
    },
}
