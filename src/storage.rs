pub mod buffer {
    use glam::f32;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Vertex {
        pub position: f32::Vec3,
        pub uvs: f32::Vec2,
    }

    unsafe impl bytemuck::Pod for Vertex {}
    unsafe impl bytemuck::Zeroable for Vertex {}

    impl Vertex {
        const ATTRIBS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

        pub fn desc() -> wgpu::VertexBufferLayout<'static> {
            use std::mem;

            wgpu::VertexBufferLayout {
                array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &Self::ATTRIBS,
            }
        }
    }
}

pub mod instance {
    use encase::ShaderType;
    use glam::f32;

    #[repr(C)]
    #[derive(Copy, Clone, ShaderType)]
    pub struct Instance {
        pub position: f32::Vec3,
        pub color: Color,
    }

    unsafe impl bytemuck::Pod for Instance {}
    unsafe impl bytemuck::Zeroable for Instance {}

    #[repr(C)]
    #[derive(Copy, Clone, ShaderType)]
    pub struct Color {
        pub value: u32,
    }

    unsafe impl bytemuck::Pod for Color {}
    unsafe impl bytemuck::Zeroable for Color {}

    impl Color {
        pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
            let value = ((a as u32) << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32);
            Self { value }
        }

        pub fn rgba(&self) -> (u8, u8, u8, u8) {
            let r = (self.value & 0x000000FF) as u8;
            let g = ((self.value & 0x0000FF00) >> 8) as u8;
            let b = ((self.value & 0x00FF0000) >> 16) as u8;
            let a = ((self.value & 0xFF000000) >> 24) as u8;
            (r, g, b, a)
        }
    }
}

pub mod uniform {
    use encase::ShaderType;
    use glam::f32;

    #[repr(C)]
    #[derive(Copy, Clone, ShaderType)]
    pub struct Camera {
        pub focal_view: f32::Vec3,
        pub world_space_position: f32::Vec3,
        pub view_matrix: f32::Mat4,
        pub projection_matrix: f32::Mat4,
        pub near_clip: f32,
        pub far_clip: f32,
    }

    unsafe impl bytemuck::Pod for Camera {}
    unsafe impl bytemuck::Zeroable for Camera {}

    #[repr(C)]
    #[derive(Copy, Clone, ShaderType)]
    pub struct Globals {
        pub frame: u32,
        pub time_since_start: f32,
        pub time_since_last_frame: f32,
    }

    unsafe impl bytemuck::Pod for Globals {}
    unsafe impl bytemuck::Zeroable for Globals {}
}
