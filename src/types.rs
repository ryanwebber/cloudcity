use glam::{Mat3, f32};

#[derive(Clone, PartialEq, Debug)]
pub struct Camera {
    pub transform: Transform,
    pub lens: Lens,
    pub clipping: Clipping,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Clipping {
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Lens {
    Perspective { fov: f32, focal_distance: f32 },
}

#[derive(Clone, PartialEq, Debug)]
pub struct Transform {
    pub position: f32::Vec3,
    pub rotation: f32::Vec3,
    pub scale: f32::Vec3,
}

impl Into<Transform> for glam::Mat4 {
    fn into(self) -> Transform {
        let (scale, rotation, position) = self.to_scale_rotation_translation();
        let rotation = rotation.to_euler(glam::EulerRot::XYZ);
        let rotation = glam::f32::Vec3::new(rotation.0, rotation.1, rotation.2);

        Transform {
            position,
            rotation,
            scale,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Timings {
    pub frame: u32,
    pub time_since_start: std::time::Duration,
    pub time_since_last_frame: std::time::Duration,
}
