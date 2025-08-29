use glam::f32::{Mat4, Quat, Vec3};
use std::time::Duration;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};

use crate::types::{Camera, Clipping, Lens, Transform};

pub struct CameraController {
    // Camera state
    position: Vec3,
    rotation: Vec3,

    // Movement state
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,

    // Mouse state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,

    // Camera settings
    move_speed: f32,
    mouse_sensitivity: f32,

    // Camera properties
    fov: f32,
    focal_distance: f32,
    near: f32,
    far: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),

            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,

            mouse_pressed: false,
            last_mouse_pos: None,

            move_speed: 5.0,
            mouse_sensitivity: 0.002,

            fov: 50.0,
            focal_distance: 10.5,
            near: 0.1,
            far: 2000.0,
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;

                match logical_key.as_ref() {
                    winit::keyboard::Key::Character("w") | winit::keyboard::Key::Character("W") => {
                        self.forward = pressed
                    }
                    winit::keyboard::Key::Character("s") | winit::keyboard::Key::Character("S") => {
                        self.backward = pressed
                    }
                    winit::keyboard::Key::Character("a") | winit::keyboard::Key::Character("A") => {
                        self.left = pressed
                    }
                    winit::keyboard::Key::Character("d") | winit::keyboard::Key::Character("D") => {
                        self.right = pressed
                    }
                    winit::keyboard::Key::Character(" ") => self.up = pressed,
                    winit::keyboard::Key::Character("shift") => self.down = pressed,
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_mouse_pos = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    let (x, y) = (position.x, position.y);

                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let delta_x = (x - last_x) as f32 * self.mouse_sensitivity;
                        let delta_y = (y - last_y) as f32 * self.mouse_sensitivity;

                        // Update rotation (yaw and pitch)
                        self.rotation.y -= delta_x;
                        self.rotation.x -= delta_y;

                        // Clamp pitch to prevent gimbal lock
                        self.rotation.x = self.rotation.x.clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.01,
                            std::f32::consts::FRAC_PI_2 - 0.01,
                        );
                    }

                    self.last_mouse_pos = Some((x, y));
                }
            }

            _ => {}
        }
    }

    pub fn update(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();
        let move_distance = self.move_speed * dt;

        // Calculate forward and right vectors based on current rotation
        let forward_vec = self.get_forward_vector();
        let right_vec = self.get_right_vector();
        let up_vec = Vec3::new(0.0, 1.0, 0.0);

        // Update position based on input
        if self.forward {
            self.position += forward_vec * move_distance;
        }
        if self.backward {
            self.position -= forward_vec * move_distance;
        }
        if self.left {
            self.position -= right_vec * move_distance;
        }
        if self.right {
            self.position += right_vec * move_distance;
        }
        if self.up {
            self.position += up_vec * move_distance;
        }
        if self.down {
            self.position -= up_vec * move_distance;
        }
    }

    pub fn get_camera(&self) -> Camera {
        Camera {
            transform: Transform {
                position: self.position,
                rotation: self.rotation,
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            lens: Lens::Perspective {
                fov: self.fov,
                focal_distance: self.focal_distance,
            },
            clipping: Clipping {
                near: self.near,
                far: self.far,
            },
        }
    }

    fn get_forward_vector(&self) -> Vec3 {
        let rotation_matrix = Mat4::from_quat(Quat::from_euler(
            glam::EulerRot::XYZ,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        ));

        (rotation_matrix * Vec3::new(0.0, 0.0, -1.0).extend(1.0))
            .truncate()
            .normalize()
    }

    fn get_right_vector(&self) -> Vec3 {
        let rotation_matrix = Mat4::from_quat(Quat::from_euler(
            glam::EulerRot::XYZ,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        ));

        (rotation_matrix * Vec3::new(1.0, 0.0, 0.0).extend(1.0))
            .truncate()
            .normalize()
    }
}
