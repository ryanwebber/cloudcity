use glam::f32::{Mat4, Quat, Vec3};
use std::time::Duration;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

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

    // Mouse smoothing and interpolation
    target_rotation: Vec3,
    rotation_smoothing: f32,

    // Cursor state
    cursor_locked: bool,

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
            target_rotation: Vec3::new(0.0, 0.0, 0.0),

            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,

            cursor_locked: false,

            move_speed: 5.0,
            mouse_sensitivity: 0.002, // Reduced sensitivity for smoother movement
            rotation_smoothing: 0.4,  // Slightly more smoothing for better feel

            fov: 50.0,
            focal_distance: 10.5,
            near: 0.1,
            far: 2000.0,
        }
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent, window: &Window) {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;

                match logical_key.as_ref() {
                    Key::Character("w") | Key::Character("W") => self.forward = pressed,
                    Key::Character("s") | Key::Character("S") => self.backward = pressed,
                    Key::Character("a") | Key::Character("A") => self.left = pressed,
                    Key::Character("d") | Key::Character("D") => self.right = pressed,
                    Key::Named(NamedKey::Space) => self.up = pressed,
                    Key::Named(NamedKey::Shift) => self.down = pressed,
                    Key::Named(NamedKey::Escape) => {
                        if pressed {
                            self.toggle_cursor_lock(window);
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                // Enter cursor lock when left mouse is pressed
                if !self.cursor_locked && *state == ElementState::Released {
                    self.lock_cursor(window);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Adjust movement speed with scroll wheel
                let speed_change = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y as f32 * 0.5,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };

                // Update movement speed (clamp to reasonable bounds)
                self.move_speed = (self.move_speed - speed_change).clamp(0.1, 50.0);
            }

            _ => {}
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        // Handle raw mouse events for camera control when cursor is locked
        if self.cursor_locked {
            match event {
                DeviceEvent::MouseMotion { delta } => {
                    let delta_x = delta.0 as f32 * self.mouse_sensitivity;
                    let delta_y = delta.1 as f32 * self.mouse_sensitivity;

                    // Update target rotation (yaw and pitch)
                    self.target_rotation.y -= delta_x;
                    self.target_rotation.x -= delta_y;

                    // Clamp pitch to prevent gimbal lock
                    self.target_rotation.x = self.target_rotation.x.clamp(
                        -std::f32::consts::FRAC_PI_2 + 0.1,
                        std::f32::consts::FRAC_PI_2 - 0.1,
                    );
                }
                _ => {}
            }
        }
    }

    fn toggle_cursor_lock(&mut self, window: &Window) {
        if self.cursor_locked {
            self.unlock_cursor(window);
        } else {
            self.lock_cursor(window);
        }
    }

    fn lock_cursor(&mut self, window: &Window) {
        self.cursor_locked = true;
        window.set_cursor_visible(false);
        if let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::Locked) {
            log::warn!("Failed to lock cursor: {:?}", e);
        }
    }

    fn unlock_cursor(&mut self, window: &Window) {
        self.cursor_locked = false;
        window.set_cursor_visible(true);
        if let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::None) {
            log::warn!("Failed to unlock cursor: {:?}", e);
        }
    }

    pub fn update(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();

        // Smooth rotation interpolation with frame-rate independent smoothing
        let smoothing_factor = (1.0 - self.rotation_smoothing).powf(dt * 60.0);
        self.rotation = self
            .rotation
            .lerp(self.target_rotation, 1.0 - smoothing_factor);

        let move_distance = self.move_speed * dt;

        // Calculate forward and right vectors based on current rotation
        let forward_vec = self.get_forward_vector();
        let right_vec = self.get_right_vector();
        let up_vec = Vec3::Y; // Use world up

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
