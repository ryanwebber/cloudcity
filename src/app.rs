use std::sync::Arc;

use glam::f32;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    renderer::Renderer,
    types::{self, Camera, Clipping, Lens, Transform},
};

pub fn run() -> anyhow::Result<()> {
    let event_loop: EventLoop<()> = EventLoop::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<()> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let state = self.state.get_or_insert_with(|| {
            log::debug!("Initializing application state...");
            let size = LogicalSize::new(800, 600);
            let attributes = Window::default_attributes()
                .with_title(env!("CARGO_PKG_NAME"))
                .with_inner_size(size);

            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .expect("Failed to create window"),
            );

            let renderer = Renderer::try_new(window.clone()).expect("Failed to create renderer");
            let frame_timer = FrameTimer::new();
            let camera = Camera {
                lens: Lens::Perspective {
                    fov: 50.0,
                    focal_distance: 10.5,
                },
                transform: Transform {
                    position: f32::Vec3::new(0.0, 0.0, 0.0),
                    rotation: f32::Vec3::new(0.0, 0.0, 0.0),
                    scale: f32::Vec3::new(1.0, 1.0, 1.0),
                },
                clipping: Clipping {
                    near: 0.1,
                    far: 2000.0,
                },
            };

            State {
                window,
                renderer,
                camera,
                frame_timer,
            }
        });

        _ = state;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        if state.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // First frame, just record the time and request another redraw
                // rather than having a big emulation chunk up front
                if let Some(timings) = state.frame_timer.tick() {
                    match state.renderer.render(&state.camera, &timings) {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Rendering error: {:?}", e);
                            event_loop.exit();
                        }
                    }
                }

                state.window.request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                state.renderer.resize_surface(new_size);
            }
            _ => (),
        }
    }
}

pub struct State {
    window: Arc<Window>,
    renderer: Renderer,
    camera: Camera,
    frame_timer: FrameTimer,
}

struct FrameTimer {
    frame: u32,
    start_time: std::time::Instant,
    last_frame_time: Option<std::time::Instant>,
}

impl FrameTimer {
    pub fn new() -> Self {
        Self {
            frame: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: None,
        }
    }

    pub fn tick(&mut self) -> Option<types::Timings> {
        let timings = if let Some(last_frame_time) = self.last_frame_time {
            Some(types::Timings {
                frame: self.frame,
                time_since_start: self.start_time.elapsed(),
                time_since_last_frame: last_frame_time.elapsed(),
            })
        } else {
            None
        };

        self.frame = self.frame.wrapping_add(1);
        self.last_frame_time = Some(std::time::Instant::now());

        timings
    }
}
