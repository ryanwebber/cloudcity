use std::sync::Arc;

use crossbeam::channel::Receiver;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    controller::CameraController,
    layer::scene::{DebugEvent, SceneLayer},
    renderer::Renderer,
    types,
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

            let frame_timer = FrameTimer::new();
            let camera_controller = CameraController::new();

            let mut renderer =
                Renderer::try_new(window.clone()).expect("Failed to create renderer");

            let scene_layer =
                SceneLayer::try_new(&renderer.graphics()).expect("Failed to create scene layer");

            let debug_receiver = scene_layer.debug_rx().clone();

            renderer.add_layer(Box::new(scene_layer));

            State {
                window,
                renderer,
                debug_receiver,
                camera_controller,
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

        // Handle camera controller events first
        state
            .camera_controller
            .handle_window_event(&event, &state.window);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Update camera controller with frame timing
                if let Some(timings) = state.frame_timer.tick() {
                    // Update camera controller
                    state
                        .camera_controller
                        .update(timings.time_since_last_frame);

                    // Get current camera from controller
                    let camera = state.camera_controller.get_camera();

                    match state.renderer.render(&camera, &timings) {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Rendering error: {:?}", e);
                            event_loop.exit();
                        }
                    }

                    while let Ok(debug_event) = state.debug_receiver.try_recv() {
                        match debug_event {
                            DebugEvent::CullingStats {
                                total_points,
                                visible_points,
                            } => {
                                let title = format!(
                                    "{} - Points: {}/{} ({:.1}% culled)",
                                    env!("CARGO_PKG_NAME"),
                                    visible_points,
                                    total_points,
                                    (total_points - visible_points) as f32 / total_points as f32
                                        * 100.0
                                );
                                state.window.set_title(&title);
                            }
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        // Handle raw mouse events for camera control when cursor is locked
        state.camera_controller.handle_device_event(&event);
    }
}

pub struct State {
    window: Arc<Window>,
    renderer: Renderer,
    debug_receiver: Receiver<DebugEvent>,
    camera_controller: CameraController,
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
