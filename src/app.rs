use std::{collections::VecDeque, sync::Arc};

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    controller::CameraController,
    layer::{Layer, gui::GuiLayer, scene::SceneLayer},
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

            let renderer = Renderer::try_new(window.clone()).expect("Failed to create renderer");

            let layers = Layers {
                scene_layer: SceneLayer::try_new(&renderer.graphics())
                    .expect("Failed to create scene layer"),
                gui_layer: GuiLayer::try_new(&renderer.graphics(), window.clone())
                    .expect("Failed to create gui layer"),
            };

            State {
                window,
                renderer,
                layers,
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

                    let render_result =
                        state
                            .renderer
                            .render_with(&camera, &timings, |mut context| {
                                context.render(&mut state.layers.scene_layer)?;
                                context.render(&mut state.layers.gui_layer)?;
                                Ok(())
                            });

                    match render_result {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Rendering error: {:?}", e);
                            event_loop.exit();
                        }
                    }

                    // Handle debug events
                    while let Some(debug_event) = state.layers.scene_layer.poll_debug_event() {
                        state.layers.gui_layer.handle_debug_event(debug_event);
                    }
                }

                state.window.request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                state.renderer.resize_surface(new_size);

                state
                    .layers
                    .scene_layer
                    .resize(&state.renderer.graphics(), new_size);

                state
                    .layers
                    .gui_layer
                    .resize(&state.renderer.graphics(), new_size);
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
    layers: Layers,
    camera_controller: CameraController,
    frame_timer: FrameTimer,
}

pub struct Layers {
    scene_layer: SceneLayer,
    gui_layer: GuiLayer,
}

struct FrameTimer {
    frame: u32,
    start_time: std::time::Instant,
    last_frame_time: Option<std::time::Instant>,
    average_frame_times: VecDeque<std::time::Duration>,
}

impl FrameTimer {
    const RUNNING_AVERAGE_FRAME_TIMES_COUNT: usize = 30;

    pub fn new() -> Self {
        Self {
            frame: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: None,
            average_frame_times: VecDeque::new(),
        }
    }

    pub fn tick(&mut self) -> Option<types::Timings> {
        let timings = if let Some(last_frame_time) = self.last_frame_time {
            let elapsed_frame_time = last_frame_time.elapsed();

            // Store the last 10 frame times
            self.average_frame_times.push_back(elapsed_frame_time);
            if self.average_frame_times.len() > Self::RUNNING_AVERAGE_FRAME_TIMES_COUNT {
                self.average_frame_times.pop_front();
            }

            // Calculate the average frame time
            let average_frame_time = self.average_frame_times.iter().sum::<std::time::Duration>()
                / self.average_frame_times.len() as u32;

            let average_fps = 1.0 / average_frame_time.as_secs_f32();

            Some(types::Timings {
                frame: self.frame,
                average_fps,
                average_frame_time,
                time_since_start: self.start_time.elapsed(),
                time_since_last_frame: elapsed_frame_time,
            })
        } else {
            None
        };

        self.frame = self.frame.wrapping_add(1);
        self.last_frame_time = Some(std::time::Instant::now());

        timings
    }
}
