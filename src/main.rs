use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, arg, command};
use glam::Vec3;

mod app;
mod controller;
mod layer;
mod loader;
mod pipeline;
mod renderer;
mod storage;
mod types;

/// A point-cloud renderer for lidar data generated for city surveying and mapping.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the renderer
    Visualize {
        /// Path to the LIDAR data file (ASPRS formatted .las file)
        #[arg(required = true)]
        data: PathBuf,

        #[command(flatten)]
        options: VisualizationOptions,
    },

    /// Inspect the point cloud dataset
    Inspect {
        /// Paths to the LIDAR data files (ASPRS formatted .las files)
        #[arg(required = true)]
        data: Vec<PathBuf>,
    },
}

#[derive(Debug, Args)]
struct VisualizationOptions {
    /// Offset to the first point in the data file to visualize
    #[arg(long, required = false, default_value_t = 0)]
    offset: usize,

    /// Mazimium number of points to visualize
    #[arg(long, required = false, default_value_t = 8_000_000)]
    max_count: usize,

    /// The default scale factor for the points
    #[arg(long, required = false, default_value_t = 10.0)]
    scale: f32,

    /// Amount of random jitter to apply to the points
    #[arg(long, required = false, default_value_t = 1.0)]
    jitter: f32,

    /// The maximum view range in meters
    #[arg(long, required = false)]
    far_clip: Option<f32>,

    /// The minimum view range in meters
    #[arg(long, required = false)]
    near_clip: Option<f32>,
}

fn main() {
    env_logger::Builder::new().parse_env("CLOUDCITY_LOG").init();

    let args = Arguments::parse();

    log::info!("Initial arguments: {:#?}", args);

    match args.command {
        Command::Visualize { data, options } => {
            let points = loader::load_points(&data, &options).expect("Error loading points");

            if points.is_empty() {
                eprintln!("No points found in the LAS file");
                return;
            }

            let camera_hint = types::CameraHints {
                origin: Some(points[points.len() / 3].position + Vec3::new(0.0, 100.0, 10.0)),
                near_clip: options.near_clip,
                far_clip: options.far_clip,
            };

            app::run(points, camera_hint).expect("Error running application");
        }
        Command::Inspect { data } => {
            for path in data {
                let reader = las::Reader::from_path(&path).expect("Error reading LAS file");
                let header = reader.header();
                println!("Header: {:#?}", header);
            }
        }
    }
}
