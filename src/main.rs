use std::path::PathBuf;

use clap::{Parser, Subcommand, arg, command};

mod app;
mod controller;
mod layer;
mod pipeline;
mod renderer;
mod storage;
mod types;

/// A point-cloud renderer for lidar data generated for city surveying and mapping.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the renderer
    Visualize {
        /// Paths to the LIDAR data files (ASPRS formatted .las files)
        #[arg(required = true)]
        data: Vec<PathBuf>,
    },
}

fn main() {
    env_logger::Builder::new().parse_env("CLOUDCITY_LOG").init();
    app::run().expect("Error running application");

    // let args = Args::parse();

    // match args.command {
    //     Command::Visualize { .. } => {
    //         app::run().expect("Error running application");
    //     }
    // }
}
