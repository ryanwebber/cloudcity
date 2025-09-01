use crate::{VisualizationOptions, types::RenderPoint};
use std::path::Path;

pub fn load_points(
    data: &Path,
    options: &VisualizationOptions,
) -> anyhow::Result<Vec<RenderPoint>> {
    let mut reader = las::Reader::from_path(&data).expect("Error reading LAS file");
    let header = reader.header();
    let origin = {
        let max = header.bounds().max;
        let min = header.bounds().min;
        let center_x = (max.x + min.x) / 2.0;
        let center_y = (max.y + min.y) / 2.0;
        let center_z = max.z;
        glam::f32::vec3(center_x as f32, center_y as f32, center_z as f32)
    };

    log::debug!("Header: {:#?}", header);

    log::debug!("Seeking to offset: {}", options.offset);
    if let Err(e) = reader.seek(options.offset as u64) {
        log::error!("Error seeking to offset: {}", e);
    }

    let mut rng = rand::thread_rng();
    let mut points = vec![];

    for (i, point) in reader.points().take(options.max_count).enumerate() {
        if let Ok(point) = point {
            if i == 0 {
                log::debug!("Points[0] = {:#?}", point);
            }

            // Load the position
            let position = glam::f32::vec3(point.x as f32, point.y as f32, point.z as f32);

            // Offset and scale it
            let position = (position - origin) * options.scale;

            // Switch the Z axis and Y axis
            let position = glam::f32::vec3(-position.x, position.z, position.y);

            // Wiggle each point by a random amount
            let position = position
                + glam::f32::vec3(
                    rand::Rng::gen_range(&mut rng, -options.jitter..options.jitter),
                    rand::Rng::gen_range(&mut rng, -options.jitter..options.jitter),
                    rand::Rng::gen_range(&mut rng, -options.jitter..options.jitter),
                );

            let point = RenderPoint {
                position,
                color: point
                    .color
                    .map(|color| {
                        let red_f = color.red as f32 / u16::MAX as f32;
                        let green_f = color.green as f32 / u16::MAX as f32;
                        let blue_f = color.blue as f32 / u16::MAX as f32;
                        glam::U8Vec4::new(
                            (red_f * 255.0) as u8,
                            (green_f * 255.0) as u8,
                            (blue_f * 255.0) as u8,
                            u8::MAX,
                        )
                    })
                    .unwrap_or_else(|| glam::U8Vec4::new(255, 0, 255, 255)),
            };

            points.push(point);
        }
    }
    Ok(points)
}
