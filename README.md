# CloudCity

A point-cloud renderer for lidar data generated for city surveying and mapping.

## Getting Started

I built this specifically to render the LAS lidar survey data of the city of Vancouver, BC (Canada),
which is publically available [here](https://opendata.vancouver.ca/explore/dataset/lidar-2022/information/?location=14,49.27461,-123.1306&basemap=jawg.streets).

With such data downloaded and decompressed, the renderer can be run with:

```sh
cargo run --release -- visualize </path/to/las/file>
```

Thanks to being a GPU renderer, I can load about 8,000,000 points at a time and still render at 120fps
before I actually exceed memory limits on my platform. If streaming point data from files into and
out of the GPU was implemented, it could theoretically support an infinite amount data at the level of detail
available from this data source at buttery smooth frame rates.

### Movement
- **W/A/S/D**: Move forward/left/backward/right in the direction you're looking
- **Space**: Move up along the global Y-axis
- **Left Shift**: Move down along the global Y-axis
- **Mouse Wheel**: Scroll up to increase movement speed, scroll down to decrease movement speed

### Look Around
- **Left Mouse Button**: Enter cursor lock
- **Escape**: Exit cursor lock
