struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uvs: vec2<f32>,
}

struct InstanceInput {
    @location(2) position: vec3<f32>,
    @location(3) color: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) model_pos: vec3<f32>,
}

struct CameraUniforms {
    focal_view: vec3<f32>,
    world_space_position: vec3<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    near_clip: f32,
    far_clip: f32,
    fov: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(0) @binding(1)
var<storage, read> instances: array<InstanceInput>;

@group(0) @binding(2)
var<storage, read> compacted_indices: array<u32>;

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // Access the instance data using the actual index
    let compact_index = compacted_indices[instance_index];
    let instance_data = instances[compact_index];
    var instance_position = instance_data.position;
    let instance_color = instance_data.color;

    // Point size in world units
    let point_size = 0.1;
    
    // Transform instance position to view space
    let view_pos = camera.view_matrix * vec4<f32>(instance_position, 1.0);
    
    // Create billboard by offsetting the quad vertices in view space
    // This ensures the quad always faces the camera
    let billboard_offset = vec3<f32>(
        model.position.x * point_size,
        model.position.y * point_size,
        0.0
    );
    
    // Add billboard offset to view position
    let final_view_pos = view_pos.xyz + billboard_offset;
    
    // Apply projection matrix
    let clip_pos = camera.projection_matrix * vec4<f32>(final_view_pos, 1.0);
    
    // Convert color from packed u32 to vec4
    let r = f32(instance_color & 0x000000FFu) / 255.0;
    let g = f32((instance_color & 0x0000FF00u) >> 8u) / 255.0;
    let b = f32((instance_color & 0x00FF0000u) >> 16u) / 255.0;

    let color = vec4<f32>(r, g, b, 1.0);

    return VertexOutput(
        clip_pos,
        color,
        model.position
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from the center of the hexagon (0,0,0 in model space)
    let distance = length(input.model_pos);

    // Discard fragments outside the circle (radius 0.5 in model space)
    if distance > 0.5 {
        discard;
    }

    return input.color;
}
