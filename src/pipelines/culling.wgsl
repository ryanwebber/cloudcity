// GPU-based frustum culling compute shader
// This shader processes all points in parallel to determine visibility

// Input buffers
@group(0) @binding(0) var<storage, read> point_positions: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> camera_uniforms: CameraUniforms;
@group(0) @binding(2) var<storage, read_write> visibility_buffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect_draw_args: IndirectDrawArgs;
@group(0) @binding(4) var<storage, read_write> culling_stats: CullingStats;
@group(0) @binding(5) var<storage, read_write> compacted_indices: array<u32>;

// Camera uniforms structure
struct CameraUniforms {
    focal_view: vec3<f32>,
    world_space_position: vec3<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    near_clip: f32,
    far_clip: f32,
    fov: f32,
}

// Indirect draw arguments for the render pass
struct IndirectDrawArgs {
    vertex_count: u32,      // Number of vertices per instance (12 for hexagon)
    instance_count: u32,    // Number of visible instances
    first_index: u32,       // First index to use (0)
    base_vertex: u32,       // Base vertex offset (0)
    first_instance: u32,    // First instance index (0)
}

// Culling statistics for debugging - using atomic types
struct CullingStats {
    total_points: atomic<u32>,
    visible_points: atomic<u32>,
    culled_points: atomic<u32>,
}

fn point_in_frustum(point: vec3<f32>, camera_uniform: CameraUniforms) -> bool {
    // 1. Extend to homogeneous coordinate
    let p_world = vec4<f32>(point, 1.0);

    // 2. Transform to clip space
    let p_view = camera_uniform.view_matrix * p_world;
    let p_clip = camera_uniform.projection_matrix * p_view;

    // 3. Test against clip-space bounds (OpenGL-style)
    // If you're using WebGPU/WGSL, depth is 0..1 (DirectX/Vulkan-style),
    // so z uses [0, w] instead of [-w, w].
    let x_ok = (-p_clip.w <= p_clip.x) && (p_clip.x <= p_clip.w);
    let y_ok = (-p_clip.w <= p_clip.y) && (p_clip.y <= p_clip.w);
    let z_ok = (0.0 <= p_clip.z) && (p_clip.z <= p_clip.w);

    return x_ok && y_ok && z_ok;
}

@compute @workgroup_size(64)
fn cull_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_index = global_id.x;
    
    // Check if this thread is processing a valid point
    if (point_index >= arrayLength(&point_positions)) {
        return;
    }
    
    // Get point position
    let point = point_positions[point_index];

    // Check if point is visible
    let is_visible = point_in_frustum(point, camera_uniforms);
    
    // Set visibility (1 = visible, 0 = culled)
    visibility_buffer[point_index] = select(0u, 1u, is_visible);
    
    // Update statistics atomically
    if (is_visible) {
        atomicAdd(&culling_stats.visible_points, 1u);
    } else {
        atomicAdd(&culling_stats.culled_points, 1u);
    }
    
    // Set total points count (only once, but doesn't matter if done multiple times)
    atomicStore(&culling_stats.total_points, arrayLength(&point_positions));
}

// Second pass: compact visible points and update indirect draw args
@compute @workgroup_size(64)
fn compact_visible_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_index = global_id.x;
    
    if (point_index >= arrayLength(&point_positions)) {
        return;
    }
    
    // Only the first thread should update the indirect draw args
    if (point_index == 0u) {
        // Count visible points and build compacted index buffer
        var visible_count = 0u;
        for (var i = 0u; i < arrayLength(&point_positions); i++) {
            if (visibility_buffer[i] == 1u) {
                // Write the index of this visible point to the compacted buffer
                // This ensures we render the actual visible points, not just the first N
                compacted_indices[visible_count] = i;
                visible_count = visible_count + 1u;
            }
        }
        
        // Update indirect draw args with the correct values
        indirect_draw_args.vertex_count = 12u; // Hexagon has 12 vertices
        indirect_draw_args.instance_count = visible_count; // Count of actually visible points
        indirect_draw_args.first_index = 0u;
        indirect_draw_args.base_vertex = 0u;
        indirect_draw_args.first_instance = 0u;
    }
}
