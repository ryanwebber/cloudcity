// GPU-based frustum culling compute shader
// This shader processes all points in parallel to determine visibility

// Input buffers
@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
@group(0) @binding(1) var<storage, read> point_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> visibility_buffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect_draw_args: IndirectDrawArgs;
@group(0) @binding(4) var<storage, read_write> compacted_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> culling_stats: CullingStats;

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
    index_count: u32,       // Number of vertex indices per instance (12 for hexagon)
    instance_count: u32,    // Number of visible instances
    first_index: u32,       // First index to use (0)
    base_vertex: i32,       // Base vertex offset (0)
    first_instance: u32,    // First instance index (0)
}

// Culling statistics for debugging - using atomic types for atomic operations
struct CullingStats {
    total_points: u32,
    visible_points: atomic<u32>,
}

fn point_in_frustum(point: vec3<f32>, camera_uniform: CameraUniforms) -> bool {
    // 1. Extend to homogeneous coordinate
    let p_world = vec4<f32>(point, 1.0);

    // 2. Transform to clip space
    let p_view = camera_uniform.view_matrix * p_world;
    let p_clip = camera_uniform.projection_matrix * p_view;

    // 3. Test against clip-space bounds
    let x_ok = (-p_clip.w <= p_clip.x) && (p_clip.x <= p_clip.w);
    let y_ok = (-p_clip.w <= p_clip.y) && (p_clip.y <= p_clip.w);
    let z_ok = (0.0 <= p_clip.z) && (p_clip.z <= p_clip.w);

    return x_ok && y_ok && z_ok;
}

@compute @workgroup_size(256)
fn cull_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_index = global_id.x;

    // Check if this thread is processing a valid point
    if (point_index >= arrayLength(&point_positions)) {
        return;
    }

    // Get point position
    let point = point_positions[point_index].xyz;

    // Check if point is visible
    let is_visible = point_in_frustum(point, camera_uniforms);

    // Set visibility (1 = visible, 0 = culled)
    visibility_buffer[point_index] = select(0u, 1u, is_visible);
}

// Two-pass parallel compaction to avoid race conditions
@compute @workgroup_size(256)
fn compact_visible_points(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let point_index = global_id.x;
    let point_count = arrayLength(&point_positions);

    if (point_index >= point_count) {
        return;
    }

    // PASS 1: Count visible points per workgroup and calculate global offsets
    if (local_id.x == 0u) {
        var workgroup_visible_count = 0u;
        let workgroup_start = workgroup_id.x * 256u;
        let workgroup_end = min(workgroup_start + 256u, point_count);

        // Count visible points in this workgroup
        for (var i = workgroup_start; i < workgroup_end; i++) {
            if (visibility_buffer[i] == 1u) {
                workgroup_visible_count = workgroup_visible_count + 1u;
            }
        }

        // Use atomic operations to accumulate total visible count across workgroups
        // This gives us the global offset for this workgroup
        let global_offset = atomicAdd(&culling_stats.visible_points, workgroup_visible_count);
        
        // Store the global offset in the visibility buffer temporarily
        // We'll use the high bit to mark this as an offset value
        visibility_buffer[workgroup_start] = global_offset | 0x80000000u;
    }

    // Wait for all workgroups to complete their counting
    workgroupBarrier();

    // PASS 2: Compact visible points using the calculated offsets
    if (visibility_buffer[point_index] == 1u) {
        // Find the workgroup this point belongs to
        let workgroup_id_for_point = point_index / 256u;
        let workgroup_start = workgroup_id_for_point * 256u;

        // Get the global offset for this workgroup (stored in the first element)
        let global_offset = visibility_buffer[workgroup_start] & 0x7FFFFFFFu;

        // Calculate local offset within this workgroup
        var local_offset = 0u;
        for (var i = workgroup_start; i < point_index; i++) {
            if (visibility_buffer[i] == 1u) {
                local_offset = local_offset + 1u;
            }
        }

        // Calculate final position in compacted buffer
        let final_offset = global_offset + local_offset;

        if (final_offset < point_count) {
            compacted_indices[final_offset] = point_index;
        }
    }

    workgroupBarrier();

    // Only the first thread updates the indirect draw args and final stats
    if (global_id.x == 0u) {
        // Update the culling stats
        culling_stats.total_points = point_count;

        // Update indirect draw args with the correct values
        indirect_draw_args.index_count = 12u; // Hexagon has 12 indices
        indirect_draw_args.instance_count = atomicLoad(&culling_stats.visible_points);
        indirect_draw_args.first_index = 0u;
        indirect_draw_args.base_vertex = 0;
        indirect_draw_args.first_instance = 0u;
    }
}

