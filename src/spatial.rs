use glam::{
    Vec4Swizzles,
    f32::{Mat4, Vec3, Vec4},
};

/// Axis-aligned bounding box for efficient frustum culling
#[derive(Debug, Clone, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        if points.is_empty() {
            return Self::new(Vec3::ZERO, Vec3::ZERO);
        }

        let mut min = points[0];
        let mut max = points[0];

        for &point in &points[1..] {
            min = min.min(point);
            max = max.max(point);
        }

        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn intersects_aabb(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn expand_aabb(&mut self, other: &AABB) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

/// Frustum for camera view culling
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 6], // Left, Right, Bottom, Top, Near, Far
}

impl Frustum {
    pub fn from_view_projection(view_proj: Mat4) -> Self {
        let matrix = view_proj.to_cols_array_2d();

        // Extract frustum planes from view-projection matrix
        let mut planes = [Vec4::ZERO; 6];

        // Left plane
        planes[0] = Vec4::new(
            matrix[0][3] + matrix[0][0],
            matrix[1][3] + matrix[1][0],
            matrix[2][3] + matrix[2][0],
            matrix[3][3] + matrix[3][0],
        );

        // Right plane
        planes[1] = Vec4::new(
            matrix[0][3] - matrix[0][0],
            matrix[1][3] - matrix[1][0],
            matrix[2][3] - matrix[2][0],
            matrix[3][3] - matrix[3][0],
        );

        // Bottom plane
        planes[2] = Vec4::new(
            matrix[0][3] + matrix[0][1],
            matrix[1][3] + matrix[1][1],
            matrix[2][3] + matrix[2][1],
            matrix[3][3] + matrix[3][1],
        );

        // Top plane
        planes[3] = Vec4::new(
            matrix[0][3] - matrix[0][1],
            matrix[1][3] - matrix[1][1],
            matrix[2][3] - matrix[2][1],
            matrix[3][3] - matrix[3][1],
        );

        // Near plane
        planes[4] = Vec4::new(
            matrix[0][3] + matrix[0][2],
            matrix[1][3] + matrix[1][2],
            matrix[2][3] + matrix[2][2],
            matrix[3][3] + matrix[3][2],
        );

        // Far plane
        planes[5] = Vec4::new(
            matrix[0][3] - matrix[0][2],
            matrix[1][3] - matrix[1][2],
            matrix[2][3] - matrix[2][2],
            matrix[3][3] - matrix[3][2],
        );

        // Normalize all planes
        for plane in &mut planes {
            let length = plane.xyz().length();
            if length > 0.0 {
                *plane = *plane / length;
            }
        }

        Self { planes }
    }

    /// Check if a point is inside the frustum
    pub fn contains_point(&self, point: Vec3) -> bool {
        for plane in &self.planes {
            if plane.xyz().dot(point) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }

    /// Check if an AABB intersects with the frustum
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        for (i, plane) in self.planes.iter().enumerate() {
            let normal = plane.xyz();
            let d = plane.w;

            // Find the most positive and most negative vertices
            let mut min_dot = f32::INFINITY;
            let mut max_dot = f32::NEG_INFINITY;

            for x in [aabb.min.x, aabb.max.x] {
                for y in [aabb.min.y, aabb.max.y] {
                    for z in [aabb.min.z, aabb.max.z] {
                        let dot = normal.dot(Vec3::new(x, y, z));
                        min_dot = min_dot.min(dot);
                        max_dot = max_dot.max(dot);
                    }
                }
            }

            // If both extremes are on the negative side of the plane, AABB is outside
            // For a point to be visible: normal.dot(point) + d >= 0
            // For AABB to be visible: at least one vertex should be visible
            if min_dot + d < 0.0 && max_dot + d < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Octree node for spatial partitioning
#[derive(Debug)]
pub struct OctreeNode {
    pub bounds: AABB,
    pub children: Option<[Box<OctreeNode>; 8]>,
    pub point_indices: Vec<usize>,
    pub max_points_per_node: usize,
    pub max_depth: usize,
}

impl OctreeNode {
    pub fn new(bounds: AABB, max_points_per_node: usize, max_depth: usize) -> Self {
        Self {
            bounds,
            children: None,
            point_indices: Vec::new(),
            max_points_per_node,
            max_depth,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn subdivide(&mut self, points: &[Vec3]) {
        if self.point_indices.len() <= self.max_points_per_node || self.max_depth == 0 {
            return;
        }

        let center = self.bounds.center();
        let _half_size = self.bounds.size() * 0.5;

        // Create 8 child octants
        let mut children = [
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(self.bounds.min.x, self.bounds.min.y, self.bounds.min.z),
                    Vec3::new(center.x, center.y, center.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(center.x, self.bounds.min.y, self.bounds.min.z),
                    Vec3::new(self.bounds.max.x, center.y, center.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(self.bounds.min.x, center.y, self.bounds.min.z),
                    Vec3::new(center.x, self.bounds.max.y, center.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(center.x, center.y, self.bounds.min.z),
                    Vec3::new(self.bounds.max.x, self.bounds.max.y, center.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(self.bounds.min.x, self.bounds.min.y, center.z),
                    Vec3::new(center.x, center.y, self.bounds.max.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(center.x, self.bounds.min.y, center.z),
                    Vec3::new(self.bounds.max.x, center.y, self.bounds.max.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(self.bounds.min.x, center.y, center.z),
                    Vec3::new(center.x, self.bounds.max.y, self.bounds.max.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
            Box::new(OctreeNode::new(
                AABB::new(
                    Vec3::new(center.x, center.y, center.z),
                    Vec3::new(self.bounds.max.x, self.bounds.max.y, self.bounds.max.z),
                ),
                self.max_points_per_node,
                self.max_depth - 1,
            )),
        ];

        // Distribute points to children
        for &point_idx in &self.point_indices {
            let point = points[point_idx];
            for child in &mut children {
                if child.bounds.contains_point(point) {
                    child.point_indices.push(point_idx);
                    break;
                }
            }
        }

        // Recursively subdivide children
        for child in &mut children {
            if !child.point_indices.is_empty() {
                child.subdivide(points);
            }
        }

        self.children = Some(children);
    }

    /// Get visible point indices using frustum culling
    pub fn get_visible_points(
        &self,
        frustum: &Frustum,
        points: &[Vec3],
        visible_indices: &mut Vec<usize>,
    ) {
        // Check if this node's bounds intersect with the frustum
        if !frustum.intersects_aabb(&self.bounds) {
            return;
        }

        if self.is_leaf() {
            // Add all points in this leaf node
            visible_indices.extend_from_slice(&self.point_indices);
        } else if let Some(ref children) = self.children {
            // Recursively check children
            for child in children {
                child.get_visible_points(frustum, points, visible_indices);
            }
        }
    }
}

/// Octree for efficient spatial partitioning and frustum culling
#[derive(Debug)]
pub struct Octree {
    pub root: OctreeNode,
    pub points: Vec<Vec3>,
}

impl Octree {
    pub fn new(points: Vec<Vec3>, max_points_per_node: usize, max_depth: usize) -> Self {
        if points.is_empty() {
            let bounds = AABB::new(Vec3::ZERO, Vec3::ZERO);
            let root = OctreeNode::new(bounds, max_points_per_node, max_depth);
            return Self { root, points };
        }

        // Calculate bounds
        let bounds = AABB::from_points(&points);
        let mut root = OctreeNode::new(bounds, max_points_per_node, max_depth);

        // Add all point indices to root
        root.point_indices = (0..points.len()).collect();

        // Build the octree
        root.subdivide(&points);

        Self { root, points }
    }

    /// Get all points visible from the given camera view
    pub fn get_visible_points(&self, view_projection: Mat4) -> Vec<usize> {
        let frustum = Frustum::from_view_projection(view_projection);
        let mut visible_indices = Vec::new();

        self.root
            .get_visible_points(&frustum, &self.points, &mut visible_indices);

        visible_indices
    }

    /// Get points within a radius of a center point
    pub fn get_points_in_radius(&self, center: Vec3, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        let radius_sq = radius * radius;

        self.root
            .get_points_in_radius_recursive(center, radius_sq, &self.points, &mut result);

        result
    }
}

impl OctreeNode {
    fn get_points_in_radius_recursive(
        &self,
        center: Vec3,
        radius_sq: f32,
        points: &[Vec3],
        result: &mut Vec<usize>,
    ) {
        // Check if this node's bounds could contain points within radius
        let node_center = self.bounds.center();
        let node_size = self.bounds.size();
        let max_distance = (node_size * 0.5).length() + radius_sq.sqrt();

        if (node_center - center).length_squared() > max_distance * max_distance {
            return;
        }

        if self.is_leaf() {
            // Check individual points in this leaf
            for &point_idx in &self.point_indices {
                let point = points[point_idx];
                if (point - center).length_squared() <= radius_sq {
                    result.push(point_idx);
                }
            }
        } else if let Some(ref children) = self.children {
            // Recursively check children
            for child in children {
                child.get_points_in_radius_recursive(center, radius_sq, points, result);
            }
        }
    }
}

/// Spatial index for efficient point cloud management
pub struct SpatialIndex {
    octree: Octree,
    pub total_points: usize,
}

impl SpatialIndex {
    pub fn new(points: Vec<Vec3>) -> Self {
        let total_points = points.len();

        // Heuristic: max 64 points per node, max depth 8
        // These values can be tuned based on your specific use case
        let max_points_per_node = 64;
        let max_depth = 8;

        let octree = Octree::new(points, max_points_per_node, max_depth);

        Self {
            octree,
            total_points,
        }
    }

    /// Get visible points for rendering
    pub fn get_visible_points(&self, view_projection: Mat4) -> Vec<usize> {
        self.octree.get_visible_points(view_projection)
    }

    /// Get statistics about the spatial index
    pub fn get_stats(&self) -> SpatialIndexStats {
        let mut stats = SpatialIndexStats::default();
        self.octree.root.collect_stats(&mut stats);
        stats
    }
}

#[derive(Debug, Default)]
pub struct SpatialIndexStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub avg_points_per_leaf: f32,
}

impl OctreeNode {
    fn collect_stats(&self, stats: &mut SpatialIndexStats) {
        stats.total_nodes += 1;

        if self.is_leaf() {
            stats.leaf_nodes += 1;
            stats.avg_points_per_leaf += self.point_indices.len() as f32;
        } else if let Some(ref children) = self.children {
            for child in children {
                child.collect_stats(stats);
            }
        }
    }
}

impl SpatialIndexStats {
    pub fn finalize(&mut self) {
        if self.leaf_nodes > 0 {
            self.avg_points_per_leaf /= self.leaf_nodes as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustum_culling() {
        // Create a simple point cloud
        let mut points = Vec::new();
        for x in -10..10 {
            for y in -10..10 {
                for z in -10..10 {
                    points.push(Vec3::new(x as f32, y as f32, z as f32));
                }
            }
        }

        let total_points = points.len();

        // Create spatial index
        let spatial_index = SpatialIndex::new(points);

        // Create a simple view-projection matrix (looking down Z axis)
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 20.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4, // 45 degrees
            1.0,                         // square aspect ratio
            0.1,
            100.0,
        );
        let view_projection = projection * view;

        // Get visible points
        let visible_indices = spatial_index.get_visible_points(view_projection);

        // Should cull some points (not all 8000 points should be visible)
        assert!(visible_indices.len() < total_points);
        assert!(visible_indices.len() > 0);

        println!(
            "Total points: {}, Visible points: {}, Culled: {}%",
            total_points,
            visible_indices.len(),
            (1.0 - visible_indices.len() as f32 / total_points as f32) * 100.0
        );
    }

    #[test]
    fn test_octree_construction() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(10.0, 10.0, 10.0),
        ];

        let spatial_index = SpatialIndex::new(points);
        let stats = spatial_index.get_stats();

        // Should have at least the root node
        assert!(stats.total_nodes >= 1);
        println!("Octree stats: {:?}", stats);
    }
}
