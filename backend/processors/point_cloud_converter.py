"""
Point Cloud to Mesh Converter for MeshBuilder
Converts 3D Gaussian Splatting point clouds to 3D meshes

This module handles:
- Point cloud loading from 3DGS training output
- Mesh reconstruction using various algorithms
- Mesh optimization and cleanup
- Export to multiple formats (OBJ, PLY, etc.)

Author: MeshBuilder Team
Version: 2.0.0
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MeshBuilder.PointCloudConverter")

# Import dependencies with fallbacks
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("Trimesh not available. Install with: pip install trimesh")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available. Install with: pip install open3d")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy is required but not available")


class PointCloudToMeshConverter:
    """
    Converts 3D Gaussian Splatting point clouds to meshes
    
    Supports multiple reconstruction algorithms and export formats
    """
    
    def __init__(self, base_output_dir: str = "./output/"):
        """
        Initialize the point cloud to mesh converter
        
        Args:
            base_output_dir: Base output directory for projects
        """
        self.base_output_dir = Path(base_output_dir)
        
        # Check dependencies
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for point cloud operations")
        
        self.mesh_methods = {
            'poisson': self._poisson_reconstruction,
            'ball_pivoting': self._ball_pivoting_reconstruction,
            'marching_cubes': self._marching_cubes_reconstruction,
            'alpha_shape': self._alpha_shape_reconstruction
        }
        
        logger.info("PointCloudToMeshConverter initialized")
        logger.info(f"Available methods: {list(self.mesh_methods.keys())}")
        logger.info(f"Trimesh available: {TRIMESH_AVAILABLE}")
        logger.info(f"Open3D available: {OPEN3D_AVAILABLE}")
    
    def convert_project(self, 
                       project_name: str,
                       method: str = 'poisson',
                       target_triangles: int = 100000,
                       callback: Optional[Callable[[str, int], None]] = None) -> bool:
        """
        Convert a project's point cloud to mesh
        
        Args:
            project_name: Name of the project
            method: Reconstruction method ('poisson', 'ball_pivoting', etc.)
            target_triangles: Target number of triangles
            callback: Progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting mesh conversion for project: {project_name}")
            logger.info(f"Method: {method}, Target triangles: {target_triangles}")
            
            if callback:
                callback("Loading point cloud data", 10)
            
            # Load point cloud from 3DGS output
            point_cloud_path = self._find_point_cloud(project_name)
            if not point_cloud_path:
                logger.error("Point cloud file not found")
                return False
            
            point_cloud = self._load_point_cloud(point_cloud_path)
            if point_cloud is None:
                logger.error("Failed to load point cloud")
                return False
            
            if callback:
                callback("Reconstructing mesh", 40)
            
            # Convert to mesh using specified method
            mesh = self._convert_to_mesh(point_cloud, method)
            if mesh is None:
                logger.error("Mesh reconstruction failed")
                return False
            
            if callback:
                callback("Optimizing mesh", 70)
            
            # Optimize mesh
            mesh = self._optimize_mesh(mesh, target_triangles)
            
            if callback:
                callback("Saving mesh", 90)
            
            # Save mesh
            output_path = self._save_mesh(project_name, mesh)
            if not output_path:
                logger.error("Failed to save mesh")
                return False
            
            if callback:
                callback("Mesh conversion completed", 100)
            
            logger.info(f"Mesh conversion completed successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Mesh conversion failed: {e}")
            if callback:
                callback(f"Conversion failed: {e}", -1)
            return False
    
    def _find_point_cloud(self, project_name: str) -> Optional[Path]:
        """Find point cloud file in project output"""
        project_dir = self.base_output_dir / project_name
        
        # Look for common point cloud file locations
        search_paths = [
            project_dir / "Output_main" / "point_cloud.ply",
            project_dir / "training_output" / "point_cloud.ply",
            project_dir / "Output_main" / "output.ply",
            project_dir / "training_output" / "output.ply"
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found point cloud: {path}")
                return path
        
        # Search for any .ply files
        for path in project_dir.rglob("*.ply"):
            if path.is_file():
                logger.info(f"Found point cloud file: {path}")
                return path
        
        logger.warning(f"No point cloud files found in project: {project_name}")
        return None
    
    def _load_point_cloud(self, file_path: Path) -> Optional[Any]:
        """Load point cloud from file"""
        try:
            if OPEN3D_AVAILABLE:
                # Use Open3D for loading
                pcd = o3d.io.read_point_cloud(str(file_path))
                if len(pcd.points) == 0:
                    raise ValueError("Empty point cloud")
                logger.info(f"Loaded point cloud with {len(pcd.points)} points using Open3D")
                return pcd
            
            elif TRIMESH_AVAILABLE:
                # Use Trimesh for loading
                mesh = trimesh.load(str(file_path))
                if hasattr(mesh, 'vertices'):
                    logger.info(f"Loaded point cloud with {len(mesh.vertices)} points using Trimesh")
                    return mesh
                else:
                    raise ValueError("Invalid point cloud format")
            
            else:
                # Fallback: basic PLY loading with NumPy
                return self._load_ply_numpy(file_path)
                
        except Exception as e:
            logger.error(f"Failed to load point cloud {file_path}: {e}")
            return None
    
    def _load_ply_numpy(self, file_path: Path) -> Optional[np.ndarray]:
        """Basic PLY loading using NumPy (fallback)"""
        try:
            # This is a very basic PLY parser - for production use Open3D or Trimesh
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find vertex count
            vertex_count = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    header_end = i + 1
                    break
            
            if vertex_count == 0:
                raise ValueError("No vertices found in PLY file")
            
            # Read vertex data
            vertices = []
            for i in range(header_end, min(header_end + vertex_count, len(lines))):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            points = np.array(vertices)
            logger.info(f"Loaded {len(points)} points using NumPy fallback")
            return points
            
        except Exception as e:
            logger.error(f"NumPy PLY loading failed: {e}")
            return None
    
    def _convert_to_mesh(self, point_cloud: Any, method: str) -> Optional[Any]:
        """Convert point cloud to mesh using specified method"""
        if method not in self.mesh_methods:
            logger.error(f"Unknown mesh method: {method}")
            return None
        
        try:
            return self.mesh_methods[method](point_cloud)
        except Exception as e:
            logger.error(f"Mesh conversion failed with method {method}: {e}")
            return None
    
    def _poisson_reconstruction(self, point_cloud: Any) -> Optional[Any]:
        """Poisson surface reconstruction"""
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D required for Poisson reconstruction")
            return None
        
        try:
            if isinstance(point_cloud, np.ndarray):
                # Convert numpy array to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                point_cloud = pcd
            
            # Estimate normals
            point_cloud.estimate_normals()
            
            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=9
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            logger.info(f"Poisson reconstruction: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Poisson reconstruction failed: {e}")
            return None
    
    def _ball_pivoting_reconstruction(self, point_cloud: Any) -> Optional[Any]:
        """Ball pivoting algorithm reconstruction"""
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D required for ball pivoting reconstruction")
            return None
        
        try:
            if isinstance(point_cloud, np.ndarray):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                point_cloud = pcd
            
            # Estimate normals
            point_cloud.estimate_normals()
            
            # Calculate radii for ball pivoting
            distances = point_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2]
            
            # Ball pivoting reconstruction
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, o3d.utility.DoubleVector(radii)
            )
            
            logger.info(f"Ball pivoting: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Ball pivoting reconstruction failed: {e}")
            return None
    
    def _marching_cubes_reconstruction(self, point_cloud: Any) -> Optional[Any]:
        """Marching cubes reconstruction (placeholder)"""
        logger.warning("Marching cubes not yet implemented, using Poisson instead")
        return self._poisson_reconstruction(point_cloud)
    
    def _alpha_shape_reconstruction(self, point_cloud: Any) -> Optional[Any]:
        """Alpha shape reconstruction (placeholder)"""
        logger.warning("Alpha shape not yet implemented, using Poisson instead")
        return self._poisson_reconstruction(point_cloud)
    
    def _optimize_mesh(self, mesh: Any, target_triangles: int) -> Any:
        """Optimize mesh (simplification, smoothing, etc.)"""
        try:
            if OPEN3D_AVAILABLE and hasattr(mesh, 'triangles'):
                # Simplify mesh if it has too many triangles
                current_triangles = len(mesh.triangles)
                if current_triangles > target_triangles:
                    reduction_factor = target_triangles / current_triangles
                    mesh = mesh.simplify_quadric_decimation(target_triangles)
                    logger.info(f"Simplified mesh: {current_triangles} -> {len(mesh.triangles)} triangles")
                
                # Remove degenerate triangles
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                # Smooth mesh slightly
                mesh = mesh.filter_smooth_simple(number_of_iterations=1)
                
                logger.info(f"Optimized mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Mesh optimization failed: {e}, returning original mesh")
            return mesh
    
    def _save_mesh(self, project_name: str, mesh: Any) -> Optional[Path]:
        """Save mesh to file"""
        try:
            # Create output directory
            output_dir = self.base_output_dir / project_name / "mesh_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as OBJ (primary format)
            obj_path = output_dir / "model.obj"
            
            if OPEN3D_AVAILABLE and hasattr(mesh, 'triangles'):
                success = o3d.io.write_triangle_mesh(str(obj_path), mesh)
                if success:
                    logger.info(f"Saved mesh as OBJ: {obj_path}")
                    
                    # Also save as PLY
                    ply_path = output_dir / "model.ply"
                    o3d.io.write_triangle_mesh(str(ply_path), mesh)
                    logger.info(f"Saved mesh as PLY: {ply_path}")
                    
                    return obj_path
            
            elif TRIMESH_AVAILABLE:
                mesh.export(str(obj_path))
                logger.info(f"Saved mesh using Trimesh: {obj_path}")
                return obj_path
            
            else:
                logger.error("No suitable library available for saving mesh")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save mesh: {e}")
            return None
    
    def get_conversion_info(self, project_name: str) -> Dict[str, Any]:
        """Get information about mesh conversion for a project"""
        info = {
            "project_name": project_name,
            "point_cloud_found": False,
            "point_cloud_path": None,
            "mesh_output_exists": False,
            "mesh_output_path": None,
            "available_methods": list(self.mesh_methods.keys()),
            "recommended_method": "poisson"
        }
        
        try:
            # Check for point cloud
            point_cloud_path = self._find_point_cloud(project_name)
            if point_cloud_path:
                info["point_cloud_found"] = True
                info["point_cloud_path"] = str(point_cloud_path)
            
            # Check for existing mesh output
            mesh_path = self.base_output_dir / project_name / "mesh_output" / "model.obj"
            if mesh_path.exists():
                info["mesh_output_exists"] = True
                info["mesh_output_path"] = str(mesh_path)
            
            # Recommend method based on available libraries
            if OPEN3D_AVAILABLE:
                info["recommended_method"] = "poisson"
            elif TRIMESH_AVAILABLE:
                info["recommended_method"] = "poisson"
            else:
                info["recommended_method"] = "basic"
        
        except Exception as e:
            logger.debug(f"Error getting conversion info: {e}")
        
        return info


def test_point_cloud_converter():
    """Test the point cloud converter"""
    try:
        print("Testing PointCloudToMeshConverter...")
        
        # Create converter
        converter = PointCloudToMeshConverter()
        
        print(f"Available methods: {list(converter.mesh_methods.keys())}")
        print(f"Trimesh available: {TRIMESH_AVAILABLE}")
        print(f"Open3D available: {OPEN3D_AVAILABLE}")
        
        # Test getting conversion info
        info = converter.get_conversion_info("test_project")
        print(f"Conversion info structure: {len(info)} fields")
        
        print("PointCloudToMeshConverter test completed successfully!")
        return True
        
    except Exception as e:
        print(f"PointCloudToMeshConverter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_point_cloud_converter()