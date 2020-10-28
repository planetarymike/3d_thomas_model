// grid_plane_parallel_gpu.cu --- device routines for spherical grid
#include "grid_plane_parallel.hpp"

// we have only one fixed-size grid of any given kind.

// the size is defined using static const file-scope variables in
// generate_source_function.cpp or elsewhere

//because each grid has a unique parent with different size arrays, we
//define these here also

// __constant__ atmo_voxel d_grid_pp_voxels[(n_radial_boundaries-1)*(n_sza_boundaries-1)];
// __constant__ atmo_ray   d_grid_pp_rays[n_rays_theta*n_rays_phi];

__constant__ Real  d_grid_pp_radial_boundaries[n_radial_boundaries];
__constant__ Real  d_grid_pp_pts_radii[n_radial_boundaries-1];
__constant__ plane d_grid_pp_radial_boundary_planes[n_radial_boundaries];

template <int N_RADIAL_BOUNDARIES, int N_RAYS_THETA>
void plane_parallel_grid<N_RADIAL_BOUNDARIES,
			 N_RAYS_THETA>::to_device(plane_parallel_grid
						  <N_RADIAL_BOUNDARIES,
						  N_RAYS_THETA> *device_grid)
{
  //for each pointer, copy the data from this object into the constant
  //memory allocated above, then point the device pointer at the
  //location of the constant array

  // grid_member_to_device(&(device_grid->voxels), d_grid_pp_voxels, this->voxels);
  // grid_member_to_device(&(device_grid->rays), d_grid_pp_rays, this->rays);

  grid_member_to_device(&(device_grid->radial_boundaries), d_grid_pp_radial_boundaries, radial_boundaries);
  grid_member_to_device(&(device_grid->pts_radii), d_grid_pp_pts_radii, pts_radii);
  grid_member_to_device(&(device_grid->radial_boundary_planes), d_grid_pp_radial_boundary_planes, radial_boundary_planes);
}
