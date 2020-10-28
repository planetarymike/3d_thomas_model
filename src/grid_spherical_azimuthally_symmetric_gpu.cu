// grid_spherical_azimuthally_symmetric_gpu.cu --- device routines for spherical grid
#include "grid_spherical_azimuthally_symmetric.hpp"

// we have only one fixed-size grid of any given kind.

// the size is defined using static const file-scope variables in
// generate_source_function.cpp or elsewhere

//because each grid has a unique parent with different size arrays, we
//define these here also

// __device__ __constant__ atmo_voxel d_grid_sph_azi_sym_voxels[(n_radial_boundaries-1)*(n_sza_boundaries-1)];
// __device__ __constant__ atmo_ray   d_grid_sph_azi_sym_rays[n_rays_theta*n_rays_phi];

__device__ __constant__ Real   d_grid_sph_azi_sym_radial_boundaries[n_radial_boundaries];
__device__ __constant__ Real   d_grid_sph_azi_sym_pts_radii[n_radial_boundaries-1];
__device__ __constant__ Real   d_grid_sph_azi_sym_log_pts_radii[n_radial_boundaries-1];
__device__ __constant__ sphere d_grid_sph_azi_sym_radial_boundary_spheres[n_radial_boundaries];

__device__ __constant__ Real d_grid_sph_azi_sym_sza_boundaries[n_sza_boundaries];
__device__ __constant__ Real d_grid_sph_azi_sym_pts_sza[n_sza_boundaries-1];
__device__ __constant__ cone d_grid_sph_azi_sym_sza_boundary_cones[n_sza_boundaries];

__device__ __constant__ Real d_grid_sph_azi_sym_ray_theta[n_rays_theta];
__device__ __constant__ Real d_grid_sph_azi_sym_ray_phi[n_rays_phi];


template <int N_RADIAL_BOUNDARIES, int N_SZA_BOUNDARIES, int N_RAY_THETA, int N_RAY_PHI>
void spherical_azimuthally_symmetric_grid<N_RADIAL_BOUNDARIES,
					  N_SZA_BOUNDARIES,
					  N_RAY_THETA,
					  N_RAY_PHI>::to_device(spherical_azimuthally_symmetric_grid
								<N_RADIAL_BOUNDARIES,
								N_SZA_BOUNDARIES,
								N_RAY_THETA,
								N_RAY_PHI> *device_grid)
{
  //for each pointer, copy the data from this object into the constant
  //memory allocated above, then point the device pointer at the
  //location of the constant array

  // grid_member_to_device(&(device_grid->voxels), d_grid_sph_azi_sym_voxels, this->voxels);
  // grid_member_to_device(&(device_grid->rays), d_grid_sph_azi_sym_rays, this->rays);

  grid_member_to_device(&(device_grid->radial_boundaries), d_grid_sph_azi_sym_radial_boundaries, radial_boundaries);
  grid_member_to_device(&(device_grid->pts_radii), d_grid_sph_azi_sym_pts_radii, pts_radii);
  grid_member_to_device(&(device_grid->log_pts_radii), d_grid_sph_azi_sym_log_pts_radii, log_pts_radii);
  grid_member_to_device(&(device_grid->radial_boundary_spheres), d_grid_sph_azi_sym_radial_boundary_spheres, radial_boundary_spheres);

  grid_member_to_device(&(device_grid->sza_boundaries), d_grid_sph_azi_sym_sza_boundaries, sza_boundaries);
  grid_member_to_device(&(device_grid->pts_sza), d_grid_sph_azi_sym_pts_sza, pts_sza);
  grid_member_to_device(&(device_grid->sza_boundary_cones), d_grid_sph_azi_sym_sza_boundary_cones, sza_boundary_cones);

  grid_member_to_device(&(device_grid->ray_theta), d_grid_sph_azi_sym_ray_theta, ray_theta);
  grid_member_to_device(&(device_grid->ray_phi), d_grid_sph_azi_sym_ray_phi, ray_phi);
}
