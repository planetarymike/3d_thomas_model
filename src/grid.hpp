//grid.h -- base class for atmosphere grids

#ifndef __GRID_H
#define __GRID_H

#include "Real.hpp"
#include "cuda_compatibility.hpp"
#include "emission.hpp"
#include "boundaries.hpp"
#include "atm/atmosphere_base.hpp"

template <int NDIM, int NVOXELS, int NRAYS, int N_MAX_INTERSECTIONS>
struct grid {
  static const int n_dimensions = NDIM; //dimensionality of the grid
  static const int n_voxels = NVOXELS;//number of grid voxels


  // !!!!! Note: Even though the next three functions are commented
  // !!!!! out, they MUST be implemented in any child class of grid.
  // !!!!! (They are commented out because CUDA cannot deal with the
  // !!!!! vtable built on the host for these methods when the object
  // !!!!! is copied to the device. No idea why it works for the
  // !!!!! other virtual functions in this class.)
  //
  // !!!!! Each derived class should also define a to_device() function
  // !!!!! that stores the grid members in CUDA constant memory
  //helper functions to swap between voxel and coordinate indices
  //CUDA_CALLABLE_MEMBER virtual void indices_to_voxel(const int (&/*indices*/)[n_dimensions], int & ret) const { };
  //CUDA_CALLABLE_MEMBER virtual void voxel_to_indices(const int /*i_voxel*/, int (&/*indices*/)[n_dimensions]) const { };
  //CUDA_CALLABLE_MEMBER virtual void point_to_indices(const atmo_point &/*pt*/, int (&/*indices*/)[n_dimensions]) const = 0;
  // void to_device(grid_type *device_ptr);

  virtual void setup_voxels(const atmosphere& atm) = 0;
  Real rmax,rmin;//max and min altitudes in the atmosphere
  
  //points inside the voxels to shoot rays from
  atmo_voxel voxels[NVOXELS];
  //  atmo_voxel *voxels;
  
  //ray info
  static const int n_rays = NRAYS;
  atmo_ray rays[NRAYS];
  //  atmo_ray *rays;
  virtual void setup_rays() = 0; 
  
  //how to intersect rays with voxel boundaries
  static const int n_max_intersections = N_MAX_INTERSECTIONS;
  CUDA_CALLABLE_MEMBER
  virtual void ray_voxel_intersections(const atmo_vector &vec,
				       boundary_intersection_stepper<n_dimensions, n_max_intersections> &stepper) const = 0; 
  
  //where the sun is, for single scattering
  const Real sun_direction[3] = {0.,0.,1.};
  
  //function to get interpolation coefs
  static const int n_interp_points = 2*n_dimensions;
  CUDA_CALLABLE_MEMBER
  virtual void interp_weights(const int &ivoxel, const atmo_point &pt,
			      int (&/*indices*/)[n_interp_points], Real (&/*weights*/)[n_interp_points] ) const = 0;

  virtual void save_S(const string &fname, const emission<n_voxels> *emiss, const int n_emissions) const = 0;

  // grid() {
  //   voxels = new atmo_voxel[NVOXELS];
  //   rays = new atmo_ray[NRAYS];
  // }
  // ~grid() {
  //   delete [] voxels;
  //   delete [] rays;
  // }
};

#ifdef __CUDACC__
template<typename T, int N>
void grid_member_to_device(T** device_loc, T (&device_constant)[N], const T* host_loc) {
  checkCudaErrors(cudaMemcpyToSymbol(device_constant, host_loc, N * sizeof(T)));

  T* symbol_loc = NULL;
  checkCudaErrors(cudaGetSymbolAddress((void **)&symbol_loc, device_constant));

  checkCudaErrors(cudaMemcpy((void**) device_loc, &symbol_loc, sizeof(T*), cudaMemcpyHostToDevice));
}
#endif

#endif

