// grid_plane_parallel.h -- plane parallel RT grid

#ifndef __grid_plane_parallel
#define __grid_plane_parallel

#include "Real.hpp"
#include "grid_dims.hpp"
#include "grid.hpp"
#include "coordinate_generation.hpp"
#include "boundaries.hpp"
#include "atm/atmosphere_base.hpp"
#include <fstream>
#include <cmath>
#include "intersections.hpp"

template <int N_RADIAL_BOUNDARIES, int N_RAYS_THETA>
struct plane_parallel_grid : grid<1,//this is a 1d grid
				  N_RADIAL_BOUNDARIES-1,
				  N_RAYS_THETA,
				  N_RADIAL_BOUNDARIES> 
{

  using parent_grid = grid<1,//N_DIMENSIONS
			   N_RADIAL_BOUNDARIES-1,//N_VOXELS
			   N_RAYS_THETA,//N_RAYS
			   N_RADIAL_BOUNDARIES>;//N_MAX_INTERSECTIONS

  int rmethod;
  static const int rmethod_altitude = 0;
  static const int rmethod_log_n_species= 1;

  static const int n_radial_boundaries = N_RADIAL_BOUNDARIES;
  Real *radial_boundaries;
  Real *pts_radii;
  plane *radial_boundary_planes;
  
  plane_parallel_grid() {
    rmethod = rmethod_log_n_species;

    radial_boundaries = new Real[n_radial_boundaries];
    pts_radii = new Real[n_radial_boundaries-1];
    radial_boundary_planes = new plane[n_radial_boundaries];
  }
  ~plane_parallel_grid() {
    delete [] radial_boundaries;
    delete [] pts_radii;
    delete [] radial_boundary_planes;
  }
  void to_device(plane_parallel_grid
		 <N_RADIAL_BOUNDARIES,
		 N_RAYS_THETA> *device_grid);
    
  void setup_voxels(const atmosphere &atm)
  {
    this->rmin = atm.rmin;
    this->rmax = atm.rmax;

    assert((rmethod == rmethod_altitude || rmethod == rmethod_log_n_species)
	   && "rmethod must match a defined radial points method");
    // don't define a tau radial points method; tau < 0.1 is
    // important and max(tau) > 10; this leads to many required
    // gridpoints
    if (rmethod == rmethod_altitude) {
      vector<Real> radial_boundaries_vector;
      get_radial_log_linear_points(radial_boundaries_vector, n_radial_boundaries,
				   atm.rmin, atm.rexo, atm.rmax);
      for (int i=0;i<n_radial_boundaries;i++)
	radial_boundaries[i] = radial_boundaries_vector[i];
    }
    if (rmethod == rmethod_log_n_species) {
      Real log_n_species_max = log(atm.n_species(atm.rmin));
      Real log_n_species_min = log(atm.n_species(atm.rmax));
      Real log_n_species_step = (log_n_species_max-log_n_species_min)/(n_radial_boundaries-1.);
      
      for(int i=0;i<n_radial_boundaries;i++) {
	Real n_species_target=exp(log_n_species_max-i*log_n_species_step);
	radial_boundaries[i]=atm.r_from_n_species(n_species_target);
      }
    }

    for (int i=0; i<n_radial_boundaries-1; i++) {
      this->voxels[i].rbounds[0] = radial_boundaries[i];
      this->voxels[i].rbounds[1] = radial_boundaries[i+1];
      this->voxels[i].tbounds[0] = 0;
      this->voxels[i].tbounds[1] = pi;
      this->voxels[i].pbounds[0] = 0;
      this->voxels[i].pbounds[1] = 2*pi;
      this->voxels[i].i_voxel = i;
      // this->voxels[i].init = true;

      pts_radii[i]=sqrt(radial_boundaries[i]*radial_boundaries[i+1]);
      // Real frac = 0.5;
      // pts_radii[i]=(frac*radial_boundaries[i]+(1-frac)*radial_boundaries[i+1]);
      this->voxels[i].pt.xyz(0.,0.,pts_radii[i]);
      this->voxels[i].pt.set_voxel_index(i);
    }

    for (int i=0; i<n_radial_boundaries; i++) 
      radial_boundary_planes[i].z = radial_boundaries[i];
  }
  
  void setup_rays() {
    //in a plane-parallel grid only the angle with the vertical matters
    vector<Real> ray_theta;
    vector<Real> ray_weights;

    gauss_quadrature_points(ray_theta,ray_weights,0,pi,parent_grid::n_rays);
    for (int i=0;i<parent_grid::n_rays;i++)
      ray_weights[i]*=std::sin(ray_theta[i]);
    
    for (int i=0;i<parent_grid::n_rays;i++) {
      this->rays[i].tp(ray_theta[i],0.0);
      this->rays[i].set_ray_index(i,ray_weights[i],2*pi);
    }
  }

  CUDA_CALLABLE_MEMBER 
  void indices_to_voxel(const int &comp_idx, int & ret) const {
    if ((comp_idx < 0) || (comp_idx > (int) n_radial_boundaries-2))
      ret = -1;
    else
      ret = comp_idx;
  }
  CUDA_CALLABLE_MEMBER 
  void indices_to_voxel(const int (&indices)[parent_grid::n_dimensions], int & ret) const {
    indices_to_voxel(indices[0], ret);
  }
  CUDA_CALLABLE_MEMBER 
  void voxel_to_indices(const int i_voxel, int (&indices)[parent_grid::n_dimensions]) const {
    if ((i_voxel < 0) || (i_voxel > parent_grid::n_voxels-1))
      indices[0]=-1;
    else
      indices[0]=i_voxel;
  }

  template <class V>
  CUDA_CALLABLE_MEMBER 
  int find_coordinate_index(const Real &pt_coord, const V &boundaries, int n_boundaries) const {
    int i;
    
    for (i=0;i<n_boundaries;i++)
      if (pt_coord < boundaries[i])
	break;

    i--;

    assert((boundaries[n_boundaries-1]<=pt_coord ||
	    pt_coord < boundaries[0] ||
	    (boundaries[i]<=pt_coord &&pt_coord<boundaries[i+1]))
	   && "we have found the appropriate point index");
    
    return i;
  }

  CUDA_CALLABLE_MEMBER 
  void point_to_indices(const atmo_point pt, int (&indices)[parent_grid::n_dimensions]) const {
    indices[0] = find_coordinate_index(pt.r,radial_boundaries,n_radial_boundaries);
  }
  
  CUDA_CALLABLE_MEMBER 
  void ray_voxel_intersections(const atmo_vector &vec,
			       boundary_intersection_stepper<parent_grid::n_dimensions,
			                                     parent_grid::n_max_intersections> &stepper) const {
    stepper.vec = vec;
    stepper.boundaries.reset();

    //define the origin
    boundary<parent_grid::n_dimensions> origin;
    origin.entering = vec.pt.i_voxel;
    if (vec.pt.i_voxel == -1)
      point_to_indices(vec.pt, origin.entering_indices);
    else
      voxel_to_indices(origin.entering, origin.entering_indices);
    origin.distance = 0.0;
    stepper.boundaries.append(origin);
    
    //do the intersections for each coordinate
    int n_hits = 0;
    Real temp_distances[2] = {-1,-1};
    for (unsigned int ir=0;ir<n_radial_boundaries;ir++) {
      radial_boundary_planes[ir].intersections(vec, temp_distances, n_hits);
      stepper.boundaries.add_intersections(vec.pt.r, 0,
					   ir, radial_boundaries[ir],
					   temp_distances, n_hits);
    }

    //sort the list of intersections by distance & trim
    stepper.boundaries.sort();
    stepper.boundaries.propagate_indices();
    stepper.boundaries.assign_voxel_indices(this);
    stepper.boundaries.trim();

    stepper.init_stepper();
  }

  CUDA_CALLABLE_MEMBER 
  void interp_weights(__attribute__((unused)) const int &ivoxel, __attribute__((unused)) const atmo_point &ptt,
		      __attribute__((unused)) int (&indices)[parent_grid::n_interp_points],
		      __attribute__((unused)) Real (&weights)[parent_grid::n_interp_points]) const {
    assert(false && "interp_weights not implemented in grid_plane_parallel");
  }

  
  void save_S(const string &fname, const emission<parent_grid::n_voxels> *emissions, const int n_emissions) const {
    std::ofstream file(fname.c_str());
    if (file.is_open())
      {

	VectorX r_boundaries_write_out = Eigen::Map<const VectorX>(radial_boundaries,
								     n_radial_boundaries);

	file << "radial boundaries [cm]: " << r_boundaries_write_out.transpose() << "\n\n";

	VectorX r_pts_write_out = Eigen::Map<const VectorX>(pts_radii,
							      n_radial_boundaries-1);

	file << "pts radii [cm]: " << r_pts_write_out.transpose() << "\n\n";
       	
	for (int i_emission=0;i_emission<n_emissions;i_emission++)
	  file << "For " << emissions[i_emission].name << ",\n"
	       << "Species density [cm-3]: "
	       <<	emissions[i_emission].species_density.transpose() << "\n"

	       << "Species single scattering tau: "
	       <<	emissions[i_emission].tau_species_single_scattering.transpose() << "\n"

	       << "Species cross section [cm2]: "
	       << (emissions[i_emission].species_sigma_T_ref
		   *std::sqrt(emissions[i_emission].species_T_ref)
		   /emissions[i_emission].species_T.array().sqrt()).transpose() << "\n"

	       << "Absorber density [cm-3]: "
	       << emissions[i_emission].absorber_density.transpose() << "\n"

	       << "Absorber single scattering tau: "
	       <<	emissions[i_emission].tau_absorber_single_scattering.transpose() << "\n"

	       << "Species single scattering source function S0: "
	       <<	emissions[i_emission].singlescat.transpose() << "\n"

	       << "Source function: "
	       << emissions[i_emission].sourcefn.transpose() << "\n\n";
      }
  }

};

#ifdef __CUDACC__
#include "grid_plane_parallel_gpu.cu"
#endif

#endif
