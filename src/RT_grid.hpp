 //RT_grid.h -- basic RT grid definitions
//             specifics for different geometries are in other files

#ifndef __RT_grid_H
#define __RT_grid_H

#include "Real.hpp"
#include <iostream> // for file output and dialog
#include <cmath>    // for cos and sin
#include "constants.hpp" // basic parameter definitions
#include "my_clock.hpp"
#include "atmo_vec.hpp"
#include "boundaries.hpp"
#include "influence.hpp"
#include "emission.hpp"
#include "observation.hpp"
#include "los_tracker.hpp"
#include <Eigen/Dense> //matrix solvers
#include <Eigen/StdVector>
#include <string>
#include <cassert>

//structure to hold the atmosphere grid
template<int N_EMISS, typename grid_type, typename influence_type>
struct RT_grid {
  static const int n_emissions = N_EMISS;//number of emissions to evaluate at each point in the grid
  emission<grid_type::n_voxels> emissions[n_emissions];

  grid_type grid;//this stores all of the geometrical info

  //need to refactor so this lives inside each emission
  //the influence function
  influence_type transmission;

  //initialization parameters
  bool all_emissions_init;
  
  RT_grid(const string (&emission_names)[n_emissions])
  {
    all_emissions_init = false;
    
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {
      emissions[i_emission].name = emission_names[i_emission];
      emissions[i_emission].resize();//grid.n_voxels);
    }
  }
  
  template<typename C>
  void define_emission(string emission_name,
		       Real emission_branching_ratio,
		       C &atmosphere,
		       Real (C::*species_density_function)(const atmo_point) ,
		       Real (C::*species_sigma_function)(const atmo_point),
		       Real (C::*absorber_density_function)(const atmo_point),
		       Real (C::*absorber_sigma_function)(const atmo_point)) {

    //find emission name (dumb search but n_emissions is small and these are called infrequently)
    int n;
    for (n=0;n<n_emissions;n++) {
      if (emissions[n].name == emission_name)
	break;
    }
    if (n == n_emissions) {
      assert(false && "can't find emission name in define_emission");
    } else {
      assert((0<=n&&n<n_emissions) && "attempt to set invalid emission in define_emitter");
      
      emissions[n].define(emission_branching_ratio,
			  atmosphere,
			  species_density_function,
			  species_sigma_function,
			  absorber_density_function,
			  absorber_sigma_function,
			  grid.pts);
      
      all_emissions_init = true;
      for (int i_emission=0;i_emission<n_emissions;i_emission++)
	if (!emissions[i_emission].init)
	  all_emissions_init = false;
    }
  }

  
  template <typename R>
  void voxel_traverse(const atmo_vector &v,
		      void (RT_grid::*function)(boundary_intersection_stepper<grid_type::n_dimensions,
						                              grid_type::n_max_intersections>& ,R& ),
		      R &retval)
  {
    assert(all_emissions_init && "!nitialize grid and influence function before calling voxel_traverse.");
  
    //get boundary intersections
    boundary_intersection_stepper<grid_type::n_dimensions,
				  grid_type::n_max_intersections> stepper;
    grid.ray_voxel_intersections(v, stepper);

    if (stepper.boundaries.size() == 0)
      return;
    
    stepper.origin();

    while(stepper.inside) {
      //call the desired function in each voxel
      (this->*function)(stepper, retval);
    
      stepper.next();
    }
  }


  void influence_update(boundary_intersection_stepper<grid_type::n_dimensions,
			                              grid_type::n_max_intersections>& stepper,
			tau_tracker<n_emissions>& los) {
    //update the influence matrix for each emission

    los.update_start(stepper,emissions);
      
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {
      
      //see Bishop1999 for derivation of this formula
      Real coef = stepper.vec.ray.domega;

      //bishop formulation
      coef *= ((transmission.T_lerp(los.tau_species_initial[i_emission])
		- transmission.T_lerp(los.tau_species_final[i_emission]))
	       
      	       *exp(-0.5*(los.tau_absorber_initial[i_emission]
      			  +los.tau_absorber_final[i_emission])));
      
      emissions[i_emission].influence_matrix(stepper.start_voxel,
					     stepper.current_voxel) += coef;
    
    }

    los.update_end();

  }

  void get_single_scattering_optical_depths(boundary_intersection_stepper<grid_type::n_dimensions,
					                                  grid_type::n_max_intersections>& stepper,
					    Real& max_tau_species)
  {
    for (int i_emission=0;i_emission<n_emissions;i_emission++) {
      emissions[i_emission].tau_species_single_scattering(stepper.start_voxel) += (emissions[i_emission].dtau_species(stepper.current_voxel)
										   * stepper.pathlength);
      emissions[i_emission].tau_absorber_single_scattering(stepper.start_voxel) += (emissions[i_emission].dtau_absorber(stepper.current_voxel)
										    * stepper.pathlength);
      
      Real tscomp = emissions[i_emission].tau_species_single_scattering(stepper.start_voxel);
      max_tau_species = tscomp > max_tau_species ? tscomp : max_tau_species;    
    }
  }
  

  void get_single_scattering(const atmo_point &pt, Real &max_tau_species) {

    if (pt.z<0&&pt.x*pt.x+pt.y*pt.y<grid.rmin*grid.rmin) {
      //if the point is behind the planet, no single scattering
      for (int i_emission=0;i_emission<n_emissions;i_emission++)
	emissions[i_emission].singlescat(pt.i_voxel)=0.0;
    } else {
      atmo_vector vec = atmo_vector(pt, grid.sun_direction);
      voxel_traverse(vec,
		     &RT_grid::get_single_scattering_optical_depths,
		     max_tau_species);
      
      for (int i_emission=0;i_emission<n_emissions;i_emission++)
	emissions[i_emission].singlescat(pt.i_voxel) = ( (transmission.T_lerp(emissions[i_emission].tau_species_single_scattering(pt.i_voxel))
							  * exp(-emissions[i_emission].tau_absorber_single_scattering(pt.i_voxel)) )
							 / emissions[i_emission].species_sigma(pt.i_voxel));
    }
  }

  void solve() {
    for (int i_emission=0;i_emission<n_emissions;i_emission++) {
      emissions[i_emission].influence_matrix *= emissions[i_emission].branching_ratio;

      MatrixX kernel = MatrixX::Identity(grid.n_voxels,grid.n_voxels);
      kernel -= emissions[i_emission].influence_matrix;

      emissions[i_emission].sourcefn=kernel.partialPivLu().solve(emissions[i_emission].singlescat);//partialPivLu has multithreading support

      // // iterative solution.
      // Real err = 1;
      // int it = 0;
      // VectorX sourcefn_old(grid.n_voxels);
      // sourcefn_old = emissions[i_emission].singlescat;
      // while (err > ABS && it < 500) {
      // 	emissions[i_emission].sourcefn = emissions[i_emission].singlescat + emissions[i_emission].influence_matrix * sourcefn_old;

      // 	err=((emissions[i_emission].sourcefn-sourcefn_old).array().abs()/sourcefn_old.array()).maxCoeff();
      // 	sourcefn_old = emissions[i_emission].sourcefn;
      // 	it++;
      // }
      // std::cout << "For " << emissions[i_emission].name << std::endl;
      // std::cout << "  Scattering up to order: " << it << " included.\n";
      // std::cout << "  Error at final order is: " << err << " .\n";
      
      for (int i=0;i<grid.n_voxels;i++)
	emissions[i_emission].log_sourcefn(i) = emissions[i_emission].sourcefn(i) == 0 ? -1e5 : log(emissions[i_emission].sourcefn(i));
      emissions[i_emission].solved=true;

      emissions[i_emission].eigen_to_vec();
    }
  }

  //generate source functions on the grid
  void generate_S() {
  
    //start timing
    my_clock clk;
    clk.start();

    for (int i_emission=0;i_emission<n_emissions;i_emission++) {
      emissions[i_emission].influence_matrix.setZero();
      emissions[i_emission].tau_species_single_scattering.setZero();
      emissions[i_emission].tau_absorber_single_scattering.setZero();
    }
    
    atmo_vector vec;
    Real max_tau_species = 0;
  
#pragma omp parallel for firstprivate(vec) shared(max_tau_species,std::cout) default(none)
    for (int i_pt = 0; i_pt < grid.n_voxels; i_pt++) {
      Real omega = 0.0; // make sure sum(domega) = 4*pi
    
      //now integrate outward along the ray grid:
      for (int i_ray=0; i_ray < grid.n_rays; i_ray++) {
	vec = atmo_vector(grid.pts[i_pt], grid.rays[i_ray]);
	omega += vec.ray.domega;

	tau_tracker<n_emissions> los;
      
	voxel_traverse(vec, &RT_grid::influence_update, los);

	if (los.max_tau_species > max_tau_species)
	  max_tau_species = los.max_tau_species;
      }

      if ((omega - 1.0 > ABS) || (omega - 1.0 < -ABS)) {
	std::cout << "omega != 4*pi\n";
	throw(10);
      }
    
      //now compute the single scattering function:
      get_single_scattering(grid.pts[i_pt], max_tau_species);
    
    }
  
    //solve for the source function
    solve();

    //std::cout << "max_tau_species = " << max_tau_species << std::endl;
  
    // print time elapsed
    clk.stop();
    clk.print_elapsed("source function generation takes ");
    std::cout << std::endl;
    
    return;
  }

  void save_influence(const string fname = "test/influence_matrix.dat") {
    std::ofstream file(fname);
    if (file.is_open())
      for (int i_emission=0;i_emission<n_emissions;i_emission++)
	file << "Here is the influence matrix for " << emissions[i_emission].name <<":\n" 
	     << emissions[i_emission].influence_matrix << "\n\n";
  }

  void save_S(const string fname) {
    grid.save_S(fname,emissions,n_emissions);
  }



  CUDA_CALLABLE_MEMBER
  Real interp_array(const int *indices, const Real *weights, const Real *arr) const {
    Real retval=0;
    for (int i=0;i<grid.n_interp_points;i++)
      retval+=weights[i]*arr[indices[i]];
    return retval;
  }
  CUDA_CALLABLE_MEMBER
  void interp(const int &ivoxel, const atmo_point &pt, interpolated_values<n_emissions> &retval) const {

    int indices[grid_type::n_interp_points];
    Real weights[grid_type::n_interp_points];

    grid.interp_weights(ivoxel,pt,indices,weights);
    
    for (int i_emission=0;i_emission<n_emissions;i_emission++) {
      retval.dtau_species_interp[i_emission]  = exp(interp_array(indices, weights,
								 emissions[i_emission].log_dtau_species_vec));
      retval.dtau_absorber_interp[i_emission] = exp(interp_array(indices, weights,
								 emissions[i_emission].log_dtau_absorber_vec));
      retval.sourcefn_interp[i_emission]      = exp(interp_array(indices, weights,
								 emissions[i_emission].log_sourcefn_vec));
    }
  }
  
  //interpolated brightness routine
  CUDA_CALLABLE_MEMBER
  void brightness(const atmo_vector &vec, const Real *g,
		  brightness_tracker<n_emissions> &los,
		  const int n_subsamples=5) const {
    assert(n_subsamples!=1 && "choose either 0 or n>1 voxel subsamples.");
    
    boundary_intersection_stepper<grid_type::n_dimensions,
				  grid_type::n_max_intersections> stepper;
    grid.ray_voxel_intersections(vec, stepper);

    los.reset();

    atmo_point pt;
    interpolated_values<n_emissions> interp_vals;
    
    //brightness is zero if we do not intersect the grid
    if (stepper.boundaries.size() == 0)
      return;

    int n_subsamples_distance = n_subsamples;
    if (n_subsamples==0)
      n_subsamples_distance=2;
    
    for(unsigned int i_bound=1;i_bound<stepper.boundaries.size();i_bound++) {
      Real d_start=stepper.boundaries[i_bound-1].distance;
      Real d_step=(stepper.boundaries[i_bound].distance-d_start)/(n_subsamples_distance-1);

      //account for small rounding errors in boundary crossings
      const Real eps = ABS;
      d_start += eps/2.*d_step;
      d_step *= 1-eps;

      int current_voxel = stepper.boundaries[i_bound-1].entering;

      
      for (int i_step=1;i_step<n_subsamples_distance;i_step++) {

	pt = vec.extend(d_start+i_step*d_step);
	
	if (n_subsamples!=0) {
	  interp(current_voxel,pt,interp_vals);
	}
	for (int i_emission=0;i_emission<n_emissions;i_emission++) {

	  Real dtau_species_temp, dtau_absorber_temp, sourcefn_temp;
	  
	  if (n_subsamples == 0) {
	    dtau_species_temp  = emissions[i_emission].dtau_species_vec[current_voxel];
	    dtau_absorber_temp = emissions[i_emission].dtau_absorber_vec[current_voxel];
	    sourcefn_temp      = emissions[i_emission].sourcefn_vec[current_voxel];
	  } else {
	    dtau_species_temp  = interp_vals.dtau_species_interp[i_emission];
	    dtau_absorber_temp = interp_vals.dtau_absorber_interp[i_emission];
	    sourcefn_temp      = interp_vals.sourcefn_interp[i_emission];
	  } 


	  los.tau_species_final[i_emission] = ( los.tau_species_initial[i_emission]
						+ dtau_species_temp*d_step);
	  los.tau_absorber_final[i_emission] = ( los.tau_absorber_initial[i_emission]
						 + dtau_absorber_temp*d_step);


	  //bishop formulation
	  los.brightness[i_emission] += (sourcefn_temp
					 
	  				 *emissions[i_emission].branching_ratio
					 
	  				 *(transmission.Tint_lerp(los.tau_species_final[i_emission])
	  				   - transmission.Tint_lerp(los.tau_species_initial[i_emission]))
					 
	  				 *exp(-0.5*(los.tau_absorber_initial[i_emission]
	  					    +los.tau_absorber_final[i_emission])));

	  los.tau_species_initial[i_emission]=los.tau_species_final[i_emission];
	  los.tau_absorber_initial[i_emission]=los.tau_absorber_final[i_emission];
	}
      }
    }

    if (stepper.exits_bottom) {
      for (int i_emission=0;i_emission<n_emissions;i_emission++) {
	los.tau_absorber_final[i_emission] = -1;
      }
    }

    //convert to kR
    for (int i_emission=0;i_emission<n_emissions;i_emission++)
      los.brightness[i_emission] *= g[i_emission]/1e9; //megaphoton/cm2/s * 1e-3 = kR, see C&H pg 280-282

  }

  vector<brightness_tracker<n_emissions>> brightness(const vector<atmo_vector> &vecs,
						     const Real (&g)[n_emissions],
						     const int n_subsamples=5) const {
    vector<brightness_tracker<n_emissions>> retval;
    
    retval.resize(vecs.size(),brightness_tracker<n_emissions>());

#pragma omp parallel for shared(retval) firstprivate(vecs,g,n_subsamples) default(none)
    for(unsigned int i=0; i<vecs.size(); i++)
      brightness(vecs[i],g,
		 retval[i],
		 n_subsamples);
    
    return retval;
  }

  void brightness(observation<n_emissions> &obs, const int n_subsamples=5) const {
    assert(obs.size()>0 && "there must be at least one observation to simulate!");
    for (int i_emission=0;i_emission<n_emissions;i_emission++)
      assert(obs.emission_g_factors[i_emission] != 0. && "set emission g factors before simulating brightness");

    Real g[n_emissions];
    obs.get_emission_g_factors(g);

    my_clock clk;
    clk.start();
    vector<brightness_tracker<n_emissions>> los = brightness(obs.get_vecs(), g, n_subsamples);
    for (int i_obs=0;i_obs<obs.size();i_obs++) {
      for (int i_emission=0;i_emission<n_emissions;i_emission++) {
	obs.brightness[i_obs][i_emission]   = los[i_obs].brightness[i_emission];
	obs.tau_species[i_obs][i_emission]  = los[i_obs].tau_species_final[i_emission];
	obs.tau_absorber[i_obs][i_emission] = los[i_obs].tau_absorber_final[i_emission];
      }
    }
    clk.stop();
    clk.print_elapsed("brightness calculation takes ");
  }

  void brightness_nointerp(observation<n_emissions> &obs) const {
    brightness(obs,0);
  }

  //hooks for porting to gpu
  void brightness_gpu(observation<n_emissions> &obs, const int n_subsamples=5);
  
};

//import the CUDA code if NVCC is the compiler
#ifdef __CUDACC__
#include "RT_gpu.cu"
#endif

#endif