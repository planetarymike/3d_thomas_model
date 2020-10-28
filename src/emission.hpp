//emission.h -- structure defining an atmospheric emission

#ifndef __emission_h_
#define __emission_h_

#include "Real.hpp"
#include <string>
#include "atmo_vec.hpp"
#include "cuda_compatibility.hpp"
#include "voxel_vector.hpp"
#include <boost/type_traits/type_identity.hpp> //for type deduction in define

using std::string;
using std::isnan;

template <int N_VOXELS>
struct emission {
  static const unsigned int n_voxels = N_VOXELS;
  string name;
  bool init;
  bool solved;
  
  Real branching_ratio;
  Real species_T_ref;//reference temp, K (anything near mean works OK as long
                     //as variations are less than a factor of 5ish)
  Real species_sigma_T_ref;//species cross section at this temperature


  //these store physical atmospheric parameters on the grid (dimension n_voxels)
  //dynamic arrays are required for dim>~32 for Eigen
  //the vectors point to Eigen objects so that these can be used interchangably
  typedef voxel_vector<N_VOXELS> vv;
  typedef voxel_matrix<N_VOXELS> vm;
  vv species_density; //average and point densities of species on the grid
  vv species_density_pt;
  vv species_T; //average and point temperature of species on the grid
  vv species_T_pt;
  vv dtau_species; // species density * species_sigma_T_ref * sqrt(species_T_ref/species_T) 
  vv dtau_species_pt;
  //  vv log_dtau_species; //in case we want to do interpolation in log space
  //  vv log_dtau_species_pt; 

  vv absorber_density; //same as above for absorber
  vv absorber_density_pt; 
  vv absorber_sigma; //pure absorption does not need T info; can be implicitly T-dependent
  vv absorber_sigma_pt;
  vv dtau_absorber; // absorber_density * absorber_sigma for each voxel independently
  vv dtau_absorber_pt;
  //  vv log_dtau_absorber;
  //  vv log_dtau_absorber_pt; 

  vv abs; //ratio (dtau_absorber / dtau_species)
  vv abs_pt; 
  //  vv log_abs; 
  //  vv log_abs_pt; 

  //Radiative transfer parameters
  vm influence_matrix; //influence matrix has dimensions n_voxels, n_voxels)

  //vectors to compute the single scattering have dimensions (n_voxels)  
  vv tau_species_single_scattering;
  vv tau_absorber_single_scattering;
  vv singlescat; 

  vv sourcefn; //have dimensions (n_voxels)  
  //  vv log_sourcefn; 

  emission() { init=false; }


  // CUDA_CALLABLE_MEMBER
  // emission& operator=(const emission<N_VOXELS> &rhs) {
  //   if(this == &rhs) return *this;

  //   //    name = rhs.name;
  //   init = rhs.init;
  //   solved = rhs.solved;
  
  //   branching_ratio = rhs.branching_ratio;
  //   species_T_ref = rhs.species_T_ref;
  //   species_sigma_T_ref = rhs.species_sigma_T_ref;

  //   species_density = rhs.species_density;
  //   species_density_pt = rhs.species_density_pt;
  //   species_T = rhs.species_T;
  //   species_T_pt = rhs.species_T_pt;
  //   dtau_species = rhs.dtau_species;
  //   dtau_species_pt = rhs.dtau_species_pt;
  //   // log_dtau_species = rhs.log_dtau_species;
  //   // log_dtau_species_pt = rhs. log_dtau_species_pt; 

  //   absorber_density = rhs.absorber_density;
  //   absorber_density_pt = rhs. absorber_density_pt; 
  //   absorber_sigma = rhs.absorber_sigma;
  //   absorber_sigma_pt = rhs.absorber_sigma_pt;
  //   dtau_absorber = rhs.dtau_absorber;
  //   dtau_absorber_pt = rhs.dtau_absorber_pt;
  //   // log_dtau_absorber = rhs.log_dtau_absorber;
  //   // log_dtau_absorber_pt = rhs. log_dtau_absorber_pt; 

  //   abs = rhs.abs;
  //   abs_pt = rhs. abs_pt; 
  //   // log_abs = rhs. log_abs; 
  //   // log_abs_pt = rhs. log_abs_pt; 

  //   influence_matrix = rhs.influence_matrix;

  //   tau_species_single_scattering = rhs.tau_species_single_scattering;
  //   tau_absorber_single_scattering = rhs.tau_absorber_single_scattering;
  //   singlescat = rhs. singlescat; 

  //   sourcefn = rhs.sourcefn;
  //   //    log_sourcefn = rhs.log_sourcefn; 
  // return *this;
  // }


  template<typename C>
  void define(const Real &emission_branching_ratio,
	      const Real &species_T_reff, const Real &species_sigma_T_reff,
	      const C &atmosphere,
	      void (boost::type_identity<C>::type::*species_density_function)(const atmo_voxel &vox, Real &ret_avg, Real &ret_pt) const,
	      void (boost::type_identity<C>::type::*species_T_function)(const atmo_voxel &vox, Real &ret_avg, Real &ret_pt) const,
	      void (boost::type_identity<C>::type::*absorber_density_function)(const atmo_voxel &vox, Real &ret_avg, Real &ret_pt) const,
	      Real (boost::type_identity<C>::type::*absorber_sigma_function)(const Real &T) const,
	      const atmo_voxel *voxels) {
    
    branching_ratio     = emission_branching_ratio;
    species_T_ref       = species_T_reff;
    species_sigma_T_ref = species_sigma_T_reff;
    
    for (unsigned int i_voxel=0;i_voxel<n_voxels;i_voxel++) {
      (atmosphere.*species_density_function)(voxels[i_voxel],
					     species_density[i_voxel],
					     species_density_pt[i_voxel]);
      assert(!isnan(species_density[i_voxel])
	     && species_density[i_voxel] >= 0
	     && "densities must be real and positive");
      assert(!isnan(species_density_pt[i_voxel])
	     && species_density_pt[i_voxel] >= 0
	     && "densities must be real and positive");
      
      (atmosphere.*species_T_function)(voxels[i_voxel],
				       species_T[i_voxel],
				       species_T_pt[i_voxel]);
      assert(!isnan(species_T[i_voxel])
	     && species_T[i_voxel] >= 0
	     && "temperatures must be real and positive");
      assert(!isnan(species_T_pt[i_voxel])
	     && species_T_pt[i_voxel] >= 0
	     && "temperatures must be real and positive");
      
      (atmosphere.*absorber_density_function)(voxels[i_voxel],
					      absorber_density[i_voxel],
					      absorber_density_pt[i_voxel]);
      assert(!isnan(absorber_density[i_voxel])
	     && absorber_density[i_voxel] >= 0
	     && "densities must be real and positive");
      assert(!isnan(absorber_density_pt[i_voxel])
	     && absorber_density_pt[i_voxel] >= 0
	     && "densities must be real and positive");
      

      absorber_sigma[i_voxel] = (atmosphere.*absorber_sigma_function)(species_T[i_voxel]);
      absorber_sigma_pt[i_voxel] = (atmosphere.*absorber_sigma_function)(species_T_pt[i_voxel]);
      assert(!isnan(absorber_sigma[i_voxel])
	     && absorber_sigma[i_voxel] >= 0
	     && "cross sections must be real and positive");
      assert(!isnan(absorber_sigma_pt[i_voxel])
	     && absorber_sigma_pt[i_voxel] >= 0
	     && "cross sections must be real and positive");
    }
    
    //define differential optical depths by coefficientwise multiplication
    dtau_species = species_density.array() * species_sigma_T_ref * sqrt(species_T_ref)/species_T.array().sqrt();
    dtau_species_pt = species_density_pt.array() * species_sigma_T_ref * sqrt(species_T_ref)/species_T_pt.array().sqrt();;
    dtau_absorber = absorber_density.array() * absorber_sigma.array();
    dtau_absorber_pt = absorber_density_pt.array() * absorber_sigma_pt.array();
    abs = dtau_absorber.array() / dtau_species.array();
    abs_pt = dtau_absorber_pt.array() / dtau_species_pt.array();
    
    // for (unsigned int i=0;i<log_dtau_species.size();i++) {
    //   log_dtau_species(i) = dtau_species(i) == 0 ? -1e5 : log(dtau_species(i));
    //   log_dtau_species_pt(i) = dtau_species_pt(i) == 0 ? -1e5 : log(dtau_species_pt(i));
    //   log_dtau_absorber(i) = dtau_absorber(i) == 0 ? -1e5 : log(dtau_species(i));
    //   log_dtau_absorber_pt(i) = dtau_absorber_pt(i) == 0 ? -1e5 : log(dtau_species_pt(i));
    //   log_abs(i) = abs(i) == 0 ? -1e5 : log(abs(i));
    //   log_abs_pt(i) = abs_pt(i) == 0 ? -1e5 : log(abs_pt(i));
    // }
    
    init=true;
  }


  //methods to transfer objects to device
  void copy_to_device_influence(emission<N_VOXELS> *device_emission);
  void copy_to_device_brightness(emission<N_VOXELS> *device_emission);
#ifdef __CUDACC__
  __device__ void copy_to_shared_brightness();
  __device__ void from_shared_brightness();
#endif


  void vector_to_device(vv & device_vec, vv & host_vec, bool transfer = true);
  void matrix_to_device(vm & device_vec, vm & host_vec, bool transfer = true);

  void copy_influence_to_host();
  void copy_solved_to_host();
  void vector_to_host(vv & host_vec);
  void matrix_to_host(vm & host_vec);
};

#ifdef __CUDACC__
#include "emission_gpu.cu"
#endif

#endif
