//emission.h -- structure defining an atmospheric emission

#ifndef __emission_h_
#define __emission_h_

#include <string>
#include <Eigen/Dense>
#include "atmo_vec.h"

using std::string;
using Eigen::VectorXd;
using Eigen::MatrixXd;

template <int N_VOXELS>
struct emission {
  static const unsigned int n_voxels = N_VOXELS;
  string name;
  bool init;
  bool solved;
  
  double branching_ratio;
  
  //these store physical atmospheric parameters on the grid (dimension n_voxels)
  //even though n_voxels is known at compile time, Eigen docs
  //recommend using dynamic matrices for arrays larger than 16x16
  //don't make these fixed-size
  VectorXd species_density; //densities of scatterers and absorbers on the tabulated grid
  VectorXd absorber_density; 
  VectorXd species_sigma;//scatterer and absorber cross section on the tabulated grid
  VectorXd absorber_sigma;

  VectorXd dtau_species;
  VectorXd log_dtau_species; //quantities that need to be interpolated are also stored as log
  VectorXd dtau_absorber;
  VectorXd log_dtau_absorber;
  VectorXd abs; //ratio of dtau_abs to dtau_species
  VectorXd log_abs; 

  //Radiative transfer parameters
  MatrixXd influence_matrix; //influence matrix has dimensions n_voxels, n_voxels)

  VectorXd singlescat; //have dimensions (n_voxels)  
  VectorXd sourcefn; 
  VectorXd log_sourcefn; 

  //vectors to compute the single scattering have dimensions (n_voxels)  
  VectorXd tau_species_single_scattering;
  VectorXd tau_absorber_single_scattering;

  //cuda vectors
  double dtau_species_vec[n_voxels];
  double log_dtau_species_vec[n_voxels];
  double dtau_absorber_vec[n_voxels];
  double log_dtau_absorber_vec[n_voxels];
  double sourcefn_vec[n_voxels];
  double log_sourcefn_vec[n_voxels];
  
  
  emission() {
    init=false;
  }
  
  void resize() {//int n_voxelss) {
    //n_voxels = n_voxelss;
    
    species_density.resize(n_voxels);
    absorber_density.resize(n_voxels);
    species_sigma.resize(n_voxels);
    absorber_sigma.resize(n_voxels);
    dtau_species.resize(n_voxels);
    log_dtau_species.resize(n_voxels);
    dtau_absorber.resize(n_voxels);
    log_dtau_absorber.resize(n_voxels);
    abs.resize(n_voxels);
    log_abs.resize(n_voxels);
    influence_matrix.resize(n_voxels,n_voxels);
    singlescat.resize(n_voxels);
    sourcefn.resize(n_voxels);
    log_sourcefn.resize(n_voxels);
    tau_species_single_scattering.resize(n_voxels);
    tau_absorber_single_scattering.resize(n_voxels);
  }
  
  template<typename C, typename V>
  void define(double emission_branching_ratio,
	      C &atmosphere,
	      double (C::*species_density_function)(const atmo_point),
	      double (C::*species_sigma_function)(const atmo_point),
	      double (C::*absorber_density_function)(const atmo_point),
	      double (C::*absorber_sigma_function)(const atmo_point),
	      const V &pts) {
    
    branching_ratio = emission_branching_ratio;
    
    for (unsigned int i_voxel=0;i_voxel<n_voxels;i_voxel++) {
      species_density[i_voxel]=(atmosphere.*species_density_function)(pts[i_voxel]);
      species_sigma[i_voxel]=(atmosphere.*species_sigma_function)(pts[i_voxel]);
      absorber_density[i_voxel]=(atmosphere.*absorber_density_function)(pts[i_voxel]);
      absorber_sigma[i_voxel]=(atmosphere.*absorber_sigma_function)(pts[i_voxel]);
    }
    
    //define differential optical depths by coefficientwise multiplication
    dtau_species = species_density.array() * species_sigma.array();
    dtau_absorber = absorber_density.array() * absorber_sigma.array();
    abs = dtau_absorber.array() / dtau_species.array();
    
    for (unsigned int i=0;i<log_dtau_species.size();i++) {
      log_dtau_species(i) = dtau_species(i) == 0 ? -1e5 : log(dtau_species(i));
      log_dtau_absorber(i) = dtau_absorber(i) == 0 ? -1e5 : log(dtau_species(i));
      log_abs(i) = abs(i) == 0 ? -1e5 : log(abs(i));
    }
    
    init=true;
  }

  void eigen_to_vec() {
    for (unsigned int i_voxel=0;i_voxel<n_voxels;i_voxel++) {
      dtau_species_vec[i_voxel] = dtau_species(i_voxel);
      log_dtau_species_vec[i_voxel] = log_dtau_species(i_voxel);
      dtau_absorber_vec[i_voxel] = dtau_absorber(i_voxel);
      log_dtau_absorber_vec[i_voxel] = log_dtau_absorber(i_voxel);
      sourcefn_vec[i_voxel] = sourcefn(i_voxel);
      log_sourcefn_vec[i_voxel] = log_sourcefn(i_voxel);
    }
  }


};


#endif
