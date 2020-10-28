//observation_fit.hpp -- routines to fit an atmosphere observation

#ifndef __OBSERVATION_FIT_H
#define __OBSERVATION_FIT_H

#include "Real.hpp"
#include "observation.hpp"
#include "atm/temperature.hpp"
#include "atm/chamberlain_exosphere.hpp"
#include "atm/chamb_diff_1d.hpp"
#include "atm/chamb_diff_1d_asymmetric.hpp"
#include "atm/tabular_1d.hpp"
#include "RT_grid.hpp"
#include "grid_plane_parallel.hpp"
#include "grid_spherical_azimuthally_symmetric.hpp"

class observation_fit {
protected:
  static const int n_emissions = 2;
  const std::string emission_names[n_emissions];// = {"H Lyman alpha",
                                                //    "H Lyman beta"};
					        // nvcc complains about
					        // inline definition, this
					        // needs to go in
					        // constructor

  static const int n_parameters = 2; // nH_exo and some T_exo type
  static const int n_pts_per_derivative = 2; // central difference

  static const int n_simulate_per_emission = n_parameters*n_pts_per_derivative + 1;
  static const int n_simulate = n_emissions*n_simulate_per_emission;
  std::string simulate_names[n_simulate];
  
  observation<n_emissions> obs;
  observation<n_simulate> obs_deriv;
  
  krasnopolsky_temperature temp;
  const Real CO2_exobase_density = 2e8;//cm-3
  chamb_diff_1d atm;//make sure to use the same exobase alt as in Tconv
  chamb_diff_1d_asymmetric atm_asym;//make sure to use the same quantities as in atm
  tabular_1d atm_tabular;

  typedef holstein_approx influence_type;

  typedef plane_parallel_grid<n_radial_boundaries,
			      n_rays_theta> plane_parallel_grid_type;
  RT_grid<n_emissions,
	  plane_parallel_grid_type,
	  influence_type> RT_pp;
  
  typedef spherical_azimuthally_symmetric_grid<n_radial_boundaries,
					       n_sza_boundaries,
					       n_rays_theta,
					       n_rays_phi> sph_azi_sym_grid_type;
  RT_grid<n_emissions,
	  sph_azi_sym_grid_type,
	  influence_type> RT;

  RT_grid<n_simulate,
	  sph_azi_sym_grid_type,
	  influence_type> RT_deriv;

public:
  observation_fit();

  Temp_converter Tconv;//also takes exobase alt argument

  void add_observation(const std::vector<vector<Real>> &MSO_locations,
		       const std::vector<vector<Real>> &MSO_directions);

  void add_observed_brightness(const std::vector<Real> &brightness,
			       const std::vector<Real> &sigma,
			       const int emission = 0);
  
  void set_g_factor(vector<Real> &g);

  void generate_source_function(const Real &nHexo, const Real &Texo,
				const string atmosphere_fname = "",
				const string sourcefn_fname = "",
				bool plane_parallel=false);
  void generate_source_function_effv(const Real &nHexo, const Real &effv_exo,
				     const string atmosphere_fname = "",
				     const string sourcefn_fname = "",
				     bool plane_parallel=false);
  void generate_source_function_lc(const Real &nHexo, const Real &lc_exo,
				   const string atmosphere_fname = "",
				   const string sourcefn_fname = "",
				   bool plane_parallel=false);

  template <typename A>
  void generate_source_function_plane_parallel(const A &atm, const Real &Texo,
					       const string sourcefn_fname = "");
  template <typename A>
  void generate_source_function_sph_azi_sym(const A &atm, const Real &Texo,
					    const string sourcefn_fname = "");
  

  void generate_source_function_asym(const Real &nHexo, const Real &Texo,
				     const Real &asym,
				     const string sourcefn_fname = "");
  
  void generate_source_function_tabular_atmosphere(const Real rmin, const Real rexo, const Real rmax,
						   const std::vector<Real> &alt_nH, const std::vector<Real> &log_nH,
						   const std::vector<Real> &alt_nCO2, const std::vector<Real> &log_nCO2,
						   const std::vector<Real> &alt_temp, const std::vector<Real> &temp,
						   const bool compute_exosphere = false,
						   const bool plane_parallel = false,
						   const string sourcefn_fname="");

  void set_use_CO2_absorption(const bool use_CO2_absorption = true);
  void set_use_temp_dependent_sH(const bool use_temp_dependent_sH = true, const Real constant_temp_sH = -1);

  void set_sza_method_uniform();
  void set_sza_method_uniform_cos();
  
  void reset_H_lya_xsec_coef(const Real xsec_coef = lyman_alpha_line_center_cross_section_coef);
  void reset_H_lyb_xsec_coef(const Real xsec_coef = lyman_beta_line_center_cross_section_coef);
  void reset_CO2_lya_xsec(const Real xsec = CO2_lyman_alpha_absorption_cross_section);
  void reset_CO2_lyb_xsec(const Real xsec = CO2_lyman_beta_absorption_cross_section);
  
  std::vector<std::vector<Real>> brightness();
  std::vector<std::vector<Real>> tau_species_final();
  std::vector<std::vector<Real>> tau_absorber_final();

  std::vector<Real> likelihood_and_derivatives(const Real &nHexo, const Real &Texo);
  void logl();
  void logl_gpu();
};

//might be needed to instantiate template members
//#include "observation_fit.cpp"
//observation_fit hello;

#endif
