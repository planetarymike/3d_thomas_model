//generate_source_function.cpp -- program to generate a source
//function for comparison with analytic solutions and other models

#include "Real.hpp"
#include "grid_dims.hpp"
#include "cuda_compatibility.hpp"
#include "atm/temperature.hpp"
#include "atm/chamb_diff_1d.hpp"
#include "RT_grid.hpp"
#include "grid_plane_parallel.hpp"
#include "grid_spherical_azimuthally_symmetric.hpp"

int main(__attribute__((unused)) int argc, __attribute__((unused)) char* argv[]) {

  //define the physical atmosphere
  Real exobase_temp = 200;//K
  Real H_T_ref = exobase_temp;//K
  krasnopolsky_temperature temp(exobase_temp);

  Real H_exobase_density = 5e5;// cm-3
  Real CO2_exobase_density = 2e8;//cm-3
  chamb_diff_1d atm(H_exobase_density,
		    CO2_exobase_density,
		    temp);
  // // fix temperature to the exobase temp, eliminate CO2 absorption to compare with JY
  atm.temp_dependent_sH=false;
  atm.constant_temp_sH=exobase_temp;
  //atm.no_CO2_absorption = true;
  atm.save("test/test_atmosphere.dat");
  
  //use holstein functions to compute influence integrals
  typedef holstein_approx influence_function; //setting up holstein_approx is the most time consuming part of startup, ~0.4s

  //define the RT grid
  static const int n_emissions = 2;
  string emission_names[n_emissions] = {"H Lyman alpha", "H Lyman beta"};
  
  // typedef plane_parallel_grid<n_radial_boundaries,
  // 		                 n_rays_theta> grid_type;
  // atm.spherical = false;
  typedef spherical_azimuthally_symmetric_grid<n_radial_boundaries,
					       n_sza_boundaries,
					       n_rays_phi,
					       n_rays_theta> grid_type;
  RT_grid<n_emissions,
	  grid_type,
	  influence_function> RT(emission_names);
  std::cout << "size of RT grid: " << sizeof(RT) << "\n";

  //RT.grid.rmethod = RT.grid.rmethod_altitude;
  RT.grid.rmethod = RT.grid.rmethod_log_n_species;
  //RT.grid.szamethod = RT.grid.szamethod_uniform; //requires CONEABS = 1e-2 in Real.hpp
  RT.grid.szamethod = RT.grid.szamethod_uniform_cos;

  RT.grid.raymethod_theta = RT.grid.raymethod_theta_uniform;
  
  RT.grid.setup_voxels(atm);
  RT.grid.setup_rays();
  
  

  //solve for H lyman alpha
  RT.define_emission("H Lyman alpha",
		     1.0,
		     H_T_ref, atm.sH_lya(H_T_ref),
		     atm,
		     &chamb_diff_1d::nH,   &chamb_diff_1d::H_Temp,
		     &chamb_diff_1d::nCO2, &chamb_diff_1d::sCO2_lya);
  //solve for H lyman beta
  RT.define_emission("H Lyman beta",
		     lyman_beta_branching_ratio,
		     H_T_ref, atm.sH_lyb(H_T_ref),
		     atm,
		     &chamb_diff_1d::nH,   &chamb_diff_1d::H_Temp,
		     &chamb_diff_1d::nCO2, &chamb_diff_1d::sCO2_lyb);

  //solve for the source function
  //  RT.save_influence = true;
#ifndef __CUDACC__
  RT.generate_S();
#else
  //RT.generate_S();
  RT.generate_S_gpu();
  RT.generate_S_gpu();
#endif
  //now print out the output
  RT.save_S("test/test_source_function.dat");

  string sfn_name = "test/test_source_function_";
  sfn_name += std::to_string(n_radial_boundaries) + "x";
  sfn_name += std::to_string(n_sza_boundaries) + "x";
  sfn_name += std::to_string(n_rays_phi) + "x";
  sfn_name += std::to_string(n_rays_theta) + ".dat";
  RT.save_S(sfn_name);


  //simulate a fake observation
  observation<n_emissions> obs(emission_names);

  Real g[n_emissions] = {lyman_alpha_typical_g_factor, lyman_beta_typical_g_factor};
  obs.set_emission_g_factors(g);
  
  std::cout << "lyman alpha g factor is: " << lyman_alpha_typical_g_factor << std::endl;
  std::cout << "lyman alpha line center cross section coef is: "
  	    <<  lyman_alpha_line_center_cross_section_coef << std::endl;
  std::cout << "lyman alpha tau=1 brightness at " << exobase_temp << " K : "
  	    <<   (lyman_alpha_typical_g_factor/
  		  (lyman_alpha_line_center_cross_section_coef/std::sqrt(exobase_temp))
  		  /1e9)
  	    << " kR" << std::endl;
  
  std::cout << "lyman beta g factor is: " << lyman_beta_typical_g_factor << std::endl;
  std::cout << "lyman beta line center cross section coef is: "
  	    <<  lyman_beta_line_center_cross_section_coef << std::endl;
  std::cout << "lyman beta tau=1 brightness at " << exobase_temp << " K : "
  	    <<   (lyman_beta_typical_g_factor/
  		  (lyman_beta_line_center_cross_section_coef/std::sqrt(exobase_temp))
  		  /1e6)
  	    << " R" << std::endl;
  std::cout << std::endl;

  Real dist = 30*rMars;

#ifndef __CUDACC__
  //CPU-only code
  obs.fake(dist,30,600);
  observation<n_emissions> obs_nointerp(emission_names);
  obs_nointerp = obs;

  std::cout << "Performing brightness calculation without interpolation...\n";
  RT.brightness_nointerp(obs_nointerp);
  obs_nointerp.save_brightness("test/test_brightness_nointerp.dat");
  std::cout << "Performing interpolated brightness calculation...\n";
  RT.brightness(obs);
  obs.save_brightness("test/test_brightness.dat");
#else
  //GPU code
  vector<int> sizes = {10,100,300,600/*,1200,2400*/};

  for (auto&& size: sizes) {
    std::cout << "simulating image size "<< size << "x" << size << ":" << std::endl;
    obs.fake(dist,30,size);
    RT.brightness_gpu(obs);

    my_clock save_clk;
    save_clk.start();
    string fname = "test/test_brightness_gpu" + std::to_string(size) + "x" + std::to_string(size) + ".dat";
    obs.save_brightness(fname);  
    save_clk.stop();
    save_clk.print_elapsed("saving file takes ");
    
    std::cout << std::endl;
  }
#endif

  return 0; 
}

