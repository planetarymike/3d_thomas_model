//RT_grid.h -- basic RT grid definitions
//             specifics for different geometries are in other files

#ifndef __RT_grid_H
#define __RT_grid_H

#include <iostream> // for file output and dialog
#include <cmath>    // for cos and sin
#include "constants.h" // basic parameter definitions
#include "my_clock.h"
#include "atmo_vec.h"
#include "boundaries.h"
#include "influence.h"
#include "observation.h"
#include <Eigen/Dense> //matrix solvers
#include <Eigen/StdVector>
#include <string>
#include <cassert>

using std::string;
using Eigen::VectorXd;
using Eigen::MatrixXd;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(MatrixXd)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(VectorXd)


//structure to hold the atmosphere grid
struct RT_grid {
  int n_emissions;//number of emissions to evaluate at each point in the grid
  vector<string> emission_names;

  int n_dimensions; //dimensionality of the grid
  int n_voxels;//number of grid voxels
  //helper functions to swap between voxel and coordinate indices
  virtual int indices_to_voxel(const vector<int> /*indices*/) { return -1; };
  virtual vector<int> voxel_to_indices(const int /*i_voxel*/) { return vector<int>(); };
  virtual vector<int> point_to_indices(const atmo_point /*pt*/) { return vector<int>(); };

  virtual void setup_voxels() {};
  double rmax,rmin;//max and min altitudes in the atmosphere

  //points inside the voxels to shoot rays from
  vector<atmo_point> pts;
    
  int n_rays;
  vector<atmo_ray> rays;
  virtual void setup_rays() {}; 

  virtual boundary_intersection_stepper ray_voxel_intersections(const atmo_vector &vec) {
    return boundary_intersection_stepper();
  }; 
  
  vector<double> sun_direction; //defined by geometry of derived class
  
  //objects to store source functions, influence matrices

  //these parameters are arrays of dimension (n_emissions, n_voxels)
  //these store physical atmospheric parameters on the grid
  vector<VectorXd> species_density; //densities of scatterers and absorbers on the tabulated grid
  vector<VectorXd> absorber_density; 
  vector<VectorXd> species_sigma;//scatterer cross section on the tabulated grid
  vector<VectorXd> absorber_sigma;//absorber cross section on the tabulated grid
  vector<VectorXd> dtau_species;
  vector<VectorXd> dtau_absorber;
  vector<VectorXd> abs; //ratio of dtau_abs to dtau_species
  
  //Radiative transfer parameters
  vector<MatrixXd> influence_matrix; //influence matrix has dimensions (n_emissions, n_voxels, n_voxels)

  vector<VectorXd> singlescat; //have dimensions (n_emissions, n_voxels)  
  vector<VectorXd> sourcefn; //have dimensions (n_emissions, n_voxels)  
  
  //vectors to compute the single scattering have dimensions (n_emissions, n_voxels)  
  vector<VectorXd> tau_species_single_scattering;
  vector<VectorXd> tau_absorber_single_scattering;


  class qinterp {
  public:
    virtual double interp(const atmo_point pt) { return 0.0; }
  };

  class grid_interpolator {
  public:
    qinterp* dtau_species;
    qinterp* dtau_absorber;
    qinterp* abs;
    qinterp* sourcefn;
  }; 
  
  vector<grid_interpolator> interp;
  virtual void setup_interp() { };
  
  influence &transmission;

  //initialization parameters
  bool voxels_init; //is the geometry of the grid initialized?
  bool rays_init; //is the geometry of the grid initialized?
  bool grid_init; //is the geometry of the grid initialized?
  vector<bool> emission_init;//are the density and sigma values populated?
  bool all_emissions_init;
  bool influence_matrix_init, singlescat_init;//are the RT properties initialized?
  bool solved;//have we solved for the source function?
  bool interp_init;//is the interpolator initialized?

  virtual void save_S(string fname) {}; //save source function to file
  
  RT_grid(int n_emissionss, influence &transmissionn) :
    n_emissions(n_emissionss), transmission(transmissionn) {

    voxels_init = false; 
    rays_init = false; 
    grid_init = false;
    emission_init.resize(n_emissions,false);
    all_emissions_init = false;
    influence_matrix_init = false;
    singlescat_init = false;
    solved = false;
    interp_init = false;

    emission_names.resize(n_emissions);

    species_density.resize(n_emissions);
    absorber_density.resize(n_emissions);
    species_sigma.resize(n_emissions);
    absorber_sigma.resize(n_emissions);
    dtau_species.resize(n_emissions);
    dtau_absorber.resize(n_emissions);
    abs.resize(n_emissions);
    influence_matrix.resize(n_emissions);
    singlescat.resize(n_emissions);
    sourcefn.resize(n_emissions);
    tau_species_single_scattering.resize(n_emissions);
    tau_absorber_single_scattering.resize(n_emissions);
  }

  void init_arrays(int n) {
    n_voxels = n;
    
    for (int i=0;i<n_emissions;i++) {
      species_density[i].resize(n_voxels);
      absorber_density[i].resize(n_voxels);
      species_sigma[i].resize(n_voxels);
      absorber_sigma[i].resize(n_voxels);
      dtau_species[i].resize(n_voxels);
      dtau_absorber[i].resize(n_voxels);
      abs[i].resize(n_voxels);
      influence_matrix[i].resize(n_voxels,n_voxels);
      singlescat[i].resize(n_voxels);
      sourcefn[i].resize(n_voxels);
      tau_species_single_scattering[i].resize(n_voxels);
      tau_absorber_single_scattering[i].resize(n_voxels);
    }
  }
  
  //evaluate a member function of class C at the coordinates for a
  //given voxel, storing the result in v
  template <typename C, typename V>
  void voxel_function(C &obj, double (C::*function)(const atmo_point pt), V &v) {
    assert(grid_init && "initialize grid before calling member function");
      
    for (int row=0;row<n_voxels;row++)
      v[row]=(obj.*function)(pts[row]);

  }
  

  template<typename C>
  void define_emission(int n,
		       string emission_name,
		       C &atmosphere,
		       double (C::*species_density_function)(const atmo_point) ,
		       double (C::*species_sigma_function)(const atmo_point),
		       double (C::*absorber_density_function)(const atmo_point),
		       double (C::*absorber_sigma_function)(const atmo_point)) {
    assert(n<n_emissions && n>=0 && "attempt to set invalid emission in define_emitter");
    assert(grid_init && "attempt to set emissions before initializing grid in define_emitter.");

    emission_names[n] = emission_name;
  
    voxel_function(atmosphere,species_density_function,species_density[n]);
    voxel_function(atmosphere,species_sigma_function,species_sigma[n]);

    voxel_function(atmosphere,absorber_density_function,absorber_density[n]);
    voxel_function(atmosphere,absorber_sigma_function,absorber_sigma[n]);

    //define differential optical depths by coefficientwise multiplication
    dtau_species[n] = species_density[n].array() * species_sigma[n].array();
    dtau_absorber[n] = absorber_density[n].array() * absorber_sigma[n].array();
    abs[n] = dtau_absorber[n].array() / dtau_species[n].array();

    emission_init[n] = true;

    all_emissions_init = true;
    for (int i=0;i<n_emissions;i++)
      if (!emission_init[i])
	all_emissions_init = false;
  }





  template <typename R>
  void voxel_traverse(const atmo_vector &v,
		      void (RT_grid::*function)(boundary_intersection_stepper& ,R& ),
		      R &retval)
  {
    assert(all_emissions_init && "!nitialize grid and influence function before calling voxel_traverse.");
  
    //get boundary intersections
    boundary_intersection_stepper stepper = ray_voxel_intersections(v);

    if (stepper.boundaries.size() == 0)
      return;
    
    stepper.origin();

    while(stepper.inside) {
      //call the desired function in each voxel
      (this->*function)(stepper, retval);
    
      stepper.next();
    }
  }


  struct tau_tracker {
    int n_emissions;
    vector<double> tau_species_initial;
    vector<double> tau_species_final;
    vector<double> tau_absorber_initial;
    vector<double> tau_absorber_final;
    double max_tau_species;
    
    tau_tracker(int n_emissionss)
      : n_emissions(n_emissionss)
    {
      tau_species_initial.resize(n_emissions,0.);
      tau_species_final.resize(n_emissions,0.);
      tau_absorber_initial.resize(n_emissions,0.);
      tau_absorber_final.resize(n_emissions,0.);
      
      max_tau_species = 0;
    }

    void check_max_tau() {
      for (int i_emission=0; i_emission < n_emissions; i_emission++) {
	if (tau_species_final[i_emission] > max_tau_species)
	  max_tau_species = tau_species_final[i_emission];
      }
    }
    
    void update_start(boundary_intersection_stepper& stepper,
		      vector<VectorXd> &dtau_species,
		      vector<VectorXd> &dtau_absorber) {
      for (int i_emission=0; i_emission < n_emissions; i_emission++) {
	tau_species_final[i_emission] = ( tau_species_initial[i_emission]
					  + (dtau_species[i_emission](stepper.current_voxel)
					     * stepper.pathlength));
	tau_absorber_final[i_emission] = ( tau_absorber_initial[i_emission]
					   + (dtau_absorber[i_emission](stepper.current_voxel)
					      * stepper.pathlength)); 
      }
    }
    
    void update_end() {
      for (int i_emission=0; i_emission < n_emissions; i_emission++) {
	tau_species_initial[i_emission]=tau_species_final[i_emission];
	tau_absorber_initial[i_emission]=tau_absorber_final[i_emission];
      }
      check_max_tau();
    }
  };


  
  void influence_update(boundary_intersection_stepper& stepper, tau_tracker& los) {
    //update the influence matrix for each emission

    los.update_start(stepper,dtau_species,dtau_absorber);
      
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {   
    
      //see Bishop1999 for derivation of this formula
      double coef = stepper.vec.ray.domega;
      coef *= (transmission.T(los.tau_species_initial[i_emission])
	       - transmission.T(los.tau_species_final[i_emission]) 
	       - abs[i_emission](stepper.current_voxel)*( transmission.Tint(los.tau_species_final[i_emission])
							  - transmission.Tint(los.tau_species_initial[i_emission]) ));
    
      influence_matrix[i_emission](stepper.start_voxel,
				   stepper.current_voxel) += coef;
    
    }

    los.update_end();

  }








  void get_single_scattering_optical_depths(boundary_intersection_stepper& stepper, double& max_tau_species) {
    
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {
      tau_species_single_scattering[i_emission](stepper.start_voxel) += (dtau_species[i_emission](stepper.current_voxel)
									 * stepper.pathlength);
      tau_absorber_single_scattering[i_emission](stepper.start_voxel) += (dtau_absorber[i_emission](stepper.current_voxel)
									  * stepper.pathlength);
    
      double tscomp = tau_species_single_scattering[i_emission](stepper.start_voxel);
      max_tau_species = tscomp > max_tau_species ? tscomp : max_tau_species;    
    }
  }


  void get_single_scattering(atmo_point &pt, double &max_tau_species) {

    if (pt.z<0&&pt.x*pt.x+pt.y*pt.y<rmin*rmin) {
      //if the point is behind the planet, no single scattering
      for (int i_emission=0; i_emission < n_emissions; i_emission++)
	singlescat[i_emission](pt.i_voxel)=0.0;
    } else {
      atmo_vector vec = atmo_vector(pt, sun_direction);
      voxel_traverse(vec,
		     &RT_grid::get_single_scattering_optical_depths,
		     max_tau_species);
      
      for (int i_emission=0; i_emission < n_emissions; i_emission++) {
	singlescat[i_emission](pt.i_voxel) = ( (transmission.T(tau_species_single_scattering[i_emission](pt.i_voxel))
						* exp(-tau_absorber_single_scattering[i_emission](pt.i_voxel)) )
					       / species_sigma[i_emission](pt.i_voxel));
      }
    }
  }

  void solve() {
    assert(all_emissions_init && influence_matrix_init && singlescat_init && "initialize before solving!");
  
    MatrixXd kernel(n_voxels,n_voxels);
  
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {
      for (int row=0;row<n_voxels;row++) {
	for (int col=0;col<n_voxels;col++) {
	  kernel(row,col) = row==col ? 1.0-influence_matrix[i_emission](row,col) : -influence_matrix[i_emission](row,col);
	}
      }


      sourcefn[i_emission]=kernel.partialPivLu().solve(singlescat[i_emission]);//partialPivLu has multithreading support

      // double err = 1;
      // int it = 0;
      // VectorXd sourcefn_old(n_voxels);
      // sourcefn_old.setZero();
      // while (err > 1e-3 && it < 500) {
      // 	sourcefn[i_emission] = singlescat[i_emission] + influence_matrix[i_emission] * sourcefn_old;

      // 	err=((sourcefn[i_emission]-sourcefn_old).array().abs()/sourcefn_old.array()).maxCoeff();
      // 	sourcefn_old = sourcefn[i_emission];
      // 	it++;
      // }
      // std::cout << "Scattering up to order: " << it << " included.\n";
      // std::cout << "Error at final order is: " << err << " .\n";
    
      //check solution:
      /* double check = (( */
      /* 		      sourcefn[i_emission].array() */
      /* 		      / */
      /* 		      (singlescat[i_emission]+influence_matrix[i_emission]*sourcefn[i_emission]).array() */
      /* 		       )-1.0).matrix().norm(); */
      /* std::cout << "check for " << emission_names[i_emission] << " is = " << check << std::endl; */

    }
  
    solved=true;
  
  }

  //generate source functions on the grid
  void generate_S() {
  
    //start timing
    my_clock clk;
    clk.start();

    for (int i=0;i<n_emissions;i++) {
      influence_matrix[i].setZero();
      tau_species_single_scattering[i].setZero();
      tau_absorber_single_scattering[i].setZero();
    }

    
    atmo_vector vec;
    double max_tau_species = 0;
  
#pragma omp parallel for firstprivate(vec) shared(max_tau_species,std::cout) default(none)
    for (int i_pt = 0; i_pt < n_voxels; i_pt++) {
      double omega = 0.0; // make sure sum(domega) = 4*pi
    
      //now integrate outward along the ray grid:
      for (int i_ray=0; i_ray < n_rays; i_ray++) {
	vec = atmo_vector(pts[i_pt], rays[i_ray]);
	omega += vec.ray.domega;

	tau_tracker ttrack(n_emissions);
      
	voxel_traverse(vec, &RT_grid::influence_update, ttrack);

	if (ttrack.max_tau_species > max_tau_species)
	  max_tau_species = ttrack.max_tau_species;
      }

      if ((omega - 1.0 > 1e-6) || (omega - 1.0 < -1e-6)) {
	std::cout << "omega != 4*pi\n";
	throw(10);
      }
    
      //now compute the single scattering function:
      get_single_scattering(pts[i_pt], max_tau_species);
    
    }
  
    //now we've initialized the important parameters
    influence_matrix_init=true;
    singlescat_init=true;
  
    //solve for the source function
    solve();
    solved = true;

    setup_interp();
    //std::cout << "max_tau_species = " << max_tau_species << std::endl;
  
    // print time elapsed
    clk.stop();
    clk.print_elapsed();
    
    return;
  }
  
  struct brightness_tracker : tau_tracker {
    vector<double> brightness;
    
    brightness_tracker(int n_emissionss) : tau_tracker(n_emissionss) {
      brightness.resize(n_emissionss,0.);
    }
  };

  void brightness_traverse(boundary_intersection_stepper& stepper, brightness_tracker &los) {
    los.update_start(stepper,dtau_species,dtau_absorber);
    
    for (int i_emission=0; i_emission < n_emissions; i_emission++) {
      if (stepper.exits_bottom) 
	los.tau_absorber_final[i_emission] = -1;
      
      los.brightness[i_emission] += (sourcefn[i_emission](stepper.current_voxel)

				     *(transmission.Tint(los.tau_species_final[i_emission])
				       - transmission.Tint(los.tau_species_initial[i_emission]) 

				       - (abs[i_emission](stepper.current_voxel)

					  *(transmission.Tintint(los.tau_species_final[i_emission])
					    - transmission.Tintint(los.tau_species_initial[i_emission])))));
    }

    los.update_end();
  }


  virtual vector<double> brightness(atmo_vector &vec) {
    assert(solved && "equation must be solved before brightness can be integrated!");
    
    brightness_tracker los(n_emissions);
    
    voxel_traverse(vec,
		   &RT_grid::brightness_traverse,
		   los);

    
    return los.brightness;
  }
   virtual vector<vector<double>> brightness(vector<atmo_vector> &vecs) {
    vector<vector<double>> retval;
    
    retval.resize(vecs.size(),vector<double>(n_emissions,0.));
#pragma omp parallel for shared(retval,vecs) default(none)
    for(unsigned int i=0; i<vecs.size(); i++) {
      retval[i].resize(n_emissions);
      retval[i] = brightness(vecs[i]);
    }

    return retval;
  }
  virtual void brightness(observation &obs) {
    obs.brightness = brightness(obs.obs_vecs);
  }






  
  //interpolated brightness routine
  vector<double> interp_brightness(atmo_vector &vec) {
    assert(interp_init && "interpolation object must be defined before brightness can be integrated!");
    
    boundary_intersection_stepper stepper = ray_voxel_intersections(vec);

    if (stepper.boundaries.size() == 0)
      return vector<double>(n_emissions,0);

    brightness_tracker los(n_emissions);
    
    //we can use the stepper intersections to simplify the calculation
    int n_subsamples = 10;
    for(unsigned int i_bound=1;i_bound<stepper.boundaries.size();i_bound++) {
      double d_start=stepper.boundaries[i_bound-1].distance;
      double d_step=(stepper.boundaries[i_bound].distance-d_start)/(n_subsamples-1);

      for (int i_step=1;i_step<n_subsamples;i_step++) {

	atmo_point pt = vec.extend(d_start+i_step*d_step);
	
	for (int i_emission=0;i_emission<n_emissions;i_emission++) {

	  los.tau_species_final[i_emission] = ( los.tau_species_initial[i_emission]
						+ (interp[i_emission].dtau_species->interp(pt)
						   * d_step));
	  los.tau_absorber_final[i_emission] = ( los.tau_absorber_initial[i_emission]
						 + (interp[i_emission].dtau_absorber->interp(pt)
						    * d_step)); 

	  
	  los.brightness[i_emission] += (interp[i_emission].sourcefn->interp(pt)
					 
					 *(transmission.Tint(los.tau_species_final[i_emission])
					   - transmission.Tint(los.tau_species_initial[i_emission]) 
					   
					   - (interp[i_emission].abs->interp(pt)
					      
					      *(transmission.Tintint(los.tau_species_final[i_emission])
						- transmission.Tintint(los.tau_species_initial[i_emission])))));


	  los.tau_species_initial[i_emission]=los.tau_species_final[i_emission];
	  los.tau_absorber_initial[i_emission]=los.tau_absorber_final[i_emission];

	}
      }
    }

    if (stepper.exits_bottom) 
      los.tau_absorber_final = vector<double>(n_emissions,-1);
    
    return los.brightness;
  }
  virtual vector<vector<double>> interp_brightness(vector<atmo_vector> &vecs) {
    vector<vector<double>> retval;
    
    retval.resize(vecs.size(),vector<double>(n_emissions,0.));
#pragma omp parallel for shared(retval,vecs) default(none)
    for(unsigned int i=0; i<vecs.size(); i++) {
      retval[i].resize(n_emissions);
      retval[i] = interp_brightness(vecs[i]);
    }

    return retval;
  }
  virtual void interp_brightness(observation &obs) {
    obs.brightness = interp_brightness(obs.obs_vecs);
  }


  


  
};


#endif
