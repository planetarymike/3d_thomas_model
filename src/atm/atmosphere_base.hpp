#ifndef __ATMOSPHERE_BASE_H_
#define __ATMOSPHERE_BASE_H_

#include "Real.hpp"
#include "atmo_vec.hpp"

struct atmosphere {
  double rmin;// cm, minimum altitude in model atmosphere
  double rexo;// cm, exobase altitude. Needs to remain unless you want
                  // to rewrite rmethod_altitude in grid_plane_parallel and
                  // grid_spherical_azimuthally_symmetric
  double rmax;// cm, max altitude in model atmosphere

  bool init;//whether n_species, r_from_n_species, n_absorber, and
	    //Temp are ready to use
  
  //species density at a given radius (either subsolar or average)
  virtual double n_species(const double &r) const = 0;

  //returns radius (at subsolar point, or average) from species density
  virtual double r_from_n_species(const double &n_species) const = 0;

  //atmosphere temperature
  virtual double Temp(const double &r) const = 0;

  //absorber density
  virtual double n_absorber(const double &r) const = 0; 
  
  //function for cross sections should be defined for use with RT
  //code, but this is not required as some species (H) have multiple
  //emissions with different cross sections
  //
  //virtual double species_sigma(const double &T) const = 0; virtual
  //double absorber_sigma(const double &T) const = 0;

  atmosphere(double rminn, double rexoo, double rmaxx);

  virtual ~atmosphere() = default;
};

#endif
