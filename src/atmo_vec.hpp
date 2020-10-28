//atmo_vec.hpp -- header for points, rays, and vectors

#ifndef __atmo_vec_H
#define __atmo_vec_H

#include "Real.hpp"
#include "cuda_compatibility.hpp"
#include <cmath>
#include <cassert>

struct atmo_point {
  Real x, y, z;
  Real r, t, p;
  int i_voxel; //voxel index to which this point belongs
  // bool init;

  // CUDA_CALLABLE_MEMBER
  // atmo_point();
  // CUDA_CALLABLE_MEMBER
  // ~atmo_point();
  // CUDA_CALLABLE_MEMBER
  // atmo_point(const atmo_point &copy);
  // CUDA_CALLABLE_MEMBER
  // atmo_point & operator=(const atmo_point &rhs);

  CUDA_CALLABLE_MEMBER
  void rtp(const Real &rr, const Real &tt, const Real &pp);
  CUDA_CALLABLE_MEMBER  
  void xyz(const Real &xx, const Real &yy, const Real &zz);

  CUDA_CALLABLE_MEMBER
  void set_voxel_index(const int &ii);

  CUDA_CALLABLE_MEMBER
  atmo_point operator*(const Real & scale) const;
  CUDA_CALLABLE_MEMBER
  atmo_point operator/(const Real & scale) const;
};

struct atmo_voxel {
  Real rbounds[2];
  Real tbounds[2];
  Real pbounds[2];

  int i_voxel; //voxel index to which this point belongs
  // bool init;

  atmo_point pt;
  //atmo_point influence_pt;
  //atmo_point singlescat_pt;

  // CUDA_CALLABLE_MEMBER
  // atmo_voxel();
  // CUDA_CALLABLE_MEMBER
  // ~atmo_voxel();
  // CUDA_CALLABLE_MEMBER
  // atmo_voxel(const atmo_voxel &copy);
  // CUDA_CALLABLE_MEMBER
  // atmo_voxel & operator=(const atmo_voxel &rhs);
};



struct atmo_ray {
  Real t, p; // angles measured from the local vertical of the point in question
  Real cost, sint; //needed for sphere intersections later on, might as well get them now
  int i_ray; //ray index, if the ray comes from the grid
  Real domega; //solid angle belonging to this ray, if on grid; if not on a grid, 1.0 

  // bool init;

  // CUDA_CALLABLE_MEMBER
  // atmo_ray();
  // CUDA_CALLABLE_MEMBER
  // ~atmo_ray();
  // CUDA_CALLABLE_MEMBER
  // atmo_ray(const atmo_ray &copy);
  // CUDA_CALLABLE_MEMBER
  // atmo_ray & operator=(const atmo_ray &rhs);

  CUDA_CALLABLE_MEMBER
  void tp(const Real &tt, const Real &pp);

  CUDA_CALLABLE_MEMBER
  void set_ray_index(const int &ii, const Real &twt, const Real &pwt);  
};



struct atmo_vector {
  atmo_point pt;
  atmo_ray ray;
  
  Real line_x, line_y, line_z;//cartesian vector elements 
  // bool init;

  // CUDA_CALLABLE_MEMBER
  // atmo_vector(); 
  // CUDA_CALLABLE_MEMBER
  // ~atmo_vector();
  // CUDA_CALLABLE_MEMBER
  // atmo_vector(const atmo_vector &copy);
  // CUDA_CALLABLE_MEMBER
  // atmo_vector & operator=(const atmo_vector &rhs);

  CUDA_CALLABLE_MEMBER
  void ptray(atmo_point &ptt, atmo_ray &rayy);
  CUDA_CALLABLE_MEMBER
  void ptvec(const atmo_point &ptt, const Real (&vec)[3]);
  CUDA_CALLABLE_MEMBER
  void ptxyz(const atmo_point &ptt, const Real &line_xx, const Real &line_yy, const Real &line_zz);

  CUDA_CALLABLE_MEMBER
  atmo_point extend(const Real &dist) const;

  CUDA_CALLABLE_MEMBER
  atmo_vector operator*(const Real & scale) const;
  CUDA_CALLABLE_MEMBER
  atmo_vector operator/(const Real & scale) const;  
};

#endif

