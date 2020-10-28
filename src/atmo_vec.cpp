 //atmo_vec.cpp -- basic definitions for points, rays, and vectors
#include "atmo_vec.hpp"

//atmo_point
CUDA_CALLABLE_MEMBER
// atmo_point::atmo_point() {
//   // init=false;
// }

// CUDA_CALLABLE_MEMBER
// atmo_point::~atmo_point() { };

// CUDA_CALLABLE_MEMBER
// atmo_point::atmo_point(const atmo_point &copy) {
//   x=copy.x;
//   y=copy.y;
//   z=copy.z;
//   r=copy.r;
//   t=copy.t;
//   p=copy.p;
//   i_voxel=copy.i_voxel;
//   // init=copy.init;
// }

// CUDA_CALLABLE_MEMBER
// atmo_point & atmo_point::operator=(const atmo_point &rhs) {
//   if(this == &rhs) return *this;
//   x=rhs.x;
//   y=rhs.y;
//   z=rhs.z;
//   r=rhs.r;
//   t=rhs.t;
//   p=rhs.p;
//   i_voxel=rhs.i_voxel;
//   // init=rhs.init;

//   return *this;
// }

CUDA_CALLABLE_MEMBER
void atmo_point::rtp(const Real &rr, const Real &tt, const Real &pp) {
  r = rr; t = tt; p = pp;
  x = r*sin(t)*cos(p);
  y = r*sin(t)*sin(p);
  z = r*cos(t);
  // init=true;
  i_voxel = -1;
}

CUDA_CALLABLE_MEMBER  
void atmo_point::xyz(const Real &xx, const Real &yy, const Real &zz) {
  x = xx; y = yy; z = zz;
  r = hypot(hypot(x,y),z);
  t = acos(z/r);
  p = atan2(y,x);
  if (p<0)
    p+=2*M_PI;//puts the negative values on the right branch [0,2pi]
  // init=true;
  i_voxel = -1;
}

CUDA_CALLABLE_MEMBER
void atmo_point::set_voxel_index(const int &ii) {
  // assert(init && "point must be initialized to assign a voxel index.");
  i_voxel = ii;
}

CUDA_CALLABLE_MEMBER
atmo_point atmo_point::operator*(const Real & scale) const {
  atmo_point pt;
  pt.x=x*scale;
  pt.y=y*scale;
  pt.z=z*scale;
  pt.r=r*scale;
  pt.t=t;
  pt.p=p;
  pt.i_voxel=i_voxel;
  return pt;
}

CUDA_CALLABLE_MEMBER
atmo_point atmo_point::operator/(const Real & scale) const {
  atmo_point pt;
  pt.x=x/scale;
  pt.y=y/scale;
  pt.z=z/scale;
  pt.r=r/scale;
  pt.t=t;
  pt.p=p;
  pt.i_voxel=i_voxel;
  return pt;
}


//atmo_voxel
// CUDA_CALLABLE_MEMBER
// atmo_voxel::atmo_voxel() {
//   // init = false;
//   //i_voxel = -1;
// }

// CUDA_CALLABLE_MEMBER
// atmo_voxel::~atmo_voxel() { };

// CUDA_CALLABLE_MEMBER
// atmo_voxel::atmo_voxel(const atmo_voxel &copy) {
//   rbounds[0]=copy.rbounds[0];
//   rbounds[1]=copy.rbounds[1];
//   tbounds[0]=copy.tbounds[0];
//   tbounds[1]=copy.tbounds[1];
//   pbounds[0]=copy.pbounds[0];
//   pbounds[1]=copy.pbounds[1];
//   i_voxel=copy.i_voxel;
//   // init=copy.init;
//   pt=copy.pt;
// }

// CUDA_CALLABLE_MEMBER
// atmo_voxel & atmo_voxel::operator=(const atmo_voxel &rhs) {
//   if(this == &rhs) return *this;
//   rbounds[0]=rhs.rbounds[0];
//   rbounds[1]=rhs.rbounds[1];
//   tbounds[0]=rhs.tbounds[0];
//   tbounds[1]=rhs.tbounds[1];
//   pbounds[0]=rhs.pbounds[0];
//   pbounds[1]=rhs.pbounds[1];
//   i_voxel=rhs.i_voxel;
//   // init=rhs.init;
//   pt=rhs.pt;

//   return *this;
// }




//atmo_ray

// CUDA_CALLABLE_MEMBER
// atmo_ray::atmo_ray() {
//   // init=false;
//   // i_ray=-1;
//   // domega=1.0;
// }
// CUDA_CALLABLE_MEMBER
// atmo_ray::~atmo_ray() { }
// CUDA_CALLABLE_MEMBER
// atmo_ray::atmo_ray(const atmo_ray &copy) {
//   t=copy.t;
//   p=copy.p;
//   cost=copy.cost;
//   sint=copy.sint;
//   i_ray=copy.i_ray;
//   domega=copy.domega;
//   // init=copy.init;
// }
// CUDA_CALLABLE_MEMBER
// atmo_ray & atmo_ray::operator=(const atmo_ray &rhs) {
//   if(this == &rhs) return *this;
//   t=rhs.t;
//   p=rhs.p;
//   cost=rhs.cost;
//   sint=rhs.sint;
//   i_ray=rhs.i_ray;
//   domega=rhs.domega;
//   // init=rhs.init;

//   return *this;
// }

CUDA_CALLABLE_MEMBER
void atmo_ray::tp(const Real &tt, const Real &pp) {
  t=tt;
  p=pp;

  cost = std::cos(t);
  sint = std::sin(t);
  i_ray=-1;
  domega=1.0;
  // init = true;
}

CUDA_CALLABLE_MEMBER
void atmo_ray::set_ray_index(const int &ii, const Real &twt, const Real &pwt) {
  // if (init) {
  i_ray = ii;
  domega = twt*pwt/4/M_PI;
  // }
}







//atmo_vector

// CUDA_CALLABLE_MEMBER
// atmo_vector::atmo_vector() {
//   // init=false;
// }
// CUDA_CALLABLE_MEMBER
// atmo_vector::~atmo_vector() { }
// CUDA_CALLABLE_MEMBER
// atmo_vector::atmo_vector(const atmo_vector &copy) {
//   pt=copy.pt;
//   ray=copy.ray;
//   line_x=copy.line_x;
//   line_y=copy.line_y;
//   line_z=copy.line_z;
//   // init=copy.init;
// }
// CUDA_CALLABLE_MEMBER
// atmo_vector & atmo_vector::operator=(const atmo_vector &rhs) {
//   if(this == &rhs) return *this;
//   pt=rhs.pt;
//   ray=rhs.ray;
//   line_x=rhs.line_x;
//   line_y=rhs.line_y;
//   line_z=rhs.line_z;
//   // init=rhs.init;

//   return *this;
// }
  
CUDA_CALLABLE_MEMBER
void atmo_vector::ptray(atmo_point &ptt, atmo_ray &rayy)
{
  pt=ptt;
  ray=rayy;
  
  //construct the cartesian elements from the point and ray
  //spherical coordinates.

  //Product of four rotation matrices.
  //probably worth refactoring to make this obvious
  //refactoring needs to be independent of Eigen
  line_x = (cos(pt.p)*cos(ray.p)*cos(pt.t)*ray.sint
	    +cos(pt.p)*ray.cost*sin(pt.t)
	    -ray.sint*sin(ray.p)*sin(pt.p));
  line_y = (ray.cost*sin(pt.p)*sin(pt.t)
	    +ray.sint*cos(ray.p)*sin(pt.p)*cos(pt.t)
	    +ray.sint*sin(ray.p)*cos(pt.p));
  line_z = ray.cost*cos(pt.t)-cos(ray.p)*ray.sint*sin(pt.t);

  // init = true;
}

CUDA_CALLABLE_MEMBER
void atmo_vector::ptvec(const atmo_point &ptt, const Real (&vec)[3]) {
  ptxyz(ptt, vec[0], vec[1], vec[2]);
}

CUDA_CALLABLE_MEMBER 
void atmo_vector::ptxyz(const atmo_point &ptt,
			const Real &line_xx,
			const Real &line_yy,
			const Real &line_zz)
{
  //this exists to construct rays from points toward the
  //sun, or rays from the spacecraft through the atmosphere. In both
  //cases the integration is along this line only so we can set
  //domega = 1. We need only set the theta values since these are
  //used in computing intersections.

  pt = ptt;
  
  Real line_mag=hypot(hypot(line_xx,line_yy),line_zz);
  line_x = line_xx/line_mag;
  line_y = line_yy/line_mag;
  line_z = line_zz/line_mag;

  //CUDA coerces to NaN if this isn't split up like this???
  Real costx = line_x*(pt.x/pt.r);
  Real costy = line_y*(pt.y/pt.r);
  Real costz = line_z*(pt.z/pt.r);
  
  ray.cost=costx+costy+costz;
  ray.sint=sqrt(1.0-ray.cost*ray.cost);

  ray.t=acos(ray.cost);
  ray.p=-1.0;
    
  ray.domega=1.0;
  // ray.init=true;

  // init=true;
}

CUDA_CALLABLE_MEMBER
atmo_point atmo_vector::extend(const Real &dist) const {
  atmo_point retpt;

  const Real scale = 1e9;

  Real newx = pt.x/scale; //more baffling CUDA shenanigans here
  newx += (line_x * dist)/scale;
  const Real newy = pt.y/scale + (line_y * dist)/scale;
  const Real newz = pt.z/scale + (line_z * dist)/scale;

  retpt.xyz(newx,newy,newz);

  return retpt*scale;
}

CUDA_CALLABLE_MEMBER
atmo_vector atmo_vector::operator*(const Real & scale) const {
  atmo_vector vec;
  vec.pt=pt*scale;
  vec.ray=ray;
  vec.line_x=line_x;
  vec.line_y=line_y;
  vec.line_z=line_z;
  // vec.init=init;
  return vec;
}

CUDA_CALLABLE_MEMBER
atmo_vector atmo_vector::operator/(const Real & scale) const {
  atmo_vector vec;
  vec.pt=pt/scale;
  vec.ray=ray;
  vec.line_x=line_x;
  vec.line_y=line_y;
  vec.line_z=line_z;
  // vec.init=init;
  return vec;
}
