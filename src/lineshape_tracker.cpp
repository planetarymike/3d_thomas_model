//lineshape_tracker.cpp --- holstein integrals computed JIT as lines of sight are traversed
#include "lineshape_tracker.hpp"

// CUDA_CALLABLE_MEMBER
// lineshape_tracker::lineshape_tracker() { }
// CUDA_CALLABLE_MEMBER
// lineshape_tracker::~lineshape_tracker() { }
// CUDA_CALLABLE_MEMBER
// lineshape_tracker::lineshape_tracker(const lineshape_tracker &copy)
// {
//   //lambda = {SHIZGAL_LAMBDAS};
//   //weight = {SHIZGAL_WEIGHTS};
  
//   for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
//     // lambda[i_lambda]   = copy.lambda[i_lambda];
//     // weight[i_lambda]   = copy.weight[i_lambda];

//     //lambda2[i_lambda]  = copy.lambda2[i_lambda];
//     //weightfn[i_lambda] = copy.weightfn[i_lambda];

//     lineshape_at_origin[i_lambda] = copy.lineshape_at_origin[i_lambda];
//     //lineshape[i_lambda] = copy.lineshape[i_lambda];

//     tau_species_lambda_initial[i_lambda] = copy.tau_species_lambda_initial[i_lambda];
//     //tau_species_lambda_final[i_lambda] = copy.tau_species_lambda_final[i_lambda];

//     transfer_probability_lambda_initial[i_lambda] = copy.transfer_probability_lambda_initial[i_lambda];
//     // transfer_probability_lambda_voxel[i_lambda] = copy.transfer_probability_lambda_voxel[i_lambda];
//     // transfer_probability_lambda_final[i_lambda] = copy.transfer_probability_lambda_final[i_lambda];
//   }

//   tau_species_initial=copy.tau_species_initial;
//   tau_species_final=copy.tau_species_final;

//   tau_absorber_initial=copy.tau_absorber_initial;
//   tau_absorber_final=copy.tau_absorber_final;

//   max_tau_species=copy.max_tau_species;

//   holstein_T_initial = copy.holstein_T_final;
//   holstein_T_final = copy.holstein_T_final;
//   holstein_T_int = copy.holstein_T_int;
//   holstein_G_int = copy.holstein_G_int;
// }
// CUDA_CALLABLE_MEMBER
// lineshape_tracker& lineshape_tracker::operator=(const lineshape_tracker &rhs) {
//   if(this == &rhs) return *this;

//   for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
//     //lambda2[i_lambda]  = rhs.lambda2[i_lambda];
//     //weightfn[i_lambda] = rhs.weightfn[i_lambda];

//     lineshape_at_origin[i_lambda] = rhs.lineshape_at_origin[i_lambda];
//     //lineshape[i_lambda] = rhs.lineshape[i_lambda];

//     tau_species_lambda_initial[i_lambda] = rhs.tau_species_lambda_initial[i_lambda];
//     //tau_species_lambda_final[i_lambda] = rhs.tau_species_lambda_final[i_lambda];

//     transfer_probability_lambda_initial[i_lambda] = rhs.transfer_probability_lambda_initial[i_lambda];
//     // transfer_probability_lambda_voxel[i_lambda] = rhs.transfer_probability_lambda_voxel[i_lambda];
//     // transfer_probability_lambda_final[i_lambda] = rhs.transfer_probability_lambda_final[i_lambda];
//   }

//   tau_species_initial=rhs.tau_species_initial;
//   tau_species_final=rhs.tau_species_final;

//   tau_absorber_initial=rhs.tau_absorber_initial;
//   tau_absorber_final=rhs.tau_absorber_final;

//   max_tau_species=rhs.max_tau_species;

//   holstein_T_initial = rhs.holstein_T_final;
//   holstein_T_final = rhs.holstein_T_final;
//   holstein_T_int = rhs.holstein_T_int;
//   holstein_G_int = rhs.holstein_G_int;

//   return *this;
// }

CUDA_CALLABLE_MEMBER
Real lineshape_tracker::lambda(int i_lambda) {
  return i_lambda*lambda_max/(n_lambda-1);
}
CUDA_CALLABLE_MEMBER
Real lineshape_tracker::weight(int i_lambda) {
  Real thisweight = lambda_max/(n_lambda-1);
  if (i_lambda==0 || i_lambda==n_lambda-1)
    thisweight *= 0.5;
  return thisweight;
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::init() {
  max_tau_species = 0;
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::reset(const Real &T, const Real &T_ref) {
  tau_species_initial=0;
  tau_absorber_initial=0;
  holstein_T_initial=1.0;

  for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
    lineshape_at_origin[i_lambda] = std::sqrt(T_ref/T)*exp(-lambda(i_lambda)*lambda(i_lambda)*T_ref/T);
    assert(!std::isnan(lineshape_at_origin[i_lambda])
	   && lineshape_at_origin[i_lambda]>0 && "lineshape must be real and positive");

    tau_species_lambda_initial[i_lambda] = 0.0;
    transfer_probability_lambda_initial[i_lambda] = 1.0;
  }
  //max_tau_species not reset because we want to track this across
  //all lines of sight
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::check_max_tau() {
  //we only need to check the line center where the optical depth is greatest
  if (tau_species_final > max_tau_species)
    max_tau_species = tau_species_final;
}
  
CUDA_CALLABLE_MEMBER
void lineshape_tracker::update_start(const Real &T, const Real &T_ref,
				     const Real &dtau_species,
				     const Real &dtau_absorber,
				     const Real &abs,
				     const Real &pathlength)
{
  tau_species_final = (tau_species_initial + dtau_species * pathlength);
  assert(!std::isnan(tau_species_final)
	 && tau_species_final>=0
	 && "optical depths must be real numbers");

  tau_absorber_final = ( tau_absorber_initial + dtau_absorber * pathlength ); 
  assert(!std::isnan(tau_absorber_final)
	 && tau_absorber_final>=0
	 && "optical depths must be real numbers");
    
  //
  holstein_T_final = 0;
  holstein_T_int = 0;
  holstein_G_int = 0;
  Real holTcoef;
  Real holstein_T_int_coef;

  for (int i_lambda = 0; i_lambda < n_lambda; i_lambda++) {
    Real lineshape = std::exp(-lambda(i_lambda) * lambda(i_lambda) * T_ref / T);
    assert(!std::isnan(lineshape) && lineshape > 0 &&
           "lineshape must be real and positive");

    Real tau_species_lambda_final = (tau_species_lambda_initial[i_lambda] +
                                     (dtau_species * pathlength * lineshape));
    assert(!std::isnan(tau_species_lambda_final) &&
           tau_species_lambda_final >= 0 &&
           "optical depth must be real and positive");

    Real transfer_probability_lambda_voxel = std::exp(-(dtau_absorber + dtau_species * lineshape)
						      * pathlength);
    assert(!std::isnan(transfer_probability_lambda_voxel) &&
           transfer_probability_lambda_voxel >= 0 &&
           transfer_probability_lambda_voxel <= 1 &&
           "transfer probability is a probability.");

    Real transfer_probability_lambda_final = (transfer_probability_lambda_initial[i_lambda]
					      * transfer_probability_lambda_voxel);

    holTcoef = (M_2_SQRTPI * weight(i_lambda));
    //		/ weightfn[i_lambda]);

    // holstein T final represents a frequency-averaged absorption and
    // uses lineshape_at_origin
    holstein_T_final += (holTcoef
			 * lineshape_at_origin[i_lambda]
			 * transfer_probability_lambda_final);
    assert(!std::isnan(holstein_T_final) && holstein_T_final >= 0 && holstein_T_final <= 1 &&
           "holstein function represents a probability");

    // holstein_T_int represents a frequency averaged emission and
    // therefore uses lineshape in this voxel
    holstein_T_int_coef = (holTcoef
			   * lineshape
			   * transfer_probability_lambda_initial[i_lambda]
			   * (1.0 - transfer_probability_lambda_voxel)
			   / (abs + lineshape));
    holstein_T_int += holstein_T_int_coef;
    assert(!std::isnan(holstein_T_int) && holstein_T_int >= 0 &&
           (holstein_T_int <= tau_species_final - tau_species_initial ||
            std::abs(holstein_T_int - (tau_species_final - tau_species_initial)) < ABS)
           //  ^^ this allows for small rounding errors
           &&
           "holstein integral must be between 0 and Delta tau b/c 0<=HolT<=1");

    // holstein_G_int represents the frequency averaged emission
    // followed by absorption and uses both lineshape in this voxel
    //(via holstein_T_int_coef) and the lineshape at origin
    holstein_G_int += holstein_T_int_coef * lineshape_at_origin[i_lambda];
    assert(!std::isnan(holstein_G_int) && holstein_G_int >= 0 && holstein_G_int <= 1 &&
           "holstein G integral represents a probability");

    //update the initial values
    tau_species_lambda_initial[i_lambda] = tau_species_lambda_final;
    transfer_probability_lambda_initial[i_lambda] = transfer_probability_lambda_final;
  }
  //check that the last element is not contributing too much to the integral
  assert(!((holstein_T_int > STRICTABS) && (holstein_T_int_coef/holstein_T_int > 1e-2))
	 && "wings of line contribute too much to transmission. Increase lambda_max in lineshape_tracker.");
  //if holstein T is larger than physically possible due to rounding errors, reduce it to the physical limit
  if (holstein_T_int > tau_species_final-tau_species_initial)
      holstein_T_int = tau_species_final-tau_species_initial;
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::update_end() {
  tau_species_initial=tau_species_final;
  tau_absorber_initial=tau_absorber_final;
  holstein_T_initial=holstein_T_final;
  
  // for (int i_lambda=0; i_lambda < n_lambda; i_lambda++) {
  //   tau_species_lambda_initial[i_lambda] = tau_species_lambda_final[i_lambda];
  //   transfer_probability_lambda_initial[i_lambda] = transfer_probability_lambda_final[i_lambda];
  // }
  
  check_max_tau();
}

