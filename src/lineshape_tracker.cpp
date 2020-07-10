//lineshape_tracker.cpp --- holstein integrals computed JIT as lines of sight are traversed
#include "lineshape_tracker.hpp"

#define LAMBDA_MAX 5.0

CUDA_CALLABLE_MEMBER
lineshape_tracker::lineshape_tracker()
{
  //for compatibility with Bishop (1999) --- worse than trapezoidal quadrature
  //lambda = {SHIZGAL_LAMBDAS};
  //weight = {SHIZGAL_WEIGHTS};

  lambda_max = 5.0;

  for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
    lambda[i_lambda] = i_lambda*lambda_max/(n_lambda-1);
    weight[i_lambda] = lambda_max/(n_lambda-1);
    if (i_lambda==0 || i_lambda==n_lambda-1)
      weight[i_lambda] *= 0.5;

    lambda2[i_lambda]            = lambda[i_lambda]*lambda[i_lambda];
    weightfn[i_lambda]           = 1.0;//exp(-lambda2[i_lambda]);
    tau_species_lambda_initial[i_lambda] = 0.0;
  }

  tau_species_initial=0;
  tau_absorber_initial=0;
  holstein_T_initial=1.0;

  max_tau_species = 0;
}
CUDA_CALLABLE_MEMBER
lineshape_tracker::~lineshape_tracker() { }
CUDA_CALLABLE_MEMBER
lineshape_tracker::lineshape_tracker(const lineshape_tracker &copy)
{
  //lambda = {SHIZGAL_LAMBDAS};
  //weight = {SHIZGAL_WEIGHTS};
  lambda_max = lambda_max;
  
  for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
    lambda[i_lambda]   = copy.lambda[i_lambda];
    weight[i_lambda]   = copy.weight[i_lambda];

    lambda2[i_lambda]  = copy.lambda2[i_lambda];
    weightfn[i_lambda] = copy.weightfn[i_lambda];

    lineshape_at_origin[i_lambda] = copy.lineshape_at_origin[i_lambda];
    lineshape[i_lambda] = copy.lineshape[i_lambda];

    tau_species_lambda_initial[i_lambda] = copy.tau_species_lambda_initial[i_lambda];
    tau_species_lambda_final[i_lambda] = copy.tau_species_lambda_final[i_lambda];
  }

  tau_species_initial=copy.tau_species_initial;
  tau_species_final=copy.tau_species_final;

  tau_absorber_initial=copy.tau_absorber_initial;
  tau_absorber_final=copy.tau_absorber_final;

  max_tau_species=copy.max_tau_species;

  holstein_T_initial = copy.holstein_T_final;
  holstein_T_final = copy.holstein_T_final;
  holstein_T_int = copy.holstein_T_int;
  holstein_G_int = copy.holstein_G_int;
}
CUDA_CALLABLE_MEMBER
lineshape_tracker& lineshape_tracker::operator=(const lineshape_tracker &rhs) {
  if(this == &rhs) return *this;

  assert(n_lambda == rhs.n_lambda);

  for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
    lambda2[i_lambda]  = rhs.lambda2[i_lambda];
    weightfn[i_lambda] = rhs.weightfn[i_lambda];

    lineshape_at_origin[i_lambda] = rhs.lineshape_at_origin[i_lambda];
    lineshape[i_lambda] = rhs.lineshape[i_lambda];

    tau_species_lambda_initial[i_lambda] = rhs.tau_species_lambda_initial[i_lambda];
    tau_species_lambda_final[i_lambda] = rhs.tau_species_lambda_final[i_lambda];
  }

  tau_species_initial=rhs.tau_species_initial;
  tau_species_final=rhs.tau_species_final;

  tau_absorber_initial=rhs.tau_absorber_initial;
  tau_absorber_final=rhs.tau_absorber_final;

  max_tau_species=rhs.max_tau_species;

  holstein_T_initial = rhs.holstein_T_final;
  holstein_T_final = rhs.holstein_T_final;
  holstein_T_int = rhs.holstein_T_int;
  holstein_G_int = rhs.holstein_G_int;

  return *this;
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::reset(const Real &T, const Real &T_ref) {
  tau_species_initial=0;
  tau_absorber_initial=0;
  holstein_T_initial=1.0;

  for (int i_lambda = 0; i_lambda<n_lambda; i_lambda++) {
    lineshape_at_origin[i_lambda] = std::sqrt(T_ref/T)*exp(-lambda2[i_lambda]*T/T_ref);
    assert(!std::isnan(lineshape_at_origin[i_lambda])
	   && lineshape_at_origin[i_lambda]>0 && "lineshape must be real and positive");

    tau_species_lambda_initial[i_lambda] = 0.0;
  }
  //max_tau_species not reset because we want to track this across
  //all lines of sight
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::check_max_tau() {
  if (tau_species_final > max_tau_species)
    //we only need to check the line center where the optical depth is greatest
    max_tau_species = tau_species_final;
}
  
CUDA_CALLABLE_MEMBER
void lineshape_tracker::update_start(const Real &T, const Real &T_ref,
				     const Real &dtau_species,
				     const Real &dtau_absorber,
				     const Real &abs,
				     const Real &pathlength)
{
  tau_species_final = (tau_species_initial
		       + dtau_species * pathlength * std::sqrt(T_ref/T));
  assert(!std::isnan(tau_species_final)
	 && tau_species_final>=0
	 && "optical depths must be real numbers");

  tau_absorber_final = ( tau_absorber_initial + dtau_absorber * pathlength ); 
  assert(!std::isnan(tau_absorber_final)
	 && tau_absorber_final>=0
	 && "optical depths must be real numbers");
    

  holstein_T_final = 0;
  holstein_T_int = 0;
  holstein_G_int = 0;
  Real holTcoef;
  Real holstein_T_int_coef;

  for (int i_lambda=0; i_lambda < n_lambda; i_lambda++) {
    lineshape[i_lambda] = std::sqrt(T_ref/T)*exp(-lambda2[i_lambda]*T/T_ref);
    assert(!std::isnan(lineshape[i_lambda])
	   && lineshape[i_lambda]>0 && "lineshape must be real and positive");

    tau_species_lambda_final[i_lambda] = (tau_species_lambda_initial[i_lambda]
					  + dtau_species * pathlength * lineshape[i_lambda]);
    assert(!std::isnan(tau_species_lambda_final[i_lambda])
	   && tau_species_lambda_final[i_lambda]>0 && "optical depth must be real and positive");


    holTcoef = (M_2_SQRTPI
		* weight[i_lambda]
		* lineshape_at_origin[i_lambda]
		/ weightfn[i_lambda]);
    
    holstein_T_final += holTcoef * std::exp(-(tau_absorber_final
					      + tau_species_lambda_final[i_lambda]));
    assert(!std::isnan(holstein_T_final)
	   && holstein_T_final>=0 && "holstein function represents a probability");
    
    
    holstein_T_int_coef = (holTcoef					  
			   *std::exp(-(tau_absorber_initial
				       + tau_species_lambda_initial[i_lambda]))
			   *(1.0 - std::exp(-(dtau_absorber
					      + dtau_species * lineshape[i_lambda])*pathlength))
			   /(abs + lineshape[i_lambda]));
    holstein_T_int += holstein_T_int_coef;
    assert(!std::isnan(holstein_T_int)
	   && holstein_T_int>=0 && "holstein integral must be real and positive");
    
    holstein_G_int += holstein_T_int_coef * lineshape[i_lambda];
    assert(!std::isnan(holstein_G_int)
	   && holstein_G_int>=0 && "holstein integral must be real and positive");
  }
  //check that the last element is not contributing too much to the integral
  assert(!((holstein_T_int > STRICTABS) && (holstein_T_int_coef/holstein_T_int > 1e-2))
	 && "wings of line contribute too much to transmission. Increase lambda_max in lineshape_tracker.");
}

CUDA_CALLABLE_MEMBER
void lineshape_tracker::update_end() {
  tau_species_initial=tau_species_final;
  tau_absorber_initial=tau_absorber_final;
  holstein_T_initial=holstein_T_final;
  
  for (int i_lambda=0; i_lambda < n_lambda; i_lambda++)
    tau_species_lambda_initial[i_lambda] = tau_species_lambda_final[i_lambda];

  check_max_tau();
}
