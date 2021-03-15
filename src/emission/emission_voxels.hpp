//emission_voxels.hpp --- emissions associated with a voxel grid

#ifndef __emission_voxels_h_
#define __emission_voxels_h_

#include "emission.hpp"
#include "voxel_vector.hpp"

template <int N_VOXELS, typename emission_type, template<bool,int> class los_tracker_type>
struct emission_voxels : emission<emission_type, los_tracker_type> {
protected:
  typedef emission<emission_type, los_tracker_type> parent;
  typedef emission_voxels<N_VOXELS,emission_type,los_tracker_type> this_emission_type;
  
  using parent::internal_name;
  using parent::internal_init;
  using parent::internal_solved;
  
  static const unsigned int n_voxels = N_VOXELS;

  //these store physical atmospheric parameters on the grid (dimension n_voxels)
  //dynamic arrays are required for dim>~32 for Eigen
  //the vectors point to Eigen objects so that these can be used interchangably
  typedef voxel_vector<N_VOXELS> vv;
  typedef voxel_matrix<N_VOXELS> vm;

  //Radiative transfer parameters
  vm influence_matrix; //influence matrix has dimensions n_voxels, n_voxels)
  
  //vectors to compute the single scattering have dimensions (n_voxels)  
  vv tau_species_single_scattering;
  vv tau_absorber_single_scattering;
  vv singlescat; 
  
  vv sourcefn; //have dimensions (n_voxels)  

  template <typename V>
  CUDA_CALLABLE_MEMBER
  Real interp_array(const int n_interp_points,
		    const int *indices,
		    const Real *weights,
		    const V &arr) const {
    //basic interpolation method for use in emission_type class update_interp
    Real retval=0;
    for (int i=0;i<n_interp_points;i++)
      retval+=weights[i]*arr[indices[i]];
    return retval;
  }

#ifdef __CUDACC__
  void vector_to_device(vv & device_vec,
			vv & host_vec,
			const bool transfer/*=true*/)
  {
    //if transfer = false vector is allocated on device but not copied
    host_vec.to_device(transfer);
    
    //point the device pointer at the same location we just moved memory to
    checkCudaErrors(
		    cudaMemcpy(&device_vec.vec,
			       &host_vec.d_vec,
			       sizeof(Real*),
			       cudaMemcpyHostToDevice)
		    );
  }
  
  void matrix_to_device(vm & device_mat,
			vm & host_mat,
			const bool transfer/*=true*/) {
    //if transfer = false vector is allocated on device but not copied
    host_mat.to_device(transfer);
    
    //point the device pointer at the same location we just moved memory to
    checkCudaErrors(
		    cudaMemcpy(&device_mat.mat,
			       &host_mat.d_mat,
			       sizeof(Real*),
			       cudaMemcpyHostToDevice)
		    );
  }
#endif
  
public:
  ~emission_voxels() {
#if defined(__CUDACC__) and not defined(__CUDA_ARCH__)
    device_clear();
#endif
  }

  CUDA_CALLABLE_MEMBER
  void reset_solution(__attribute__((unused)) const int i_vox=0) {
    //we only need to reset influence_matrix
#ifndef __CUDA_ARCH__
    //we are running on the CPU, reset using Eigen
    influence_matrix.eigen().setZero();
#else
    // we are inside a GPU kernel, each block resets one row (specified
    // by i_vox), using all threads
    for (int j_vox = threadIdx.x; j_vox < n_voxels; j_vox+=blockDim.x)
      influence_matrix(i_vox, j_vox) = 0;
#endif
    internal_solved=false;
  }

  template <bool transmission, int NV>
  using los = los_tracker_type<transmission, NV>;
  typedef los<true, n_voxels> influence_tracker;
  typedef los<false, n_voxels> brightness_tracker;
  
  using parent::update_tracker_end;

  // update the influence tracker with the contribution from this voxel
  CUDA_CALLABLE_MEMBER
  void update_tracker_influence(const int &current_voxel,
				const Real &pathlength,
				const Real &domega,
				influence_tracker &tracker) const {
    //update influence functions for this voxel
    static_cast<const emission_type*>(this)->update_tracker_start(current_voxel, pathlength, tracker);
    
    //see Bishop1999 for derivation of this formula
    Real coef = domega;
    coef *= tracker.holstein_G_int;
    
    assert(!isnan(coef) && "influence coefficients must be real numbers");
    assert(0<=coef && coef<=1 && "influence coefficients represent transition probabilities");
    
    tracker.influence[current_voxel] += coef;
    
    static_cast<const emission_type*>(this)->update_tracker_end(tracker);
  }

  // compute the single scattering from the input tracker
  template<bool influence>
  CUDA_CALLABLE_MEMBER
  void compute_single_scattering(const int &start_voxel, los<influence,n_voxels> &tracker) {
    tau_species_single_scattering(start_voxel) = tracker.tau_species_final;//line center optical depth
    assert(!isnan(tau_species_single_scattering(start_voxel))
	   && (tau_species_single_scattering(start_voxel) >= 0
	       || tau_species_single_scattering(start_voxel) == -1)
	   && "optical depth must be real and positive, or -1 if point is behind limb");
    
    tau_absorber_single_scattering(start_voxel) = tracker.tau_absorber_final;
    assert(!isnan(tau_absorber_single_scattering(start_voxel))
	   && (tau_absorber_single_scattering(start_voxel) >= 0
	       ||  tau_absorber_single_scattering(start_voxel) == -1)
	   && "optical depth must be real and positive, or -1 if point is behind limb");
    
    singlescat(start_voxel) = tracker.holstein_T_final;
    assert(!isnan(singlescat(start_voxel))
	   && singlescat(start_voxel) >= 0
	   && "single scattering coefficient must be real and positive");
  }

  
  CUDA_CALLABLE_MEMBER
  void accumulate_influence(const int & start_voxel,
			    influence_tracker &tracker,
			    __attribute__((unused)) const int offset=0) {
    // offset allows parallel kernels to write to the same row without
    // collision, this is only used on the GPU
#ifndef __CUDA_ARCH__    
    Real rowsum = 0.0;
    for (unsigned int j_vox = 0; j_vox < n_voxels; j_vox++) {
      influence_matrix(start_voxel, j_vox) += tracker.influence[j_vox];
      rowsum += influence_matrix(start_voxel, j_vox);
    }
    assert(0.0 <= rowsum && rowsum <= 1.0 && "row represents scattering probability from this voxel");
#else
    for (unsigned int j_vox = 0; j_vox < n_voxels; j_vox++) {
      int j_vox_offset = (j_vox + offset) % n_voxels; 
      // ^^^ this starts parallel threads off in different columns
      influence_matrix(start_voxel, j_vox_offset) += tracker.influence[j_vox_offset];
      __syncthreads();
    }
#endif
  }

  void solve() {
    static_cast<emission_type*>(this)->pre_solve();

    MatrixX kernel = MatrixX::Identity(n_voxels, n_voxels);
    kernel -= influence_matrix.eigen();
    
    sourcefn = kernel.partialPivLu().solve(singlescat.eigen()); //partialPivLu has multithreading support
    
    // // iterative solution.
    // Real err = 1;
    // int it = 0;
    // VectorX sourcefn_old(n_voxels);
    // sourcefn_old = singlescat;
    // while (err > ABS && it < 500) {
    // 	sourcefn = singlescat.eigen() + influence_matrix.eigen() * sourcefn_old.eigen();

    // 	err=((sourcefn.eigen()-sourcefn_old.eigen().array().abs()/sourcefn_old.eigen().array()).maxCoeff();
    // 	sourcefn_old = sourcefn;
    // 	it++;
    // }
    // std::cout << "For " << internal_name << std::endl;
    // std::cout << "  Scattering up to order: " << it << " included.\n";
    // std::cout << "  Error at final order is: " << err << " .\n";
      
    internal_solved=true;
  }
  void solve_gpu();
  void transpose_influence_gpu();
  
  // update the brightness using an interpolation point inside a voxel
  CUDA_CALLABLE_MEMBER
  void update_tracker_brightness_interp(const int n_interp_points,
					const int *indices,
					const Real *weights,
					const Real &pathlength,
					brightness_tracker &tracker) const {
    static_cast<const emission_type*>(this)->update_tracker_start_interp(n_interp_points,
									 indices,
									 weights,
									 pathlength,
									 tracker);
    
    Real sourcefn_temp = interp_array(n_interp_points, indices, weights, sourcefn);
    
    static_cast<const emission_type*>(this)->update_tracker_brightness(sourcefn_temp, tracker);
    static_cast<const emission_type*>(this)->update_tracker_end(tracker);
  }

  // update the brightness with the contribution from this voxel
  CUDA_CALLABLE_MEMBER
  void update_tracker_brightness_nointerp(const int &current_voxel,
					  const Real &pathlength,
					  brightness_tracker &tracker) const {
    static_cast<const emission_type*>(this)->update_tracker_start(current_voxel, pathlength, tracker);
    static_cast<const emission_type*>(this)->update_tracker_brightness(sourcefn(current_voxel), tracker);
    static_cast<const emission_type*>(this)->update_tracker_end(tracker);
  }
  
  void save_influence(std::ostream &file) const {
    file << "Here is the influence matrix for " << parent::name() <<":\n" 
	 << influence_matrix.eigen() << "\n\n";
  }

#ifdef __CUDACC__
  using parent::device_emission;
  using parent::copy_trivial_member_to_device;
  using parent::copy_trivial_members_to_device;
  
  // methods to move voxel_vec pointers to/from device
  void copy_to_device_influence() {
    this_emission_type* typed_device_emission = static_cast<this_emission_type*>(device_emission);

    // instantiate the arrays we populate on the device
    bool transfer = false;

    matrix_to_device(typed_device_emission->influence_matrix, influence_matrix, transfer);
    
    vector_to_device(typed_device_emission->tau_species_single_scattering, tau_species_single_scattering, transfer);
    vector_to_device(typed_device_emission->tau_absorber_single_scattering, tau_absorber_single_scattering, transfer);
    vector_to_device(typed_device_emission->singlescat, singlescat, transfer);
    vector_to_device(typed_device_emission->sourcefn, sourcefn, transfer);
  }

  void copy_to_device_brightness() {
    //free some of the influnce stuff if we used it
    influence_matrix.free_d_mat();
    tau_species_single_scattering.free_d_vec();
    tau_absorber_single_scattering.free_d_vec();
    singlescat.free_d_vec();

    this_emission_type* typed_device_emission = static_cast<this_emission_type*>(device_emission);
    
    //copy the stuff we need to do the calculation on the device
    bool transfer = true;
    vector_to_device(typed_device_emission->sourcefn, sourcefn, transfer); 
  }

  using parent::copy_trivial_member_to_host;
  using parent::copy_trivial_members_to_host;

  void copy_influence_to_host() {
    copy_solved_to_host();

    influence_matrix.to_host();
  }
  
  void copy_solved_to_host() {
    static_cast<emission_type*>(this)->copy_trivial_members_to_host();
    
    tau_species_single_scattering.to_host();
    tau_absorber_single_scattering.to_host();
    singlescat.to_host();

    sourcefn.to_host();
  }

  void device_clear() {
    influence_matrix.free_d_mat();
    
    tau_species_single_scattering.free_d_vec();
    tau_absorber_single_scattering.free_d_vec();
    singlescat.free_d_vec(); 
    
    sourcefn.free_d_vec();
    
    parent::device_clear();
  }
#endif
};

template <int N_VOXELS, typename emission_type, template<bool,int> class los_tracker_type>
void emission_voxels<N_VOXELS, emission_type, los_tracker_type>::solve_gpu() {
  //the CUDA matrix library has a LOT more boilerplate than Eigen
  //this is adapted from the LU dense example here:
  //https://docs.nvidia.com/cuda/pdf/CUSOLVER_Library.pdf

  static_cast<emission_type*>(this)->pre_solve_gpu();
  // this ensures that:
  // 1) emiss->influence_matrix represents the kernel, not the influence matrix
  // 2) emiss->sourcefn = emiss->singlescat (solution is found in-place)

#ifdef EIGEN_ROWMAJOR
  // we need to transpose the influence matrix from row-major to
  // column-major before the CUDA utils can solve it
  transpose_influence_gpu();
#endif

  cusolverDnHandle_t cusolverH = NULL;

  cudaStream_t stream = NULL;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  const int m = n_voxels;
  const int lda = m;
  const int ldb = m;

  int info = 0; /* host copy of error info */

  int *d_Ipiv = NULL; /* pivoting sequence */
  int *d_info = NULL; /* error info */
  int lwork = 0; /* size of workspace */
  Real *d_work = NULL; /* device workspace for getrf */

  /* step 1: create cusolver handle, bind a stream */
  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  assert(cudaSuccess == cudaStat1);
  status = cusolverDnSetStream(cusolverH, stream);
  assert(CUSOLVER_STATUS_SUCCESS == status);


  /* step 2: allocate device solver parameters */
  cudaStat1 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
  cudaStat2 = cudaMalloc ((void**)&d_info, sizeof(int));
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);

  /* step 3: determine and allocate working space of getrf */
  //DnS refers to dense single-precision, need to change this if working with doubles
  status = cusolverDnSgetrf_bufferSize(cusolverH,
				       m, m, influence_matrix.d_mat,
				       lda,
				       &lwork);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  cudaStat1 = cudaMalloc((void**)&d_work, sizeof(Real)*lwork);
  assert(cudaSuccess == cudaStat1);


  /* step 4: now we can do LU factorization */
  //this does the work in place--- matrix is replaced with solution!
  //DnS refers to dense single-precision, need to change this if working with doubles
  status = cusolverDnSgetrf(cusolverH,
			    m, m, influence_matrix.d_mat,
			    lda, d_work,
			    d_Ipiv, d_info);

  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);

  cudaStat2 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat2);

  if ( 0 > info ){
    printf("%d-th parameter is wrong \n", -info);
    exit(1);
  }

  /* step 5: finally, we get the solution */
  //this is also done in-place, replacing the vector to be solved with the solution
  //DnS refers to dense single-precision, need to change this if working with doubles
  status = cusolverDnSgetrs(cusolverH,
			    CUBLAS_OP_N,
			    m,
			    1, /* nrhs */
			    influence_matrix.d_mat,
			    lda,
			    d_Ipiv,
			    sourcefn.d_vec,
			    ldb,
			    d_info);

  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);

#ifdef NDEBUG
  //check errors since the assert statements are missing
  if (status != CUSOLVER_STATUS_SUCCESS)
    printf("status not successful in CUDA solve\n");
  checkCudaErrors(cudaStat1);
  checkCudaErrors(cudaStat2);
#endif
  
  /* free resources */
  if (d_Ipiv ) cudaFree(d_Ipiv);
  if (d_info ) cudaFree(d_info);
  if (d_work ) cudaFree(d_work);
  if (cusolverH ) cusolverDnDestroy(cusolverH);
  if (stream ) cudaStreamDestroy(stream);
}

template <int N_VOXELS, typename emission_type, template<bool,int> class los_tracker_type>
void emission_voxels<N_VOXELS, emission_type, los_tracker_type>::transpose_influence_gpu() {
  //transpose the influence matrix so it's column major
  const Real unity = 1.0;
  const Real null  = 0.0;
  cublasHandle_t handle;
  //gettimeofday(&t1, NULL);
  cublasCreate(&handle);

  //allocate space for transpose;
  const int N = n_voxels;
  Real *d_transpose = NULL;
  checkCudaErrors(
		  cudaMalloc((void **) &d_transpose,
			     N*N*sizeof(Real))
		  );
  
  //S here means single precision
  //transpose into swap memory
  cublasSgeam(handle,
	      CUBLAS_OP_T,
	      CUBLAS_OP_N,
	      N, N,
	      &unity, influence_matrix.d_mat, N,
	      &null, influence_matrix.d_mat, N,
	      d_transpose, N);
  
  checkCudaErrors( cudaPeekAtLastError() );
  checkCudaErrors( cudaDeviceSynchronize() );
  
  //reassign to original location
  cublasSgeam(handle, 
	      CUBLAS_OP_N, CUBLAS_OP_N,
	      N, N,
	      &unity, d_transpose, N,
	      &null, d_transpose, N,
	      influence_matrix.d_mat, N);
  
  checkCudaErrors( cudaPeekAtLastError() );
  checkCudaErrors( cudaDeviceSynchronize() );
  
  if (d_transpose) cudaFree(d_transpose);
  cublasDestroy(handle);
}




#endif