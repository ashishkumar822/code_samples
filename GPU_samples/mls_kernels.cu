#include"../includes/mls_kernels.h"
#include"../includes/flann_routines.h"
#include<time.h>
#include<common/time_util.h>

#define point_dim 4


#include<pthread.h>


typedef  struct _estimate_coeffs_args_t_
{
    float* dev_P;
    float* dev_wxP;
    float* dev_weights;
    float* dev_f;
    float* dev_Pxf;
    char* host_valid_mask_chunk;
    int* host_n_nbrs;
    int max_nbrs;
    int n_points;
    int n_coeffs;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusolverDnHandle_t cusolver_handleDn;
    cudaStream_t* cuda_streams;
    pthread_mutex_t* cuda_mutex;
    int thread_id;
}estimate_coeffs_args_t;


typedef struct _knn_args_t_
{
    flann::KDTreeCuda3dIndex<flann::L2<float> >* cuda_3d_index;
    float* dev_points3d_chunk;
    int n_points;
    int* dev_indices;
    float* dev_distances;
    int max_nbrs;
    float srch_radii;
    int  knn_or_radial;
    int thread_id;
}knn_args_t;




__global__ void compute_centroid(float* dev_points3d , float* dev_points3d_chunk, int* dev_indices,
                                 float* dev_distances,int n_points,
                                 int max_nbrs, int min_nbrs,
                                 float sqr_srch_radii,
                                 char* dev_valid_mask,int* dev_n_nbrs, float* dev_centroid)
{

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;


    if(index < n_points )
    {
        float centroid[3];

        centroid[0] = 0.0f;
        centroid[1] = 0.0f;
        centroid[2] = 0.0f;

        int* indices = dev_indices + index * max_nbrs;
        float* distances = dev_distances + index * max_nbrs;

        int n_nbrs = 0;

        //        for(int i=0; (i< max_nbrs) && (indices[i] > -1);i++)
        for(int i=0; (i< max_nbrs) && (distances[i] < sqr_srch_radii);i++)
            n_nbrs++;


         if(dev_valid_mask)
              dev_valid_mask[index] = 0;

          if(n_nbrs > min_nbrs)
        {
            float* ptr_point = dev_points3d_chunk + index * 4;

            for(int i=0; i< n_nbrs;i++)
            {
                ptr_point = dev_points3d + indices[i]*4;

                centroid[0] += ptr_point[0];
                centroid[1] += ptr_point[1];
                centroid[2] += ptr_point[2];
            }


            float* ptr_dev_centroid = dev_centroid + index *3;
            ptr_dev_centroid[0] = centroid[0] / n_nbrs;
            ptr_dev_centroid[1] = centroid[1] / n_nbrs;
            ptr_dev_centroid[2] = centroid[2] / n_nbrs;

            if(dev_valid_mask)
                dev_valid_mask[index] = 1;

            dev_n_nbrs[index] = n_nbrs;
        }
        else
            dev_n_nbrs[index] = 0;

    }
}


__global__ void compute_cov_mtx(float* dev_points3d,int* dev_indices,int n_points,
                                int max_nbrs, int min_nbrs,
                                char* dev_valid_mask ,int* dev_n_nbrs, float* dev_centroid, float* dev_cov_matrices)
{
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;


    if(index < n_points )
    {
        char mask =1;

        if(dev_valid_mask)
            mask = dev_valid_mask[index];

        if(mask)
        {

            float cov_mtx[9];
            float diff[3];

            // UPPER TRIANGULAR
            cov_mtx[0] = 0.0f;    // 0 1 2
            cov_mtx[1] = 0.0f;
            cov_mtx[2] = 0.0f;
            //    cov_mtx[3] = 0.0f;    // 3 4 5
            cov_mtx[4] = 0.0f;
            cov_mtx[5] = 0.0f;
            //    cov_mtx[6] = 0.0f;    // 6 7 8
            //    cov_mtx[7] = 0.0f;
            cov_mtx[8] = 0.0f;

            int* indices = dev_indices + index*max_nbrs;

            float* ptr_point;
            float* ptr_centroid;

            int n_nbrs = dev_n_nbrs[index];

            for(int i=0; i<n_nbrs;i++)
            {
                ptr_point = dev_points3d + indices[i]*4;
                ptr_centroid = dev_centroid + index*3 ;

                diff[0] =  ptr_point[0] - ptr_centroid[0];
                diff[1] =  ptr_point[1] - ptr_centroid[1];
                diff[2] =  ptr_point[2] - ptr_centroid[2];

                cov_mtx[0] += diff[0] * diff[0];
                cov_mtx[1] += diff[0] * diff[1];
                cov_mtx[2] += diff[0] * diff[2];
                cov_mtx[4] += diff[1] * diff[1];
                cov_mtx[5] += diff[1] * diff[2];
                cov_mtx[8] += diff[2] * diff[2];
            }

            //        cov_mtx[3]  = cov_mtx[1];
            //        cov_mtx[6]  = cov_mtx[2];
            //        cov_mtx[7]  = cov_mtx[5];

            float* ptr_dev_cov_matrices = dev_cov_matrices+ index * 9;
            ptr_dev_cov_matrices[0] = cov_mtx[0];
            ptr_dev_cov_matrices[1] = cov_mtx[1];
            ptr_dev_cov_matrices[2] = cov_mtx[2];
            ptr_dev_cov_matrices[3] = cov_mtx[1];
            ptr_dev_cov_matrices[4] = cov_mtx[4];
            ptr_dev_cov_matrices[5] = cov_mtx[5];
            ptr_dev_cov_matrices[6] = cov_mtx[2];
            ptr_dev_cov_matrices[7] = cov_mtx[5];
            ptr_dev_cov_matrices[8] = cov_mtx[8];


            //        cov_mtx[0] /= (n_nbrs*n_nbrs);
            //        cov_mtx[1] /= (n_nbrs*n_nbrs);
            //        cov_mtx[2] /= (n_nbrs*n_nbrs);
            //        cov_mtx[3]  = cov_mtx[1];
            //        cov_mtx[4] /= (n_nbrs*n_nbrs);
            //        cov_mtx[5] /= (n_nbrs*n_nbrs);
            //        cov_mtx[6]  = cov_mtx[2];
            //        cov_mtx[7]  = cov_mtx[5];
            //        cov_mtx[8] /= (n_nbrs*n_nbrs);
        }
    }
}


__global__ void compute_normal(int n_points,char* dev_valid_mask,
                               float* dev_cov_matrices,float* dev_normal)
{
    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // real-valued, because the matrix is symmetric.

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;

    if(index < n_points )
    {
        char mask =1;

        if(dev_valid_mask)
            mask = dev_valid_mask[index];

        if(mask)
        {
            float* ptr_dev_cov_matrices = dev_cov_matrices+ index * 9;

            float ptr_cov_mtx[9];
            ptr_cov_mtx[0] = ptr_dev_cov_matrices[0];
            ptr_cov_mtx[1] = ptr_dev_cov_matrices[1];
            ptr_cov_mtx[2] = ptr_dev_cov_matrices[2];
            ptr_cov_mtx[3] = ptr_dev_cov_matrices[3];
            ptr_cov_mtx[4] = ptr_dev_cov_matrices[4];
            ptr_cov_mtx[5] = ptr_dev_cov_matrices[5];
            ptr_cov_mtx[6] = ptr_dev_cov_matrices[6];
            ptr_cov_mtx[7] = ptr_dev_cov_matrices[7];
            ptr_cov_mtx[8] = ptr_dev_cov_matrices[8];


            // LOOP UNROLLING FO EFFICIENCY AND
            // ONLY 6 ELEMENTS DUE TO SYMMETRIC COVARIANCE MATRIX
            float max  = -1000000000.0f;
            float abs_scalar ;

            abs_scalar = fabsf(ptr_cov_mtx[0]);
            if(max < abs_scalar)max = abs_scalar;
            abs_scalar = fabsf(ptr_cov_mtx[1]);
            if(max < abs_scalar)max = abs_scalar;
            abs_scalar = fabsf(ptr_cov_mtx[2]);
            if(max < abs_scalar)max = abs_scalar;
            abs_scalar = fabsf(ptr_cov_mtx[4]);
            if(max < abs_scalar)max = abs_scalar;
            abs_scalar = fabsf(ptr_cov_mtx[5]);
            if(max < abs_scalar)max = abs_scalar;
            abs_scalar = fabsf(ptr_cov_mtx[8]);
            if(max < abs_scalar)max = abs_scalar;

            if (abs_scalar <= FLT_MIN)
                abs_scalar = 1.0f;

            ptr_cov_mtx[0] /= abs_scalar;
            ptr_cov_mtx[1] /= abs_scalar;
            ptr_cov_mtx[2] /= abs_scalar;
            ptr_cov_mtx[3] /= abs_scalar;
            ptr_cov_mtx[4] /= abs_scalar;
            ptr_cov_mtx[5] /= abs_scalar;
            ptr_cov_mtx[6] /= abs_scalar;
            ptr_cov_mtx[7] /= abs_scalar;
            ptr_cov_mtx[8] /= abs_scalar;

            // COMPUTE ROOTS


            float c0 =      ptr_cov_mtx[0] * ptr_cov_mtx[4] * ptr_cov_mtx[8]
                    + 2.0 * ptr_cov_mtx[1] * ptr_cov_mtx[2] * ptr_cov_mtx[5]
                    - ptr_cov_mtx[0] * ptr_cov_mtx[5] * ptr_cov_mtx[5]
                    - ptr_cov_mtx[4] * ptr_cov_mtx[2] * ptr_cov_mtx[2]
                    - ptr_cov_mtx[8] * ptr_cov_mtx[1] * ptr_cov_mtx[1];

            float c1 = ptr_cov_mtx[0] * ptr_cov_mtx[4] -
                    ptr_cov_mtx[1] * ptr_cov_mtx[1] +
                    ptr_cov_mtx[0] * ptr_cov_mtx[8] -
                    ptr_cov_mtx[2] * ptr_cov_mtx[2] +
                    ptr_cov_mtx[4] * ptr_cov_mtx[8] -
                    ptr_cov_mtx[5] * ptr_cov_mtx[5];

            float c2 = ptr_cov_mtx[0] + ptr_cov_mtx[4] + ptr_cov_mtx[8];

            float roots[3];

            if(fabsf (c0) < FLT_EPSILON)  // one root is 0 -> quadratic equation
            {
                roots[0] = 0.0f;
                float d = c2 * c2 - 4.0f * c1;
                if (d < 0.0)  // no real roots ! THIS SHOULD NOT HAPPEN!
                {
                    roots[2] = 0.5f * c2;   // cz both roots equal no need to multiply again
                    roots[1] = roots[2];
                }
                else
                {
                    float sd = sqrtf(d);
                    roots[2] = 0.5f * (c2 + sd);
                    roots[1] = 0.5f * (c2 - sd);
                }
            }
            else
            {
#define s_inv3 0.33333333333f
#define s_sqrt3 1.73205080757f

                float c2_over_3 = c2 * s_inv3;
                float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;

                float half_b = 0.5f * (c0 + c2_over_3 * (2.0f * c2_over_3 * c2_over_3 - c1));

                float q;
                float rho;
                float theta;

                if (a_over_3 > 0.0f)
                {
                    // a=0.0f;
                    q= half_b * half_b ;
                    rho =0.0f;
                }
                else
                {
                    q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
                    rho = sqrtf(-a_over_3);
                }

                if (q > 0.0f)
                    q=0.0f;

                theta = atan2f(sqrtf(-q), half_b) * s_inv3;

                float cos_theta = cosf(theta);
                float sin_theta = sinf(theta);
                roots[0] = c2_over_3 + 2.0f * rho * cos_theta;
                roots[1] = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
                roots[2] = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

                // Sort in increasing order.
                if (roots[0] > roots[1])
                {
                    float temp = roots[0];
                    roots[0] = roots[1];
                    roots[1] = temp;
                }
                if (roots[1] > roots[2])
                {
                    float temp = roots[1];
                    roots[1] = roots[2];
                    roots[2] = temp;
                    if (roots[0] > roots[1])
                    {
                        temp = roots[0];
                        roots[0] = roots[1];
                        roots[1] = temp;
                    }

                }


                if (roots[0] <= 0)  // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
                {
                    roots[0] = 0.0f;
                    float d = c2 * c2 - 4.0f * c1;
                    if (d < 0.0)  // no real roots ! THIS SHOULD NOT HAPPEN!
                    {
                        roots[2] = 0.5f * c2;   // cz both roots equal no need to multiply again
                        roots[1] = roots[2];
                    }
                    else
                    {
                        float sd = sqrtf(d);
                        roots[2] = 0.5f * (c2 + sd);
                        roots[1] = 0.5f * (c2 - sd);
                    }
                }
            }

            //COMPUTE EIGEN_VECS


            // A-lambda*I   for minimm eigen val
            ptr_cov_mtx[0] -= roots[0];
            ptr_cov_mtx[4] -= roots[0];
            ptr_cov_mtx[8] -= roots[0];

            float vec1[3];
            float vec2[3];
            float vec3[3];

            vec1[0] =   ptr_cov_mtx[1] * ptr_cov_mtx[5] - ptr_cov_mtx[2] * ptr_cov_mtx[4];
            vec1[1] = - ptr_cov_mtx[0] * ptr_cov_mtx[5] + ptr_cov_mtx[2] * ptr_cov_mtx[3];
            vec1[2] =   ptr_cov_mtx[0] * ptr_cov_mtx[4] - ptr_cov_mtx[1] * ptr_cov_mtx[3];

            vec2[0] =   ptr_cov_mtx[1] * ptr_cov_mtx[8] - ptr_cov_mtx[2] * ptr_cov_mtx[7];
            vec2[1] = - ptr_cov_mtx[0] * ptr_cov_mtx[8] + ptr_cov_mtx[2] * ptr_cov_mtx[6];
            vec2[2] =   ptr_cov_mtx[0] * ptr_cov_mtx[7] - ptr_cov_mtx[1] * ptr_cov_mtx[6];

            vec3[0] =   ptr_cov_mtx[4] * ptr_cov_mtx[8] - ptr_cov_mtx[5] * ptr_cov_mtx[7];
            vec3[1] = - ptr_cov_mtx[3] * ptr_cov_mtx[8] + ptr_cov_mtx[5] * ptr_cov_mtx[6];
            vec3[2] =   ptr_cov_mtx[3] * ptr_cov_mtx[7] - ptr_cov_mtx[4] * ptr_cov_mtx[6];

            float len1 = vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2];
            float len2 = vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2];
            float len3 = vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2];

            float* vec;
            float sqrt_len;


            if (len1 > len2 && len1 > len3)
            {
                vec =vec1;
                sqrt_len = sqrtf(len1);

            }
            else if (len2 > len1 && len2 > len3)
            {
                vec =vec2;
                sqrt_len = sqrtf(len2);
            }
            else
            {
                vec =vec3;
                sqrt_len = sqrtf(len3);
            }


            float* ptr_normal = dev_normal+index*3;

            ptr_normal[0] = vec[0] / sqrt_len;
            ptr_normal[1] = vec[1] / sqrt_len;
            ptr_normal[2] = vec[2] / sqrt_len;

            //            ptr_normal[0] = abs_scalar * roots[0];
            //            ptr_normal[1] = abs_scalar * roots[1];
            //            ptr_normal[2] = abs_scalar * roots[2];

            //TO RETURN EIGEN VALUES
            //            roots[0] = abs_scalar * roots[0];
            //            roots[1] = abs_scalar * roots[1];
            //            roots[2] = abs_scalar * roots[2];

        }


    }

}

__global__ void mls_project_on_plane(float* dev_points3d_chunk,float* dev_centroid, int n_points, char* dev_valid_mask,
                                     float* dev_normals,float* dev_mls_points_3d)
{
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;

    if(index < n_points )
    {
        char mask =1;

        if(dev_valid_mask)
            mask = dev_valid_mask[index];

        if(mask)
        {


            float* ptr_dev_normal = dev_normals + index *3;

            float normal[3];
            normal[0] = ptr_dev_normal[0];
            normal[1] = ptr_dev_normal[1];
            normal[2] = ptr_dev_normal[2];

            float* ptr_dev_centroid = dev_centroid + index *3;

            float centroid[3];
            centroid[0] =ptr_dev_centroid[0];
            centroid[1] =ptr_dev_centroid[1];
            centroid[2] =ptr_dev_centroid[2];


            float*  ptr_dev_points3d = dev_points3d_chunk +index*4;

            float point[3];
            point[0] = ptr_dev_points3d[0];
            point[1] = ptr_dev_points3d[1];
            point[2] = ptr_dev_points3d[2];

            float distance = (point[0] - centroid[0])*normal[0]
                    + (point[1] - centroid[1])*normal[1]
                    + (point[2] - centroid[2])*normal[2];


            float*  ptr_dev_mls_points3d = dev_mls_points_3d +index*3;

            ptr_dev_mls_points3d[0] = point[0] - distance * normal[0];
            ptr_dev_mls_points3d[1] = point[1] - distance * normal[1];
            ptr_dev_mls_points3d[2] = point[2] - distance * normal[2];


            //OPTIMIZED NO NEED OF "distance" VARIABLE
            //            ptr_dev_mls_points3d[0] = point[0] -(point[0] - centroid[0])*normal[0]* normal[0];
            //            ptr_dev_mls_points3d[1] = point[1] -(point[1] - centroid[1])*normal[1]* normal[1];
            //            ptr_dev_mls_points3d[2] = point[2] -(point[2] - centroid[2])*normal[2]* normal[2];

        }
    }
}


__global__ void mls_compute_u_v_axis( int n_points, int* dev_n_nbrs,
                                      float* dev_normals,float* dev_u_v_axis,
                                      char* dev_valid_mask ,int polynomial_order)
{
    float precision = 0.00001f;

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;

    if(index < n_points )
    {
        char mask =1;

        if(dev_valid_mask)
            mask = dev_valid_mask[index];

        if(mask)
        {
            int n_nbrs = dev_n_nbrs[index];
            int n_coeffs = ((polynomial_order+2)*(polynomial_order+1))/2;

            if (n_nbrs>= n_coeffs)
            {
                float* ptr_dev_normal = dev_normals + index *3;

                float normal[3];
                normal[0] = ptr_dev_normal[0];
                normal[1] = ptr_dev_normal[1];
                normal[2] = ptr_dev_normal[2];

                float u_axis[3] = {0.0f,0.0f,0.0f};
                float v_axis[3] = {0.0f,0.0f,0.0f};

                //                // Get local coordinate system (Darboux frame)

                if(!( fabsf(normal[0]) < precision* fabsf(normal[2])) || !(fabsf(normal[1]) < precision* fabsf(normal[2])))
                {
                    float inv_norm = 1.0f / sqrtf(normal[0]*normal[0]+normal[1]*normal[1]);

                    v_axis[0] = -normal[1]*inv_norm;
                    v_axis[1] =  normal[0]*inv_norm;
                    v_axis[2] =  0;
                }
                else
                {
                    float inv_norm = 1.0f / sqrtf(normal[1]*normal[1]+normal[2]*normal[2]);

                    v_axis[0] =  0;
                    v_axis[1] = -normal[2]*inv_norm;
                    v_axis[2] =  normal[1]*inv_norm;
                }

                u_axis[0] =   normal[1] * v_axis[2] - normal[2] * v_axis[1];
                u_axis[1] = - normal[0] * v_axis[2] + normal[2] * v_axis[0];
                u_axis[2] =   normal[0] * v_axis[1] - normal[1] * v_axis[0];

                float* ptr_u_axis = dev_u_v_axis + index * 6;
                float* ptr_v_axis = dev_u_v_axis + index * 6 + 3;

                ptr_u_axis[0] = u_axis[0];
                ptr_u_axis[1] = u_axis[1];
                ptr_u_axis[2] = u_axis[2];

                ptr_v_axis[0] = v_axis[0];
                ptr_v_axis[1] = v_axis[1];
                ptr_v_axis[2] = v_axis[2];

            }
        }
    }
}

__global__ void mls_fill_P_mtx_weights(float* dev_points3d,float* point,
                                       int* ptr_indices, int n_nbrs,
                                       float* ptr_u_v_axis,
                                       float* ptr_dev_P,float* ptr_weight,
                                       float* ptr_f, float* normal,
                                       int polynomial_order, float sqr_gauss_param)
{

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;

    if(index < n_nbrs )
    {

        float* u_axis = ptr_u_v_axis;
        float* v_axis = ptr_u_v_axis + 3;

        float  point_nbr_wrt_plane[3];

        float u_coord;
        float v_coord;


        float* ptr_point_nbr = dev_points3d + ptr_indices[index]*4;

        point_nbr_wrt_plane[0] = ptr_point_nbr[0] - point[0];
        point_nbr_wrt_plane[1] = ptr_point_nbr[1] - point[1];
        point_nbr_wrt_plane[2] = ptr_point_nbr[2] - point[2];


        float sqr_dist = point_nbr_wrt_plane[0]*point_nbr_wrt_plane[0]
                +point_nbr_wrt_plane[1]*point_nbr_wrt_plane[1]
                +point_nbr_wrt_plane[2]*point_nbr_wrt_plane[2];


        float weight = expf(-sqr_dist / sqr_gauss_param);
        //weight =1.0f;
        ptr_weight[index] = weight;


        // Transforming coordinates
        u_coord = point_nbr_wrt_plane[0]*u_axis[0]
                + point_nbr_wrt_plane[1]*u_axis[1]
                + point_nbr_wrt_plane[2]*u_axis[2];

        v_coord = point_nbr_wrt_plane[0]*v_axis[0]
                + point_nbr_wrt_plane[1]*v_axis[1]
                + point_nbr_wrt_plane[2]*v_axis[2];

        float ptr_fn = point_nbr_wrt_plane[0] * normal[0]
                + point_nbr_wrt_plane[1] * normal[1]
                + point_nbr_wrt_plane[2] * normal[2];


        ptr_f[index] = ptr_fn * weight;

        //                        if(i==0)
        //                           ptr_f[i] =1.0f;


        // Compute the polynomial's terms at the current point
        float u_pow,v_pow;

        int j = 0;
        u_pow = 1;

        for (int ui = 0; ui <= polynomial_order; ++ui)
        {
            v_pow = 1;
            for (int vi = 0; vi <= polynomial_order - ui; ++vi)
            {
                ptr_dev_P[j*n_nbrs+index] = u_pow * v_pow;
                //  ptr_dev_P[j*n_nbrs+i] =j*i;
                v_pow *= v_coord;
                j++;
            }
            u_pow *= u_coord;
        }
    }
}



__global__ void mls_project_points_on_polynomial( float* dev_mls_points3d,
                                                  float* dev_coeffs,
                                                  float* dev_u_v_axis,
                                                  int polynomial_order,
                                                  int n_coeffs,
                                                  int n_points,
                                                  char* dev_valid_mask,
                                                  int compute_normals,
                                                  float* dev_normals
                                                  )
{

    float precision = 0.00001f;

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ;

    if(index < n_points )
    {
        char mask =1;

        if(dev_valid_mask)
            mask = dev_valid_mask[index];

        if(mask)
        {

            float* ptr_dev_normal = dev_normals + index *3;

            float normal[3];
            normal[0] = ptr_dev_normal[0];
            normal[1] = ptr_dev_normal[1];
            normal[2] = ptr_dev_normal[2];


            float* ptr_dev_coeffs = dev_coeffs + index * n_coeffs;


            float* mls_point =  dev_mls_points3d +index*3;


            //            mls_point[0] +=  1.0f * normal[0];
            //            mls_point[1] +=  1.0f * normal[1];
            //            mls_point[2] +=  1.0f * normal[2];

            mls_point[0] +=  ptr_dev_coeffs[0]* normal[0];
            mls_point[1] +=  ptr_dev_coeffs[0]* normal[1];
            mls_point[2] +=  ptr_dev_coeffs[0]* normal[2];

            //             Compute tangent vectors using the
            //partial derivates evaluated at (0,0)
            //which is c_vec[order_+1] and c_vec[1]
            // REFER TO THE ORIGINAL PAPER

            if (compute_normals)
            {
                float* u_axis = dev_u_v_axis + index * 6;
                float* v_axis = dev_u_v_axis + index * 6 + 3;


                ptr_dev_normal[0] = normal[0]
                        - ptr_dev_coeffs[polynomial_order + 1] * u_axis[0]
                        - ptr_dev_coeffs[1] * v_axis[0];

                ptr_dev_normal[1] = normal[1]
                        - ptr_dev_coeffs[polynomial_order + 1] * u_axis[1]
                        - ptr_dev_coeffs[1] * v_axis[1];

                ptr_dev_normal[2] = normal[2]
                        - ptr_dev_coeffs[polynomial_order + 1] * u_axis[2]
                        - ptr_dev_coeffs[1] * v_axis[2];

            }



        }

    }
}

void compute_P_mtx_weights_with_streams(float* dev_points3d,float* dev_mls_points3d,
                                        int* dev_indices, int n_points,
                                        float* dev_P,float* dev_weights,
                                        float* dev_f,int max_nbrs,
                                        int* host_n_nbrs, float* dev_u_v_axis,
                                        float* dev_normals, char* host_valid_mask_chunk,
                                        int polynomial_order,int n_coeffs,
                                        float sqr_gauss_param,
                                        cudaStream_t* cuda_streams)
{


    int nbrs_processed  = 0;

    for(int i=0;i<n_points;i++)
    {
        if(host_valid_mask_chunk[i])
        {
            int n_nbr = host_n_nbrs[i];

            float* point = dev_mls_points3d + i * 3;
            int* ptr_indices = dev_indices + i * max_nbrs;
            float* ptr_weight = dev_weights + nbrs_processed;
            float* ptr_f = dev_f + nbrs_processed;

            float* ptr_dev_P = dev_P + nbrs_processed*n_coeffs;

            float* ptr_dev_normal = dev_normals + i * 3;

            float* ptr_u_v_axis = dev_u_v_axis + i*6;

            double blocks_required = (double)n_nbr / THREADS_PER_BLOCK;
            int n_blocks = ceil(blocks_required);


            dim3 blocks_per_grid(n_blocks,1,1);
            dim3 thrds_per_block(THREADS_PER_BLOCK,1,1);

            // '0' IS SHARED MEMORY SIZE
            mls_fill_P_mtx_weights<<<blocks_per_grid,thrds_per_block,0, cuda_streams[i]>>>(dev_points3d,
                                                                                           point,
                                                                                           ptr_indices,
                                                                                           n_nbr,
                                                                                           ptr_u_v_axis,
                                                                                           ptr_dev_P,
                                                                                           ptr_weight,
                                                                                           ptr_f,
                                                                                           ptr_dev_normal,
                                                                                           polynomial_order,
                                                                                           sqr_gauss_param);
        }
        nbrs_processed += host_n_nbrs[i];
    }


}





void* estimate_coeffs_multi_gpu(void* arg)
{



    estimate_coeffs_args_t* estimate_coeffs_args = (estimate_coeffs_args_t*)arg;

    float* dev_P = estimate_coeffs_args->dev_P;
    float* dev_wxP = estimate_coeffs_args->dev_wxP;
    float* dev_weights = estimate_coeffs_args->dev_weights;
    float* dev_f = estimate_coeffs_args->dev_f;
    float* dev_Pxf = estimate_coeffs_args->dev_Pxf;
    char* host_valid_mask_chunk = estimate_coeffs_args->host_valid_mask_chunk;
    int* host_n_nbrs = estimate_coeffs_args->host_n_nbrs;
    int max_nbrs = estimate_coeffs_args->max_nbrs;
    int n_points = estimate_coeffs_args->n_points;
    int n_coeffs = estimate_coeffs_args->n_coeffs;
    cublasHandle_t cublas_handle = estimate_coeffs_args->cublas_handle;
    cusparseHandle_t cusparse_handle = estimate_coeffs_args->cusparse_handle;
    cusolverDnHandle_t cusolver_handleDn = estimate_coeffs_args->cusolver_handleDn;
    cudaStream_t* cuda_streams = estimate_coeffs_args->cuda_streams;
    int thread_id = estimate_coeffs_args->thread_id;

    cudaSetDevice(thread_id);

    //    clock_t t1,t2;

    //BLAS  AND SOLVER
    {

        //t1= clock();
        {

            //COMPUTE A , b  MATRIX
            float* dev_A = NULL;
            float* dev_B = NULL;

            cudaMalloc(&dev_A,n_points*n_coeffs*n_coeffs*sizeof(float));

            dev_B = dev_Pxf;

            //BLAS COMPUTATION TO COMPUTE MATRIX "A" and "b"

            {

                float one = 1.0f;
                float zero = 0.0f;

                cusparseMatDescr_t mat_desc_A;
                cusparseCreateMatDescr(&mat_desc_A);

                cusparseSetMatType(mat_desc_A,CUSPARSE_MATRIX_TYPE_GENERAL);
                //            cusparseSetMatIndexBase(mat_desc_A,CUSPARSE_INDEX_BASE_ZERO);


                int host_csr_row_ptr_A[max_nbrs+1];
                int host_csr_col_ptr_A[max_nbrs];


                for(int i=0;i<max_nbrs;i++)
                {
                    host_csr_row_ptr_A[i] =i;
                    host_csr_col_ptr_A[i] =i;
                }
                host_csr_row_ptr_A[max_nbrs] = max_nbrs+0;


                int* dev_csr_row_ptr_A = NULL;
                int* dev_csr_col_ptr_A = NULL;



                cudaMalloc(&dev_csr_row_ptr_A, (max_nbrs+1)*sizeof(int));
                cudaMalloc(&dev_csr_col_ptr_A,max_nbrs*sizeof(int));

                cudaMemcpy(dev_csr_row_ptr_A,host_csr_row_ptr_A,(max_nbrs+1)*sizeof(int),cudaMemcpyHostToDevice);
                cudaMemcpy(dev_csr_col_ptr_A,host_csr_col_ptr_A,max_nbrs*sizeof(int),cudaMemcpyHostToDevice);

                //                t1=clock();

                int nbrs_processed = 0;

                for(int j=0;j<n_points;j++)
                {

                    if(host_valid_mask_chunk[j])
                    {

                        cudaStreamSynchronize(cuda_streams[j]);


                        int n_nbrs = host_n_nbrs[j];
                        int nnz = n_nbrs;

                        int rows_A = n_nbrs;
                        int cols_A = n_nbrs;
                        int rows_B = n_nbrs;
                        int cols_B = n_coeffs;
                        int rows_C = n_nbrs;
                        //     int cols_C = n_coeffs;

                        float* ptr_dev_A = dev_weights +  nbrs_processed;
                        float* ptr_dev_B = dev_P + n_coeffs * nbrs_processed;
                        float* ptr_dev_C = dev_wxP + nbrs_processed* n_coeffs;

                        cusparseSetStream(cusparse_handle,cuda_streams[j]);

                        cusparseScsrmm2(cusparse_handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        rows_A,
                                        cols_B,
                                        cols_A,
                                        nnz,
                                        &one,
                                        mat_desc_A,
                                        ptr_dev_A,
                                        dev_csr_row_ptr_A,
                                        dev_csr_col_ptr_A,
                                        ptr_dev_B,
                                        rows_B,
                                        &zero,
                                        ptr_dev_C,
                                        rows_C);
                    }
                    nbrs_processed += host_n_nbrs[j];
                }

                cusparseDestroyMatDescr(mat_desc_A);

                //                t2 = clock(); std::cout << "MTX 1 TIME = "<< (t2-t1)/1000000.0f << std::endl;

                //                t1=clock();

                nbrs_processed = 0;

                for(int j=0;j<n_points;j++)
                {
                    if(host_valid_mask_chunk[j])
                    {
                        cudaStreamSynchronize(cuda_streams[j]);

                        int n_nbrs = host_n_nbrs[j];

                        int rows_A = n_coeffs; //
                        int cols_A = n_nbrs;
                        int rows_B = n_nbrs;  //OPERATED B means TRANSPOSE
                        int cols_B = n_coeffs;
                        int rows_C = n_coeffs;
                        //  int cols_C = n_coeffs;

                        float* ptr_dev_A = dev_P + n_coeffs * nbrs_processed;
                        float* ptr_dev_B = dev_wxP + n_coeffs * nbrs_processed;
                        float* ptr_dev_C = dev_A +   j*n_coeffs* n_coeffs;

                        cublasSetStream_v2(cublas_handle,cuda_streams[j]);
                        cublasSgemm_v2(cublas_handle,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       rows_A,
                                       cols_B,
                                       cols_A,
                                       &one,
                                       ptr_dev_A,
                                       cols_A,
                                       ptr_dev_B,
                                       rows_B,
                                       &zero,
                                       ptr_dev_C,
                                       rows_C
                                       );
                    }
                    nbrs_processed += host_n_nbrs[j];
                }


                //                t2 = clock(); std::cout << "MTX 2 TIME = "<< (t2-t1)/1000000.0f << std::endl;

                //                t1=clock();

                nbrs_processed = 0;

                for(int j=0;j<n_points;j++)
                {
                    if(host_valid_mask_chunk[j])
                    {
                        cudaStreamSynchronize(cuda_streams[j]);

                        int n_nbrs = host_n_nbrs[j];

                        int rows_A = n_coeffs;
                        int cols_A = n_nbrs;
                        //                    int rows_B = n_nbrs;  //OPERATED B means TRANSPOSE
                        //                    int cols_B = 1;  // NO NEDD CZ B IS A VECTOR THIS TIME
                        //                    int rows_C = n_coeffs;
                        //                    int cols_C = 1;

                        float* ptr_dev_A = dev_P +  n_coeffs * nbrs_processed;
                        float* ptr_dev_B = dev_f +  nbrs_processed;
                        float* ptr_dev_C = dev_B + j * n_coeffs;

                        //ROWS AND COULMNS ARE SWAPPED AS PER DOCUMENT
                        // BECAUSE DUE TO CUBLAS_OP_T SIZE OF "x" changes
                        // HENCE WE HAVE TO SWAP COLS AND ROWS
                        cublasSetStream_v2(cublas_handle,cuda_streams[j]);
                        cublasSgemv_v2(cublas_handle,
                                       CUBLAS_OP_T,
                                       cols_A,
                                       rows_A,
                                       &one,
                                       ptr_dev_A,
                                       cols_A,
                                       ptr_dev_B,
                                       1,
                                       &zero,
                                       ptr_dev_C,
                                       1
                                       );

                    }
                    nbrs_processed += host_n_nbrs[j];
                }

                //                t2 = clock(); std::cout << "MTX 3 TIME = "<< (t2-t1)/1000000.0f << std::endl;



                cudaFree(dev_csr_col_ptr_A);
                cudaFree(dev_csr_row_ptr_A);


            }

            // CUSOLVER

            {

                int* dev_info = NULL;
                cudaMalloc(&dev_info,n_points*sizeof(int));


                int Lwork = 0;
                cusolverDnSpotrf_bufferSize(cusolver_handleDn, CUBLAS_FILL_MODE_LOWER, n_coeffs, dev_A, n_coeffs, &Lwork);


                float* dev_workspace = NULL;
                cudaMalloc(&dev_workspace,n_points*Lwork*sizeof(float));


                for (int j = 0; j < n_points; j++)
                {
                    if (host_valid_mask_chunk[j])
                    {
                        float* dev_A_chunk = dev_A + j*n_coeffs*n_coeffs;
                        float* dev_workspace_chunk = dev_workspace + j*Lwork;
                        int* dev_info_chunk = dev_info + j;
                        cusolverDnSetStream(cusolver_handleDn, cuda_streams[j]);
                        int stat = cusolverDnSpotrf(cusolver_handleDn, CUBLAS_FILL_MODE_LOWER, n_coeffs, dev_A_chunk, n_coeffs, dev_workspace_chunk, Lwork, dev_info_chunk);
                        //                  std::cout << "STATUS = " << stat<< std::endl;
                        //                    break;
                    }
                }

                for (int j = 0; j < n_points; j++)
                {
                    if (host_valid_mask_chunk[j])
                    {
                        cudaStreamSynchronize(cuda_streams[j]);

                        int host_info;
                        cudaMemcpy( &host_info,dev_info + j, sizeof(int), cudaMemcpyDeviceToHost);



                        if (!host_info)
                        {
                            float* dev_A_chunk = dev_A + j*n_coeffs*n_coeffs;
                            float* dev_B_chunk = dev_B + j*n_coeffs;
                            int* dev_info_chunk = dev_info + j;

                            cusolverDnSetStream(cusolver_handleDn, cuda_streams[j]);
                            cusolverDnSpotrs(cusolver_handleDn, CUBLAS_FILL_MODE_LOWER, n_coeffs, 1, dev_A_chunk, n_coeffs, dev_B_chunk, n_coeffs, dev_info_chunk);
                        }
                        else
                            host_valid_mask_chunk[j] = 0;
                    }
                }
                cudaFree(dev_workspace);


                for (int j = 0; j < n_points; j++)
                {
                    if (host_valid_mask_chunk[j])
                    {
                        cudaStreamSynchronize(cuda_streams[j]);

                        int host_info;
                        cudaMemcpy( &host_info,dev_info + j, sizeof(int), cudaMemcpyDeviceToHost);

                        if (host_info)
                            host_valid_mask_chunk[j] = 0;
                    }
                }

                cudaFree(dev_info);

            }

            cudaFree(dev_A);
        }


        //  t2 = clock(); std::cout << (t2-t1)/1000000.0f << std::endl;
    }


}





void* flann_compute_neighbors_multi_gpu(void* args)
{


    knn_args_t* knn_args = (knn_args_t*)args;

    flann::KDTreeCuda3dIndex<flann::L2<float> >*  cuda3d_index = knn_args->cuda_3d_index;
    float* dev_points3d = knn_args->dev_points3d_chunk;
    int no_of_points = knn_args->n_points;
    int* dev_indices = knn_args->dev_indices;
    float* dev_distances = knn_args->dev_distances;
    int  max_knn = knn_args->max_nbrs;
    float  srch_radii = knn_args->srch_radii;
    int  knn_or_radial = knn_args->knn_or_radial;
    int thread_id = knn_args->thread_id;

    cudaSetDevice(thread_id);

    flann::Matrix<float> flann_mtx_points3d(dev_points3d,no_of_points,3,4*sizeof(float));
    flann::Matrix<float> flann_mtx_distance(dev_distances,no_of_points,max_knn);
    flann::Matrix<int> flann_mtx_indices(dev_indices,no_of_points,max_knn);

    flann::SearchParams srch_params;
    srch_params.matrices_in_gpu_ram = true;
    srch_params.use_heap = FLANN_True;

    if(knn_or_radial)
        cuda3d_index->knnSearch(flann_mtx_points3d,flann_mtx_indices,flann_mtx_distance,max_knn,srch_params);
    else
        cuda3d_index->radiusSearch(flann_mtx_points3d,flann_mtx_indices,flann_mtx_distance,srch_radii,srch_params);

}





float total_time;
int total_processed_pts;

bool done = false;

void* print_ud(void* arg)
{
    int counts =0;
    while(!done)
    {
        counts = printf("\rTOTAL TIME FOR %d POINTS = %f SECONDS",total_processed_pts,total_time);

        timeutil_usleep(10000);

        printf("\r");
        for(int i=0;i<counts;i++)
            printf("\b");
    }
    printf("\n");

    return 0;
}


int mls_process_multi_gpu(float* host_points3d,int total_no_of_points, int no_of_points,
                          int max_nbrs ,int min_nbrs, float* host_normals,
                          float srch_radii,char* host_valid_mask,bool knn_or_radial,
                          bool fit_polynomial,int polynomial_order,bool compute_normals,
                          float sqr_gauss_param,  float* host_mls_points3d, bool upsample, float upsample_radii)
{

    float sqr_srch_radii = srch_radii * srch_radii;

    // CUDA DEVICE SETUP

    int cuda_status = -1;

    int n_devices  = -1;
    cudaGetDeviceCount(&n_devices);


    if(n_devices < 1 )
    {
        std::cout << "NO GPU COMPUTING DEVICE FOUND\n";
        return -1;
    }
    else
        std::cout << "NO OF GPU(s) " << n_devices << "\n";

    for(int i=0;i<n_devices;i++)
    {
        cudaSetDevice(i);
        cudaFree(0);
    }


    int n_points_per_chunk = 1024*8;
    int no_of_chunks = ceil((float)no_of_points / n_points_per_chunk);

    if(!no_of_chunks && no_of_points)
        no_of_chunks = 1;


    int n_blocks = n_points_per_chunk / THREADS_PER_BLOCK ;

    dim3 blocks_per_grid(n_blocks,1,1);
    dim3 thrds_per_block(THREADS_PER_BLOCK,1,1);

    cublasHandle_t* cublas_handle = (cublasHandle_t*)calloc(n_devices,sizeof(cublasHandle_t));
    cusparseHandle_t* cusparse_handle = (cusparseHandle_t*)calloc(n_devices,sizeof(cusparseHandle_t));
    cusolverDnHandle_t* cusolver_handleDn = (cusolverDnHandle_t*)calloc(n_devices,sizeof(cusolverDnHandle_t));


    for(int i=0;i<n_devices;i++)
    {
        cudaSetDevice(i);
        cublasCreate_v2(cublas_handle+i);
        cusparseCreate(cusparse_handle+i);
        cusolverDnCreate(cusolver_handleDn+i);
    }

    cudaStream_t** cuda_streams = (cudaStream_t**)calloc(n_devices,sizeof(cudaStream_t*));

    for(int i=0;i<n_devices;i++)
    {
        cudaSetDevice(i);
        cuda_streams[i] = (cudaStream_t*)calloc(n_points_per_chunk,sizeof(cudaStream_t));
        for (int j = 0; j < n_points_per_chunk; j++)
            cudaStreamCreate(cuda_streams[i]+j);
    }

    float** dev_points3d = (float**)calloc(n_devices,sizeof(float*));

    int** dev_indices = (int**)calloc(n_devices,sizeof(int*));
    float** dev_distances = (float**)calloc(n_devices,sizeof(float*));

    char** dev_valid_mask = (char**)calloc(n_devices,sizeof(char*));
    int** dev_n_nbrs = (int**)calloc(n_devices,sizeof(int*));

    float** dev_centroids = (float**)calloc(n_devices,sizeof(float*));
    float** dev_covariance_matrices = (float**)calloc(n_devices,sizeof(float*));
    float** dev_normals = (float**)calloc(n_devices,sizeof(float*));
    float** dev_mls_points3d = (float**)calloc(n_devices,sizeof(float*));

    float** dev_points3d_chunk = (float**)calloc(n_devices,sizeof(float*)) ;
    char** host_valid_mask_chunk = (char**)calloc(n_devices,sizeof(char*)) ;
    float** host_mls_points3d_chunk = (float**)calloc(n_devices,sizeof(float*)) ;
    float** host_normals_chunk = (float**)calloc(n_devices,sizeof(float*)) ;

    flann::KDTreeCuda3dIndex<flann::L2<float> >** cuda3d_index =
            (flann::KDTreeCuda3dIndex<flann::L2<float> >**)calloc(n_devices,sizeof(flann::KDTreeCuda3dIndex<flann::L2<float> >*)) ;

    for(int i=0;i<n_devices;i++)
    {
        cudaSetDevice(i);

        //--MEMORY ALLOCATION ON GPU FOR POINTS
        cuda_status = cudaMalloc(&dev_points3d[i],total_no_of_points * point_dim * sizeof(float));
        cuda_status = cudaMemcpy(dev_points3d[i],host_points3d,total_no_of_points * point_dim * sizeof(float),cudaMemcpyHostToDevice);

        //--MEMORY ALLOCATION ON GPU FOR DISTANCES AND INDICES
        cuda_status = cudaMalloc(&dev_indices[i],n_points_per_chunk*max_nbrs*sizeof(int));
        cuda_status = cudaMalloc(&dev_distances[i],n_points_per_chunk*max_nbrs*sizeof(float));

        //--MASK MEMORY ALLOCATION TO CHECK FILTERED POINTS
        cuda_status = cudaMalloc(&dev_valid_mask[i],n_points_per_chunk*sizeof(char));

        //--NEIGHBORS COUNT MEMORY ALLOCATION
        cuda_status = cudaMalloc(&dev_n_nbrs[i],n_points_per_chunk*sizeof(int));

        //--CENTROIDS MEMORY ALLOCATION
        cuda_status = cudaMalloc(&dev_centroids[i],n_points_per_chunk*3*sizeof(float));

        //--COVARIANCE MATRICES MEMORY ALLOCATION
        cuda_status = cudaMalloc(&dev_covariance_matrices[i],n_points_per_chunk*9*sizeof(float));

        //--NORMALS MEMORY ALLOCATION
        cuda_status = cudaMalloc(&dev_normals[i],n_points_per_chunk*3*sizeof(float));

        //--MLS POINTS ON GPU MEMORY ALLOCATION
        cuda_status = cudaMalloc(&dev_mls_points3d[i],n_points_per_chunk*3*sizeof(float));

        // std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
    }

    {

        std::cout << "CONSTRUCTING TREES ON GPU(s)\n";
        for(int i=0;i<n_devices;i++)
        {
            cudaSetDevice(i);
            cuda3d_index[i] = flann_construct_tree(dev_points3d[i],total_no_of_points);
        }
        std::cout << "TREES CONSTRUCTED\n";

        pthread_t print_updates;
        pthread_create(&print_updates,NULL,print_ud,NULL);


        clock_t t1,t2,t3,t4;

        int points_processed = 0 ;

        //--SURFACE SMOOTHING KERNEL LAUNCHES
        {
            for(int chunks_processed=0;chunks_processed < no_of_chunks;)
            {
                t3 = clock();


                int n_points[n_devices];


                // COMPUTING NUMBER OF POINTS PER GPU
                {
                    int points_remaining = no_of_points - points_processed;

                    for(int j=0;j<n_devices;j++)
                    {

                        int num_points  = n_points_per_chunk;

                        if(points_remaining > 0 && points_remaining < n_points_per_chunk )
                            num_points = points_remaining;
                        else if(points_remaining <= 0)
                            num_points = 0;

                        n_points[j] = num_points;

                        if(num_points)
                        {
                            dev_points3d_chunk[j] = dev_points3d[j] + points_processed * point_dim ;
                            host_valid_mask_chunk[j] = host_valid_mask + points_processed ;
                            host_mls_points3d_chunk[j] = host_mls_points3d + points_processed * 3;
                            host_normals_chunk[j] = host_normals + points_processed * 3;

                            if(j)
                            {
                                dev_points3d_chunk[j] += j * n_points_per_chunk * point_dim;
                                host_valid_mask_chunk[j] +=  j * n_points_per_chunk ;
                                host_mls_points3d_chunk[j] += j* n_points_per_chunk * 3;
                                host_normals_chunk[j] += j* n_points_per_chunk * 3;
                            }
                        }
                        points_remaining -= n_points[j];
                    }
                }


                //--COMPUTE NEIGHBORS on GPU
                {

                    knn_args_t knn_args[n_devices];
                    pthread_t knn_threads[n_devices];

                    for(int j=0;j< n_devices;j++)
                    {
                        if(chunks_processed < no_of_chunks)
                        {
                            if(n_points[j])
                            {
                                knn_args[j].cuda_3d_index = cuda3d_index[j];
                                knn_args[j].dev_points3d_chunk =   dev_points3d_chunk[j];
                                knn_args[j].n_points =   n_points[j];
                                knn_args[j].dev_indices =   dev_indices[j];
                                knn_args[j].dev_distances = dev_distances[j];
                                knn_args[j].max_nbrs =    max_nbrs;
                                knn_args[j].srch_radii =   srch_radii;
                                knn_args[j].knn_or_radial =  knn_or_radial;
                                knn_args[j].thread_id = j;


                                pthread_create(&knn_threads[j],NULL,flann_compute_neighbors_multi_gpu,&knn_args[j]);

                                chunks_processed++;
                            }
                        }
                    }

                    for(int j=0;j<n_devices;j++)
                        if(n_points[j])
                            pthread_join(knn_threads[j],NULL);

                    cudaDeviceSynchronize();
                }


                //--COMPUTE CENTROID ON GPU
                {
                    // t1 = clock();

                    for(int j=0;j<n_devices;j++)
                    {
                        if(n_points[j])
                        {
                            cudaSetDevice(j);
                            compute_centroid<<<blocks_per_grid , thrds_per_block,0,cuda_streams[j][0]>>>(dev_points3d[j],
                                                                                                         dev_points3d_chunk[j],
                                                                                                         dev_indices[j],
                                                                                                         dev_distances[j],
                                                                                                         n_points[j],
                                                                                                         max_nbrs,
                                                                                                         min_nbrs,
                                                                                                         sqr_srch_radii,
                                                                                                         dev_valid_mask[j],
                                                                                                         dev_n_nbrs[j],
                                                                                                         dev_centroids[j]);

                        }
                    }

//                    cuda_status = cudaGetLastError();
//                    std::cout << "CUDA STATUS = " << cuda_status <<std::endl;

                    cuda_status = cudaDeviceSynchronize();
//                     std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
                    // t2 = clock(); std::cout << "CENTROID COMPUTATION TIME = " << (t2-t1)/1000000.0f << std::endl;
                }


//                //--COMPUTE COVARIANCE MATRICES ON GPU
                {
                    //            t1 = clock();
                    for(int j=0;j<n_devices;j++)
                    {
                        if(n_points[j])
                        {
                            cudaSetDevice(j);

                            compute_cov_mtx<<<blocks_per_grid , thrds_per_block ,0,cuda_streams[j][0]>>>(dev_points3d[j],
                                                                                                         dev_indices[j],
                                                                                                         n_points[j],
                                                                                                         max_nbrs,
                                                                                                         min_nbrs,
                                                                                                         dev_valid_mask[j],
                                                                                                         dev_n_nbrs[j],
                                                                                                         dev_centroids[j],
                                                                                                         dev_covariance_matrices[j]);

                        }
                    }

                    cuda_status = cudaDeviceSynchronize();
                    // std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
                    // t2 = clock(); std::cout << "COVARIANCE MTX COMP TIME = " << (t2-t1)/1000000.0f << std::endl;
                }


                //--COMPUTE NORMALS ON GPU
                {
                    // t1 = clock();
                    for(int j=0;j<n_devices;j++)
                    {
                        if(n_points[j])
                        {
                            cudaSetDevice(j);
                            compute_normal<<<blocks_per_grid , thrds_per_block ,0,cuda_streams[j][0]>>>(n_points[j],
                                                                                                        dev_valid_mask[j],
                                                                                                        dev_covariance_matrices[j],
                                                                                                        dev_normals[j]);
                        }
                    }
                    cuda_status = cudaDeviceSynchronize();
                    //std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
                    //t2 = clock(); std::cout << "NORMAL ESTIMATION TIME = " <<  (t2-t1)/1000000.0f << std::endl;
                }


                //--PROJECT POINT ON THE PLANE
                {
                    // t1 = clock();
                    for(int j=0;j<n_devices;j++)
                    {
                        if(n_points[j])
                        {
                            cudaSetDevice(j);
                            mls_project_on_plane<<<blocks_per_grid , thrds_per_block,0,cuda_streams[j][0] >>>(dev_points3d_chunk[j],
                                                                                                              dev_centroids[j],
                                                                                                              n_points[j],
                                                                                                              dev_valid_mask[j],
                                                                                                              dev_normals[j],
                                                                                                              dev_mls_points3d[j]);
                        }
                    }
                    cuda_status = cudaDeviceSynchronize();
                    // std::cout << "CUDA STATUS = " << cuda_status << std::endl;
                    // t2 = clock(); std::cout << "MLS PROJECTION TIME = " <<  (t2-t1)/1000000.0f << std::endl;
                }


                //--COPY MASK TO CPU MEMORY
                {
                    if(host_valid_mask)
                    {
                        for(int j=0;j<n_devices;j++)
                        {
                            cudaSetDevice(j);
                            cudaMemcpyAsync(host_valid_mask_chunk[j],
                                            dev_valid_mask[j],
                                            n_points[j]*sizeof(char),
                                            cudaMemcpyDeviceToHost,
                                            cuda_streams[j][0]);
                        }
                        cudaDeviceSynchronize();
                    }
                }


                //--FIT POLYNOMIAL
                {
                    if(fit_polynomial)
                    {
                        int n_coeffs;
                        n_coeffs = ((polynomial_order + 2) * (polynomial_order + 1))/ 2;

                        float** dev_P = (float**)calloc(n_devices,sizeof(float*)) ;
                        float** dev_wxP = (float**)calloc(n_devices,sizeof(float*)) ;
                        float** dev_weights = (float**)calloc(n_devices,sizeof(float*)) ;
                        float** dev_f = (float**)calloc(n_devices,sizeof(float*)) ;
                        float** dev_Pxf = (float**)calloc(n_devices,sizeof(float*)) ;

                        float** dev_u_v_axis = (float**)calloc(n_devices,sizeof(float*)) ;
                        int** host_n_nbrs = (int**)calloc(n_devices,sizeof(int*));


                        //--MEMORY ALLOCATION FOR POLYOMIAL COEFFICIENTS ESTIMATION
                        {
                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    host_n_nbrs[j] = (int*)calloc(n_points[j],sizeof(int));

                                    cudaMemcpy(host_n_nbrs[j],dev_n_nbrs[j],n_points[j]*sizeof(int),cudaMemcpyDeviceToHost);

                                    int total_nbrs =0;
                                    for(int k=0;k<n_points[j];k++)
                                        total_nbrs += host_n_nbrs[j][k];

                                    cudaMalloc(&dev_P[j], total_nbrs * n_coeffs * sizeof(float));
                                    cudaMalloc(&dev_wxP[j], total_nbrs * n_coeffs * sizeof(float));
                                    cudaMalloc(&dev_weights[j], total_nbrs * sizeof(float));
                                    cudaMalloc(&dev_f[j], total_nbrs * sizeof(float));
                                    cudaMalloc(&dev_Pxf[j], n_points[j] * n_coeffs * sizeof(float));
                                    cudaMalloc(&dev_u_v_axis[j],n_points[j] * 6 * sizeof(float));
                                }
                            }
                        }

                        //--WEGHT COMPUTATION
                        {
                            // t1 = clock();

                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    mls_compute_u_v_axis<<<blocks_per_grid , thrds_per_block ,0,cuda_streams[j][0] >>>(n_points[j],
                                                                                                                       dev_n_nbrs[j],
                                                                                                                       dev_normals[j],
                                                                                                                       dev_u_v_axis[j],
                                                                                                                       dev_valid_mask[j],
                                                                                                                       polynomial_order);
                                }
                            }

                            cuda_status = cudaDeviceSynchronize();
                            // std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
                            // t2 = clock(); std::cout << "WEIGHT MTX COMPUTATION TIME = " <<  (t2-t1)/1000000.0f << std::endl;
                        }

                        //--PxW COMPUTATION
                        {

                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    compute_P_mtx_weights_with_streams(dev_points3d[j],
                                                                       dev_mls_points3d[j],
                                                                       dev_indices[j],
                                                                       n_points[j],
                                                                       dev_P[j],
                                                                       dev_weights[j],
                                                                       dev_f[j],
                                                                       max_nbrs,
                                                                       host_n_nbrs[j],
                                                                       dev_u_v_axis[j],
                                                                       dev_normals[j],
                                                                       host_valid_mask_chunk[j],
                                                                       polynomial_order,
                                                                       n_coeffs,
                                                                       sqr_gauss_param,
                                                                       cuda_streams[j]);
                                }
                            }
                            cuda_status = cudaDeviceSynchronize();
                        }



                        //--COEFFICIENTS ESTIMATION
                        {
                            // t1 = clock();

                            pthread_t PxW_threads[n_devices];
                            estimate_coeffs_args_t PxW_info[n_devices];

                            for(int j=0;j<n_devices;j++)
                            {
                                //                                std::cout << "DEPLOYING THREADS FOR COEFFS ESTIMATION" << std::endl;
                                if(n_points[j])
                                {

                                    PxW_info[j].dev_P =                      dev_P[j];
                                    PxW_info[j].dev_wxP =                    dev_wxP[j];
                                    PxW_info[j].dev_weights =                dev_weights[j];
                                    PxW_info[j].dev_f =                      dev_f[j];
                                    PxW_info[j].dev_Pxf =                    dev_Pxf[j];
                                    PxW_info[j].host_valid_mask_chunk =      host_valid_mask_chunk[j];
                                    PxW_info[j].host_n_nbrs =                host_n_nbrs[j];
                                    PxW_info[j].max_nbrs =                   max_nbrs;
                                    PxW_info[j].n_points =                   n_points[j];
                                    PxW_info[j].n_coeffs =                   n_coeffs;
                                    PxW_info[j].cublas_handle =              cublas_handle[j];
                                    PxW_info[j].cusparse_handle =            cusparse_handle[j];
                                    PxW_info[j].cusolver_handleDn =          cusolver_handleDn[j];
                                    PxW_info[j].cuda_streams =               cuda_streams[j];
                                    PxW_info[j].thread_id = j;

                                    pthread_create(&PxW_threads[j],NULL, estimate_coeffs_multi_gpu,(void*)(&PxW_info[j]));

                                }
                            }

                            for(int j=0;j<n_devices;j++)
                                if(n_points[j])
                                    pthread_join(PxW_threads[j],NULL);
                        }

                        //--COPYING MASKS
                        {
                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    cudaMemcpyAsync(dev_valid_mask[j],
                                                    host_valid_mask_chunk[j],
                                                    n_points[j]*sizeof(char),
                                                    cudaMemcpyHostToDevice,
                                                    cuda_streams[j][0]);
                                }
                            }
                            cudaDeviceSynchronize();
                        }

                        //--PROJECTING POINTS ON POLYNOMIAL
                        {
                            //  t1 = clock();
                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    float* dev_coeffs = dev_Pxf[j];

                                    mls_project_points_on_polynomial<<<blocks_per_grid , thrds_per_block,0,cuda_streams[j][0] >>>(dev_mls_points3d[j],
                                                                                                                                  dev_coeffs,
                                                                                                                                  dev_u_v_axis[j],
                                                                                                                                  polynomial_order,
                                                                                                                                  n_coeffs,
                                                                                                                                  n_points[j],
                                                                                                                                  dev_valid_mask[j],
                                                                                                                                  compute_normals,
                                                                                                                                  dev_normals[j]);
                                }
                            }

                            cuda_status = cudaDeviceSynchronize();
                            // std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
                            // t2 = clock(); std::cout << "PROJECTION ON SMOOTH SURFACE COMPUTATION TIME = " <<  (t2-t1)/1000000.0f << std::endl;
                        }

                        //--MEMORY DEALLOCATION FOR POLYOMIAL COEFFICIENTS ESTIMATION
                        {
                            for(int j=0;j<n_devices;j++)
                            {
                                if(n_points[j])
                                {
                                    cudaSetDevice(j);

                                    free(host_n_nbrs[j]);

                                    cudaFree(dev_P[j]);
                                    cudaFree(dev_wxP[j]);
                                    cudaFree(dev_weights[j]);
                                    cudaFree(dev_f[j]);
                                    cudaFree(dev_Pxf[j]);
                                    cudaFree(dev_u_v_axis[j]);
                                }
                            }

                            free(host_n_nbrs);
                            free(dev_P);
                            free(dev_wxP);
                            free(dev_weights);
                            free(dev_f);
                            free(dev_Pxf);
                            free(dev_u_v_axis);
                        }



                    }
                }


                //--COPY PROCESSED MLS POINTS TO CPU
                {
                    if(host_mls_points3d)
                    {
                        for(int j=0;j<n_devices;j++)
                        {
                            cudaSetDevice(j);
                            cudaMemcpyAsync(host_mls_points3d_chunk[j],
                                            dev_mls_points3d[j],
                                            n_points[j]*3*sizeof(float),
                                            cudaMemcpyDeviceToHost,
                                            cuda_streams[j][0]);

                        }
                        cudaDeviceSynchronize();
                    }
                }

                //--COPY NORMALS TO CPU
                {
                    if(compute_normals)
                    {
                        for(int j=0;j<n_devices;j++)
                        {
                            cudaSetDevice(j);
                            cudaMemcpyAsync(host_normals_chunk[j],
                                            dev_normals[j],
                                            n_points[j]*3*sizeof(float),
                                            cudaMemcpyDeviceToHost,
                                            cuda_streams[j][0]);

                        }
                        cudaDeviceSynchronize();
                    }
                }


                for(int j=0;j<n_devices;j++)
                    points_processed += n_points[j];


                t4 = clock();

                total_time +=(t4-t3)/1000000.0f;
                total_processed_pts = points_processed;
                //                std::cout << "CHUNK NO = " << chunks_processed << " TOTAL POINTS PROCESSED = " << points_processed << "\n";

            }
        }


        //--DESTROYING CUBLAS CUSOLVER CUSPARSE HANDLES
        {
            for(int i=0;i<n_devices;i++)
            {
                cudaSetDevice(i);
                cublasDestroy_v2(cublas_handle[i]);
                cusparseDestroy(cusparse_handle[i]);
                cusolverDnDestroy(cusolver_handleDn[i]);
            }
            free(cublas_handle);
            free(cusparse_handle);
            free(cusolver_handleDn);
        }


        //--DESTROYING STREAMS
        {
            for(int i=0;i<n_devices;i++)
            {
                cudaSetDevice(i);

                for (int j = 0; j < n_points_per_chunk; j++)
                    cudaStreamDestroy(cuda_streams[i][j]);

                free(cuda_streams[i]);
            }
            free(cuda_streams);
        }

        //--FREE TREES ON GPUS
        {
            for(int i=0;i<n_devices;i++)
                delete cuda3d_index[i];

            free(cuda3d_index);
        }

        //--FREE THE MEMORY RESOURCES
        {
            for(int i=0;i<n_devices;i++)
            {
                cudaSetDevice(i);

                //--MEMORY DEALLOCATION ON GPU FOR POINTS
                cudaFree(dev_points3d[i]);

                //--MEMORY DEALLOCATION ON GPU FOR DISTANCES AND INDICES
                cudaFree(dev_indices[i]);
                cudaFree(dev_distances[i]);

                //--MASK MEMORY DEALLOCATION TO CHECK FILTERED POINTS
                cudaFree(dev_valid_mask[i]);

                //--NEIGHBORS COUNT MEMORY DEALLOCATION
                cudaFree(dev_n_nbrs[i]);

                //--CENTROIDS MEMORY DEALLOCATION
                cudaFree(dev_centroids[i]);

                //--COVARIANCE MATRICES MEMORY DEALLOCATION
                cudaFree(dev_covariance_matrices[i]);

                //--NORMALS MEMORY DEALLOCATION
                cudaFree(dev_normals[i]);

                //--MLS POINTS ON GPU MEMORY DEALLOCATION
                cudaFree(dev_mls_points3d[i]);

                // std::cout << "CUDA STATUS = " << cuda_status <<std::endl;
            }

            free(dev_points3d);
            free(dev_indices);
            free(dev_distances);
            free(dev_valid_mask);
            free(dev_n_nbrs);
            free(dev_centroids);
            free(dev_covariance_matrices);
            free(dev_normals);
            free(dev_mls_points3d);
        }


        done =true;
        pthread_join(print_updates,NULL);
        done =false;
        total_time =0;
        total_processed_pts =0;
    }



    return 0;

}



