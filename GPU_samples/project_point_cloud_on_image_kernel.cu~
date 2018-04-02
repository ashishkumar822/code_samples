#include<iostream>

#include<arc17_computer_vision/project_point_cloud_on_image_kernel.h>

__global__ void project_point_cloud_kernel(int n_points,
                                           int image_width,
                                           int image_height,
                                           float* dev_points,
                                           float* dev_points_projected,
                                           unsigned char* dev_points_rgb,
                                           double* dev_camera_intrinsics,
                                           double* dev_camera_extrinsics,
                                           unsigned char* dev_image,
                                           unsigned char* dev_mask_valid_points
                                           )
{
    int index = blockIdx.x * MAX_THREADS + threadIdx.x;

    if(index < n_points)
    {
        float* point_xyz = dev_points + index * 3;

        dev_mask_valid_points[index] = 0;

        if(!isnan(point_xyz[2]))
        {
            float x = point_xyz[0];
            float y = point_xyz[1];
            float z = point_xyz[2];

            x = dev_camera_extrinsics[0]* x + dev_camera_extrinsics[1]* y + dev_camera_extrinsics[2]* z + dev_camera_extrinsics[3];
            y = dev_camera_extrinsics[4]* x + dev_camera_extrinsics[5]* y + dev_camera_extrinsics[6]* z + dev_camera_extrinsics[7];
            z = dev_camera_extrinsics[8]* x + dev_camera_extrinsics[9]* y + dev_camera_extrinsics[10]* z + dev_camera_extrinsics[11];

            point_xyz[0] = x;
            point_xyz[1] = y;
            point_xyz[2] = z;

            if(z>0)
            {
                x = dev_camera_intrinsics[0]*x + dev_camera_intrinsics[2]*z;
                y = dev_camera_intrinsics[4]*y + dev_camera_intrinsics[5]*z;

                int pixel_x = x/z;
                int pixel_y = y/z;

                if(pixel_x>=0 && pixel_x < image_width && pixel_y>=0 && pixel_y < image_height)
                {
                    float* point_projected = dev_points_projected + index*3;

                    point_projected[0] = pixel_x;
                    point_projected[1] = pixel_y;
                    point_projected[2] = 0;

                    unsigned char* pixel = dev_image + pixel_y* 3* image_width + pixel_x * 3;

                    unsigned char* point_rgb = dev_points_rgb + index*3;

                    point_rgb[0] = pixel[2];
                    point_rgb[1] = pixel[1];
                    point_rgb[2] = pixel[0];


                    dev_mask_valid_points[index] = 1;
                }
            }
        }
    }
}

void project_point_cloud_kernel_GPU(pcl::PointCloud<pcl::PointXYZ>::Ptr& ptr_cloud,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ptr_cloud_RGB,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& ptr_projection_cloud,
                                    cv::Mat& image,
                                    cv::Mat& camera_intrinsics,
                                    cv::Mat& camera_extrinsics)
{

    float* dev_points;
    float* dev_points_projected;
    unsigned char* dev_image;
    unsigned char* dev_points_rgb;
    unsigned char* dev_mask_valid_points;
    double* dev_camera_intrinsics;
    double* dev_camera_extrinsics;

    float* host_points;
    float* host_points_projected;
    unsigned char* host_image;
    unsigned char* host_points_rgb;
    unsigned char* host_mask_valid_points;
    double* host_camera_intrinsics;
    double* host_camera_extrinsics;


    int image_width = image.cols;
    int image_height = image.rows;

    int cloud_width = ptr_cloud->width;
    int cloud_height = ptr_cloud->height;

    int n_points = cloud_width * cloud_height;
    int n_blocks = ceil((float)n_points/MAX_THREADS);

    host_points = (float*)malloc(n_points * 3 * sizeof(float));
    host_points_projected = (float*)malloc(n_points * 3 * sizeof(float));
    host_image = image.data;
    host_points_rgb = (unsigned char*)malloc(n_points * 3 * sizeof(unsigned char));
    host_mask_valid_points = (unsigned char*)malloc(n_points * 3 * sizeof(unsigned char));
    host_camera_intrinsics = (double*)camera_intrinsics.data;
    host_camera_extrinsics = (double*)camera_extrinsics.data;

    cudaMalloc(&dev_points, n_points * 3 * sizeof(float));
    cudaMalloc(&dev_points_projected, n_points * 3 * sizeof(float));
    cudaMalloc(&dev_image, image_width* image_height* 3 * sizeof(unsigned char));
    cudaMalloc(&dev_points_rgb, n_points * 3 * sizeof(unsigned char));
    cudaMalloc(&dev_mask_valid_points, n_points * sizeof(unsigned char));
    cudaMalloc(&dev_camera_intrinsics, 3 * 3 * sizeof(double));
    cudaMalloc(&dev_camera_extrinsics, 4 * 4 * sizeof(double));

    for(int i=0;i<ptr_cloud->height;i++)
        for(int j=0;j<ptr_cloud->width;j++)
        {
            float* point = host_points + i*ptr_cloud->width*3 + j*3;
            point[0] = ptr_cloud->at(j,i).x;
            point[1] = ptr_cloud->at(j,i).y;
            point[2] = ptr_cloud->at(j,i).z;
        }

    cudaError_t err;

    cudaMemcpy(dev_points,host_points,n_points * 3 * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_image,host_image,image_width* image_height* 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_camera_intrinsics,host_camera_intrinsics,3 * 3 * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_camera_extrinsics,host_camera_extrinsics,4 * 4 * sizeof(double),cudaMemcpyHostToDevice);

    dim3 thread_per_block(MAX_THREADS,1, 1);
    dim3 blocks_per_grid(n_blocks,1,1);

    project_point_cloud_kernel<<<blocks_per_grid, thread_per_block>>>( n_points,
                                                                       image_width,
                                                                       image_height,
                                                                       dev_points,
                                                                       dev_points_projected,
                                                                       dev_points_rgb,
                                                                       dev_camera_intrinsics,
                                                                       dev_camera_extrinsics,
                                                                       dev_image,
                                                                       dev_mask_valid_points
                                                                       );
    cudaThreadSynchronize();

    cudaMemcpy(host_points,dev_points,n_points * 3 * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_points_projected,dev_points_projected,n_points * 3 * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_points_rgb,dev_points_rgb,n_points * 3 * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mask_valid_points,dev_mask_valid_points,n_points * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    ptr_cloud->clear();

    for(int i=0;i<n_points;i++)
    {
        if(host_mask_valid_points[i])
        {

            pcl::PointXYZ point;
            pcl::PointXYZRGB point_xyzrgb;
            pcl::PointXYZ point_projected;

            float* point_xyz = host_points + i*3;
            float* point_projected_xyz = host_points_projected + i*3;
            unsigned char* point_rgb = host_points_rgb + i*3;

            point.x = point_xyz[0];
            point.y = point_xyz[1];
            point.z = point_xyz[2];

            point_xyzrgb.x = point_xyz[0];
            point_xyzrgb.y = point_xyz[1];
            point_xyzrgb.z = point_xyz[2];

            point_xyzrgb.r = point_rgb[0];
            point_xyzrgb.g = point_rgb[1];
            point_xyzrgb.b = point_rgb[2];

            point_projected.x = point_projected_xyz[0];
            point_projected.y = point_projected_xyz[1];
            point_projected.z = point_projected_xyz[2];

            ptr_cloud->push_back(point);
            ptr_cloud_RGB->push_back(point_xyzrgb);
            ptr_projection_cloud->push_back(point_projected);
        }
    }

    cudaFree(dev_points);
    cudaFree(dev_points_projected);
    cudaFree(dev_image);
    cudaFree(dev_points_rgb);
    cudaFree(dev_mask_valid_points);
    cudaFree(dev_camera_intrinsics);
    cudaFree(dev_camera_extrinsics);
}
