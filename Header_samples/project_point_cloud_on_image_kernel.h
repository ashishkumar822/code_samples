#ifndef __PROJECT_POINT_CLOUD_KERNEL_GPU_H__
#define __PROJECT_POINT_CLOUD_KERNEL_GPU_H__

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include<pcl/point_cloud.h>
#include<pcl/point_types.h>

#include<opencv2/opencv.hpp>

#define MAX_THREADS 64


void project_point_cloud_kernel_GPU(pcl::PointCloud<pcl::PointXYZ>::Ptr& ptr_cloud,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ptr_cloud_RGB,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& ptr_projection_cloud,
                                    cv::Mat& image,
                                    cv::Mat& camera_intrinsics,
                                    cv::Mat& camera_extrinsics);

#endif
