#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <signal.h>

#include<arc17_computer_vision/arc17_computer_vision.h>
#include<arc17_computer_vision/project_point_cloud_on_image_kernel.h>

cv::Mat foscam_cam_matrix(3, 3, CV_64FC1);
cv::Mat foscam_dist_coeff(3, 3, CV_64FC1);
cv::Mat foscam_depthsensor_R(3, 3, CV_64FC1);
cv::Mat foscam_depthsensor_t(3, 3, CV_64FC1);

Eigen::Matrix4d T_depthsensor;
Eigen::MatrixXd F_depthsensor(3, 4);


std::string str_file_intrinsics_foscam_kinect = "/home/isl-server/ashish/workspace_ros/data/camera_calibration/foscam_kinect_intrinsic.yml";
std::string str_file_extrinsics_foscam_kinect = "/home/isl-server/ashish/workspace_ros/data/camera_calibration/foscam_kinect_extrinsic.yml";
std::string str_file_intrinsics_foscam_ensenso = "/home/isl-server/ashish/workspace_ros/data/camera_calibration/foscam_ensenso_intrinsic.yml";
std::string str_file_extrinsics_foscam_ensenso = "/home/isl-server/ashish/workspace_ros/data/camera_calibration/foscam_ensenso_extrinsic.yml";


void arc17_computer_vision_t::read_calibration_parameteres(void)
{
    cv::FileStorage fs_intrinsics_foscam_depthsensor;
    cv::FileStorage fs_extrinsics_foscam_depthsensor;

    if(DEPTH_SENSOR == "KINECT")
    {
        fs_intrinsics_foscam_depthsensor = cv::FileStorage(str_file_intrinsics_foscam_kinect, cv::FileStorage::READ);
        fs_extrinsics_foscam_depthsensor =  cv::FileStorage(str_file_extrinsics_foscam_kinect, cv::FileStorage::READ);
    }
    else if(DEPTH_SENSOR == "ENSENSO")
    {
        fs_intrinsics_foscam_depthsensor = cv::FileStorage(str_file_intrinsics_foscam_ensenso, cv::FileStorage::READ);
        fs_extrinsics_foscam_depthsensor =  cv::FileStorage(str_file_extrinsics_foscam_ensenso, cv::FileStorage::READ);
    }

    fs_intrinsics_foscam_depthsensor["cam_matrix_hd"] >> foscam_cam_matrix;
    fs_intrinsics_foscam_depthsensor["dist_coeff_hd"] >> foscam_dist_coeff;

    fs_extrinsics_foscam_depthsensor["R"] >> foscam_depthsensor_R;
    fs_extrinsics_foscam_depthsensor["t"] >> foscam_depthsensor_t;

    fs_intrinsics_foscam_depthsensor.release();
    fs_extrinsics_foscam_depthsensor.release();

    for(int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            T_depthsensor(i, j) = foscam_depthsensor_R.at<double>(i, j);
    T_depthsensor(3, 0) = 0.f;
    T_depthsensor(3, 1) = 0.f;
    T_depthsensor(3, 2) = 0.f;
    T_depthsensor(3, 3) = 1.f;

    T_depthsensor(0, 3) = foscam_depthsensor_t.at<double>(0, 0);
    T_depthsensor(1, 3) = foscam_depthsensor_t.at<double>(1, 0);
    T_depthsensor(2, 3) = foscam_depthsensor_t.at<double>(2, 0);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) {
            if(j < 3)
                F_depthsensor(i, j) = foscam_cam_matrix.at<double>(i, j);
            else
                F_depthsensor(i, j) = 0.0f;
        }

    std::cout<< "T_depthsensor\n" << T_depthsensor << std::endl;
    std::cout<< "F_depthsensor\n" << F_depthsensor << std::endl;

}

bool arc17_computer_vision_t::fetch_foscam_image(void)
{
    srv_fetch_foscam_image.request.key.data = KEY;
    if(service_client_foscam.call(srv_fetch_foscam_image))
    {
        cv_bridge::CvImagePtr m_img_ptr = cv_bridge::toCvCopy(srv_fetch_foscam_image.response.middle_rgb_img);
        cv::Mat image = m_img_ptr->image;

        cv::undistort(image, foscam_image, foscam_cam_matrix, foscam_dist_coeff);

        return  true;
    }
    else
        return false;
}

void arc17_computer_vision_t::project_point_cloud(void)
{

    ptr_projection_cloud->clear();
    ptr_cloud_RGB->clear();

    for(int i = 0; i < ptr_cloud->width; i++)
        for(int j = 0; j < ptr_cloud->height; j++)
        {

            pcl::PointXYZRGB point;
            point.x = ptr_cloud->at(i, j).x;
            point.y = ptr_cloud->at(i, j).y;
            point.z = ptr_cloud->at(i, j).z;

            Eigen::Vector4d k_point;
            if(pcl::isFinite(point))
            {
                k_point << point.x, point.y, point.z, 1.0f;

                Eigen::Vector4d c_point = T_depthsensor * k_point;
                Eigen::Vector3d h_point = F_depthsensor * c_point;


                if(h_point(2))
                {
                    int x = std::floor(h_point(0)/h_point(2));
                    int y = std::floor(h_point(1)/h_point(2));

                    if(x < detection_roi.x + detection_roi.width && x >= detection_roi.x && y < detection_roi.y + detection_roi.height && y >= detection_roi.y)
                    {
                        unsigned char* pixel = foscam_image.data + y * foscam_image.step[0] + x * foscam_image.step[1];

                        point.r = pixel[2];
                        point.g = pixel[1];
                        point.b = pixel[0];

                        pcl::PointXYZ point_projected;
                        point_projected.x = x;
                        point_projected.y = y;
                        point_projected.z = 0;

                        ptr_projection_cloud->push_back(point_projected);
                        ptr_cloud_RGB->push_back(point);
                    }
                }


            }

        }

    this->projection_kdtree.setInputCloud(ptr_projection_cloud);
}

void arc17_computer_vision_t::project_point_cloud_GPU(void)
{
    ptr_projection_cloud->clear();
    ptr_cloud_RGB->clear();

    cv::Mat mat_T_depthsensor(4,4,CV_64FC1);

    for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                mat_T_depthsensor.at<double>(i,j) = T_depthsensor(i,j);

    project_point_cloud_kernel_GPU(ptr_cloud,
                                   ptr_cloud_RGB,
                                   ptr_projection_cloud,
                                   foscam_image,
                                   foscam_cam_matrix,
                                   mat_T_depthsensor);

    this->projection_kdtree.setInputCloud(ptr_projection_cloud);
}

void* arc17_computer_vision_t::project_point_cloud_on_image(void* args)
{

    cudaSetDevice(GPU_ID_POINT_PROJECTION);
    while(1)
    {
        usleep(100);

        if(this->flag_project_points_on_image)
        {
            //            clockid_t t1,t2;
            //            t1=clock();

            if(this->point_projection_mode == arc17_computer_vision_t::CPU)
                this->project_point_cloud();
            else if(this->point_projection_mode == arc17_computer_vision_t::GPU)
                this->project_point_cloud_GPU();


            //            t2=clock();
            //            std::cout << "Time Taken = " << (t2-t1)/1000000.0 << "\n";

            flag_project_points_on_image = false;
        }
    }
}

