#include<Eigen/Dense>

#include<lcm/lcm-cpp.hpp>

#include<iostream>
#include<gps_t.hpp>
#include<image_t.hpp>
#include<points3d_t.hpp>
#include<pose_t.hpp>

#include<opencv2/opencv.hpp>

#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/visualization/cloud_viewer.h>


int n_frames = 267;
std::string drive_id = "2011_09_26_drive_0095_sync";
std::string str_drive_log = "/home/isl-server/ashish/KITTI/2011_09_26/" + drive_id + "/";
std::string str_calib_cam_T_cam = str_drive_log + "../2011_09_26_calib/calib_cam_to_cam.txt";
std::string str_calib_imu_T_vel = str_drive_log + "../2011_09_26_calib/calib_imu_to_velo.txt";
std::string str_calib_vel_T_cam = str_drive_log + "../2011_09_26_calib/calib_velo_to_cam.txt";

Eigen::Affine3f imu_To_vel;  // vehicle to velodyne transform
Eigen::Affine3f vel_To_imu;  // velodyne to vehicle transform
Eigen::Affine3f vel_To_cam;  // camera to velodyne transform


typedef struct _frame_t_
{
    int id;
    cv::Mat im_left_gray;
    cv::Mat im_right_gray;
    cv::Mat im_left_color; //image_02
    cv::Mat im_right_color;

    std::vector<cv::Mat> images;
    int n_points;
    std::vector<Eigen::Vector4f> xyz;
    std::vector<unsigned char> intensity;

    Eigen::Affine3f pose;
    Eigen::Quaternionf orientation;
}frame_t;

#define NUM_LB3_CAMS 6

typedef struct _lb3_camera_info_t_
{
    Eigen::Affine3f imu_To_head;
    std::vector<Eigen::Affine3f> head_To_cams;
    std::vector<Eigen::Affine3f> K;
}lb3_camera_info_t;

bool get_frame(lcm::LogFile& log_file, std::string str_pose, frame_t& frame)
{

    const lcm::LogEvent* logevent = log_file.readNextEvent();

    if(!logevent)
        return false;

    if(logevent->channel == "GPS")
    {
        gps_t gps;
        gps.decode(logevent->data,0,logevent->datalen);

    }

    logevent = log_file.readNextEvent();
    if(logevent->channel == str_pose)
    {
        pose_t pose;
        pose.decode(logevent->data,0,logevent->datalen);

        Eigen::Quaternionf q(pose.orientation[0],pose.orientation[1],pose.orientation[2],pose.orientation[3]);

        frame.pose = Eigen::Affine3f::Identity();

        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                frame.pose(i,j) = q.matrix()(i,j);

        frame.pose(0,3) = pose.pos[0];
        frame.pose(1,3) = pose.pos[1];
        frame.pose(2,3) = pose.pos[2];
    }

    logevent = log_file.readNextEvent();
    if(logevent->channel == "VELODYNE")
    {
        points3d_t points;
        points.decode(logevent->data,0,logevent->datalen);

        int n_points = points.datalen/3;
        for(int i =0;i<n_points;i++)
        {
            Eigen::Vector4f point;
            point(0) = points.data[i*3+0];
            point(1) = points.data[i*3+1];
            point(2) = points.data[i*3+2];
            point(3) = 1.0f;

            frame.xyz.push_back(point);
        }

        frame.n_points = frame.xyz.size();

    }

    logevent = log_file.readNextEvent();
    if(logevent->channel == "IMAGE_02")
    {
        image_t image;
        image.decode(logevent->data,0,logevent->datalen);

        frame.im_left_color =cv::Mat(image.height,image.width,CV_8UC3);
        memcpy(frame.im_left_color.data,image.data.data(),image.height*image.width*sizeof(unsigned char)*3);

    }

    frame.id = logevent->eventnum / 4;
    //    std::cout << "POSE = " << frame.pose.matrix() << "\n";
    //    cv::imshow("image",frame.im_left_color);
    //    cv::waitKey(1);

    return true;
}



bool get_ford_frame(lcm::LogFile& log_file, std::string str_pose, frame_t& frame)
{

    const lcm::LogEvent* logevent = log_file.readNextEvent();

    if(!logevent)
        return false;

    if(logevent->channel == "GPS")
    {
        gps_t gps;
        gps.decode(logevent->data,0,logevent->datalen);

    }

    logevent = log_file.readNextEvent();
    if(logevent->channel == str_pose)
    {
        pose_t pose;
        pose.decode(logevent->data,0,logevent->datalen);

        Eigen::Quaternionf q(pose.orientation[0],pose.orientation[1],pose.orientation[2],pose.orientation[3]);

        frame.pose = Eigen::Affine3f::Identity();

        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                frame.pose(i,j) = q.matrix()(i,j);

        frame.pose(0,3) = pose.pos[0];
        frame.pose(1,3) = pose.pos[1];
        frame.pose(2,3) = pose.pos[2];
    }

    logevent = log_file.readNextEvent();
    if(logevent->channel == "VELODYNE")
    {
        points3d_t points;
        points.decode(logevent->data,0,logevent->datalen);

        int n_points = points.datalen/3;
        for(int i =0;i<n_points;i++)
        {
            Eigen::Vector4f point;
            point(0) = points.data[i*3+0];
            point(1) = points.data[i*3+1];
            point(2) = points.data[i*3+2];
            point(3) = 1.0f;

            frame.xyz.push_back(point);
        }

        frame.n_points = frame.xyz.size();

    }

    frame.images.resize(NUM_LB3_CAMS);

    for(int i=NUM_LB3_CAMS-1;i >= 0;i--)
    {
        logevent = log_file.readNextEvent();

        std::stringstream image_channel;
        image_channel << "IMAGE_0" << i;

        if(logevent->channel == image_channel.str())
        {
            image_t image;
            image.decode(logevent->data,0,logevent->datalen);

            frame.images[i] = cv::Mat(image.height,image.width,CV_8UC3);
            memcpy(frame.images[i].data,image.data.data(),image.height*image.width*sizeof(unsigned char)*3);
        }
    }

    frame.id = logevent->eventnum / 9;
    //    std::cout << "POSE = " << frame.pose.matrix() << "\n";
    //    cv::imshow("image",frame.im_left_color);
    //    cv::waitKey(1);

    return true;
}

//camera_info_t get_calibration_data(camera_info_t& kitti_camera)
//{
//    //READ IMU_To_Vel matrix
//    {
//        FILE* file_imu_T_vel = fopen(str_calib_imu_T_vel.c_str(),"r");
//        char text[100];

//        fscanf(file_imu_T_vel,"%s%s%s",text,text,text);
//        fscanf(file_imu_T_vel,"%s",text);

//        for(int i=0;i<3;i++)
//            for(int j=0;j<3;j++)
//                fscanf(file_imu_T_vel,"%f",&imu_To_vel(i,j));


//        fscanf(file_imu_T_vel,"%s",text);

//        for(int i=0;i<3;i++)
//            fscanf(file_imu_T_vel,"%f",&imu_To_vel(i,3));

//        imu_To_vel(3,0) = 0;
//        imu_To_vel(3,1) = 0;
//        imu_To_vel(3,2) = 0;
//        imu_To_vel(3,3) = 1;

//        fclose(file_imu_T_vel);

//        vel_To_imu = imu_To_vel.inverse();
//    }

//    // READ VEL_To_Cam
//    {
//        FILE* file_vel_T_cam = fopen(str_calib_vel_T_cam.c_str(),"r");
//        char text[100];

//        fscanf(file_vel_T_cam,"%s%s%s",text,text,text);
//        fscanf(file_vel_T_cam,"%s",text);

//        for(int i=0;i<3;i++)
//            for(int j=0;j<3;j++)
//                fscanf(file_vel_T_cam,"%f",&vel_To_cam(i,j));


//        fscanf(file_vel_T_cam,"%s",text);

//        for(int i=0;i<3;i++)
//            fscanf(file_vel_T_cam,"%f",&vel_To_cam(i,3));

//        vel_To_cam(3,0) = 0;
//        vel_To_cam(3,1) = 0;
//        vel_To_cam(3,2) = 0;
//        vel_To_cam(3,3) = 1;

//        fclose(file_vel_T_cam);

//    }

//    // READ CAMERA INTRINSICS
//    {

//        kitti_camera.K << 7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
//                0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
//                0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03,
//                0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00;


//        Eigen::Affine3f R_vel_to_cam0;
//        R_vel_to_cam0.matrix() << 9.999239e-01, 9.837760e-03, -7.445048e-03, 0.000000e+00,
//                -9.869795e-03, 9.999421e-01, -4.278459e-03, 0.000000e+00,
//                7.402527e-03, 4.351614e-03,  9.999631e-01, 0.000000e+00,
//                0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00;

//        kitti_camera.channel = "IMAGE_02";

//        kitti_camera.Vel_To_Cam = R_vel_to_cam0 * vel_To_cam;
//        kitti_camera.B_To_Cam = kitti_camera.Vel_To_Cam * imu_To_vel;

//        return kitti_camera;
//    }
//}


void get_ford_calibration(lb3_camera_info_t& lb3_camera_info)
{
    float fx;
    float fy;
    float cx;
    float cy;

    float tx;
    float ty;
    float tz;

    float rx;
    float ry;
    float rz;

    Eigen::Matrix3f Rx;
    Eigen::Matrix3f Ry;
    Eigen::Matrix3f Rz;
    Eigen::Matrix3f R;

    lb3_camera_info.head_To_cams.resize(NUM_LB3_CAMS);
    lb3_camera_info.K.resize(NUM_LB3_CAMS);

    char text[10];

    FILE* file_ford_calib = fopen("/home/isl-server/ashish/ford/lb3_params.txt","r");


    // FIL BASE TO HEAD
    {
        fscanf(file_ford_calib, "%s",text);
        fscanf(file_ford_calib,"%f %f %f %f %f %f",&tx, &ty, &tz, &rx, &ry, &rz);

        rx *= M_PI / 180.0f;
        ry *= M_PI / 180.0f;
        rz *= M_PI / 180.0f;

        Rx << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx);
        Ry << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry);
        Rz << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1;
        R  =  Rz * Ry * Rx ;

        lb3_camera_info.imu_To_head = Eigen::Affine3f::Identity();

        for(int i=0;i < 3;i++)
            for(int j=0;j < 3;j++)
                lb3_camera_info.imu_To_head(i,j) = R(i,j);

        lb3_camera_info.imu_To_head(0,3) = tx;
        lb3_camera_info.imu_To_head(1,3) = ty;
        lb3_camera_info.imu_To_head(2,3) = tz;
    }

    // FIL CAMERA INTRINSICS AND EXTRINSICS
    {

        int fx_multiplier = 1;
        int fy_multiplier = 1;

        int cx_multiplier = 1;
        int cy_multiplier = 2;

        for(int i=0; i < NUM_LB3_CAMS; i++)
        {
            fscanf(file_ford_calib, "%s",text);
            fscanf(file_ford_calib,"%f %f %f %f %f %f",&tx, &ty, &tz, &rx, &ry, &rz);

            rx *= M_PI / 180.0f;
            ry *= M_PI / 180.0f;
            rz *= M_PI / 180.0f;

            Rx << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx);
            Ry << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry);
            Rz << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1;
            R  =  Rz * Ry * Rx ;

            lb3_camera_info.head_To_cams[i] = Eigen::Affine3f::Identity();

            for(int j=0;j < 3;j++)
                for(int k=0; k < 3;k++)
                    lb3_camera_info.head_To_cams[i](j,k) = R(j,k);

            lb3_camera_info.head_To_cams[i](0,3) = tx;
            lb3_camera_info.head_To_cams[i](1,3) = ty;
            lb3_camera_info.head_To_cams[i](2,3) = tz;

            fscanf(file_ford_calib, "%s",text);
            fscanf(file_ford_calib,"%f %f %f",&fx, &cx, &cy);

            fy = fx;

            lb3_camera_info.K[i] = Eigen::Affine3f::Identity();

            lb3_camera_info.K[i](0,0) = fx * fx_multiplier;
            lb3_camera_info.K[i](1,1) = fy * fy_multiplier;
            lb3_camera_info.K[i](0,2) = cx * cx_multiplier;
            lb3_camera_info.K[i](1,2) = cy * cy_multiplier;  // BCZ FORD HAS CALIBRATED THE CAMERA WITH HALF Y and DONT KNOW ABOUT X;
        }
    }

    fclose(file_ford_calib);
}


int main(void)
{
    pcl::visualization::CloudViewer cloud_ciewer("CLOUD");


    lb3_camera_info_t lb3_camera_info;
    get_ford_calibration(lb3_camera_info);

//    std::cout << lb3_camera_info.imu_To_head.matrix() << "\n\n";

//    for(int i=0;i<NUM_LB3_CAMS;i++)
//    {
//        std::cout << lb3_camera_info.head_To_cams[i].matrix() << "\n\n";
//        std::cout << lb3_camera_info.K[i].matrix() << "\n\n";
//    }

//    return 0;

    lcm::LogFile logfile("/home/isl-server/ashish/KITTI/ford_map_logs/north_campus.lcmlog","r");

    bool done = true;

    while(done)
    {
        frame_t frame;
        //        done = get_frame(logfile,"POSE",frame);
        done = get_ford_frame(logfile,"POSE",frame);

        if(!done)
            break;

        pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for(int i=0;i<frame.n_points;i++)
        {
            pcl::PointXYZ point;

            point.x = frame.xyz[i](0);
            point.y = frame.xyz[i](1);
            point.z = frame.xyz[i](2);

            ptr_cloud->push_back(point);
        }

        cloud_ciewer.showCloud(ptr_cloud);


        //        for(int i=0;i<NUM_LB3_CAMS;i++)
        //        {
        //            std::stringstream image_channel;
        //            image_channel << "IMAGE_0" << i;
        //            cv::imshow(image_channel.str(),frame.images[i]);
        //        }
        //        cv::waitKey(1);

        std::cout << "FRAME VISUALIZED = " << frame.id << "\n";
    }

    while(!cloud_ciewer.wasStopped());
    return 0;
}




























