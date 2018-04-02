#include <iostream>


#include<stdlib.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>


#include<pcl/io/pcd_io.h>
#include<pcl/filters/statistical_outlier_removal.h>
#include<pcl/common/centroid.h>
#include<pcl/features/normal_3d.h>
#include<pcl/common/pca.h>

#include<arc17_computer_vision/arc17_computer_vision.h>
//#include<arc17_computer_vision/object_information.h>
#include<iitktcs_msgs_srvs/objects_info.h>

#include<CImg.h>

void arc17_computer_vision_t::init(void)
{
    ROS_INFO("IN INIT");


    if(TASK == "PICK")
    {
        detection_roi_picking = new cv::Rect[N_BINS];
        detection_roi_mask_picking = new cv::Mat[N_BINS];
    }
    point_projection_mode = GPU;

    flag_update_point_cloud = false;
    flag_update_foscam_image = false;
    flag_do_forward_pass = false;
    flag_project_points_on_image = false;
    flag_continue_spinning_pcl_visualizer = true;
    flag_update_pcl_visualizer = false;
    flag_update_images = false;

    ptr_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ptr_cloud_RGB.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    ptr_projection_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    ROS_INFO("READING CALIBRATION PARAMS");
    //----read_camera_params
    read_calibration_parameteres();

    ROS_INFO("READING MASKS");
    //----read detection_mask
    read_detection_roi_masks();

    ROS_INFO("READING DETECTION ROI");


    //----SUPRRESS CAFFE LOGS
    putenv("GLOG_minloglevel=2");
    //----Initialize caffe
    read_network();

    std::cout << " LAUNCHING ALL THREADS\n";

    pthread_create(&forward_pass_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::forward_pass),this);
    pthread_create(&project_point_cloud_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::project_point_cloud_on_image),this);
    pthread_create(&foscam_image_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::callback_update_foscam_image),this);

#ifdef COMPILE_ENSENSO
    if(DEPTH_SENSOR == "ENSENSO")
    {
        //----Opening ENSENSO camera
        open_ensenso_camera();
        pthread_create(&ensenso_cloud_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::callback_fetch_ensenso_cloud),this);
    }
#endif


    if(VISUALIZE_IMAGE)
    {
        pthread_create(&update_images_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::update_images),this);
    }

    if(VISUALIZE_PCL)
    {
        pthread_create(&spin_pcl_visualizer_thread,NULL,reinterpret_cast<void* (*)(void*)>(&arc17_computer_vision_t::spin_pcl_visualizer),this);
    }

    std::cout << "*****-----COMPETITION SET-----*****\n";
    std::cout << "OBJECT ID       ACTUAL ID      OBJECT NAME\n";

    for(int i=0;i<object_names.size();i++)
        std::cout << i+1 << "  "  << map_object_set_to_competition_set[i+1] <<  "  " << object_names[i] << "\n";

    std::cout << TASK << " VISION SYSTEM IS READY TO ACCEPT CLIENT CALLS\n";

}

#ifdef COMPILE_ENSENSO

void arc17_computer_vision_t::open_ensenso_camera(void)
{
    try
    {
        std::cout << "Initializing ENSENSO SDK..."<< std::endl;

        nxLibInitialize(true);
        nxLibOpenTcpPort(24001);

        NxLibItem root;
        camera = root[itmCameras][itmBySerialNo][ENSENSO_ID];

        std::cout << "Opening ENSENSO Camera ..."<< std::endl;
        NxLibCommand open(cmdOpen);
        open.parameters()[itmCameras] = ENSENSO_ID;
        open.execute();
    }
    catch(...)
    {
        std::cout << "NXLIB ERROR IN OPENING THE CAMERA\n";
    }

}

void arc17_computer_vision_t::close_ensenso_camera(void)
{
    try
    {
        NxLibCommand close(cmdClose);
        close.parameters()[itmCameras] = ENSENSO_ID;
        close.execute();
    }
    catch(...)
    {
        std::cout << "NXLIB ERROR IN CLOSING THE CAMERA\n";
    }
}

void* arc17_computer_vision_t::callback_fetch_ensenso_cloud(void *args)
{
    while(1)
    {
        usleep(10000);
        try {
            if(flag_update_point_cloud)
            {
                // Capturing Images
                NxLibCommand (cmdCapture).execute ();

                // Stereo matching task
                NxLibCommand (cmdComputeDisparityMap).execute ();

                // Convert disparity map into XYZ data for each pixel
                NxLibCommand (cmdComputePointMap).execute ();

                // Get info about the computed point map and copy it into a std::vector
                double timestamp;
                std::vector<float> pointMap;
                int width, height;

                int channels;
                int bpe;
                bool isfloat;
                int err_code;

                //            camera[itmImages][itmRaw][itmLeft].getBinaryDataInfo (&err_code, &width,&height,&channels,&bpe,&isfloat,0);  // Get raw image timestamp
                //            camera[itmImages][itmPointMap].getBinaryData (pointMap,0);

                camera[itmImages][itmRaw][itmLeft].getBinaryDataInfo(0, 0, 0, 0, 0, &timestamp);  // Get raw image timestamp
                camera[itmImages][itmPointMap].getBinaryDataInfo(&width, &height, 0, 0, 0, 0);
                camera[itmImages][itmPointMap].getBinaryData(pointMap, 0);

                // Copy point cloud and convert in meters

                std::cout << "SIZE = " << width << "  " << height << "  "  << pointMap.size () << "\n";

                ptr_cloud->resize(height * width);
                ptr_cloud->width = width;
                ptr_cloud->height = height;
                ptr_cloud->is_dense = false;

                for (int i = 0; i < pointMap.size(); i += 3)
                {
                    ptr_cloud->at(i / 3).x = pointMap[i] / 1000.0;
                    ptr_cloud->at(i / 3).y = pointMap[i + 1] / 1000.0;
                    ptr_cloud->at(i / 3).z = pointMap[i + 2] / 1000.0;
                }
                flag_update_point_cloud = false;
            }
        }
        catch (...) {
            this->close_ensenso_camera();
            std::cout<<"RANDOM ENSENSO NXLIB EXCEPTION"<<std::endl;
            std::cout<<"waiting for 5 seconds to retry"<<std::endl;
            usleep(5000000);
            this->open_ensenso_camera();
        }
    }

}

#endif


void* arc17_computer_vision_t::callback_update_foscam_image(void* args)
{
    while(1)
    {
        usleep(1000);

        if(this->flag_update_foscam_image)
        {
            fetch_foscam_image();
            flag_update_foscam_image = false;
        }
    }
}


void arc17_computer_vision_t::callback_kinect_point_cloud_subscriber(
        const sensor_msgs::PointCloud2ConstPtr& ptr_cloud)
{

    if(flag_update_point_cloud)
    {
        pcl::fromROSMsg(*ptr_cloud,*(this->ptr_cloud));
        flag_update_point_cloud = false;
    }
}

void* arc17_computer_vision_t::spin_pcl_visualizer(void* args)
{
    pcl_visualizer = new pcl::visualization::PCLVisualizer("PCL VISUALIZER");
    pcl_visualizer->addCoordinateSystem(0.5,0,0,0);

    while(1)
    {
        usleep(1000);
        //        flag_update_ensenso_point_cloud = true;
        //        while(flag_update_ensenso_point_cloud)
        //            usleep(1000);

        if(flag_continue_spinning_pcl_visualizer)
        {
            pcl_visualizer->spinOnce();
            flag_update_pcl_visualizer = false;
        }
        else
            flag_update_pcl_visualizer = true;
    }
}

void* arc17_computer_vision_t::update_images(void* args)
{
    cv::Mat input_image = cv::Mat::zeros(detection_roi_to_display.height,detection_roi_to_display.width,CV_8UC3);
    cv::Mat colorized_probability_image = cv::Mat::zeros(detection_roi_to_display.height,detection_roi_to_display.width,CV_8UC3);

    int i =0;
    while(1)
    {
        usleep(100000);

        if(flag_update_images)
        {
            this->foscam_image(detection_roi_to_display).copyTo(input_image);
            this->foscam_color_labels(detection_roi_to_display).copyTo(colorized_probability_image);

            flag_update_images = false;

            std::stringstream img_name;
            img_name << "/home/isl-server/ashish/trial_run_images/img_" << i <<".png";
            std::stringstream seg_name;
            seg_name << "/home/isl-server/ashish/trial_run_images/seg_" << i++ <<".png";

            cv::imwrite(img_name.str(), input_image);
            cv::imwrite(seg_name.str(), colorized_probability_image);
        }
        else
        {
            cv::imshow("Input Image", input_image);
            cv::imshow("Colorizzed Prob Image", colorized_probability_image);
            cv::waitKey(1);
        }
    }
}


void arc17_computer_vision_t::update_image_and_point_clouds(void)
{
    flag_update_foscam_image = true;
    while(flag_update_foscam_image)
        usleep(10000);

    flag_update_point_cloud = true;
    while(flag_update_point_cloud)
        usleep(10000);

    flag_project_points_on_image = true;
    while(flag_project_points_on_image)
        usleep(10000);
}



void arc17_computer_vision_t::draw_rectangles(cv::Rect roi)
{
    ROS_INFO("IN DRAW RECTANGLES");

    cv::Mat detections_image = foscam_color_labels(roi);

    std::vector<cv::Rect> detections;
    std::vector<int> detections_label;

    for(int i=0;i<object_rects.size();i++)
    {
        if(object_rects[i].size())
        {
            cv::Rect object_rect = object_rects[i][0];
            for(int j=1;j<object_rects[i].size();j++)
                object_rect |= object_rects[i][j];

            detections.push_back(object_rect);
            detections_label.push_back(i+1);
            cv::rectangle(detections_image,object_rect, cv::Scalar(0,255,0),2);
        }
    }

    cimg_library::CImg<unsigned char> drawing_image(detections_image.cols,detections_image.rows,1,3);

    for(int i=0;i<detections_image.rows;i++)
        for(int j=0;j<detections_image.cols;j++)
        {
            unsigned char* pixel = detections_image.data + i* detections_image.step[0] + j* detections_image.step[1];
            unsigned char* pixel_draw = (unsigned char*)(drawing_image.data()+i*detections_image.cols+j);
            pixel_draw[0] = pixel[2];
            (pixel_draw+ detections_image.rows*detections_image.cols)[0] = pixel[1];
            (pixel_draw+ detections_image.rows*detections_image.cols*2)[0] = pixel[0];

        }

    for(int i=0;i<detections.size();i++)
    {
        unsigned char foreground[]={0,0,0};
        unsigned char background[]={255,0,0};

        int text_height = 20;


        std::stringstream text;
        text << object_names[detections_label[i]-1].c_str();

        drawing_image.draw_text(detections[i].x,
                                detections[i].y,
                                text.str().c_str(),
                                foreground,
                                background,
                                0.5,
                                text_height);
    }

    for(int i=0;i<detections_image.rows;i++)
        for(int j=0;j<detections_image.cols;j++)
        {
            unsigned char* pixel = detections_image.data + i* detections_image.step[0] + j* detections_image.step[1];
            unsigned char* pixel_draw = (unsigned char*)(drawing_image.data()+i*detections_image.cols+j);
            pixel[2] =  pixel_draw[0];
            pixel[1] = (pixel_draw+ detections_image.rows*detections_image.cols)[0];
            pixel[0] = (pixel_draw+ detections_image.rows*detections_image.cols*2)[0];

        }
    ROS_INFO("OUT DRAW RECTANGLES");
}

void arc17_computer_vision_t::get_rectangles(cv::Rect decision_roi,unsigned char* availability)
{
    ROS_INFO("IN GET RECTANGLES");

    cv::Size erosion_size(3,3);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, erosion_size);

    for(int i=0;i<NUM_OBJECTS;i++)
    {
        object_rects[i].clear();
        object_rotated_rects[i].clear();

        if(availability[i])
        {
            cv::Mat object_mask = object_masks[i](decision_roi);

            cv::erode(object_mask,object_mask,element, cv::Point(-1,-1), 3);
            cv::dilate(object_mask,object_mask,element, cv::Point(-1,-1), 3 );

            cv::Mat gray_mask;
            object_masks[i](decision_roi).copyTo(gray_mask);

            std::vector<std::vector<cv::Point2i> > contours;
            cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

            /*  if(contours.size() > 1)
            {
                cv::Mat patch_mask;
                object_masks[i](decision_roi).copyTo(patch_mask);

                std::vector<int> contours_area(contours.size());
                std::vector<cv::Rect> contours_rect(contours.size());

                int max_area = 0;

                for(int j=0;j<contours.size();j++)
                {
                    cv::Rect contour_rect = cv::boundingRect(contours[j]);

                    int contour_area = cv::countNonZero(patch_mask(contour_rect));
                    contours_area.push_back(contour_area);
                    contours_rect.push_back(contour_rect);

                    if(max_area < contour_area)
                        max_area = contour_area;
                }

                if(max_area > 0)
                {
                    for(int j=0;j<contours.size();j++)
                    {
                        int contour_area = contours_area[j];
                        float ratio = contour_area / static_cast<float>(max_area);

                        if(ratio > 0.30f)
                            object_rects[i].push_back(contours_rect[j]);
                    }
                }

            }
            else*/

            if(contours.size())
            {
                for(int j=0;j<contours.size();j++)
                {
                    object_rects[i].push_back(cv::boundingRect(contours[j]));

                    std::vector<cv::Point2i> points;
                    for(int k=0;k<contours[j].size();k++)
                        points.push_back(contours[j][k]);

                    cv::RotatedRect rotated_bounding_rect =  cv::minAreaRect(points);
                    object_rotated_rects[i].push_back(rotated_bounding_rect);
                }
            }
        }
        else
            object_masks[i](decision_roi) = cv::Mat::zeros(decision_roi.height,decision_roi.width,CV_8UC1);
    }
    ROS_INFO("OUT GET RECTANGLES");
}
