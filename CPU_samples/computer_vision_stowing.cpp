#include <iostream>

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
#include<pcl/common/norms.h>
#include<pcl/features/normal_3d.h>
#include<pcl/common/pca.h>

#include<arc17_computer_vision/arc17_computer_vision.h>
#include<arc17_computer_vision/decision_making.h>
#include<arc17_computer_vision/object_information.h>
#include<arc17_computer_vision/occlusion_detection.hpp>

#include<iitktcs_msgs_srvs/objects_info.h>



bool arc17_computer_vision_t::callback_service_computer_vision(
        iitktcs_msgs_srvs::computer_vision::Request& req,
        iitktcs_msgs_srvs::computer_vision::Response& res )
{
    std::cout << "IN COMPUTER VISION\n";

    unsigned char availability[NUM_OBJECTS];
    memset(availability,255,sizeof(unsigned char)*NUM_OBJECTS);

    //----IMAGE PROCESSING

    //    update_image_and_point_clouds();

    flag_update_foscam_image = true;
    while(flag_update_foscam_image)
        usleep(10000);

    flag_do_forward_pass = true;
    while(flag_do_forward_pass)
        usleep(10000);

    merge_decisions();
    get_rectangles(detection_roi_stowing,availability);
    draw_rectangles(detection_roi_stowing);

    detection_roi_to_display = detection_roi_stowing;
    flag_update_images = true;
    while(!flag_update_images)
        usleep(1000);


    //    //------DONE

    //    int object_id = 0;

    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_object_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    //    decision_making_based_on_largest_point_cloud_stowing(availability,
    //                                                         ptr_object_cloud_rgb,
    //                                                         object_id);

    //    if(object_id)
    //    {
    //        if(ptr_object_cloud_rgb->size())
    //        {
    //            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_removal;
    //            statistical_removal.setMeanK(10);
    //            statistical_removal.setStddevMulThresh(2.0);
    //            statistical_removal.setInputCloud(ptr_object_cloud_rgb);
    //            statistical_removal.filter(*ptr_object_cloud_rgb);

    //            Eigen::Vector4f centroid;
    //            pcl::compute3DCentroid(*ptr_object_cloud_rgb,centroid);

    //            pcl::PointCloud<pcl::Normal>::Ptr ptr_normal_cloud(new pcl::PointCloud<pcl::Normal>);

    //            pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    //            normal_estimator.setRadiusSearch(0.02);
    //            normal_estimator.setInputCloud(ptr_object_cloud_rgb);
    //            normal_estimator.compute(*ptr_normal_cloud);

    //            pcl::search::KdTree<pcl::PointXYZRGB> kd_tree;
    //            kd_tree.setInputCloud(ptr_object_cloud_rgb);

    //            std::vector<int> indices;
    //            std::vector<float> distances;

    //            pcl::PointXYZRGB point_centroid;
    //            pcl::PointXYZ axis;
    //            pcl::PointXYZ normal;



    //            point_centroid.x = centroid[0];
    //            point_centroid.y = centroid[1];
    //            point_centroid.z = centroid[2];

    //            kd_tree.nearestKSearch(point_centroid,1,indices,distances);

    //            point_centroid.x = ptr_object_cloud_rgb->at(indices[0]).x;
    //            point_centroid.y = ptr_object_cloud_rgb->at(indices[0]).y;
    //            point_centroid.z = ptr_object_cloud_rgb->at(indices[0]).z;

    //            normal.x = ptr_normal_cloud->at(indices[0]).normal_x;
    //            normal.y = ptr_normal_cloud->at(indices[0]).normal_y;
    //            normal.z = ptr_normal_cloud->at(indices[0]).normal_z;

    //            pcl::PCA<pcl::PointXYZRGB> pca;
    //            pca.setInputCloud(ptr_object_cloud_rgb);
    //            Eigen::Matrix3f E = pca.getEigenVectors();
    //            axis.x = E(0,0);
    //            axis.y = E(1,0);
    //            axis.z = E(2,0);

    //            //        normal.x = E(0,2);
    //            //        normal.y = E(1,2);
    //            //        normal.z = E(2,2);

    //            //        pcl::flipNormalTowardsViewpoint(point_centroid,0.0f,0.0f,0.0f,normal.x,normal.y,normal.z);

    //            //        normal.x /= sqrt(normal.x*normal.x + normal.y *normal.y + normal.z*normal.z);
    //            //        normal.y /= sqrt(normal.x*normal.x + normal.y *normal.y + normal.z*normal.z);
    //            //        normal.z /= sqrt(normal.x*normal.x + normal.y *normal.y + normal.z*normal.z);

    //            /*std::vector<float> axis_v;
    //                axis_v.push_back(axis.x);
    //                axis_v.push_back(axis.y);
    //                axis_v.push_back(axis.z);

    //                std::vector<float> origin_v;
    //                axis_v.push_back(0.f);
    //                axis_v.push_back(0.f);
    //                axis_v.push_back(0.f);

    //                std::cout<<"Length of axis"<<pcl::L2_Norm(origin_v, axis_v, 3);*/

    //            // For mohit's testing
    //            res.object_info.data.push_back((float) object_id);


    //            res.object_info.data.push_back(point_centroid.x);
    //            res.object_info.data.push_back(point_centroid.y);
    //            res.object_info.data.push_back(point_centroid.z);

    //            res.object_info.data.push_back(normal.x);
    //            res.object_info.data.push_back(normal.y);
    //            res.object_info.data.push_back(normal.z);

    //            res.object_info.data.push_back(axis.x);
    //            res.object_info.data.push_back(axis.y);
    //            res.object_info.data.push_back(axis.z);

    //            std::cout << point_centroid.x << " " << point_centroid.y << " " << point_centroid.z << std::endl;
    //            std::cout << normal.x << " " << normal.y << " " << normal.z << std::endl;
    //            std::cout << axis.x << " " << axis.y << " " << axis.z << std::endl;


    //            pcl::PointXYZRGB end_point_normal;
    //            pcl::PointXYZRGB end_point_axis;

    //            end_point_normal.x = normal.x + point_centroid.x;
    //            end_point_normal.y = normal.y + point_centroid.y;
    //            end_point_normal.z = normal.z + point_centroid.z;

    //            end_point_axis.x = axis.x + point_centroid.x;
    //            end_point_axis.y = axis.y + point_centroid.y;
    //            end_point_axis.z = axis.z + point_centroid.z;

    //            std::cout << "OBJECT NAME = " << object_names[object_id-1] << "\n";

    //            if(VISUALIZE_IMAGE)
    //            {
    //                draw_rectangles(detection_roi_stowing);

    //                detection_roi_to_display = detection_roi_stowing;
    //                flag_update_images = true;
    //                while(!flag_update_images)
    //                    usleep(1000);
    //            }

    //            if(VISUALIZE_PCL)
    //            {
    //                flag_continue_spinning_pcl_visualizer = false;

    //                while(!flag_update_pcl_visualizer)
    //                    usleep(1000);

    //                pcl_visualizer->removeAllPointClouds();
    //                pcl_visualizer->removeAllShapes();

    //                pcl_visualizer->addPointCloud(ptr_object_cloud_rgb,"OBJECT CLOUD");

    //                pcl_visualizer->addArrow(point_centroid, end_point_normal, 0, 1.0, 0, true, "normal");
    //                pcl_visualizer->addArrow(point_centroid, end_point_axis, 0, 0, 1.0, true, "axis");

    //                flag_continue_spinning_pcl_visualizer = true;
    //                while(flag_update_pcl_visualizer)
    //                    usleep(1000);
    //            }
    //        }
    //    }
    //    return true;

}

bool arc17_computer_vision_t::callback_service_computer_vision_stowing(
        iitktcs_msgs_srvs::computer_vision_stowing::Request &req,
        iitktcs_msgs_srvs::computer_vision_stowing::Response &res)

{
    std::cout << "IN COMPUTER VISION STOWING\n";

    if(req.task.data == "STOW")
    {

        unsigned char availability[NUM_OBJECTS];
        memset(availability,0,sizeof(unsigned char)*NUM_OBJECTS);

        std::vector<int> mapped_availability;

        for(int i=0;i<req.ids_available.data.size();i++)
        {
            mapped_availability.push_back(map_competition_set_to_object_set[req.ids_available.data[i]]);
            availability[mapped_availability[i]-1] = 1;

            std::cout << "AVAILABILIY = " << map_competition_set_to_object_set[req.ids_available.data[i]] << " " << mapped_availability[i] << "\n";
        }

        std::cout << "TASK = " << req.task.data << "\n";

        std::cout << "\n-----AVAILABLE OBJECTS-----\n";
        for(int i=0;i<req.ids_available.data.size();i++)
            std::cout << req.ids_available.data[i] << "  " << object_names[mapped_availability[i]-1] << "\n";


        //----IMAGE PROCESSING

        update_image_and_point_clouds();

        flag_do_forward_pass = true;
        while(flag_do_forward_pass)
            usleep(100);

        merge_decisions();
        get_rectangles(detection_roi_stowing,availability);

        //------DONE


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_object_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

        int object_id = 0;


//                decision_making_based_on_largest_point_cloud_stowing(availability,
//                                                                     ptr_object_cloud_rgb,
//                                                                     object_id);


        //-----DECISION MAKING WITH OCCLUSION

        //        {
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ptr_objects_cloud_rgb;

        //            for(int i=0;i<NUM_OBJECTS;i++)
        //                ptr_objects_cloud_rgb.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));

        ////            decision_making_based_on_largest_point_cloud_stowing(availability,
        ////                                                                 ptr_objects_cloud_rgb,
        ////                                                                 mapped_availability,
        ////                                                                 detection_roi_stowing
        ////                                                                 );


        //            //-----DETECT OCCLUSION
        //            std::vector<std::vector<int> > occluding_obj_id;

        //            detect_occlusion(availability,
        //                             detection_roi_stowing,
        //                             mapped_availability,
        //                             occluding_obj_id
        //                             );

        //            int n_targets = mapped_availability.size();

        //            for(int i=0;i<n_targets;i++)
        //            {
        //                if(occluding_obj_id[i].size())
        //                {
        //                    std::cout << "TARGET " << object_names[mapped_availability[i]-1] << "  IS OCCLUDED BY \n";

        //                    for(int j=0;j<occluding_obj_id[i].size();j++)
        //                        std::cout << object_names[occluding_obj_id[i][j]-1] << "\n";
        //                }
        //                else
        //                    std::cout << "TARGET " << object_names[mapped_availability[i]-1] << "  IS FULLY VISIBLE \n";

        //            }


        //            //-----SEPARATE OCCLUDED AND UNOCCLUDED OBJECTS

        //            std::vector<std::vector<int> > unoccluded_object_ids_set;

        //            for(int i=0;i<n_targets;i++)
        //            {
        //                if(!occluding_obj_id[i].size())
        //                {
        //                    std::vector<int> object_set;
        //                    object_set.push_back(mapped_availability[i]);

        //                    unoccluded_object_ids_set.push_back(object_set);
        //                }
        //            }



        //            float min  = 10000000.0f;
        //            int index = 0;

        //            for(int i=0;i < unoccluded_object_ids_set.size() ;i++)
        //            {
        //                int obj_id = unoccluded_object_ids_set[i][0];
        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ptr_object_cloud = ptr_objects_cloud_rgb[obj_id-1];

        //                Eigen::Vector4d centroid;

        //                pcl::compute3DCentroid(*ptr_object_cloud,centroid);

        //                if(centroid(2) < min)
        //                {
        //                    min = centroid(2);
        //                    index = obj_id;
        //                }

        //            }

        //            object_id = index;

        //            if(object_id)
        //                pcl::copyPointCloud(*ptr_objects_cloud_rgb[object_id-1],*ptr_object_cloud_rgb);

        //        }


        //-----DECISION MAKING WITH MIN Z
        //        {
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ptr_objects_cloud_rgb;

        //            for(int i=0;i<NUM_OBJECTS;i++)
        //                ptr_objects_cloud_rgb.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));

        //            decision_making_based_on_largest_point_cloud_stowing(availability,
        //                                                                 ptr_objects_cloud_rgb,
        //                                                                 mapped_availability,
        //                                                                 detection_roi_stowing
        //                                                                 );

        //            float min_height = 100.f;
        //            int min_index = 0;

        //            for(int i=0; i < ptr_objects_cloud_rgb.size(); i++) {

        //                int obj_id = mapped_availability[i];

        //                Eigen::Vector4d cn;
        //                Eigen::Vector3f centroid;


        //                std::cout << "SIZE = " << ptr_objects_cloud_rgb[i]->size() << "\n";

        //                if(ptr_objects_cloud_rgb[i]->size())
        //                {
        //                    pcl::compute3DCentroid(*ptr_objects_cloud_rgb[i], cn);
        //                    centroid(0) = cn(0);
        //                    centroid(1) = cn(1);
        //                    centroid(2) = cn(2);

        //                    Eigen::Vector3f scene_centroid;
        //                    searchKNN(centroid, scene_centroid, ptr_objects_cloud_rgb[i]);


        //                    if(scene_centroid[2] < min_height) {

        //                        std::cout << "MIN HEIGHT\n";

        //                        min_height = scene_centroid[2];
        //                        min_index = obj_id;
        //                    }
        //                }
        //            }


        //            std::cout << "OBJECT ID IN MIN Z = " << min_index << "\n";

        //            object_id = min_index;

        //            std::cout << "COPYING CLOUD " << min_index << "\n";

        //            if(object_id)
        //                pcl::copyPointCloud(*ptr_objects_cloud_rgb[object_id-1], *ptr_object_cloud_rgb);

        //        }

//        //-----DECISION MAKING WITH LARGEST AND LESSER NO OF PATCHES
//        {
//            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ptr_objects_cloud_rgb;

//            for(int i=0;i<NUM_OBJECTS;i++)
//                ptr_objects_cloud_rgb.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));

//            decision_making_based_on_largest_point_cloud_stowing(availability,
//                                                                 ptr_objects_cloud_rgb,
//                                                                 mapped_availability,
//                                                                 detection_roi_stowing
//                                                                 );


//            if(ptr_objects_cloud_rgb[0]->size())
//            {
//                int target_id = mapped_availability[0] - 1;

//                cv::Mat target_mask;
//                object_masks[target_id](detection_roi_stowing).copyTo(target_mask);

//                cv::Mat gray_mask;
//                target_mask.copyTo(gray_mask);

//                std::vector<std::vector<cv::Point2i> > contours;
//                cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

//                for(int i=0;i<contours.size();i++)
//                    cv::fillPoly(gray_mask,contours,cv::Scalar(255));

//                int n_objects = NUM_OBJECTS;

//                for(int i=0; i < n_objects ; i++)
//                {
//                    if(availability[i])
//                    {
//                        if(i != target_id)
//                        {

//                            cv::Mat object_mask;
//                            object_masks[i](detection_roi_stowing).copyTo(object_mask);

//                            cv::Mat object_gray_mask;
//                            object_mask.copyTo(object_gray_mask);

//                            std::vector<std::vector<cv::Point2i> > object_contours;
//                            cv::findContours(object_gray_mask, object_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

//                            for(int i=0;i<contours.size();i++)
//                                cv::fillPoly(object_gray_mask,object_contours,cv::Scalar(255));

//                            cv::Mat anded_mask;
//                            cv::bitwise_and(gray_mask,object_gray_mask, anded_mask);

//                            std::vector<std::vector<cv::Point2i> > anded_contours;
//                            cv::findContours(anded_mask, anded_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

//                            if(anded_contours.size())
//                            {
//                            cv::Rect rect_intersection = cv::boundingRect(anded_contours[0]);

//                            for(int j=0; j< anded_contours.size();j++)
//                                rect_intersection |= anded_contours[j];

//                            rect_intersection = object_rects[target_id] & object_rects[i];

//                            float intersection_area = static_cast<float>(rect_intersection.area());



//                            float others_area = static_cast<float>(object_rects[i].area());

//                            float ratio = intersection_area / others_area;

//                            if(ratio > 0.4)
//                            {
//                                int white_pixel_target = cv::countNonZero(target_mask(rect_intersection));
//                                int white_pixel_others = cv::countNonZero(rect_masks[i](rect_intersection));

//                                //-----IF WHITE PIXELS OF BOTH TARGET AND OBJECT TO CHECK IS ZERO
//                                //-----IT WILL BE TAKEN CARE OF IN SENDING THE RESPONSE BY CHECKING SIZE OF
//                                //-----TARGET POINT CLOUD

//                                if(white_pixel_others > white_pixel_target )
//                                {
//                                    //                    std::cout << " object id: " << object_names[target_id]
//                                    //                                 << " is occluded by object id: " << object_names[i] << "\n";
//                                    object_id.push_back(i+1);
//                                }
//                            }
//                        }
//                        }
//                    }
//                }
//            }

//            std::cout << "OBJECT ID IN MIN Z = " << min_index << "\n";

//            object_id = min_index;

//            std::cout << "COPYING CLOUD " << min_index << "\n";

//            if(object_id)
//                pcl::copyPointCloud(*ptr_objects_cloud_rgb[object_id-1], *ptr_object_cloud_rgb);

//        }


        //-----DECISION MAKING WITH MINIMUM PATCHES
        {
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ptr_objects_cloud_rgb;

            for(int i=0;i<NUM_OBJECTS;i++)
                ptr_objects_cloud_rgb.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));

            decision_making_based_on_largest_point_cloud_stowing(availability,
                                                                 ptr_objects_cloud_rgb,
                                                                 mapped_availability,
                                                                 detection_roi_stowing
                                                                 );


            int min_n_polygons = 10000;
            int index = 0;

            for(int i=0;i<mapped_availability.size();i++)
            {
                int obj_id = mapped_availability[i] - 1;

                cv::Mat gray_mask;
                object_masks[obj_id](detection_roi_stowing).copyTo(gray_mask);

                std::vector<std::vector<cv::Point2i> > contours;
                cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

                if(contours.size())
                {
                    if(contours.size() && contours.size() < min_n_polygons)
                    {
                        min_n_polygons = contours.size();
                        index = obj_id + 1;
                    }
                }

            }

            object_id = index;

            if(object_id)
                pcl::copyPointCloud(*ptr_objects_cloud_rgb[object_id-1], *ptr_object_cloud_rgb);

        }


        if(object_id)
        {
            std::cout << "\n-----LARGEST OBJECT----- " << object_names[object_id-1] << "\n";

            res.object_info.push_back(iitktcs_msgs_srvs::objects_info());

            iitktcs_msgs_srvs::objects_info& target_object_info =  res.object_info[0];

            target_object_info.id.data = map_object_set_to_competition_set[object_id];
            pcl::toROSMsg(*ptr_object_cloud_rgb,target_object_info.roi);
        }


        if(VISUALIZE_IMAGE)
        {
            draw_rectangles(detection_roi_stowing);

            detection_roi_to_display = detection_roi_stowing;
            flag_update_images = true;
            while(!flag_update_images)
                usleep(1000);
        }

        if(VISUALIZE_PCL)
        {
            flag_continue_spinning_pcl_visualizer = false;

            while(!flag_update_pcl_visualizer)
                usleep(1000);

            pcl_visualizer->removeAllPointClouds();

            pcl_visualizer->addPointCloud(ptr_object_cloud_rgb,"OBJECT CLOUD");

            flag_continue_spinning_pcl_visualizer = true;
            while(flag_update_pcl_visualizer)
                usleep(1000);
        }

        return true;
    }
    else
        return true;


}
