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
#include<pcl/features/normal_3d.h>
#include<pcl/common/pca.h>



#include<arc17_computer_vision/arc17_computer_vision.h>

#include<iitktcs_msgs_srvs/objects_info.h>
#include<iitktcs_msgs_srvs/objects_info_vector.h>

#include<pcl/filters/voxel_grid.h>

//#include<arc17_computer_vision/object_information.h>

bool arc17_computer_vision_t::callback_service_computer_vision_picking(
        iitktcs_msgs_srvs::computer_vision_picking::Request &req,
        iitktcs_msgs_srvs::computer_vision_picking::Response &res )

{
    std::cout << "IN COMPUTER VISION PICKING\n";


    if(req.task.data == "PICK")
    {

        int bin_id = req.bin_id.data;


        unsigned char availability[NUM_OBJECTS];
        memset(availability,0,sizeof(unsigned char)*NUM_OBJECTS);

        int n_targets = req.ids_target.data.size();

        std::vector<int> target_objects_id;
        std::vector<int> mapped_target_objects_id;
        std::vector<int> mapped_availability;

        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ptr_objects_cloud_rgb;

        for(int i=0;i<NUM_OBJECTS;i++)
            ptr_objects_cloud_rgb.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));


        for(int i=0;i<n_targets;i++)
        {
            mapped_target_objects_id.push_back(map_competition_set_to_object_set[req.ids_target.data[i]]);
            target_objects_id.push_back(mapped_target_objects_id[i]);
        }

        for(int i=0;i<req.ids_available.data.size();i++)
        {
            mapped_availability.push_back(map_competition_set_to_object_set[req.ids_available.data[i]]);
            availability[mapped_availability[i]-1] = 1;
        }


        std::cout << "TASK = " << req.task.data << "\n";
        std::cout << "BIN ID = " << req.bin_id << "\n";

        std::cout << "\n-----AVAILABLE OBJECTS-----\n";
        for(int i=0;i<req.ids_available.data.size();i++)
            std::cout << req.ids_available.data[i] << "  " << object_names[mapped_availability[i]-1] << "\n";

        std::cout << "\n-----TARGET OBJECTS-----\n";
        for(int i=0;i<req.ids_target.data.size();i++)
            std::cout << req.ids_target.data[i] << "  " << object_names[mapped_target_objects_id[i]-1] << "\n";
        std::cout << "\n";


        //----IMAGE PROCESSING

        update_image_and_point_clouds();

        flag_do_forward_pass = true;
        while(flag_do_forward_pass)
            usleep(100);

        merge_decisions();
        get_rectangles(detection_roi_picking[bin_id],availability);

        //------DONE


        //        //-----WITHOUT OCCLUSION
        {
            decision_making_based_on_largest_point_cloud_picking(availability,
                                                                 ptr_objects_cloud_rgb,
                                                                 mapped_target_objects_id,
                                                                 detection_roi_picking[bin_id]);

//            std::vector<cv::Mat> polygon_masks;
//            std::vector<cv::Rect> polygon_rects;

//            decision_making_based_on_largest_point_cloud_picking_largset(availability,
//                                                                         ptr_objects_cloud_rgb,
//                                                                         mapped_target_objects_id,
//                                                                         detection_roi_picking[bin_id],
//                                                                         polygon_masks,
//                                                                         polygon_rects);

            //-----PUSH UNOCCLUDED TARGET CLOUDS

            ROS_INFO("PUSHING OBJECT INFO");
            for(int i=0;i<mapped_target_objects_id.size();i++)
            {
                int object_id = mapped_target_objects_id[i];

                std::cout << "PUSHING " << object_names[object_id - 1] << "\n";

                if(ptr_objects_cloud_rgb[object_id-1]->size())
                {

                    iitktcs_msgs_srvs::objects_info_vector object_infos;

                    iitktcs_msgs_srvs::objects_info object_info;
                    object_info.id.data = map_object_set_to_competition_set[object_id];

                    pcl::toROSMsg(*ptr_objects_cloud_rgb[object_id-1], object_info.roi);
                    object_infos.obj_info_vect.push_back(object_info);

                    res.object_info.push_back(object_infos);
                }
            }
            ROS_INFO("PUSHED OBJECTS");

        }

//        //-----WITH OCCLUSION
//        {

//            std::vector<cv::Mat> polygon_masks;
//            std::vector<cv::Rect> polygon_rects;

//            decision_making_based_on_largest_point_cloud_picking_largset(availability,
//                                                                         ptr_objects_cloud_rgb,
//                                                                         mapped_target_objects_id,
//                                                                         detection_roi_picking[bin_id],
//                                                                         polygon_masks,
//                                                                         polygon_rects);



//            //-----DETECT OCCLUSION
//            std::vector<std::vector<int> > occluding_obj_id;

//            detect_occlusion_largest(availability,
//                                     mapped_target_objects_id,
//                                     polygon_masks,
//                                     polygon_rects,
//                                     occluding_obj_id
//                                     );


//            for(int i=0;i<n_targets;i++)
//            {
//                if(occluding_obj_id[i].size())
//                {
//                    std::cout << "TARGET " << object_names[mapped_target_objects_id[i]-1] << "  IS OCCLUDED BY \n";

//                    for(int j=0;j<occluding_obj_id[i].size();j++)
//                        std::cout << object_names[occluding_obj_id[i][j]-1] << "\n";
//                }
//                else
//                {
//                    if(ptr_objects_cloud_rgb[mapped_target_objects_id[i]-1]->size())
//                        std::cout << "TARGET " << object_names[mapped_target_objects_id[i]-1] << "  IS FULLY VISIBLE \n";
//                }
//            }

//            //-----SEPARATE OCCLUDED AND UNOCCLUDED OBJECTS

//            std::vector<std::vector<int> > unoccluded_object_ids_set;
//            std::vector<std::vector<int> > occluded_object_ids_set;

//            for(int i=0;i<n_targets;i++)
//            {
//                if(occluding_obj_id[i].size())
//                {
//                    std::vector<int> object_set;
//                    object_set.push_back(mapped_target_objects_id[i]);

//                    for(int j=0;j < occluding_obj_id[i].size(); j++)
//                        object_set.push_back(occluding_obj_id[i][j]);

//                    occluded_object_ids_set.push_back(object_set);
//                }
//                else
//                {
//                    std::vector<int> object_set;
//                    object_set.push_back(mapped_target_objects_id[i]);

//                    unoccluded_object_ids_set.push_back(object_set);
//                }
//            }

//            //-----PUSH UNOCCLUDED TARGET CLOUDS

//            for(int i=0;i<unoccluded_object_ids_set.size();i++)
//            {
//                int object_id = unoccluded_object_ids_set[i][0];

//                std::cout << "PUSHING " << object_names[object_id - 1] << "\n";

//                if(ptr_objects_cloud_rgb[object_id-1]->size())
//                {
//                    ROS_INFO("PUSHING UNOCCLUDED OBJECT INFO");

//                    iitktcs_msgs_srvs::objects_info_vector object_infos;

//                    iitktcs_msgs_srvs::objects_info object_info;
//                    object_info.id.data = map_object_set_to_competition_set[object_id];

//                    pcl::toROSMsg(*ptr_objects_cloud_rgb[object_id-1], object_info.roi);
//                    object_infos.obj_info_vect.push_back(object_info);

//                    res.object_info.push_back(object_infos);

//                    ROS_INFO("PUSHED UNOCCLUDED OBJECTS");
//                }
//            }


//            //-----PUSH OCCLUDED TARGET CLOUDS

//            for(int i=0;i<occluded_object_ids_set.size();i++)
//            {
//                int object_id = occluded_object_ids_set[i][0];
//                std::cout << "PUSHING " << object_names[object_id - 1] << "\n";

//                if(ptr_objects_cloud_rgb[object_id-1]->size())
//                {
//                    ROS_INFO("PUSHING OCCLUDED OBJECT INFO");

//                    iitktcs_msgs_srvs::objects_info_vector object_infos;

//                    iitktcs_msgs_srvs::objects_info object_info;
//                    object_info.id.data = map_object_set_to_competition_set[object_id];

//                    pcl::toROSMsg(*ptr_objects_cloud_rgb[object_id-1], object_info.roi);
//                    object_infos.obj_info_vect.push_back(object_info);

//                    //-----BECAUSE FIRST ID IS THE TARGET ITSELF
//                    for(int j=1;j<occluded_object_ids_set[i].size();j++)
//                    {

//                        iitktcs_msgs_srvs::objects_info object_info;
//                        object_info.id.data = map_object_set_to_competition_set[occluded_object_ids_set[i][j]];

//                        pcl::toROSMsg(*ptr_objects_cloud_rgb[occluded_object_ids_set[i][j]-1],object_info.roi);
//                        object_infos.obj_info_vect.push_back(object_info);
//                    }

//                    res.object_info.push_back(object_infos);

//                    ROS_INFO("PUSHED OCCLUDED OBJECTS");
//                }
//            }
//        }
//


        //-----PUSH BIN CLOUD
        //        const int x_loc = 0;
        //        const int y_loc = 1;
        //        const int z_loc = 2;

        //        float x_min = BIN_BOUNDARIES[bin_id][x_loc];
        //        float y_min = BIN_BOUNDARIES[bin_id][y_loc];
        //        float z_min = BIN_BOUNDARIES[bin_id][z_loc];

        //        float x_max = BIN_BOUNDARIES[bin_id][x_loc];
        //        float y_max = BIN_BOUNDARIES[bin_id][y_loc];
        //        float z_max = BIN_BOUNDARIES[bin_id][z_loc];

        //        for(int i=1;i<4;i++)
        //        {
        //            x_min = std::min(x_min, BIN_BOUNDARIES[bin_id][3*i+x_loc]);
        //            y_min = std::min(y_min, BIN_BOUNDARIES[bin_id][3*i+y_loc]);
        //            z_min = std::min(z_min, BIN_BOUNDARIES[bin_id][3*i+z_loc]);

        //            x_max = std::max(x_max, BIN_BOUNDARIES[bin_id][3*i+x_loc]);
        //            y_max = std::max(y_max, BIN_BOUNDARIES[bin_id][3*i+y_loc]);
        //            z_max = std::max(z_max, BIN_BOUNDARIES[bin_id][3*i+z_loc]);
        //        }

        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_bin_cloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);

        //        for(int i=0;i<ptr_cloud_RGB->size();i++)
        //        {
        //            pcl::PointXYZRGB& point = ptr_cloud_RGB->at(i);
        //            if(point.x > x_min && point.x < x_max)
        //                if(point.y > y_min && point.y < y_max)
        //                    if(point.z > z_min && point.z < z_max)
        //                    {
        //                        ptr_bin_cloud_RGB->push_back(ptr_cloud_RGB->at(i));
        //                    }
        //        }



        //--------WITHOUT OCCLUSION
        {
            //        for(int i=0;i<n_targets;i++)
            //        {
            //            int object_id = mapped_target_objects_id[i];
            //            std::cout << "PUSHING " << object_names[object_id - 1] << "\n";

            //            if(ptr_objects_cloud_rgb[object_id-1]->size())
            //            {
            //                ROS_INFO("PUSHING OBJECT INFO");

            //                iitktcs_msgs_srvs::objects_info_vector object_infos;

            //                iitktcs_msgs_srvs::objects_info object_info;
            //                object_info.id.data = map_object_set_to_competition_set[object_id];

            //                pcl::toROSMsg(*ptr_objects_cloud_rgb[object_id-1], object_info.roi);
            //                object_infos.obj_info_vect.push_back(object_info);

            //                res.object_info.push_back(object_infos);

            //                ROS_INFO("PUSHED OBJECT");
            //            }
            //        }
        }

        int x_min = detection_roi_picking[bin_id].x;
        int y_min = detection_roi_picking[bin_id].y;
        int x_max = detection_roi_picking[bin_id].x + detection_roi_picking[bin_id].width;
        int y_max = detection_roi_picking[bin_id].y + detection_roi_picking[bin_id].height;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_bin_cloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);

        for(int i=y_min;i<y_max;i++)
            for(int j=x_min;j<x_max;j++)
            {
                pcl::PointXYZ point;

                point.x = j;
                point.y = i;
                point.z = 0;

                std::vector<int> indices;
                std::vector<float> distances;

                projection_kdtree.nearestKSearch(point,1,indices,distances);

                if(indices.size())
                    ptr_bin_cloud_RGB->push_back(ptr_cloud_RGB->at(indices[0]));
            }


        //        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        //        voxel_grid.setLeafSize(0.01,0.01,0.01);
        //        voxel_grid.setInputCloud(ptr_bin_cloud_RGB);
        //        voxel_grid.filter(*ptr_bin_cloud_RGB);

        pcl::toROSMsg(*ptr_bin_cloud_RGB,res.bin_cloud);

        //        pcl::io::savePCDFileBinary("/home/isl-server/ashish/bin_cloud_rgb.pcd",*ptr_bin_cloud_RGB);


        //-----SAVING BIN CLOUDS

        for(int i=0;i<n_targets;i++)
        {
            std::stringstream cloud_name;
            cloud_name << "object_cloud_" << i;

            std::stringstream cloud_to_save_name;
            cloud_to_save_name << "/home/isl-server/ashish/object_" << i << ".pcd";

            //            if(ptr_objects_cloud_rgb[i]->size())
            //                pcl::io::savePCDFileBinary(cloud_to_save_name.str(),*ptr_objects_cloud_rgb[i]);
        }


        std::cout << "LARGEST OBJECT = " << object_names[mapped_target_objects_id[0]-1] << "\n";
        std::cout << "SENDIND TARGETS CLOUDS\n";
        std::cout << "DONE\n";

        if(VISUALIZE_IMAGE)
        {
            draw_rectangles(detection_roi_picking[bin_id]);

            detection_roi_to_display = detection_roi_picking[bin_id];
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

            for(int i=0;i<n_targets;i++)
            {
                std::stringstream cloud_name;
                cloud_name << "object_cloud_" << i;

                pcl_visualizer->addPointCloud(ptr_objects_cloud_rgb[i],cloud_name.str());
            }

            flag_continue_spinning_pcl_visualizer = true;
            while(flag_update_pcl_visualizer)
                usleep(1000);
        }

        return true;
    }
    else
        return false;

}
