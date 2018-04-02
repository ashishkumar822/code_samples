#ifndef __ARC_2017_COMPUTER_VISION_H__
#define __ARC_2017_COMPUTER_VISION_H__

#include<opencv2/opencv.hpp>
#include<eigen3/Eigen/Dense>


#define COMPILE_ENSENSO

//-----ENSENSO

#ifdef COMPILE_ENSENSO
#include<nxLib.h>
#endif

//------PCL

#include<pcl_ros/point_cloud.h>
#include<pcl/visualization/cloud_viewer.h>
#include<pcl/search/kdtree.h>

//------CAFFE

#include<caffe/caffe.hpp>
#include<caffe/util/upgrade_proto.hpp>
#include<caffe/common.hpp>

#include<pthread.h>

//------SERVICES

#include<iitktcs_msgs_srvs/fetch_foscam_image.h>
#include<iitktcs_msgs_srvs/computer_vision.h>
#include<iitktcs_msgs_srvs/computer_vision_picking.h>
#include<iitktcs_msgs_srvs/computer_vision_stowing.h>


//-----GPU

#include<cuda.h>
#include<cuda_runtime.h>



typedef struct _params_t_
{

    std::string TASK;
    std::string KEY;
    std::string DEPTH_SENSOR;

    std::string ENSENSO_ID;

    bool VISUALIZE_PCL;
    bool VISUALIZE_IMAGE;
    bool USE_RECTANGLE_ROI_PICKING;
    bool USE_ROTATED_RECTANGLE_ROI_PICKING;
    bool USE_CRF;

    float CONFIDENCE_THRESHOLD;

    int FOSCAM_WIDTH;
    int FOSCAM_HEIGHT;

    int ENSENSO_WIDTH;
    int ENSENSO_HEIGHT;

    int N_BINS;

    // BIN BOUNDARIES
    std::vector<std::vector<float> > BIN_BOUNDARIES;

    //-----STOWING PARAMS
    std::string IMAGE_CHANNEL;

    std::string NET_TEST_PROTO;
    std::string NET_PRETRAINED;

    std::string CRF_NET_TEST_PROTO;

    int GPU_ID_SCENE_PARSING;
    int GPU_ID_POINT_PROJECTION;
}params_t;




class arc17_computer_vision_t
{
public:

    //-----PARAMS
    //------------------********************************-------------------//

    std::string TASK;
    std::string KEY;
    std::string DEPTH_SENSOR;

    std::string ENSENSO_ID;

    bool VISUALIZE_PCL;
    bool VISUALIZE_IMAGE;

    bool USE_RECTANGLE_ROI_PICKING;
    bool USE_ROTATED_RECTANGLE_ROI_PICKING;

    float CONFIDENCE_THRESHOLD;

    int FOSCAM_WIDTH;
    int FOSCAM_HEIGHT;

    int ENSENSO_WIDTH;
    int ENSENSO_HEIGHT;

    int N_BINS;

    std::string IMAGE_CHANNEL;

    int GPU_ID_SCENE_PARSING;
    int GPU_ID_POINT_PROJECTION;

    bool USE_CRF;
    //------------------********************************-------------------//

    std::vector<std::vector<float> > BIN_BOUNDARIES;

    //-----ENUM VARS

    enum DEPTH_SENSOR_TYPE
    {
        KINECT,
        ENSENSO
    };

    enum POINT_PROJECTION_ON_IMAGE_MODE
    {
        CPU,
        GPU
    };

    DEPTH_SENSOR_TYPE depth_sensor;
    POINT_PROJECTION_ON_IMAGE_MODE point_projection_mode;

    //----- GENERAL

    bool flag_update_point_cloud;
    bool flag_update_foscam_image;
    bool flag_do_forward_pass;
    bool flag_project_points_on_image;
    bool flag_continue_spinning_pcl_visualizer;
    bool flag_update_pcl_visualizer;
    bool flag_update_images;
    bool flag_compute_rotated_rect_roi;

    //-----OPENCV VARS

    cv::Mat foscam_image;
    cv::Mat foscam_labels;
    cv::Mat foscam_color_labels;

    cv::Mat detection_roi_mask_stowing;
    cv::Mat* detection_roi_mask_picking;
    cv::Mat detection_roi_mask;

    cv::Rect detection_roi_stowing;
    cv::Rect* detection_roi_picking;
    cv::Rect detection_roi;
    cv::Rect detection_roi_to_display;


    std::vector<cv::Mat> object_masks;
    std::vector<std::vector<cv::Rect> > object_rects;
    std::vector<std::vector<cv::RotatedRect> > object_rotated_rects;

    std::vector<std::string> object_names;
    std::vector<int> competition_set;
    std::map<int,int> map_competition_set_to_object_set;
    std::map<int,int> map_object_set_to_competition_set;

    int NUM_OBJECTS;

#ifdef COMPILE_ENSENSO
    //-----ENSENSO VARS
    NxLibItem camera;
#endif

    //-----PCL VARS

    pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_cloud_RGB;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_projection_cloud;

    pcl::search::KdTree<pcl::PointXYZ> projection_kdtree;

    pcl::visualization::PCLVisualizer* pcl_visualizer;
    //-----EIGEN VARS

    //-----ROS VARS
    ros::ServiceServer service_server_computer_vision;
    ros::ServiceServer service_server_computer_vision_picking;
    ros::ServiceServer service_server_computer_vision_stowing;

    ros::ServiceClient service_client_foscam;
    ros::Subscriber kinect_point_cloud_subscriber;

    iitktcs_msgs_srvs::fetch_foscam_image srv_fetch_foscam_image;

    //-----CAFFE VARS

    std::string str_net_test_proto;
    std::string str_net_pretrained;
    std::string str_crf_net_test_proto;

    //-----NET
    caffe::Net<float>* net;
    caffe::Net<float>* crf_net;

    std::vector<caffe::Blob<float>* > net_blobs_input;
    std::vector<caffe::Blob<float>* > net_blobs_output;

    caffe::Blob<float>* net_input_blob;
    caffe::Blob<float>* net_output_blob;

    std::vector<int> net_input_blob_dims;
    std::vector<int> net_output_blob_dims;

    float* net_data_output_all;


    //-----PTHREAD
    pthread_t forward_pass_thread;
    pthread_t project_point_cloud_thread;
    pthread_t foscam_image_thread;
    pthread_t ensenso_cloud_thread;
    pthread_t spin_pcl_visualizer_thread;
    pthread_t update_images_thread;

    //-----
    arc17_computer_vision_t(params_t& params,
                            std::vector<std::string>& all_object_names,
                            std::vector<int>& ids_competition_set)
    {
        TASK = params.TASK;
        KEY = params.KEY;
        N_BINS = params.N_BINS;

        DEPTH_SENSOR = params.DEPTH_SENSOR;
        ENSENSO_ID = params.ENSENSO_ID;
        IMAGE_CHANNEL = params.IMAGE_CHANNEL;

        CONFIDENCE_THRESHOLD = params.CONFIDENCE_THRESHOLD;

        VISUALIZE_PCL = params.VISUALIZE_PCL;
        VISUALIZE_IMAGE = params.VISUALIZE_IMAGE;

        USE_RECTANGLE_ROI_PICKING = params.USE_RECTANGLE_ROI_PICKING;
        USE_ROTATED_RECTANGLE_ROI_PICKING = params.USE_ROTATED_RECTANGLE_ROI_PICKING;

        FOSCAM_WIDTH = params.FOSCAM_WIDTH;
        FOSCAM_HEIGHT = params.FOSCAM_HEIGHT;

        ENSENSO_WIDTH = params.ENSENSO_WIDTH;
        ENSENSO_HEIGHT = params.ENSENSO_HEIGHT;

        GPU_ID_SCENE_PARSING = params.GPU_ID_SCENE_PARSING;
        GPU_ID_POINT_PROJECTION = params.GPU_ID_POINT_PROJECTION;


        USE_CRF = params.USE_CRF;
        str_crf_net_test_proto = params.CRF_NET_TEST_PROTO;

        if(TASK == "PICK")
        {
            for(int i=0;i<N_BINS;i++)
                BIN_BOUNDARIES.push_back(params.BIN_BOUNDARIES[i]);
        }

        for(int i=0;i<ids_competition_set.size();i++)
        {
            int actual_object_id = ids_competition_set[i];
            map_competition_set_to_object_set[actual_object_id] = i+1;
            map_object_set_to_competition_set[i+1] = actual_object_id;

            object_names.push_back(all_object_names[actual_object_id-1]);
        }


        NUM_OBJECTS = ids_competition_set.size();

        std::cout << NUM_OBJECTS << "\n";

        str_net_test_proto = params.NET_TEST_PROTO;
        str_net_pretrained = params.NET_PRETRAINED;

        init();
    }

    ~arc17_computer_vision_t(void)
    {
        pthread_cancel(forward_pass_thread);

        pthread_cancel(project_point_cloud_thread);
        pthread_cancel(foscam_image_thread);

#ifdef COMPILE_ENSENSO
        if(DEPTH_SENSOR == "ENSENSO")
        {
            pthread_cancel(ensenso_cloud_thread);
            close_ensenso_camera();
        }
#endif

        if(VISUALIZE_PCL)
        {
            pthread_cancel(spin_pcl_visualizer_thread);
        }

        if(VISUALIZE_IMAGE)
        {
            pthread_cancel(update_images_thread);
        }

    }

    void init(void);

    bool fetch_foscam_image(void);
    void fetch_ensenso_cloud(void);

    void callback_kinect_point_cloud_subscriber(const sensor_msgs::PointCloud2ConstPtr& ptr_cloud);

    bool callback_service_computer_vision(iitktcs_msgs_srvs::computer_vision::Request &req,
                                          iitktcs_msgs_srvs::computer_vision::Response &res);

    bool callback_service_computer_vision_stowing(iitktcs_msgs_srvs::computer_vision_stowing::Request &req,
                                                  iitktcs_msgs_srvs::computer_vision_stowing::Response &res);

    bool callback_service_computer_vision_picking(iitktcs_msgs_srvs::computer_vision_picking::Request &req,
                                                  iitktcs_msgs_srvs::computer_vision_picking::Response &res );


    void* callback_update_foscam_image(void* args);

    void read_detection_roi_masks(void);
    void read_network(void);
    void read_network_known(void);
    void read_network_unknown(void);
    void *forward_pass(void* args);
    void *forward_pass_unknown(void* args);
    void merge_decisions(void);

    void project_point_cloud(void);
    void project_point_cloud_GPU(void);
    void read_calibration_parameteres(void);

    void *project_point_cloud_on_image(void* args);
    void *callback_fetch_ensenso_cloud(void*args);

    void* spin_pcl_visualizer(void* args);
    void* update_images(void* args);

    void update_image_and_point_clouds(void);

    void process_net_output(void);
    void get_rectangles(cv::Rect decision_roi, unsigned char *availability);
    void draw_rectangles(cv::Rect roi);

    void decision_making_based_on_largest_point_cloud_stowing(unsigned char *availability,
                                                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ptr_object_cloud_rgb,
                                                              int& object_id);

    void decision_making_based_on_largest_point_cloud_picking(unsigned char *availability,
                                                              std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& ptr_object_cloud_rgb,
                                                              std::vector<int>& target_objects_id,
                                                              cv::Rect detection_roi_bin);

    void decision_making_based_on_largest_point_cloud_picking_largset(unsigned char *availability,
                                                                      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& ptr_object_cloud_rgb,
                                                                      std::vector<int>& target_objects_id,
                                                                      cv::Rect detection_roi_bin,
                                                                      std::vector<cv::Mat>& polygon_masks,
                                                                      std::vector<cv::Rect>& polygon_rects);

    void decision_making_based_on_largest_point_cloud_stowing(unsigned char *availability,
                                                              std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& ptr_object_cloud_rgb,
                                                              std::vector<int>& target_objects_id,
                                                              cv::Rect detection_roi_stowing
                                                              );


    void detect_occlusion(unsigned char *availability,
                          cv::Rect detection_roi,
                          std::vector<int>& target_objects_id,
                          std::vector<std::vector<int> >& occluding_obj_id);

    void detect_occlusion_largest(unsigned char *availability,
                                  std::vector<int>& target_objects_id,
                                  std::vector<cv::Mat>& polygon_masks,
                                  std::vector<cv::Rect> &polygon_rects,
                                  std::vector<std::vector<int> >& occluding_obj_id
                                  );

#ifdef COMPILE_ENSENSO
    //    -----ENSENSO ROUTINES
    void open_ensenso_camera(void);
    void close_ensenso_camera(void);
#endif

};

#endif



