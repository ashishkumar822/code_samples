#include<stdio.h>
#include<ros/ros.h>

#include<signal.h>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<boost/filesystem.hpp>


std::string data_set_path;// = "/home/isl-server/ashish/apc_new_objects_tote_data_foscam";
std::string TASK;
std::string CLUTTER_TYPE;

std::vector<std::string> all_object_names;
std::vector<int> ids_competition_set;

int CROP_SIZE;
float MIN_VISIBILITY;
float MIN_VISIBILITY_WRT_LARGEST_CONTOUR;
int n_images_multiclass;
int n_background_images;

int n_org_images;
const int n_threads = 24;

int NUM_OBJECTS;

std::stringstream background_images_dir;
std::stringstream clutter_objects_dir;
std::stringstream clutter_annotations_dir;
std::stringstream objects_cropped_objects_dir;
std::stringstream objects_annotations_dir;

std::vector<cv::Mat> cluttered_images(n_threads);
std::vector<cv::Mat> cluttered_masks(n_threads);


std::vector<int> n_images;//(NUM_NEW_OBJECTS);
std::vector<int> file_offset;


bool flag_data_gen_completed = false;
bool flag_generate_clutter = false;
bool flag_exit = false;

int thread_images_processed = 0;


void signal_handler_(int signal_id)
{
    std::cout << "\nBREAK SIGNAL CAUGHT\n";
    std::cout << "EXITING\n";
    flag_data_gen_completed =  true;
    flag_exit = true;
}


pthread_mutex_t mutex_rng;

cv::RNG rng;

void* generate_clutter_normal(void* args)
{


    int thread_id = *((int*)args);

    while(!flag_data_gen_completed)
    {
        if(flag_generate_clutter)
        {
            int file_number;

            pthread_mutex_lock(&mutex_rng);
            {
                file_number = rng.uniform(1,n_background_images+1);
            }
            pthread_mutex_unlock(&mutex_rng);


            std::stringstream img_file_name;
            img_file_name <<  background_images_dir.str() << file_number << ".png";

            cv::Mat& cluttered_image = cluttered_images[thread_id];
            cv::Mat& cluttered_mask = cluttered_masks[thread_id];

            cluttered_image = cv::imread(img_file_name.str());


            cv::Rect roi;
            roi.x = (cluttered_image.cols -CROP_SIZE)/2.0;
            roi.y = (cluttered_image.rows -CROP_SIZE)/2.0;
            roi.width = CROP_SIZE;
            roi.height = CROP_SIZE;

            cluttered_image = cluttered_image(roi);


            cluttered_mask = cv::Mat::zeros(CROP_SIZE, CROP_SIZE, CV_8UC3);

            // 3 levels of clutter low, medium, high
            //4 levels of occlusion 0,25,50,75

            int clutter_level;

            pthread_mutex_lock(&mutex_rng);
            {
                clutter_level = rng(2);
            }
            pthread_mutex_unlock(&mutex_rng);

            int clutter_divisions[] = {4,5};


            int max_x_div = cluttered_image.cols / clutter_divisions[clutter_level];
            int max_y_div = cluttered_image.rows / clutter_divisions[clutter_level];

            unsigned char obj_processed[NUM_OBJECTS];

            memset(obj_processed,0,NUM_OBJECTS);


            for(int j=0;j<clutter_divisions[clutter_level]-1;j++)
                for(int k=0;k<clutter_divisions[clutter_level]-1;k++)
                {

                    for(int l=0;l<2;l++)
                    {
                        int object_id = rng.uniform(1,NUM_OBJECTS+1)-1;

                        //                        //                            if(!obj_processed[object_id])
                        {
                            obj_processed[object_id] = 1;

                            int max_images = n_images[object_id];
                            int file_num;

                            pthread_mutex_lock(&mutex_rng);
                            {
                                file_num = rng.uniform(1,max_images+1);
                            }
                            pthread_mutex_unlock(&mutex_rng);


                            int exact_file_num = file_num + file_offset[object_id];

                            std::stringstream clutter_sample_image_name;
                            clutter_sample_image_name << objects_cropped_objects_dir.str() << exact_file_num << ".png";


                            std::stringstream clutter_sample_mask_name;
                            clutter_sample_mask_name << objects_annotations_dir.str() <<  "mask_" << exact_file_num << ".png";


                            cv::Mat cluttered_sample_image = cv::imread(clutter_sample_image_name.str());
                            cv::Mat cluttered_sample_mask  = cv::imread(clutter_sample_mask_name.str());


                            cv::Mat cluttered_gray_mask;
                            cv::cvtColor(cluttered_sample_mask,cluttered_gray_mask,CV_BGR2GRAY);
                            cv::threshold(cluttered_gray_mask,cluttered_gray_mask,0,255,CV_THRESH_BINARY);


                            std::vector<std::vector<cv::Point2i> > clutter_contours;
                            cv::findContours(cluttered_gray_mask,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                            cv::Rect clutter_bounding_rect;

                            if(clutter_contours.size())
                                clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                            else
                                continue;

                            cv::Point2i center;
                            center.x  = clutter_bounding_rect.x + clutter_bounding_rect.width/2.0f;
                            center.y  = clutter_bounding_rect.y + clutter_bounding_rect.height/2.0f;

                            cv::Point2i anchor;
                            anchor.x = max_x_div*(k+1);
                            anchor.y = max_y_div*(j+1);


                            clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                            for(int l=1;l<clutter_contours.size();l++)
                                clutter_bounding_rect |= cv::boundingRect(clutter_contours[l]);

                            //                                    //-----TO PREVENT FALSE OBJECT CROPPING AT THE BOUNDARIES
                            //                                    int pad = 2;
                            //                                    if(clutter_bounding_rect.x > 0)
                            //                                        clutter_bounding_rect.x -= pad/2;

                            //                                    if(clutter_bounding_rect.y > 0)
                            //                                        clutter_bounding_rect.y -= pad/2;

                            //                                    if(clutter_bounding_rect.x + clutter_bounding_rect.width < cluttered_sample_image.cols)
                            //                                        clutter_bounding_rect.width += pad;


                            //                                    if(clutter_bounding_rect.y + clutter_bounding_rect.height < cluttered_sample_image.rows)
                            //                                        clutter_bounding_rect.height += pad;


                            //                                    cv::Rect roi_to_copy;
                            //                                    roi_to_copy.x = anchor.x  - clutter_bounding_rect.width / 2;
                            //                                    roi_to_copy.y = anchor.y  - clutter_bounding_rect.height / 2;
                            //                                    roi_to_copy.width = clutter_bounding_rect.width;
                            //                                    roi_to_copy.height = clutter_bounding_rect.height;

                            //                                    if(roi_to_copy.x < 0)
                            //                                    {
                            //                                        //-----REDUCE THE WIDTH
                            //                                        roi_to_copy.width += roi_to_copy.x ;

                            //                                        clutter_bounding_rect.x -= roi_to_copy.x;
                            //                                        clutter_bounding_rect.width = roi_to_copy.width;

                            //                                        roi_to_copy.x = 0;

                            //                                    }

                            //                                    if(roi_to_copy.y < 0)
                            //                                    {

                            //                                        //-----REDUCE THE HEIGHT
                            //                                        roi_to_copy.height += roi_to_copy.y ;

                            //                                        clutter_bounding_rect.y -= roi_to_copy.y;
                            //                                        clutter_bounding_rect.height = roi_to_copy.height;

                            //                                        roi_to_copy.y = 0;
                            //                                    }

                            //                                    if(roi_to_copy.x + roi_to_copy.width > cluttered_sample_image.cols)
                            //                                    {

                            //                                        //-----REDUCE THE WIDTH
                            //                                        roi_to_copy.width -=  roi_to_copy.x + roi_to_copy.width - cluttered_sample_image.cols;
                            //                                        clutter_bounding_rect.width = roi_to_copy.width;
                            //                                    }

                            //                                    if(roi_to_copy.y + roi_to_copy.height > cluttered_sample_image.rows)
                            //                                    {
                            //                                        //-----REDUCE THE HEIGHT
                            //                                        roi_to_copy.height -=  roi_to_copy.y + roi_to_copy.height - cluttered_sample_image.rows;
                            //                                        clutter_bounding_rect.height = roi_to_copy.height;
                            //                                    }



                            //                                    cluttered_sample_image(clutter_bounding_rect).copyTo(cluttered_image(roi_to_copy),cluttered_sample_mask(clutter_bounding_rect));
                            //                                    cluttered_sample_mask(clutter_bounding_rect).copyTo(cluttered_mask(roi_to_copy),cluttered_sample_mask(clutter_bounding_rect));


                            for(int l=0;l<cluttered_sample_image.rows;l++)
                                for(int m=0;m<cluttered_sample_image.cols;m++)
                                {
                                    unsigned char* cluttered_sample_mask_pixel = cluttered_sample_mask.data + l* cluttered_sample_mask.step[0] + m * cluttered_sample_mask.step[1];
                                    unsigned char* cluttered_sample_image_pixel = cluttered_sample_image.data + l * cluttered_sample_image.step[0] + m * cluttered_sample_image.step[1];

                                    if(cluttered_sample_mask_pixel[0])
                                    {
                                        int loc_x = m - center.x + anchor.x;
                                        int loc_y = l - center.y + anchor.y;

                                        if(loc_x > -1 && loc_x < cluttered_sample_image.cols && loc_y > -1 && loc_y < cluttered_sample_image.rows )
                                        {
                                            unsigned char* cluttered_image_pixel = cluttered_image.data + loc_y * cluttered_image.step[0] + loc_x * cluttered_image.step[1];
                                            unsigned char* cluttered_mask_pixel = cluttered_mask.data + loc_y * cluttered_mask.step[0] + loc_x * cluttered_mask.step[1];


                                            cluttered_image_pixel[0] = cluttered_sample_image_pixel[0];
                                            cluttered_image_pixel[1] = cluttered_sample_image_pixel[1];
                                            cluttered_image_pixel[2] = cluttered_sample_image_pixel[2];

                                            cluttered_mask_pixel[0] = cluttered_sample_mask_pixel[0];
                                            cluttered_mask_pixel[1] = cluttered_sample_mask_pixel[1];
                                            cluttered_mask_pixel[2] = cluttered_sample_mask_pixel[2];

                                        }
                                    }
                                }
                        }
                    }
                }
            pthread_mutex_lock(&mutex_rng);
            {
                thread_images_processed++;
                if(thread_images_processed == n_threads)
                {
                    flag_generate_clutter = false;
                    thread_images_processed = 0;
                }

            }
            pthread_mutex_unlock(&mutex_rng);
        }

    }
}


int main(int argc, char** argv)
{
    ros::init(argc,argv,"synthetic_clutter_generation_online");

    ros::NodeHandle nh;


    signal(SIGINT,signal_handler_);

    //-----READ PARAMS FROM ROS SERVER


    nh.getParam("/ARC17_DATASET_PATH",data_set_path);
    nh.getParam("/ARC17_OBJECT_NAMES",all_object_names);
    nh.getParam("/ARC17_COMPETITION_SET",ids_competition_set);
    nh.getParam("/ARC17_TASK",TASK);
    nh.getParam("/ARC17_CROP_SIZE",CROP_SIZE);
    nh.getParam("/ARC17_MIN_VISIBILITYT",MIN_VISIBILITY);
    nh.getParam("/ARC17_MIN_VISIBILITY_WRT_LARGEST_CONTOUR",MIN_VISIBILITY_WRT_LARGEST_CONTOUR);
    nh.getParam("/ARC17_CLUTTER_TYPE",CLUTTER_TYPE);
    nh.getParam("/ARC17_N_IMAGES_MULTICLASS",n_images_multiclass);

    //-----


    std::cout << "GENERATING SYNTHETIC CLUTTER\n";

    //    int dummy;
    //    std::cout << "*****-----PRESS 0 TO START GENERATING DATA\n";

    //    std::cin >> dummy;

    std::string str_task_dir;
    std::stringstream str_file_n_images_new_objects;

    if(TASK == "STOW")
    {
        str_task_dir = "objects_dataset_stowing/";
    }
    if(TASK == "PICK")
    {
        str_task_dir = "objects_dataset_picking/";
    }
    if(TASK == "STOW-PICK")
    {
        str_task_dir = "objects_dataset_stowing_picking/";
    }

    str_file_n_images_new_objects << data_set_path << str_task_dir << "new_objects_n_files.yaml";

    clutter_objects_dir << data_set_path << str_task_dir << "/clutter_objects/";

    clutter_annotations_dir << data_set_path << str_task_dir << "/clutter_annotations/";


    objects_cropped_objects_dir << data_set_path << str_task_dir << "/objects/";

    objects_annotations_dir << data_set_path << str_task_dir << "/annotations/";

    boost::filesystem::create_directories(clutter_objects_dir.str());
    boost::filesystem::create_directories(clutter_annotations_dir.str());

    if(CLUTTER_TYPE == "NEW_OBJECTS")
    {

        background_images_dir << data_set_path << "/background_against_unknown/";

        n_background_images  = 0;

        boost::filesystem::directory_iterator dir_iterator(background_images_dir.str());
        while( dir_iterator != boost::filesystem::directory_iterator())
        {
            n_background_images++;
            *dir_iterator++;
        }


        //----READ TOTAL FILES IN ORDER TO APPEND THE CLUTTER

        cv::FileStorage file_n_images_new_objects(str_file_n_images_new_objects.str(),cv::FileStorage::READ);

        //-----READING NUMBER OF IMAGES PER OBJECT FOR ALL 10/16(old) + 10/16(new) objects
        NUM_OBJECTS = ids_competition_set.size();

        int NUM_OLD_OBJECTS = NUM_OBJECTS / 2;
        int NUM_NEW_OBJECTS = NUM_OBJECTS / 2;

        std::vector<int> object_ids_mapping;

        for(int i = 0; i < NUM_NEW_OBJECTS;i++)
        {
            int index =  ids_competition_set[i+NUM_OLD_OBJECTS] - 1;
            std::stringstream n_images_obj;
            n_images_obj << "n_images_" << all_object_names[index];

            int n_image;
            file_n_images_new_objects[n_images_obj.str()] >> n_image;

            n_images.push_back(n_image);
            //-----COMPUTE OFFSET and ID MAPPING TO IMAGES FILES FOR EACH OBJECT

            int offset = 0;
            for(int j=0;j<i;j++)
                offset += n_images[j];

            file_offset.push_back(offset);
            object_ids_mapping.push_back(index);
        }

        n_org_images = 0;
        for(int i=0;i<n_images.size();i++)
            n_org_images += n_images[i];

        //-----GENERATE ACTUAL CLUTTER
        int total_files =  1;
        cv::RNG rng;

        for(int i=0;i<n_images_multiclass;i++)
        {
            cv::Mat cluttered_image;
            cv::Mat cluttered_mask;

            int obj_image_or_background = rng.uniform(0,2);
            obj_image_or_background = 0;

            if(obj_image_or_background)
            {
                int file_number = rng.uniform(1,n_org_images+1);

                std::stringstream img_file_name;
                img_file_name <<  objects_cropped_objects_dir.str() << file_number << ".png";

                std::stringstream mask_file_name;
                mask_file_name << objects_annotations_dir.str() <<  "mask_" << file_number << ".png";

                cluttered_image = cv::imread(img_file_name.str());
                cluttered_mask = cv::imread(mask_file_name.str());

            }
            else
            {

                int file_number = rng.uniform(1,n_background_images+1);

                std::stringstream img_file_name;
                img_file_name <<  background_images_dir.str() << file_number << ".png";

                cluttered_image = cv::imread(img_file_name.str());

                cv::Rect roi;
                roi.x = (cluttered_image.cols -CROP_SIZE)/2.0;
                roi.y = (cluttered_image.rows -CROP_SIZE)/2.0;
                roi.width = CROP_SIZE;
                roi.height = CROP_SIZE;

                cluttered_image = cluttered_image(roi);

                cv::Mat loc_cluttered_image;
                cluttered_image.copyTo(loc_cluttered_image);

                cv::Mat loc_cluttered_mask = cv::Mat::zeros(roi.height,roi.width,CV_8UC1);

                // 3 levels of clutter low, medium, high
                //4 levels of occlusion 0,25,50,75

                int clutter_level = rng.uniform(0,3);
                int clutter_divisions[] = {3,4,5};

                int max_x_div = cluttered_image.cols / clutter_divisions[clutter_level];
                int max_y_div = cluttered_image.rows / clutter_divisions[clutter_level];


                //-----TO ENSURE EACH OBJECT OCCURRS ONCE
                //-----CAN BE DISABLED AS WELL...
                //-----JUST TO PREVENT DATASET BIASING

                unsigned char obj_processed[NUM_NEW_OBJECTS];
                memset(obj_processed,0,NUM_NEW_OBJECTS);

                //                std::vector<cv::Mat> images;
                std::vector<int> labels;
                std::vector<int> sizes;
                std::vector<cv::Rect> object_rects;

                int n_clutter_filled = 1;


                //-----GENERATES CLUTTER MASK SINGAL CHANNEL
                //-----EACH CLASS IS ALLOWED ONCE
                //-----LABELS IN THE MASK IS INSTANCE ID not the exact labels
                //-----THIS IS DONE IN ORDER TO ABLE TO POST CHECK FOR VISIBILITY
                //-----AND TO CREATE BOXES IN CASE OF MULTIPLE INSTANCES OF THE SAME OBJECT

                for(int j=0;j<clutter_divisions[clutter_level]-1;j++)
                    for(int k=0;k<clutter_divisions[clutter_level]-1;k++)
                    {

                        int object_id = rng.uniform(1,NUM_OBJECTS+1)-1;

                        if(!obj_processed[object_id])
                        {
                            obj_processed[object_id] = 1;

                            int max_images = n_images[object_id];
                            int file_num = rng.uniform(1,max_images+1);

                            int exact_file_num = 0;
                            for(int l=0;l< object_id ;l++)
                                exact_file_num += n_images[l];

                            exact_file_num += file_num;

                            std::stringstream clutter_sample_image_name;
                            clutter_sample_image_name << objects_cropped_objects_dir.str() << exact_file_num << ".png";


                            std::stringstream clutter_sample_mask_name;
                            clutter_sample_mask_name << objects_annotations_dir.str() <<  "mask_" << exact_file_num << ".png";

                            cv::Mat cluttered_sample_image = cv::imread(clutter_sample_image_name.str());
                            cv::Mat cluttered_sample_mask  = cv::imread(clutter_sample_mask_name.str());


                            cv::Mat cluttered_gray_mask;
                            cv::cvtColor(cluttered_sample_mask,cluttered_gray_mask,CV_BGR2GRAY);
                            cv::threshold(cluttered_gray_mask,cluttered_gray_mask,0,255,CV_THRESH_BINARY);


                            std::vector<std::vector<cv::Point2i> > clutter_contours;
                            cv::findContours(cluttered_gray_mask,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                            if(!clutter_contours.size())
                                continue;


                            cv::Rect clutter_bounding_rect;
                            clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                            for(int l=1;l<clutter_contours.size();l++)
                                clutter_bounding_rect |= cv::boundingRect(clutter_contours[l]);

                            int pad = 2;
                            if(clutter_bounding_rect.x > 0)
                                clutter_bounding_rect.x -= pad/2;

                            if(clutter_bounding_rect.y > 0)
                                clutter_bounding_rect.y -= pad/2;

                            if(clutter_bounding_rect.x + clutter_bounding_rect.width < cluttered_sample_image.cols)
                                clutter_bounding_rect.width += pad;


                            if(clutter_bounding_rect.y + clutter_bounding_rect.height < cluttered_sample_image.rows)
                                clutter_bounding_rect.height += pad;


                            cv::Point2i anchor;
                            anchor.x = max_x_div*(k+1) ;
                            anchor.y = max_y_div*(j+1);

                            cv::Rect roi_to_copy;
                            roi_to_copy.x = anchor.x  - clutter_bounding_rect.width / 2;
                            roi_to_copy.y = anchor.y  - clutter_bounding_rect.height / 2;
                            roi_to_copy.width = clutter_bounding_rect.width;
                            roi_to_copy.height = clutter_bounding_rect.height;

                            if(roi_to_copy.x < 0)
                            {
                                //-----REDUCE THE WIDTH
                                roi_to_copy.width += roi_to_copy.x ;

                                clutter_bounding_rect.x -= roi_to_copy.x;
                                clutter_bounding_rect.width = roi_to_copy.width;

                                roi_to_copy.x = 0;

                            }

                            if(roi_to_copy.y < 0)
                            {

                                //-----REDUCE THE HEIGHT
                                roi_to_copy.height += roi_to_copy.y ;

                                clutter_bounding_rect.y -= roi_to_copy.y;
                                clutter_bounding_rect.height = roi_to_copy.height;

                                roi_to_copy.y = 0;
                            }

                            if(roi_to_copy.x + roi_to_copy.width > cluttered_sample_image.cols)
                            {
                                //-----REDUCE THE WIDTH
                                roi_to_copy.width -=  roi_to_copy.x + roi_to_copy.width - cluttered_sample_image.cols;
                                clutter_bounding_rect.width = roi_to_copy.width;
                            }

                            if(roi_to_copy.y + roi_to_copy.height > cluttered_sample_image.rows)
                            {
                                //-----REDUCE THE HEIGHT
                                roi_to_copy.height -=  roi_to_copy.y + roi_to_copy.height - cluttered_sample_image.rows;
                                clutter_bounding_rect.height = roi_to_copy.height;
                            }

                            int object_pixel_counts = 0;

                            int loc_x = roi_to_copy.x;
                            int loc_y = roi_to_copy.y;



                            for(int l=clutter_bounding_rect.y;l<clutter_bounding_rect.y + clutter_bounding_rect.height;l++)
                            {
                                for(int m=clutter_bounding_rect.x;m< clutter_bounding_rect.x +  clutter_bounding_rect.width ;m++)
                                {
                                    unsigned char* cluttered_sample_mask_pixel = cluttered_sample_mask.data + l* cluttered_sample_mask.step[0] + m * cluttered_sample_mask.step[1];
                                    unsigned char* cluttered_sample_image_pixel = cluttered_sample_image.data + l * cluttered_sample_image.step[0] + m * cluttered_sample_image.step[1];

                                    if(cluttered_sample_mask_pixel[0] == object_id +1)
                                    {
                                        unsigned char* cluttered_image_pixel = cluttered_image.data + loc_y * cluttered_image.step[0] + loc_x * cluttered_image.step[1];
                                        unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + loc_y * loc_cluttered_mask.step[0] + loc_x * loc_cluttered_mask.step[1];

                                        cluttered_image_pixel[0] = cluttered_sample_image_pixel[0];
                                        cluttered_image_pixel[1] = cluttered_sample_image_pixel[1];
                                        cluttered_image_pixel[2] = cluttered_sample_image_pixel[2];

                                        //                                        cluttered_mask_pixel[0] = cluttered_sample_mask_pixel[0];
                                        //                                        cluttered_mask_pixel[1] = cluttered_sample_mask_pixel[1];
                                        //                                        cluttered_mask_pixel[2] = cluttered_sample_mask_pixel[2];

                                        loc_cluttered_mask_pixel[0] = n_clutter_filled;
                                        object_pixel_counts++;
                                    }
                                    loc_x++;
                                }
                                loc_y++;
                                loc_x = roi_to_copy.x;
                            }

                            n_clutter_filled++;

                            //                            images.push_back(cluttered_sample_image);
                            labels.push_back(object_id+1);
                            sizes.push_back(object_pixel_counts);
                            object_rects.push_back(roi_to_copy);

                        }
                    }


                //-----CHECKS FOR THE PERCENTAGE VISIBILTY
                //-----IF ONLY ONE CONTOUR AND VISIBILITY IS GREATER THEN A THRESHOLD
                //-----MASK IS KEPT INTACT
                //-----IN CASE OF MULTIPLE CONTOURS AFTER CLUTTERING, EACH CONTOUR IS CHECKED AND
                //-----CONTOUR IS LEFT INTACT IN CASE IT IS GREATER THEN A CERTAIN PERCENTAGE OF
                //-----THE LARGEST CONTOUR

                for(int j=0;j<labels.size();j++)
                {
                    int size = sizes[j];
                    cv::Rect& object_rect = object_rects[j];

                    cv::Mat object_cluttered_mask = (loc_cluttered_mask(object_rect) == j+1);

                    int object_size = cv::countNonZero(object_cluttered_mask);
                    float visibility = (float)object_size / size;

                    if(visibility > MIN_VISIBILITY)
                    {

                        cv::Mat cropped_cluttered_mask_contours;
                        object_cluttered_mask.copyTo(cropped_cluttered_mask_contours);

                        std::vector<std::vector<cv::Point2i> > clutter_contours;
                        cv::findContours(cropped_cluttered_mask_contours,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                        if(clutter_contours.size() > 1)
                        {

                            std::vector<int> contour_areas;

                            int max_area = -10000000;

                            for(int k=0;k<clutter_contours.size();k++)
                            {
                                int area = cv::contourArea(clutter_contours[k]);
                                contour_areas.push_back(area);

                                if(max_area < area)
                                    max_area = area;
                            }

                            //-----FILL THE SMALLER CONTOURS AS COMPARED TO LARGEST CONTOUR WITH BACKGROUND
                            cv::Mat cropped_cluttered_mask = loc_cluttered_mask(object_rect);

                            for(int k=0;k<clutter_contours.size();k++)
                            {
                                float visbility_wrt_largest_contour = (float)contour_areas[k] / max_area;

                                if(visbility_wrt_largest_contour < MIN_VISIBILITY_WRT_LARGEST_CONTOUR)
                                {
                                    std::vector<std::vector<cv::Point> > contour;
                                    contour.push_back(clutter_contours[k]);
                                    cv::fillPoly(cropped_cluttered_mask,contour,cv::Scalar(0));
                                }
                            }
                        }
                    }
                    else
                    {
                        for(int k=object_rect.y; k < object_rect.y + object_rect.height;k++)
                            for(int l=object_rect.x; l < object_rect.x + object_rect.width ;l++)
                            {
                                unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + k*loc_cluttered_mask.step[0] + l* loc_cluttered_mask.step[1];

                                if(loc_cluttered_mask_pixel[0] == j+1)
                                    loc_cluttered_mask_pixel[0] = 0;
                            }
                    }
                }

                cluttered_mask = cv::Mat::zeros(roi.height,roi.width,CV_8UC3);

                for(int j=0;j<cluttered_mask.rows;j++)
                    for(int k=0;k<cluttered_mask.cols;k++)
                    {
                        unsigned char* cluttered_mask_pixel = cluttered_mask.data + j* cluttered_mask.step[0] + k* cluttered_mask.step[1];
                        unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + j* loc_cluttered_mask.step[0] + k* loc_cluttered_mask.step[1];

                        int index = loc_cluttered_mask_pixel[0];

                        if(index > 0)
                        {
                            int label = labels[index-1];

                            cluttered_mask_pixel[0] = label;
                            cluttered_mask_pixel[1] = label;
                            cluttered_mask_pixel[2] = label;

                        }
                        else if(index == 0)
                        {
                            unsigned char* cluttered_image_pixel  = cluttered_image.data + j* cluttered_image.step[0] + k* cluttered_image.step[1];
                            unsigned char* loc_cluttered_image_pixel  = loc_cluttered_image.data + j* loc_cluttered_image.step[0] + k* loc_cluttered_image.step[1];

                            cluttered_image_pixel[0] = loc_cluttered_image_pixel[0];
                            cluttered_image_pixel[1] = loc_cluttered_image_pixel[1];
                            cluttered_image_pixel[2] = loc_cluttered_image_pixel[2];
                        }

                    }

            }

            cv::imshow("cluttered image",cluttered_image);
            cv::imshow("cluttered mask",5*cluttered_mask);
            cv::waitKey(0);

            std::stringstream cluttered_img_file_name;
            cluttered_img_file_name << objects_cropped_objects_dir.str() << total_files << ".png";

            std::stringstream cluttered_mask_file_name;
            cluttered_mask_file_name << objects_annotations_dir.str() <<  "mask_" << total_files << ".png";

            //        cv::imwrite(cluttered_img_file_name.str(),cluttered_image);
            //        cv::imwrite(cluttered_mask_file_name.str(),cluttered_mask);
            total_files++;
        }
    }
    else if(CLUTTER_TYPE == "COMPETITION_SET")
    {

        usleep(10000);

        background_images_dir << data_set_path << "/background_common/";

        n_background_images  = 0;

        boost::filesystem::directory_iterator dir_iterator(background_images_dir.str());
        while( dir_iterator != boost::filesystem::directory_iterator())
        {
            n_background_images++;
            *dir_iterator++;
        }

        //----READ TOTAL FILES IN ORDER TO APPEND THE CLUTTER
        cv::FileStorage file_n_images_new_objects(str_file_n_images_new_objects.str(),cv::FileStorage::READ);

        //-----READING NUMBER OF IMAGES PER OBJECT FOR ALL 10/16(old) + 10/16(new) objects
        NUM_OBJECTS = ids_competition_set.size();

        std::vector<int> object_ids_mapping;


        for(int i = 0; i < NUM_OBJECTS;i++)
        {
            int index =  ids_competition_set[i] - 1;
            std::stringstream n_images_obj;
            n_images_obj << "n_images_" << all_object_names[index];

            int n_image;
            file_n_images_new_objects[n_images_obj.str()] >> n_image;

            n_images.push_back(n_image);

            //-----COMPUTE OFFSET and ID MAPPING TO IMAGES FILES FOR EACH OBJECT

            int offset = 0;
            for(int j=0;j<i;j++)
                offset += n_images[j];

            file_offset.push_back(offset);
            object_ids_mapping.push_back(index+1);

        }

        n_org_images = 0;
        for(int i=0;i<n_images.size();i++)
            n_org_images += n_images[i];

        //-----GENERATE ACTUAL CLUTTER
        int total_files = 1;
        cv::RNG rng;

        std::cout << "N ORG IMAGES = " << n_org_images << "\n";



        pthread_t clutter_gen_threads[n_threads];

        int thread_id[n_threads];
        for(int i=0;i<n_threads;i++)
        {
            thread_id[i] = i;
            pthread_create(&clutter_gen_threads[i],NULL,generate_clutter_normal,&thread_id[i]);
        }

        for(int i=0;i<n_images_multiclass;i++)
        {

            if(flag_exit)
                break;

            usleep(10000);

            printf("\r TOTAL CLUTTER GENERATED = %d", total_files);
            cv::Mat cluttered_image;
            cv::Mat cluttered_mask;

            int obj_image_or_background = rng.uniform(0,2);
            obj_image_or_background = 0;

            if(obj_image_or_background)
            {
                int file_number = rng.uniform(1,n_org_images+1);


                std::stringstream img_file_name;
                img_file_name <<  objects_cropped_objects_dir.str() << file_number << ".png";

                std::stringstream mask_file_name;
                mask_file_name << objects_annotations_dir.str() <<  "mask_" << file_number << ".png";

                cluttered_image = cv::imread(img_file_name.str());
                cluttered_mask = cv::imread(mask_file_name.str());

            }
            else
            {

                u_int64_t t1,t2;

                t1 = (double)ros::Time::now().toNSec() / 1000.0;


                if(1)
                {
                    flag_generate_clutter = true;
                    while(flag_generate_clutter)
                        usleep(100000);
                }
                else
                {
                    int file_number = rng.uniform(1,n_background_images+1);

                    std::stringstream img_file_name;
                    img_file_name <<  background_images_dir.str() << file_number << ".png";

                    cluttered_image = cv::imread(img_file_name.str());

                    cv::Rect roi;
                    roi.x = (cluttered_image.cols -CROP_SIZE)/2.0;
                    roi.y = (cluttered_image.rows -CROP_SIZE)/2.0;
                    roi.width = CROP_SIZE;
                    roi.height = CROP_SIZE;

                    cluttered_image = cluttered_image(roi);

                    cv::Mat loc_cluttered_image;
                    cluttered_image.copyTo(loc_cluttered_image);

                    cv::Mat loc_cluttered_mask = cv::Mat::zeros(roi.height,roi.width,CV_8UC1);

                    // 3 levels of clutter low, medium, high
                    //4 levels of occlusion 0,25,50,75

                    int clutter_level = rng.uniform(0,3);
                    int clutter_divisions[] = {3,4,5};

                    int max_x_div = cluttered_image.cols / clutter_divisions[clutter_level];
                    int max_y_div = cluttered_image.rows / clutter_divisions[clutter_level];


                    //-----TO ENSURE EACH OBJECT OCCURRS ONCE
                    //-----CAN BE DISABLED AS WELL...
                    //-----JUST TO PREVENT DATASET BIASING

                    unsigned char obj_processed[NUM_OBJECTS];
                    memset(obj_processed,0,NUM_OBJECTS);

                    //                std::vector<cv::Mat> images;
                    std::vector<int> labels;
                    std::vector<int> sizes;
                    std::vector<cv::Rect> object_rects;

                    int n_clutter_filled = 1;


                    //-----GENERATES CLUTTER MASK SINGAL CHANNEL
                    //-----EACH CLASS IS ALLOWED ONCE
                    //-----LABELS IN THE MASK IS INSTANCE ID not the exact labels
                    //-----THIS IS DONE IN ORDER TO ABLE TO POST CHECK FOR VISIBILITY
                    //-----AND TO CREATE BOXES IN CASE OF MULTIPLE INSTANCES OF THE SAME OBJECT

                    for(int j=0;j<clutter_divisions[clutter_level]-1;j++)
                        for(int k=0;k<clutter_divisions[clutter_level]-1;k++)
                        {

                            int object_id = rng.uniform(1,NUM_OBJECTS+1)-1;

                            //                            if(!obj_processed[object_id])
                            {
                                obj_processed[object_id] = 1;

                                int max_images = n_images[object_id];
                                int file_num = rng.uniform(1,max_images+1);

                                int exact_file_num = file_num + file_offset[object_id];

                                int mapped_object_id = object_ids_mapping[object_id];

                                std::stringstream clutter_sample_image_name;
                                clutter_sample_image_name << objects_cropped_objects_dir.str() << exact_file_num << ".png";


                                std::stringstream clutter_sample_mask_name;
                                clutter_sample_mask_name << objects_annotations_dir.str() <<  "mask_" << exact_file_num << ".png";

                                cv::Mat cluttered_sample_image = cv::imread(clutter_sample_image_name.str());
                                cv::Mat cluttered_sample_mask  = cv::imread(clutter_sample_mask_name.str());

                                cv::Mat cluttered_gray_mask;
                                cv::cvtColor(cluttered_sample_mask,cluttered_gray_mask,CV_BGR2GRAY);
                                cv::threshold(cluttered_gray_mask,cluttered_gray_mask,0,255,CV_THRESH_BINARY);


                                std::vector<std::vector<cv::Point2i> > clutter_contours;
                                cv::findContours(cluttered_gray_mask,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                                if(!clutter_contours.size())
                                    continue;


                                cv::Rect clutter_bounding_rect;
                                clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                                for(int l=1;l<clutter_contours.size();l++)
                                    clutter_bounding_rect |= cv::boundingRect(clutter_contours[l]);

                                //-----TO PREVENT FALSE OBJECT CROPPING AT THE BOUNDARIES
                                int pad = 2;
                                if(clutter_bounding_rect.x > 0)
                                    clutter_bounding_rect.x -= pad/2;

                                if(clutter_bounding_rect.y > 0)
                                    clutter_bounding_rect.y -= pad/2;

                                if(clutter_bounding_rect.x + clutter_bounding_rect.width < cluttered_sample_image.cols)
                                    clutter_bounding_rect.width += pad;


                                if(clutter_bounding_rect.y + clutter_bounding_rect.height < cluttered_sample_image.rows)
                                    clutter_bounding_rect.height += pad;


                                cv::Point2i anchor;
                                anchor.x = max_x_div*(k+1) ;
                                anchor.y = max_y_div*(j+1);

                                cv::Rect roi_to_copy;
                                roi_to_copy.x = anchor.x  - clutter_bounding_rect.width / 2;
                                roi_to_copy.y = anchor.y  - clutter_bounding_rect.height / 2;
                                roi_to_copy.width = clutter_bounding_rect.width;
                                roi_to_copy.height = clutter_bounding_rect.height;

                                if(roi_to_copy.x < 0)
                                {
                                    //-----REDUCE THE WIDTH
                                    roi_to_copy.width += roi_to_copy.x ;

                                    clutter_bounding_rect.x -= roi_to_copy.x;
                                    clutter_bounding_rect.width = roi_to_copy.width;

                                    roi_to_copy.x = 0;

                                }

                                if(roi_to_copy.y < 0)
                                {

                                    //-----REDUCE THE HEIGHT
                                    roi_to_copy.height += roi_to_copy.y ;

                                    clutter_bounding_rect.y -= roi_to_copy.y;
                                    clutter_bounding_rect.height = roi_to_copy.height;

                                    roi_to_copy.y = 0;
                                }

                                if(roi_to_copy.x + roi_to_copy.width > cluttered_sample_image.cols)
                                {

                                    //-----REDUCE THE WIDTH
                                    roi_to_copy.width -=  roi_to_copy.x + roi_to_copy.width - cluttered_sample_image.cols;
                                    clutter_bounding_rect.width = roi_to_copy.width;
                                }

                                if(roi_to_copy.y + roi_to_copy.height > cluttered_sample_image.rows)
                                {
                                    //-----REDUCE THE HEIGHT
                                    roi_to_copy.height -=  roi_to_copy.y + roi_to_copy.height - cluttered_sample_image.rows;
                                    clutter_bounding_rect.height = roi_to_copy.height;
                                }

                                int object_pixel_counts = 0;

                                int loc_x = roi_to_copy.x;
                                int loc_y = roi_to_copy.y;



                                for(int l=clutter_bounding_rect.y;l<clutter_bounding_rect.y + clutter_bounding_rect.height;l++)
                                {
                                    for(int m=clutter_bounding_rect.x;m< clutter_bounding_rect.x +  clutter_bounding_rect.width ;m++)
                                    {
                                        unsigned char* cluttered_sample_mask_pixel = cluttered_sample_mask.data + l* cluttered_sample_mask.step[0] + m * cluttered_sample_mask.step[1];
                                        unsigned char* cluttered_sample_image_pixel = cluttered_sample_image.data + l * cluttered_sample_image.step[0] + m * cluttered_sample_image.step[1];

                                        if(cluttered_sample_mask_pixel[0] == mapped_object_id)
                                        {
                                            unsigned char* cluttered_image_pixel = cluttered_image.data + loc_y * cluttered_image.step[0] + loc_x * cluttered_image.step[1];
                                            unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + loc_y * loc_cluttered_mask.step[0] + loc_x * loc_cluttered_mask.step[1];

                                            cluttered_image_pixel[0] = cluttered_sample_image_pixel[0];
                                            cluttered_image_pixel[1] = cluttered_sample_image_pixel[1];
                                            cluttered_image_pixel[2] = cluttered_sample_image_pixel[2];

                                            //                                        cluttered_mask_pixel[0] = cluttered_sample_mask_pixel[0];
                                            //                                        cluttered_mask_pixel[1] = cluttered_sample_mask_pixel[1];
                                            //                                        cluttered_mask_pixel[2] = cluttered_sample_mask_pixel[2];

                                            loc_cluttered_mask_pixel[0] = n_clutter_filled;
                                            object_pixel_counts++;
                                        }
                                        loc_x++;
                                    }
                                    loc_y++;
                                    loc_x = roi_to_copy.x;
                                }

                                n_clutter_filled++;

                                //                            images.push_back(cluttered_sample_image);
                                labels.push_back(mapped_object_id);
                                sizes.push_back(object_pixel_counts);
                                object_rects.push_back(roi_to_copy);
                            }
                        }



                    //-----CHECKS FOR THE PERCENTAGE VISIBILTY
                    //-----IF ONLY ONE CONTOUR AND VISIBILITY IS GREATER THEN A THRESHOLD
                    //-----MASK IS KEPT INTACT
                    //-----IN CASE OF MULTIPLE CONTOURS AFTER CLUTTERING, EACH CONTOUR IS CHECKED AND
                    //-----CONTOUR IS LEFT INTACT IN CASE IT IS GREATER THEN A CERTAIN PERCENTAGE OF
                    //-----THE LARGEST CONTOUR


                    for(int j=0;j<labels.size();j++)
                    {
                        int size = sizes[j];
                        cv::Rect& object_rect = object_rects[j];

                        cv::Mat object_cluttered_mask = (loc_cluttered_mask(object_rect) == j+1);

                        int object_size = cv::countNonZero(object_cluttered_mask);
                        float visibility = (float)object_size / size;

                        if(visibility > MIN_VISIBILITY)
                        {

                            cv::Mat cropped_cluttered_mask_contours;
                            object_cluttered_mask.copyTo(cropped_cluttered_mask_contours);

                            std::vector<std::vector<cv::Point2i> > clutter_contours;
                            cv::findContours(cropped_cluttered_mask_contours,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                            if(clutter_contours.size() > 1)
                            {

                                std::vector<int> contour_areas;

                                int max_area = -10000000;

                                for(int k=0;k<clutter_contours.size();k++)
                                {
                                    int area = cv::contourArea(clutter_contours[k]);
                                    contour_areas.push_back(area);

                                    if(max_area < area)
                                        max_area = area;
                                }

                                //-----FILL THE SMALLER CONTOURS AS COMPARED TO LARGEST CONTOUR WITH BACKGROUND
                                cv::Mat cropped_cluttered_mask = loc_cluttered_mask(object_rect);

                                for(int k=0;k<clutter_contours.size();k++)
                                {
                                    float visbility_wrt_largest_contour = (float)contour_areas[k] / max_area;

                                    if(visbility_wrt_largest_contour < MIN_VISIBILITY_WRT_LARGEST_CONTOUR)
                                    {
                                        std::vector<std::vector<cv::Point> > contour;
                                        contour.push_back(clutter_contours[k]);
                                        cv::fillPoly(cropped_cluttered_mask,contour,cv::Scalar(0));
                                    }
                                }
                            }
                        }
                        else
                        {
                            for(int k=object_rect.y; k < object_rect.y + object_rect.height;k++)
                                for(int l=object_rect.x; l < object_rect.x + object_rect.width ;l++)
                                {
                                    unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + k*loc_cluttered_mask.step[0] + l* loc_cluttered_mask.step[1];

                                    if(loc_cluttered_mask_pixel[0] == j+1)
                                        loc_cluttered_mask_pixel[0] = 0;
                                }
                        }
                    }

                    cluttered_mask = cv::Mat::zeros(roi.height,roi.width,CV_8UC3);

                    for(int j=0;j<cluttered_mask.rows;j++)
                        for(int k=0;k<cluttered_mask.cols;k++)
                        {
                            unsigned char* cluttered_mask_pixel = cluttered_mask.data + j* cluttered_mask.step[0] + k* cluttered_mask.step[1];
                            unsigned char* loc_cluttered_mask_pixel = loc_cluttered_mask.data + j* loc_cluttered_mask.step[0] + k* loc_cluttered_mask.step[1];

                            int index = loc_cluttered_mask_pixel[0];

                            if(index > 0)
                            {
                                int label = labels[index-1];

                                cluttered_mask_pixel[0] = label;
                                cluttered_mask_pixel[1] = label;
                                cluttered_mask_pixel[2] = label;

                            }
                            else if(index == 0)
                            {
                                unsigned char* cluttered_image_pixel  = cluttered_image.data + j* cluttered_image.step[0] + k* cluttered_image.step[1];
                                unsigned char* loc_cluttered_image_pixel  = loc_cluttered_image.data + j* loc_cluttered_image.step[0] + k* loc_cluttered_image.step[1];

                                cluttered_image_pixel[0] = loc_cluttered_image_pixel[0];
                                cluttered_image_pixel[1] = loc_cluttered_image_pixel[1];
                                cluttered_image_pixel[2] = loc_cluttered_image_pixel[2];
                            }

                        }


                }

                t2 = (double)ros::Time::now().toNSec() / 1000.0;

                std::cout << "TIME TAKEN IN GENERATING CLUTTER = " << (t2-t1) << "  micro seconds\n";
            }



            for(int j=0;j<n_threads;j++)
            {
                cv::imshow("cluttered image",cluttered_images[j]);
                cv::imshow("cluttered mask",5*cluttered_masks[j]);
                cv::waitKey(1);
            }



            for(int j=0;j<n_threads;j++)
            {
                std::stringstream cluttered_img_file_name;
                cluttered_img_file_name << clutter_objects_dir.str() << total_files << ".png";

                std::stringstream cluttered_mask_file_name;
                cluttered_mask_file_name << clutter_annotations_dir.str() <<  "mask_" << total_files << ".png";

//                cv::imwrite(cluttered_img_file_name.str(),cluttered_images[j]);
//                cv::imwrite(cluttered_mask_file_name.str(),cluttered_masks[j]);
                total_files++;
            }
        }

        flag_data_gen_completed =  true;
        for(int i=0;i<n_threads;i++)
            pthread_join(clutter_gen_threads[i],NULL);

    }
}

