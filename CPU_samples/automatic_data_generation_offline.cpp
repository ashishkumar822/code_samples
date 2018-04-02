#include<iostream>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>

#include<opencv2/opencv.hpp>

#include<caffe/caffe.hpp>

#include<automatic_data_generation/rotating_platform.h>
#include<iitktcs_msgs_srvs/fetch_foscam_all_images.h>

#include<boost/filesystem.hpp>

#include<signal.h>

#include<stdio.h>

#include<queue>

#include<pthread.h>

//---------_FOSCAM MODELS

int NUM_OLD_OBJECTS;
int N_CAMS;
int GPU_ID_AUTOMATIC_DATA_GENERATION_PSPNET;
int GPU_ID_AUTOMATIC_DATA_GENERATION_SSD;
int N_OBJECTS;
double TIME_PER_REV_MICRO_SECONDS;
int N_IMAGES_PER_REV;
int N_BATCHES;
int LABEL_OFFSET;

std::string TASK;

std::string dataset_path;
std::string CLUTTER_TYPE;
std::vector<std::string> all_object_names;
std::vector<int> ids_competition_set;


std::string str_pspnet_net_test_proto;
std::string str_pspnet_net_pretrained;
std::string str_ssd_net_test_proto;
std::string str_ssd_net_pretrained;

std::string str_detection_roi_masks_path;
std::string dataset_images_dir_path;
std::string dataset_annotations_dir_path;
std::string str_image_channel;

std::string str_list_file;

std::string str_serial_port;

std::string str_dataset_info_file;
std::stringstream str_file_n_images_old_objects;

int CROP_SIZE;
int crop_height;
int crop_width;

int total_images;

typedef struct batch_t_
{
    cv::Mat* images;
    int label;
    bool end;
}batch_t;


class AutomaticDataGeneration
{
public:

    //-----ENUMS
    enum
    {
        BACKGROUND,
        FOREGROUND
    };

    //-----CAFFE VARS
    caffe::Net<float>* pspnet;

    std::vector<caffe::Blob<float>* > pspnet_blobs_input;
    std::vector<caffe::Blob<float>* > pspnet_blobs_output;

    caffe::Blob<float>* pspnet_input_blob;
    caffe::Blob<float>* pspnet_output_blob;

    std::vector<int> pspnet_input_blob_dims;
    std::vector<int> pspnet_output_blob_dims;

    //-----SSD NETWORK
    caffe::Net<float>* ssd;

    std::vector<caffe::Blob<float>* > ssd_blobs_input;
    std::vector<caffe::Blob<float>* > ssd_blobs_output;

    caffe::Blob<float>* ssd_input_blob;
    caffe::Blob<float>* ssd_output_blob;

    std::vector<int> ssd_input_blob_dims;
    std::vector<int> ssd_output_blob_dims;


    //-----OPENCV VARS
    cv::Rect detection_roi;
    cv::Rect* detection_roi_cams;

    cv::Mat* detection_roi_masks;


    cv::Mat input_image;
    cv::Mat output_label;
    int class_label;
    std::vector<cv::Rect> object_rois;


    int input_width;
    int input_height;
    int image_offset;
    int prev_label;
    //-----BATCHES

    int n_batches;
    std::queue<batch_t*> prefetch_free;
    std::queue<batch_t*> prefetch_full;

    pthread_mutex_t prefetch_mutex_free;
    pthread_mutex_t prefetch_mutex_full;

    pthread_cond_t prefetch_cond_free;
    pthread_cond_t prefetch_cond_full;

    pthread_t forward_pass_thread;
    pthread_t pspnet_thread;
    pthread_t ssd_thread;

    //-----ROS VARS
    iitktcs_msgs_srvs::fetch_foscam_all_images srv_fetch_foscam_image;
    ros::ServiceClient service_client_foscam;

    //----flags
    bool flag_data_gen_completed;
    bool flag_forward_passing_finished;
    bool flag_do_forward_pass_pspnet;
    bool flag_do_forward_pass_ssd;
    bool flag_update_n_images;
    bool flag_was_object;

    AutomaticDataGeneration()
    {

        flag_data_gen_completed = false;
        flag_forward_passing_finished = false;
        flag_do_forward_pass_pspnet = false;
        flag_do_forward_pass_ssd = false;
        flag_update_n_images =false;

        detection_roi_cams = new cv::Rect[N_CAMS];
        detection_roi_masks = new cv::Mat[N_CAMS];

        crop_width = CROP_SIZE;
        crop_height = CROP_SIZE;


        //-----READ DETECTION ROIS
        read_detection_roi_masks();

        //        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        //        caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION);

        //-----READ NETWORK
        read_scene_parsing_network();
        read_ssd_network();


        for(int i=0;i<N_CAMS;i++)
        {
            srv_fetch_foscam_image.request.roi.data.push_back(detection_roi_cams[i].x);
            srv_fetch_foscam_image.request.roi.data.push_back(detection_roi_cams[i].y);
            srv_fetch_foscam_image.request.roi.data.push_back(detection_roi_cams[i].width);
            srv_fetch_foscam_image.request.roi.data.push_back(detection_roi_cams[i].height);
        }

        output_label = cv::Mat::zeros(input_height,input_width,CV_8UC3);


        n_batches = N_BATCHES;

        for(int i=0;i<N_BATCHES;i++)
            for(int j=0;j<N_CAMS;j++)
            {
                batch_t* batch = new batch_t;

                batch->images = new cv::Mat[N_CAMS];
                for(int k=0; k < N_CAMS; k++)
                    batch->images[k].create(crop_height,crop_width,CV_8UC3);

                prefetch_free.push(batch);

            }


        pthread_mutex_init(&prefetch_mutex_free,NULL);
        pthread_mutex_init(&prefetch_mutex_full,NULL);
        pthread_cond_init(&prefetch_cond_free,NULL);
        pthread_cond_init(&prefetch_cond_full,NULL);

        pthread_create(&forward_pass_thread,NULL,reinterpret_cast<void* (*)(void*)>(&AutomaticDataGeneration::forward_pass),this);
        pthread_create(&pspnet_thread,NULL,reinterpret_cast<void* (*)(void*)>(&AutomaticDataGeneration::forward_pass_pspnet),this);
        pthread_create(&ssd_thread,NULL,reinterpret_cast<void* (*)(void*)>(&AutomaticDataGeneration::forward_pass_ssd),this);

    }

    ~AutomaticDataGeneration()
    {

        for(int i=0;i<N_BATCHES;i++)
        {
            batch_t* batch = prefetch_free.front();
            prefetch_free.pop();

            delete batch;
        }

        //        delete detection_roi_cams;
        //        delete detection_roi_masks;
        //        delete foscam_images;
        //        delete foscam_labels;
        //        pthread_join(forward_pass_thread,NULL);
        //                pthread_cancel(pspnet_thread);
        //        pthread_join(ssd_thread,NULL);

        delete pspnet;
    }


    void read_scene_parsing_network(void);
    void read_ssd_network(void);

    void read_detection_roi_masks(void);

    bool fetch_foscam_image(int label, bool end);
    void *forward_pass(void *args);

    void* forward_pass_pspnet(void* args);
    void* forward_pass_ssd(void* args);


};


void AutomaticDataGeneration::read_detection_roi_masks(void)
{

    for(int i=0;i<N_CAMS;i++)
    {

        std::stringstream str_detection_roi_mask_path_bin;
        str_detection_roi_mask_path_bin << str_detection_roi_masks_path << "mask_cam_" << i << ".png";

        detection_roi_masks[i] = cv::imread(str_detection_roi_mask_path_bin.str());


        cv::Mat gray_mask;
        cv::cvtColor(detection_roi_masks[i],gray_mask,CV_BGR2GRAY);

        std::vector<std::vector<cv::Point2i> > contours;
        cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        detection_roi_cams[i] =  cv::boundingRect(contours[0]);

    }




    //-----A RECTANGLE WHICH COVERS THE RECTANGLES OF STOWING ROI AND ALL BINS ROI
    //-----THIS ALL IS DONE SO THAT NETWORK INPUT CAN BE ADJUSTED BECAUSE NETWORK'S
    //-----INTERP LAYER USES INTERPAOLATION FACTOR OF 8
    int factor = 8;

    //    cv::Rect aggregate_rect;

    //    aggregate_rect = detection_roi_cams[0];

    //    for(int i=1;i<N_CAMS;i++)
    //        aggregate_rect |= detection_roi_cams[i];

    //    int remainder_width  = aggregate_rect.width % factor;
    //    int remainder_height = aggregate_rect.height % factor;

    //    int extra_width =  aggregate_rect.width  - ((int)((float)aggregate_rect.width / factor))*factor;
    //    int extra_height =  aggregate_rect.height  - ((int)((float)aggregate_rect.height / factor))*factor;

    //    std::cout << extra_width << "  " << extra_height << "\n";


    //    if(remainder_width)
    //        aggregate_rect.width = ((int)(aggregate_rect.width / factor))*factor + factor;

    //    if(remainder_height)
    //        aggregate_rect.height = ((int)(aggregate_rect.height / factor))*factor + factor;

    //    crop_width = aggregate_rect.width + 1;
    //    crop_height = aggregate_rect.height + 1;

    //    detection_roi.x = aggregate_rect.x - extra_width / 2.0f;
    //    detection_roi.y = aggregate_rect.y - extra_height / 2.0f;
    //    detection_roi.width = crop_width;
    //    detection_roi.height = crop_height;


    int remainder_width  = crop_width % factor;
    int remainder_height = crop_height % factor;

    int extra_width =  crop_width  - ((int)((float)crop_width / factor))*factor;
    int extra_height =  crop_height  - ((int)((float)crop_height / factor))*factor;

    std::cout << extra_width << "  " << extra_height << "\n";


    if(remainder_width)
        crop_width = ((int)(crop_width / factor))*factor + factor;

    if(remainder_height)
        crop_height = ((int)(crop_height / factor))*factor + factor;

    crop_width = crop_height + 1;
    crop_height = crop_height + 1;


    for(int i=0;i<N_CAMS;i++)
    {
        detection_roi_cams[i].x = detection_roi_cams[i].x + (detection_roi_cams[i].width -crop_width)/2.0;
        detection_roi_cams[i].y = detection_roi_cams[i].y + (detection_roi_cams[i].height -crop_height)/2.0;
        detection_roi_cams[i].width = crop_width;
        detection_roi_cams[i].height = crop_height;

        if(detection_roi_cams[i].x < 0)
            detection_roi_cams[i].x = 0;
        if(detection_roi_cams[i].y < 0 )
            detection_roi_cams[i].y = 0;

        detection_roi_masks[i] = detection_roi_masks[i](detection_roi_cams[i]);

    }

    input_width = crop_width;
    input_height = crop_height;
}


void AutomaticDataGeneration::read_scene_parsing_network(void)
{
    std::cout << "Readin Net\n";

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION_PSPNET);

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(str_pspnet_net_test_proto,&net_param);

    //      int input_height = 600+1;
    //      int input_width = 600+1;

    //    int input_height = 472+1;
    //    int input_width = 472+1;

    int factor = 8;

    int interp_width = (input_width-1)/factor + 1;
    int interp_height = (input_height-1)/factor + 1;

    for(int i=0;i<net_param.layer_size();i++)
    {
        if(net_param.mutable_layer(i)->type() == "Interp")
        {
            if(net_param.mutable_layer(i)->interp_param().has_height() && net_param.mutable_layer(i)->interp_param().has_height())
            {
                net_param.mutable_layer(i)->mutable_interp_param()->set_height(interp_height);
                net_param.mutable_layer(i)->mutable_interp_param()->set_width(interp_width);
            }
        }

        if(net_param.mutable_layer(i)->type() == "Input")
        {
            net_param.mutable_layer(i)->mutable_input_param()->clear_shape();
            caffe::BlobShape* input = net_param.mutable_layer(i)->mutable_input_param()->add_shape();

            input->add_dim(1);
            input->add_dim(3);
            input->add_dim(input_height);
            input->add_dim(input_width);

        }
    }

    pspnet = new caffe::Net<float>(net_param);
    pspnet->CopyTrainedLayersFrom(str_pspnet_net_pretrained);

    pspnet_blobs_input = pspnet->input_blobs();
    pspnet_blobs_output = pspnet->output_blobs();

    pspnet_input_blob = pspnet_blobs_input[0];
    pspnet_output_blob = pspnet_blobs_output[0];


    for(int i=0;i<pspnet_blobs_input[0]->shape().size();i++)
        pspnet_input_blob_dims.push_back(pspnet_blobs_input[0]->shape(i));

    std::cout << "Input Blob Dims = ";
    for(int i=0;i<pspnet_input_blob_dims.size();i++)
        std::cout << pspnet_input_blob_dims[i] << " x ";
    std::cout << std::endl;


    for(int i=0;i<pspnet_output_blob->shape().size();i++)
        pspnet_output_blob_dims.push_back(pspnet_output_blob->shape(i)) ;

    std::cout << "Output Blob Dims = ";
    for(int i=0;i<pspnet_output_blob_dims.size();i++)
        std::cout << pspnet_output_blob_dims[i] << " x ";
    std::cout << std::endl;


}

void AutomaticDataGeneration::read_ssd_network(void)
{
    std::cout << "Readin Net\n";

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION_SSD);

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(str_ssd_net_test_proto,&net_param);

    int factor = 8;

    int interp_width = (input_width-1)/factor + 1;
    int interp_height = (input_height-1)/factor + 1;

    for(int i=0;i<net_param.layer_size();i++)
    {
        if(net_param.mutable_layer(i)->type() == "Interp")
        {
            if(net_param.mutable_layer(i)->interp_param().has_height() && net_param.mutable_layer(i)->interp_param().has_height())
            {
                net_param.mutable_layer(i)->mutable_interp_param()->set_height(interp_height);
                net_param.mutable_layer(i)->mutable_interp_param()->set_width(interp_width);
            }
        }

        if(net_param.mutable_layer(i)->type() == "Input")
        {
            net_param.mutable_layer(i)->mutable_input_param()->clear_shape();
            caffe::BlobShape* input = net_param.mutable_layer(i)->mutable_input_param()->add_shape();

            input->add_dim(1);
            input->add_dim(3);
            input->add_dim(input_height);
            input->add_dim(input_width);

        }
    }

    ssd = new caffe::Net<float>(net_param);
    ssd->CopyTrainedLayersFrom(str_ssd_net_pretrained);



    ssd_blobs_input = ssd->input_blobs();
    ssd_blobs_output = ssd->output_blobs();

    ssd_input_blob = ssd_blobs_input[0];
    ssd_output_blob = ssd_blobs_output[0];


    for(int i=0;i<ssd_blobs_input[0]->shape().size();i++)
        ssd_input_blob_dims.push_back(ssd_blobs_input[0]->shape(i));

    std::cout << "Input Blob Dims = ";
    for(int i=0;i<ssd_input_blob_dims.size();i++)
        std::cout << ssd_input_blob_dims[i] << " x ";
    std::cout << std::endl;


    for(int i=0;i<ssd_output_blob->shape().size();i++)
        ssd_output_blob_dims.push_back(ssd_output_blob->shape(i)) ;

    std::cout << "Output Blob Dims = ";
    for(int i=0;i<ssd_output_blob_dims.size();i++)
        std::cout << ssd_output_blob_dims[i] << " x ";
    std::cout << std::endl;


}



bool AutomaticDataGeneration::fetch_foscam_image(int label,bool end)
{

    batch_t* batch = NULL;

    pthread_mutex_lock(&prefetch_mutex_free);
    {
        if(prefetch_free.empty())
            pthread_cond_wait(&prefetch_cond_free, &prefetch_mutex_free);

        batch = prefetch_free.front();
        prefetch_free.pop();
    }
    pthread_mutex_unlock(&prefetch_mutex_free);

    if(!end)
    {
        if(!this->service_client_foscam.call(srv_fetch_foscam_image))
            return false;

        for(int i=0;i<N_CAMS;i++)
        {
            cv_bridge::CvImagePtr m_img_ptr = cv_bridge::toCvCopy(srv_fetch_foscam_image.response.images[i]);
            cv::Mat image = m_img_ptr->image;
            image.copyTo(batch->images[i]);
        }
    }

    batch->label = label;
    batch->end = end;

    pthread_mutex_lock(&prefetch_mutex_full);
    {
        prefetch_full.push(batch);
        pthread_cond_signal(&prefetch_cond_full);
    }
    pthread_mutex_unlock(&prefetch_mutex_full);

    return  true;
}

void* AutomaticDataGeneration::forward_pass_pspnet(void* args)
{

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION_PSPNET);

    while(!flag_forward_passing_finished)
    {
        usleep(10000);

        if(flag_do_forward_pass_pspnet)
        {
            float* net_data_input = ((caffe::Blob<float>*)(this->pspnet_input_blob))->mutable_cpu_data();

            int channel_size = this->pspnet_input_blob_dims[2] * this->pspnet_input_blob_dims[3];

            float* net_data_input_ch1 = net_data_input;
            float* net_data_input_ch2 = net_data_input + channel_size;
            float* net_data_input_ch3 = net_data_input + channel_size * 2;

            for(int i=0;i< input_height;i++)
                for(int j=0;j< input_width;j++)
                {
                    unsigned char* pixel = input_image.data + i * input_image.step[0] + j * input_image.step[1];

                    (net_data_input_ch1 + i * input_width)[j] = (float)pixel[0];
                    (net_data_input_ch2 + i * input_width)[j] = (float)pixel[1];
                    (net_data_input_ch3 + i * input_width)[j] = (float)pixel[2];

                }

            pspnet->ForwardFrom(0);

            float* net_data_output_foreground = pspnet_output_blob->mutable_cpu_data() + channel_size * FOREGROUND;

            memset(output_label.data,0,channel_size*3);

            for(int i=0;i<output_label.rows;i++)
                for(int j=0;j<output_label.cols;j++)
                {
                    float* prob_pixel = net_data_output_foreground + i* output_label.cols + j;

                    if(prob_pixel[0] > 0.99)
                    {
                        unsigned char*  pixel_output_label   = output_label.data + i* output_label.step[0] + j* output_label.step[1];

                        pixel_output_label[0] = class_label;
                        pixel_output_label[1] = class_label;
                        pixel_output_label[2] = class_label;
                    }
                }


            flag_do_forward_pass_pspnet = false;
        }
    }
}

void* AutomaticDataGeneration::forward_pass_ssd(void* args)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION_SSD);

    while(!flag_forward_passing_finished)
    {
        usleep(10000);

        if(flag_do_forward_pass_ssd)
        {

            float* net_data_input = ((caffe::Blob<float>*)(this->ssd_input_blob))->mutable_cpu_data();

            int channel_size = this->ssd_input_blob_dims[2] * this->ssd_input_blob_dims[3];

            float* net_data_input_ch1 = net_data_input;
            float* net_data_input_ch2 = net_data_input + channel_size;
            float* net_data_input_ch3 = net_data_input + channel_size * 2;

            for(int i=0;i< input_height;i++)
                for(int j=0;j< input_width;j++)
                {
                    unsigned char* pixel = input_image.data + i * input_image.step[0] + j * input_image.step[1];

                    (net_data_input_ch1 + i * input_width)[j] = (float)pixel[0];
                    (net_data_input_ch2 + i * input_width)[j] = (float)pixel[1];
                    (net_data_input_ch3 + i * input_width)[j] = (float)pixel[2];

                }

            ssd->ForwardFrom(0);

            float* net_data_output = ssd_output_blob->mutable_cpu_data();

            int num_detections = ssd_output_blob->height();

            int height = input_height;
            int width = input_width;

            //    const int label_idx = 1;
            const int score_idx = 2;
            const int xmin_idx = 3;
            const int ymin_idx = 4;
            const int xmax_idx = 5;
            const int ymax_idx = 6;

            int max_index = 0;
            float max_score = 0;

            for(int i=0;i<num_detections;i++)
            {
                //        if(net_data_output[i*7+score_idx] > 0.950)
                //        {
                //                labeled_roi labeled_roi_obj;
                //                labeled_roi_obj.label = net_data_output[i*7+label_idx];
                //                labeled_roi_obj.score = net_data_output[i*7+score_idx];
                //                labeled_roi_obj.roi.x = width * net_data_output[i*7 + xmin_idx];
                //                labeled_roi_obj.roi.y = height * net_data_output[i*7 + ymin_idx];
                //                labeled_roi_obj.roi.width = width * ( net_data_output[i*7 + xmax_idx] - net_data_output[i*7 + xmin_idx]);
                //                labeled_roi_obj.roi.height = height * ( net_data_output[i*7 + ymax_idx] - net_data_output[i*7 + ymin_idx]);
                //                labeled_rois.push_back(labeled_roi_obj);
                //        }

                if(max_score <  net_data_output[i*7+score_idx])
                {
                    max_score =  net_data_output[i*7+score_idx];
                    max_index = i;
                }
            }

            object_rois.clear();

            if(max_score >  0.90f)
            {
                cv::Rect roi_obj;
                //        roi_obj.label = net_data_output[max_index*7+label_idx];
                //        roi_obj.score = net_data_output[max_index*7+score_idx];
                roi_obj.x = width * net_data_output[max_index*7 + xmin_idx];
                roi_obj.y = height * net_data_output[max_index*7 + ymin_idx];
                roi_obj.width = width * ( net_data_output[max_index*7 + xmax_idx] - net_data_output[max_index*7 + xmin_idx]);
                roi_obj.height = height * ( net_data_output[max_index*7 + ymax_idx] - net_data_output[max_index*7 + ymin_idx]);

                object_rois.push_back(roi_obj);
            }

            flag_do_forward_pass_ssd = false;
        }
    }
}

void* AutomaticDataGeneration::forward_pass(void* args)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_AUTOMATIC_DATA_GENERATION_SSD);

    int image_no = total_images;

    int images_per_object = 0;
    bool continue_forward_pass = true;

    while(continue_forward_pass)
    {

        batch_t* batch;

        pthread_mutex_lock(&prefetch_mutex_full);
        {
            if(!flag_data_gen_completed && prefetch_full.empty())
                pthread_cond_wait(&prefetch_cond_full,&prefetch_mutex_full);

            if(flag_data_gen_completed && prefetch_full.empty())
                continue_forward_pass = false;
            else
            {
                batch = prefetch_full.front();
                prefetch_full.pop();
            }
        }
        pthread_mutex_unlock(&prefetch_mutex_full);

        if(batch && batch->end)
        {
            cv::FileStorage file_dataset_info(str_dataset_info_file,cv::FileStorage::APPEND);

            std::stringstream n_images_obj;
            n_images_obj << "n_images_" << batch->label;

            file_dataset_info << n_images_obj.str() << images_per_object;

            images_per_object = 0;
            file_dataset_info.release();
        }

        if(!continue_forward_pass)
            break;

        if(!batch->end)
        {
            for(int cam_no = 0; cam_no < N_CAMS; cam_no++)
            {
                cv::Mat& image = batch->images[cam_no];
                cv::Mat& detection_mask = detection_roi_masks[cam_no];

                cv::multiply(image,detection_mask,input_image,1.0/255);

                class_label = batch->label;

                flag_do_forward_pass_ssd = true;
                flag_do_forward_pass_pspnet = true;

                while(flag_do_forward_pass_ssd || flag_do_forward_pass_pspnet )
                    usleep(10000);

                if(object_rois.size())
                {
                    std::cout << "OBJECT ROI = " << object_rois[0] << "\n";

                    if(object_rois[0].x < input_width && object_rois[0].y < input_height)
                    {
                        cv::Mat object_mask = cv::Mat::zeros(input_height,input_width,CV_8UC3);

                        if(object_rois[0].x < 0)
                            object_rois[0].x = 0;

                        if(object_rois[0].y < 0)
                            object_rois[0].y = 0;

                        if(object_rois[0].x + object_rois[0].width > input_width)
                            object_rois[0].width -= object_rois[0].x + object_rois[0].width - input_width;

                        if(object_rois[0].y + object_rois[0].height > input_height)
                            object_rois[0].height -= object_rois[0].y + object_rois[0].height - input_height;


                        cv::Mat cropped_roi = object_mask(object_rois[0]);

                        for(int i=0;i<cropped_roi.rows;i++)
                            for(int j=0;j<cropped_roi.cols;j++)
                            {
                                unsigned char* pixel = cropped_roi.data +cropped_roi.step[0] *i + cropped_roi.step[1]*j;
                                pixel[0] = 1;
                                pixel[1] = 1;
                                pixel[2] = 1;
                            }

                        cv::multiply(object_mask,output_label,output_label);

                        std::stringstream str_image_file_name;
                        std::stringstream str_annotated_file_name;

                        str_image_file_name << dataset_images_dir_path << image_no << ".png";
                        str_annotated_file_name << dataset_annotations_dir_path << "mask_" << image_no << ".png";

                        cv::imwrite(str_image_file_name.str(),image(cv::Rect(0,0,CROP_SIZE,CROP_SIZE)));
                        cv::imwrite(str_annotated_file_name.str(),output_label(cv::Rect(0,0,CROP_SIZE,CROP_SIZE)));

                        FILE* list_file = fopen(str_list_file.c_str(),"a");

                        std::stringstream list_img_file_name;
                        list_img_file_name << "/objects/" << image_no << ".png";

                        std::stringstream list_mask_file_name;
                        list_mask_file_name << "/annotations/mask_" <<  image_no << ".png";

                        fprintf(list_file,"%s %s\n",list_img_file_name.str().c_str(),list_mask_file_name.str().c_str());

                        fclose(list_file);

                        image_no++;
                        images_per_object++;

                        float rotation_degree = 40;

                        for(int k=0;k<2;k++)
                        {
                            cv::Mat rotated_image;
                            cv::Mat rotated_mask;

                            cv::Mat r = cv::getRotationMatrix2D(cv::Point2i(CROP_SIZE/2.0f,CROP_SIZE/2.0f), rotation_degree, 1.0);
                            cv::warpAffine(image(cv::Rect(0,0,CROP_SIZE,CROP_SIZE)), rotated_image, r, cv::Size(CROP_SIZE, CROP_SIZE), cv::INTER_LINEAR);
                            cv::warpAffine(output_label(cv::Rect(0,0,CROP_SIZE,CROP_SIZE)), rotated_mask, r, cv::Size(CROP_SIZE, CROP_SIZE), cv::INTER_NEAREST);

                            std::stringstream str_rotated_image_file_name;
                            std::stringstream str_rotated_annotated_file_name;

                            str_rotated_image_file_name << dataset_images_dir_path << image_no << ".png";
                            str_rotated_annotated_file_name << dataset_annotations_dir_path << "mask_" << image_no << ".png";

                            cv::imwrite(str_rotated_image_file_name.str(),rotated_image);
                            cv::imwrite(str_rotated_annotated_file_name.str(),rotated_mask);

                            FILE* rotated_list_file = fopen(str_list_file.c_str(),"a");

                            std::stringstream list_rotated_img_file_name;
                            list_rotated_img_file_name << "/objects/" << image_no << ".png";

                            std::stringstream list_rotated_mask_file_name;
                            list_rotated_mask_file_name << "/annotations/mask_" <<  image_no << ".png";

                            fprintf(rotated_list_file,"%s %s\n",list_rotated_img_file_name.str().c_str(),list_rotated_mask_file_name.str().c_str());

                            fclose(rotated_list_file);

                            rotation_degree = 80.0f;

                            image_no++;
                            images_per_object++;
                        }

                        cv::rectangle(image,object_rois[0],cv::Scalar(0,255,0),1);

                        cv::imshow("input_image", image);
                        cv::imshow("output_label",output_label);
                        cv::waitKey(1);
                    }
                }

            }
        }

        pthread_mutex_lock(&prefetch_mutex_free);
        {
            batch->end = false;
            prefetch_free.push(batch);
            pthread_cond_signal(&prefetch_cond_free);
        }
        pthread_mutex_unlock(&prefetch_mutex_free);
    }


    flag_forward_passing_finished = true;

    std::cout << "EXITING FORWARD PASS THREAD\n";
}

void callback_signal(int signal_id)
{
    std::cout << "CAUGHT SIGNAL...EXITING\n";
    exit(0);
}



bool flag_get_keyboard_val = false;
int keyboard_val = -1;

void* get_input_from_keyboard(void* args)
{
    while(1)
    {
        usleep(10000);

        if(flag_get_keyboard_val)
        {
            int a;
            std::cin >> a;
            keyboard_val = a;
            flag_get_keyboard_val = false;
        }
    }
}

int main(int argc, char** argv)
{

    ros::init(argc,argv,"automatic_data_generation");

    ros::NodeHandle nh;

    nh.getParam("/ARC17_GPU_ID_AUTOMATIC_DATA_GENERATION_PSPNET", GPU_ID_AUTOMATIC_DATA_GENERATION_PSPNET);
    nh.getParam("/ARC17_GPU_ID_AUTOMATIC_DATA_GENERATION_SSD", GPU_ID_AUTOMATIC_DATA_GENERATION_SSD);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_SERIAL_PORT", str_serial_port);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_NET_PSPNET_TEST_PROTO", str_pspnet_net_test_proto);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_NET_PSPNET_PRETRAINED", str_pspnet_net_pretrained);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_NET_SSD_TEST_PROTO", str_ssd_net_test_proto);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_NET_SSD_PRETRAINED", str_ssd_net_pretrained);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_ROI_MASKS_PATH", str_detection_roi_masks_path);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_N_CAMS", N_CAMS);
    nh.getParam("/ARC17_TASK", TASK);
    nh.getParam("/ARC17_TIME_PER_REV_MICRO_SECONDS", TIME_PER_REV_MICRO_SECONDS);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_IMAGES_PER_REV", N_IMAGES_PER_REV);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_IMAGE_CHANNEL", str_image_channel);
    nh.getParam("/ARC17_AUTO_DATA_GENERATION_N_BATCHES", N_BATCHES);
    nh.getParam("/ARC17_LABEL_OFFSET",LABEL_OFFSET);
    nh.getParam("/ARC17_CROP_SIZE",CROP_SIZE);

    nh.getParam("/ARC17_NUM_OLD_OBJECTS", NUM_OLD_OBJECTS);
    nh.getParam("/ARC17_DATASET_PATH",dataset_path);
    nh.getParam("/ARC17_OBJECT_NAMES",all_object_names);
    nh.getParam("/ARC17_COMPETITION_SET",ids_competition_set);
    nh.getParam("/ARC17_CLUTTER_TYPE",CLUTTER_TYPE);


    N_OBJECTS = ids_competition_set.size() / 2;

    std::string str_task_dir;

    if(TASK == "STOW")
        str_task_dir = "objects_dataset_stowing/";
    if(TASK == "PICK")
        str_task_dir = "objects_dataset_picking/";
    if(TASK == "STOW-PICK")
        str_task_dir = "objects_dataset_stowing_picking/";

    dataset_images_dir_path  = dataset_path + str_task_dir +  "/objects/";
    dataset_annotations_dir_path = dataset_path + str_task_dir + "/annotations/";

    boost::filesystem::create_directories(dataset_images_dir_path);
    boost::filesystem::create_directories(dataset_annotations_dir_path);

    str_file_n_images_old_objects << dataset_path << "old_objects_n_files.yaml";


    str_dataset_info_file = dataset_path + str_task_dir + "dataset_info.yaml";
    str_list_file = dataset_path + str_task_dir + "list.txt";

    cv::Mat id_mapping(N_OBJECTS*2,2,CV_8UC1);

    for(int i=0;i<id_mapping.rows;i++)
    {
        int actual_id = ids_competition_set[i];
        int mapped_id = i+1;

        id_mapping.at<unsigned char>(i,0) = actual_id;
        id_mapping.at<unsigned char>(i,1) = mapped_id;
    }

    cv::FileStorage file_dataset_info(str_dataset_info_file,cv::FileStorage::WRITE);

    int n_threads;
    int n_images_per_thread;

    nh.getParam("/ARC17_CLUTTER_GEN_N_THREADS",n_threads);
    nh.getParam("/ARC17_CLUTTER_GEN_N_IMAGES_PER_THREAD",n_images_per_thread);

    file_dataset_info << "n_threads" << n_threads;
    file_dataset_info << "n_images_per_thread" << n_images_per_thread;
    file_dataset_info << "NUM_OBJECTS" << N_OBJECTS*2;
    file_dataset_info << "CROP_SIZE" << CROP_SIZE;
    file_dataset_info << "ID_MAPPING" << id_mapping;

    file_dataset_info.release();

    total_images = 1;

    if(CLUTTER_TYPE == "COMPETITION_SET")
    {

        std::stringstream old_objects_annotations_dir;
        old_objects_annotations_dir << dataset_path <<  "/annotations/";

        std::stringstream old_objects_cropped_objects_dir;
        old_objects_cropped_objects_dir << dataset_path << "/objects/";

        //----READ TOTAL FILES IN ORDER TO CREATE SYMLINKS
        cv::FileStorage file_n_images_old_objects(str_file_n_images_old_objects.str(),cv::FileStorage::READ);

        //-----READIN ALL IMAGE NUMBERS OF OLD OBJECTS
        std::vector<int> n_images(NUM_OLD_OBJECTS);

        for(int i = 0; i < NUM_OLD_OBJECTS;i++)
        {
            int index =  i + 1;
            std::stringstream n_images_obj;
            n_images_obj << "n_images_" << index;

            file_n_images_old_objects[n_images_obj.str()] >> n_images[i];
        }

        file_n_images_old_objects.release();

        cv::FileStorage file_dataset_info(str_dataset_info_file,cv::FileStorage::APPEND);


        FILE* list_file = fopen(str_list_file.c_str(),"w");

        //-----READING NUMBER OF IMAGES OF ALL KNOWN IMAGES
        int NUM_OBJECTS = ids_competition_set.size();

        for(int i = 0; i < NUM_OBJECTS /2;i++)
        {
            int index =  ids_competition_set[i] - 1;
            std::stringstream n_images_obj;
            n_images_obj << "n_images_" << index + 1;

            file_dataset_info << n_images_obj.str() << n_images[index];

            //-----COMPUTE OFFSET and ID MAPPING TO IMAGES FILES FOR EACH OBJECT
            long int offset = 0;
            for(int j=0;j<index;j++)
                offset += n_images[j];

            for(int j=0;j<n_images[i];j++)
            {
                std::stringstream source_image_symlink;
                std::stringstream source_mask_symlink;

                source_image_symlink << old_objects_cropped_objects_dir.str() << offset+j+1 << ".png";
                source_mask_symlink << old_objects_annotations_dir.str() << "mask_" << offset+j+1 << ".png";

                std::stringstream target_image_symlink;
                std::stringstream target_mask_symlink;

                target_image_symlink << dataset_images_dir_path << total_images << ".png";
                target_mask_symlink << dataset_annotations_dir_path << "mask_" << total_images << ".png";


                boost::filesystem::create_symlink(source_image_symlink.str() , target_image_symlink.str());
                boost::filesystem::create_symlink(source_mask_symlink.str() , target_mask_symlink.str());

                std::stringstream list_img_file_name;
                list_img_file_name << "/objects/" << total_images << ".png";

                std::stringstream list_mask_file_name;
                list_mask_file_name << "/annotations/mask_" <<  total_images << ".png";

                fprintf(list_file,"%s %s\n",list_img_file_name.str().c_str(),list_mask_file_name.str().c_str());

                total_images++;
            }

        }

        file_dataset_info.release();
        fclose(list_file);

    }

    RotatingPlatform rotating_platform(str_serial_port);
    //SET SPEED
    rotating_platform.stop();
    rotating_platform.set_speed(2);

    //ALIGN TO SENSOR

    std::cout << "ALIGNING PLATFORM TO ZERO POSITION...\n";
    rotating_platform.rotate_motor_cw(1);
    rotating_platform.wait_for_rotation_completion();
    std::cout << "DONE\n";

    usleep(10000);

    AutomaticDataGeneration automatic_data_generation;

    automatic_data_generation.service_client_foscam = nh.serviceClient<iitktcs_msgs_srvs::fetch_foscam_all_images>(str_image_channel);


    pthread_t get_input_thread;
    pthread_create(&get_input_thread,NULL,get_input_from_keyboard,NULL);

    bool continue_with_same_object;

    for(int i=0;i<N_OBJECTS;i++)
    {

        std::cout << "*****-----PLACE THE OBJECT AND ENTER ANY NUMBER-----*****\n";
        std::flush(std::cout);

        flag_get_keyboard_val = true;
        while(flag_get_keyboard_val)
            usleep(10000);

        int input = keyboard_val;

        input = -1;

        int object_label = LABEL_OFFSET+i+1;

        continue_with_same_object = true;

        while(continue_with_same_object)
        {
            std::cout << "*****-----CASE 0....NEXT OBJECT-----*****\n";
            std::cout << "*****-----CASE 1....SAME OBJECT-----*****\n";

            //-----ADD BUTTON TO THE IMAGE WINDOW
            flag_get_keyboard_val = true;
            while(flag_get_keyboard_val)
                usleep(10000);

            int input = keyboard_val;

            //-----GET FEED FROM BUTTON

            if(input == 0)
                continue_with_same_object = false;

            if(continue_with_same_object)
            {
                //-----CONTINUOUS ROTATION OF PLATFORM
                rotating_platform.rotate_motor_cw(0);

                for(int j=0;j < N_IMAGES_PER_REV;j++)
                {
                    automatic_data_generation.fetch_foscam_image(object_label,false);

                    usleep(TIME_PER_REV_MICRO_SECONDS/(double)N_IMAGES_PER_REV);
                }
                rotating_platform.stop();
            }
            else
                //-----PUSH A DUMMY BATCH TO INDICATE END OF BATCHES FOR AN OBJECT
                automatic_data_generation.fetch_foscam_image(object_label,true);
        }

    }

    automatic_data_generation.flag_data_gen_completed = true;

    //-----TO PREVENT INFINTE WAITING OF FORWARD PASS THREAD
    pthread_mutex_lock(&automatic_data_generation.prefetch_mutex_full);
    {
        pthread_cond_signal(&automatic_data_generation.prefetch_cond_full);
    }
    pthread_mutex_unlock(&automatic_data_generation.prefetch_mutex_full);

    std::cout << "JOINING ALL THREADS\n";

    pthread_join(automatic_data_generation.forward_pass_thread,NULL);
    pthread_join(automatic_data_generation.pspnet_thread,NULL);
    pthread_join(automatic_data_generation.ssd_thread,NULL);

    pthread_cancel(get_input_thread);

    std::cout << "EXITING SUCCESFULLY\n";
}
