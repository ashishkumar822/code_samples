#include<caffe/caffe.hpp>
#include<caffe/util/upgrade_proto.hpp>
#include<caffe/common.hpp>
#include<opencv2/opencv.hpp>

#include<arc17_computer_vision/arc17_computer_vision.h>
#include<arc17_computer_vision/object_information.h>


//---------_FOSCAM

//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/vgg16_pspnet_test_with_bn.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/vgg16_foscam/vgg16_with_data_aug_foscam_iter_1400.caffemodel";


//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/pspnet18_ADE20K_473.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_18_foscam/pspnet_18_with_data_aug_foscam_iter_800.caffemodel";

//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/pspnet18_ADE20K_473.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_18_foscam/resnet_18_with_data_aug_foscam_iter_1000.caffemodel";


//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/pspnet50_ADE20K_473.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_50_foscam/pspnet_50_with_data_aug_foscam_iter_150000.caffemodel";

//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/pspnet50_FPN_test.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_50_FPN_foscam/pspnet_50_FPN_foscam_iter_1500.caffemodel";


//#define USE_CRF
//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/vgg16_pspnet_test_with_bn_dense_crf.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/vgg16_foscam_with_bn/vgg16_foscam_with_bn_iter_1000.caffemodel";

//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/vgg16_pspnet_test_with_bn.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/vgg16_foscam_with_bn/vgg16_foscam_with_bn_iter_1000.caffemodel";


//std::string str_net_test_proto = path_prefix + "prototxt_testing_modified/vgg16_pspnet_test_with_bn.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/vgg16_foscam_with_bn/vgg16_foscam_with_bn_iter_2000.caffemodel";

//std::string str_net_test_proto = path_prefix + "/prototxt_testing_modified/pspnet18_ADE20K_473_background_seg.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_18_foscam_background_seg/pspnet_18_foscam_background_seg__iter_30000.caffemodel";

//std::string str_net_test_proto = path_prefix + "/prototxt_testing_modified/pspnet50_train_background_seg_mpi_gather.prototxt";
//std::string str_net_pretrained = path_prefix + "trained_models/pspnet_50_foscam_background_seg/pspnet_50_foscam_background_seg__iter_50000.caffemodel";


//---- MASKS

std::string str_detection_roi_masks_path = "/home/isl-server/ashish/workspace_ros/data/detection_masks/";

int crop_width;
int crop_height;

void arc17_computer_vision_t::read_detection_roi_masks(void)
{
    ROS_INFO("IN READ DETECTION ROI MASKS");

    foscam_image = cv::Mat::zeros(FOSCAM_HEIGHT,FOSCAM_WIDTH,CV_8UC3);
    foscam_color_labels = cv::Mat::zeros(FOSCAM_HEIGHT,FOSCAM_WIDTH,CV_8UC3);
    foscam_labels = cv::Mat::zeros(FOSCAM_HEIGHT,FOSCAM_WIDTH,CV_8UC1);
    detection_roi_mask = cv::Mat::zeros(FOSCAM_HEIGHT,FOSCAM_WIDTH,CV_8UC1);

    for(int i=0;i<NUM_OBJECTS;i++)
    {
        object_masks.push_back(cv::Mat::zeros(FOSCAM_HEIGHT,FOSCAM_WIDTH,CV_8UC1));
        object_rects.push_back(std::vector<cv::Rect>());
        object_rotated_rects.push_back(std::vector<cv::RotatedRect>());
    }

    if(TASK == "STOW")
    {
        detection_roi_mask_stowing = cv::imread(str_detection_roi_masks_path + "detection_roi_mask_stowing.png");

        cv::Mat gray_mask;
        cv::cvtColor(detection_roi_mask_stowing,gray_mask,CV_BGR2GRAY);

        gray_mask.copyTo(detection_roi_mask);

        std::vector<std::vector<cv::Point2i> > contours;
        cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        detection_roi_stowing =  cv::boundingRect(contours[0]);

    }
    else if(TASK == "PICK")
    {
        for(int i=0;i< N_BINS;i++)
        {
            std::stringstream str_detection_roi_mask_path_bin;
            str_detection_roi_mask_path_bin << str_detection_roi_masks_path << "detection_roi_mask_picking_bin_" << i << ".png";

            detection_roi_mask_picking[i] = cv::imread(str_detection_roi_mask_path_bin.str());

            cv::Mat gray_mask;
            cv::cvtColor(detection_roi_mask_picking[i],gray_mask,CV_BGR2GRAY);

            cv::bitwise_or(gray_mask,detection_roi_mask,detection_roi_mask);

            std::vector<std::vector<cv::Point2i> > contours;
            cv::findContours(gray_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

            detection_roi_picking[i] =  cv::boundingRect(contours[0]);

        }
    }

    //-----A RECTANGLE WHICH COVERS THE RECTANGLES OF ALL BINS ROI
    //-----THIS ALL IS DONE SO THAT NETWORK INPUT CAN BE ADJUSTED BECAUSE NETWORK'S
    //-----INTERP LAYER USES INTERPAOLATION FACTOR OF 8
    int factor = 8;

    cv::Rect aggregate_rect;

    if(TASK == "STOW")
    {
        aggregate_rect = detection_roi_stowing;
    }
    else if(TASK == "PICK")
    {
        aggregate_rect = detection_roi_picking[0];

        for(int i=1;i <N_BINS;i++)
            aggregate_rect |= detection_roi_picking[i];

    }

    int remainder_width  = aggregate_rect.width % factor;
    int remainder_height = aggregate_rect.height % factor;

    int extra_width =  aggregate_rect.width  - ((int)((float)aggregate_rect.width / factor))*factor;
    int extra_height =  aggregate_rect.height  - ((int)((float)aggregate_rect.height / factor))*factor;

    if(remainder_width)
        aggregate_rect.width = ((int)(aggregate_rect.width / factor))*factor + factor;

    if(remainder_height)
        aggregate_rect.height = ((int)(aggregate_rect.height / factor))*factor + factor;

    crop_width = aggregate_rect.width + 1;
    crop_height = aggregate_rect.height + 1;

    //    crop_width = 512;
    //    crop_height = 512;


    detection_roi.x = aggregate_rect.x - extra_width / 2.0f;
    detection_roi.y = aggregate_rect.y - extra_height / 2.0f;
    detection_roi.width = crop_width;
    detection_roi.height = crop_height;

    detection_roi_to_display = detection_roi;

    std::cout << "DETECTION_ROI" << detection_roi << std::endl;
    ROS_INFO("OUT READ DETECTION ROI MASKS");
}

void arc17_computer_vision_t::read_network(void)
{
    ROS_INFO("IN READ NETWORK");

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_SCENE_PARSING);

    std::cout << "Readin Net\n";

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(str_net_test_proto,&net_param);

    int input_height = crop_height;
    int input_width = crop_width;


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

    net = new caffe::Net<float>(net_param);
    net->CopyTrainedLayersFrom(str_net_pretrained);


    net_blobs_input = net->input_blobs();
    net_blobs_output = net->output_blobs();

    net_input_blob = net_blobs_input[0];
    net_output_blob = net_blobs_output[0];


    for(int i=0;i<net_blobs_input[0]->shape().size();i++)
        net_input_blob_dims.push_back(net_blobs_input[0]->shape(i));


    std::cout << "Input Blob Dims = ";
    for(int i=0;i<net_input_blob_dims.size();i++)
        std::cout << net_input_blob_dims[i] << " x ";
    std::cout << std::endl;


    for(int i=0;i<net_output_blob->shape().size();i++)
        net_output_blob_dims.push_back(net_output_blob->shape(i)) ;

    std::cout << "Output Blob Dims = ";
    for(int i=0;i<net_output_blob_dims.size();i++)
        std::cout << net_output_blob_dims[i] << " x ";
    std::cout << std::endl;

    caffe::NetParameter crf_net_param;
    caffe::ReadNetParamsFromTextFileOrDie(str_crf_net_test_proto,&crf_net_param);

    for(int i=0;i<crf_net_param.layer_size();i++)
    {

        if(crf_net_param.mutable_layer(i)->type() == "Input")
        {

            crf_net_param.mutable_layer(i)->mutable_input_param()->clear_shape();
            caffe::BlobShape* blob_softmax = crf_net_param.mutable_layer(i)->mutable_input_param()->add_shape();

            blob_softmax->add_dim(1);
            blob_softmax->add_dim(NUM_OBJECTS+1);
            blob_softmax->add_dim(input_height);
            blob_softmax->add_dim(input_width);

            caffe::BlobShape* blob_img_dim = crf_net_param.mutable_layer(i)->mutable_input_param()->add_shape();

            blob_img_dim->add_dim(1);
            blob_img_dim->add_dim(1);
            blob_img_dim->add_dim(1);
            blob_img_dim->add_dim(2);

            caffe::BlobShape* blob_data = crf_net_param.mutable_layer(i)->mutable_input_param()->add_shape();

            blob_data->add_dim(1);
            blob_data->add_dim(3);
            blob_data->add_dim(input_height);
            blob_data->add_dim(input_width);

        }
    }

    crf_net = new caffe::Net<float>(crf_net_param);

    ROS_INFO("OUT READ NETWORK");

}

void arc17_computer_vision_t::merge_decisions(void)
{
    ROS_INFO("IN MERGE DECISIONS");

    cv::Mat colorized_prob_image = cv::Mat::zeros(detection_roi.height,detection_roi.width,CV_8UC3);
    cv::Mat op_labeled_image = foscam_labels(detection_roi);

    this->foscam_image(detection_roi).copyTo(foscam_color_labels(detection_roi));
    cv::cvtColor(colorized_prob_image,colorized_prob_image,CV_BGR2HSV);

    std::vector<cv::Mat>& obj_masks = this->object_masks;
    for(int i=0;i<obj_masks.size();i++)
        obj_masks[i](detection_roi) = cv::Mat::zeros(detection_roi.height, detection_roi.width ,CV_8UC1);

    //    op_labeled_image = 0;

    cv::Mat detection_mask = detection_roi_mask(detection_roi);

    for(int i=0;i<colorized_prob_image.rows;i++)
        for(int j=0;j<colorized_prob_image.cols;j++)
        {
            unsigned char* detection_mask_pixel = detection_mask.data + i* detection_mask.step[0] + j* detection_mask.step[1];
            unsigned char* color_pixel = colorized_prob_image.data + i * colorized_prob_image.step[0] + j * colorized_prob_image.step[1];

            if(detection_mask_pixel[0])
            {

                float max = -1000000000.0;
                int index =0;

                float* prob_pixel = net_data_output_all + i* colorized_prob_image.cols + j;
                int plane_offset = colorized_prob_image.cols * colorized_prob_image.rows;

                for(int k=0;k<net_output_blob_dims[1];k++)
                {
                    if((prob_pixel + plane_offset*k)[0] >  max)
                    {
                        max = (prob_pixel + plane_offset*k)[0];
                        index = k;
                    }
                }


                if(index && max > CONFIDENCE_THRESHOLD)
                {
                    int h;
                    int s;
                    int v;

                    h = (180.0*index) / NUM_OBJECTS;
                    s = 100  + (150.0*index) / NUM_OBJECTS;
                    v = 100  + (150.0*index) / NUM_OBJECTS;

                    color_pixel[0] = h;//alpha*color_pixel[0] + (1-alpha)*h;
                    color_pixel[1] = s;//alpha*color_pixel[1] + (1-alpha)*s;
                    color_pixel[2] = v;//alpha*color_pixel[2] + (1-alpha)*v;


                    unsigned char*  op_labeled_image_pixel   = op_labeled_image.data + i* op_labeled_image.step[0] + j* op_labeled_image.step[1];

                    op_labeled_image_pixel[0] = index;

                    unsigned char*  object_mask_pixel   = obj_masks[index-1](detection_roi).data + i* obj_masks[index-1](detection_roi).step[0] + j* obj_masks[index-1](detection_roi).step[1];

                    object_mask_pixel[0] = 255;
                }
                else
                {
                    unsigned char*  op_labeled_image_pixel   = op_labeled_image.data + i* op_labeled_image.step[0] + j* op_labeled_image.step[1];

                    op_labeled_image_pixel[0] = 0;
                }
            }
        }

    cv::cvtColor(colorized_prob_image,colorized_prob_image,CV_HSV2BGR);

    cv::Mat cropped_foscam_color_labels = foscam_color_labels(detection_roi);

    for(int i=0;i<colorized_prob_image.rows;i++)
        for(int j=0;j<colorized_prob_image.cols;j++)
        {
            unsigned char* detection_mask_pixel = detection_mask.data + i* detection_mask.step[0] + j* detection_mask.step[1];

            if(detection_mask_pixel[0])
            {
                unsigned char* color_pixel = colorized_prob_image.data + i * colorized_prob_image.step[0] + j * colorized_prob_image.step[1];
                unsigned char* cropped_foscam_color_labels_pixel = cropped_foscam_color_labels.data + i * cropped_foscam_color_labels.step[0] + j * cropped_foscam_color_labels.step[1];
                unsigned char*  op_labeled_image_pixel   = op_labeled_image.data + i* op_labeled_image.step[0] + j* op_labeled_image.step[1];

                int index =  op_labeled_image_pixel[0];

                if(index)
                {
                    float alpha = 0.1;
                    cropped_foscam_color_labels_pixel[0] = alpha* cropped_foscam_color_labels_pixel[0] + (1-alpha)*color_pixel[0];
                    cropped_foscam_color_labels_pixel[1] = alpha* cropped_foscam_color_labels_pixel[0] + (1-alpha)*color_pixel[1];
                    cropped_foscam_color_labels_pixel[2] = alpha* cropped_foscam_color_labels_pixel[0] + (1-alpha)*color_pixel[2];
                }
            }
        }


    ROS_INFO("OUT MERGE DECISIONS");
}

//void arc17_computer_vision_t::merge_decisions(void)
//{
//    ROS_INFO("IN MERGE DECISIONS");

//    cv::Mat colorized_prob_image = foscam_color_labels(detection_roi);
//    cv::Mat op_labeled_image = foscam_labels(detection_roi);


//    op_labeled_image = cv::Mat::zeros(detection_roi.height, detection_roi.width,CV_8UC1);

//    this->foscam_image(detection_roi).copyTo(foscam_color_labels(detection_roi));
//    cv::cvtColor(colorized_prob_image,colorized_prob_image,CV_BGR2HSV);

//    std::vector<cv::Mat>& obj_masks = this->object_masks;
//    for(int i=0;i<obj_masks.size();i++)
//        obj_masks[i](detection_roi) = cv::Mat::zeros(detection_roi.height, detection_roi.width ,CV_8UC1);

//    cv::Mat detection_mask = detection_roi_mask(detection_roi);

//    for(int i=0;i<colorized_prob_image.rows;i++)
//        for(int j=0;j<colorized_prob_image.cols;j++)
//        {
//            unsigned char* detection_mask_pixel = detection_mask.data + i* detection_mask.step[0] + j* detection_mask.step[1];
//            unsigned char* color_pixel = colorized_prob_image.data + i * colorized_prob_image.step[0] + j * colorized_prob_image.step[1];

//            unsigned char*  op_labeled_image_pixel   = op_labeled_image.data + i* op_labeled_image.step[0] + j* op_labeled_image.step[1];
//            op_labeled_image_pixel[0] = 0;

//            if(detection_mask_pixel[0])
//            {
//                float max = -1000000000.0;
//                int index =0;

//                float* prob_pixel = net_data_output_all + i* colorized_prob_image.cols + j;
//                int plane_offset = colorized_prob_image.cols * colorized_prob_image.rows;

//                for(int k=0;k<net_output_blob_dims[1];k++)
//                {
//                    if((prob_pixel + plane_offset*k)[0] >  max)
//                    {
//                        max = (prob_pixel + plane_offset*k)[0];
//                        index = k;
//                    }
//                }

//                if(index && max > CONFIDENCE_THRESHOLD)
//                {
//                    int h;
//                    int s;
//                    int v;

//                    h = (180.0*index) / NUM_OBJECTS;
//                    s = 25 + (250.0*index) / NUM_OBJECTS;
//                    v = 25 + (250.0*index) / NUM_OBJECTS;

//                    float alpha = 0.1;
//                    color_pixel[0] = alpha*color_pixel[0] + (1-alpha)*h;
//                    color_pixel[1] = alpha*color_pixel[1] + (1-alpha)*s;
//                    color_pixel[2] = alpha*color_pixel[2] + (1-alpha)*v;

//                    op_labeled_image_pixel[0] = index;

//                    unsigned char*  object_mask_pixel   = obj_masks[index-1](detection_roi).data + i* obj_masks[index-1](detection_roi).step[0] + j* obj_masks[index-1](detection_roi).step[1];
//                    object_mask_pixel[0] = 255;

//                }
//            }
//            else
//            {
//                color_pixel[0] = 0;
//                color_pixel[1] = 0;
//                color_pixel[2] = 0;
//            }
//        }

//    cv::cvtColor(colorized_prob_image,colorized_prob_image,CV_HSV2BGR);
//    ROS_INFO("OUT MERGE DECISIONS");
//}



void *arc17_computer_vision_t::forward_pass(void *args)
{

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(GPU_ID_SCENE_PARSING);


    while(1)
    {
        usleep(100);

        if(this->flag_do_forward_pass)
        {

            ROS_INFO("FORWARD PASSING");

            cv::Mat input_image = foscam_image(this->detection_roi);

            //            if(USE_CRF)
            //            {
            //                float* img_data_dim = ((caffe::Blob<float>*)(net_blobs_input[1]))->mutable_cpu_data();

            //                img_data_dim[0] = net_input_blob_dims[2];
            //                img_data_dim[1] = net_input_blob_dims[3];
            //            }

            float* net_data_input = ((caffe::Blob<float>*)(this->net_input_blob))->mutable_cpu_data();

            int channel_size = this->net_input_blob_dims[2] * this->net_input_blob_dims[3];

            float* net_data_input_ch1 = net_data_input;
            float* net_data_input_ch2 = net_data_input + channel_size;
            float* net_data_input_ch3 = net_data_input + channel_size * 2;

            for(int i=0;i<input_image.rows;i++)
                for(int j=0;j<input_image.cols;j++)
                {
                    unsigned char* pixel = input_image.data + i * input_image.step[0] + j * input_image.step[1];
                    (net_data_input_ch1 + i * input_image.cols)[j] = (float)pixel[0];
                    (net_data_input_ch2 + i * input_image.cols)[j] = (float)pixel[1];
                    (net_data_input_ch3 + i * input_image.cols)[j] = (float)pixel[2];
                }


            //            std::cout << "Forward passing\n";
            this->net->ForwardFrom(0);
            //            std::cout << "Forward passing Done\n";

            net_data_output_all = net_output_blob->mutable_cpu_data();

            if(USE_CRF)
            {

                ROS_INFO("CRF FORWARD PASSING");

                std::vector<caffe::Blob<float>* > blobs_input = crf_net->input_blobs();

                caffe::Blob<float>*  blob_softmax = blobs_input[0];
                caffe::Blob<float>*  blob_img_dim = blobs_input[1];
                caffe::Blob<float>*  blob_data = blobs_input[2];

                float* crf_net_data_input  = blob_softmax->mutable_cpu_data();

                for(int i=0;i<crop_height;i++)
                    for(int j=0;j<crop_width;j++)
                    {

                        for(int k=0;k< NUM_OBJECTS+1;k++)
                        {
                            //                    std::cout << (prob_pixel + plane_offset*k)[0] << "  ";

                            float* prob_pixel = net_data_output_all + i* crop_width + j;
                            float* crf_prob_pixel = crf_net_data_input + i* crop_width + j;


                            int plane_offset = channel_size;

                            if((prob_pixel + plane_offset*k)[0] <  CONFIDENCE_THRESHOLD)
                                (crf_prob_pixel + plane_offset*k)[0] = 0;//1.0/output_blob_dims[1] ;
                            else
                                (crf_prob_pixel + plane_offset*k)[0]  = (prob_pixel + plane_offset*k)[0];
                        }
                    }


                float* img_data_dim = blob_img_dim->mutable_cpu_data();

                img_data_dim[0] = this->net_input_blob_dims[2];
                img_data_dim[1] = this->net_input_blob_dims[3];

                crf_net_data_input = blob_data->mutable_cpu_data();

                net_data_input_ch1 = crf_net_data_input;
                net_data_input_ch2 = crf_net_data_input + channel_size;
                net_data_input_ch3 = crf_net_data_input + channel_size * 2;

                for(int i=0;i<input_image.rows;i++)
                    for(int j=0;j<input_image.cols;j++)
                    {
                        unsigned char* pixel = input_image.data + i * input_image.step[0] + j * input_image.step[1];
                        (net_data_input_ch1 + i * input_image.cols)[j] = (float)pixel[0];//-mean_rgb[0];
                        (net_data_input_ch2 + i * input_image.cols)[j] = (float)pixel[1];//-mean_rgb[1];
                        (net_data_input_ch3 + i * input_image.cols)[j] = (float)pixel[2];//-mean_rgb[2];
                    }

                crf_net->Forward(0);


                caffe::shared_ptr<caffe::Blob<float> > blob_dense_crf_op = crf_net->blob_by_name("dense_crf_op");

                net_data_output_all = blob_dense_crf_op->mutable_cpu_data();
                ROS_INFO("CRF FORWARD PASS DONE");

            }

            flag_do_forward_pass = false;

            ROS_INFO("FORWARD PASS DONE");
        }
    }
}


