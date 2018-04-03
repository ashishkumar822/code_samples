#include<iostream>
#include<stdio.h>

#include<opencv2/opencv.hpp>

#include<Eigen/Dense>
#include <Eigen/Geometry>

#include<stdlib.h>

#include<time.h>

#include<gps_t.hpp>
#include<pose_t.hpp>
#include<points3d_t.hpp>
#include<image_t.hpp>

#include<lcm/lcm-cpp.hpp>


#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<pcl/filters/voxel_grid.h>

//#include<boost/math/quaternion.hpp>


//#include<pcl/common/angles.h>
//#include<pcl/common/transforms.h>

//int n_frames = 107;
std::string drive_id = "2011_09_26_drive_0095_sync";
//std::string drive_id = "2011_09_26_drive_0001_sync";
std::string str_drive_log = "/home/isl-server/ashish/KITTI/2011_09_26/" + drive_id + "/";

int n_frames = 185;
//std::string drive_id = "2011_09_26_drive_0104_sync";
//std::string str_drive_log = "/home/isl-server/ashish/KITTI/2011_09_26/" + drive_id + "/";

std::string str_left_cam_gray = str_drive_log + "image_00/data/";
std::string str_right_cam_gray = str_drive_log + "image_01/data/";
std::string str_left_cam_color = str_drive_log + "image_02/data/";
std::string str_right_cam_color = str_drive_log + "image_03/data/";
std::string str_velodyne = str_drive_log + "velodyne_points/data/";
std::string str_gps_ins = str_drive_log + "oxts/data/";

std::string str_calib_cam_T_cam = str_drive_log + "../2011_09_26_calib/calib_cam_to_cam.txt";
std::string str_calib_imu_T_vel = str_drive_log + "../2011_09_26_calib/calib_imu_to_velo.txt";
std::string str_calib_vel_T_cam = str_drive_log + "../2011_09_26_calib/calib_velo_to_cam.txt";


std::string str_timestamp_vel = str_drive_log + "/velodyne_points/timestamps.txt";
std::string str_timestamp_gps_ins = str_drive_log + "/oxts/timestamps.txt";
std::string str_timestamp_image_02 = str_drive_log + "/image_02/timestamps.txt";

Eigen::Affine3f imu_T_vel;  // velodyne to vehicle transform
Eigen::Affine3f vel_T_cam;  // velodyne to cam transform

typedef struct _frame_t_
{
    cv::Mat im_left_gray;
    cv::Mat im_right_gray;
    cv::Mat im_left_color;
    cv::Mat im_right_color;

    unsigned long int utime_ins;
    unsigned long int utime_velodyne;

    int n_points;
    float* xyz;
    unsigned char* i;

    Eigen::Affine3f pose;
    Eigen::Quaternionf orientation;

    _frame_t_()
    {

    }

    ~_frame_t_()
    {
        free(xyz);
        free(i);
    }

    typedef struct _gps_ins_t_
    {
        float lat;          //   latitude of the oxts-unit (deg)
        float lon;          //   longitude of the oxts-unit (deg)
        float alt;          //   altitude of the oxts-unit (m)
        float roll;          //  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
        float pitch;          // pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
        float yaw;          //   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
        float vn;          //    velocity towards north (m/s)
        float ve;          //    velocity towards east (m/s)
        float vf;          //    forward velocity, i.e. parallel to earth-surface (m/s)
        float vl;          //    leftward velocity, i.e. parallel to earth-surface (m/s)
        float vu;          //    upward velocity, i.e. perpendicular to earth-surface (m/s)
        float ax;          //    acceleration in x, i.e. in direction of vehicle front (m/s^2)
        float ay;          //    acceleration in y, i.e. in direction of vehicle left (m/s^2)
        float az;          //    acceleration in z, i.e. in direction of vehicle top (m/s^2)
        float af;          //    forward acceleration (m/s^2)
        float al;          //    leftward acceleration (m/s^2)
        float au;          //    upward acceleration (m/s^2)
        float wx;          //    angular rate around x (rad/s)
        float wy;          //    angular rate around y (rad/s)
        float wz;          //    angular rate around z (rad/s)
        float wf;          //    angular rate around forward axis (rad/s)
        float wl;          //    angular rate around leftward axis (rad/s)
        float wu;          //    angular rate around upward axis (rad/s)
        float pos_accuracy;          //  velocity accuracy (north/east in m)
        float vel_accuracy;          //  velocity accuracy (north/east in m/s)
        float navstat;          //       navigation status (see navstat_to_string)
        float numsats;          //       number of satellites tracked by primary GPS receiver
        float posmode;          //       position mode of primary GPS receiver (see gps_mode_to_string)
        float velmode;          //       velocity mode of primary GPS receiver (see gps_mode_to_string)
        float orimode;          //       orientation mode of primary GPS receiver (see gps_mode_to_string)
    }gps_ins_t;

    void prinft_gps_ins(void)
    {
        std::cout << "------******-------\n"
                  << "lat " <<  gps_ins.lat  <<  " \n"
                  << "lon " <<  gps_ins.lon  <<  " \n"
                  << "alt " <<  gps_ins.alt  <<  " \n"
                  << "roll " <<  gps_ins.roll  <<  " \n"
                  << "pitch " <<  gps_ins.pitch  <<  " \n"
                  << "yaw " <<  gps_ins.yaw  <<  " \n"
                  << "vn " <<  gps_ins.vn  <<  " \n"
                  << "ve " << gps_ins.ve   <<  " \n"
                  << "vf " <<  gps_ins.vf  <<  " \n"
                  << "vl " <<  gps_ins.vl  <<  " \n"
                  << "vu " <<  gps_ins.vu  <<  " \n"
                  << "ax " <<  gps_ins.ax  <<  " \n"
                  << "ay " << gps_ins.ay   <<  " \n"
                  << "az " << gps_ins.az   <<  " \n"
                  << "af " << gps_ins.af   <<  " \n"
                  << "al " <<  gps_ins.al  <<  " \n"
                  << "au " <<  gps_ins.au  <<  " \n"
                  << "wx " <<  gps_ins.wx  <<  " \n"
                  << "wy " <<  gps_ins.wy  <<  " \n"
                  << "wz " << gps_ins.wz  <<  " \n"
                  << "wf " << gps_ins.wf   <<  " \n"
                  << "wl " << gps_ins.wl   <<  " \n"
                  << "wu " << gps_ins.wu   <<  " \n"
                  << "pos_accuracy " << gps_ins.pos_accuracy   <<  " \n"
                  << "vel_accuracy " <<  gps_ins.vel_accuracy  <<  " \n"
                  << "navstat " <<  gps_ins.navstat  <<  " \n"
                  << "numsats " <<  gps_ins.numsats  <<  " \n"
                  << "posmode " <<  gps_ins.posmode  <<  " \n"
                  << "velmode " <<  gps_ins.velmode  <<  " \n"
                  << "orimode " <<  gps_ins.orimode  <<  "\n"
                  << "------******-------\n";


    }

    gps_ins_t gps_ins;

}frame_t;

void get_time_stamp(FILE* time_stamp_file, unsigned long int& timestamp)
{
    std::stringstream str_year;
    std::stringstream str_month;
    std::stringstream str_day;
    std::stringstream str_hour;
    std::stringstream str_min;
    std::stringstream str_sec;
    std::stringstream str_decimal;

    char text[50];

    fscanf(time_stamp_file,"%s",text);

    int index = 0;
    while(text[index]!='-')
        str_year << text[index++];

    index++;
    while(text[index]!='-')
        str_month << text[index++];

    index++;
    while(text[index]!='\0')
        str_day << text[index++];

    fscanf(time_stamp_file,"%s",text);

    index = 0;
    while(text[index]!=':')
        str_hour << text[index++];

    index++;
    while(text[index]!=':')
        str_min << text[index++];

    index++;
    while(text[index]!='.')
        str_sec << text[index++];

    while(text[index]!='\0')
        str_decimal << text[index++];


    struct tm time;
    time.tm_sec = atoi(str_sec.str().c_str());			/* Seconds.	[0-60] (1 leap second) */
    time.tm_min = atoi(str_min.str().c_str());			/* Minutes.	[0-59] */
    time.tm_hour = atoi(str_hour.str().c_str());			/* Hours.	[0-23] */
    time.tm_mday = atoi(str_day.str().c_str());			/* Day.		[1-31] */
    time.tm_mon = atoi(str_month.str().c_str());			/* Month.	[0-11] */
    time.tm_year = atoi(str_year.str().c_str());			/* Year	- 1900.  */

    time_t time_since_epoch = mktime(&time);

    timestamp = time_since_epoch * 1000000;
    timestamp +=  atof(str_decimal.str().c_str()) * 1000000;

    //    std::cout << atof(str_decimal.str().c_str()) * 1000000 << "\n";
    //    std::cout << timestamp << "  "
    //              << str_sec.str() << "  "
    //                 << str_min.str() << "  "
    //                    << str_hour.str() << "  "
    //                       << str_day.str() << "  "
    //                          << str_month.str() << "  "
    //                             << str_year.str() << "  "
    //                                << str_decimal.str() << "\n";

}

FILE* new_poses_file = fopen("/home/isl-server/ashish/KITTI/new_pose.txt","r");

void get_frame(int frame_id, frame_t& frame)
{
    unsigned int text[10];
    memset(text,0,10);

    int id = frame_id;
    for(int i=9;i>=0;i--)
    {
        text[i] = id%10;
        id /= 10;
    }

    std::stringstream str_file;
    for(int i=0;i<10;i++)
        str_file << text[i];


    // READ IMAGES

    {
        frame.im_left_gray =  cv::imread(str_left_cam_gray + str_file.str() + ".png", cv::IMREAD_GRAYSCALE);
        frame.im_right_gray =  cv::imread(str_right_cam_gray + str_file.str() + ".png", cv::IMREAD_GRAYSCALE);
        frame.im_left_color =  cv::imread(str_left_cam_color + str_file.str() + ".png", cv::IMREAD_COLOR);
        frame.im_right_color =  cv::imread(str_left_cam_color + str_file.str() + ".png", cv::IMREAD_COLOR);
    }

    //READ VELODYNE POINTS

    {
        FILE* velodyne_file = fopen((str_velodyne + str_file.str() + ".bin").c_str(),"rb");

        int n_points = 1e6;

        float* xyzi = (float*)malloc(sizeof(float)*n_points);
        n_points = fread(xyzi,sizeof(float),n_points, velodyne_file) / 4;

        frame.n_points = n_points;
        frame.xyz = (float*)malloc(sizeof(float) * 3 * frame.n_points);
        frame.i = (unsigned char*)malloc(sizeof(unsigned char) * frame.n_points);

        for(int i=0;i<n_points;i++)
        {
            memcpy(frame.xyz + i*3,xyzi + i*4,sizeof(float)*3);
            frame.i[i] = (xyzi + i*4)[0] + 127;
        }

        free(xyzi);

        fclose(velodyne_file);
    }

    //READ GPS INS

    {
        FILE* gps_ins_file = fopen((str_gps_ins + str_file.str() + ".txt" ).c_str(),"r");

        fscanf(gps_ins_file,
               "%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f",
               &frame.gps_ins.lat,
               &frame.gps_ins.lon,
               &frame.gps_ins.alt,
               &frame.gps_ins.roll,
               &frame.gps_ins.pitch,
               &frame.gps_ins.yaw,
               &frame.gps_ins.vn,
               &frame.gps_ins.ve,
               &frame.gps_ins.vf,
               &frame.gps_ins.vl,
               &frame.gps_ins.vu,
               &frame.gps_ins.ax,
               &frame.gps_ins.ay,
               &frame.gps_ins.az,
               &frame.gps_ins.af,
               &frame.gps_ins.al,
               &frame.gps_ins.au,
               &frame.gps_ins.wx,
               &frame.gps_ins.wy,
               &frame.gps_ins.wz,
               &frame.gps_ins.wf,
               &frame.gps_ins.wl,
               &frame.gps_ins.wu,
               &frame.gps_ins.pos_accuracy,
               &frame.gps_ins.vel_accuracy,
               &frame.gps_ins.navstat,
               &frame.gps_ins.numsats,
               &frame.gps_ins.posmode,
               &frame.gps_ins.velmode,
               &frame.gps_ins.orimode);

        fclose(gps_ins_file);
    }

    // COMPUTE POSE

    {
        frame_t::gps_ins_t& gps_ins = frame.gps_ins;

        //        float scale = cos(gps_ins.lat * M_PI / 180.0);

        float r_earth = 6378137.0;
        float e = 8.1819190842622e-2;
        //        //        float tx = scale * gps_ins.lon * M_PI * r_earth / 180;
        //        //        float ty = scale * r_earth * log( tan((90+ gps_ins.lat) * M_PI / 360) );

        //        static float first_lat;
        //        static float first_lon;
        //        static float first_alt;

        //        if(!frame_id)
        //        {
        //            first_lat = gps_ins.lat;
        //            first_lon = gps_ins.lon;
        //            first_alt = -gps_ins.alt;
        //        }

        //        float tx = r_earth * sin((gps_ins.lat-first_lat) * M_PI / 180);
        //        float ty = scale* r_earth * sin((gps_ins.lon-first_lon) * M_PI / 180.0);
        //        float tz = - gps_ins.alt - first_alt;


        //        xy[0] = sin (dlat) * lin->radius_ns;
        //        xy[1] = sin (dlon) * lin->radius_ew * cos (to_radians (lin->lat0_deg));

        static Eigen::Vector3f first_lat_lon_alt;
        static Eigen::Vector3f first_earth_xyz;
        static Eigen::Matrix3f R_gps_ned;



        if(!frame_id)
        {
            first_lat_lon_alt(0) = gps_ins.lat * M_PI / 180;
            first_lat_lon_alt(1) = gps_ins.lon * M_PI / 180;
            first_lat_lon_alt(2) = -gps_ins.alt;

            first_earth_xyz(0) = (r_earth +  gps_ins.alt) * cos(gps_ins.lat * M_PI / 180) * cos(gps_ins.lon * M_PI / 180);
            first_earth_xyz(1) = (r_earth +  gps_ins.alt) * cos(gps_ins.lat * M_PI / 180) * sin(gps_ins.lon * M_PI / 180);
            first_earth_xyz(2) = (r_earth +  gps_ins.alt) * sin(gps_ins.lat * M_PI / 180);

            //            first_earth_xyz(0) = (r_earth ) * sin((gps_ins.lat * M_PI / 180) - first_lat_lon_alt(0));
            //            first_earth_xyz(1) = (r_earth ) * cos((gps_ins.lat * M_PI / 180) - first_lat_lon_alt(0))
            //                    * sin((gps_ins.lon * M_PI / 180)-first_lat_lon_alt(1));
            //            first_earth_xyz(2) = 0;//- gps_ins.alt ;

            R_gps_ned << - sin(first_lat_lon_alt(0)) * cos(first_lat_lon_alt(1)),
                    - sin(first_lat_lon_alt(0)) * sin(first_lat_lon_alt(1)),
                    cos(first_lat_lon_alt(0)),
                    -sin(first_lat_lon_alt(1)),
                    cos(first_lat_lon_alt(1)),
                    0,
                    - cos(first_lat_lon_alt(0)) * cos(first_lat_lon_alt(1)),
                    - cos(first_lat_lon_alt(0)) * sin(first_lat_lon_alt(1)),
                    -sin(first_lat_lon_alt(0));
        }


        Eigen::Vector3f earth_xyz;
        earth_xyz(0) = (r_earth +  gps_ins.alt) * cos(gps_ins.lat * M_PI / 180) * cos(gps_ins.lon * M_PI / 180);
        earth_xyz(1) = (r_earth +  gps_ins.alt) * cos(gps_ins.lat * M_PI / 180) * sin(gps_ins.lon * M_PI / 180);
        earth_xyz(2) = (r_earth +  gps_ins.alt) * sin(gps_ins.lat * M_PI / 180);

        //        earth_xyz(0) = (r_earth ) * sin((gps_ins.lat * M_PI / 180) - first_lat_lon_alt(0));
        //        earth_xyz(1) = (r_earth ) * cos((gps_ins.lat * M_PI / 180) - first_lat_lon_alt(0))
        //                * sin((gps_ins.lon * M_PI / 180)-first_lat_lon_alt(1));
        //        earth_xyz(2) = - gps_ins.alt - first_lat_lon_alt(2) ;

        //        earth_xyz(0) = (r_earth ) * sin(gps_ins.lat * M_PI / 180);
        //        earth_xyz(1) = (r_earth ) * cos(gps_ins.lat * M_PI / 180) * sin(gps_ins.lon * M_PI / 180);
        //        earth_xyz(2) = 0;//- gps_ins.alt;

        Eigen::Matrix3f R_ned_enu;
        R_ned_enu << 0, 1, 0, 1, 0, 0, 0, 0,-1;

        Eigen::Vector3f ned_xyz;
        ned_xyz = R_gps_ned * (earth_xyz- first_earth_xyz);

        Eigen::Vector3f local_xyz;
        local_xyz = ned_xyz;

        float rx = gps_ins.roll;
        float ry = gps_ins.pitch;
        float rz = gps_ins.yaw;

        Eigen::Matrix3f Rx;
        Eigen::Matrix3f Ry;
        Eigen::Matrix3f Rz;
        Eigen::Matrix3f R;

        Rx << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx);
        Ry << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry);
        Rz << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1;
        R  =  R_ned_enu * Rz * Ry * Rx ;


        Eigen::Affine3f pose = Eigen::Affine3f::Identity();

        for(int i=0;i<3;i++)
            pose(i,3) = local_xyz(i);

        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                pose(i,j) = R(i,j);


        Eigen::Affine3f transformation = Eigen::Affine3f::Identity();

        if(frame_id)
        {
            for(int i=0;i<4;i++)
                for(int j=0;j<4;j++)
                    fscanf(new_poses_file,"%f",&transformation(i,j));
        }

        frame.pose = transformation * pose;
        frame.orientation = Eigen::Quaternionf( frame.pose.rotation());

        std::cout <<  frame.pose.matrix() << "\n\n" << transformation.matrix() << "\n\n" << pose.matrix() << "\n\n------------------\n\n";
    }


}

void get_calibration_data(void)
{

    FILE* file_imu_T_vel = fopen(str_calib_imu_T_vel.c_str(),"r");

    char text[100];

    fscanf(file_imu_T_vel,"%s%s%s",text,text,text);
    fscanf(file_imu_T_vel,"%s",text);

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            fscanf(file_imu_T_vel,"%f",&imu_T_vel(i,j));


    fscanf(file_imu_T_vel,"%s",text);

    for(int i=0;i<3;i++)
        fscanf(file_imu_T_vel,"%f",&imu_T_vel(i,3));

    imu_T_vel(3,0) = 0;
    imu_T_vel(3,1) = 0;
    imu_T_vel(3,2) = 0;
    imu_T_vel(3,3) = 1;


    fclose(file_imu_T_vel);


    FILE* file_vel_T_cam = fopen(str_calib_vel_T_cam.c_str(),"r");

    fscanf(file_vel_T_cam,"%s%s%s",text,text,text);
    fscanf(file_vel_T_cam,"%s",text);

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            fscanf(file_vel_T_cam,"%f",&vel_T_cam(i,j));


    fscanf(file_vel_T_cam,"%s",text);

    for(int i=0;i<3;i++)
        fscanf(file_vel_T_cam,"%f",&vel_T_cam(i,3));

    vel_T_cam(3,0) = 0;
    vel_T_cam(3,1) = 0;
    vel_T_cam(3,2) = 0;
    vel_T_cam(3,3) = 1;


    fclose(file_vel_T_cam);
}



int main(void)
{

    get_calibration_data();


    Eigen::Affine3f vel_T_imu;  // vechile to velodyne transform
    vel_T_imu = imu_T_vel.inverse();

    float roll;
    float pitch;
    float yaw;

    //    pcl::getEulerAngles(vel_T_imu,roll,pitch,yaw);

    //    std::cout << roll*180/M_PI << " " << pitch*180/M_PI << " " << yaw*180/M_PI  << "\n\n" << vel_T_imu.matrix() << "\n";
    //    Eigen::Affine3f vel_T_global;
    //    vel_T_global = frame.pose * vel_T_imu;

    int frame_id = 0;

    FILE* gps_ins_timestamps_file = fopen(str_timestamp_gps_ins.c_str(),"r");
    FILE* vel_timestamps_file = fopen(str_timestamp_vel.c_str(),"r");
    FILE* image_02_timestamps_file = fopen(str_timestamp_image_02.c_str(),"r");


    lcm::LogFile lcm_logfile(str_drive_log + drive_id + ".lcmlog","r");

    //    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    //    voxel_filter.setLeafSize(Eigen::Vector4f(0.1,0.1,0.1,1.0));

    int event_id = 0;
    while(frame_id < n_frames)
    {

        frame_t frame;
        get_frame(frame_id,frame);

        frame_id++;
        continue;
        //        pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        //        for(int i=0;i<frame.n_points;i++)
        //        {
        //            pcl::PointXYZ point;

        //            point.x = frame.xyz[i*3+0];
        //            point.y = frame.xyz[i*3+1];
        //            point.z = frame.xyz[i*3+2];

        //            ptr_cloud->push_back(point);
        //        }

        //        voxel_filter.setInputCloud(ptr_cloud);
        //        voxel_filter.filter(*ptr_cloud);


        gps_t gps;
        pose_t pose;
        points3d_t points_3d;
        image_t image;

        unsigned long int timestamp;

        get_time_stamp(gps_ins_timestamps_file,timestamp);

        gps.utime = timestamp;
        gps.source = gps.SOURCE_RTNAV;
        gps.alt = frame.gps_ins.alt;
        gps.lat = frame.gps_ins.lat * M_PI/180.0f;
        gps.lon = frame.gps_ins.lon * M_PI/180.0f;
        //        gps.lat = frame_id * 0.1 * M_PI/180.0f;
        //        gps.lon = frame_id * 0.1 * M_PI/180.0f;
        gps.heading = frame.gps_ins.yaw * M_PI/180.0f;


        pose.utime = timestamp;
        pose.accel[0] = frame.gps_ins.ax;
        pose.accel[1] = frame.gps_ins.ay;
        pose.accel[2] = frame.gps_ins.az;
        pose.vel[0] = frame.gps_ins.vf;
        pose.vel[1] = frame.gps_ins.vl;
        pose.vel[2] = frame.gps_ins.vu;
        pose.rotation_rate[0] = frame.gps_ins.wx;
        pose.rotation_rate[1] = frame.gps_ins.wy;
        pose.rotation_rate[2] = frame.gps_ins.wz;
        pose.pos[0] = frame.pose(0,3);
        pose.pos[1] = frame.pose(1,3);
        pose.pos[2] = frame.pose(2,3);
        pose.orientation[0] = frame.orientation.w();
        pose.orientation[1] = frame.orientation.x();
        pose.orientation[2] = frame.orientation.y();
        pose.orientation[3] = frame.orientation.z();

        get_time_stamp(vel_timestamps_file,timestamp);

        points_3d.utime = timestamp;
        points_3d.datalen = frame.n_points*3;

        for(int i=0;i<points_3d.datalen;i++)
            points_3d.data.push_back(frame.xyz[i]);

        //        points_3d.utime = timestamp;
        //        points_3d.datalen = ptr_cloud->size()*3;

        //        for(int i=0;i<points_3d.datalen/3;i++)
        //        {
        //            points_3d.data.push_back(ptr_cloud->at(i).x);
        //            points_3d.data.push_back(ptr_cloud->at(i).y);
        //            points_3d.data.push_back(ptr_cloud->at(i).z);
        //        }

        // IMAGE
        get_time_stamp(image_02_timestamps_file,timestamp);

        image.utime = timestamp;
        image.height = frame.im_left_color.rows;
        image.width = frame.im_left_color.cols;
        image.size = image.height*image.width*3;
        image.data.resize(image.size);
        memcpy(image.data.data(),frame.im_left_color.data, image.size*sizeof(unsigned char));


        //WRITE TO LCM LOG

        lcm::LogEvent log_event;

        int datalen = gps.getEncodedSize();
        unsigned char* buffer = (unsigned char*)malloc(sizeof(unsigned char)*datalen);
        gps.encode(buffer,0,datalen);

        log_event.channel= "GPS";
        log_event.eventnum = event_id++;
        log_event.timestamp = gps.utime;
        log_event.datalen = datalen;
        log_event.data = buffer;

        lcm_logfile.writeEvent(&log_event);
        free(buffer);


        datalen = pose.getEncodedSize();
        buffer = (unsigned char*)malloc(sizeof(unsigned char)*datalen);
        pose.encode(buffer,0,datalen);

        log_event.channel= "POSE";
        log_event.eventnum = event_id++;
        log_event.timestamp = pose.utime;
        log_event.datalen = datalen;
        log_event.data = buffer;

        lcm_logfile.writeEvent(&log_event);
        free(buffer);

        datalen = points_3d.getEncodedSize();
        buffer = (unsigned char*)malloc(sizeof(unsigned char)*datalen);
        points_3d.encode(buffer,0,datalen);

        log_event.channel= "VELODYNE";
        log_event.eventnum = event_id++;
        log_event.timestamp = points_3d.utime;
        log_event.datalen = datalen;
        log_event.data = buffer;

        lcm_logfile.writeEvent(&log_event);
        free(buffer);


        datalen = image.getEncodedSize();
        buffer = (unsigned char*)malloc(sizeof(unsigned char)*datalen);
        image.encode(buffer,0,datalen);

        log_event.channel = "IMAGE_02";
        log_event.eventnum = event_id++;
        log_event.timestamp = image.utime;
        log_event.datalen = datalen;
        log_event.data = buffer;

        lcm_logfile.writeEvent(&log_event);
        free(buffer);

        std::cout << "Processing Frame = " << frame_id << "\n";
        frame_id++;
    }


    fclose(gps_ins_timestamps_file);
    fclose(vel_timestamps_file);
    fclose(image_02_timestamps_file);

//    fclose(new_poses_file);

    return 0;

}


