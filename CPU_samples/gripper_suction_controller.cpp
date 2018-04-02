#include <iostream>

#include <ros/ros.h>

#include<iitktcs_msgs_srvs/gripper_suction_controller.h>

#include<gripper_suction_controller/suction_controller.h>


SuctionController* suction_controller;

bool callback_service_gripper_suction_controller(
        iitktcs_msgs_srvs::gripper_suction_controller::Request& req,
        iitktcs_msgs_srvs::gripper_suction_controller::Response& res )
{

    std::cout << "IN SUCTION CONTROLLER\n";

    unsigned char nozel_angle = (unsigned char)req.nozel_angle.data;
    unsigned char vacuum_cleaner = (unsigned char)req.vacuum_cleaner.data;
    unsigned char suction_valve = (unsigned char)req.suction_valve.data;

    int result = SuctionController::SUCTION_FAILURE;

    // CONVERT TO in range min max

    float scale = (suction_controller->max_limit-suction_controller->min_limit)/(255.0-0.0);

    nozel_angle = nozel_angle * scale ;

    result = suction_controller->set_actuators_position(suction_controller->max_limit - nozel_angle);

    result = suction_controller->get_actuators_position(&nozel_angle);

    std::cout << "CURRENT POSITION = " << (int)nozel_angle << "\n";

    if(vacuum_cleaner == 1)
        result = suction_controller->set_vacuum_controller(SuctionController::VACUUM_ON);
    else if(vacuum_cleaner == 0)
        result = suction_controller->set_vacuum_controller(SuctionController::VACUUM_OFF);

    std::cout << "OUT SUCTION CONTROLLER\n";

//    if(result == SuctionController::SUCTION_SUCCESS)
        return true;
//    else if(result == SuctionController::SUCTION_FAILURE)
//        return false;

}



int main(int argc, char* argv[])
{
    ros::init(argc, argv, "gripper_suction_controller");
    ros::NodeHandle nh;

    ros::MultiThreadedSpinner multithreaded_spinner(2);

    std::string str_serial_port;
    nh.getParam("/ARC17_SUCTION_SERIAL_PORT", str_serial_port);

    int ACTUATOR_MIN_LIMIT;
    int ACTUATOR_MAX_LIMIT;

    nh.getParam("/ARC17_ACTUATOR_MIN_LIMIT", ACTUATOR_MIN_LIMIT);
    nh.getParam("/ARC17_ACTUATOR_MAX_LIMIT", ACTUATOR_MAX_LIMIT);

    suction_controller = new SuctionController(str_serial_port);

    suction_controller->set_actuator_min_limit(ACTUATOR_MIN_LIMIT);
    suction_controller->set_actuator_max_limit(ACTUATOR_MAX_LIMIT);

    ros::ServiceServer service_server_gripper_suction_controller = nh.advertiseService("/iitktcs/gripper_suction_controller",
                                                                         &callback_service_gripper_suction_controller);

    std::cout << "READY TO ACCEPT CLIENT CALLS\n";
    multithreaded_spinner.spin();

    return 0;
}

