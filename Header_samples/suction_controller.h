#ifndef __SUCTION_CONTROLLER_H__
#define __SUCTION_CONTROLLER_H__

#include<stdio.h>
#include<iostream>
#include<SerialPort.h>


class SuctionController
{

private:

#define CMD_SET_ACTUATORS_POSITION      'A'
#define CMD_GET_ACTUATORS_POSITION      'B'

#define CMD_TURN_ON_VACUUM_SUCTION      'C'
#define CMD_TURN_OFF_VACUUM_SUCTION     'D'

#define CMD_SET_ACTUATOR_MIN_LIMIT      'E'
#define CMD_SET_ACTUATOR_MAX_LIMIT      'F'


#define CMD_SUCTION_CONTROLLER_CHECK     'Y'

#define RES_SUCCESS      'S'
#define RES_FAILURE      'F'


#define DUMMY 255

#define RW_TIME_OUT 10000

    unsigned char response;
    SerialPort* serial_port;

public:

    enum SUCTION_RESPONSE{ SUCTION_SUCCESS, SUCTION_FAILURE};

    enum VACUUM_CONTROLLER{ VACUUM_ON, VACUUM_OFF};

    int max_limit;
    int min_limit;

    SuctionController(std::string str_serial_port)
    {
        min_limit =2;
        max_limit =250;


        serial_port = new SerialPort(str_serial_port);

        try
        {
            serial_port->Open(SerialPort::BAUD_9600,
                              SerialPort::CHAR_SIZE_8,
                              SerialPort::PARITY_NONE,
                              SerialPort::STOP_BITS_1,
                              SerialPort::FLOW_CONTROL_NONE
                              );
            response = 'N';


            serial_port->WriteByte(CMD_SUCTION_CONTROLLER_CHECK);
            serial_port->WriteByte(DUMMY);

            response = serial_port->ReadByte(RW_TIME_OUT);
            if(response == RES_SUCCESS)
                std::cout << "SUCTION CONTROLLER IS AVAILABLE\n";
            else
                std::cout << "SUCTION CONTROLLER IS NOT AVAILABLE EXITING\n";
        }
        catch(SerialPort::AlreadyOpen already_open)
        {
            std::cout << "ALREADY OPEN, ERROR STRING =  " << already_open.what() << "\n";
        }
        catch(SerialPort::OpenFailed open_failed)
        {
            std::cout << "OPEN FAILED , ERROR STRING =  " << open_failed.what() << "\n";
        }
        catch(SerialPort::UnsupportedBaudRate unsupported_baudrate)
        {
            std::cout << "UNSUPPORTED RATE, ERROR STRING =  " << unsupported_baudrate.what() << "\n";
        }
        catch(std::invalid_argument std_invalid_argument)
        {
            std::cout << "INVALID ARGUMENT , ERROR STRING =  " << std_invalid_argument.what() << "\n";
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }
    }



    ~SuctionController()
    {
        try
        {
            serial_port->Close();
            delete serial_port;
            std::cout << "SUCTION CONTROLLER EXITING\n";
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }
    }


    int set_actuator_min_limit(unsigned char min_limit)
    {
        this->min_limit = min_limit;

        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_SET_ACTUATOR_MIN_LIMIT);
            serial_port->WriteByte(min_limit);

            return SUCTION_SUCCESS;
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }

    }

    int set_actuator_max_limit(unsigned char max_limit)
    {
        this->max_limit = max_limit;

        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_SET_ACTUATOR_MAX_LIMIT);
            serial_port->WriteByte(max_limit);

            return SUCTION_SUCCESS;
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }

    }


    int set_actuators_position(unsigned char position)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_SET_ACTUATORS_POSITION);
            serial_port->WriteByte(position);

            return SUCTION_SUCCESS;
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }

    }

    int get_actuators_position(unsigned char* position)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_GET_ACTUATORS_POSITION);
            serial_port->WriteByte(DUMMY);

            *position = serial_port->ReadByte(RW_TIME_OUT);
            response = serial_port->ReadByte(RW_TIME_OUT);
            if(response == RES_SUCCESS)
            {
                return SUCTION_SUCCESS;
            }
            else
                return SUCTION_FAILURE;
        }
        catch(SerialPort::NotOpen not_open)
        {
            std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
        }
        catch(SerialPort::ReadTimeout read_timeout)
        {
            std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
        }
        catch(std::runtime_error runtime_error)
        {
            std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
        }
    }

    int set_vacuum_controller(unsigned char vacuum_controller_on_off)
    {
        response = 'N';

        if(vacuum_controller_on_off == VACUUM_ON)
        {

            try
            {
                serial_port->WriteByte(CMD_TURN_ON_VACUUM_SUCTION);
                serial_port->WriteByte(DUMMY);

                response = serial_port->ReadByte(RW_TIME_OUT);
                if(response == RES_SUCCESS)
                    return SUCTION_SUCCESS;
                else
                    return SUCTION_FAILURE;
            }
            catch(SerialPort::NotOpen not_open)
            {
                std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
            }
            catch(SerialPort::ReadTimeout read_timeout)
            {
                std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
            }
            catch(std::runtime_error runtime_error)
            {
                std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
            }
        }

        else if(vacuum_controller_on_off == VACUUM_OFF)
        {

            try
            {
                serial_port->WriteByte(CMD_TURN_OFF_VACUUM_SUCTION);
                serial_port->WriteByte(DUMMY);

                response = serial_port->ReadByte(RW_TIME_OUT);
                if(response == RES_SUCCESS)
                    return SUCTION_SUCCESS;
                else
                    return SUCTION_FAILURE;
            }
            catch(SerialPort::NotOpen not_open)
            {
                std::cout << "NOT OPEN, ERROR STRING =  " << not_open.what() << "\n";
            }
            catch(SerialPort::ReadTimeout read_timeout)
            {
                std::cout << "READ TIMEOUT, ERROR STRING =  " << read_timeout.what() << "\n";
            }
            catch(std::runtime_error runtime_error)
            {
                std::cout << "RUNTIME ERROR, ERROR STRING =  " << runtime_error.what() << "\n";
            }
        }
    }


};

#endif
