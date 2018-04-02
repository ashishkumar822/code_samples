#ifndef _ROTATING_PLATFORM_H_
#define _ROTATING_PLATFORM_H_

#include<SerialPort.h>



class RotatingPlatform
{

private:

#define CMD_ROTATE_PLATFORM_CW      'A'
#define CMD_ROTATE_PLATFORM_ACW      'B'
#define CMD_SET_SPEED      'C'
#define CMD_STOP      'D'



#define CMD_ROTATE_PLATFORM_CHECK     'Z'

#define RES_SUCCESS      'S'
#define RES_FAILURE      'F'
#define RES_ROTATION     'C'

#define DUMMY 255

#define RW_TIME_OUT 100000

#define MAX_ROTATIONS 253

    unsigned char response;

    SerialPort* serial_port;

public:

    enum ROTATE_PLATFORM{ ROTATE_PLATFORM_SUCCESS, ROTATE_PLATFORM_FAILURE};


    RotatingPlatform(std::string str_serial_port)
    {

        try
        {
            serial_port = new SerialPort(str_serial_port);

            serial_port->Open(SerialPort::BAUD_9600,
                              SerialPort::CHAR_SIZE_8,
                              SerialPort::PARITY_NONE,
                              SerialPort::STOP_BITS_1,
                              SerialPort::FLOW_CONTROL_NONE
                              );


            response = 'N';

            serial_port->WriteByte(CMD_ROTATE_PLATFORM_CHECK);
            serial_port->WriteByte(DUMMY);


            response = serial_port->ReadByte(RW_TIME_OUT);
            if(response == RES_SUCCESS)
                std::cout << "ROTATING PLATFORM IS AVAILABLE\n";
            else
                std::cout << "ROTATING PLATFORM IS NOT AVAILABLE EXITING\n";
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }

    ~RotatingPlatform()
    {
        try
        {
            serial_port->Close();
            delete serial_port;

            std::cout << "SUCTION CONTROLLER EXITING\n";
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }

    int rotate_motor_cw(unsigned char num_rotations)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_ROTATE_PLATFORM_CW);
            serial_port->WriteByte(num_rotations);

            if(num_rotations == 0 || num_rotations > MAX_ROTATIONS)
            {
                response = serial_port->ReadByte(RW_TIME_OUT);
                if(response == RES_SUCCESS)
                    return ROTATE_PLATFORM_SUCCESS;
                else
                    return ROTATE_PLATFORM_FAILURE;
            }
            else
            {
                return ROTATE_PLATFORM_SUCCESS;
            }
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }



    int rotate_motor_acw(unsigned char num_rotations)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_ROTATE_PLATFORM_ACW);
            serial_port->WriteByte(num_rotations);

            if(num_rotations > MAX_ROTATIONS)
            {
                response = serial_port->ReadByte(RW_TIME_OUT);
                if(response == RES_SUCCESS)
                    return ROTATE_PLATFORM_SUCCESS;
                else
                    return ROTATE_PLATFORM_FAILURE;
            }
            else
            {
                return ROTATE_PLATFORM_SUCCESS;
            }

        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }


    int wait_for_rotation_completion(void)
    {
        response = 'N';

        try
        {

            while(serial_port->ReadByte(RW_TIME_OUT) != RES_ROTATION);
            std::cout << "ROTATION COMPLETED\n";

                return ROTATE_PLATFORM_SUCCESS;

        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }


    int set_speed(unsigned char speed)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_SET_SPEED);
            serial_port->WriteByte(speed);

            response = serial_port->ReadByte(RW_TIME_OUT);
            if(response == RES_SUCCESS)
                return ROTATE_PLATFORM_SUCCESS;
            else
                return ROTATE_PLATFORM_FAILURE;
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }

    int stop(void)
    {
        response = 'N';

        try
        {
            serial_port->WriteByte(CMD_STOP);
            serial_port->WriteByte(DUMMY);

            response = serial_port->ReadByte(RW_TIME_OUT);
            if(response == RES_SUCCESS)
                return ROTATE_PLATFORM_SUCCESS;
            else
                return ROTATE_PLATFORM_FAILURE;
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }

    int flush_receiver_buffer(void)
    {
        try
        {
            std::vector<unsigned char> buffer;
            serial_port->Read(buffer);

                return ROTATE_PLATFORM_SUCCESS;
        }
        catch(...)
        {
            std::cout << "READ WRITE ERROR ON SERIAL PORT\n";
        }
    }
};





#endif
