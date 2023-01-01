//
// Created by meng on 2021/2/24.
//
#include "eskf_flow.h"
#include "tool.h"

#include <fstream>
#include <yaml-cpp/yaml.h>
Eigen::Matrix4d Vector2Matrix(const Eigen::Vector3d& vec){
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    matrix.block<3,1>(0,3) = vec;

    return matrix;
}

ESKFFlow::ESKFFlow(const std::string &work_space_path)
        : work_space_path_(work_space_path){

    std::string config_file_path = work_space_path_ + "/config/config.yaml";
    YAML::Node config_node = YAML::LoadFile(config_file_path);
    eskf_ptr_ = std::make_shared<ESKF>(config_node);
}

bool ESKFFlow::ReadData() 
{
    const std::string data_path = work_space_path_ + "/data/raw_data";

    if (imu_flow_ptr_->ReadIMUData(data_path, imu_data_buff_) &&
        gps_flow_ptr_->ReadGPSData(data_path, gps_data_buff_))
    {
        return false;
    }

    return false;
}


bool ESKFFlow::ValidGPSAndIMUData() {
    curr_imu_data_ = imu_data_buff_.front();
    curr_gps_data_ = gps_data_buff_.front();

    double delta_time = curr_imu_data_.time - curr_gps_data_.time;

    if (delta_time > 0.01){
        gps_data_buff_.pop_front();//pop_front删除列表的第一个
        return false;
    }

    if (delta_time < -0.01){
        imu_data_buff_.pop_front();
        return false;
    }

    imu_data_buff_.pop_front();
    gps_data_buff_.pop_front();

    return true;
}

bool ESKFFlow::ValidIMUData() {
    curr_imu_data_ = imu_data_buff_.front();
    imu_data_buff_.front();//front提取出列表的第一个

    return true;
}

bool ESKFFlow::ValidGPSData() {
    curr_gps_data_ = gps_data_buff_.front();
    gps_data_buff_.pop_front();

    return true;
}

bool ESKFFlow::Run() 
{
    ReadData();
    std::cout<<" enter run"<<std::endl;
//只要IMU和GPS数据缓冲区内还有数据，while循环就不会停止
    while (!imu_data_buff_.empty() && !gps_data_buff_.empty())//empty函数用于测试双段队列是否为空，为空返回true，不为空则返回0
    {
        if (!ValidGPSAndIMUData())
        {
            continue;
        } else//当两类数据时间戳对齐的时候就会进入init函数
        {
            eskf_ptr_->Init(curr_gps_data_, curr_imu_data_);//初始化
            break;
        }
    }

//三个应该是输出
    std::ofstream gt_file(work_space_path_+"/data/gt.txt", std::ios::trunc);//gt: ground truth 真实的无误差值
    std::ofstream fused_file(work_space_path_+"/data/fused.txt", std::ios::trunc);//fused： 通过滤波器融合之后的结果
    std::ofstream measured_file(work_space_path_+"/data/measured.txt", std::ios::trunc);//measured：直接从触感器那里获得的测量值

//当IMU和GPS数据缓冲区有数据时才能进入while循环
    while (!imu_data_buff_.empty() && !gps_data_buff_.empty())
    {
        curr_imu_data_ = imu_data_buff_.front();//返回IMU数据的第一个元素
        curr_gps_data_ = gps_data_buff_.front();
        if (curr_imu_data_.time < curr_gps_data_.time)
        {
            std::cout<<"only predict"<<std::endl;
            eskf_ptr_->Predict(curr_imu_data_);//两个时间戳没对齐说明这一时刻没有GPS的数据输入，只进行预测
            imu_data_buff_.pop_front();
        } else
        {
            std::cout<<"predict and correct"<<std::endl;
            eskf_ptr_->Predict(curr_imu_data_);//Kalman预测部分
            imu_data_buff_.pop_front();

            eskf_ptr_->Correct(curr_gps_data_);//Kalman校正部分
            std::cout<<"start correct"<<std::endl;
            SavePose(fused_file, eskf_ptr_->GetPose());
            SavePose(measured_file,Vector2Matrix(curr_gps_data_.position_ned));

            SavePose(gt_file, Vector2Matrix(GPSFlow::LLA2NED(curr_gps_data_.true_position_lla)));
            gps_data_buff_.pop_front();
        }

        if (use_observability_analysis_) {
            Eigen::Matrix<double, 15, 15> F;
            Eigen::Matrix<double, 3, 15> G;
            Eigen::Matrix<double, 3, 1> Y;
            eskf_ptr_->GetFGY(F, G, Y);
            observability_analysis.SaveFG(F, G, Y, curr_gps_data_.time);
        }
    }

    if (use_observability_analysis_) {
        observability_analysis.ComputeSOM();
        observability_analysis.ComputeObservability();
    }
    return true;
}

bool ESKFFlow::TestRun() {
    ReadData();

    while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
        if (!ValidGPSAndIMUData()) {
            continue;
        } else {
            eskf_ptr_->Init(curr_gps_data_, curr_imu_data_);
            std::cout << "\ntime: " << curr_gps_data_.time << std::endl;
            std::cout << "vel: " << eskf_ptr_->GetVelocity().transpose() << std::endl;
            std::cout << "measure vel: " << curr_gps_data_.velocity.transpose() << std::endl;
            std::cout << "true vel: " << curr_gps_data_.true_velocity.transpose() << std::endl;
            std::cout << "time: " << curr_gps_data_.time << std::endl;
            break;
        }
    }

    std::ofstream gt_file(work_space_path_ + "/data/gt.txt", std::ios::trunc);
    std::ofstream fused_file(work_space_path_ + "/data/fused.txt", std::ios::trunc);
    std::ofstream measured_file(work_space_path_ + "/data/measured.txt", std::ios::trunc);

    while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
        curr_imu_data_ = imu_data_buff_.front();
        curr_gps_data_ = gps_data_buff_.front();
            eskf_ptr_->Predict(curr_imu_data_);
            imu_data_buff_.pop_front();
            SavePose(fused_file, eskf_ptr_->GetPose());
    }
    return true;
}

void ESKFFlow::SavePose(std::ofstream &ofs, const Eigen::Matrix4d &pose) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ofs << pose(i, j);

            if (i == 2 && j == 3) {
                ofs << std::endl;
            } else {
                ofs << " ";
            }
        }
    }
}

