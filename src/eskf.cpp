//
// Created by meng on 2021/2/19.
//
#include "eskf.h"
#include "../3rd/sophus/se3.hpp"

constexpr double kDegree2Radian = M_PI / 180.0;

Eigen::Matrix3d BuildSkewMatrix(const Eigen::Vector3d& vec){
    Eigen::Matrix3d matrix;
    matrix << 0.0,     -vec[2],   vec[1],
              vec[2],    0.0,     -vec[0],
              -vec[1],   vec[0],    0.0;

    return matrix;
}

ESKF::ESKF(const YAML::Node &node) {
    double gravity = node["earth"]["gravity"].as<double>();
    double earth_rotation_speed = node["earth"]["rotation_speed"].as<double>();
    double cov_prior_posi = node["covariance"]["prior"]["posi"].as<double>();
    double cov_prior_vel = node["covariance"]["prior"]["vel"].as<double>();
    double cov_prior_ori = node["covariance"]["prior"]["ori"].as<double>();
    double cov_prior_epsilon = node["covariance"]["prior"]["epsilon"].as<double>();
    double cov_prior_delta = node["covariance"]["prior"]["delta"].as<double>();
    double cov_measurement_posi = node["covariance"]["measurement"]["posi"].as<double>();
    double cov_process_gyro = node["covariance"]["process"]["gyro"].as<double>();
    double cov_process_accel = node["covariance"]["process"]["accel"].as<double>();
    L_ = node["earth"]["latitude"].as<double>();
    g_ = Eigen::Vector3d(0.0, 0.0, -gravity);
    w_ = Eigen::Vector3d(0.0, earth_rotation_speed * cos(L_ * kDegree2Radian),
                         earth_rotation_speed * sin(L_ * kDegree2Radian));

    SetCovarianceP(cov_prior_posi, cov_prior_vel, cov_prior_ori,
                   cov_prior_epsilon, cov_prior_delta);
    SetCovarianceR(cov_measurement_posi);
    SetCovarianceQ(cov_process_gyro, cov_process_accel);

    X_.setZero();
    F_.setZero();
    C_.setIdentity();
    G_.block<3,3>(INDEX_MEASUREMENT_POSI,INDEX_MEASUREMENT_POSI) = Eigen::Matrix3d::Identity();

    F_.block<3,3>(INDEX_STATE_POSI, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity();
    F_.block<3,3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = BuildSkewMatrix(-w_);
}

void ESKF::SetCovarianceQ(double gyro_noise, double accel_noise) {
    Q_.setZero();
    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * gyro_noise * gyro_noise;
    Q_.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * accel_noise * accel_noise;
}

void ESKF::SetCovarianceR(double posi_noise) {
    R_.setZero();
    R_ = Eigen::Matrix3d::Identity() * posi_noise * posi_noise;
}

void ESKF::SetCovarianceP(double posi_noise, double velo_noise, double ori_noise,
                          double gyro_noise, double accel_noise) {
    P_.setZero();
    P_.block<3,3>(INDEX_STATE_POSI, INDEX_STATE_POSI) = Eigen::Matrix3d::Identity() * posi_noise;
    P_.block<3,3>(INDEX_STATE_VEL, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity() * velo_noise;
    P_.block<3,3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = Eigen::Matrix3d::Identity() * ori_noise;
    P_.block<3,3>(INDEX_STATE_GYRO_BIAS, INDEX_STATE_GYRO_BIAS) = Eigen::Matrix3d::Identity() * gyro_noise;
    P_.block<3,3>(INDEX_STATE_ACC_BIAS, INDEX_STATE_ACC_BIAS) = Eigen::Matrix3d::Identity() * accel_noise;
}

bool ESKF::Init(const GPSData &curr_gps_data, const IMUData &curr_imu_data) {
    init_velocity_ = curr_gps_data.true_velocity;
    velocity_ = init_velocity_;
    //Q 四元数，由欧拉角旋转过来==等价于先转X轴->Y轴->Z轴（先转后乘）X轴180度，Y轴0度，Z轴90度
    Eigen::Quaterniond Q = Eigen::AngleAxisd(90 * kDegree2Radian, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(180 * kDegree2Radian, Eigen::Vector3d::UnitX());
    init_pose_.block<3,3>(0,0) = Q.toRotationMatrix();//初始时刻（Init）IMU body frame 到导航坐标系的变换矩阵就等于Cbn
    pose_ = init_pose_;

    imu_data_buff_.clear();
    imu_data_buff_.push_back(curr_imu_data);

    curr_gps_data_ = curr_gps_data;

    return true;
}

void ESKF::GetFGY(TypeMatrixF &F, TypeMatrixG &G, TypeVectorY &Y) {
    F = Ft_;
    G = G_;
    Y = Y_;
}

bool ESKF::Correct(const GPSData &curr_gps_data) {
    curr_gps_data_ = curr_gps_data;

    Y_ = pose_.block<3,1>(0,3) - curr_gps_data.position_ned;

    K_ = P_ * G_.transpose() * (G_ * P_ * G_.transpose() + C_ * R_ * C_.transpose()).inverse();//卡尔曼增益Kk

    P_ = (TypeMatrixP::Identity() - K_ * G_) * P_;//更新后验误差的协防差矩阵
    X_ = X_ + K_ * (Y_ - G_ * X_);//后验估计

    EliminateError();

    ResetState();

    return true;
}

bool ESKF::Predict(const IMUData &curr_imu_data) //误差状态kalman滤波的预测部分
{
    imu_data_buff_.push_back(curr_imu_data);//push_back，在imu_buff的尾部插入一个数据

    UpdateOdomEstimation();

    double delta_t = curr_imu_data.time - imu_data_buff_.front().time;//两次预测之间的时间差

    Eigen::Vector3d curr_accel = pose_.block<3, 3>(0, 0)//eigen::block矩阵块操作    
    //将pose矩阵（4x4）中（从第0行第0列开始）的大小为<3x3的矩阵>提出来
                                 * curr_imu_data.linear_accel;//imu的线加速度

    UpdateErrorState(delta_t, curr_accel);

    imu_data_buff_.pop_front();
    return true;
}

bool ESKF::UpdateErrorState(double t, const Eigen::Vector3d &accel) {
    Eigen::Matrix3d F_23 = BuildSkewMatrix(accel);//IMU测量的加速度的矩阵

    F_.block<3,3>(INDEX_STATE_VEL/*3*/, INDEX_STATE_ORI/*6*/) = F_23;//放在Ft矩阵的第[3,6]位置开始的F23
    F_.block<3,3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS/*12*/) = pose_.block<3,3>(0,0);//Cbn
    F_.block<3,3>(INDEX_STATE_ORI/*6*/, INDEX_STATE_GYRO_BIAS/*9*/) = -pose_.block<3,3>(0,0);//Cbn
    B_.block<3,3>(INDEX_STATE_VEL, 3) = pose_.block<3,3>(0,0);//Cbn
    B_.block<3,3>(INDEX_STATE_ORI, 0) = -pose_.block<3,3>(0,0);//-Cbn

    TypeMatrixF Fk = TypeMatrixF::Identity() + F_ * t;//Ft的离散化，使用一阶泰勒近似
    TypeMatrixB Bk = B_ * t;

    Ft_ = F_ * t;

    X_ = Fk * X_;//ESKF的预测方程
    P_ = Fk * P_ * Fk.transpose() + Bk * Q_ * Bk.transpose();//先验误差的协防差矩阵

    return true;
}

bool ESKF::UpdateOdomEstimation() {
    Eigen::Vector3d angular_delta;
    ComputeAngularDelta(angular_delta);//计算角度变化

    Eigen::Matrix3d R_nm_nm_1;
    ComputeEarthTranform(R_nm_nm_1);

    Eigen::Matrix3d curr_R, last_R;
    ComputeOrientation(angular_delta, R_nm_nm_1, curr_R, last_R);

    Eigen::Vector3d curr_vel, last_vel;
    ComputeVelocity(curr_vel, last_vel, curr_R, last_R);

    ComputePosition(curr_vel, last_vel);

    return true;
}

bool ESKF::ComputeAngularDelta(Eigen::Vector3d &angular_delta) //计算delta角度
{
    IMUData curr_imu_data = imu_data_buff_.at(1);//curr的时间是这一次的时间戳时刻
    IMUData last_imu_data = imu_data_buff_.at(0);//last的时间是上一次的时间时刻
    //因为pop_front是在计算完了之后才执行，所以进入位姿更新的时候还没有去除上一时刻
    double delta_t = curr_imu_data.time - last_imu_data.time;

    if (delta_t <= 0){
        return false;
    }

    Eigen::Vector3d curr_angular_vel = curr_imu_data.angle_velocity;

    Eigen::Vector3d last_angular_vel = last_imu_data.angle_velocity;

    Eigen::Vector3d curr_unbias_angular_vel = curr_angular_vel;
    Eigen::Vector3d last_unbias_angular_vel = last_angular_vel;

    angular_delta = 0.5 * (curr_unbias_angular_vel + last_unbias_angular_vel) * delta_t;//两次角速度的平均乘以运动时间

    return true;
}

bool ESKF::ComputeEarthTranform(Eigen::Matrix3d &R_nm_nm_1) {
    IMUData curr_imu_data = imu_data_buff_.at(1);
    IMUData last_imu_data = imu_data_buff_.at(0);
    std::cout<<"start earth"<<std::endl;
    double delta_t = curr_imu_data.time - last_imu_data.time;

    constexpr double rm = 6353346.18315;
    constexpr double rn = 6384140.52699;
    Eigen::Vector3d w_en_n(-velocity_[1] / (rm + curr_gps_data_.position_lla[2]),
                           velocity_[0] / (rn + curr_gps_data_.position_lla[2]),
                           velocity_[0] / (rn + curr_gps_data_.position_lla[2])
                           * std::tan(curr_gps_data_.position_lla[0] * kDegree2Radian));

    Eigen::Vector3d w_in_n = w_en_n + w_;

    auto angular = delta_t * w_in_n;

    Eigen::AngleAxisd angle_axisd(angular.norm(), angular.normalized());

    R_nm_nm_1 = angle_axisd.toRotationMatrix().transpose();
    std::cout<<"earth ok"<<std::endl;
    return true;
}

bool ESKF::ComputeOrientation(const Eigen::Vector3d &angular_delta,
                              const Eigen::Matrix3d R_nm_nm_1,
                              Eigen::Matrix3d &curr_R,
                              Eigen::Matrix3d &last_R) 
{
    Eigen::AngleAxisd angle_axisd(angular_delta.norm(), angular_delta.normalized());
    last_R = pose_.block<3, 3>(0, 0);

    curr_R = R_nm_nm_1 * pose_.block<3, 3>(0, 0) * angle_axisd.toRotationMatrix();

    pose_.block<3, 3>(0, 0) = curr_R;

    return true;
}

bool ESKF::ComputeVelocity(Eigen::Vector3d &curr_vel, Eigen::Vector3d& last_vel,
                                             const Eigen::Matrix3d &curr_R,
                                             const Eigen::Matrix3d last_R) {
    IMUData curr_imu_data = imu_data_buff_.at(1);
    IMUData last_imu_data = imu_data_buff_.at(0);
    double delta_t = curr_imu_data.time - last_imu_data.time;
    if (delta_t <=0 ){
        return false;
    }

    Eigen::Vector3d curr_accel = curr_imu_data.linear_accel;
    Eigen::Vector3d curr_unbias_accel = GetUnbiasAccel(curr_R * curr_accel);

    Eigen::Vector3d last_accel = last_imu_data.linear_accel;
    Eigen::Vector3d last_unbias_accel = GetUnbiasAccel(last_R * last_accel);

    last_vel = velocity_;

    velocity_ += delta_t * 0.5 * (curr_unbias_accel + last_unbias_accel);
    curr_vel = velocity_;

    return true;
}

Eigen::Vector3d ESKF::GetUnbiasAccel(const Eigen::Vector3d &accel) {
//    return accel - accel_bias_ + g_;
    return accel + g_;
}

bool ESKF::ComputePosition(const Eigen::Vector3d& curr_vel, const Eigen::Vector3d& last_vel){
    double delta_t = imu_data_buff_.at(1).time - imu_data_buff_.at(0).time;

    pose_.block<3,1>(0,3) += 0.5 * delta_t * (curr_vel + last_vel);

    return true;
}

void ESKF::ResetState() {
    X_.setZero();
}

void ESKF::EliminateError() {
    pose_.block<3,1>(0,3) = pose_.block<3,1>(0,3) - X_.block<3,1>(INDEX_STATE_POSI, 0);

    velocity_ = velocity_ - X_.block<3,1>(INDEX_STATE_VEL, 0);
    Eigen::Matrix3d C_nn = Sophus::SO3d::exp(X_.block<3,1>(INDEX_STATE_ORI, 0)).matrix();
    pose_.block<3,3>(0,0) = C_nn * pose_.block<3,3>(0,0);

    gyro_bias_ = gyro_bias_ - X_.block<3,1>(INDEX_STATE_GYRO_BIAS, 0);
    accel_bias_ = accel_bias_ - X_.block<3,1>(INDEX_STATE_ACC_BIAS, 0);
}

Eigen::Matrix4d ESKF::GetPose() const {
    return pose_;
}