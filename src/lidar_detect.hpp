/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#define PCL_NO_PRECOMPILE

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include "common_lib.h"
#include <limits>

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;

    // 存储中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr filtered_cloud_;
    pcl::PointCloud<Common::Point>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

    struct CircleFitResult
    {
        double x = 0.0;
        double y = 0.0;
        double radius = 0.0;
        double mean_abs_error = std::numeric_limits<double>::max();
        bool valid = false;
    };

    float percentile(std::vector<float> values, double ratio) const
    {
        if (values.empty()) return 0.0f;
        ratio = std::max(0.0, std::min(1.0, ratio));
        const size_t idx = static_cast<size_t>(std::round(ratio * static_cast<double>(values.size() - 1)));
        std::nth_element(values.begin(), values.begin() + idx, values.end());
        return values[idx];
    }

    bool fitCircleRobust(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                         CircleFitResult& result) const
    {
        if (!cluster || cluster->size() < 8) return false;

        Eigen::MatrixXd A(cluster->size(), 3);
        Eigen::VectorXd b(cluster->size());
        for (size_t i = 0; i < cluster->size(); ++i)
        {
            const double x = cluster->points[i].x;
            const double y = cluster->points[i].y;
            A(static_cast<int>(i), 0) = x;
            A(static_cast<int>(i), 1) = y;
            A(static_cast<int>(i), 2) = 1.0;
            b(static_cast<int>(i)) = -(x * x + y * y);
        }

        Eigen::Vector3d algebraic = A.colPivHouseholderQr().solve(b);
        double cx = -0.5 * algebraic(0);
        double cy = -0.5 * algebraic(1);
        double r2 = cx * cx + cy * cy - algebraic(2);
        if (!std::isfinite(r2) || r2 <= 0.0) return false;
        double radius = std::sqrt(r2);

        const double huber_delta = 0.02;
        for (int iter = 0; iter < 20; ++iter)
        {
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            Eigen::Vector3d g = Eigen::Vector3d::Zero();

            for (const auto& p : *cluster)
            {
                const double dx = cx - static_cast<double>(p.x);
                const double dy = cy - static_cast<double>(p.y);
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < 1e-8) continue;

                const double residual = dist - radius;
                const double abs_residual = std::fabs(residual);
                const double weight = (abs_residual <= huber_delta) ? 1.0 : (huber_delta / abs_residual);

                Eigen::Vector3d J(dx / dist, dy / dist, -1.0);
                H += weight * J * J.transpose();
                g += weight * J * residual;
            }

            Eigen::Vector3d step = H.ldlt().solve(-g);
            if (!step.allFinite()) return false;

            cx += step(0);
            cy += step(1);
            radius += step(2);
            if (radius <= 0.0 || !std::isfinite(radius)) return false;

            if (step.norm() < 1e-7) break;
        }

        double error_sum = 0.0;
        int valid_count = 0;
        for (const auto& p : *cluster)
        {
            const double dx = static_cast<double>(p.x) - cx;
            const double dy = static_cast<double>(p.y) - cy;
            const double dist = std::sqrt(dx * dx + dy * dy);
            if (!std::isfinite(dist)) continue;
            error_sum += std::fabs(dist - radius);
            ++valid_count;
        }

        if (valid_count == 0) return false;
        result.x = cx;
        result.y = cy;
        result.radius = radius;
        result.mean_abs_error = error_sum / static_cast<double>(valid_count);
        result.valid = std::isfinite(result.x) && std::isfinite(result.y) &&
                       std::isfinite(result.radius) && std::isfinite(result.mean_abs_error);
        return result.valid;
    }

    bool selectGeometryConsistentCenters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& candidates,
                                         std::vector<int>& selected_indices) const
    {
        selected_indices.clear();
        if (!candidates || candidates->size() < TARGET_NUM_CIRCLES) return false;

        std::vector<std::vector<int>> groups;
        comb(candidates->size(), TARGET_NUM_CIRCLES, groups);

        int best_candidate_idx = -1;
        double best_candidate_score = std::numeric_limits<double>::max();

        for (int i = 0; i < static_cast<int>(groups.size()); ++i)
        {
            std::vector<pcl::PointXYZ> candidate_group;
            candidate_group.reserve(TARGET_NUM_CIRCLES);
            for (int idx : groups[i])
            {
                candidate_group.push_back(candidates->points[idx]);
            }

            Square square_candidate(candidate_group, delta_width_circles_, delta_height_circles_);
            if (!square_candidate.is_valid()) continue;

            pcl::PointCloud<pcl::PointXYZ>::Ptr sorted(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& p : candidate_group) tmp->push_back(p);
            sortPatternCenters(tmp, sorted, "camera");

            const double s01 = distance3D(sorted->points[0], sorted->points[1]);
            const double s12 = distance3D(sorted->points[1], sorted->points[2]);
            const double s23 = distance3D(sorted->points[2], sorted->points[3]);
            const double s30 = distance3D(sorted->points[3], sorted->points[0]);

            const double score_width_height =
                std::pow(s01 - delta_width_circles_, 2) +
                std::pow(s12 - delta_height_circles_, 2) +
                std::pow(s23 - delta_width_circles_, 2) +
                std::pow(s30 - delta_height_circles_, 2);
            const double score_height_width =
                std::pow(s01 - delta_height_circles_, 2) +
                std::pow(s12 - delta_width_circles_, 2) +
                std::pow(s23 - delta_height_circles_, 2) +
                std::pow(s30 - delta_width_circles_, 2);
            const double score = std::min(score_width_height, score_height_width);
            if (score < best_candidate_score)
            {
                best_candidate_score = score;
                best_candidate_idx = i;
            }
        }

        if (best_candidate_idx < 0) return false;
        selected_indices = groups[best_candidate_idx];
        return true;
    }

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    LidarDetect(ros::NodeHandle &nh, Params &params)
        : filtered_cloud_(new pcl::PointCloud<Common::Point>),
          plane_cloud_(new pcl::PointCloud<Common::Point>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        delta_width_circles_ = params.delta_width_circles;
        delta_height_circles_ = params.delta_height_circles;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    void extractHighIntensityRingPoints(const pcl::PointCloud<Common::Point>::Ptr& plane_cloud,const pcl::ModelCoefficients::Ptr& plane_coefficients,pcl::PointCloud<pcl::PointXYZ>::Ptr ring_cloud)
    {
        ring_cloud->clear();

        if (!plane_cloud || plane_cloud->empty())
        {
            ROS_WARN("[LiDAR] Empty plane cloud, cannot extract high-intensity ring points.");
            return;
        }
        if (!plane_coefficients || plane_coefficients->values.size() < 4)
        {
            ROS_WARN("[LiDAR] Invalid plane coefficients.");
            return;
        }

        const auto& c = plane_coefficients->values;
        const double norm_n = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);

        if (norm_n < 1e-6)
        {
            ROS_WARN("[LiDAR] Invalid plane normal.");
            return;
        }

        const double plane_distance_threshold = 0.03;

        std::vector<float> intensities;
        intensities.reserve(plane_cloud->size());

        for (const auto& p : *plane_cloud)
        {
            double dist_plane = std::fabs(c[0] * p.x + c[1] * p.y + c[2] * p.z + c[3]) / norm_n;
            if (dist_plane < plane_distance_threshold)
            {
                intensities.push_back(p.intensity);
            }
        }

        if (intensities.size() < 20)
        {
            ROS_WARN("[LiDAR] Too few plane points for intensity segmentation.");
            return;
        }

        auto minmax = std::minmax_element(intensities.begin(), intensities.end());
        const float min_i = *minmax.first;
        const float max_i = *minmax.second;

        if (max_i - min_i < 1e-3f)
        {
            ROS_WARN("[LiDAR] Intensity range too small: %.3f - %.3f", min_i, max_i);
            return;
        }

        const int bins = 128;
        std::vector<int> hist(bins, 0);

        for (float v : intensities)
        {
            int b = static_cast<int>((v - min_i) / (max_i - min_i) * (bins - 1));
            b = std::max(0, std::min(bins - 1, b));
            hist[b]++;
        }

        const double total = static_cast<double>(intensities.size());
        double sum_total = 0.0;

        for (int i = 0; i < bins; ++i)
        {
            sum_total += static_cast<double>(i * hist[i]);
        }

        double sum_background = 0.0;
        double weight_background = 0.0;
        double max_between_class_variance = -1.0;
        int best_bin = 0;

        for (int i = 0; i < bins; ++i)
        {
            weight_background += hist[i];
            if (weight_background <= 0.0) continue;

            double weight_foreground = total - weight_background;
            if (weight_foreground <= 0.0) break;

            sum_background += static_cast<double>(i * hist[i]);

            double mean_background = sum_background / weight_background;
            double mean_foreground = (sum_total - sum_background) / weight_foreground;
            double between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) * (mean_background - mean_foreground);

            if (between_class_variance > max_between_class_variance)
            {
                max_between_class_variance = between_class_variance;
                best_bin = i;
            }
        }

        const float otsu_threshold = min_i + (max_i - min_i) * static_cast<float>(best_bin) / static_cast<float>(bins - 1);

        std::vector<float> foreground_intensities;
        foreground_intensities.reserve(intensities.size());
        for (float v : intensities)
        {
            if (v >= otsu_threshold) foreground_intensities.push_back(v);
        }

        if (foreground_intensities.size() < 20)
        {
            ROS_WARN("[LiDAR] Too few foreground points after Otsu intensity segmentation.");
            return;
        }

        const float foreground_low = percentile(foreground_intensities, 0.15);
        const float high_quantile = percentile(intensities, 0.92);
        const float relative_high = otsu_threshold + 0.55f * (max_i - otsu_threshold);

        // 新标定板：圆环强度明显高于背景。Otsu 先分出前景，再用前景低分位
        // 和强度范围内的相对高强度侧约束阈值，避免固定 margin 受雷达强度尺度影响。
        const float ring_threshold = std::max(std::max(otsu_threshold, foreground_low), relative_high);

        for (const auto& p : *plane_cloud)
        {
            double dist_plane = std::fabs(c[0] * p.x + c[1] * p.y + c[2] * p.z + c[3]) / norm_n;
            if (dist_plane >= plane_distance_threshold) continue;
            if (p.intensity >= ring_threshold)
            {
                ring_cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
            }
        }

        ROS_INFO("[LiDAR] Intensity range: %.3f - %.3f", min_i, max_i);
        ROS_INFO("[LiDAR] Otsu threshold: %.3f, foreground p15: %.3f, p92: %.3f, relative high: %.3f, ring threshold: %.3f",
                 otsu_threshold, foreground_low, high_quantile, relative_high, ring_threshold);
        ROS_INFO("[LiDAR] Extracted %zu high-intensity ring points.", ring_cloud->size());
    }

    void detect_mech_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);

        ROS_INFO("Depth filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 拟合平面，提取法向量
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
    
        // // 3. 根据每条ring相邻点距离提取边缘点
        // edge_cloud_->reserve(filtered_cloud_->size());

        // // 先按 ring 分组
        // std::unordered_map<unsigned int, std::vector<int>> ring2indices;
        // ring2indices.reserve(64);

        // for (int i = 0; i < static_cast<int>(filtered_cloud_->size()); ++i)
        // {
        //     const auto &pt = filtered_cloud_->points[i];
        //     ring2indices[pt.ring].push_back(i);
        // }

        // const auto &c = plane_coefficients->values; // [a, b, c, d]
        // Eigen::Vector3d n(c[0], c[1], c[2]);
        // double norm_n = n.norm();
        // Eigen::Vector3d normal = n / norm_n;

        // // 在每条 ring 内，用相邻点距离检测跳变点作为边缘点
        // const double neighbor_gap_threshold = 0.10;  // 邻近点距离阈值
        // const int    min_points_per_ring    = 10;    // 太短的 ring 不处理

        // for (auto &kv : ring2indices)
        // {
        //     auto &idx_vec = kv.second;
        //     if (static_cast<int>(idx_vec.size()) < min_points_per_ring) continue;

        //     for (size_t k = 1; k + 1 < idx_vec.size(); ++k)
        //     {
        //         const auto &p_prev = filtered_cloud_->points[idx_vec[k - 1]];
        //         const auto &p_cur  = filtered_cloud_->points[idx_vec[k]];
        //         const auto &p_next = filtered_cloud_->points[idx_vec[k + 1]];

        //         // 只保留落在拟合平面附近的点
        //         double dist_plane = std::fabs(c[0]*p_cur.x + c[1]*p_cur.y + c[2]*p_cur.z + c[3]) / norm_n;
        //         if (dist_plane >= 0.03) continue;

        //         // cur 与 prev 的距离
        //         double dx1 = static_cast<double>(p_cur.x) - static_cast<double>(p_prev.x);
        //         double dy1 = static_cast<double>(p_cur.y) - static_cast<double>(p_prev.y);
        //         double dz1 = static_cast<double>(p_cur.z) - static_cast<double>(p_prev.z);
        //         double dist_prev = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        //         // cur 与 next 的距离
        //         double dx2 = static_cast<double>(p_cur.x) - static_cast<double>(p_next.x);
        //         double dy2 = static_cast<double>(p_cur.y) - static_cast<double>(p_next.y);
        //         double dz2 = static_cast<double>(p_cur.z) - static_cast<double>(p_next.z);
        //         double dist_next = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        //         // 只要和前一个或后一个之间有一个距离超过阈值，就认为当前点是边缘点
        //         if (dist_prev > neighbor_gap_threshold || dist_next > neighbor_gap_threshold)
        //         {
        //             edge_cloud_->push_back(pcl::PointXYZ(p_cur.x, p_cur.y, p_cur.z));
        //         }
        //     }
        // }

        // ROS_INFO("Extracted %zu edge points (mechanical LiDAR by neighbor distance).", edge_cloud_->size());

        // 3. 根据强度提取圆环点
        edge_cloud_->clear();
        edge_cloud_->reserve(plane_cloud_->size());
        extractHighIntensityRingPoints(plane_cloud_, plane_coefficients, edge_cloud_);
        ROS_INFO("Extracted %zu high-intensity ring points.", edge_cloud_->size());
        if (edge_cloud_->size() < 12)
        {
            ROS_WARN("[LiDAR] Too few ring points after intensity extraction.");
            return;
        }

        // // 4. 将边缘点对齐到 Z=0 平面
        // aligned_cloud_->reserve(edge_cloud_->size());

        // Eigen::Vector3d z_axis(0, 0, 1);
        // Eigen::Vector3d axis = normal.cross(z_axis);
        // double angle = acos(normal.dot(z_axis));

        // Eigen::AngleAxisd rotation(angle, axis);
        // Eigen::Matrix3d R_align = rotation.toRotationMatrix();

        // float average_z = 0.0;
        // int cnt = 0;
        // for (const auto& pt : *edge_cloud_) {
        //     Eigen::Vector3d point(pt.x, pt.y, pt.z);
        //     Eigen::Vector3d aligned_point = R_align * point;
        //     aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
        //     average_z += aligned_point.z();
        //     cnt++;
        // }
        // average_z /= cnt;

        // 4. 将圆环点对齐到 Z=0 平面
        aligned_cloud_->reserve(edge_cloud_->size());

        const auto &c = plane_coefficients->values; // [a, b, c, d]
        Eigen::Vector3d n(c[0], c[1], c[2]);
        double norm_n = n.norm();

        if (norm_n < 1e-6)
        {
            ROS_WARN("[LiDAR] Invalid plane normal.");
            return;
        }

        Eigen::Vector3d normal = n / norm_n;

        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);

        double dot = std::max(-1.0, std::min(1.0, normal.dot(z_axis)));
        double angle = std::acos(dot);

        Eigen::Matrix3d R_align = Eigen::Matrix3d::Identity();

        if (axis.norm() > 1e-6)
        {
            axis.normalize();
            Eigen::AngleAxisd rotation(angle, axis);
            R_align = rotation.toRotationMatrix();
        }

        float average_z = 0.0;
        int cnt = 0;

        for (const auto& pt : *edge_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }

        if (cnt == 0)
        {
            ROS_WARN("[LiDAR] No ring points to align.");
            return;
        }
        average_z /= cnt;

        // 5. 在对齐后的点云中检测圆形，提取圆心
        
        // 拷贝一份工作点云，后面要不停“删掉已拟合的圆”
        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));

        ROS_INFO("[LiDAR] Start circle detection, initial cloud size = %zu", xy_cloud->points.size());

        // 在对齐后的平面上，用 RANSAC 反复检测 2D 圆
        pcl::SACSegmentation<pcl::PointXYZ> circle_segmentation;
        circle_segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
        circle_segmentation.setMethodType(pcl::SAC_RANSAC);
        circle_segmentation.setDistanceThreshold(0.02);
        circle_segmentation.setOptimizeCoefficients(true);
        circle_segmentation.setMaxIterations(1000);
        circle_segmentation.setRadiusLimits(circle_radius_ - 0.03, circle_radius_ + 0.03);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ExtractIndices<pcl::PointXYZ> extract2;

        // 不停在剩余点云中找圆
        while (xy_cloud->points.size() > 3)
        {
            ROS_INFO("[LiDAR] RANSAC on cloud of size %lu", xy_cloud->points.size());

            circle_segmentation.setInputCloud(xy_cloud);
            circle_segmentation.segment(*inliers, *coefficients);

            // 没有 inliers，说明没有可用的圆，结束
            if (inliers->indices.empty())
            {
                ROS_INFO("[LiDAR] No more circles can be found, stop.");
                break;
            }

            // 内点太少就认为是噪声，直接结束
            if ((int)inliers->indices.size() < 5)
            {
                ROS_INFO("[LiDAR] Found circle but inliers too few (%zu < 3), stop.",
                            inliers->indices.size());
                break;
            }

            // ROS_INFO("[LiDAR] Circle found: inliers = %zu, coeffs size = %zu",
            //         inliers->indices.size(), coefficients->values.size());
            // 对 Circle2D 而言，coeffs 通常是 [xc, yc, r]
            // if (coefficients->values.size() >= 3)
            // {
            //     ROS_INFO("[LiDAR]   center = (%.4f, %.4f), r = %.4f",
            //             coefficients->values[0],
            //             coefficients->values[1],
            //             coefficients->values[2]);
            // }
            
            // 记录当前这个圆
            pcl::PointXYZ center_point;
            center_point.x = coefficients->values[0];
            center_point.y = coefficients->values[1];
            center_point.z = 0;
            center_z0_cloud_->push_back(center_point);

            // 把当前圆的 inliers 从点云中移除，继续在剩余点中找
            extract2.setInputCloud(xy_cloud);
            extract2.setIndices(inliers);
            extract2.setNegative(true);  // 保留非 inliers（即剩余点）
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract2.filter(*remaining);
            xy_cloud.swap(remaining);

            // 清空 inliers，避免下一轮残留
            inliers->indices.clear();
        }

        // 6. Geometric consistency check
        std::vector<std::vector<int>> groups;
        comb(center_z0_cloud_->size(), TARGET_NUM_CIRCLES, groups);
        double groups_scores[groups.size()];  // -1: invalid; 0-1 normalized score
        for (int i = 0; i < groups.size(); ++i) 
        {
            std::vector<pcl::PointXYZ> candidates;
            // Build candidates set
            for (int j = 0; j < groups[i].size(); ++j) 
            {
                pcl::PointXYZ center;
                center.x = center_z0_cloud_->at(groups[i][j]).x;
                center.y = center_z0_cloud_->at(groups[i][j]).y;
                center.z = center_z0_cloud_->at(groups[i][j]).z;
                candidates.push_back(center);
            }

            // Compute candidates score
            Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
            groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1;  // -1 when it's not valid, 1 otherwise
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (int i = 0; i < groups.size(); ++i) 
        {
            if (best_candidate_score == 1 && groups_scores[i] == 1) 
            {
                // Exit 4: Several candidates fit target's geometry
                ROS_ERROR(
                    "[LiDAR] More than one set of candidates fit target's geometry. "
                    "Please, make sure your parameters are well set. Exiting callback");
                return;
            }
            if (groups_scores[i] > best_candidate_score) 
            {
                best_candidate_score = groups_scores[i];
                best_candidate_idx = i;
            }
        }
        if (best_candidate_idx == -1) 
        {
            // Exit 5: No candidates fit target's geometry
            ROS_WARN(
                "[LiDAR] Unable to find a candidate set that matches target's "
                "geometry");
            return;
        }
        
        // 7. 将选中的圆心逆变换回原始坐标系
        Eigen::Matrix3d R_inv = R_align.inverse();
        for (int j = 0; j < groups[best_candidate_idx].size(); ++j) 
        {
            pcl::PointXYZ center;
            center.x = center_z0_cloud_->at(groups[best_candidate_idx][j]).x;
            center.y = center_z0_cloud_->at(groups[best_candidate_idx][j]).y;
            center.z = center_z0_cloud_->at(groups[best_candidate_idx][j]).z;

            // 将圆心坐标逆变换回原始坐标系
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }

    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();

        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());

        pcl::PointCloud<Common::Point>::Ptr raw_filtered_cloud(new pcl::PointCloud<Common::Point>(*filtered_cloud_));
        
        // 平面拟合可以下采样提速，但高反圆环提取必须回到原始点云，避免 voxel
        // 平均改变 intensity 和圆环采样分布。
        pcl::VoxelGrid<Common::Point> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(std::max(-1.0, std::min(1.0, normal.dot(z_axis))));

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (axis.norm() > 1e-6)
        {
            axis.normalize();
            Eigen::AngleAxisd rotation(angle, axis);
            R = rotation.toRotationMatrix();
        }

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // // 4. 提取边缘点
        // edge_cloud_->reserve(aligned_cloud_->size());

        // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // normal_estimator.setInputCloud(aligned_cloud_);
        // normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        // normal_estimator.compute(*normals);
    
        // pcl::PointCloud<pcl::Boundary> boundaries;
        // pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        // boundary_estimator.setInputCloud(aligned_cloud_);
        // boundary_estimator.setInputNormals(normals);
        // boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        // boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        // boundary_estimator.compute(boundaries);
    
        // for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
        //     if (boundaries.points[i].boundary_point > 0) {
        //         edge_cloud_->push_back(aligned_cloud_->points[i]);
        //     }
        // }
        // ROS_INFO("Extracted %zu edge points.", edge_cloud_->size());

        // 4. 根据强度提取圆环点，并对齐到 Z=0 平面
        pcl::PointCloud<pcl::PointXYZ>::Ptr ring_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        extractHighIntensityRingPoints(raw_filtered_cloud, plane_coefficients, ring_cloud_raw);
        edge_cloud_->clear();
        edge_cloud_->reserve(ring_cloud_raw->size());

        for (const auto& pt : *ring_cloud_raw)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            edge_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
        }
        ROS_INFO("Extracted %zu high-intensity ring points.", edge_cloud_->size());

        if (edge_cloud_->size() < 20)
        {
            ROS_WARN("[LiDAR] Too few ring points after intensity extraction.");
            return;
        }

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02); // 设置聚类距离阈值
        ec.setMinClusterSize(200);    // 最小点数
        ec.setMaxClusterSize(30000);  // 原始点云未下采样时，圆环 cluster 会明显大于 voxel 后结果
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        ROS_INFO("Number of edge clusters: %zu", cluster_indices.size());
    
        // 6. 对每个聚类做全点鲁棒圆环拟合。RANSAC 只选局部内点时容易被环带采样密度带偏。
        pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_centers(new pcl::PointCloud<pcl::PointXYZ>);
        candidate_centers->reserve(cluster_indices.size());
        Eigen::Matrix3d R_inv = R.inverse();
    
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            CircleFitResult fit;
            if (!fitCircleRobust(cluster, fit))
            {
                ROS_WARN("[LiDAR] Circle fit failed for cluster %zu with %zu points.", i, cluster->size());
                continue;
            }

            const bool radius_ok = std::fabs(fit.radius - circle_radius_) < 0.05;
            const bool error_ok = fit.mean_abs_error < 0.03;
            ROS_INFO("[LiDAR] Ring cluster %zu: points=%zu, center=(%.4f, %.4f), radius=%.4f, mean abs error=%.4f",
                     i, cluster->size(), fit.x, fit.y, fit.radius, fit.mean_abs_error);

            if (!radius_ok || !error_ok)
            {
                ROS_WARN("[LiDAR] Reject cluster %zu: radius_ok=%d, error_ok=%d.", i, radius_ok, error_ok);
                continue;
            }

            pcl::PointXYZ center_point;
            center_point.x = static_cast<float>(fit.x);
            center_point.y = static_cast<float>(fit.y);
            center_point.z = 0.0f;
            candidate_centers->push_back(center_point);
        }

        std::vector<int> selected_indices;
        if (!selectGeometryConsistentCenters(candidate_centers, selected_indices))
        {
            ROS_WARN("[LiDAR] Unable to select 4 geometry-consistent ring centers from %zu candidates.",
                     candidate_centers->size());
            return;
        }

        center_z0_cloud_->clear();
        center_z0_cloud_->reserve(TARGET_NUM_CIRCLES);
        center_cloud->clear();
        center_cloud->reserve(TARGET_NUM_CIRCLES);

        for (int idx : selected_indices)
        {
            const pcl::PointXYZ& center_point = candidate_centers->points[idx];
            center_z0_cloud_->push_back(center_point);

            Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }
    // 获取中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
