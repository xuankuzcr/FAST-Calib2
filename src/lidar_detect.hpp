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
#include <unordered_map>

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;
    double board_width_, board_height_, board_roi_margin_, board_roi_depth_;
    double auto_roi_voxel_leaf_, annulus_voxel_leaf_;
    bool use_auto_lidar_roi_;

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

    struct VoxelKey
    {
        long long x = 0;
        long long y = 0;
        long long z = 0;

        // 判断两个体素索引是否指向同一个空间体素
        bool operator==(const VoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash
    {
        // 为体素索引生成哈希值，供 unordered_map 使用
        std::size_t operator()(const VoxelKey& key) const
        {
            std::size_t seed = 0;
            auto hashCombine = [&seed](long long value) {
                seed ^= std::hash<long long>()(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            };
            hashCombine(key.x);
            hashCombine(key.y);
            hashCombine(key.z);
            return seed;
        }
    };

    struct BoardFrame
    {
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX();
        Eigen::Vector3d y_axis = Eigen::Vector3d::UnitY();
        Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    };

    struct PlaneAlignment
    {
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        double average_z = 0.0;
    };

    // 将三维坐标按叶子尺寸映射到整数体素索引
    VoxelKey voxelKey(double x, double y, double z, double leaf) const
    {
        return VoxelKey{
            static_cast<long long>(std::floor(x / leaf)),
            static_cast<long long>(std::floor(y / leaf)),
            static_cast<long long>(std::floor(z / leaf))
        };
    }

    // 计算一组强度值的指定分位数
    float percentile(std::vector<float> values, double ratio) const
    {
        if (values.empty()) return 0.0f;
        ratio = std::max(0.0, std::min(1.0, ratio));
        const size_t idx = static_cast<size_t>(std::round(ratio * static_cast<double>(values.size() - 1)));
        std::nth_element(values.begin(), values.begin() + idx, values.end());
        return values[idx];
    }

    // 判断点的坐标和强度是否都是有限值
    bool isFinitePoint(const Common::Point& p) const
    {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
               std::isfinite(p.intensity);
    }

    // 计算点到给定平面模型的垂直距离
    double planeDistance(const pcl::ModelCoefficients::Ptr& coefficients,
                         const Common::Point& p) const
    {
        const auto& c = coefficients->values;
        const double norm_n = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        if (norm_n < 1e-6) return std::numeric_limits<double>::max();
        return std::fabs(c[0] * p.x + c[1] * p.y + c[2] * p.z + c[3]) / norm_n;
    }

    // 从输入点云中复制有限且距离合理的点
    void copyFiniteRangePoints(const pcl::PointCloud<Common::Point>::Ptr& input,
                               pcl::PointCloud<Common::Point>::Ptr output) const
    {
        output->clear();
        output->reserve(input->size());
        const double min_range = 0.3;
        const double max_range = 50.0;
        for (const auto& p : *input)
        {
            if (!isFinitePoint(p)) continue;
            const double range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (range < min_range || range > max_range) continue;
            output->push_back(p);
        }
    }

    // 按空间体素降采样，每个体素保留强度最高的原始点
    void voxelDownsampleKeepMaxIntensity(const pcl::PointCloud<Common::Point>::Ptr& input,
                                         double leaf_size,
                                         pcl::PointCloud<Common::Point>::Ptr output) const
    {
        output->clear();
        if (!input || input->empty()) return;
        if (leaf_size <= 0.0)
        {
            *output = *input;
            return;
        }

        std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_to_index;
        voxel_to_index.reserve(input->size());

        // 自动 ROI 只需要粗略高反布局；保留最强原始点可避免平均 intensity。
        for (int i = 0; i < static_cast<int>(input->size()); ++i)
        {
            const auto& p = input->points[i];
            if (!isFinitePoint(p)) continue;
            const VoxelKey key = voxelKey(p.x, p.y, p.z, leaf_size);
            auto it = voxel_to_index.find(key);
            if (it == voxel_to_index.end() ||
                p.intensity > input->points[it->second].intensity)
            {
                voxel_to_index[key] = i;
            }
        }

        output->reserve(voxel_to_index.size());
        for (const auto& kv : voxel_to_index)
        {
            output->push_back(input->points[kv.second]);
        }
    }

    // 按空间体素降采样，每个体素保留离体素质心最近的真实测量点
    void voxelDownsampleClosestToCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                                          double leaf_size,
                                          pcl::PointCloud<pcl::PointXYZ>::Ptr output) const
    {
        output->clear();
        if (!input || input->empty()) return;
        if (leaf_size <= 0.0)
        {
            *output = *input;
            return;
        }

        struct VoxelAccumulator
        {
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            std::vector<int> indices;
        };

        std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
        voxels.reserve(input->size());

        // 最终 annulus 拟合不使用合成质心点，只选最接近质心的原始测量点。
        for (int i = 0; i < static_cast<int>(input->size()); ++i)
        {
            const auto& p = input->points[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const VoxelKey key = voxelKey(p.x, p.y, p.z, leaf_size);
            auto& voxel = voxels[key];
            voxel.sum += Eigen::Vector3d(p.x, p.y, p.z);
            voxel.indices.push_back(i);
        }

        output->reserve(voxels.size());
        for (const auto& kv : voxels)
        {
            const auto& voxel = kv.second;
            if (voxel.indices.empty()) continue;
            const Eigen::Vector3d centroid = voxel.sum / static_cast<double>(voxel.indices.size());

            int best_index = voxel.indices.front();
            double best_distance2 = std::numeric_limits<double>::max();
            for (int idx : voxel.indices)
            {
                const auto& p = input->points[idx];
                const Eigen::Vector3d point(p.x, p.y, p.z);
                const double distance2 = (point - centroid).squaredNorm();
                if (distance2 < best_distance2)
                {
                    best_distance2 = distance2;
                    best_index = idx;
                }
            }
            output->push_back(input->points[best_index]);
        }
    }

    // 使用手动配置的 x/y/z 范围做直通滤波
    void manualPassThroughFilter(const pcl::PointCloud<Common::Point>::Ptr& input,
                                 pcl::PointCloud<Common::Point>::Ptr output) const
    {
        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(input);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*output);

        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(output);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*output);

        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(output);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*output);
    }

    // 根据强度直方图自适应计算高反点阈值
    bool computeHighIntensityThreshold(const std::vector<float>& intensities,
                                       float& threshold,
                                       float& min_i,
                                       float& max_i,
                                       float& otsu_threshold,
                                       float& foreground_low,
                                       float& high_quantile,
                                       float& relative_high) const
    {
        if (intensities.size() < 20) return false;

        auto minmax = std::minmax_element(intensities.begin(), intensities.end());
        min_i = *minmax.first;
        max_i = *minmax.second;
        if (max_i - min_i < 1e-3f) return false;

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
            double between_class_variance = weight_background * weight_foreground *
                                            (mean_background - mean_foreground) *
                                            (mean_background - mean_foreground);

            if (between_class_variance > max_between_class_variance)
            {
                max_between_class_variance = between_class_variance;
                best_bin = i;
            }
        }

        otsu_threshold = min_i + (max_i - min_i) * static_cast<float>(best_bin) / static_cast<float>(bins - 1);

        std::vector<float> foreground_intensities;
        foreground_intensities.reserve(intensities.size());
        for (float v : intensities)
        {
            if (v >= otsu_threshold) foreground_intensities.push_back(v);
        }
        if (foreground_intensities.size() < 20) return false;

        // Otsu 给出数据驱动的前景分割，再用高强度侧约束排除弱反射背景点。
        foreground_low = percentile(foreground_intensities, 0.15);
        high_quantile = percentile(intensities, 0.92);
        relative_high = otsu_threshold + 0.55f * (max_i - otsu_threshold);
        threshold = std::max(std::max(otsu_threshold, foreground_low), relative_high);
        return true;
    }

    // 从输入点云中提取强度高于自适应阈值的点
    bool extractHighIntensityPoints(const pcl::PointCloud<Common::Point>::Ptr& input,
                                    pcl::PointCloud<Common::Point>::Ptr output,
                                    const std::string& label) const
    {
        output->clear();
        std::vector<float> intensities;
        intensities.reserve(input->size());
        for (const auto& p : *input)
        {
            if (isFinitePoint(p)) intensities.push_back(p.intensity);
        }

        float threshold = 0.0f;
        float min_i = 0.0f;
        float max_i = 0.0f;
        float otsu_threshold = 0.0f;
        float foreground_low = 0.0f;
        float high_quantile = 0.0f;
        float relative_high = 0.0f;
        if (!computeHighIntensityThreshold(intensities, threshold, min_i, max_i,
                                           otsu_threshold, foreground_low,
                                           high_quantile, relative_high))
        {
            ROS_WARN("[LiDAR] Unable to compute high-intensity threshold for %s.", label.c_str());
            return false;
        }

        output->reserve(input->size() / 10);
        for (const auto& p : *input)
        {
            if (!isFinitePoint(p)) continue;
            if (p.intensity >= threshold) output->push_back(p);
        }

        ROS_INFO("[LiDAR] %s intensity range: %.3f - %.3f", label.c_str(), min_i, max_i);
        ROS_INFO("[LiDAR] %s Otsu: %.3f, foreground p15: %.3f, p92: %.3f, relative high: %.3f, threshold: %.3f",
                 label.c_str(), otsu_threshold, foreground_low, high_quantile, relative_high, threshold);
        ROS_INFO("[LiDAR] %s high-intensity points: %zu", label.c_str(), output->size());
        return !output->empty();
    }

    // 在给定平面距离范围内提取高反点，并复用统一的 intensity 阈值策略
    bool extractHighIntensityPointsNearPlane(const pcl::PointCloud<Common::Point>::Ptr& input,
                                             const pcl::ModelCoefficients::Ptr& plane_coefficients,
                                             double plane_distance_threshold,
                                             pcl::PointCloud<Common::Point>::Ptr output,
                                             const std::string& label) const
    {
        output->clear();
        if (!input || input->empty())
        {
            ROS_WARN("[LiDAR] Empty input cloud, cannot extract high-intensity points for %s.", label.c_str());
            return false;
        }
        if (!plane_coefficients || plane_coefficients->values.size() < 4)
        {
            ROS_WARN("[LiDAR] Invalid plane coefficients for %s.", label.c_str());
            return false;
        }

        std::vector<float> intensities;
        intensities.reserve(input->size());
        for (const auto& p : *input)
        {
            if (!isFinitePoint(p)) continue;
            if (planeDistance(plane_coefficients, p) < plane_distance_threshold)
            {
                intensities.push_back(p.intensity);
            }
        }

        float threshold = 0.0f;
        float min_i = 0.0f;
        float max_i = 0.0f;
        float otsu_threshold = 0.0f;
        float foreground_low = 0.0f;
        float high_quantile = 0.0f;
        float relative_high = 0.0f;
        if (!computeHighIntensityThreshold(intensities, threshold, min_i, max_i,
                                           otsu_threshold, foreground_low,
                                           high_quantile, relative_high))
        {
            ROS_WARN("[LiDAR] Unable to compute high-intensity threshold for %s.", label.c_str());
            return false;
        }

        output->reserve(input->size() / 10);
        for (const auto& p : *input)
        {
            if (!isFinitePoint(p)) continue;
            if (planeDistance(plane_coefficients, p) >= plane_distance_threshold) continue;
            if (p.intensity >= threshold) output->push_back(p);
        }

        ROS_INFO("[LiDAR] %s intensity range: %.3f - %.3f", label.c_str(), min_i, max_i);
        ROS_INFO("[LiDAR] %s Otsu: %.3f, foreground p15: %.3f, p92: %.3f, relative high: %.3f, threshold: %.3f",
                 label.c_str(), otsu_threshold, foreground_low, high_quantile, relative_high, threshold);
        ROS_INFO("[LiDAR] %s high-intensity points: %zu", label.c_str(), output->size());
        return !output->empty();
    }

    // 使用 RANSAC 拟合平面模型
    bool fitPlaneRansac(const pcl::PointCloud<Common::Point>::Ptr& input,
                        pcl::ModelCoefficients::Ptr coefficients,
                        double distance_threshold) const
    {
        if (!input || input->size() < 50) return false;

        pcl::PointCloud<Common::Point>::Ptr work_cloud(new pcl::PointCloud<Common::Point>);
        if (input->size() > 80000)
        {
            // 这里只为 RANSAC 提速；平面模型会回到原始 ROI 上提取 inliers。
            pcl::VoxelGrid<Common::Point> voxel_filter;
            voxel_filter.setInputCloud(input);
            voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
            voxel_filter.filter(*work_cloud);
        }
        else
        {
            *work_cloud = *input;
        }

        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(distance_threshold);
        plane_segmentation.setInputCloud(work_cloud);
        plane_segmentation.segment(*plane_inliers, *coefficients);

        return !plane_inliers->indices.empty() && coefficients->values.size() >= 4;
    }

    // 根据平面模型从原始点云中提取平面内点
    void extractPlaneInliers(const pcl::PointCloud<Common::Point>::Ptr& input,
                             const pcl::ModelCoefficients::Ptr& coefficients,
                             pcl::PointCloud<Common::Point>::Ptr output,
                             double distance_threshold) const
    {
        output->clear();
        output->reserve(input->size());
        for (const auto& p : *input)
        {
            if (planeDistance(coefficients, p) < distance_threshold)
            {
                output->push_back(p);
            }
        }
    }

    // 仅根据两两距离筛选符合标定板几何的 4 个候选中心
    bool selectGeometryConsistentCentersByDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr& candidates,
                                                    std::vector<int>& selected_indices,
                                                    double max_error_threshold) const
    {
        selected_indices.clear();
        if (!candidates || candidates->size() < TARGET_NUM_CIRCLES) return false;

        std::vector<std::vector<int>> groups;
        comb(candidates->size(), TARGET_NUM_CIRCLES, groups);

        const double target_diagonal = std::sqrt(delta_width_circles_ * delta_width_circles_ +
                                                delta_height_circles_ * delta_height_circles_);
        std::vector<double> target_distances = {
            delta_height_circles_, delta_height_circles_,
            delta_width_circles_, delta_width_circles_,
            target_diagonal, target_diagonal
        };

        double best_score = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (int i = 0; i < static_cast<int>(groups.size()); ++i)
        {
            std::vector<double> measured;
            measured.reserve(6);
            for (int a = 0; a < TARGET_NUM_CIRCLES; ++a)
            {
                for (int b = a + 1; b < TARGET_NUM_CIRCLES; ++b)
                {
                    measured.push_back(distance3D(candidates->points[groups[i][a]],
                                                  candidates->points[groups[i][b]]));
                }
            }
            std::sort(measured.begin(), measured.end());

            double score = 0.0;
            double max_error = 0.0;
            for (size_t k = 0; k < measured.size(); ++k)
            {
                const double error = measured[k] - target_distances[k];
                score += error * error;
                max_error = std::max(max_error, std::fabs(error));
            }

            if (max_error <= max_error_threshold && score < best_score)
            {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_idx < 0) return false;
        selected_indices = groups[best_idx];
        return true;
    }

    // 对单个 annulus 聚类做鲁棒圆拟合，输出中心、半径和平均残差
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

    // 从候选中心中选择最符合 0.5m x 0.4m 几何约束的 4 个中心
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

    // 根据聚类索引计算每个高反聚类的三维质心
    void computeClusterCentroids(const pcl::PointCloud<Common::Point>::Ptr& cloud,
                                 const std::vector<pcl::PointIndices>& cluster_indices,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_centers,
                                 std::vector<pcl::PointIndices>& candidate_clusters) const
    {
        candidate_centers->clear();
        candidate_clusters.clear();
        candidate_centers->reserve(cluster_indices.size());
        candidate_clusters.reserve(cluster_indices.size());

        for (const auto& cluster_idx : cluster_indices)
        {
            if (cluster_idx.indices.empty()) continue;

            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (int idx : cluster_idx.indices)
            {
                const auto& p = cloud->points[idx];
                centroid += Eigen::Vector3d(p.x, p.y, p.z);
            }
            centroid /= static_cast<double>(cluster_idx.indices.size());

            pcl::PointXYZ center;
            center.x = static_cast<float>(centroid.x());
            center.y = static_cast<float>(centroid.y());
            center.z = static_cast<float>(centroid.z());
            candidate_centers->push_back(center);
            candidate_clusters.push_back(cluster_idx);
        }
    }

    // 收集通过几何筛选的 annulus 聚类中心和对应高反点
    void collectSelectedAutoRoiClusters(const pcl::PointCloud<Common::Point>::Ptr& high_cloud,
                                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& candidate_centers,
                                        const std::vector<pcl::PointIndices>& candidate_clusters,
                                        const std::vector<int>& selected_indices,
                                        pcl::PointCloud<Common::Point>::Ptr selected_high_points,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr selected_centers) const
    {
        selected_high_points->clear();
        selected_centers->clear();
        selected_high_points->reserve(50000);
        selected_centers->reserve(TARGET_NUM_CIRCLES);

        for (int selected_idx : selected_indices)
        {
            selected_centers->push_back(candidate_centers->points[selected_idx]);
            for (int point_idx : candidate_clusters[selected_idx].indices)
            {
                selected_high_points->push_back(high_cloud->points[point_idx]);
            }
        }
    }

    // 根据粗平面和 4 个 annulus 中心估计标定板局部坐标系
    bool estimateBoardFrame(const pcl::ModelCoefficients::Ptr& rough_plane,
                            const pcl::PointCloud<pcl::PointXYZ>::Ptr& selected_centers,
                            BoardFrame& frame) const
    {
        Eigen::Vector3d normal(rough_plane->values[0], rough_plane->values[1], rough_plane->values[2]);
        if (normal.norm() < 1e-6)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: invalid rough plane normal.");
            return false;
        }
        normal.normalize();

        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        for (const auto& p : *selected_centers)
        {
            origin += Eigen::Vector3d(p.x, p.y, p.z);
        }
        origin /= static_cast<double>(selected_centers->size());

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto& p : *selected_centers)
        {
            Eigen::Vector3d v(p.x, p.y, p.z);
            v -= origin;
            v -= normal * v.dot(normal);
            covariance += v * v.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: cannot build board axes.");
            return false;
        }

        Eigen::Vector3d x_axis = solver.eigenvectors().col(2);
        x_axis -= normal * x_axis.dot(normal);
        if (x_axis.norm() < 1e-6)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: invalid board x-axis.");
            return false;
        }
        x_axis.normalize();

        Eigen::Vector3d y_axis = normal.cross(x_axis);
        if (y_axis.norm() < 1e-6)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: invalid board y-axis.");
            return false;
        }
        y_axis.normalize();

        frame.origin = origin;
        frame.x_axis = x_axis;
        frame.y_axis = y_axis;
        frame.normal = normal;
        return true;
    }

    // 按标定板局部坐标系和实际尺寸从原始点云裁剪整块板 ROI
    void cropBoardRoiByFrame(const pcl::PointCloud<Common::Point>::Ptr& finite_cloud,
                             const BoardFrame& frame,
                             pcl::PointCloud<Common::Point>::Ptr board_roi_cloud) const
    {
        const double half_width = board_width_ * 0.5 + board_roi_margin_;
        const double half_height = board_height_ * 0.5 + board_roi_margin_;
        const double half_depth = board_roi_depth_ * 0.5;

        board_roi_cloud->clear();
        board_roi_cloud->reserve(finite_cloud->size() / 10);
        for (const auto& p : *finite_cloud)
        {
            Eigen::Vector3d v(p.x, p.y, p.z);
            v -= frame.origin;
            const double u = v.dot(frame.x_axis);
            const double w = v.dot(frame.y_axis);
            const double depth = v.dot(frame.normal);
            if (std::fabs(u) <= half_width &&
                std::fabs(w) <= half_height &&
                std::fabs(depth) <= half_depth)
            {
                board_roi_cloud->push_back(p);
            }
        }
    }

    // 根据高反 annulus 自动裁剪整块标定板 ROI
    bool extractAutoBoardRoi(const pcl::PointCloud<Common::Point>::Ptr& cloud,
                             pcl::PointCloud<Common::Point>::Ptr board_roi_cloud) const
    {
        board_roi_cloud->clear();

        pcl::PointCloud<Common::Point>::Ptr finite_cloud(new pcl::PointCloud<Common::Point>);
        copyFiniteRangePoints(cloud, finite_cloud);
        ROS_INFO("[LiDAR] Auto ROI finite cloud size: %zu", finite_cloud->size());
        if (finite_cloud->size() < 1000)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: too few finite points.");
            return false;
        }

        pcl::PointCloud<Common::Point>::Ptr coarse_cloud(new pcl::PointCloud<Common::Point>);
        // 粗定位阶段降低采集时长对点数的影响，同时保留高反 annulus 回波。
        voxelDownsampleKeepMaxIntensity(finite_cloud, auto_roi_voxel_leaf_, coarse_cloud);
        ROS_INFO("[LiDAR] Auto ROI max-intensity voxel cloud size: %zu (leaf %.3f m)",
                 coarse_cloud->size(), auto_roi_voxel_leaf_);
        if (coarse_cloud->size() < 1000)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: too few points after max-intensity voxel sampling.");
            return false;
        }

        pcl::PointCloud<Common::Point>::Ptr high_cloud(new pcl::PointCloud<Common::Point>);
        // 高反聚类用于估计粗略板位姿，避免依赖手动直通滤波范围。
        if (!extractHighIntensityPoints(coarse_cloud, high_cloud, "Auto ROI"))
        {
            return false;
        }

        if (high_cloud->size() < 1000)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: too few high-intensity points.");
            return false;
        }

        pcl::search::KdTree<Common::Point>::Ptr tree(new pcl::search::KdTree<Common::Point>);
        tree->setInputCloud(high_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<Common::Point> ec;
        ec.setClusterTolerance(0.035);
        ec.setMinClusterSize(200);
        ec.setMaxClusterSize(60000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(high_cloud);
        ec.extract(cluster_indices);

        ROS_INFO("[LiDAR] Auto ROI high-intensity clusters: %zu", cluster_indices.size());
        if (cluster_indices.size() < TARGET_NUM_CIRCLES)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: fewer than 4 high-intensity clusters.");
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_centers(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<pcl::PointIndices> candidate_clusters;
        computeClusterCentroids(high_cloud, cluster_indices, candidate_centers, candidate_clusters);

        std::vector<int> selected_indices;
        // 按已知 0.5m x 0.4m annulus 中心几何关系剔除无关高反物。
        if (!selectGeometryConsistentCentersByDistances(candidate_centers, selected_indices, 0.18))
        {
            ROS_WARN("[LiDAR] Auto ROI failed: no high-intensity cluster set matches target geometry.");
            return false;
        }

        pcl::PointCloud<Common::Point>::Ptr selected_high_points(new pcl::PointCloud<Common::Point>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr selected_centers(new pcl::PointCloud<pcl::PointXYZ>);
        collectSelectedAutoRoiClusters(high_cloud, candidate_centers, candidate_clusters,
                                       selected_indices, selected_high_points, selected_centers);

        pcl::ModelCoefficients::Ptr rough_plane(new pcl::ModelCoefficients);
        if (!fitPlaneRansac(selected_high_points, rough_plane, 0.03))
        {
            ROS_WARN("[LiDAR] Auto ROI failed: cannot fit rough plane from selected reflective annuli.");
            return false;
        }

        // 裁剪原始有限点云，不裁剪粗采样点云，保证后续平面拟合保留原始测量。
        BoardFrame board_frame;
        if (!estimateBoardFrame(rough_plane, selected_centers, board_frame))
        {
            return false;
        }
        cropBoardRoiByFrame(finite_cloud, board_frame, board_roi_cloud);

        ROS_INFO("[LiDAR] Auto ROI board cloud size: %zu", board_roi_cloud->size());
        if (board_roi_cloud->size() < 1000)
        {
            ROS_WARN("[LiDAR] Auto ROI failed: board ROI is too small.");
            return false;
        }

        return true;
    }

    // 清空本次检测使用的中间点云
    void clearDetectionClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();
    }

    // 根据配置选择自动 ROI 或手动直通滤波来得到板子区域
    bool extractBoardRoi(const pcl::PointCloud<Common::Point>::Ptr& cloud,
                         pcl::PointCloud<Common::Point>::Ptr board_roi_cloud) const
    {
        if (use_auto_lidar_roi_)
        {
            ROS_INFO("[LiDAR] Using automatic board ROI extraction.");
            if (!extractAutoBoardRoi(cloud, board_roi_cloud))
            {
                ROS_WARN("[LiDAR] Automatic board ROI extraction failed.");
                return false;
            }
        }
        else
        {
            ROS_INFO("[LiDAR] Using manual pass-through board ROI extraction.");
            manualPassThroughFilter(cloud, board_roi_cloud);
        }
        return true;
    }

    // 用整块板 ROI 拟合最终平面，并回到原始 ROI 上提取平面内点
    bool fitBoardPlaneFromRoi(const pcl::PointCloud<Common::Point>::Ptr& board_roi_cloud,
                              pcl::ModelCoefficients::Ptr plane_coefficients)
    {
        *filtered_cloud_ = *board_roi_cloud;
        ROS_INFO("[LiDAR] Board ROI cloud size: %zu", filtered_cloud_->size());
        if (filtered_cloud_->size() < 1000)
        {
            ROS_WARN("[LiDAR] Too few board ROI points.");
            return false;
        }

        if (!fitPlaneRansac(filtered_cloud_, plane_coefficients, 0.01))
        {
            ROS_WARN("[LiDAR] Unable to fit board plane from ROI cloud.");
            return false;
        }

        extractPlaneInliers(filtered_cloud_, plane_coefficients, plane_cloud_, 0.015);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
        if (plane_cloud_->size() < 500)
        {
            ROS_WARN("[LiDAR] Too few board plane inliers.");
            return false;
        }
        return true;
    }

    // 计算板平面到 Z=0 平面的旋转，并生成对齐后的板点云
    bool alignBoardPlaneToZ0(const pcl::ModelCoefficients::Ptr& plane_coefficients,
                             PlaneAlignment& alignment)
    {
        Eigen::Vector3d normal(plane_coefficients->values[0],
                               plane_coefficients->values[1],
                               plane_coefficients->values[2]);
        if (normal.norm() < 1e-6)
        {
            ROS_WARN("[LiDAR] Invalid plane normal.");
            return false;
        }
        normal.normalize();

        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);
        const double angle = acos(std::max(-1.0, std::min(1.0, normal.dot(z_axis))));

        alignment.rotation = Eigen::Matrix3d::Identity();
        if (axis.norm() > 1e-6)
        {
            axis.normalize();
            Eigen::AngleAxisd rotation(angle, axis);
            alignment.rotation = rotation.toRotationMatrix();
        }

        aligned_cloud_->clear();
        aligned_cloud_->reserve(plane_cloud_->size());

        double average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = alignment.rotation * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }

        if (cnt == 0)
        {
            ROS_WARN("[LiDAR] No board plane points to align.");
            return false;
        }
        alignment.average_z = average_z / static_cast<double>(cnt);
        return true;
    }

    // 在板平面内提取高反 annulus 点、对齐到 Z=0 并做最终空间稀疏
    bool extractAlignedAnnulusCloud(const pcl::ModelCoefficients::Ptr& plane_coefficients,
                                    const PlaneAlignment& alignment)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr annulus_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        extractHighIntensityAnnulusPoints(plane_cloud_, plane_coefficients, annulus_cloud_raw);

        edge_cloud_->clear();
        edge_cloud_->reserve(annulus_cloud_raw->size());
        for (const auto& pt : *annulus_cloud_raw)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = alignment.rotation * point;
            edge_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
        }
        ROS_INFO("Extracted %zu high-intensity annulus points before voxel sampling.", edge_cloud_->size());

        if (edge_cloud_->size() < 20)
        {
            ROS_WARN("[LiDAR] Too few annulus points after intensity extraction.");
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr annulus_cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        // 聚类/拟合前只做空间稀疏，限制 cluster 点数且不生成虚拟点。
        voxelDownsampleClosestToCentroid(edge_cloud_, annulus_voxel_leaf_, annulus_cloud_downsampled);
        edge_cloud_.swap(annulus_cloud_downsampled);
        ROS_INFO("[LiDAR] Annulus points after closest-to-centroid voxel sampling: %zu (leaf %.3f m)",
                 edge_cloud_->size(), annulus_voxel_leaf_);
        return true;
    }

    // 对高反 annulus 点云做欧式聚类，分出 4 个 annulus 候选
    void clusterAnnulusCloud(std::vector<pcl::PointIndices>& cluster_indices) const
    {
        cluster_indices.clear();
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02);
        ec.setMinClusterSize(200);
        ec.setMaxClusterSize(50000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);

        ROS_INFO("Number of edge clusters: %zu", cluster_indices.size());
    }

    // 对每个 annulus 聚类拟合圆，并输出通过半径和残差检查的候选中心
    void fitAnnulusCentersFromClusters(const std::vector<pcl::PointIndices>& cluster_indices,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_centers) const
    {
        candidate_centers->clear();
        candidate_centers->reserve(cluster_indices.size());

        for (size_t i = 0; i < cluster_indices.size(); ++i)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices)
            {
                cluster->push_back(edge_cloud_->points[idx]);
            }

            CircleFitResult fit;
            if (!fitCircleRobust(cluster, fit))
            {
                ROS_WARN("[LiDAR] Annulus fit failed for cluster %zu with %zu points.", i, cluster->size());
                continue;
            }

            const bool radius_ok = std::fabs(fit.radius - circle_radius_) < 0.05;
            const bool error_ok = fit.mean_abs_error < 0.03;
            ROS_INFO("[LiDAR] Annulus cluster %zu: points=%zu, center=(%.4f, %.4f), radius=%.4f, mean abs error=%.4f",
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
    }

    // 将 Z=0 平面上的 annulus 中心逆变换回原始 LiDAR 坐标系
    void transformCentersBackToLidar(const pcl::PointCloud<pcl::PointXYZ>::Ptr& candidate_centers,
                                     const std::vector<int>& selected_indices,
                                     const PlaneAlignment& alignment,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        center_z0_cloud_->clear();
        center_z0_cloud_->reserve(TARGET_NUM_CIRCLES);
        center_cloud->clear();
        center_cloud->reserve(TARGET_NUM_CIRCLES);

        const Eigen::Matrix3d R_inv = alignment.rotation.inverse();
        for (int idx : selected_indices)
        {
            const pcl::PointXYZ& center_point = candidate_centers->points[idx];
            center_z0_cloud_->push_back(center_point);

            Eigen::Vector3d aligned_point(center_point.x, center_point.y,
                                          center_point.z + alignment.average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    // 构造 LiDAR 检测器并读取参数、初始化调试点云发布器
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
        board_width_ = params.board_width;
        board_height_ = params.board_height;
        board_roi_margin_ = params.board_roi_margin;
        board_roi_depth_ = params.board_roi_depth;
        auto_roi_voxel_leaf_ = params.auto_roi_voxel_leaf;
        annulus_voxel_leaf_ = params.annulus_voxel_leaf;
        use_auto_lidar_roi_ = params.use_auto_lidar_roi;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    // 在已拟合的板平面附近，根据 intensity 提取高反 annulus 点
    void extractHighIntensityAnnulusPoints(const pcl::PointCloud<Common::Point>::Ptr& plane_cloud,const pcl::ModelCoefficients::Ptr& plane_coefficients,pcl::PointCloud<pcl::PointXYZ>::Ptr annulus_cloud)
    {
        annulus_cloud->clear();

        const double plane_distance_threshold = 0.03;
        pcl::PointCloud<Common::Point>::Ptr annulus_common(new pcl::PointCloud<Common::Point>);
        if (!extractHighIntensityPointsNearPlane(plane_cloud, plane_coefficients,
                                                 plane_distance_threshold,
                                                 annulus_common, "Annulus"))
        {
            return;
        }

        annulus_cloud->reserve(annulus_common->size());
        for (const auto& p : *annulus_common)
        {
            annulus_cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
        }
        ROS_INFO("[LiDAR] Extracted %zu high-intensity annulus points.", annulus_cloud->size());
    }

    // 处理机械式 LiDAR 点云并提取 4 个 annulus 中心
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
    
        // 3. 根据强度提取圆环点
        edge_cloud_->clear();
        edge_cloud_->reserve(plane_cloud_->size());
        extractHighIntensityAnnulusPoints(plane_cloud_, plane_coefficients, edge_cloud_);
        ROS_INFO("Extracted %zu high-intensity annulus points.", edge_cloud_->size());
        if (edge_cloud_->size() < 12)
        {
            ROS_WARN("[LiDAR] Too few annulus points after intensity extraction.");
            return;
        }

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
            ROS_WARN("[LiDAR] No annulus points to align.");
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
                ROS_INFO("[LiDAR] No more annulus candidates can be found, stop.");
                break;
            }

            // 内点太少就认为是噪声，直接结束
            if ((int)inliers->indices.size() < 5)
            {
                ROS_INFO("[LiDAR] Found circle but inliers too few (%zu < 3), stop.",
                            inliers->indices.size());
                break;
            }

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

    // 处理固态 LiDAR 点云并提取 4 个 annulus 中心
    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        clearDetectionClouds(center_cloud);

        pcl::PointCloud<Common::Point>::Ptr board_roi_cloud(new pcl::PointCloud<Common::Point>);
        if (!extractBoardRoi(cloud, board_roi_cloud))
        {
            return;
        }

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        if (!fitBoardPlaneFromRoi(board_roi_cloud, plane_coefficients))
        {
            return;
        }

        PlaneAlignment alignment;
        if (!alignBoardPlaneToZ0(plane_coefficients, alignment))
        {
            return;
        }

        if (!extractAlignedAnnulusCloud(plane_coefficients, alignment))
        {
            return;
        }

        std::vector<pcl::PointIndices> cluster_indices;
        clusterAnnulusCloud(cluster_indices);

        pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_centers(new pcl::PointCloud<pcl::PointXYZ>);
        fitAnnulusCentersFromClusters(cluster_indices, candidate_centers);

        std::vector<int> selected_indices;
        if (!selectGeometryConsistentCenters(candidate_centers, selected_indices))
        {
            ROS_WARN("[LiDAR] Unable to select 4 geometry-consistent annulus centers from %zu candidates.",
                     candidate_centers->size());
            return;
        }

        transformCentersBackToLidar(candidate_centers, selected_indices, alignment, center_cloud);
    }
    // 获取中间结果的点云
    // 获取 ROI 滤波后的点云
    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    // 获取最终板平面的内点点云
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    // 获取对齐到 Z=0 平面的板点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    // 获取对齐后的高反 annulus 点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    // 获取 Z=0 平面上的 annulus 中心
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
