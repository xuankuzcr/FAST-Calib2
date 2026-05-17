/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

LiDAR-only batch test entry for target annulus center extraction.
*/

#include "data_preprocess.hpp"
#include "lidar_detect.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/impl/filter_indices.hpp>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sys/stat.h>

namespace
{
// 判断 PointCloud2 消息中是否包含指定字段
bool hasField(const sensor_msgs::PointCloud2& msg, const std::string& name)
{
    for (const auto& field : msg.fields)
    {
        if (field.name == name) return true;
    }
    return false;
}

// 从 rosbag 中读取指定 LiDAR topic，并统一转换为 Common::Point 点云
bool loadCloudFromBag(const std::string& bag_path,
                      const std::string& lidar_topic,
                      pcl::PointCloud<Common::Point>::Ptr cloud,
                      LiDARType& detected_type)
{
    cloud->clear();
    detected_type = LiDARType::Unknown;

    rosbag::Bag bag;
    try
    {
        bag.open(bag_path, rosbag::bagmode::Read);
    }
    catch (const rosbag::BagException& e)
    {
        ROS_ERROR_STREAM("[LiDAR Test] Failed to open bag " << bag_path << ": " << e.what());
        return false;
    }

    rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{lidar_topic}));
    size_t message_count = 0;

    for (const rosbag::MessageInstance& m : view)
    {
        if (auto livox_custom_msg = m.instantiate<livox_ros_driver::CustomMsg>())
        {
            detected_type = LiDARType::Solid;
            cloud->reserve(cloud->size() + livox_custom_msg->point_num);
            for (uint32_t i = 0; i < livox_custom_msg->point_num; ++i)
            {
                Common::Point p;
                p.x = livox_custom_msg->points[i].x;
                p.y = livox_custom_msg->points[i].y;
                p.z = livox_custom_msg->points[i].z;
                p.intensity = static_cast<float>(livox_custom_msg->points[i].reflectivity);
                p.ring = static_cast<std::uint16_t>(livox_custom_msg->points[i].line);
                cloud->push_back(p);
            }
            ++message_count;
            continue;
        }

        if (auto pcl_msg = m.instantiate<sensor_msgs::PointCloud2>())
        {
            const bool has_ring = hasField(*pcl_msg, "ring");
            const bool has_intensity = hasField(*pcl_msg, "intensity");
            const bool has_reflectivity = hasField(*pcl_msg, "reflectivity");

            if (detected_type == LiDARType::Unknown)
            {
                detected_type = has_ring ? LiDARType::Mech : LiDARType::Solid;
            }

            sensor_msgs::PointCloud2ConstIterator<float> it_x(*pcl_msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> it_y(*pcl_msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> it_z(*pcl_msg, "z");

            std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<std::uint16_t>> it_ring_ptr;
            if (has_ring)
            {
                it_ring_ptr.reset(new sensor_msgs::PointCloud2ConstIterator<std::uint16_t>(*pcl_msg, "ring"));
            }

            std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_intensity_ptr;
            if (has_intensity)
            {
                it_intensity_ptr.reset(new sensor_msgs::PointCloud2ConstIterator<float>(*pcl_msg, "intensity"));
            }
            else if (has_reflectivity)
            {
                it_intensity_ptr.reset(new sensor_msgs::PointCloud2ConstIterator<float>(*pcl_msg, "reflectivity"));
            }

            const size_t n = static_cast<size_t>(pcl_msg->width) * pcl_msg->height;
            cloud->reserve(cloud->size() + n);
            for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z)
            {
                Common::Point p;
                p.x = *it_x;
                p.y = *it_y;
                p.z = *it_z;
                p.ring = 0xFFFF;
                p.intensity = 0.0f;

                if (it_ring_ptr)
                {
                    p.ring = **it_ring_ptr;
                    ++(*it_ring_ptr);
                }
                if (it_intensity_ptr)
                {
                    p.intensity = **it_intensity_ptr;
                    ++(*it_intensity_ptr);
                }

                cloud->push_back(p);
            }
            ++message_count;
        }
    }

    ROS_INFO("[LiDAR Test] Loaded %zu messages, %zu points from %s",
             message_count, cloud->size(), bag_path.c_str());
    return message_count > 0 && !cloud->empty();
}

// 将 LiDAR 类型枚举转换为日志可读字符串
std::string lidarTypeName(LiDARType type)
{
    switch (type)
    {
        case LiDARType::Solid: return "solid";
        case LiDARType::Mech: return "mech";
        default: return "unknown";
    }
}

// 去掉路径末尾多余的斜杠
std::string trimTrailingSlash(std::string path)
{
    while (!path.empty() && path.back() == '/')
    {
        path.pop_back();
    }
    return path;
}

// 获取路径最后一级文件名或目录名
std::string pathBaseName(const std::string& path)
{
    const std::string clean_path = trimTrailingSlash(path);
    const size_t pos = clean_path.find_last_of('/');
    return pos == std::string::npos ? clean_path : clean_path.substr(pos + 1);
}

// 获取路径父目录的最后一级名称
std::string pathParentName(const std::string& path)
{
    const std::string clean_path = trimTrailingSlash(path);
    const size_t last_slash = clean_path.find_last_of('/');
    if (last_slash == std::string::npos) return "";
    return pathBaseName(clean_path.substr(0, last_slash));
}

// 去掉文件名扩展名
std::string stripExtension(const std::string& filename)
{
    const size_t pos = filename.find_last_of('.');
    return pos == std::string::npos ? filename : filename.substr(0, pos);
}

// 将任意字符串转换为安全的文件名前缀片段
std::string sanitizeFilePart(std::string value)
{
    for (char& c : value)
    {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'))
        {
            c = '_';
        }
    }
    return value;
}

// 确保输出目录存在
void ensureDirectory(const std::string& path)
{
    if (path.empty()) return;
    mkdir(path.c_str(), 0755);
}

// 解析测试输出目录，兼容未展开的 ROS launch 变量
std::string resolveOutputDirectory(const Params& params)
{
    std::string output_dir = params.output_path;
    if (output_dir.empty() || output_dir.find("$(") != std::string::npos)
    {
        output_dir = "/home/chunran/02_calib_ws/src/FAST-Calib/output";
    }
    output_dir = trimTrailingSlash(output_dir);
    ensureDirectory(output_dir);
    return output_dir;
}

// 根据 bag 所在目录和文件名生成输出文件前缀
std::string outputPrefixForBag(const std::string& bag_path)
{
    return sanitizeFilePart(pathParentName(bag_path) + "_" +
                            stripExtension(pathBaseName(bag_path)));
}

// 构造带 RGB 颜色的 PCL 点
pcl::PointXYZRGB makeRgbPoint(float x, float y, float z, std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    pcl::PointXYZRGB p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.r = r;
    p.g = g;
    p.b = b;
    return p;
}

// 在线性颜色表中按比例插值
std::array<std::uint8_t, 3> lerpColor(const std::array<std::uint8_t, 3>& a,
                                      const std::array<std::uint8_t, 3>& b,
                                      float t)
{
    t = std::max(0.0f, std::min(1.0f, t));
    return {{
        static_cast<std::uint8_t>(std::round(a[0] + t * (b[0] - a[0]))),
        static_cast<std::uint8_t>(std::round(a[1] + t * (b[1] - a[1]))),
        static_cast<std::uint8_t>(std::round(a[2] + t * (b[2] - a[2])))
    }};
}

// 计算浮点数组的指定分位数
float percentile(std::vector<float> values, float ratio)
{
    if (values.empty()) return 0.0f;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    const size_t idx = static_cast<size_t>(std::round(ratio * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

// 将 intensity 映射为便于观察的伪彩色
std::array<std::uint8_t, 3> intensityColor(float intensity, float min_intensity, float max_intensity)
{
    float t = 0.0f;
    if (max_intensity > min_intensity + 1e-3f)
    {
        t = (intensity - min_intensity) / (max_intensity - min_intensity);
    }
    t = std::max(0.0f, std::min(1.0f, t));
    t = std::pow(t, 0.75f);

    const std::array<std::uint8_t, 3> low = {{30, 38, 55}};
    const std::array<std::uint8_t, 3> mid = {{70, 135, 190}};
    const std::array<std::uint8_t, 3> high = {{245, 220, 145}};
    if (t < 0.55f)
    {
        return lerpColor(low, mid, t / 0.55f);
    }
    return lerpColor(mid, high, (t - 0.55f) / 0.45f);
}

// 在调试点云中生成实心球标记
void addSphereMarker(const pcl::PointXYZ& center,
                     float radius,
                     float step,
                     const std::array<std::uint8_t, 3>& color,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr output)
{
    for (float dx = -radius; dx <= radius; dx += step)
    {
        for (float dy = -radius; dy <= radius; dy += step)
        {
            for (float dz = -radius; dz <= radius; dz += step)
            {
                if (dx * dx + dy * dy + dz * dz <= radius * radius)
                {
                    output->push_back(makeRgbPoint(center.x + dx, center.y + dy, center.z + dz,
                                                   color[0], color[1], color[2]));
                }
            }
        }
    }
}

// 用纯白小球标记最终圆心
void addCenterMarker(const pcl::PointXYZ& center,
                     const std::array<std::uint8_t, 3>& color,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr output)
{
    addSphereMarker(center, 0.040f, 0.006f, color, output);
}

// 保存调试 PCD：板子按 intensity 着色，annulus 为绿色，边界为红色，圆心为白色
bool saveDebugCloud(const pcl::PointCloud<Common::Point>::Ptr& board_cloud,
                    const pcl::PointCloud<Common::Point>::Ptr& annulus_cloud,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& boundary_cloud,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& centers,
                    const Params& params,
                    const std::string& bag_path)
{
    const std::string output_dir = resolveOutputDirectory(params);
    const std::string prefix = outputPrefixForBag(bag_path);
    const std::string output_path = output_dir + "/" + prefix + "_debug_cloud.pcd";

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    const size_t board_size = board_cloud ? board_cloud->size() : 0;
    const size_t annulus_size = annulus_cloud ? annulus_cloud->size() : 0;
    const size_t boundary_size = boundary_cloud ? boundary_cloud->size() : 0;
    const size_t board_stride = std::max<size_t>(1, board_size / 220000);
    const size_t visual_board_size = board_stride > 0 ? (board_size + board_stride - 1) / board_stride : board_size;
    debug_cloud->reserve(visual_board_size + annulus_size + boundary_size + 6000);

    std::vector<float> board_intensities;
    board_intensities.reserve(std::min<size_t>(board_size, 300000));
    if (board_cloud)
    {
        for (size_t i = 0; i < board_cloud->size(); i += board_stride)
        {
            const auto& p = board_cloud->points[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            if (!std::isfinite(p.intensity)) continue;
            board_intensities.push_back(p.intensity);
        }
    }

    float min_intensity = percentile(board_intensities, 0.02f);
    float max_intensity = percentile(board_intensities, 0.98f);
    if (!std::isfinite(min_intensity) || !std::isfinite(max_intensity) ||
        max_intensity <= min_intensity + 1e-3f)
    {
        min_intensity = 0.0f;
        max_intensity = 255.0f;
    }

    size_t board_output_count = 0;
    if (board_cloud)
    {
        for (size_t i = 0; i < board_cloud->size(); i += board_stride)
        {
            const auto& p = board_cloud->points[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(p.intensity))
            {
                continue;
            }
            const auto color = intensityColor(p.intensity, min_intensity, max_intensity);
            debug_cloud->push_back(makeRgbPoint(p.x, p.y, p.z, color[0], color[1], color[2]));
            ++board_output_count;
        }
    }

    const std::array<std::uint8_t, 3> annulus_color = {{40, 255, 40}};
    if (annulus_cloud)
    {
        for (const auto& p : *annulus_cloud)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                continue;
            }
            debug_cloud->push_back(makeRgbPoint(p.x, p.y, p.z,
                                                annulus_color[0],
                                                annulus_color[1],
                                                annulus_color[2]));
        }
    }

    const std::array<std::uint8_t, 3> boundary_color = {{255, 35, 35}};
    if (boundary_cloud)
    {
        for (const auto& p : *boundary_cloud)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                continue;
            }
            debug_cloud->push_back(makeRgbPoint(p.x, p.y, p.z,
                                                boundary_color[0],
                                                boundary_color[1],
                                                boundary_color[2]));
        }
    }

    const std::array<std::uint8_t, 3> center_color = {{255, 255, 255}};

    for (size_t i = 0; i < centers->size(); ++i)
    {
        addCenterMarker(centers->points[i], center_color, debug_cloud);
    }

    debug_cloud->width = static_cast<uint32_t>(debug_cloud->size());
    debug_cloud->height = 1;
    debug_cloud->is_dense = false;

    const int ret = pcl::io::savePCDFileBinaryCompressed(output_path, *debug_cloud);
    if (ret != 0)
    {
        ROS_ERROR_STREAM("[LiDAR Test] Failed to save debug cloud to " << output_path);
        return false;
    }

    std::cout << "[LiDAR Test] Saved debug cloud: " << output_path
              << " (" << board_output_count << "/" << board_size << " board points, "
              << annulus_size << " green annulus points, "
              << boundary_size << " red boundary points, "
              << centers->size() << " white centers)" << std::endl;
    return true;
}

// 保存最终圆心坐标到 output 文本文件
bool saveCenterCoordinates(const pcl::PointCloud<pcl::PointXYZ>::Ptr& centers,
                           const Params& params,
                           const std::string& bag_path)
{
    const std::string output_dir = resolveOutputDirectory(params);
    const std::string output_path = output_dir + "/" + outputPrefixForBag(bag_path) + "_centers.txt";

    std::ofstream fout(output_path);
    if (!fout.is_open())
    {
        ROS_ERROR_STREAM("[LiDAR Test] Failed to save center coordinates to " << output_path);
        return false;
    }

    fout << std::fixed << std::setprecision(9);
    fout << "# x y z\n";
    for (size_t i = 0; i < centers->size(); ++i)
    {
        const auto& p = centers->points[i];
        fout << i << " " << p.x << " " << p.y << " " << p.z << "\n";
    }

    std::cout << "[LiDAR Test] Saved center coordinates: " << output_path << std::endl;
    return true;
}

// 计算 double 数组的指定分位数
double quantileDouble(std::vector<double> values, double ratio)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    ratio = std::max(0.0, std::min(1.0, ratio));
    const size_t idx = static_cast<size_t>(std::round(ratio * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

// 输出固态 LiDAR 单圆半径质检统计
void printSolidRadiusQuality(const std::array<std::vector<double>, TARGET_NUM_CIRCLES>& radii_by_center,
                             double target_radius)
{
    std::vector<double> radius_errors;
    radius_errors.reserve(TARGET_NUM_CIRCLES);

    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
    {
        const double radius = quantileDouble(radii_by_center[i], 0.5);
        if (!std::isfinite(radius))
        {
            std::cout << "[Radius][LiDAR] circle " << i << ": insufficient support" << std::endl;
            continue;
        }
        const double p10 = quantileDouble(radii_by_center[i], 0.1);
        const double p90 = quantileDouble(radii_by_center[i], 0.9);
        const double error = radius - target_radius;
        radius_errors.push_back(error);
        std::cout << "[Radius][LiDAR] circle " << i
                  << ": support=" << radii_by_center[i].size()
                  << ", radius=" << radius * 1000.0 << " mm"
                  << ", error=" << error * 1000.0 << " mm"
                  << ", p10-p90=[" << p10 * 1000.0 << ", " << p90 * 1000.0 << "] mm"
                  << std::endl;
    }

    double max_abs_error = 0.0;
    double rmse = 0.0;
    for (double error : radius_errors)
    {
        max_abs_error = std::max(max_abs_error, std::fabs(error));
        rmse += error * error;
    }
    if (!radius_errors.empty())
    {
        rmse = std::sqrt(rmse / static_cast<double>(radius_errors.size()));
    }
    std::cout << "[Radius][LiDAR] max radius error = " << max_abs_error * 1000.0
              << " mm, RMSE = " << rmse * 1000.0 << " mm" << std::endl;
}

// 输出机械 LiDAR 内外边界半径和环宽质检统计
void printMechanicalRadiusQuality(
    const std::array<std::vector<double>, TARGET_NUM_CIRCLES>& inner_radii_by_center,
    const std::array<std::vector<double>, TARGET_NUM_CIRCLES>& outer_radii_by_center,
    double target_radius,
    double target_half_width)
{
    std::vector<double> centerline_errors;
    std::vector<double> half_width_errors;
    centerline_errors.reserve(TARGET_NUM_CIRCLES);
    half_width_errors.reserve(TARGET_NUM_CIRCLES);

    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
    {
        const double inner_radius = quantileDouble(inner_radii_by_center[i], 0.5);
        const double outer_radius = quantileDouble(outer_radii_by_center[i], 0.5);
        if (!std::isfinite(inner_radius) || !std::isfinite(outer_radius))
        {
            std::cout << "[Radius][LiDAR] circle " << i << ": insufficient inner/outer support"
                      << " (inner=" << inner_radii_by_center[i].size()
                      << ", outer=" << outer_radii_by_center[i].size() << ")" << std::endl;
            continue;
        }

        const double centerline_radius = 0.5 * (inner_radius + outer_radius);
        const double half_width = 0.5 * (outer_radius - inner_radius);
        const double centerline_error = centerline_radius - target_radius;
        const double half_width_error = half_width - target_half_width;
        centerline_errors.push_back(centerline_error);
        half_width_errors.push_back(half_width_error);

        std::cout << "[Radius][LiDAR] circle " << i
                  << ": inner=" << inner_radius * 1000.0 << " mm"
                  << ", outer=" << outer_radius * 1000.0 << " mm"
                  << ", centerline error=" << centerline_error * 1000.0 << " mm"
                  << ", half-width error=" << half_width_error * 1000.0 << " mm"
                  << ", support=[" << inner_radii_by_center[i].size()
                  << ", " << outer_radii_by_center[i].size() << "]"
                  << std::endl;
    }

    double max_centerline_error = 0.0;
    double centerline_rmse = 0.0;
    double max_half_width_error = 0.0;
    for (double error : centerline_errors)
    {
        max_centerline_error = std::max(max_centerline_error, std::fabs(error));
        centerline_rmse += error * error;
    }
    for (double error : half_width_errors)
    {
        max_half_width_error = std::max(max_half_width_error, std::fabs(error));
    }
    if (!centerline_errors.empty())
    {
        centerline_rmse = std::sqrt(centerline_rmse / static_cast<double>(centerline_errors.size()));
    }

    std::cout << "[Radius][LiDAR] max centerline radius error = "
              << max_centerline_error * 1000.0
              << " mm, centerline RMSE = " << centerline_rmse * 1000.0
              << " mm, max half-width error = " << max_half_width_error * 1000.0
              << " mm" << std::endl;
}

// 根据提取到的 annulus/边界点统计半径误差，用作圆心几何质检之外的辅助检查
void validateRadiusQuality(const pcl::PointCloud<pcl::PointXYZ>::Ptr& edge_cloud,
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr& centers_z0,
                           const Params& params,
                           LiDARType run_type)
{
    if (!edge_cloud || !centers_z0 || centers_z0->size() != TARGET_NUM_CIRCLES)
    {
        std::cout << "[Radius][LiDAR] skipped: need 4 centers in aligned board frame." << std::endl;
        return;
    }

    const double target_radius = params.circle_radius;
    const double annulus_half_width = params.annulus_half_width;
    const double inner_target = std::max(0.02, target_radius - annulus_half_width);
    const double outer_target = target_radius + annulus_half_width;

    if (run_type == LiDARType::Mech)
    {
        std::array<std::vector<double>, TARGET_NUM_CIRCLES> inner_radii_by_center;
        std::array<std::vector<double>, TARGET_NUM_CIRCLES> outer_radii_by_center;
        const double gate = 0.055;
        for (const auto& p : edge_cloud->points)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;

            int best_center = -1;
            double best_residual = std::numeric_limits<double>::max();
            double best_radius = 0.0;
            bool best_inner = true;
            for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
            {
                const auto& center = centers_z0->points[i];
                const double dx = static_cast<double>(p.x) - static_cast<double>(center.x);
                const double dy = static_cast<double>(p.y) - static_cast<double>(center.y);
                const double radius = std::sqrt(dx * dx + dy * dy);
                const double inner_residual = std::fabs(radius - inner_target);
                const double outer_residual = std::fabs(radius - outer_target);
                const bool use_inner = inner_residual <= outer_residual;
                const double residual = use_inner ? inner_residual : outer_residual;
                if (residual < best_residual)
                {
                    best_center = i;
                    best_residual = residual;
                    best_radius = radius;
                    best_inner = use_inner;
                }
            }
            if (best_center < 0 || best_residual > gate) continue;
            if (best_inner)
            {
                inner_radii_by_center[best_center].push_back(best_radius);
            }
            else
            {
                outer_radii_by_center[best_center].push_back(best_radius);
            }
        }
        printMechanicalRadiusQuality(inner_radii_by_center, outer_radii_by_center,
                                     target_radius, annulus_half_width);
        return;
    }

    std::array<std::vector<double>, TARGET_NUM_CIRCLES> radii_by_center;
    const double gate = 0.07;
    for (const auto& p : edge_cloud->points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;

        int best_center = -1;
        double best_residual = std::numeric_limits<double>::max();
        double best_radius = 0.0;
        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            const auto& center = centers_z0->points[i];
            const double dx = static_cast<double>(p.x) - static_cast<double>(center.x);
            const double dy = static_cast<double>(p.y) - static_cast<double>(center.y);
            const double radius = std::sqrt(dx * dx + dy * dy);
            const double residual = std::fabs(radius - target_radius);
            if (residual < best_residual)
            {
                best_center = i;
                best_residual = residual;
                best_radius = radius;
            }
        }
        if (best_center >= 0 && best_residual <= gate)
        {
            radii_by_center[best_center].push_back(best_radius);
        }
    }
    printSolidRadiusQuality(radii_by_center, target_radius);
}

}  // namespace

// LiDAR 圆心提取批量测试入口
int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_center_test");
    ros::NodeHandle nh;

    if (argc < 3)
    {
        std::cerr << "Usage: rosrun fast_calib lidar_center_test <bag_path> <lidar_topic> [auto|solid|mech]" << std::endl;
        return 2;
    }

    Params params = loadParameters(nh);
    params.bag_path = argv[1];
    params.lidar_topic = argv[2];

    const std::string mode = argc >= 4 ? argv[3] : "auto";

    pcl::PointCloud<Common::Point>::Ptr cloud(new pcl::PointCloud<Common::Point>);
    LiDARType detected_type = LiDARType::Unknown;
    if (!loadCloudFromBag(params.bag_path, params.lidar_topic, cloud, detected_type))
    {
        return 1;
    }

    LiDARType run_type = detected_type;
    if (mode == "solid") run_type = LiDARType::Solid;
    if (mode == "mech") run_type = LiDARType::Mech;

    std::cout << "[LiDAR Test] Bag: " << params.bag_path << std::endl;
    std::cout << "[LiDAR Test] Topic: " << params.lidar_topic << std::endl;
    std::cout << "[LiDAR Test] Detected type: " << lidarTypeName(detected_type)
              << ", run type: " << lidarTypeName(run_type) << std::endl;

    LidarDetect lidar_detect(nh, params);
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_centers(new pcl::PointCloud<pcl::PointXYZ>);

    if (run_type == LiDARType::Solid)
    {
        lidar_detect.detect_solid_lidar(cloud, raw_centers);
    }
    else if (run_type == LiDARType::Mech)
    {
        lidar_detect.detect_mech_lidar(cloud, raw_centers);
    }
    else
    {
        ROS_ERROR("[LiDAR Test] Unknown LiDAR type.");
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr centers(new pcl::PointCloud<pcl::PointXYZ>);
    sortPatternCenters(raw_centers, centers, "lidar");

    std::cout << "[LiDAR Test] Raw center count: " << raw_centers->size() << std::endl;
    std::cout << "[LiDAR Test] Sorted center count: " << centers->size() << std::endl;
    for (size_t i = 0; i < centers->size(); ++i)
    {
        const auto& p = centers->points[i];
        std::cout << "[LiDAR Test] Center " << i << ": "
                  << std::fixed << std::setprecision(6)
                  << p.x << ", " << p.y << ", " << p.z << std::endl;
    }
    validateTargetGeometry(centers, params.delta_width_circles, params.delta_height_circles, "LiDAR");
    validateRadiusQuality(lidar_detect.getEdgeCloud(), lidar_detect.getCenterZ0Cloud(), params, run_type);
    saveCenterCoordinates(centers, params, params.bag_path);
    saveDebugCloud(lidar_detect.getPlaneCloud(), lidar_detect.getAnnulusOriginalCloud(),
                   lidar_detect.getBoundaryOriginalCloud(),
                   centers, params, params.bag_path);

    return centers->size() == TARGET_NUM_CIRCLES ? 0 : 1;
}
