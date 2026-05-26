# FAST-Calib2

## LiDAR-Camera Extrinsic Calibration with Reflective Annular Targets

FAST-Calib2 extends [FAST-Calib](https://github.com/hku-mars/FAST-Calib) to LiDAR-camera modules that were previously hard to calibrate due to **low-quality point clouds**. With a custom-designed reflective annular calibration target, it enables robust center extraction on **large-spot solid-state and mechanical LiDARs**, including Mid360, Avia, Ouster, XT32, JT128, Airy, E1R, and Adaps Photonics Spad LiDAR.

**Key highlights include:**

1. A self-designed 3D reflective annular calibration target that avoids center extraction errors caused by hole-edge inflation and bleeding artifacts in previous circular-hole calibration boards.
2. A robust concentric-circle fitting method that uses the fixed inner and outer annulus radii as geometric constraints.
3. Automatic calibration board ROI extraction without manual pass-through tuning.
4. Geometry and radius quality checks for extracted annulus centers.
5. Single-scene and multi-scene LiDAR-camera extrinsic calibration without initial extrinsic parameters.

📬 For further assistance or inquiries, please feel free to contact Chunran Zheng at zhengcr@connect.hku.hk.

<p align="center">
  <img src="./pics/cover.jpg" width="100%">
  <font color=#a0a0a0 size=2>Mid360 calibration example.</font>
</p>

## 1. Prerequisites

PCL>=1.8, OpenCV>=4.0.

## 2. Calibration Target

FAST-Calib2 uses four reflective annuli and four visual markers on one board. The annuli are used by LiDAR center extraction, while the visual markers are used by the camera pipeline.

Materials:

- Board: PVC
- Reflective annulus stickers: 3M engineering-grade reflective film

<p align="center">
  <img src="./pics/FAST-Calib2-board.png" width="100%">
  <font color=#a0a0a0 size=2>Reflective annular calibration target and annotated dimensions.</font>
</p>

DIY Calibration Target Tips:

1. Fabricate the board based on the schematic. Ensure a minimum thickness of 1 cm to avoid bending.
2. Apply reflective annulus stickers to the designated ring positions on the fabricated board.

## 3. Method Overview

Both LiDAR pipelines first **locate the calibration board automatically**, fit the board plane, and align the plane to `Z=0`. Center extraction is then performed in the aligned board frame.

Solid-state LiDAR pipeline:

1. Extract high-reflectivity annulus points on the fitted board plane.
2. Cluster the extracted annulus points.
3. Fit robust single circles as the default center estimate.
4. Optionally extract annulus boundary points and fit fixed inner/outer radius concentric circles.
5. Select the best result by checking four-center geometry consistency against the known target geometry.

Mechanical LiDAR pipeline:

1. Use LiDAR `ring` order to find intensity transition points on the annulus boundary.
2. Try both interpolated boundary points and high-reflectivity-side boundary points.
3. Cluster the extracted boundary points.
4. Fit fixed inner/outer radius concentric circles.
5. Select the best result by checking four-center geometry consistency against the known target geometry.

The final quality checks include center-to-center geometry error and annulus radius consistency.

## 4. Run Examples

Prepare static acquisition data in the `calib_data` folder (Download the example data from [Google Drive](https://drive.google.com/drive/folders/1VnMCsGj3Gat7dxe6IION0SfS7jYNMw1g?usp=sharing)):

- rosbag containing point cloud messages
- corresponding image

Run single-scene calibration:

```bash
roslaunch fast_calib calib.launch
```

After collecting at least three scenes, run multi-scene joint calibration:

```bash
roslaunch fast_calib multi_calib.launch
```

Typical multi-scene target placement:

<p align="center">
  <img src="./pics/multi-scene.jpg" width="100%">
  <font color=#a0a0a0 size=2>Placement of the calibration target for multi-scene data collection: (a) facing forward, (b) oriented to the right, (c) oriented to the left.</font>
</p>

## 5. Standalone LiDAR Center Extraction Test

<details>
<summary>Show Unit Test Usage</summary>

The repository also provides a LiDAR-only test tool for checking annulus center extraction before running full camera-LiDAR calibration.

Load parameters:

```bash
rosparam load config/qr_params.yaml /
rosparam set /output_path /home/chunran/02_calib_ws/src/FAST-Calib/output
```

Run solid-state LiDAR data:

```bash
rosrun fast_calib lidar_center_test calib_data/fast-calib2-data/left.bag /livox/lidar solid
rosrun fast_calib lidar_center_test calib_data/fast-calib2-data/mid.bag /livox/lidar solid
rosrun fast_calib lidar_center_test calib_data/fast-calib2-data/right.bag /livox/lidar solid
```

Run mechanical LiDAR data:

```bash
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/left.bag /lidar_points mech
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/mid.bag /lidar_points mech
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/right.bag /lidar_points mech
```

The test tool writes:

- `*_centers.txt`: extracted annulus center coordinates
- `*_debug_cloud.pcd`: board point cloud, annulus points, boundary points, and center markers for visualization

Debug PCD colors:

- Board points: intensity color map
- Annulus points: green
- Solid-LiDAR boundary points: red
- Centers: white spheres

</details>
