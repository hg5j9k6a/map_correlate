#! usr/bin/env python3

import open3d as o3d # version=0.12.0
import numpy as np
import cv2
import base64
import copy
import math
import time
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.spatial import KDTree
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def readb64_to_gray(encoded_data):
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def convert_img_to_points(file_path,resolution):
    png_path = file_path #"/home/joeclin/farobot_dev_env/data/far_app_data/app_map/jj_demo.png"
    # yaml_path = "/home/joeclin/farobot_dev_env/data/far_app_data/app_map/jj_demo.yaml"
    threshold = 127

    # imitate getying base64 format
    read_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    im_b64 = base64.b64encode(cv2.imencode('.png', read_img)[1])
    gray_img = np.asarray( readb64_to_gray(im_b64),np.uint8)
        
    height, width = gray_img.shape
    raw_img = np.asarray(gray_img).flatten()
    
    points = []

    for idx in np.where(raw_img<threshold)[0]:
        h = idx // width
        w = idx % width 
        h_point, w_point = (resolution*(height-h),resolution*w)
        for i in range(5):
            points.append([h_point,w_point,0.01*i])
    return np.array(points)

def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def convert_points_to_img(points,resolution):
    max_h = max(points, key=lambda item: item[0])[0]/resolution
    min_h = min(points, key=lambda item: item[0])[0]/resolution
    max_w = max(points, key=lambda item: item[1])[1]/resolution
    min_w = min(points, key=lambda item: item[1])[1]/resolution

    canvas_h = math.ceil(max_h - min_h) + 1
    canvas_w = math.ceil(max_w - min_w) + 1
    canvas = np.zeros((canvas_h,canvas_w))
    weights = []

    for point in points:
        h_pixel = int(point[0]/resolution - min_h)
        w_pixel = int(point[1]/resolution - min_w)
        canvas[h_pixel,w_pixel] = 1
        weights.append([h_pixel*resolution,w_pixel*resolution])
    return canvas,weights

def convert_flip_points_to_img(points,resolution):
    max_h = max(points, key=lambda item: item[0])[0]/resolution
    min_h = min(points, key=lambda item: item[0])[0]/resolution
    max_w = max(points, key=lambda item: item[1])[1]/resolution
    min_w = min(points, key=lambda item: item[1])[1]/resolution

    canvas_h = math.ceil(max_h - min_h) + 1
    canvas_w = math.ceil(max_w - min_w) + 1
    canvas = np.zeros((canvas_h,canvas_w))

    for point in points:
        h_pixel = int(max_h - point[0]/resolution)
        w_pixel = int(point[1]/resolution - min_w)
        canvas[h_pixel,w_pixel] = 1
    return canvas

def rotate_img(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(0))

def draw_registration_result(source, target,transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,transformation):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5 * 10 # voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

if __name__ == '__main__':
    resolution = 0.03
    voxel_size = resolution
    trans_init = np.identity(4)

    source_file = "/home/joeclin/farobot_dev_env/data/far_app_data/app_map/AUO.png"
    target_file = "/home/joeclin/farobot_dev_env/data/far_app_data/app_map/AUO.png"

    source_points = convert_img_to_points(source_file,resolution)
    target_points = convert_img_to_points(target_file,resolution)

    target_pcd = points_to_pcd(target_points)

    # source_offset = np.mean(source_points, axis=0)
    # target_offset = np.mean(target_points, axis=0)

    source_points = (source_points)
    target_points = (target_points)

    # source_points = np.dot(source_points,Rz(np.deg2rad(32)))

    source_pcd = points_to_pcd(source_points)
    target_pcd = points_to_pcd(target_points)

    # draw_registration_result(normalize_source_pcd, normalize_target_pcd,trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    print(result_ransac.transformation)


    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                 voxel_size,result_ransac.transformation)
    print(result_icp)
    print(result_icp.transformation)
    draw_registration_result(source_pcd, target_pcd, result_icp.transformation)

    res_source_points = np.asarray(source_pcd.transform(result_icp.transformation).points)

    threshold = 245
    # imitate getying base64 format
    read_img = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
    im_b64 = base64.b64encode(cv2.imencode('.png', read_img)[1])
    gray_img = np.asarray( readb64_to_gray(im_b64),np.uint8)
        
    height, width = gray_img.shape

    gray_img_show = cv2.resize(gray_img, (1200, 900), interpolation=cv2.INTER_AREA)
    cv2.imshow('Target_img', gray_img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    raw_img = np.asarray(gray_img).flatten()
    gray_canvas = np.ones((height*width)) * 205
    gray_canvas[np.where(raw_img > threshold)] = 255
    gray_canvas = gray_canvas.reshape((height,width))

    for point in res_source_points:
        h_pixel = height - int(point[0]//resolution)
        w_pixel = int(point[1]//resolution)
        cv2.circle(gray_canvas,(w_pixel, h_pixel), 1, 0, -1)

    gray_canvas_show = cv2.resize(gray_canvas, (1200, 900), interpolation=cv2.INTER_AREA)
    cv2.imshow('Res_img', gray_canvas_show.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_three_channel = cv2.cvtColor(gray_canvas.astype('uint8'), cv2.COLOR_GRAY2BGR)
    im_b64 = base64.b64encode(cv2.imencode('.png', gray_three_channel)[1])
    path = "/home/joeclin/ros2_ws/src/output.txt"
    f = open(path, 'w')
    f.write(im_b64.decode('ascii'))
    f.close()


        