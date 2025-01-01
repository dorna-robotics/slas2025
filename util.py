import numpy as np
import cv2 as cv
from shapely.geometry import LineString, Polygon as ShapelyPolygon
import time

def free_line(start, end, bb_list, padding):
    """
    Check if a line between 'start' and 'end' intersects any padded bounding box.
    """
    # Define the line segment
    line = LineString([start, end])

    # Check intersection with all bounding boxes
    for bb in bb_list:
        original_poly = ShapelyPolygon(bb)
        padded_bb = np.array(bb) + np.array([[-padding, -padding],
                                             [padding, -padding],
                                             [padding, padding],
                                             [-padding, padding]])
        padded_poly = ShapelyPolygon(padded_bb)

        # Check if line intersects padded bounding box
        if line.intersects(padded_poly):
            return False  # Invalid if it intersects

    return True


def valid_pick(pose, bb_list, robot, cam_T, camera_matrix, dist_coeffs, padding, length):

    T_target_to_frame = robot.kinematic.xyzabc_to_mat(pose)
    T_target_to_cam = np.matmul(np.linalg.inv(cam_T), T_target_to_frame)

    #  X, Y, Z, O
    X = np.array([T_target_to_cam[i, 0] for i in range(3)])
    O = np.array([T_target_to_cam[i, 3] for i in range(3)])

    #Perform projection
    point_list = np.array([O+(length/2)*X, O-(length/2)*X])

    res_list, _ =  cv.projectPoints(point_list, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
    start = (int(res_list[0][0][0]), int(res_list[0][0][1]))
    end = (int(res_list[1][0][0]), int(res_list[1][0][1]))

    valid = free_line(start, end, bb_list, padding)

    return valid, start, end


def best_pick(detection_result, rvec_base, joint, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening):
    # init
    retval = None
    rotation = [i*360/16 for i in range(32)]
    
    for i in range(len(detection_result)):
        # tcp
        robot.kinematic.set_tcp_xyzabc(detection_result[i]["tcp"])

        # bb list
        bb_list = [r["corners"] for r in detection_result]
        bb_list.pop(i)

        # candidates
        xyz = detection_result[i]["xyz"]
        pose_candidate = [xyz+robot.kinematic.rotate_rvec(rvec=rvec_base, axis=[0,0,1], angle=r, local=True) for r in rotation]
        
        # valid candidates
        pose_valid = []
        pose_not_valid = []
        for pose in pose_candidate:
            valid, start, end = valid_pick(pose, bb_list, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening)
            if valid:
                pose_valid.append([pose, start, end])
            else:
                pose_not_valid.append([pose, start, end])
        
        # best pose
        if len(pose_valid):
            candiate = [x[0] for x in pose_valid]
            pose_best = robot.kinematic.nearest_pose(candiate, np.array(joint)*np.pi/180)
            label = detection_result[i]["cls"]
            indx = candiate.index(pose_best)
            start = pose_valid[indx][1]
            end = pose_valid[indx][2]
            retval = pose_best, label, start, end, pose_valid, pose_not_valid
            break

    return retval


def decap(robot, cap_type, output_gripper_config, output_decap_config, place_position, tcp, round):
    # approach
    robot.go((np.array(place_position[cap_type])+np.array([0, 0, -23, 0, 0, 0])).tolist(), tcp=tcp[cap_type], motion="lmove")

    #hold
    robot.set_output(output_decap_config[0], output_decap_config[1])
    for i in range(round):
        # close
        robot.set_output(output_gripper_config[0], output_gripper_config[1])
        # rotate
        robot.lmove(rel=1, z=1.1, a=-307.278746,b=-127.253498, vel=500, accel=4000, jerk=10000)
        # open
        robot.set_output(output_gripper_config[0], output_gripper_config[2])
        # rotate back
        robot.jmove(rel=1, j5=90, vel=200, accel=2000, jerk=6000)

    # grab the cap
    robot.set_output(output_gripper_config[0], output_gripper_config[1]) # close


def barcode_read(robot, detection=None, num_img=10, bound=[-90, 90]):
    num_img = 10
    bound = [-90, 90]

    for i in range(num_img):
        robot.jmove(rel=0, j5=bound[0]+i*(bound[1]-bound[0])/num_img)
        time.sleep(0.1)
