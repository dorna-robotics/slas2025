import numpy as np
import cv2 as cv
from shapely.geometry import LineString, Point, Polygon, box
from shapely.affinity import scale
from itertools import permutations
import time

def volume_estimator(h, tube):
    ref = {
        "15ml_cap": [[0, 0.0], [2, 87.28], [3, 111.16999999999999], [4, 135.97], [5, 157.1], [6, 180.07], [7, 203.04], [8, 225.09], [9, 248.06], [10, 269.19], [11, 292.15999999999997], [12, 312.37], [13, 331.67], [14, 353.71999999999997], [15, 372.09]],
        "50ml_cap": [[0, 0.0], [5, 51.45], [10, 90.04], [15, 128.63], [20, 169.97000000000003], [25, 208.56], [30, 245.31], [35, 283.89000000000004], [40, 320.64000000000004], [45, 359.23], [50, 395.98]]
    }

    for i in range(len(ref[tube])-1):
        v1, h1 = ref[tube][i]
        v2, h2 = ref[tube][i+1]

        # Check if h is between the second values of consecutive elements (Linear)
        if h1 <= h <= h2:
            # Linear interpolation between (x1, y1) and (x2, y2)
            if i == 0:
                volume = v1 + (v2 - v1) * ((h - h1) / (h2 - h1))**2
            else:
                volume = v1 + (v2 - v1) * (h - h1) / (h2 - h1)
            return volume

    # h is larger 
    volume = v2 * (h/h2)  
    # If no interval is found, return None or a specific value
    return volume


def free_line(pixel_list, center, bb_list, padding, thickness):
    """
    Check if a thickened line between 'start' and 'end' intersects any ellipse 
    inscribed within the bounding boxes (adjusted for padding).
    """
    # Define the line segment and thicken it
    line_list = [LineString([pixel, center]) for pixel in pixel_list]
    thick_line_list = [line.buffer(thickness / 2, cap_style=1) for line in line_list]

    # List to store ellipses and padded bounding boxes for optional plotting
    ellipses = []
    padded_bbs = []

    # Check intersection with all ellipses
    for bb in bb_list:
        # Calculate the coordinates of the expanded bounding box
        bb_array = np.array(bb)
        # Create new expanded bounding box with padding
        padded_bb = np.array([
            [bb_array[0][0] - padding, bb_array[0][1] - padding],
            [bb_array[1][0] + padding, bb_array[1][1] - padding],
            [bb_array[2][0] + padding, bb_array[2][1] + padding],
            [bb_array[3][0] - padding, bb_array[3][1] + padding]
        ])
        
        # Calculate the center and width/height of the padded bounding box
        center = padded_bb.mean(axis=0)
        width = np.abs(padded_bb[1][0] - padded_bb[0][0])
        height = np.abs(padded_bb[2][1] - padded_bb[1][1])

        # Define the inscribed ellipse using scaling
        unit_circle = Point(center).buffer(1)  # Create a unit circle
        ellipse = scale(unit_circle, xfact=width / 2, yfact=height / 2, origin=(center[0], center[1]))
        ellipses.append(ellipse)
        padded_bbs.append(padded_bb)
        # Check if thickened line intersects the ellipse
        if any([thick_line.intersects(ellipse) for thick_line in thick_line_list]):
            return False  # Invalid if it intersects

    return True

def free_line_old(start, end, bb_list, padding, thickness):
    """
    Check if a thickened line between 'start' and 'end' intersects any ellipse 
    inscribed within the bounding boxes (adjusted for padding).
    """
    # Define the line segment and thicken it
    line = LineString([start, end])
    thick_line = line.buffer(thickness / 2, cap_style=2)  # Thickness as radius

    # List to store ellipses and padded bounding boxes for optional plotting
    ellipses = []
    padded_bbs = []

    # Check intersection with all ellipses
    for bb in bb_list:
        # Calculate the coordinates of the expanded bounding box
        bb_array = np.array(bb)
        # Create new expanded bounding box with padding
        padded_bb = np.array([
            [bb_array[0][0] - padding, bb_array[0][1] - padding],
            [bb_array[1][0] + padding, bb_array[1][1] - padding],
            [bb_array[2][0] + padding, bb_array[2][1] + padding],
            [bb_array[3][0] - padding, bb_array[3][1] + padding]
        ])
        
        # Calculate the center and width/height of the padded bounding box
        center = padded_bb.mean(axis=0)
        width = np.abs(padded_bb[1][0] - padded_bb[0][0])
        height = np.abs(padded_bb[2][1] - padded_bb[1][1])

        # Define the inscribed ellipse using scaling
        unit_circle = Point(center).buffer(1)  # Create a unit circle
        ellipse = scale(unit_circle, xfact=width / 2, yfact=height / 2, origin=(center[0], center[1]))
        ellipses.append(ellipse)
        padded_bbs.append(padded_bb)
        # Check if thickened line intersects the ellipse
        if thick_line.intersects(ellipse):
            return False  # Invalid if it intersects

    return True


def valid_pick_old(pose, bb_list, robot, cam_T, camera_matrix, dist_coeffs, padding, length, thickness, finger_location, center):
    # iinit
    valid = False
    start = (0, 0)
    end = (0, 0)

    T_target_to_frame = robot.kinematic.xyzabc_to_mat(pose)
    T_target_to_cam = np.matmul(np.linalg.inv(cam_T), T_target_to_frame)
    #  X, Y, Z, O
    X = np.array([T_target_to_cam[i, 0] for i in range(3)])
    O = np.array([T_target_to_cam[i, 3] for i in range(3)])

    #Perform projection
    point_list = np.array([O+(length/2)*X, O-(length/2)*X])

    try:
        res_list, _ =  cv.projectPoints(point_list, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
        start = ((res_list[0][0][0]), (res_list[0][0][1]))
        end = ((res_list[1][0][0]), (res_list[1][0][1]))
        if center is not None:
            offset = (center[0]-(start[0]+end[0])/2, center[1]-(start[1]+end[1])/2)
            start = (int(start[0]+offset[0]), int(start[1]+offset[1]))
            end = (int(end[0]+offset[0]), int(end[1]+offset[1]))

        valid = free_line(start, end, bb_list, padding, thickness)
    except:
        pass
    return valid, start, end



def valid_pick(pose, bb_list, robot, cam_T, camera_matrix, dist_coeffs, padding, length, thickness, finger_location, center):
    # iinit
    valid = False
    
    pose_all = [pose[0:3]+robot.kinematic.rotate_rvec(rvec=pose[3:], axis=[0,0,1], angle=r, local=True) for r in finger_location]
    point_list = []
    for pose in pose_all:
        T_target_to_frame = robot.kinematic.xyzabc_to_mat(pose)
        T_target_to_cam = np.matmul(np.linalg.inv(cam_T), T_target_to_frame)
        #  X, Y, Z, O
        X = np.array([T_target_to_cam[i, 0] for i in range(3)])
        O = np.array([T_target_to_cam[i, 3] for i in range(3)])
        point_list.append([O+(length/2)*X])

    # add O
    point_list.append([O])
    
    #Perform projection
    try:
        res_list, _ =  cv.projectPoints(np.array(point_list), np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
        pixel_list = [((res_list[i][0][0]), (res_list[i][0][1])) for i in range(len(res_list))]
        projected_center = pixel_list.pop(-1)
        offset = (center[0]-projected_center[0], center[1]-projected_center[1])
        for i in range (len(pixel_list)):
            pixel_list[i] = (int(pixel_list[i][0]+offset[0]), int(pixel_list[i][1]+offset[1]))

        valid = free_line(pixel_list, center, bb_list, padding, thickness)
    except:
        pass
    return valid, pixel_list, center


def best_pick(detection_result, rvec_base, joint, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening, freedom, gripper_thickness=8, xyz_min=110, aspect_ratio=0.9, num_sample=32, seach_rotation=[0, 360], finger_location=[0, 180], bb_radius=float('inf')): 

    # init
    retval = None
    rotation = [seach_rotation[0]+i*(seach_rotation[1]-seach_rotation[0])/num_sample for i in range(num_sample)]
    
    for i in range(len(detection_result)):
        # belongs to the pick class
        if "tcp" not in detection_result[i]:
            continue
        
        #xyz
        xyz = detection_result[i]["xyz"]
        
        # xyz_min
        if xyz[2] < xyz_min:
            continue

        # aspect ratio
        corners = np.array(detection_result[i]["corners"])
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        aspect_ratio = np.min(sides) / np.max(sides)
        if aspect_ratio < aspect_ratio:
            continue
        
        # tcp
        robot.kinematic.set_tcp_xyzabc(detection_result[i]["tcp"])

        # bb list
        bb_list = [r["corners"] for r in detection_result[:i]+detection_result[i+1:] if np.linalg.norm(np.array(r["center"])-np.array(detection_result[i]["center"])) < bb_radius]

        # candidates
        pose_candidate = [xyz+robot.kinematic.rotate_rvec(rvec=rvec_base, axis=[0,0,1], angle=r, local=True) for r in rotation]
        
        # valid candidates
        pose_valid = []
        pose_not_valid = []
        for pose in pose_candidate:
            valid, pixel_list, o = valid_pick(pose, bb_list, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening, gripper_thickness, finger_location, center=detection_result[i]["center"])
            if valid:
                pose_valid.append([pose, pixel_list, o])
            else:
                pose_not_valid.append([pose, pixel_list, o])
        
        # best pose
        if len(pose_valid):
            candiate = [x[0] for x in pose_valid]
            pose_best = robot.kinematic.nearest_pose(candiate, joint, freedom)
            label = detection_result[i]["cls"]
            indx = candiate.index(pose_best)
            start = pose_valid[indx][1]
            end = pose_valid[indx][2]
            retval = pose_best, label, start, end, pose_valid, pose_not_valid, detection_result[i]
            break

    return retval
def best_pick_old(detection_result, rvec_base, joint, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening, freedom, gripper_thickness=8, xyz_min=110, aspect_ratio=0.9, num_sample=32, seach_rotation=[0, 360], finger_location=[0, 180]): 

    # init
    retval = None
    rotation = [seach_rotation[0]+i*(seach_rotation[1]-seach_rotation[0])/num_sample for i in range(num_sample)]
    
    for i in range(len(detection_result)):
        # belongs to the pick class
        if "tcp" not in detection_result[i]:
            continue
        
        #xyz
        xyz = detection_result[i]["xyz"]
        
        # xyz_min
        if xyz[2] < xyz_min:
            continue

        # aspect ratio
        corners = np.array(detection_result[i]["corners"])
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        aspect_ratio = np.min(sides) / np.max(sides)
        if aspect_ratio < aspect_ratio:
            continue
        
        # tcp
        robot.kinematic.set_tcp_xyzabc(detection_result[i]["tcp"])

        # bb list
        bb_list = [r["corners"] for r in detection_result]
        bb_list.pop(i)

        # candidates
        pose_candidate = [xyz+robot.kinematic.rotate_rvec(rvec=rvec_base, axis=[0,0,1], angle=r, local=True) for r in rotation]
        
        # valid candidates
        pose_valid = []
        pose_not_valid = []
        for pose in pose_candidate:
            valid, start, end = valid_pick(pose, bb_list, robot, T_cam_inv, camera_matrix, dist_coeffs, padding, gripper_opening, gripper_thickness, finger_location, center=detection_result[i]["center"])
            if valid:
                pose_valid.append([pose, start, end])
            else:
                pose_not_valid.append([pose, start, end])
        
        # best pose
        if len(pose_valid):
            candiate = [x[0] for x in pose_valid]
            pose_best = robot.kinematic.nearest_pose(candiate, joint, freedom)
            label = detection_result[i]["cls"]
            indx = candiate.index(pose_best)
            best_pxl_listart = pose_valid[indx][1]
            best_center = pose_valid[indx][2]
            retval = pose_best, label, best_pxl_listart, best_center, pose_valid, pose_not_valid, detection_result[i]
            break

    return retval


def decap(robot, cap_type, output_gripper_config, output_decap_config, place_position, place_down, tcp, round):
    # release vial
    robot.set_output(output_gripper_config[0], output_gripper_config[2])

    # go down
    robot.go((np.array(place_position[cap_type])+np.array(place_down[cap_type])).tolist(), tcp=tcp[cap_type], motion="lmove", freedom=None)

    #hold
    robot.set_output(output_decap_config[0], output_decap_config[1])
    for i in range(round):
        # close
        robot.set_output(output_gripper_config[0], output_gripper_config[1])
        # rotate
        robot.lmove(rel=1, z=1.2, a=-307.278746, b=-127.253498, vel=1500, accel=10000, jerk=40000)
        #robot.lmove(rel=1, z=2.2, a=-178.936936, b=180.238781, c=-1.284735, vel=800, accel=10000, jerk=20000)
        # open
        robot.set_output(output_gripper_config[0], output_gripper_config[2])
        # rotate back
        robot.jmove(rel=1, j5=90, vel=400, accel=10000, jerk=40000)
        #robot.jmove(rel=1, j5=180, vel=300, accel=10000, jerk=20000)

    # grab the cap
    #robot.set_output(output_gripper_config[0], output_gripper_config[1])


def barcode_read(robot, detection_level, classification_barcode, tube, num_img=5, bound=[0, 180]):
    barcode = None
    corners = None
    volume = 0
    # go to the start
    robot.jmove(rel=0, j5=-90, vel=400, accel=10000, jerk=40000)
    
    for i in range(num_img+1):
        time.sleep(0.1)

        # level
        if corners is None:
            result = detection_level.run()
            if result and result[0]["cls"] == "level":
                corners = result[0]["corners"]
                h = max([pxl[1] for pxl in corners])-min([pxl[1] for pxl in corners])
                volume = volume_estimator(h, tube)
                
        # barcode
        if barcode is None:
            result = classification_barcode.run()
            if result and result[0]["cls"] != "empty":
                barcode = result[0]["cls"]
                

        # break
        if corners is not None and barcode is not None:
            break
        
        # move
        robot.jmove(rel=1, j5=(bound[1]-bound[0])/num_img)
        time.sleep(0.1)

    return volume, corners, barcode


def tube_img_barcode(img, corners, vertical_flip=True):
    # plot the level to image
    if corners is not None:
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the rectangle (closed polygon)
        cv.polylines(img, [pts], isClosed=True, color=(203, 0, 255), thickness=1)

    if vertical_flip:
        img = cv.flip(img, 0)

    return img

class Plane_finder:
    def __init__(self, camera_matrix, dist_coeffs, frame_mat_inv, kinematic, min_error_threshold=5.0):
        """
        Initialize the Pose_estimator class.
        
        :param camera_matrix: Camera intrinsic matrix.
        :param dist_coeffs: Distortion coefficients of the camera.
        :param min_error_threshold: The minimum reprojection error threshold to accept a solution.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.frame_mat_inv = frame_mat_inv
        self.kinematic = kinematic
        self.min_error_threshold = min_error_threshold


    def calculate_reprojection_error(self, object_points, image_points, rvec, tvec):
        """
        Calculate the reprojection error.

        :param object_points: 3D object points.
        :param image_points: 2D image points.
        :param rvec: Rotation vector.
        :param tvec: Translation vector.
        :return: Average reprojection error in pixels.
        """
        # Project 3D points to 2D
        projected_points, _ = cv.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)

        # Calculate the Euclidean distance between the projected points and the actual image points
        error = np.sqrt(np.sum((projected_points - image_points) ** 2, axis=1))
        # Return the average error in pixels
        return np.mean(error)

    def estimate_pose(self, bb_plane, corners, shape, thr=50, overlap=0.7):
        """
        Estimate the pose (rvec and tvec) based on object points and image points.

        :param object_points: 3D object points.
        :param image_points: 2D image points.
        :return: (rvec, tvec) if a solution is found, None otherwise.
        """
        # Initialize variables
        best_rvec = None
        best_tvec = None
        best_reprojection_error = float('inf')  # Initialize with a large value

        # Define the 3D object points
        object_points = np.array([
        [0, 0, 0],   # Corner 1
        [shape[0], 0, 0],   # Corner 2
        [shape[0], shape[1], 0],
        [0, shape[1], 0]], dtype=np.float32)

        # contour
        bb_contour = np.array(bb_plane, dtype=np.int32).reshape((-1, 1, 2)) 
        polygon_bb_plane = box(*Polygon(bb_plane).bounds)

        # remove all the corners that are far from the plane_bb
        corners = [c for c in corners if abs(cv.pointPolygonTest(bb_contour, c, True)) < thr]
        if len(corners) < len(object_points):
            return None

        # all the combinations
        candidates = []
        for pts in permutations(corners, len(object_points)):
            # polygon
            polygon = Polygon(pts)

            # check if the polygon is valid
            if not polygon.is_valid:
                continue

            # cehck if polygon and bb overlap significantly
            polygon_bb = box(*polygon.bounds)  # Create a polygon from the bounding box
            if polygon_bb.intersection(polygon_bb_plane).area / (polygon_bb.union(polygon_bb_plane).area) < overlap:
                continue

            # add to the candidates
            candidates.append(pts)

        # Try all 
        for perm in candidates:
            # Solve PnP using RANSAC to get the rotation (rvec) and translation (tvec)
            try:
                success, rvec, tvec, _ = cv.solvePnPRansac(object_points, np.array(perm, dtype=np.float32), 
                                                                self.camera_matrix, self.dist_coeffs)    
                if success:
                    # Calculate reprojection error
                    reprojection_error = self.calculate_reprojection_error(object_points, np.array(perm, dtype=np.float32), rvec, tvec)

                    # If the reprojection error is smaller than the threshold, update the best solution
                    if reprojection_error < best_reprojection_error and reprojection_error < self.min_error_threshold:
                        best_reprojection_error = reprojection_error
                        best_rvec = rvec
                        best_tvec = tvec
            except Exception as e:
                print("Error:", e)
                pass
        if best_rvec is not None and best_tvec is not None:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv.Rodrigues(best_rvec)
            
            # Extract Z-axis direction
            z_axis = rotation_matrix[:, 2]  # Third column of rotation matrix
            if z_axis[2] < 0:
                x_axis = rotation_matrix[:, 0]  # X-axis (1st column)
                y_axis = rotation_matrix[:, 1]  # Y-axis (2nd column)

                # Flip the X and Y axes
                new_x_axis = y_axis
                new_y_axis = x_axis

                # Recalculate the Z-axis using the cross-product of new X and Y
                new_z_axis = np.cross(new_x_axis, new_y_axis)

                # Form the new rotation matrix
                rotation_matrix = np.column_stack((new_x_axis, new_y_axis, new_z_axis))


            # Create a 4x4 transformation matrix
            transformation_matrix = np.eye(4)  # Initialize as identity matrix
            transformation_matrix[:3, :3] = rotation_matrix  # Top-left 3x3 is the rotation matrix
            transformation_matrix[:3, 3] = best_tvec.flatten()    # Top-right 3x1 is the translation vector
            T_target_to_frame = np.matmul(self.frame_mat_inv, transformation_matrix)

            # adjust z axis of T_target_to_frame
            R = T_target_to_frame[:3, :3]
            t = T_target_to_frame[:3, 3]

            # Keep the original X-axis
            x_axis = R[:, 0]
            x_axis[2,0] = 0

            # Reverse the original Z-axis
            z_axis = np.array([[0], [0], [-1]])

            # Calculate the new Y-axis as the cross-product of Z and X
            y_axis = np.cross(z_axis.flatten(), x_axis.flatten()).reshape(-1, 1)

            # Normalize the axes
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)

            # Reassemble the new rotation matrix
            R_new = np.column_stack((x_axis, y_axis, z_axis))

            # Form the new transformation matrix
            T_target_to_frame[:3, :3] = R_new
            T_target_to_frame[:3, 3] = t  # Keep the original translation

            pose_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()

            # bset rvec and tvec
            self.best_rvec = best_rvec
            self.best_tvec = best_tvec
            return pose_target_to_frame
        else:
            return None


def vial_pick_candidate(candidates, r_base, joint, kinematic, freedom, rotation=[-135, -45, 45, 135], aspect_ratio=0.95): 

    # init
    retval = None
    
    for r in candidates:
        # aspect ratio
        corners = np.array(r["corners"])
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if np.min(sides) / np.max(sides) < aspect_ratio:
            continue
        
        # tcp
        kinematic.set_tcp_xyzabc(r["tcp"])

        # candidates
        pose_candidate = [r["tvec"]+kinematic.rotate_rvec(rvec=r_base, axis=[0,0,1], angle=a, local=True) for a in rotation]
        
        # best pose
        r["rvec"] = kinematic.nearest_pose(pose_candidate, joint, freedom)[3:6]
        retval = dict(r)
        break

    return retval