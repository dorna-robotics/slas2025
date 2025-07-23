#############################################################
# robot and cameras
#############################################################
robot = {
    "ip": "192.168.254.87",
    "frame_in_world": [0, 0, 0, 0, 0, 0],
    "base_in_world": [62.5, 97.5, 0, 0, 0, -90],
    "aux_dir": [[0, 0, 0], [0, 0, 0]],
    "mode": "dorna_ta",
}

camera = {
    "pick":{
        "serial_number": "218622276562",
    },
    "barcode":{
        "serial_number": "",
    }
}
#############################################################
# motion and sim
#############################################################
sim = 0
speed = 0.2
max_pick = 1

#############################################################
# items
#############################################################
tube = {
    "15ml":{
        "height":120,
        "round":7,
        "angle": 90,
        "z_step": 1.5
    },
    "50ml":{
        "height": 117,
        "round":7,
        "angle": 90,
        "z_step": 1.5
    }
}

#############################################################
# holders
#############################################################
decapper = {
    "aux": [0, 0],
    #"frame": [(12*25+12.5), -(1*25+12.5), 0, 180, 0, 0],
    "frame": [311.9916522369349, -36.17889272057014, 1.020341447481627, 179.95014005057752, -0.015487510215226494, -0.057018180725687644],
    "15ml_cap": [25, 8.75, -tube["15ml"]["height"], 0, 0, 0],
    "15ml_cap_loose": [25, 8.75, -tube["15ml"]["height"]-5, 0, 0, 0],
    "50ml_cap": [25, 34.75, -tube["50ml"]["height"], 0, 0, 0],
    "50ml_cap_loose": [25, 34.75, -tube["50ml"]["height"]-5, 0, 0, 0],
    "scanner": [(1*25), (7*25), -tube["15ml"]["height"]+(-10), 0, 0, 0],
    "safe_z": 120,
    "open": [[1, 0, 0.25]],
    "close": [[1, 1, 0.25]],
}

cap_bin = {
    "aux": [0, 0],
    "frame": [(5*25), -(5*25), 0, 180, 0, 0],
    "drop": [0, 0, -60, 0, 0, 0],
    "safe_z": 70
}

#############################################################
# grippers
#############################################################
two_finger_gripper = {
    "15ml_cap":{
        "tool": [0, 0, 29, 0, 0, 0],
        "close": [[0, 1, 0.25]],
        "open": [[0, 0, 0.25]],
        "finger_width": 10,
        "gripper_opening": 43,
        "finger_location": [0, 180],
        "rvec_base": [180, 0, 0],
        "gripper_rotation": [
            {"axis": [0, 0, 1], "angle": 0},
            {"axis": [0, 0, 1], "angle": 180},
        ]
    },
    "50ml_cap":{
        "tool": [0, 0, 39, 0, 0, 0],
        "close": [[0, 1, 0.25]],
        "open": [[0, 0, 0.25]],
        "finger_width": 10,
        "gripper_opening": 43,
        "finger_location": [0, 180],
        "rvec_base": [180, 0, 0],
        "gripper_rotation": [
            {"axis": [0, 0, 1], "angle": 0},
            {"axis": [0, 0, 1], "angle": 180},
        ],
    },
}

#############################################################
# pick
#############################################################
tube_pick = {
    "aux": [0, 0],
    "joint": [90.175781, 79.958496, -108.325195, 0.219727, -67.456055, -90], 
}

#############################################################
# detections
#############################################################
detection_preset = {
    "tube":{
        "camera_mount":{
            "type": "dorna_ta_j4",
            "ej": [0 ,0, 0, 0, 0, 0, 0, 0],
            "T": [46.5174596+1+1+0+4-(5), 32.0776662-3+1-0-1.5+(-3), -4.24772615-3, -0.27547989, 0.27691881, 89.6939516],
        },
        'detection': {'cmd': 'od', 'path': 'model/tube.pkl', 'conf': 0.2, 'cls': []},
        'roi': {'corners': [[214.51, 11.63], [220.24, 377.26], [611.69, 374.4], [613.13, 10.19]], 'inv': False, 'crop': True, 'offset': 0},
        'sort': {'cmd': 'shuffle', 'max_det': 100}, 
        #'limit': {'xyz': {'x': [-150, 110], 'y': [200, 450], 'z': [110, 130], 'inv': 0}}, 
        'display': {'label': 0, 'save_img': True, 'save_img_roi': False}
        },
    'limit': {'xyz': {'x': [-150, 110], 'y': [200, 450], 'z': [110, 130], 'inv': 0}}, 
}


#############################################################
# motions
#############################################################
rack_to_scanner = {
    "pick": {
        "type": "robot", # robot, world, joint
        "loc": [None, tube_pick["aux"]],
        "frame": [0, 0, 0, 0, 0, 0],
        "tool": [None, None],
        "output": None,
        "approach": [[0, 0, -40, 0, 0, 0]],
        "exit": [[0, 0, -140, 0, 0, 0]]
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [decapper["scanner"], decapper["aux"]],
        "frame": decapper["frame"],
        "tool": [None, None],
        "approach": [[0, 0, -140, 0, 0, 0]],
        "cmd": [
            {"cmd": "jmove", "j5": -36, "rel": 1, "vel":1000, "accel":4000, "jerk":10000},
            {"cmd": "sleep", "time": 0.1},
            {"cmd": "jmove", "j5": -36, "rel": 1},
            {"cmd": "sleep", "time": 0.1},
            {"cmd": "jmove", "j5": -36, "rel": 1},
            {"cmd": "sleep", "time": 0.1},
            {"cmd": "jmove", "j5": -36, "rel": 1},
            {"cmd": "sleep", "time": 0.1},
            {"cmd": "jmove", "j5": -36, "rel": 1},
            {"cmd": "sleep", "time": 0.1},
            {"cmd": "jmove", "j5": 180, "rel": 1},
        ]
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 250, 
    "freedom": {"num":20, "range":[0.05, 0.05, 0.05], "early_exit":False}, "timeout": -1, "sim":sim,
}


scanner_to_decapper = {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["scanner"], decapper["aux"]],
        "frame": decapper["frame"],
        "tool": [None, None],
        "output": decapper["open"],
        "exit": [[0, 0, -140, 0, 0, 0]],
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [None, decapper["aux"]],
        "frame": decapper["frame"],
        "tool": [None, None],
        "output": None,
        "approach": [[0, 0, -140, 0, 0, 0]],
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 250, 
    "freedom": {"num":20, "range":[0.05, 0.05, 0.05], "early_exit":False}, "timeout": -1, "sim":sim,
}

decapper_to_cap_bin = {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, decapper["aux"]],
        "frame": decapper["frame"],
        "tool": [None, None],
        "output": None,
        "exit": [[0, 0, -40, 0, 0, 0]],
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [cap_bin["drop"], cap_bin["aux"]],
        "frame": cap_bin["frame"],
        "tool": [None, None],
        "output": None,
        "approach": [[0, 0, -40, 0, 0, 0]],
        "exit": [[0, 0, -40, 0, 0, 0]],
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 250, 
    "freedom": {"num":20, "range":[0.05, 0.05, 0.05], "early_exit":False}, "timeout": -1, "sim":sim,
}

decapper_to_tray = {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, decapper["aux"]],
        "frame": decapper["frame"],
        "tool": [None, None],
        "output": None,
        "approach": [[0, 0, -40, 0, 0, 0]],
        "exit": [[0, 0, -140, 0, 0, 0]],
    },
    "place": {
        "type": "robot", # robot, world, joint
        "loc": [None, tube_pick["aux"]],
        "frame": [0, 0, 0, 0, 0, 0],
        "tool": [None, None],
        "output": None,
        "approach": [[0, 0, -140, 0, 0, 0]],
        "exit": [[0, 0, -40, 0, 0, 0]],
    },
    "end":{
        "type": "joint",
        "loc":[tube_pick["joint"], tube_pick["aux"]]  
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 250, 
    "freedom": {"num":20, "range":[0.05, 0.05, 0.05], "early_exit":False}, "timeout": -1, "sim":sim,
}
