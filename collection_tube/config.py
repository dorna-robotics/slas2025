#############################################################
# robot and cameras
#############################################################
robot = {
    "ip": "192.168.254.85",
    "frame_in_world": [0, 0, 0, 0, 0, 0],
    "base_in_world": [127.5, 12.5, 0, 0, 0, 0],
    "aux_dir": [[0, 0, 0], [0, 0, 0]],
    "mode": "dorna_ta",
}

#############################################################
# motion and sim
#############################################################
sim = 0
speed = 0.4
max_pick = 3
#############################################################
# items
#############################################################
"""
collection tube
"""
collection_tube = {
    "height_tube_cap": 80,
    "height_cp": 20,
    "diameter_cap": 15,
    "height_tube": 75,
    "diameter_tube": 12,
    "kp": {
        "head": [0, -40, 0], 
        "tail": [0, -20, 0],
    }
}

"""
well plate
"""
well_plate = {
    "pitch": [21.67, 26.75], # pitch in x and y direction
    "tube": None,
    "length": [85, 127, 55],
    "length_w_tube": [85, 127, 85],
}

#############################################################
# tool cahger
#############################################################
# the tool z is pointing down, x is toward rail
tool_changer = {
    "suction_gripper":{
        "aux": [0 ,0],
        "frame": [(18*25+12.5), -(12.5), 0, 180 ,0 ,0],
        "connect": [-49.899002572290726, -4.074920378115335, -166.3728667032849, 0.09395135472557684, -0.18028435789341682, -0.6898671574861631], #[-46, 0, -167.5, 0, 0, 0],
    },

    "two_finger_gripper":{
        "aux": [0 ,0],
        "frame": [(18*25+12.5), (3*25+12.5), 0, 180 ,0 ,0],
        "connect":[-50.856544867528214, -1.6908675308354901, -166.73069237211251+1, 0.6055042189972611, -0.4442753165806921, -0.983765043945477], #[-46, 0, -167.5, 0, 0, 0], 
    },
    "safe_joint": [0, 103.271484, -132.824707, 0, -63.720703, -11.337891],
}   
#############################################################
# holders
#############################################################
# tube tary
tube_tray = {
    "aux": [0, 0],
    "frame": [0 ,0 ,0 ,0 ,0 ,0],
}

"""
Frame starts from bottom center ,x toward the letters, y toward the numbers and z pointing up
"""
plate_holder = {
    "aux": [0, 0],
    "frame": [(9*25 + 12.5), -(11*25+12.5), 0, 0, 0, 180], # set this up,
    "safe": [[0, 0, 270, 0, 0, 0]],
    #"safe_joint": [-75, 70, -140, 0,-25, -15],
    "safe_joint": [-57.546387, 49.064941, -105.534668, 0.197754,-33.442383, 122.475586],
}

"""
Frame starts from the bottom of the screw close to the cap holder, x is tworda the camera, y is toward the other screw and z is pointing up
"""
decapper = {
    "aux": [0, 0],
    "frame": [(13*25+12.5), -(11*25+12.5), 0, 0, 0, 180], # set this up
    "tube_holder": [-0.3020019584569127, -31.246361751904146, 0.9066128431270215, -0.06988831881268744, 0.07744518670531331, 0.06904262016323703], #[0, -25, 0, 0, 0, 0]
    "tube_holder_clearance": collection_tube["height_tube_cap"] + 15, # clearance above the tube holder
    "cap_holder": [-30.164097232305494, 21.304509037834293, 38.62966117852052+(1), 0.06866585650933951, 0.0568587885169989, 0.049802228037648395], #[-30, 25, 37, 0, 0, 0]
    "scanner": [-5*25, -25, 10, 0, 0, 0],
    "open": [[2, 0, 0.25]],
    "close": [[2, 1, 0.25]],
}
#############################################################
# grippers
#############################################################
tool_changer_gripper = {
    "tool": [0, 0, 22, 0, 0, 0],
    "connect": [[7, 0, 0.25]],
    "disconnect": [[7, 1, 0.25]],
    "idle": [[0, 0, 0], [1, 1, 0]]
}

suction_gripper = {
    "tool": [0, 0, 78+(1), 0, 0 ,0], 
    "tool_w_tube": [0, 40, 78+(1), 69.28203230275511, 69.28203230275511, -69.28203230275511], # center #78
    "on": [[1, 1, 0], [0, 1, 0]],
    "off": [[1, 1, 0], [0, 0, 0]],
}

two_finger_gripper = {
    "tool": [0, 0, 94+(2)+(4), 180, 0, 0],
    "tool_wo_tube": [0, 0, 71+(2)+(4), 180, 0, 0],
    "tool_w_tube": None,
    "tool_w_tube_cap": None,
    "close": [[0, 0, 0], [1, 0, 0.25]],
    "open": [[1, 1, 0.5], [0, 1, 0.5], [0, 0, 0.25]],
}

#############################################################
# detections
#############################################################

det_preset = {
    "collection_tube_kp":{
        "camera_mount":{
            "type": "dorna_ta_j4",
            "ej": [0 ,0, 0, 0, 0, 0, 0, 0],
            "T": [46.5174596+1+1+0+4, 32.0776662-3+1-0-1.5, -4.24772615-(0), -0.27547989, 0.27691881, 89.6939516],
        },
        'roi': {'corners': [[244.62, 83.35], [555.77, 31.73], [630.33, 459.03], [329.22, 471.93]], 'inv': False, 'crop': True, 'offset': 0}, 
        'detection': {'cmd': 'kp', 'path': 'model/collection_tube_kp.pkl', 'conf': 0.5, 'cls': {}}, 
        'sort': {'cmd': 'shuffle', 'max_det': 1}, 
        'limit': {'xyz': {'x': [-1000, 100], 'y': [-1000, -25], 'z': [0, 30], 'inv': 0}}, 
        'pose': {'cmd': 'kp', 'kp': {'tube': collection_tube["kp"]}, 'thr': 5},
        'display': {'label': 1, 'save_img': False, 'save_img_roi': False}
    },
    "tube_exist":{
        'roi': {'corners': [[273.3, 255.42], [468.31, 258.29], [466.87, 450], [267.56, 450]], 'inv': False, 'crop': True, 'offset': 0}, 
        'detection': {'cmd': 'od', 'path': 'model/tube_exist.pkl', 'conf': 0.5}, 
        'sort': {'cmd': 'shuffle', 'max_det': 1}, 
        'display': {'label': 1, 'save_img': False, 'save_img_roi': False}
    },
}
#############################################################
# adjustments
#############################################################
# well plate
#well_plate["frame"] = [238.22910284183757+(1), -281.80043173979976, 0.97980283472927, 0.1481681240094891, -0.09008006441539647, 179.97059787642038]
#well_plate["frame"] = [236.0486797363443, -281.8631822592238, 0.190252636220848, 0.8575079189706579, -0.2813564873656528, 179.71610956686533]
well_plate["frame"] = [237.79567976233722, -281.8059603358065, 0.8261422767794073, 1.0105966612737578, -0.32651207749238487, 179.73660117704614]


well_plate["aux"] = plate_holder["aux"]
well_plate["index"] = {f"{chr(97+i)}{j+1}": [(i-1.5)* well_plate["pitch"][0], (j-2) * well_plate["pitch"][1], 6.5, 0, 0, 0] for i in range(4) for j in range(5)} # a1, a2,...
well_plate["tube"] = {f"{chr(97+i)}{j+1}": [(i-1.5)* well_plate["pitch"][0], (j-2) * well_plate["pitch"][1], 6.5, 0, 0, 0] for i in range(4) for j in range(5)}
for k in well_plate["tube"]:
    well_plate["tube"][k][2] += collection_tube["height_tube_cap"]  # add the height of the collection tube to the z coordinate
well_plate["place"] = {f"{chr(97+i)}{j+1}": [(i-1.5)* well_plate["pitch"][0], (j-2) * well_plate["pitch"][1], 45, 0, 0, 0] for i in range(4) for j in range(5)} # a1, a2,...


# two finger gripper
two_finger_gripper["tool_w_tube"] = list(two_finger_gripper["tool_wo_tube"])
two_finger_gripper["tool_w_tube"][2] += collection_tube["height_tube"]
two_finger_gripper["tool_w_tube_cap"] = list(two_finger_gripper["tool_wo_tube"])
two_finger_gripper["tool_w_tube_cap"][2] += collection_tube["height_tube_cap"]

# decpper
decapper["tube_holder_w_tube"] = list(decapper["tube_holder"])
decapper["tube_holder_w_tube"][2] += collection_tube["height_tube"]
decapper["tube_holder_w_tube_cap"] = list(decapper["tube_holder"])
decapper["tube_holder_w_tube_cap"][2] += collection_tube["height_tube_cap"] + 10
decapper["cap_holder_above"] = list(decapper["cap_holder"])
decapper["cap_holder_above"][2] = decapper["tube_holder_clearance"]


#############################################################
# motions
#############################################################
tray_to_tube_exist= {
    "pick": {
        "type": "robot", # robot, world, joint
        "loc": [None, tube_tray["aux"]],
        "tool": [suction_gripper["tool"], suction_gripper["tool"]],
        "output": suction_gripper["on"],
        "cmd": [],
        "approach": [[0, 0, -10, 0, 0, 0]],
        "exit": [[0, 0, -30, 0, 0, 0]]
    },
    "end":{
        "type": "joint", # robot, world, joint
        "loc": [[-95.625, 27.597656, -92.790527, -98.349609, 97, 113], tube_tray["aux"]],
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "jmove", "speed": 0.7, "cont": 1, "corner": 100, 
    "freedom": {"num":50, "range":[0.5, 0.5, 0.5], "early_exit":False }, "timeout": -1, "sim":sim,
}

tube_exist_to_well_plate= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, well_plate["aux"]],
        "tool": [suction_gripper["tool_w_tube"], suction_gripper["tool_w_tube"]],
        "frame": well_plate["frame"],
        "output": suction_gripper["off"],
        "approach": [[0, 0, 50, 0, 0, 0]],
        "exit": [[20, 0, 0, 0, 0, 0], [20, 0, 50, 0, 0, 0]]
        },
    "end": {
        "type": "joint", # robot, world, joint
        "loc": [[-100, 57.041016, -130.891113, 0, -19.511719, -12.634277] , [0, 0]],
        },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "jmove", "speed": 0.3, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}

tube_exist_to_tray= {
    "pick": {
        "type": "joint", # robot, world, joint
        "loc": [[-100, 57.041016, -130.891113, 0, -19.511719, -12.634277] , [0, 0]],
        "output": suction_gripper["off"],
        },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "jmove", "speed": 0.7, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}

tray_to_well_plate= {
    "pick": {
        "type": "robot", # robot, world, joint
        "loc": [None, tube_tray["aux"]],
        "tool": [suction_gripper["tool"], suction_gripper["tool"]],
        "output": suction_gripper["on"],
        "cmd": [],
        "approach": [[0, 0, -10, 0, 0, 0]],
        "exit": [[0, 0, -30, 0, 0, 0]]
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [None, well_plate["aux"]],
        "tool": [suction_gripper["tool_w_tube"], suction_gripper["tool_w_tube"]],
        "frame": well_plate["frame"],
        "output": suction_gripper["off"],
        "approach": [[0, 0, 50, 0, 0, 0]],
        "exit": [[20, 0, 0, 0, 0, 0], [20, 0, 50, 0, 0, 0]]
        },
    "middle": [
        {
        "type": "joint", # robot, world, joint
        "loc": [[-95.625, 27.597656, -92.790527, -98.349609, 92.768555, 113], tube_tray["aux"]],
        },
    ],
    "end": {
        "type": "joint", # robot, world, joint
        "loc": [[-100, 57.041016, -130.891113, 0, -19.511719, -12.634277] , [0, 0]],
        },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "jmove", "speed": speed, "cont": 1, "corner": 200, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}


well_plate_to_decapper= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, well_plate["aux"]],
        "tool": [two_finger_gripper["tool_wo_tube"], two_finger_gripper["tool_w_tube_cap"]],
        "frame": well_plate["frame"],
        "output": two_finger_gripper["close"],
        "cmd": [],
        "approach": [[0, 0, 40, 0, 0, 0]],
        "exit": [[0, 0, 50, 0, 0, 0]]
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [decapper["tube_holder"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_w_tube_cap"], two_finger_gripper["tool_w_tube_cap"]],
        "frame": decapper["frame"],
        "output": decapper["close"],
        "approach": [[0, 0, decapper["tube_holder_clearance"], 0, 0, 0]],
        "exit": [[0, 0, 3, 0, 0, 0]]
        },
    
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": 0.75*speed, "cont": 0, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}


well_plate_second_part= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["tube_holder_w_tube_cap"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_wo_tube"], two_finger_gripper["tool_wo_tube"]],
        "frame": decapper["frame"],
        "output": two_finger_gripper["close"],
        "cmd": [],
        "approach": [],
        "exit": [[0, 0, 20, 0, 0, 0]]
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}

release_cap= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["cap_holder"], decapper["aux"]],
        "tool": [two_finger_gripper["tool"], two_finger_gripper["tool"]],
        "frame": decapper["frame"],
        "output": two_finger_gripper["open"],
        "cmd": [],
        "approach": [[0, 0, 40, 0, 0, 0]],
        "exit": [[0, 0, 40, 0, 0, 0]]
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}

scan_tube= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["tube_holder_w_tube"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_wo_tube"], two_finger_gripper["tool_w_tube_cap"]],
        "frame": decapper["frame"],
        "output":decapper["open"]+ two_finger_gripper["close"],
        "approach": [[0, 0, 30, 0, 0, 0]],
        "exit": [[0, 0, 30, 0, 0, 0]]
        },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [decapper["scanner"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_w_tube_cap"], two_finger_gripper["tool_w_tube_cap"]],
        "frame": decapper["frame"],
        "output": [],
        "approach": [[0, 0, 80, 0, 0, 0]],
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
    "motion": "jmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}


tube_wo_cap_to_decapper = {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["scanner"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_w_tube_cap"], two_finger_gripper["tool_w_tube_cap"]],
        "output": two_finger_gripper["close"],
        "frame": decapper["frame"],
        "approach": [],
        "exit": [[0, 0, well_plate["length"][2]+5, 0, 0, 0]]
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [decapper["tube_holder"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_w_tube_cap"], two_finger_gripper["tool"]],
        "frame": decapper["frame"],
        "output": two_finger_gripper["open"] + decapper["close"],
        "approach": [[0, 0, decapper["tube_holder_clearance"], 0, 0, 0]],
        "exit": [[0, 0, decapper["tube_holder_clearance"], 0, 0, 0]]
        },
    
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,

}

capping= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [decapper["cap_holder"], decapper["aux"]],
        "tool": [two_finger_gripper["tool"], two_finger_gripper["tool"]],
        "frame": decapper["frame"],
        "output": two_finger_gripper["close"],
        "cmd": [],
        "approach": [[0, 0, 40, 0, 0, 0]],
        "exit": [[0, 0, 40, 0, 0, 0]]
    },
    "place": {
        "type": "world", # robot, world, joint
        "loc": [decapper["tube_holder_w_tube_cap"], decapper["aux"]],
        "tool": [two_finger_gripper["tool_wo_tube"], two_finger_gripper["tool_w_tube_cap"]],
        "frame": decapper["frame"],
        "output": decapper["open"],
        "cmd": [],
        "approach": [[0, 0, 40, 0, 0, 0]],
        "exit": [[0, 0, 20, 0, 0, 0]]
    },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
}

decapper_to_well_plate = {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, well_plate["aux"]],
        "tool": [two_finger_gripper["tool_w_tube_cap"], two_finger_gripper["tool"]],
        "frame": well_plate["frame"],
        "output": two_finger_gripper["open"],
        "cmd": [],
        "approach": [[0, 0, decapper["tube_holder_clearance"], 0, 0, 0]],
        "exit": [[0, 0, decapper["tube_holder_clearance"], 0, 0, 0]]
    },
    
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": speed, "cont": 1, "corner": 100, 
    "freedom": {"num":20, "range":[0.1, 0.1, 0.1], "early_exit":False }, "timeout": -1, "sim":sim,
 
}

tool_changer_connect= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, None],
        "tool": [tool_changer_gripper["tool"], tool_changer_gripper["tool"]],
        "frame": None,
        "output": tool_changer_gripper["connect"],
        "approach": [[0, 0, -40, 0, 0, 0]],
        "exit": [[0, 0, -4, 0, 0, 0], [-40, 0, -4, 0, 0, 0], [-40, 0, -60, 0, 0, 0]]
        },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": 0.2, "cont": 0, "corner": 100, 
    "freedom": {"num":20, "range":[0.01, 0.01, 0.01], "early_exit":False }, "timeout": -1, "sim":sim,
}

tool_changer_disconnect= {
    "pick": {
        "type": "world", # robot, world, joint
        "loc": [None, None],
        "tool": [tool_changer_gripper["tool"], tool_changer_gripper["tool"]],
        "frame": None,
        "output": tool_changer_gripper["disconnect"],
        "approach": [[-40, 0, -1, 0, 0, 0], [0, 0, -1, 0, 0, 0]],
        "exit": [[0, 0, -60, 0, 0, 0]]
        },
    "base_in_world": robot["base_in_world"],
    "aux_dir": robot["aux_dir"],
    "sleep": 0.5, 
    "motion": "lmove", "speed": 0.2, "cont": 0, "corner": 100, 
    "freedom": {"num":20, "range":[0.01, 0.01, 0.01], "early_exit":False }, "timeout": -1, "sim":sim,
}