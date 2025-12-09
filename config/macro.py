import numpy as np

CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
CAD_LINE_IDX = CAD_COMMANDS.index('Line')
CAD_ARC_IDX = CAD_COMMANDS.index('Arc')
CAD_CIRCLE_IDX = CAD_COMMANDS.index('Circle')
CAD_EOS_IDX = CAD_COMMANDS.index('EOS')
CAD_SOL_IDX = CAD_COMMANDS.index('SOL')
CAD_EXT_IDX = CAD_COMMANDS.index('Ext')

CAD_EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]
CAD_EXTENT_TYPE = ["OneSideFeatureExtentType", "SymmetricFeatureExtentType",
               "TwoSidesFeatureExtentType"]

SVG_COMMANDS = ['SOS', 'EOS', 'L', 'C']
SVG_SOS_IDX = SVG_COMMANDS.index('SOS')
SVG_EOS_IDX = SVG_COMMANDS.index('EOS')
SVG_L_IDX = SVG_COMMANDS.index('L')
SVG_C_IDX = SVG_COMMANDS.index('C')

PAD_VAL = -1
CAD_N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
CAD_N_ARGS_PLANE = 3 # sketch plane orientation: theta, phi, gamma
CAD_N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
CAD_N_ARGS_EXT_PARAM = 4 # extrusion parameters: e1, e2, b, u
CAD_N_ARGS_EXT = CAD_N_ARGS_PLANE + CAD_N_ARGS_TRANS + CAD_N_ARGS_EXT_PARAM
CAD_N_ARGS = CAD_N_ARGS_SKETCH + CAD_N_ARGS_EXT

SVG_N_ARGS_LINE = 4 # start=(x1, y1), end=(x2, y2)
SVG_N_ARGS_BEZIERCURVE = 8 # start=(x1, y1), control1=(cx1,cy1), control2=(cx2,cy2), end=(x2, y2)
SVG_N_ARGS = 8 # shared parameters: start=(x1, y1), end=(x2, y2)

CAD_SOL_VEC = np.array([CAD_SOL_IDX, *([PAD_VAL] * CAD_N_ARGS)])
CAD_EOS_VEC = np.array([CAD_EOS_IDX, *([PAD_VAL] * CAD_N_ARGS)])

SVG_SOS_VEC = np.array([SVG_SOS_IDX, *([PAD_VAL] * SVG_N_ARGS)])
SVG_EOS_VEC = np.array([SVG_EOS_IDX, *([PAD_VAL] * SVG_N_ARGS)])

CAD_CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # line
                          [1, 1, 1, 1, 0, *[0]*CAD_N_ARGS_EXT],  # arc
                          [1, 1, 0, 0, 1, *[0]*CAD_N_ARGS_EXT],  # circle
                          [0, 0, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # EOS
                          [0, 0, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # SOL
                          [*[0]*CAD_N_ARGS_SKETCH, *[1]*CAD_N_ARGS_EXT]]) # Extrude

SVG_CMD_ARGS_MASK = np.array([[*[0]*SVG_N_ARGS], # SOS
                             [*[0]*SVG_N_ARGS], # EOS
                             [1, 1, 0, 0, 0, 0, 1, 1], # L
                             [*[1]*SVG_N_ARGS]]) # C

CAD_NORM_FACTOR = 0.75 # scale factor for normalization to prevent overflow during augmentation

CAD_MAX_N_EXT = 10 # maximum number of extrusion
CAD_MAX_N_LOOPS = 6 # maximum number of loops per sketch
CAD_MAX_N_CURVES = 15 # maximum number of curves per loop

CAD_MAX_TOTAL_LEN = 60 # maximum cad sequence length
SVG_MAX_TOTAL_LEN = 100 # maximum svg sequence length
ARGS_DIM = 256