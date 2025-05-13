import math
from PIL import Image
import numpy as np

#------COLOR BOUNDS:------
R_lower = 230
R_upper = 255

G_lower = 0
G_upper = 15

B_lower = 0
B_upper = 15
#-------------------------

def in_range(pixel):
    R_value = pixel[0]
    G_value = pixel[1]
    B_value = pixel[2]
    if (R_lower <= R_value <= R_upper) and (G_lower <= G_value <= G_upper) and (B_lower <= B_value <= B_upper):
        return True
    else:
        return False

for frameNr in range(150):

    # ---------BOUNDS:---------
    Row_lower_bound = -1
    Row_upper_bound = 0
    Column_lower_bound = -1
    Column_upper_bound = 0
    # -------------------------

    path = "./../frames/moving_ball_frames_png/" + str(frameNr) + ".png"
    # Laad de afbeelding
    frame = Image.open(path).convert("RGB")

    # Converteer naar een 3D NumPy-array (hoogte, breedte, 3 kleuren)
    rgb_values = np.array(frame)

    # Print de pixel op rij 10, kolom 15
    #print(rgb_values[0, 26])  # Output: (R, G, B) #720, 1280

    for i in range(720):
        for j in range(1280):
            if in_range(rgb_values[i][j]):
                if i > Row_upper_bound:
                    Row_upper_bound = i
                elif i < Row_lower_bound or Row_lower_bound == -1:
                    Row_lower_bound = i

                if j > Column_upper_bound:
                    Column_upper_bound = j
                elif j < Column_lower_bound or Column_lower_bound == -1:
                    Column_lower_bound = j
                #print("rij: " + str(i) + ", kolom: " + str(j))

    Middle_row = math.ceil(((Row_upper_bound - Row_lower_bound)/2) + Row_lower_bound)
    Middle_column = math.ceil(((Column_upper_bound - Column_lower_bound)/2) + Column_lower_bound)

    print("Rij: " + str(Middle_row) + ", Kolom: " + str(Middle_column))
    with open("./../output/coordinates.txt", "a") as bestand:
        bestand.write(str(Middle_row) + " " + str(Middle_column) + "\n")
