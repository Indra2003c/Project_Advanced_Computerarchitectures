import cv2
import os

#-----------SETUP VARIABLES:-----------
image_type = ".bmp"
video_name = "test_video.mp4" #should be changed to be the argument with which the script is called
input_folder = "test_content" #should be changed to "Input_content" later on
output_folder = "Output_frames_bmp"
video_path = "./" + input_folder + "/" + video_name #"./" indicates the directory the script is in
#--------------------------------------

video = cv2.VideoCapture(video_path)

try:
    if not os.path.exists(output_folder): #if the output_folder doesn't exist, we create it
        os.makedirs(output_folder)

except OSError:
    print("ERROR: The output directory could not be created.")

current_frame = 0

while(True):
    remaining,frame = video.read() #is the same as "remaining = video.read(), frame = video.read()"

    if remaining:
        frame_name = str(current_frame) + image_type
        output_directory = "./" + output_folder + "/" + frame_name

        cv2.imwrite(output_directory,frame) #writing away the created frame to the output directory

        current_frame += 1
    else:
        break

#cleanup
video.release()
cv2.destroyAllWindows()