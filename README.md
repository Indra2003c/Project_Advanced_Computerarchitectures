# Project Advanced Computerarchitecture: CUDA object trajectory

In this project, we chose to track the path of an object in a video. We started with a basic implementation in Python and C++, after which we switched to CUDA to perform operations on the GPU. Finally, we applied several techniques (e.g., double reduction) and compared them with one another.


The problem we address is tracking the path of an object in a video. The input is a video clip in which a round object moves. We chose to generate this input video ourselves so that there would be clear color contrasts between the object and the background. In a real-world application, however, this could be a recorded video—for example, of a billiards game, a bowling ball, or a tennis match. The goal is to output a line that shows the path the object has traveled. This line is projected onto the first frame of the video to make the visualization more clear and organized.


This application was implemented by analyzing the RGB values of the pixels in each video frame. Pixels whose color falls within a predefined range—specifically set for the object to be tracked—are considered "hits." After identifying all rows and columns containing such pixels, the center of this region is calculated. The center is determined by taking the average of all column indices and row indices where the target color appears. This center point is then stored as the object's position in that specific frame. This procedure is repeated for all frames in the video to reconstruct the complete trajectory of the object. Finally, all collected coordinates are visualized to clearly show the object's path. 


Example input:
https://github.com/user-attachments/assets/3c146bee-575a-49f4-b70c-8980d40c984d


Example output:
![image](https://github.com/user-attachments/assets/4a9925b3-f92d-467d-9d1e-ce11959c157d)
