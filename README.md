To run this project:
. Ensure cuda environment is present.
. To get the video output ensure ffmpeg is installed
. Create two new folders in the same folder as the file called "cpu_frames" and "gpu_frames".
. Comiple the .cu file with command "nvcc c_graph.cu -o c_gr" and run "./c_gr".
