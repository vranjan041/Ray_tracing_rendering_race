// Include headers
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using namespace std;

// Constants
const int WIDTH = 1920;
const int HEIGHT = 1080;
const float AMBIENT_LIGHT = 0.1f;
const float LIGHT_INTENSITY = 1.0f;
const float CAMERA_Z = -15.0f;

struct Sphere
{
    float x, y, z, radius;
    int r, g, b;
};

// Utility functions
float degToRad(float degrees)
{
    return degrees * M_PI / 180.0f;
}

vector<Sphere> generatePlanets()
{
    srand(42);
    return {
        {0.0f, 0.0f, 5.0f, 5.0f, 255, 232, 124}, // Sun
        {7.0f, 0.05f, 0.2f, 255, 200, 100},      // Mercury
        {8.0f, 0.03f, 0.5f, 200, 200, 255},      // Venus
        {9.0f, 0.02f, 0.7f, 100, 200, 255},      // Earth
        {10.0f, 0.015f, 0.6f, 255, 100, 100},    // Mars
        {13.0f, 0.01f, 1.2f, 255, 255, 200},     // Jupiter
        {15.0f, 0.008f, 1.0f, 200, 200, 255},    // Saturn
        {17.0f, 0.006f, 0.9f, 100, 255, 255},    // Uranus
        {19.0f, 0.004f, 0.8f, 100, 200, 255}     // Neptune
    };
}

vector<Sphere> calculatePositions(const vector<Sphere> &planets, int frame)
{
    vector<Sphere> positions;
    for (const auto &planet : planets)
    {
        float angle = planet.y * frame * 2 * M_PI;
        float x = planet.x * cos(angle);
        float y = planet.x * sin(angle);
        positions.push_back({x, y, 25.0f, planet.z, planet.r, planet.g, planet.b});
    }
    return positions;
}

void rayTraceCPU(const vector<Sphere> &spheres, const float *light, const string &outputPath)
{
    vector<vector<vector<int>>> image(HEIGHT, vector<vector<int>>(WIDTH, vector<int>(3, 0)));
    float ro[3] = {0, 0, CAMERA_Z};

    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            float rd[3] = {
                (x - WIDTH / 2.0f) / WIDTH,
                (y - HEIGHT / 2.0f) / HEIGHT,
                1.0f};
            float length = sqrt(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]);
            rd[0] /= length;
            rd[1] /= length;
            rd[2] /= length;

            for (const auto &sphere : spheres)
            {
                float oc[3] = {ro[0] - sphere.x, ro[1] - sphere.y, ro[2] - sphere.z};
                float b = oc[0] * rd[0] + oc[1] * rd[1] + oc[2] * rd[2];
                float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere.radius * sphere.radius;
                float discriminant = b * b - c;

                if (discriminant > 0)
                {
                    float t = -b - sqrt(discriminant);
                    if (t > 0)
                    {
                        float hit[3] = {ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t};
                        float normal[3] = {(hit[0] - sphere.x) / sphere.radius, (hit[1] - sphere.y) / sphere.radius, (hit[2] - sphere.z) / sphere.radius};

                        float lightDir[3] = {light[0] - hit[0], light[1] - hit[1], light[2] - hit[2]};
                        length = sqrt(lightDir[0] * lightDir[0] + lightDir[1] * lightDir[1] + lightDir[2] * lightDir[2]);
                        lightDir[0] /= length;
                        lightDir[1] /= length;
                        lightDir[2] /= length;

                        float diffuse = max(0.0f, normal[0] * lightDir[0] + normal[1] * lightDir[1] + normal[2] * lightDir[2]);
                        float lightIntensity = AMBIENT_LIGHT + diffuse * LIGHT_INTENSITY;

                        image[y][x][0] = min(255, (int)(sphere.r * lightIntensity));
                        image[y][x][1] = min(255, (int)(sphere.g * lightIntensity));
                        image[y][x][2] = min(255, (int)(sphere.b * lightIntensity));
                    }
                }
            }
        }
    }

    ofstream ofs(outputPath, ios::binary);
    ofs << "P6\n"
        << WIDTH << " " << HEIGHT << "\n255\n";
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            ofs << (char)image[y][x][0] << (char)image[y][x][1] << (char)image[y][x][2];
        }
    }
    ofs.close();
}

#ifdef __CUDACC__
__global__ void rayTraceGPUKernel(unsigned char *image, Sphere *spheres, int sphereCount, float *light, float ambientLight, float lightIntensity, float cameraZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    int idx = (y * WIDTH + x) * 3;
    float ro[3] = {0, 0, cameraZ};
    float rd[3] = {
        (x - WIDTH / 2.0f) / WIDTH,
        (y - HEIGHT / 2.0f) / HEIGHT,
        1.0f};
    float length = sqrt(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]);
    rd[0] /= length;
    rd[1] /= length;
    rd[2] /= length;

    for (int i = 0; i < sphereCount; ++i)
    {
        Sphere sphere = spheres[i];
        float oc[3] = {ro[0] - sphere.x, ro[1] - sphere.y, ro[2] - sphere.z};
        float b = oc[0] * rd[0] + oc[1] * rd[1] + oc[2] * rd[2];
        float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere.radius * sphere.radius;
        float discriminant = b * b - c;

        if (discriminant > 0)
        {
            float t = -b - sqrt(discriminant);
            if (t > 0)
            {
                float hit[3] = {ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t};
                float normal[3] = {(hit[0] - sphere.x) / sphere.radius, (hit[1] - sphere.y) / sphere.radius, (hit[2] - sphere.z) / sphere.radius};

                float lightDir[3] = {light[0] - hit[0], light[1] - hit[1], light[2] - hit[2]};
                length = sqrt(lightDir[0] * lightDir[0] + lightDir[1] * lightDir[1] + lightDir[2] * lightDir[2]);
                lightDir[0] /= length;
                lightDir[1] /= length;
                lightDir[2] /= length;

                float diffuse = fmaxf(0.0f, normal[0] * lightDir[0] + normal[1] * lightDir[1] + normal[2] * lightDir[2]);
                float lightIntensityFinal = ambientLight + diffuse * lightIntensity;

                image[idx] = min(255, (int)(sphere.r * lightIntensityFinal));
                image[idx + 1] = min(255, (int)(sphere.g * lightIntensityFinal));
                image[idx + 2] = min(255, (int)(sphere.b * lightIntensityFinal));
            }
        }
    }
}

void rayTraceGPU(const vector<Sphere> &spheres, const float *light, const string &outputPath)
{
    unsigned char *d_image;
    Sphere *d_spheres;
    float *d_light;

    size_t imageSize = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    size_t sphereSize = spheres.size() * sizeof(Sphere);
    size_t lightSize = 3 * sizeof(float);

    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_spheres, sphereSize);
    cudaMalloc(&d_light, lightSize);

    cudaMemcpy(d_spheres, spheres.data(), sphereSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_light, light, lightSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    rayTraceGPUKernel<<<grid, block>>>(d_image, d_spheres, spheres.size(), d_light, AMBIENT_LIGHT, LIGHT_INTENSITY, CAMERA_Z);

    vector<unsigned char> image(WIDTH * HEIGHT * 3);
    cudaMemcpy(image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_spheres);
    cudaFree(d_light);

    ofstream ofs(outputPath, ios::binary);
    ofs << "P6\n"
        << WIDTH << " " << HEIGHT << "\n255\n";
    ofs.write((char *)image.data(), imageSize);
    ofs.close();
}
#endif

void renderCPU(int &frameCount, double &fps, double &raysPerSecond)
{
    auto planets = generatePlanets();
    int frame = 0;
    auto start = chrono::high_resolution_clock::now();

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start).count() < 5)
    {
        auto positions = calculatePositions(planets, frame);
        rayTraceCPU(positions, new float[3]{0, 0, 0}, "./cpu_frames/cpu_frame_" + to_string(frame) + ".ppm");
        frame++;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    frameCount = frame;
    fps = frame / elapsed.count();
    raysPerSecond = (frame * WIDTH * HEIGHT) / elapsed.count();

    cout << "CPU rendering complete." << endl;
    cout << "Frames generated: " << frameCount << endl;
    cout << "FPS: " << fps << endl;
    cout << "Rays calculated per second: " << raysPerSecond << endl;
}

#ifdef __CUDACC__
void renderGPU(int &frameCount, double &fps, double &raysPerSecond)
{
    auto planets = generatePlanets();
    int frame = 0;
    auto start = chrono::high_resolution_clock::now();

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start).count() < 5)
    {
        auto positions = calculatePositions(planets, frame);
        rayTraceGPU(positions, new float[3]{0, 0, 0}, "./gpu_frames/gpu_frame_" + to_string(frame) + ".ppm");
        frame++;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    frameCount = frame;
    fps = frame / elapsed.count();
    raysPerSecond = (frame * WIDTH * HEIGHT) / elapsed.count();

    cout << "GPU rendering complete." << endl;
    cout << "Frames generated: " << frameCount << endl;
    cout << "FPS: " << fps << endl;
    cout << "Rays calculated per second: " << raysPerSecond << endl;
}
#endif

void createVideoFromFrames(const string &framesPath, const string &outputVideoPath, const string &framesName)
{
    string command = "ffmpeg -framerate 60 -i " + framesPath + "/" + framesName + "_%d.ppm -c:v libx264 -pix_fmt yuv420p " + outputVideoPath + " -y";
    system(command.c_str());
}

void plotMetrics(int cpuFrameCount, double cpuFPS, double cpuRaysPerSecond, int gpuFrameCount, double gpuFPS, double gpuRaysPerSecond)
{
    // Python script to plot the metrics
    string pythonScript = R"(
import matplotlib.pyplot as plt
import numpy as np

cpu_fps = )" + to_string(cpuFPS) +
                          R"(;
cpu_rays_per_second = )" + to_string(cpuRaysPerSecond) +
                          R"(;
gpu_fps = )" + to_string(gpuFPS) +
                          R"(;
gpu_rays_per_second = )" + to_string(gpuRaysPerSecond) +
                          R"(;

labels = ['CPU', 'GPU']
fps = [cpu_fps, gpu_fps]
rays_per_second = [cpu_rays_per_second, gpu_rays_per_second]

# Plot FPS
plt.figure(1)
plt.bar(labels, fps, color=['blue', 'green'])
plt.ylabel('FPS')
plt.title('Frames Per Second (FPS) Comparison')
plt.show()

# Plot Rays Per Second
plt.figure(2)
plt.bar(labels, rays_per_second, color=['blue', 'green'])
plt.ylabel('Rays Per Second')
plt.title('Rays Per Second Comparison')
plt.show()
)";

    // Write the Python script to a file
    ofstream pythonFile("plot_metrics.py");
    pythonFile << pythonScript;
    pythonFile.close();

    // Execute the Python script
    system("python plot_metrics.py");
}

int main()
{
    int cpuFrameCount = 0;
    double cpuFPS = 0.0;
    double cpuRaysPerSecond = 0.0;
    int gpuFrameCount = 0;
    double gpuFPS = 0.0;
    double gpuRaysPerSecond = 0.0;

    thread cpuThread([&]()
                     { renderCPU(cpuFrameCount, cpuFPS, cpuRaysPerSecond); });

#ifdef __CUDACC__
    thread gpuThread([&]()
                     { renderGPU(gpuFrameCount, gpuFPS, gpuRaysPerSecond); });
    cpuThread.join();
    gpuThread.join();
#else
    cpuThread.join();
#endif

    createVideoFromFrames("./cpu_frames", "./cpu_output_video.mp4", "cpu_frame");
    createVideoFromFrames("./gpu_frames", "./gpu_output_video.mp4", "gpu_frame");

    plotMetrics(cpuFrameCount, cpuFPS, cpuRaysPerSecond, gpuFrameCount, gpuFPS, gpuRaysPerSecond);

    return 0;
}