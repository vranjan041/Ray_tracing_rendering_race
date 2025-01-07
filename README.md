# Project Name

## About the Project
This project is designed to utilize CUDA for parallel processing and generate video outputs using FFmpeg. It demonstrates efficient computation with GPU acceleration.

---

## Requirements

1. Ensure a CUDA environment is properly set up on your system.
2. Install FFmpeg for video output processing.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository
   ```
3. Verify that CUDA and FFmpeg are installed:
   ```bash
   nvcc --version
   ffmpeg -version
   ```

---

## Setup

1. Create two new folders in the project directory:
   ```bash
   mkdir cpu_frames
   mkdir gpu_frames
   ```

---

## Compilation

1. Compile the CUDA source file:
   ```bash
   nvcc c_graph.cu -o c_gr
   ```
2. Run the compiled program:
   ```bash
   ./c_gr
   ```

---

## Usage
Ensure the required folders are created before running the program. The output frames will be stored in the respective folders (`cpu_frames` and `gpu_frames`).

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## Contact
For issues or suggestions, feel free to open an issue or reach out via email.

