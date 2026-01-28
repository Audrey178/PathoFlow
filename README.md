# PathoFlow: Clinical Video Classification

PathoFlow is a containerized web application designed for the automated classification of microscopic pathology videos. It provides a streamlined interface for physicians to upload video studies and receive immediate diagnostic predictions.

---

## Features
* **Simplified Workflow:** A minimalist interface optimized for clinical throughput.
* **Format Flexibility:** Native support for MP4 and automated server-side handling for AVI files.
* **Stable Inference:** GPU-accelerated backend using a synchronized execution model to prevent CUDA conflicts.

---

## Setting up Web Demo

### 1. Prerequisites
Ensure your host machine has the following installed:
* **Docker** and **Docker Compose**
* **NVIDIA Container Toolkit** (for GPU support)
* **NVIDIA Drivers**

### 2. Build and Run
Clone the repository:

```bash
git clone https://github.com/Audrey178/PathoFlow

cd pathoflow
```
Modify `HF_TOKEN` and `WANDB_TOKEN` inside `.env` file, then launch the system using Docker Compose:
```
docker compose up --build
```

### 3. Usage
Open your browser to http://localhost:8502.

Input: Upload a video study.

Output: The predicted class Normal, Adenoma or Malignant and the model's confidence score.