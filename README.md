## 📊 Datasets Overview

This project conducts experiments on multiple public datasets, covering both general image classification and industrial defect detection tasks. The datasets used include:

### 🖼️ Image Classification Datasets

| Dataset     | # Classes | Description |
|-------------|-----------|-------------|
| **CIFAR-10**  | 10        | Contains 60,000 color images of size 32×32, divided into 10 common object categories (e.g., airplane, automobile, bird). Widely used as a benchmark for image classification. |
| **CIFAR-100** | 100       | Similar to CIFAR-10 but with finer-grained 100 classes, each containing 600 images. |
| **ImageNet**  | 1,000     | A large-scale dataset with over 1.4 million training images across 1,000 object categories. It serves as the standard benchmark for modern deep learning models (e.g., ResNet, ViT). |

---

### 🔍 Industrial Defect Detection Datasets (Object Detection)

The following four datasets focus on surface defect detection in industrial scenarios. Their original annotations are in formats such as Pascal VOC or custom JSON. To ensure compatibility with YOLO-family detectors, we converted all of them into **YOLO format**, where each image has a corresponding `.txt` label file with lines formatted as:  
`class_id center_x center_y width height`  
(all coordinates normalized to [0, 1]).

| Dataset       | # Classes | Description | Data Processing |
|---------------|-----------|-------------|-----------------|
| **NEU-DET**   | 6         | A hot-rolled steel strip surface defect dataset from Northeastern University, containing six typical defect types (e.g., crack, patch, scratch), with 1,800 images in total. | Original annotations in Pascal VOC XML format were converted to YOLO format. The dataset was split into train/val/test sets in a 7:2:1 ratio. |
| **GC10-DET**  | 10        | A steel surface defect dataset with ten industrial defect categories. Features high-resolution images and complex backgrounds. | Custom JSON annotations were parsed and converted to YOLO format. Small objects were enhanced to improve detection performance. |
| **XSDD-DET**  | 5         | An X-ray weld defect detection dataset for non-destructive testing, containing five defect types (e.g., porosity, slag inclusion). | Original COCO-style JSON annotations were converted to YOLO format using a `coco2yolo` script, preserving aspect ratios during normalization. |
| **DSPCB-SD+** | 8         | An extended printed circuit board (PCB) surface defect dataset, including eight manufacturing defect types (e.g., short circuit, open circuit, burr). | Original Pascal VOC annotations were cleaned, deduplicated, and converted to YOLO format. Images with ambiguous labels were removed. |

> ✅ All object detection datasets follow the standard YOLOv5/YOLOv8 directory structure:
> ```
> dataset/
> ├── images/
> │   ├── train/
> │   ├── val/
> │   └── test/
> └── labels/
>     ├── train/
>     ├── val/
>     └── test/
> ```

### 📥 Dataset Download

You can download the preprocessed YOLO-format datasets from the links below:

- **CIFAR-10 / CIFAR-100**: Automatically downloaded via `torchvision.datasets`.
- **ImageNet**: Available at [https://image-net.org](https://image-net.org) (registration required).
- **NEU-DET**: [https://github.com/abin24/NEU-DET-dataset](https://github.com/abin24/NEU-DET-dataset)
- **GC10-DET**: [https://www.kaggle.com/datasets/overload10/gc10det](https://www.kaggle.com/datasets/overload10/gc10det)
- **XSDD-DET**: [https://github.com/Charmve/XSDD-Dataset](https://github.com/Charmve/XSDD-Dataset)
- **DSPCB-SD+**: [https://github.com/Charmve/DSPCB-SD-plus](https://github.com/Charmve/DSPCB-SD-plus)

> 💡 We provide our processed YOLO-format versions in the [`datasets/`](./datasets) folder. If not included, please run the conversion scripts in [`tools/convert_datasets.py`](./tools/convert_datasets.py).

---

## ⚙️ Environment Setup

Ensure you have the following environment:

- **Python**: `3.10`
- **PyTorch**: `1.12.0`

You can create a virtual environment and install dependencies as follows:

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt  # if you have additional dependencies
