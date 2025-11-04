# 🦾 Smart Glasses Capstone: Vision Transformer–Based Object Detection for Low-Vision Assistance  
_By Irene Liu, Faransina Olivia Rumere & Mustapha Mbengue_  
_University of Chicago – MSc in Applied Data Science Capstone (2025)_

---

## 🎯 Project Overview  
This project presents a **proof-of-concept smart-glasses system** that uses a **Vision Transformer (ViT)**–based object detection model to assist individuals with low vision in perceiving their physical surroundings.  

The system:
- Converts LVIS annotations into YOLO format for training.  
- Fine-tunes a **custom ViT object detection architecture** (e.g., ViTDet / DETR) for long-tail, low-data object categories.  
- Benchmarks against a **YOLOv8n CNN baseline**.  
- Deploys the model on **edge hardware** (smart-glasses prototype).  
- Evaluates using **mAP**, **IoU**, **latency (FPS)**, and **attention-map interpretability**.  

---

## 📁 Repository Structure 

/Code
│
├── data_preparation/ # Scripts: LVIS → YOLO conversion, annotation cleaning
├── models/ # ViT-based and YOLOv8 baseline model definitions
├── training/ # Training pipelines, configs, hyperparameters
├── evaluation/ # Evaluation scripts for mAP/IoU/FPS + attention visualization
├── deployment/ # Quantization, pruning, and on-device demo scripts
└── notebooks/ # EDA, baseline setup, ViT training walkthrough


---

## ✅ Key Results  
| Metric | YOLOv8n (Baseline) | Custom ViT Model |
|--------|-------------------|-----------------|
| Training Time | ~1d 18h | ~2h 43m |
| mAP50–95 (val) | 0.0615 | **0.33** |
| Inference Latency | 6 ms | **7 ms** |
| Edge Feasibility | ✅ Yes | ✅ Yes |
| Notable Strength | Fast real-time CNN | Accurate rare-object detection |

> The ViT model achieved **5× higher accuracy** on rare object categories while maintaining real-time performance (< 100 ms per frame).  
> Attention heatmaps confirmed explainable behavior — a major advantage for assistive AI use.

---

## 🔍 Why Vision Transformers & Smart Glasses?  
**Technical edge:**  
ViTs use attention to capture global context, improving performance on *rare* or *unfamiliar* objects compared to CNNs.  

**Edge deployment:**  
Quantization and token pruning make ViTs feasible for low-power devices like smart glasses powered by ESP32 or Snapdragon XR2.  

**Social impact:**  
Over **2.2 billion** people live with some form of vision impairment. ViT-powered assistive AI represents a step toward **independence, safety, and accessibility**.  

**Business potential:**  
- Growing **assistive-tech market** ($30 B+ by 2030)  
- Cross-industry applications: healthcare, AR/VR, retail, manufacturing  
- Aligned with sustainability and privacy trends through **edge AI inference**  

---

## 🧠 Ethical & Interpretability Considerations  
- **Transparency:** Attention-map visualization exposes *where* and *why* the model focuses — increasing auditability.  
- **Privacy:** On-device inference keeps sensitive visual data local.  
- **Fairness:** Long-tail datasets and interpretability reduce bias toward overrepresented object classes.  
- **Human-centric AI:** Designed to **augment**, not replace, human perception.  

---

## 🔮 Future Directions  
- 🗣 **Multimodal AI** — combine vision + language for conversational assistance (“I see a staircase ahead”).  
- 🧩 **Continual learning** — adapt detection to personal environments (home, office, daily routine).  
- 🥽 **AR integration** — link detection outputs with visual/audio overlays for immersive assistive experience.  
- 🤝 **Industry collaboration** — integrate with hardware OEMs (NVIDIA Jetson, Qualcomm Snapdragon) and healthcare/AR ecosystems.

---

## 🛠 Getting Started  

### 1. Clone this repo  
```bash
git clone https://github.com/irenegliu/Smart-Glass-Capstone.git
cd Smart-Glass-Capstone/Code
2. Install dependencies
pip install -r requirements.txt
3. Prepare dataset
Convert LVIS annotations to YOLO format:
python data_preparation/convert_lvis_to_yolo.py
4. Train baseline YOLOv8
python training/train_yolov8_baseline.py
5. Train Vision Transformer model
python training/train_vit_detector.py
6. Evaluate models
python evaluation/evaluate_models.py
Generates mAP, IoU, FPS metrics and visual attention maps.
7. Deploy to edge device
python deployment/run_on_device.py
Runs quantized model on ESP32-S3 or similar smart-glasses hardware.

## 📄 Citation

If you use this work, please cite:

Liu, I., Rumere, F. O., & Mbengue, M. (2025). Image Detection Using Vision Transformers (ViT) for Smart Glasses: A Long-Tail, Edge-Deployment Benchmark.
University of Chicago, Master of Science in Applied Data Science.

## 🤝 Acknowledgements

Special thanks to:

Dr. Ming Long Lam (Faculty Supervisor)

The University of Chicago Data Science faculty

Open-source communities: Ultralytics YOLOv8, PyTorch, TIMM, and LVIS Dataset Team

## 🧩 License

This project is released under the MIT License.
See the LICENSE
 file for details.
