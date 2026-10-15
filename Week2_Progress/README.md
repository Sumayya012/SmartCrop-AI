# 🌿 SmartCropAI - Week 2 Progress (60%)

## 📘 Summary: Model Training & Performance
Week 2 focused on building and training the base deep learning model using **Transfer Learning with MobileNetV2**, pre-trained on **ImageNet**.  
The model was fine-tuned for plant leaf disease classification and achieved **benchmark-level accuracy** on the 15-class PlantVillage dataset.

---

### 🚀 Key Achievement
Trained for **10 epochs**, achieving **~89.0 % Validation Accuracy** on 15 PlantVillage leaf disease classes — a strong baseline for fine-tuning.

---

### ✅ Tasks Completed
* Preprocessed and augmented dataset using real-time data generators.  
* Implemented **MobileNetV2** (frozen convolutional base) for transfer learning.  
* Trained model with **Adam optimizer** and monitored validation accuracy.  
* Saved best-performing model weights and generated **class-label mapping (`class_map.json`)**.  
* Documented the entire workflow and results in a Colab notebook.

---

### 💾 Deliverables
| File/Link | Description |
|-----------|-------------|
| `SmartCropAI.ipynb` | End-to-end notebook for preprocessing, model definition, and training logs. |
| `class_map.json` | Label-to-class index mapping used during inference. |
| **Model Weights** | Trained model file **(SmartCropAI_BestModel.h5)** — [➡️ Download from Google Drive](https://drive.google.com/). |

---

### 🎯 Next Steps (Week 3)
* **Fine-tuning:** Unfreeze and train final MobileNetV2 layers for improved accuracy.  
* **Evaluation:** Evaluate on an unseen test set and generate a confusion matrix.  
* **Deployment:** Develop a web interface for real-time leaf disease prediction.

---

> **Current Project Progress:** **60 % Complete** — Model successfully trained and validated.
