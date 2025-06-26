# ⚽ Intelligent Football Player Tracking & Analytics with YOLOv11

Welcome to a cutting-edge AI project that leverages **YOLOv11**, **Optical Flow**, **KMeans Clustering**, and **Perspective Transformation** to deliver real-time insights and player analytics from football match videos. Designed with production-ready practices, this project pushes the boundary of sports intelligence by combining deep learning with classical computer vision.

---

## 🎯 Project Overview

- 🎥 Detect and track **players**, **referees**, and **the ball** with YOLOv11
- 🎨 Segment players into teams using **KMeans clustering on jersey colors**
- 📊 Analyze **team-wise ball possession percentage**
- 🏃 Measure **player movement**, **distance**, and **speed** accurately
- 🧠 Apply **Optical Flow** and **Perspective Transformation** for real-world motion analysis
- 🛠️ Built with a modular design and extendable for other team sports

---

## 🛠️ Core Technologies

| Tool / Library         | Purpose                                                  |
|------------------------|----------------------------------------------------------|
| **YOLOv11**            | State-of-the-art object detection (players, ball, etc.)  |
| **Ultralytics**        | Interface to run and integrate YOLOv11 models            |
| **OpenCV**             | Image processing, optical flow, and transformations      |
| **KMeans Clustering**  | Grouping players by t-shirt color (team identification)  |
| **Optical Flow**       | Frame-by-frame player movement tracking                  |
| **Perspective Warp**   | Convert movement in pixels to meters                     |
| **Supervision + DeepSORT** | Real-time multi-object tracking                     |
| **NumPy / Pandas**     | Fast computation and data manipulation                   |
| **Matplotlib**         | Graphical representation of player performance           |

---

## ✨ Key Features

✅ Real-time detection & tracking using YOLOv11  
✅ Jersey color clustering to identify team groups  
✅ Team-wise ball acquisition analysis  
✅ Accurate player movement vectors with Optical Flow  
✅ Perspective transformation for true distance calculation  
✅ Per-player speed and total distance covered in meters  
✅ Clean and modular pipeline with extendability in mind  
✅ Visualization-ready output for game analytics dashboards  

---

## 🎥 Sample Demo

<img src="assets/Screenshot 2025-06-26 235421" width="600"/>

▶️ [Watch Full Video Demo](https://drive.google.com/file/d/1wODZFIo4UDRWOJiEXdg11pSWjXkWJVAe/view?usp=sharing)

---

## 📦 Trained Models

- Custom-trained **YOLOv11** model on sports detection dataset
- Classes: `player`, `goalkeeper`, `referee`, `ball`

---

## 🔧 Installation & Requirements

To install all required dependencies:

```bash
pip install -r requirements.txt
```

### 🧰 Dependencies
- Python 3.8+
- `ultralytics`
- `opencv-python`
- `supervision`
- `scikit-learn`
- `numpy`, `pandas`, `matplotlib`

---

## 🚀 Running the Project

```bash
python main.py
```

All configuration and path settings can be modified in `config/config.yaml`.

---

## 📈 Example Output Metrics

| Player ID | Distance Covered (m) | Avg Speed (km/h) |
|-----------|----------------------|------------------|
| #7        | 934.7                | 13.2             |
| #12       | 863.2                | 11.5             |
| #4        | 1102.3               | 15.6             |

---

## 🤝 Contributing

Have ideas to improve analytics, add new metrics, or support more sports?

✅ Fork the repo  
✅ Create your branch  
✅ Submit a pull request with your enhancements

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.

---

## 👨‍💻 Author

**Keshav Prasad**  
🎓 Final Year CSE | 🧠 ML & MLOps Enthusiast  
🔗 [GitHub](https://github.com/keshav1017) | [LinkedIn](https://linkedin.com/in/keshavprasad1017)

---

⭐ **If you found this project helpful, consider giving it a star!**
