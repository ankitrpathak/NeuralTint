# [NeuralTint - Restore Colors Instantly](https://neuraltint.streamlit.app/) 🎨

NeuralTint is an AI-powered tool that converts grayscale images into vibrant, colorized versions. This project utilizes OpenCV's deep learning module with a pre-trained Caffe model to restore colors effortlessly.

## Features 🚀

- User have to upload a grayscale image (JPG, PNG, JPEG)
- AI-based automatic colorization
- Side-by-side grayscale vs. colorized preview
- Download the colorized image with one click
- Simple, user-friendly Streamlit interface

## Installation & Setup 🛠️

### Prerequisites

Ensure you have Python installed (>=3.7) and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/ankitrpathak/NeuralTint.git
cd NeuralTint
```

## Running the Streamlit App ▶️

```bash
streamlit run Blackandwhite.py
```

## Deployment 🌍

To deploy on **Streamlit Community Cloud**:

1. Push this project to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in.
3. Click **Deploy an app** → Select your GitHub repo.
4. Set `Blackandwhite.py` as the main entry point and deploy.

## Project Structure 📂

```
NeuralTint/
│── Blackandwhite.py               # Main Streamlit app
│── colorization_deploy_v2.prototxt # Model configuration
│── colorization_release_v2.caffemodel # Pre-trained model
│── pts_in_hull.npy                # Cluster centers for model
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
```

## Required Additional Files 📌

Make sure to include:
✅ `requirements.txt` (for dependencies)
✅ `.gitignore` (to exclude unnecessary files, e.g., cache)
✅ `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, `pts_in_hull.npy` (model files)

## License 📜

This project is open-source under the MIT License.

---

### 🔗 Connect with Me

If you have any questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/ankitrpathak) or contribute to this repository!
