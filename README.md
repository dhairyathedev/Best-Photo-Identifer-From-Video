# Best Photo Identifier From Video
---

### Introduction
This project deals with the problem of getting the best frame of a video. It uses advance machine learning models and neural neural networks to identify the best frame from a video. The project is implemented in Python and uses OpenCV, Keras, and TensorFlow libraries.

### Requirements
- minimum 8GB RAM
- Python 3.6 or higher
- GPU

### Installation
```bash
pip install -r requirements.txt
```
### Training model
 Open `source_code/AI/Attractive_Model_Notebook.ipynb` run all the cells to train the Attractivenesss Detection Model.

Similarly Open `source_code/AI/Emotion_Model_Notebook.ipynb` run all the cells to train the Emotion Detection Model.

### Usage
Open `source_code/Web/model.py` and update the required model placeholders. Then run the following command to start the server.
```bash
python model.py
```

This will boot up a gradio application. You can upload a video and get the best frame from the video.


### Dataset Used
- [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

### Contributors
- Sneh Shah (Model Training and Preprocessing)
- Pankil Soni (Mode Training and Preprocessing)
- Dhairya Shah (Model Training and Deployment)