# Age and Gender Detection using Deep Learning

## Objective
The main goal of this project is to build a deep learning based system which takes a user's face image and predicts their age and gender.The project follows the similar approach as any ML project,performing data-collection,feature-engineering,training and then deployment.Some of its use-cases include online document verification,facial recognition systems,social credential verification etc.

<div align="center"> 
<Img src="/Imgs/faces2.png" width="65%"/>
</div>

## Project Workflow
The project will follow the same approach as used in all ML project. We'll go through different stages of data collection,feature extraction,training and finally deployment of trained model.

```python
Data Collection --> Feature Extraction --> Training --> Deployment
```


## Data Collection

For this project we would need a bunch of facial images of people from different age groups. For this we used the following dataset which is based on UTKFace dataset. It consist of 21000 Images of people ranging in the age of (1-116) and also provides respective features like "age","gender" and "ethnicity". All the images are grayscale single channel images and have a fixed shape of (48x48).


Dataset link : [click here](https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

(See the entire data-collection notebook [here](https://github.com/deepeshdm/Age-Gender-Detection/blob/main/Colab%20Notebooks/Data_Collection_%26_Processing.ipynb))


## Model Training

The problem of predicting age and gender given a face image is divided into 2 seperate problems wherein 2 different models have been trained for each problem. The first problem is predicting the gender of the person,which falls under 'binary classification',so a binary classifier model was trained for this task. The second problem is predicting the age of the person,this problem is treated as an 'regression' problem and a different model have been trained for this task.

The Binary classifier gives an validation-accuracy of 90% , whereas the Regressor gives a MAE of less than 90. The models were saved as '.h5' files. You can find the trained models in the ['models'](https://github.com/deepeshdm/Age-Gender-Detection/tree/main/models) directory.

(See the entire model training notebook [here](https://github.com/deepeshdm/Age-Gender-Detection/blob/main/Colab%20Notebooks/Training_Age_Gender_Detection_model.ipynb))


## Web Interface & API










