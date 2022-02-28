# dl4aed-ws2122-p5: Detection of coughs in classical music recordings

The process of cutting and replacing coughing sounds in recordings of unamplified music performances is a tedious job for audio engineers. This paper describes the development of a deep learning model with the goal of automatically detecting and localizing the coughing sounds in concert recordings. The model was build with convolutional neural network layers and trained with self created training data that was autoencoded with the vggish autoencoder. Testing of the model on artificial and real data showed that a big challenge of developing an accurately predicting model is the creation of decent training datasets.

## Instructions

1. Get the required data from [here.](https://tubcloud.tu-berlin.de/s/dgY8aCHi7RBtydL)

2. Put it inside the folder of this repository.

3. Run the first three cells of the  Jupyter Notebook ``recoughnition.ipynb``.

