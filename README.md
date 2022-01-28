# Deep Java Library MNIST CNN Tutorial
A Deep Java Library MNIST CNN example for AI Tech Education
## Classification
### Setup
1. Install JDK 11: `sudo apt-get install openjdk-11-jdk`
2. Clone DJL repository: `git clone https://github.com/deepjavalibrary/djl`
3. Download and copy `TrainMnistCNN.java` from repository: `https://github.com/davor10105/djl_aitech` to ai.djl.examples.training and the inference file `ImageClassificationCNN.java` to ai.djl.examples.inference
4. Move to examples directory `djl/examples`
5. Run the training: `./gradlew run -Dmain=ai.djl.examples.training.TrainMnistCNN`
6. Run the inference: `./gradlew run -Dmain=ai.djl.examples.training.ImageClassificationCNN`

## Detection
1. Download and copy `TrainCARPK.java` from repository: `https://github.com/davor10105/djl_aitech` to ai.djl.examples.training, the dataset loader file `CARPKDetection.java` to ai.djl/basicdataset/src/main/java/ai/djl/basicdataset/cv/ and the CARPK (carpk directory) dataset to djl/basicdataset/src/test/resources/mlrepo/dataset/cv/ai/djl/basicdataset/
2. Move to examples directory `djl/examples`
3. Run the training: `./gradlew run -Dmain=ai.djl.examples.training.TrainCARPK`
4. Run inference.
