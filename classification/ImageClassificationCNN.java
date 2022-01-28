package ai.djl.examples.inference;

import ai.djl.Model;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.ModelException;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;
import ai.djl.nn.pooling.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;


public final class ImageClassificationCNN {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    private ImageClassificationCNN() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Classifications classifications = ImageClassification.predict();
        logger.info("{}", classifications);
    }

    public static Classifications predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/0.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String modelName = "mlp";
        try (Model model = Model.newInstance(modelName)) {
            model.setBlock(getCNNModel());

            // Assume you have run TrainMnist.java example, and saved model in build/model folder.
            Path modelDir = Paths.get("build/model");
            model.load(modelDir);

            List<String> classes =
                    IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
            Translator<Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(classes)
                            .build();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                return predictor.predict(img);
            }
        }
    }

    private static Block getCNNModel() {
        SequentialBlock net = new SequentialBlock();
        net
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(1, 1))
                                .optStride(new Shape(1, 1))
                                .optPadding(new Shape(0, 0))
                                .setFilters(32)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(1, 1))
        );
        
        net
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .optStride(new Shape(1, 1))
                                .optPadding(new Shape(0, 0))
                                .setFilters(64)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(1, 1))
        );
        net
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(10).build());

        return net;
    }
}
