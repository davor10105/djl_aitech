package ai.djl.examples.training;

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
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;
import ai.djl.nn.pooling.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;


public final class TrainMnistCNN {

    private TrainMnistCNN() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainMnistCNN.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        try (Model model = Model.newInstance("cnn")) {
            model.setBlock(getCNNModel());

            // get training and validation dataset
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(1, 1, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, 10, trainingSet, validateSet);

                return trainer.getTrainingResult();
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

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Mnist mnist =
                Mnist.builder()
                        .optUsage(usage)
                        .setSampling(arguments.getBatchSize(), true)
                        .optLimit(arguments.getBatchSize())
                        .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }
}
