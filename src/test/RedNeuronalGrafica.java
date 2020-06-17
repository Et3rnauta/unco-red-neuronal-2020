package test;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import red.LectorArchivos;
import red.RedNeuronal;
import red.TraductorDatos;

public class RedNeuronalGrafica extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        System.out.println("Se hace la prueba con muestra de errores graficamente");
        stage.setTitle("Red Neuronal - Error por Época - Mini Batchs");
        //definir ejes
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Iteración");
        yAxis.setLabel("Error");
        //crear el grafico
        final LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);

        lineChart.setTitle("Gráfico de Error por Época - Mini Batchs");
        //definir la serie
        XYChart.Series series = new XYChart.Series();
        series.setName("Error Obtenido");

        System.out.println("Creando la red");

        TraductorDatos traductor = new TraductorDatos(10);
        for (int i = 0; i < 5; i++) {
            traductor.addEntrada(1, 5);
            traductor.addEntrada(1, 14);
        }
        RedNeuronal redNeuronal = new RedNeuronal(new int[]{traductor.getCantEntradas(), 18, 10});

        System.out.println("Cargando los datos");
        String[][] datosTraining = LectorArchivos.leerDatosPokerTraining(),
                datosTesting = LectorArchivos.leerDatosPokerTesting();

        double[][] datosTrainingTraducidos = new double[datosTraining.length][],
                datosTestingTraducidos = new double[datosTesting.length][];

        System.out.println("Traduciendo los datos de training");
        for (int j = 0; j < datosTraining.length; j++) {
            datosTrainingTraducidos[j] = traductor.transformarEntradaBinario(datosTraining[j]);
        }

        System.out.println("Traduciendo los datos de testing");
        for (int j = 0; j < datosTesting.length; j++) {
            datosTestingTraducidos[j] = traductor.transformarEntradaBinario(datosTesting[j]);
        }

        System.out.println("Realizando un testing previo y almacenando en /output");
        double porcentajePrevio = redNeuronal.test(datosTestingTraducidos);
        System.out.println("Porcentaje: " + porcentajePrevio);
        redNeuronal.toJson("red-Mini-Batch-pre");

        System.out.println("Realizando training");
        int cantEntrenamientos = 500;
        redNeuronal.train(0.1, datosTrainingTraducidos, datosTrainingTraducidos.length, cantEntrenamientos, datosTestingTraducidos, series);

        System.out.println("Realizando un testing posterior y almacenando en /output");
        double porcentajePosterior = redNeuronal.test(datosTestingTraducidos);
        System.out.println("Posterior: " + porcentajePosterior);
        redNeuronal.toJson("red-MiniBatch-post");

        Scene scene = new Scene(lineChart, 600, 600);
        lineChart.getData().add(series);

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
