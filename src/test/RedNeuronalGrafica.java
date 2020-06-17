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
        stage.setTitle("Red Neuronal - Error por Época - Estocástico");
        //definir ejes
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Iteración");
        yAxis.setLabel("Error");
        //crear el grafico
        final LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);

        lineChart.setTitle("Gráfico de Error por Época - Estocástico"); 
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

        System.out.println("Carga los datos");
        String[][] datosTraining = LectorArchivos.leerDatosPokerTraining(),
                datosTesting = LectorArchivos.leerDatosPokerTesting(); 
        
        double[][] datosTrainingTraducidos = new double[datosTraining.length][],
                datosTestingTraducidos = new double[datosTesting.length][];

        System.out.println("Traduce los datos de training");
        for (int j = 0; j < datosTraining.length; j++) {
            datosTrainingTraducidos[j] = traductor.transformarEntradaBinario(datosTraining[j]);
        }

        System.out.println("Traduce los datos de testing");
        for (int j = 0; j < datosTesting.length; j++) {
            datosTestingTraducidos[j] = traductor.transformarEntradaBinario(datosTesting[j]);
        }

        System.out.println("Primera fase de testing");
        double porcentajePrevio = redNeuronal.testRed(datosTestingTraducidos);
        System.out.println("Previo: " + porcentajePrevio);
        redNeuronal.toJson("red-estocástica-pre");

        System.out.println("Fase de training");
        int cantEntrenamientos = 500;

        //cargar errores        
        redNeuronal.cantEntrenamientos = cantEntrenamientos;
        for (int i = 0; i < cantEntrenamientos; i++) {
            double error = redNeuronal.gradientDescent(0.1, datosTrainingTraducidos);
            series.getData().add(new XYChart.Data(i, error));
        }

        //Test con datos de Testing         
        System.out.println("Segunda fase de testing");
        double porcentajePosterior = redNeuronal.testRed(datosTestingTraducidos);
        redNeuronal.toJson("red-estocástica-post");
        System.out.println("Posterior: " + porcentajePosterior);

        Scene scene = new Scene(lineChart, 600, 600);
        lineChart.getData().add(series);

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
