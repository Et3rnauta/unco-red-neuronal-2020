package red;

import org.junit.BeforeClass;
import org.junit.Test;
import data.impar.CrearDataset;
import static org.junit.Assert.*;

/**
 *
 * @author guido
 */
public class RedNeuronalImparTest {

    public static RedNeuronal redNeuronal;
    public static TraductorDatos traductor;
    public static int cantDatos = 2000, cantEntradas = 4;

    @BeforeClass
    public static void setUpClass() {
        System.out.println("Creando la red");

        traductor = new TraductorDatos(cantEntradas);
        for (int i = 0; i < cantEntradas; i++) {
            traductor.addEntrada(0, 16);
        }
        redNeuronal = new RedNeuronal(new int[]{traductor.getCantEntradas(), 16, 2});
    }

    @Test
    public void test1() throws Exception {
        System.out.println("Cargando los datos");
        String[][] datosTraining = CrearDataset.crearDatasetMatriz(cantEntradas, cantDatos),
                datosTesting = CrearDataset.crearDatasetMatriz(cantEntradas, cantDatos / 10);

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
        redNeuronal.toJson("red-impar-test-pre");

        System.out.println("Realizando training");
        redNeuronal.train(5, datosTrainingTraducidos, 4, 100000, datosTestingTraducidos);

        System.out.println("Realizando un testing posterior y almacenando en /output");
        double porcentajePosterior = redNeuronal.test(datosTestingTraducidos);
        System.out.println("Posterior: " + porcentajePosterior);
        redNeuronal.toJson("red-impar-test-post");
        assertTrue(porcentajePrevio < porcentajePosterior);
    }
}
