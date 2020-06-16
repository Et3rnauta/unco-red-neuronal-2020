package red;

import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author guido
 */
public class RedNeuronalPokerTest {

    public static RedNeuronal redNeuronal;
    public static TraductorDatos traductor;

    @BeforeClass
    public static void setUpClass() {
        System.out.println("Creando la red");

        traductor = new TraductorDatos(10);
        for (int i = 0; i < 5; i++) {
            traductor.addEntrada(1, 5);
            traductor.addEntrada(1, 14);
        }
        redNeuronal = new RedNeuronal(new int[]{traductor.getCantEntradas(), 20, 10});
    }

    @Test
    public void test1() throws Exception {
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
        redNeuronal.toJson("red-poker-test-pre");

        System.out.println("Realizando training");
        redNeuronal.train(5, datosTrainingTraducidos, 4, 10000, datosTestingTraducidos);

        System.out.println("Realizando un testing posterior y almacenando en /output");
        double porcentajePosterior = redNeuronal.test(datosTestingTraducidos);
        System.out.println("Posterior: " + porcentajePosterior);
        redNeuronal.toJson("red-binario-poker-test-post");
        assertTrue(porcentajePrevio < porcentajePosterior);
    }
}
