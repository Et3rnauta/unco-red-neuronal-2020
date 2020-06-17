package red;

// @author guido
import com.google.gson.GsonBuilder;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import javafx.scene.chart.XYChart;

public class RedNeuronal {

    int[] topologia;
    double learningRate;
    int cantBatch;
    int cantEntrenamientos;
    double errorFinalEntrenamiento;
    Capa[] capas;

    //Variables para control de entrenamiento    
    double acumCostes, cantCostes;//promedio de los errores de cada nodo

    //Constructor
    /**
     * @param topologia : [entrada,ocultas,...,ocultas,salida]
     */
    public RedNeuronal(int[] topologia) {
        int cantCapas = topologia.length - 1;

        this.capas = new Capa[cantCapas];
        for (int i = 0; i < cantCapas; i++) {
            this.capas[i] = new Capa(topologia[i + 1], topologia[i],
                    (i + 2 > cantCapas ? 0 : topologia[i + 2]));
        }
    }

    //Optimizacion
    /**
     * Entrena una red mediante gradiant descent con un conjunto de datos de
     * entrenamiento iterando una determinada cantidad de épocas. Muestra una
     * salida cada [epoca / 100] iteraciones probando con los datos de testing
     * provistos.
     *
     * @param learningRate : valor que indica la potencia de entrenamiento.
     * Valores bajos son mas lentos y precisos, valores altos son mas rápidos y
     * poco precisos.
     *
     * @param datosTraining : matriz de datos de entrenamiento. Cada fila es una
     * tupla diferente, las primeras [n - 1] son entradas y la columna [n] es la
     * salida. La salida debe ser un arreglo con valores de 0 a N (con N siendo
     * igual a la cantidad de salidas en la topologia).
     *
     * @param cantMiniBatch : cantidad de batch a divir el dataset de
     * entrenamiento.
     *
     * @param epocas : cantidad de epocas que repetira el gradient descent.
     *
     * @param datosTesting : conjunto de testing que provee información del
     * estado actual de la red.
     *
     * @param serie : serie de datos que almacena los puntos a graficar
     */
    public void train(double learningRate, double[][] datosTraining, int cantMiniBatch,
            int epocas, double[][] datosTesting, XYChart.Series serie) {
        this.cantBatch = cantMiniBatch;

        for (int i = 0; i < epocas; i++) {
            acumCostes = 0;
            cantCostes = 0;
            double costoEntrenamiento;
            costoEntrenamiento = gradientDescent(learningRate, datosTraining, cantMiniBatch);
            //epocas debe ser mayor a 100
            if (i % (epocas / 100) == 0 && datosTesting != null) {
                //informacion de conjunto de training
                acumCostes = 0;
                cantCostes = 0;
                double aciertosTraining = test(datosTraining);
                System.out.println("**Training->Epoca: " + i + " - AvgError: "
                        + acumCostes / cantCostes + " - Aciertos: " + aciertosTraining);

                //informacion de conjunto de testing
                acumCostes = 0;
                cantCostes = 0;
                double aciertosTesting = test(datosTesting);
                System.out.println("..Testing->Epoca: " + i + " - AvgError: "
                        + acumCostes / cantCostes + " - Aciertos: " + aciertosTesting);
            }
            serie.getData().add(new XYChart.Data(i, costoEntrenamiento));
        }
    }

    /**
     * Entrena una red mediante gradiant descent con un conjunto de datos de
     * entrenamiento iterando una determinada cantidad de épocas.
     *
     * @param learningRate : valor que indica la potencia de entrenamiento.
     * Valores bajos son mas lentos y precisos, valores altos son mas rápidos y
     * poco precisos
     *
     * @param datosTraining : matriz de datos de entrenamiento. Cada fila es una
     * tupla diferente, las primeras [n - 1] son entradas y la columna [n] es la
     * salida. La salida debe ser un arreglo con valores de 0 a N (con N siendo
     * igual a la cantidad de salidas en la topologia).
     *
     * @param cantMiniBatch : cantidad de batch a divir el dataset de
     * entrenamiento.
     *
     * @param epocas : cantidad de epocas que repetira el gradient descent
     *
     * @param serie : serie de datos que almacena los puntos a graficar
     */
    public void train(double learningRate, double[][] datosTraining, int cantMiniBatch, int epocas, XYChart.Series serie) {
        train(learningRate, datosTraining, cantMiniBatch, epocas, null, serie);
    }

    private double gradientDescent(double learningRate, double[][] datosTraining, int cantMiniBatch) {
        int datosXBatch = (datosTraining.length / cantMiniBatch)
                + (datosTraining.length % cantMiniBatch > 0 ? 1 : 0),
                datosRecorridos = 0, batchRecorridos = 0;

        datosTraining = mezclarDatos(datosTraining);

        for (double[] dato : datosTraining) {
            double[][] sumasPonderadas = new double[capas.length][], //sumasPonderadas[capa][nodo]
                    salidasNodos = new double[capas.length + 1][];//salidasNodos[capa][nodo]

            //Forward pass
            forwardPass(dato, sumasPonderadas, salidasNodos);

            //Back Propagation
            backPropagation(dato, sumasPonderadas, salidasNodos);

            //evaluo fin de mini batch
            datosRecorridos++;
            if (datosRecorridos == datosXBatch) {
                //actualizo los pesos de cada arco de la red
                actualizarNodos(learningRate, datosXBatch);
                datosRecorridos = 0;
                batchRecorridos++;
                datosXBatch = (datosTraining.length / cantMiniBatch) + (datosTraining.length % cantMiniBatch > batchRecorridos ? 1 : 0);
            }
        }

        //si quedaron datos sin promediar
        if (datosRecorridos != 0) {
            //actualizo los pesos de cada arco de la red
            actualizarNodos(learningRate, datosXBatch);
        }
        return acumCostes / datosTraining.length;
    }

    private void forwardPass(double[] dato, double[][] sumasPonderadas, double[][] salidasNodos) {
        //comienzo con las entradas
        salidasNodos[0] = dato;
        //recorro cada capa
        for (int i = 0; i < capas.length; i++) {
            //calculo las sumas ponderadas
            sumasPonderadas[i] = matrizPorVector(capas[i].w, salidasNodos[i], capas[i].b);
            //calculo las funciones sigmoide
            salidasNodos[i + 1] = new double[sumasPonderadas[i].length];
            for (int j = 0; j < salidasNodos[i + 1].length; j++) {
                salidasNodos[i + 1][j] = funcionSigmoide(sumasPonderadas[i][j]);
            }
        }
        for (int i = 0; i < salidasNodos.length; i++) {
            double valorSalida = (dato[dato.length - 1] != i) ? 0 : 1;
            acumCostes += Math.pow(funcionCosteDerivada(valorSalida, salidasNodos[salidasNodos.length-1][i]), 2);
        }
    }

    private void backPropagation(double[] dato, double[][] sumasPonderadas, double[][] salidasNodos) {
        int cantSalidas = capas[capas.length - 1].b.length;
        double[][] deltas = new double[capas.length][];//deltas[capa][nodo]

        int ultimaCapa = capas.length - 1;

        //calculo los deltas de la ultima capa            
        deltas[ultimaCapa] = new double[cantSalidas];

        for (int i = 0; i < cantSalidas; i++) {
            double valorSalida = (dato[dato.length - 1] != i) ? 0 : 1;
            deltas[ultimaCapa][i] = funcionSigmoideDerivada(sumasPonderadas[ultimaCapa][i])
                    * funcionCosteDerivada(valorSalida, salidasNodos[ultimaCapa + 1][i]);
        }

        //calculo los deltas de las capas ocultas
        for (int i = ultimaCapa - 1; i >= 0; i--) {
            deltas[i] = new double[capas[i].b.length];
            //recorro los nodos de la capa
            for (int j = 0; j < capas[i].b.length; j++) {
                double sumatoriaCorreccion = 0;
                //recorro los arcos de SALIDA del nodo 
                for (int k = 0; k < capas[i + 1].b.length; k++) {
                    //accedo a los pesos que estan afectados por el nodo actual (el nodo actual es j)
                    sumatoriaCorreccion += capas[i + 1].w[k][j] * deltas[i + 1][k];
                }
                deltas[i][j] = funcionSigmoideDerivada(sumasPonderadas[i][j]) * sumatoriaCorreccion;
            }
        }

        //acumulo las actualizaciones de los pesos y los b
        for (int i = 0; i < capas.length; i++) {
            //recorro todos los nodos
            for (int j = 0; j < capas[i].w.length; j++) {
                capas[i].delAcumB[j] += deltas[i][j];
                //recorro todos sus arcos
                for (int k = 0; k < capas[i].w[j].length; k++) {
                    capas[i].delAcumW[j][k] += salidasNodos[i][k] * deltas[i][j];
                }
            }
        }
    }

    private void actualizarNodos(double learningRate, int datosXBatch) {
        //para cada capa
        for (Capa capa : capas) {
            //para cada nodo
            for (int j = 0; j < capa.b.length; j++) {
                capa.b[j] += (learningRate * (capa.delAcumB[j] / datosXBatch));
                //para cada arco
                for (int k = 0; k < capa.w[j].length; k++) {
                    capa.w[j][k] += (learningRate * (capa.delAcumW[j][k] / datosXBatch));
                }
                capa.reiniciarNablas();
            }
        }
    }

    //Testing
    /**
     * Prueba la red con los datos recibidos y almacena valores para que puedan
     * imprimirse en el entrenamiento
     *
     * @param datosTesting : los datos para realizar testing
     * @return valor que representa el porcentaje de aciertos de la red
     */
    public double test(double[][] datosTesting) {
        double aciertos = 0;

        for (double[] dato : datosTesting) {
            double[] salidasNodos = dato;

            for (Capa capa : this.capas) {
                //calculo las sumas ponderadas
                double[] sumasPonderadas;
                sumasPonderadas = matrizPorVector(capa.w, salidasNodos, capa.b);
                //calculo las funciones sigmoide
                double[] salidasNodosAux = new double[sumasPonderadas.length];
                for (int j = 0; j < salidasNodosAux.length; j++) {
                    salidasNodosAux[j] = funcionSigmoide(sumasPonderadas[j]);
                }
                salidasNodos = salidasNodosAux;
            }

            //obtengo la salida de la red y calculo el costo
            double salidaMax = -1, indiceMax = -1;
            for (int i = 0; i < salidasNodos.length; i++) {
                double valorSalida = (dato[dato.length - 1] != i) ? 0 : 1;
                acumCostes += Math.pow(funcionCosteDerivada(valorSalida, salidasNodos[i]), 2);

                if (salidaMax < salidasNodos[i]) {
                    indiceMax = i;
                    salidaMax = salidasNodos[i];
                }
            }
            cantCostes++;

            //verifico si fue un acierto y lo acumulo
            aciertos += (dato[dato.length - 1] == indiceMax) ? 1 : 0;
        }
        return aciertos / datosTesting.length;
    }

    //Conversion
    public void toJson(String name) {
        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        String red2 = builder.create().toJson(this);
        String NOMBRE_ARCHIVO = "src/output/" + name + ".txt";
        try (PrintWriter flujoDeSalida = new PrintWriter(new FileOutputStream(NOMBRE_ARCHIVO))) {
            flujoDeSalida.print(red2);
            System.out.println("Se imprimio correctamente en: " + NOMBRE_ARCHIVO);
        } catch (FileNotFoundException ex) {
            System.err.println("Archivo no encontrado");
        }
    }

    //Funciones Extra
    private double[][] mezclarDatos(double[][] datos) {
        List<double[]> lista = Arrays.asList(datos);
        Collections.shuffle(lista);
        double[][] nuevosDatos = new double[datos.length][];
        lista.toArray(nuevosDatos);
        return nuevosDatos;
    }

    private double[] matrizPorVector(double[][] matriz, double[] vector, double[] valorIni) {
        double[] res = new double[matriz.length];

        for (int i = 0; i < matriz.length; i++) {
            res[i] = valorIni[i];
            for (int j = 0; j < matriz[i].length; j++) {
                res[i] += matriz[i][j] * vector[j];
            }
        }

        return res;
    }

    private static double funcionSigmoide(double n) {
        return 1 / (1 + Math.exp(-n));
    }

    private static double funcionSigmoideDerivada(double n) {
        return 1 / (Math.exp(n) * Math.pow(1 + Math.exp(-n), 2));
    }

    private double funcionCosteDerivada(double esperado, double obtenido) {
        return esperado - obtenido;
    }
}

class Capa {

    // w[nodo][peso] | pesos
    double[][] w;
    double[][] delAcumW;
    // b[nodo] | bias
    double[] b;
    double[] delAcumB;

    Capa(int cantNodos, int cantArcos, int cantSalidas) {
        this.w = new double[cantNodos][cantArcos];
        this.delAcumW = new double[cantNodos][cantArcos];
        this.b = new double[cantNodos];
        this.delAcumB = new double[cantNodos];

        //Inicializacion utilizando la distribución Xavier Uniforme
        Random r = new Random();
        double min = -Math.sqrt(6.0) / Math.sqrt(cantArcos + cantSalidas),
                max = Math.sqrt(6.0) / Math.sqrt(cantArcos + cantSalidas);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                w[i][j] = r.nextDouble() * (max - min) + min;
            }
            b[i] = 0;
        }
    }

    void reiniciarNablas() {
        this.delAcumW = new double[w.length][w[0].length];
        this.delAcumB = new double[b.length];
    }
}
