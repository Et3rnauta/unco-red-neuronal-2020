package red;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RedNeuronal {

    int[] topologia;
    double learningRate;
    //int cantBatch;
    public int cantEntrenamientos;
    double errorFinalEntrenamiento;
    Capa[] capas;

    //Constructor
    /**
     * @param topologia : [entrada,ocultas,...,ocultas,salida]
     */
    public RedNeuronal(int[] topologia) {
        int cantCapas = topologia.length - 1;
        this.topologia = topologia;
        this.capas = new Capa[cantCapas];
        for (int i = 0; i < cantCapas; i++) {
            this.capas[i] = new Capa(topologia[i + 1], topologia[i], (i + 2 > cantCapas ? 0 : topologia[i + 2]));
        }
    }

    public RedNeuronal(String nombreArchivo) {
        RedNeuronal redNeuronal = null;
        String linea, json = "";
        Gson gson = new Gson();

        String NOMBRE_ARCHIVO = "src/salidas/" + nombreArchivo + ".txt";
        try (BufferedReader flujoDeEntrada = new BufferedReader(new FileReader(NOMBRE_ARCHIVO))) {
            while ((linea = flujoDeEntrada.readLine()) != null) {
                json += linea;
            }
            System.out.println("Se leyo correctamente del archivo: " + NOMBRE_ARCHIVO);
            redNeuronal = gson.fromJson(json, RedNeuronal.class);
        } catch (FileNotFoundException ex) {
            System.err.println("Archivo no encontrado");
        } catch (IOException ex) {
            Logger.getLogger(RedNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }        
    }

    //Optimizacion
    /**
     * Entrena una red mediante gradiant descent con un conjunto de datos de
     * entrenamiento.
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
     * entrenamiento
     *
     * @param cantEntrenamientos : cantidad de veces que se entrenó la red
     */
    public void gradientDescent(double learningRate, double[][] datosTraining, int cantMiniBatch, int cantEntrenamientos) {
        int datosXBatch = (datosTraining.length / cantMiniBatch) + (datosTraining.length % cantMiniBatch > 0 ? 1 : 0),
                datosRecorridos = 0, batchRecorridos = 0;

        double costo = 0;

        //this.cantBatch = cantMiniBatch;
        this.cantEntrenamientos = cantEntrenamientos;
        this.learningRate = learningRate;

        datosTraining = mezclarDatos(datosTraining);

        for (double[] datos : datosTraining) {
            double[][] sumasPonderadas = new double[capas.length][], //sumasPonderadas[capa][nodo]
                    salidasNodos = new double[capas.length + 1][];//salidasNodos[capa][nodo]
            //Forward Pass
            //comienzo con las entradas
            salidasNodos[0] = datos;
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

            //calculo el error que tiene la red
            costo += funcionCoste(datos[datos.length - 1], salidasNodos[this.capas.length]);
            //Back Propagation
            backPropagation(datos, sumasPonderadas, salidasNodos);

            //evaluo fin de mini batch
            datosRecorridos++;
            if (datosRecorridos == datosXBatch) {

                //actualizo los pesos de cada arco de la red
                actualizarNodos(learningRate, datosXBatch);
                datosRecorridos = 0;
                datosXBatch = (datosTraining.length / cantMiniBatch) + (datosTraining.length % cantMiniBatch > batchRecorridos ? 1 : 0);
                batchRecorridos++;
            }
        }

        //si quedaron datos sin promediar
        if (datosRecorridos != 0) {
            //actualizo los pesos de cada arco de la red
            actualizarNodos(learningRate, datosXBatch);

        }
        System.out.println("Error de la red: " + costo / datosTraining.length);
    }

    public double gradientDescent1(double learningRate, double[][] datosTraining) {
        int cantSalidas = this.capas[this.capas.length - 1].b.length;
        int ultimaCapa = capas.length - 1;
        double costo = 0.0, costoTotal, pAciertos;
        datosTraining = mezclarDatos(datosTraining);
        this.learningRate = learningRate;

        for (double[] dato : datosTraining) {
            double[][] sumasPonderadas = new double[capas.length][], //sumasPonderadas[capa][nodo]
                    salidasNodos = new double[capas.length + 1][];//salidasNodos[capa][nodo]

            //Forward Pass
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
            //calculo el error que tiene la red

            costo += funcionCoste(dato[dato.length - 1], salidasNodos[this.capas.length]);

            //Back Propagation
            backPropagation(dato, sumasPonderadas, salidasNodos, learningRate);
        }
        pAciertos = this.testRed(datosTraining);
        costoTotal = costo / datosTraining.length;
        System.out.println("Error de la red: " + costoTotal + " Porcentaje aciertos: " + pAciertos);
        this.errorFinalEntrenamiento = costoTotal;
        return costoTotal;
    }

    private void backPropagation(double[] dato, double[][] sumasPonderadas, double[][] salidasNodos, double learningRate) {
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
        for (int i = capas.length - 2; i >= 0; i--) {
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

        //sumo las actualizaciones de los pesos y los b
        for (int i = 0; i < capas.length; i++) {
            for (int j = 0; j < capas[i].w.length; j++) {
                capas[i].b[j] += learningRate * deltas[i][j];
                for (int k = 0; k < capas[i].w[j].length; k++) {
                    capas[i].w[j][k] += learningRate * salidasNodos[i][k] * deltas[i][j];
                }
                capas[i].reiniciarNablas();
            }
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
        for (int i = capas.length - 2; i >= 0; i--) {
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

        //sumo las actualizaciones de los pesos y los b
        for (int i = 0; i < capas.length; i++) {
            //recorro todos los nodos
            for (int j = 0; j < capas[i].w.length; j++) {
                capas[i].nablaB[j] += deltas[i][j];
                //recorro todos sus arcos
                for (int k = 0; k < capas[i].w[j].length; k++) {
                    capas[i].nablaW[j][k] += salidasNodos[i][k] * deltas[i][j];
                }
            }
        }
    }

    private void actualizarNodos(double learningRate, int datosXBatch) {
        for (int i = 0; i < capas.length; i++) {
            for (int j = 0; j < capas[i].w.length; j++) {
                capas[i].b[j] += learningRate * capas[i].nablaB[j] / datosXBatch;
                for (int k = 0; k < capas[i].w[j].length; k++) {
                    capas[i].w[j][k] += learningRate * capas[i].nablaW[j][k] / datosXBatch;
                }
                capas[i].reiniciarNablas();
            }
        }
    }

    //Testing
    /**
     * @param datosTesting : los datos para realizar testing
     * @return valor que representa el porcentaje de aciertos de la red
     */
    public double testRed(double[][] datosTesting) {
        double aciertos = 0;
        double costo = 0.0;
        int e = 0;

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

            //verifico si fue un acierto
            double max = -1, indiceMax = -1;
            for (int i = 0; i < salidasNodos.length; i++) {
                //if (e % 100 == 0) {
                //System.out.print(salidasNodos[i] + " ");
                //}
                if (max < salidasNodos[i]) {
                    indiceMax = i;
                    max = salidasNodos[i];
                }
            }
//            if (e % 100 == 0) {
            //System.out.println("");
//            }
            //System.out.println(indiceMax);
            aciertos += (dato[dato.length - 1] == indiceMax) ? 1 : 0;

            //calculo el error que tiene la red
            costo += funcionCoste(dato[dato.length - 1], salidasNodos);
            e++;
        }
//        System.out.println("Error de la red Testing: " + costo / datosTesting.length);
        return aciertos / datosTesting.length;
    }

    //Conversion
    public void toJson(String nombreArchivo) {
        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        Gson gson = builder.create();
        String red2 = gson.toJson(this);
        String NOMBRE_ARCHIVO = "src/salidas/" + nombreArchivo + ".txt";
        try (PrintWriter flujoDeSalida = new PrintWriter(new FileOutputStream(NOMBRE_ARCHIVO))) {
            flujoDeSalida.print(red2);
            System.out.println("Se imprimio correctamente en: " + NOMBRE_ARCHIVO);
        } catch (FileNotFoundException ex) {
            System.err.println("Archivo no encontrado");
        }
    }

//Funciones Extra
    private static double funcionSigmoide(double n) {
        return 1 / (1 + Math.exp(-n));
    }

    private static double funcionSigmoideDerivada(double n) {
        return 1 / (Math.exp(n) * Math.pow(1 + Math.exp(-n), 2));
        //return funcionSigmoide(n)*(1.0-funcionSigmoide(n));
    }

    private double funcionCoste(double salida, double[] predicho) {
        double costo = 0.0;
        for (int i = 0; i < predicho.length; i++) {
            costo += Math.pow(predicho[i] - ((salida != i) ? 0 : 1), 2);
        }
        return costo;
    }

    private double funcionCosteDerivada(double esperado, double obtenido) {
        return esperado - obtenido;
        //return obtenido-esperado;
    }

    private double[] generarArregloEsperados(double[] dato, int cantSalidas) {
        double[] arr = new double[cantSalidas];
        for (int j = 0; j < cantSalidas; j++) {
            arr[j] = (dato[dato.length - 1] == j) ? 1 : 0;
        }
        return arr;
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

    private static double[][] mezclarDatos(double[][] datosTraining) {
        List<double[]> lista = Arrays.asList(datosTraining);
        Collections.shuffle(lista);
        double[][] nuevosDatos = new double[datosTraining.length][];
        lista.toArray(nuevosDatos);
        return nuevosDatos;
    }

    void mostrarRed() {
        for (Capa capa : capas) {
            for (int i = 0; i < capa.w.length; i++) {
                for (int j = 0; j < capa.w[i].length; j++) {
                    System.out.print(capa.w[i][j]);
                }
                System.out.print(capa.b[i]);
                System.out.println("-nodo-");
            }
            System.out.println("-capa-");

        }
    }

}

class Capa {

    // w[nodo][peso] | pesos
    double[][] w;
    double[][] nablaW;
    // b[nodo] | bias
    double[] b;
    double[] nablaB;

    Capa(int cantNodos, int cantArcos, int cantSalidas) {

        this.w = new double[cantNodos][cantArcos];
        this.nablaW = new double[cantNodos][cantArcos];
        this.b = new double[cantNodos];
        this.nablaB = new double[cantNodos];

        //inicializacion utilizando la distribución Xavier Uniforme
        Random r = new Random();
        double min = -Math.sqrt(6.0) / Math.sqrt(cantArcos + cantSalidas),
                max = Math.sqrt(6.0) / Math.sqrt(cantArcos + cantSalidas);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                w[i][j] = r.nextDouble() * (max - min) + min;
            }
            b[i] = r.nextDouble();
            //b[i] = 1;
        }
    }

    Capa(double[] b, double[][] w) {
        this.b = b;
        this.w = w;
    }

    void reiniciarNablas() {
        this.nablaW = new double[w.length][w[0].length];
        this.nablaB = new double[b.length];
    }
}
