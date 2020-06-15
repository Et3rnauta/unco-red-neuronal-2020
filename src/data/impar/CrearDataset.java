package data.impar;

// @author Guido, Gaston y Seba
import java.util.Random;

public class CrearDataset {

    /**
     * Crea un dataset q genera combinaciones de entradas con una salida que
     * indica si es par o no la suma de ellos 1: impar, 0: par
     *
     * @param cantEntradas : cantidad de entradas de la red
     * @param cantDatos : cantidad de datos en la red
     * @return cadena con el dataset separado mediante ","
     */
    public static String crearDatasetString(int cantEntradas, int cantDatos) {
        String dataset = "";
        Random r = new Random();
        int acum;

        for (int i = 0; i < cantDatos; i++) {
            acum = 0;
            for (int j = 0; j < cantEntradas; j++) {
                int valor = r.nextInt(16);
                acum += valor;
                dataset += valor + ",";
            }
            dataset += (acum % 2 == 1) ? 1 : 0 + "\n";
        }

        return dataset;
    }

    /**
     * Crea un dataset q genera combinaciones de entradas con una salida que
     * indica si es par o no la suma de ellos 1: impar, 0: par
     *
     * @param cantEntradas : cantidad de entradas de la red
     * @param cantDatos : cantidad de datos en la red
     * @return matriz con el dataset
     */
    public static String[][] crearDatasetMatriz(int cantEntradas, int cantDatos) {
        String[][] dataset = new String[cantDatos][cantEntradas + 1];
        Random r = new Random();
        int acum;

        for (int i = 0; i < cantDatos; i++) {
            acum = 0;
            for (int j = 0; j < cantEntradas; j++) {
                int valor = r.nextInt(16);
                acum += valor;
                dataset[i][j] = "" + valor;
            }
            dataset[i][dataset[i].length - 1] = (acum % 2 == 1) ? "1" : "0";
        }

        return dataset;
    }
}
