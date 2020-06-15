package red;

import java.util.HashMap;

// @author Guido, Gaston y Seba
public class TraductorDatos {

    HashMap[] dominiosColumnas;//HashMap de cada columna (Rango o CjtoValores)
    int[] cantBitColumna;
    int indiceActual;
    int totalEntradasBit;

    public TraductorDatos(int cantColumnas) {
        dominiosColumnas = new HashMap[cantColumnas];
        cantBitColumna = new int[cantColumnas];
        indiceActual = 0;
        totalEntradasBit = 0;
    }

    /**
     * Agrega y mapea todas las entradas hacia enteros (CjtoValores)
     *
     * @param valores : arreglo con todos los diferentes valores de entrada
     */
    public void addEntrada(String[] valores) {
        HashMap<String, Integer> hashMapEntrada = new HashMap<>();
        int cantBit = (int) Math.ceil(Math.log10(valores.length) / Math.log10(2));
        this.totalEntradasBit += cantBit;
        for (int i = 0; i < valores.length; i++) {
            hashMapEntrada.put("-" + valores[i], i);
        }
        dominiosColumnas[indiceActual] = hashMapEntrada;
        cantBitColumna[indiceActual] = cantBit;
        indiceActual++;
    }

    /**
     * Agrega los limites del rango (Rango)
     *
     * @param min : valor minimo del rango de enteros
     * @param max : valor maximo del rango de enteros (se toma hasta el
     * anterior)
     */
    public void addEntrada(int min, int max) {
        HashMap<String, Integer> hashMapEntrada = new HashMap<>();
        int rango = Math.abs(max - min),
                cantBit = (int) Math.ceil(Math.log10(rango) / Math.log10(2));
        this.totalEntradasBit += cantBit;
        hashMapEntrada.put("min", min);
        hashMapEntrada.put("max", max);
        dominiosColumnas[indiceActual] = hashMapEntrada;
        cantBitColumna[indiceActual] = cantBit;
        indiceActual++;
    }

    /**
     * Retorna la cantidad total de bits para las entradas cargadas
     *
     * @return cantidad total de bits
     */
    public int getCantEntradas() {
        return totalEntradasBit;
    }

    /**
     * Transforma las entradas a un conjunto de bits
     *
     * @param entradas : tupla de valores de entrada a transformar
     * @return arreglo de bits de todas las entradas transformadas
     * @throws java.lang.Exception si el valor de una de las entradas no es
     * valido, ya que no se encuentra dentro de los posibles, o es un dato
     * incorrecto
     */
    public double[] transformarEntradaBinario(String[] entradas) throws Exception {
        double[] entradaBin = new double[totalEntradasBit + 1];
        int cont = 0;

        for (int i = 0; i < entradas.length - 1; i++) {
            HashMap<String, Integer> hashColumna = dominiosColumnas[i];
            Integer valor = hashColumna.get("-" + entradas[i]);
            if (valor == null) {
                //No esta en cjto de valores o no es un cjto de valores
                Integer min = hashColumna.get("min"),
                        max = hashColumna.get("max");
                if (min == null || max == null) {
                    //No esta en cjto de valores
                    throw new Exception("La entrada: " + entradas[i] + " no es un valor válido dentro del conjunto");
                }
                int entrada;
                try {
                    entrada = Integer.parseInt(entradas[i]);
                } catch (NumberFormatException ex) {
                    //No es un entero
                    throw new Exception("La entrada: " + entradas[i] + " no es un valor numerico válido");
                }
                if (entrada < min || entrada > max) {
                    //No esta en el rango
                    throw new Exception("La entrada: " + entradas[i] + " no es un valor dentro del rango posible");
                }
                valor = entrada - min;
            }
            double[] entradaBinAux = transformarBinario(valor, cantBitColumna[i]);
            for (double bit : entradaBinAux) {
                entradaBin[cont] = bit;
                cont++;
            }
        }
        entradaBin[entradaBin.length - 1] = Double.parseDouble(entradas[entradas.length - 1]);
        return entradaBin;
    }

    private static double[] transformarBinario(int num, int cantBitTotal) throws Exception {
        double[] numBinario = new double[cantBitTotal];
        int i = 0;
        int numAux = num;

        try {
            while (numAux > 0) {
                numBinario[cantBitTotal - 1 - i] = numAux % 2;
                numAux = numAux / 2;
                i++;
            }
            for (int j = i; j < cantBitTotal; j++) {
                //Si es necesario agregar mas 0 debido a la cantidad de bits totales
                numBinario[cantBitTotal - 1 - j] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException ex) {
            throw new Exception("El número no esta dentro del rango definido");
        }
        return numBinario;
    }

    public double[] transformarDouble(String[] datosString) {
        double[] datos = new double[datosString.length];
        for (int i = 0; i < datosString.length; i++) {
            datos[i] = Double.parseDouble(datosString[i]);
        }
        return datos;
    }
}
