import time
import random
import matplotlib.pyplot as plt

def greedy_determinista(sectores, lugares, costos, coberturas):
    sectores_cubiertos = set()
    lugares_elegidos = []
    
    # Mientras no se cubran todos los sectores
    while len(sectores_cubiertos) < sectores:
        mejor_costo_beneficio = float('inf')
        mejor_lugar = -1
        mejor_cobertura = []
        
        # Encontrar el lugar con el mejor costo-beneficio
        for lugar in range(lugares):
            # Calcular los sectores que cubre este lugar
            sectores_que_cubre = set()
            for sector in range(sectores):
                if lugar in coberturas[sector]:
                    sectores_que_cubre.add(sector)

            # Calcular la cantidad de sectores no cubiertos que cubre este lugar
            sectores_no_cubiertos = [s for s in sectores_que_cubre if s not in sectores_cubiertos]
            if sectores_no_cubiertos:
                costo_beneficio = costos[lugar] / len(sectores_no_cubiertos)
                if costo_beneficio < mejor_costo_beneficio:
                    mejor_costo_beneficio = costo_beneficio
                    mejor_lugar = lugar
                    mejor_cobertura = sectores_no_cubiertos
        
        # Si encontramos un lugar válido, lo añadimos a los lugares elegidos y actualizamos los sectores cubiertos
        if mejor_lugar != -1:
            lugares_elegidos.append(mejor_lugar)
            sectores_cubiertos.update(mejor_cobertura)
        else:
            # No hay más lugares que puedan cubrir sectores no cubiertos
            break
    
    return lugares_elegidos

def greedy_estocastico(sectores, lugares, costos, coberturas, alpha=0.05):
    sectores_cubiertos = set()
    lugares_elegidos = []
    
    while len(sectores_cubiertos) < sectores:
        candidatos = []
        
        for lugar in range(lugares):
            sectores_que_cubre = set()
            for sector in range(sectores):
                if lugar in coberturas[sector]:
                    sectores_que_cubre.add(sector)
            
            sectores_no_cubiertos = [s for s in sectores_que_cubre if s not in sectores_cubiertos]
            if sectores_no_cubiertos:
                costo_beneficio = costos[lugar] / len(sectores_no_cubiertos)
                candidatos.append((costo_beneficio, lugar, sectores_no_cubiertos))
        
        # Ordenar los candidatos por costo-beneficio
        candidatos.sort(key=lambda x: x[0])
        
        # Seleccionar los mejores candidatos con un factor de aleatoriedad
        k = max(1, int(alpha * len(candidatos)))
        mejor_candidato = random.choice(candidatos[:k])
        
        # Añadir el mejor candidato encontrado a los lugares elegidos y actualizar sectores cubiertos
        _, mejor_lugar, mejor_cobertura = mejor_candidato
        lugares_elegidos.append(mejor_lugar)
        sectores_cubiertos.update(mejor_cobertura)
    
    return lugares_elegidos

def leer_datos_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        m = int(archivo.readline().strip())  # cantidad de sectores
        n = int(archivo.readline().strip())  # cantidad de lugares donde instalar clínicas
        
        # Leer el vector de costos de instalación
        costos = []
        while len(costos) < n:
            line = archivo.readline().strip()
            if line:  # Asegurarse de que la línea no esté vacía
                costos.extend(map(int, line.split()))
        
        # Leer las coberturas
        coberturas = []
        while True:
            line = archivo.readline().strip()
            if not line:
                break
            cantidad = int(line)  # Leer la cantidad de lugares para el sector actual
            lugares = []
            while len(lugares) < cantidad:
                line = archivo.readline().strip()
                if line:  # Asegurarse de que la línea no esté vacía
                    lugares.extend(map(int, line.split()))
            coberturas.append(lugares)
    
    return m, n, costos, coberturas

def hill_climbing(sectores, lugares, costos, coberturas, alpha=0.02, max_iter=500, reinicios=10):
    mejor_solucion = greedy_estocastico(sectores, lugares, costos, coberturas, alpha)
    mejor_valor = sum(costos[lugar] for lugar in mejor_solucion)
    iter_sin_mejora = 0
    valores_funcion_objetivo = [mejor_valor]
    iteraciones = [0]
    
    print("Inicialización - Mejor valor:", mejor_valor)  # Impresión de la inicialización

    for i in range(1, max_iter + 1):
        vecino = list(mejor_solucion)
        indices = random.sample(range(len(vecino)), 2)
        vecino[indices[0]], vecino[indices[1]] = vecino[indices[1]], vecino[indices[0]]
        
        valor_vecino = sum(costos[lugar] for lugar in vecino)
        
        if valor_vecino < mejor_valor:
            mejor_solucion = vecino
            mejor_valor = valor_vecino
            iter_sin_mejora = 0
            valores_funcion_objetivo.append(mejor_valor)
            iteraciones.append(i)
            print(f"Iteración {i} Nuevo mejor valor = {mejor_valor}")
        else:
            iter_sin_mejora += 1
        
        if iter_sin_mejora >= reinicios:
            nuevo_intento_solucion = greedy_estocastico(sectores, lugares, costos, coberturas, alpha)
            valor_actual = sum(costos[lugar] for lugar in nuevo_intento_solucion)
            if valor_actual < mejor_valor:
                mejor_solucion = nuevo_intento_solucion
                mejor_valor = valor_actual
                valores_funcion_objetivo.append(mejor_valor)
                iteraciones.append(i)
                print(f"Iteración {i} - r - nuevo mejor valor = {mejor_valor}")
            iter_sin_mejora = 0
    
    return mejor_solucion, valores_funcion_objetivo, iteraciones

if __name__ == "__main__":
    ruta_archivo = 'C2.txt'
    sectores, lugares, costos, coberturas = leer_datos_archivo(ruta_archivo)
    
    # Ejecutar Hill Climbing 5 veces
    for i in range(5):
        start_time = time.time()  # Registrar el tiempo de inicio de la iteración
        lugares_obtenidos_hill_climbing, valores_funcion_objetivo_hill_climbing, iteraciones_hill_climbing = hill_climbing(sectores, lugares, costos, coberturas, alpha=0.05, max_iter=1000)
        end_time = time.time()  # Registrar el tiempo de finalización de la iteración
        tiempo_ejecucion_hill_climbing = end_time - start_time  # Calcular el tiempo de ejecución

        valor_funcion_objetivo_hill_climbing = sum(costos[lugar] for lugar in lugares_obtenidos_hill_climbing)

        with open(f"item2-hc-am-ra-C2.txt", 'a') as file:
            file.write(f'Iteracion: {i+1}\n')
            file.write(f'Lugares: {lugares_obtenidos_hill_climbing}\n')
            file.write(f'F obj: {valor_funcion_objetivo_hill_climbing}\n')
            file.write(f'T: {tiempo_ejecucion_hill_climbing:.8f} segundos\n')
            file.write('------------------------------------------------------------------------------------------------------\n')

        if(i==4):
            #Generar gráfico de convergencia
            plt.plot(iteraciones_hill_climbing, valores_funcion_objetivo_hill_climbing, marker='o')
            plt.xlabel('Iteraciones')
            plt.ylabel('Valor de la función objetivo')
            plt.title('Convergencia de Hill Climbing')
            plt.grid(True)
            plt.show()