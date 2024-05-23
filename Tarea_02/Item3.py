import time
import random
import matplotlib.pyplot as plt

def greedy_estocastico(sectores, lugares, costos, coberturas, alpha=0.1):
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

def eliminar_lugares(solucion, n):
    return random.sample(solucion, len(solucion) - n)

def completar_solucion(sectores, lugares, costos, coberturas, solucion_parcial):
    sectores_cubiertos = set()
    for lugar in solucion_parcial:
        for sector in range(sectores):
            if lugar in coberturas[sector]:
                sectores_cubiertos.add(sector)
    
    while len(sectores_cubiertos) < sectores:
        candidatos = []
        
        for lugar in range(lugares):
            if lugar in solucion_parcial:
                continue
            
            sectores_que_cubre = set()
            for sector in range(sectores):
                if lugar in coberturas[sector]:
                    sectores_que_cubre.add(sector)
            
            sectores_no_cubiertos = [s for s in sectores_que_cubre if s not in sectores_cubiertos]
            if sectores_no_cubiertos:
                costo_beneficio = costos[lugar] / len(sectores_no_cubiertos)
                candidatos.append((costo_beneficio, lugar, sectores_no_cubiertos))
        
        candidatos.sort(key=lambda x: x[0])
        if candidatos:
            mejor_candidato = candidatos[0]
            _, mejor_lugar, mejor_cobertura = mejor_candidato
            solucion_parcial.append(mejor_lugar)
            sectores_cubiertos.update(mejor_cobertura)
        else:
            break
    
    return solucion_parcial

def ejecutar_variante(sectores, lugares, costos, coberturas, n, m):
    start_time = time.time()
    
    solucion = greedy_estocastico(sectores, lugares, costos, coberturas, alpha=0.1)
    valor_mejor_solucion = sum(costos[lugar] for lugar in solucion)
    tiempos = [0]
    valores = [valor_mejor_solucion]
    
    for _ in range(m):
        solucion_parcial = eliminar_lugares(solucion, n)
        solucion = completar_solucion(sectores, lugares, costos, coberturas, solucion_parcial)
        valor_nueva_solucion = sum(costos[lugar] for lugar in solucion)
        
        if valor_nueva_solucion < valor_mejor_solucion:
            valor_mejor_solucion = valor_nueva_solucion
        
        tiempos.append(time.time() - start_time)
        valores.append(valor_mejor_solucion)
    
    return tiempos, valores

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


if __name__ == "__main__":
    ruta_archivo = './Enunciado/C2.txt'
    sectores, lugares, costos, coberturas = leer_datos_archivo(ruta_archivo)
    print(f"sectores: {sectores} \n lugares: {lugares}")
    
    n_eliminar = 10  # Cantidad de lugares a eliminar
    m_iteraciones = 45  # Cantidad de iteraciones
    
    # Ejecutar la variante 5 veces y graficar los resultados
    plt.figure()
    valores_finales = []
    colores = ['b', 'orange', 'g', 'r', 'purple']
    
    for i in range(5):
        tiempos, valores = ejecutar_variante(sectores, lugares, costos, coberturas, n_eliminar, m_iteraciones)
        with open('resultados_p3_C2_n10_m45.txt','a') as file:
            file.write(f'Iteración: {i+1}\n')
            file.write(f'F. objetivo: {valores}\n')
            file.write(f'Tiempo: {tiempos}\n')
            file.write('---------------------------\n\n')
        plt.plot(tiempos, valores, marker='o', color=colores[i], label=f'Ejecución {i+1} - F.O.: {valores[-1]:.2f}')
        valores_finales.append(valores[-1])

    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Valor de la función objetivo')
    plt.title('Convergencia de la solución (P3_C2_n10_m45)')
    plt.legend(loc='upper right')
    
    plt.show()