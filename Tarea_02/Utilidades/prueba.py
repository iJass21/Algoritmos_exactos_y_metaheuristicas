import time
import random

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
        contenido = archivo.readlines()  # Lee el resto del archivo de una vez
        i = 0
        while i < len(contenido):
            # Lee la cantidad de datos para el siguiente arreglo
            if contenido[i].strip():  # Asegura que no esté vacía la línea
                cantidad = int(contenido[i].strip())
                i += 1

                # Inicializa una lista vacía para almacenar los datos
                datos = []
                
                # Continúa leyendo las siguientes líneas hasta completar la cantidad necesaria
                while len(datos) < cantidad and i < len(contenido):
                    linea = contenido[i].strip()
                    if linea:  # Asegúrate de que la línea no esté vacía
                        datos.extend(map(int, linea.split()))
                    i += 1
                
                # Agrega la lista de datos a las coberturas
                coberturas.append(datos)

    return m, n, costos, coberturas

if __name__ == "__main__":
    ruta_archivo = 'C1.txt'
    sectores, lugares, costos, coberturas = leer_datos_archivo(ruta_archivo)
    
    with open('C1-determinista.txt', 'a') as file:
        # Medir el tiempo de ejecución para el greedy determinista
        start_time = time.time()
        lugares_obtenidos_determinista = greedy_determinista(sectores, lugares, costos, coberturas)
        end_time = time.time()
        tiempo_ejecucion_determinista = end_time - start_time
        valor_funcion_objetivo_determinista = sum(costos[lugar] for lugar in lugares_obtenidos_determinista)
        
        file.write(f'Lugares: {lugares_obtenidos_determinista}\n')
        file.write(f'F obj: {valor_funcion_objetivo_determinista}\n')
        file.write(f'T: {tiempo_ejecucion_determinista:.8f} segundos\n')
        file.write('-------------------------------------------------------------------------------\n')

    with open('C1-res-item1-estocastico-alpha1.txt', 'a') as file:
        # Medir el tiempo de ejecución para el greedy estocástico

        for i in range(10):
            start_time = time.time()
            lugares_obtenidos_estocastico = greedy_estocastico(sectores, lugares, costos, coberturas, alpha=0.1)
            end_time = time.time()
            tiempo_ejecucion_estocastico = end_time - start_time
            valor_funcion_objetivo_estocastico = sum(costos[lugar] for lugar in lugares_obtenidos_estocastico)
            
            file.write(f'Iteracion: {i+1}\n')
            file.write(f'Lugares: {lugares_obtenidos_estocastico}\n')
            file.write(f'F obj: {valor_funcion_objetivo_estocastico}\n')
            file.write(f'T: {tiempo_ejecucion_estocastico:.8f} segundos\n')
            file.write('-------------------------------------------------------------------------------\n')