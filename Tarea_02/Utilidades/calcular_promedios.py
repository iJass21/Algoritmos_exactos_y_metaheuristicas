def calcular_promedios(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        lineas = archivo.readlines()
    
    fo_finales = []
    tiempos_finales = []
    
    iteracion_actual = 0
    fo_valores = []
    tiempo_valores = []
    
    for linea in lineas:
        linea = linea.strip()
        if linea.startswith("Iteración:"):
            if fo_valores and tiempo_valores:
                fo_finales.append(fo_valores[-1])
                tiempos_finales.append(tiempo_valores[-1])
            
            iteracion_actual += 1
            fo_valores = []
            tiempo_valores = []
        
        elif linea.startswith("F. objetivo:"):
            valores = linea.split(":")[1].strip().strip('[]').split(',')
            fo_valores = [int(valor) for valor in valores]
        
        elif linea.startswith("Tiempo:"):
            valores = linea.split(":")[1].strip().strip('[]').split(',')
            tiempo_valores = [float(valor) for valor in valores]
    
    if fo_valores and tiempo_valores:
        fo_finales.append(fo_valores[-1])
        tiempos_finales.append(tiempo_valores[-1])
    
    promedio_fo_final = sum(fo_finales) / len(fo_finales)
    promedio_tiempo_final = sum(tiempos_finales) / len(tiempos_finales)
    
    return promedio_fo_final, promedio_tiempo_final

# Uso de la función
nombre_archivo = 'resultados_p3_C1_n30_m45.txt'
promedio_fo, promedio_tiempo = calcular_promedios(nombre_archivo)
print(f"Promedio del último valor de F.O.: {promedio_fo}")
print(f"Promedio del último valor del tiempo: {promedio_tiempo}")
