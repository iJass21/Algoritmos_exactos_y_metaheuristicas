# Definir el nombre del archivo de entrada
archivo_entrada = 'item2-hc-am-ra-C1.txt'

# Inicializar las variables para almacenar los valores de F obj y T
f_obj_values = []
t_values = []

# Leer el archivo y extraer los valores de F obj y T
with open(archivo_entrada, 'r') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        if 'F obj' in line:
            f_obj_line = line.split(': ')[1].strip()
            f_obj_values.append(int(f_obj_line))
        elif 'T' in line:
            t_line = line.split(': ')[1].strip().split()[0]
            t_values.append(float(t_line))

# Calcular el promedio de F obj y T
promedio_f_obj = sum(f_obj_values) / len(f_obj_values) if len(f_obj_values) > 0 else 0
promedio_t = sum(t_values) / len(t_values) if len(t_values) > 0 else 0

with open("prom-C1-item2-hc.txt", 'a') as file:
    file.write(f'Prom f obj: {promedio_f_obj}\n')
    file.write(f'Prom t: {promedio_t}\n')
    file.write('-------------------------------------------------------------------------------\n')
