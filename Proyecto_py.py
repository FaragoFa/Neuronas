# Librerías
import numpy as np
import matplotlib.pyplot as plt
import WholeBrain.Observables.phFCD as phFCD
import Tests.DecoEtAl2014.fig2 as fig_Deco
import WholeBrain.Integrators.EulerMaruyama as EulerMaruyama
import os

fig_Deco.BalanceFIC.integrator = EulerMaruyama

# Ruta al archivo de texto con los datos de los sujetos 25x25
ruta_archivo_25 = 'Datos/Datasets/netmats2_25.txt'

# Ruta al archivo de texto con los datos de los sujetos 200x200
ruta_archivo_200 = 'Datos/Datasets/netmats2_200.txt'

# Ruta al archivo de texto con los time series
ruta_archivo_ts = 'Datos/Datasets/100206.txt'

# Cargar los datos de los sujetos desde los archivos de texto
datos_sujetos_25 = np.loadtxt(ruta_archivo_25)
datos_sujetos_200 = np.loadtxt(ruta_archivo_200)

# Reshape para crear matriz 3D
matrices_por_sujeto_25 = datos_sujetos_25.reshape((1003, 25, 25))
matrices_por_sujeto_200 = datos_sujetos_200.reshape((1003, 200, 200))

# Calcular la matriz de conectividad promedio de todos los sujetos
matriz_conectividad_promedio = np.mean(matrices_por_sujeto_25, axis=0)
matriz_conectividad_promedio = matriz_conectividad_promedio/matriz_conectividad_promedio.max()
matriz_conectividad_promedio = matriz_conectividad_promedio*0.1
# Visualizar la matriz de conectividad promedio
plt.imshow(matriz_conectividad_promedio, cmap='coolwarm', interpolation='none')
plt.colorbar()  # Añadir la barra de color
plt.title('Matriz de Conectividad Promedio de Todos los Sujetos')
plt.xlabel('Regiones del Cerebro')
plt.ylabel('Regiones del Cerebro')

# Mostrar la visualización
plt.show()


# DMF
folder_path = 'Datos/Results/Results_test2'

fig_Deco.plotMaxFrecForAllWe(matriz_conectividad_promedio, fileName=os.path.join(folder_path, 'results_{}.txt'))