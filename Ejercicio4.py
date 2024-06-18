#4 Multiplique dos matrices sparce de más de 1000 filas y columnas

# Función para cargar una imagen y convertirla en escala de grises
def imagen_EscalaGrises(ruta_imagen):
    image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    return image

# Función para convertir una imagen en una matriz dispersa
def imagen_a_MatrizSparce(imagen):
    # Convertir la imagen en una matriz dispersa en formato CSR
    sparse_matrix = csr_matrix(imagen)
    return sparse_matrix

# Función para redimensionar una imagen si es menor a 1000x1000
def redimencionarImagen(imagen, min_size=(1000, 1000)):
    if imagen.shape[0] < min_size[0] or imagen.shape[1] < min_size[1]:
        return cv2.resize(imagen, min_size)
    return imagen

# Rutas de las imágenes en Google Drive
ruta_imagen1 = '/content/drive/MyDrive/inf_lic_silva/leon1.jpg'
ruta_imagen2 = '/content/drive/MyDrive/inf_lic_silva/leon2.jpg'

# Cargar las imágenes en escala de grises
image1 = imagen_EscalaGrises(ruta_imagen1)
image2 = imagen_EscalaGrises(ruta_imagen2)

# Redimensionar las imágenes si es necesario
image1 = redimencionarImagen(image1)
image2 = redimencionarImagen(image2)

# Convertir las imágenes en matrices dispersas
matrizSparce1 = imagen_a_MatrizSparce(image1)
matrizSparce2 = imagen_a_MatrizSparce(image2)

# Verificar el tamaño de las matrices dispersas
print(f"Tamaño de la matriz dispersa 1: {matrizSparce1.shape}")
print(f"Tamaño de la matriz dispersa 2: {matrizSparce2.shape}")

# Asegurarse de que las matrices tengan el mismo tamaño para la multiplicación
if matrizSparce1.shape != matrizSparce2.shape:
    print("Las matrices no tienen el mismo tamaño. Redimensionando la segunda matriz para que coincida con la primera.")
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    matrizSparce2 = imagen_a_MatrizSparce(image2_resized)

# Realizar la multiplicación de las dos matrices dispersas
multiplicacionMatrices = matrizSparce1.dot(matrizSparce2)

# Mostrar información sobre la matriz resultante
print("Matriz dispersa resultante de la multiplicación:")
print(multiplicacionMatrices)

# Convertir la matriz resultante a una densa para visualizarla
matrizProductodensa = multiplicacionMatrices.toarray() #opcional

# Mostrar las imágenes originales y la matriz resultante
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('imagen 1')
plt.imshow(image1, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('imagen 2')
plt.imshow(image2, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Producto de Matrices')
plt.imshow(matrizProductodensa, cmap='gray')

plt.show()