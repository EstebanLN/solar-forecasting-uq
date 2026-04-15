import numpy as np


data = np.load('data_processed/GOES_v2/MCMIPF/2022/02/20220201_00_MCMIPF.npz')

print("Arrays disponibles:", list(data.files))

for nombre in data.files:
    arr = data[nombre]
    print(f"\nArray: '{nombre}'")
    print(f"  Dimensiones: {arr.shape}")
    print(f"  Tipo de datos: {arr.dtype}")
    
data.close()