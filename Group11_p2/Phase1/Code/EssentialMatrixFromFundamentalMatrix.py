import numpy as np

def estimateEssentialMatrix(K , F):
    E = K.T @ F @ K
    return E