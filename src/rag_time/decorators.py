from time import time

def chrono(func):
    def wrapper(*args, **kwargs):
        elapsed = time()
        resultat = func(*args, **kwargs)
        elapsed = time() - elapsed 
        return resultat, elapsed
    return wrapper