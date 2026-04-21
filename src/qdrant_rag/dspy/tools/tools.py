import importlib.util
from  pathlib import Path



def load_custom_tools(path_to_tools: str, function_names:list):
    """
    Charge des fonctions personnalisées à partir d'un fichier Python donné.
    Args:
        path_to_tools (str): Chemin vers le fichier Python contenant les fonctions.
        function_names (list): Liste des noms de fonctions à charger depuis le fichier. 
    
    Returns:    
        List: Liste des fonctions chargées.
    """

    script_path = Path(path_to_tools).expanduser().resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable: {script_path}")

    spec = importlib.util.spec_from_file_location("custom_tools", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) 
    
    functions = []
    for func_name in function_names:
        if hasattr(module, func_name):
            functions.append(getattr(module, func_name))
        else:
            raise AttributeError(f"La fonction '{func_name}' n'existe pas dans le fichier {script_path}")
    
    return functions