import requests
import os
from dotenv import load_dotenv


def check_openrouter_credits(api_key):
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
       
        limit = data['data']['limit']
        usage = data['data']['usage']
        remaining = limit - usage
        
        print(f"Crédit Total (Limit) : ${limit:.4f}")
        print(f"Utilisation actuelle : ${usage:.4f}")
        print(f"---")
        print(f"Crédit RESTANT : ${remaining:.4f}")
    else:
        print(f"Erreur {response.status_code}: {response.text}")


load_dotenv() 
MY_API_KEY = os.getenv("OPENROUTER_API_KEY")
check_openrouter_credits(MY_API_KEY)