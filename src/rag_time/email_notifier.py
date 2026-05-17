import smtplib
import os
from email.message import EmailMessage

def send_dlq_email(dlq_count: int, dlq_file_path: str, recipient_email: str):
    """
    Envoie un email d'alerte avec la DLQ en pièce jointe.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("Avertissement: SMTP_USER ou SMTP_PASSWORD non configuré. L'email ne sera pas envoyé.")
        return

    msg = EmailMessage()
    msg['Subject'] = f'Alerte: {dlq_count} tickets rejetés lors de l\'ingestion RAG'
    msg['From'] = smtp_user
    msg['To'] = recipient_email
    
    content = f"""
    Bonjour,

    Lors de la dernière ingestion de données, {dlq_count} tickets ont été rejetés car ils n'avaient pas de corps ('body') ou de réponse ('answer').
    
    Veuillez trouver la liste de ces tickets en pièce jointe (DLQ).
    
    Cordialement,
    Le pipeline d'ingestion RAG.
    """
    msg.set_content(content)

    # Ajouter la pièce jointe
    try:
        with open(dlq_file_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(dlq_file_path)
        
        msg.add_attachment(file_data, maintype='text', subtype='csv', filename=file_name)
    except FileNotFoundError:
        print(f"Erreur: Le fichier DLQ {dlq_file_path} est introuvable.")
        return

    try:
        # Envoi via SMTP avec TLS
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"Email d'alerte envoyé avec succès à {recipient_email}")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")
