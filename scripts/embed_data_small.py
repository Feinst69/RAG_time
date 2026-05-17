import argparse
import os
import fastembed
from tqdm import tqdm
from rag_time.embeddings import TextSplitter, SparseEmbeddingsGenerator
from rag_time.data_loader import DataLoader, Ticket
from rag_time.config.settings import settings
from rag_time.decorators import chrono
from rag_time.email_notifier import send_dlq_email

class SmallEmbeddingsGenerator:
    """Générateur d'embeddings utilisant un petit modèle HuggingFace via fastembed."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.batch_size = batch_size
        self.model = fastembed.TextEmbedding(model_name=model_name)

    def generate_embeddings(self, sentences: list[str]) -> list:
        # fastembed retourne un générateur numpy arrays, on le convertit en liste
        return list(self.model.embed(sentences, batch_size=self.batch_size))

@chrono
def handle_ticket(line, splitter, generator, sparse_generator):
    ticket = Ticket(
                subject=line.subject,
                body=line.body,
                answer=line.answer,
                type=line.type,
                queue=line.queue,
                priority=line.priority,
                language=line.language,
                chunks=splitter.split_text(f"{line.subject}\n{line.body}" if line.subject else line.body)
            )
    ticket.embeddings = generator.generate_embeddings(ticket.chunks)
    ticket.sparse_embeddings = sparse_generator.generate_vector(ticket.chunks)
    return ticket

def build_rag_data_small(input_path: str, output_path: str, dlq_path: str = "data/dlq.csv"):
    print(f"Loading dataset from {input_path}...")
    df, dlq_df = DataLoader(input_path).load()
    
    # Prendre seulement 30% du dataset
    original_len = len(df)
    df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    df_len = len(df)
    print(f"Sampled 30% of the dataset: {df_len} tickets (out of {original_len}).")

    if not dlq_df.empty:
        print(f"Found {len(dlq_df)} invalid tickets. Saving to {dlq_path} and sending alert...")
        os.makedirs(os.path.dirname(dlq_path), exist_ok=True)
        dlq_df.to_csv(dlq_path, index=False)
        recipient = os.getenv("ALERT_EMAIL_TO", "admin@example.com")
        send_dlq_email(len(dlq_df), dlq_path, recipient)

    print("Loading Small HuggingFace Embeddings model (sentence-transformers/all-MiniLM-L6-v2)...")
    generator = SmallEmbeddingsGenerator()
    sparse_generator = SparseEmbeddingsGenerator(settings.sparse_model)
    
    print("Loading Text Splitter...")
    splitter = TextSplitter(chunk_size=settings.chunk_size)
    
    with open(output_path, "w") as f:
        current_index = 0
        elapsed = 0.0
        print(f"Processing {df_len} tickets")
        for line in tqdm(df.itertuples(), total=df_len, desc="Embedding tickets", unit="ticket"):
            current_index += 1
            try:
                ticket, line_elapsed = handle_ticket(line, splitter, generator, sparse_generator)
                ticket.time = line_elapsed
                elapsed += line_elapsed
                f.write(ticket.model_dump_json() + "\n")
            except Exception as e:
                print(f"\nError processing ticket {current_index}: {e}")
                exit(1)
    print(f"\nFinished processing {df_len} tickets in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Build RAG data from dataset using a small HF model (30% data)")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to the output JSONL file")
    args = parser.parse_args()
    build_rag_data_small(args.input, args.output)

if __name__ == "__main__":
    main()
