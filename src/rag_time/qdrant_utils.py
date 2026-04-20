from os import name
from uuid import uuid5, NAMESPACE_DNS as namespace
from qdrant_client import QdrantClient, models
from rag_time.config.settings import settings
from rag_time.data_loader import stream_tickets

client = QdrantClient(settings.qdrant_url)


def create_collection(name: str, vector_size: int):
    """Crée une collection dans Qdrant
    Args:
        name (str): Le nom de la collection à créer
        vector_size (int): La dimension des vecteurs à stocker
    """
    client.create_collection(
        collection_name=name,
        vectors_config={
            "vector": models.VectorParams(
                size=vector_size,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        },
        sparse_vectors_config={
            "sparse-vector": models.SparseVectorParams(
                modifier=models.Modifier.IDF, # Le modificateur s'applique à ce vecteur
                index=models.SparseIndexParams(
                    on_disk=True,
                )
            )
        }
    )

def upload_tickets_to_qdrant(jsonl_path: str):
    """Upload des tickets à Qdrant
    Args:        
        jsonl_path (str): Le chemin vers le fichier JSONL contenant les tickets
    """
    points = []
    for i, ticket in enumerate(stream_tickets(jsonl_path)):
        print(f"Processing ticket {i+1}", end="\r")
        for index, chunk in enumerate(ticket.chunks):
            content = f"{ticket.subject}{chunk}"
            point_id = str(uuid5(namespace, content))
            ref_id = point_id if index == 0 else ref_id
            point = models.PointStruct(
                id=point_id,
                vector={
                    "vector": ticket.embeddings[index],
                    "sparse-vector": ticket.sparse_embeddings[index]
                },
                payload={
                    "ref_id": ref_id,
                    "subject": ticket.subject if point_id == ref_id else None,
                    "body": ticket.body  if point_id == ref_id else None,
                    "answer": ticket.answer if point_id == ref_id else None,
                    "type": ticket.type if point_id == ref_id else None,
                    "queue": ticket.queue if point_id == ref_id else None,
                    "priority": ticket.priority if point_id == ref_id else None,
                    "language": ticket.language if point_id == ref_id else None,
                    "chunk": chunk
                }
            )
            points.append(point)
        
        if len(points) >= 100:
            client.upsert(collection_name=settings.collection_name, points=points)
            points = []
    if points:
        client.upsert(collection_name=settings.collection_name, points=points)
    print(f"All tickets uploaded to Qdrant: {i+1} tickets processed.")

def delete_collection(name: str):
    client.delete_collection(collection_name=name)
