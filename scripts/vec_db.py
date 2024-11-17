from utils import upsert_embeddings, prep_doc_for_upsert_document,parse_document

if __name__ == "__main__":
    import argparse
    import chromadb
    
    client_ = chromadb.PersistentClient(path="./embeds")
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',  help="Path to file")
    parser.add_argument('-c', '--collection', help='name of collection')

    args = parser.parse_args()
    upsert_embeddings(
        client=client_,
        document=prep_doc_for_upsert_document(
                    parse_document(args.file)),
        collection_name=args.collection
        )
