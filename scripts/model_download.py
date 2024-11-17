from huggingface_hub import snapshot_download
from os.path import realpath

GGUF_PATH = realpath("../models")


def get_gguf(model_path=None, filename=None):
    """Downloads LLAMA-3.2 3B from huggingface"""

    snapshot_download(repo_id=model_path,)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog="gguf_download from huggingface", 
                                     description="Downloads model from huggingface given a filename and modelpath")

    parser.add_argument('-m', '--modelpath', type=str, help="MODEL PATH e.g TheBloke/Llama-2-7B-GGUF")
    parser.add_argument('-f', '--filename', type=str, help="Filename e.g llama-2-7b.Q4_K_M.gguf")

    args = parser.parse_args()
    get_gguf(model_path=args.modelpath, filename=args.filename)
