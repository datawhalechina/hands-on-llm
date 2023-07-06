import argparse

import pnlp

from docqa.app.recaller import Recaller
from docqa.dd_model import Doc
from docqa.config import recaller_config


def main():
    parser = argparse.ArgumentParser(description="Index docs to Qdrant.")
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        help="Input docqa file")
    parser.add_argument(
        "-n",
        "--number",
        required=False,
        type=int,
        default=100,
        help="Index number of docs")
    args = parser.parse_args()

    lst = pnlp.read_json(args.filepath)
    docs = []
    has = set()
    count = 0
    for im in lst:
        if count >= args.number:
            break
        cxt = im["text"]
        uid = pnlp.generate_uuid(cxt)
        if uid in has:
            continue
        doc = Doc(uid, cxt)
        docs.append(doc)
        has.add(uid)
        count += 1

    print(f"total {len(docs)} docs will be indexed...")

    recaller_ins = Recaller(
        recaller_config.host,
        recaller_config.port
    )
    recaller_ins.vec_index_ins.create_index(
        recaller_config.collection, recaller_config.dim
    )
    recaller_ins.add_docs(
        recaller_config.collection,
        docs,
        recaller_config.index_batch_size
    )


if __name__ == "__main__":
    main()
