import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm


def _to_pare_instances(query_jsonl_path: str, rel_default: str = "NA") -> Tuple[str, int]:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    n = 0
    with open(query_jsonl_path, "r") as f_in, tmp as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d["text"]
            hpos = d["hpos"]
            tpos = d["tpos"]

            hid = str(i * 2)
            tid = str(i * 2 + 1)

            inst = {
                "text": text,
                "relation": rel_default,
                "h": {"pos": hpos, "id": hid, "name": text[hpos[0] : hpos[1]]},
                "t": {"pos": tpos, "id": tid, "name": text[tpos[0] : tpos[1]]},
            }
            f_out.write(json.dumps(inst) + "\n")
            n += 1

    return tmp.name, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--pretrain_path", default="bert-base-uncased")
    parser.add_argument(
        "--ckpt",
        default="./PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar",
    )
    parser.add_argument(
        "--rel2id_file",
        default="./PARE/benchmark/nyt10/nyt10_rel2id.json",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_entity", action="store_true")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    pare_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PARE"))
    sys.path.insert(0, pare_path)

    import encoder
    import framework
    import model1
    import model2

    with open(args.rel2id_file, "r") as f:
        rel2id: Dict[str, int] = json.load(f)

    passage_encoder = encoder.PassageEncoder(
        pretrain_path=args.pretrain_path,
        batch_size=args.batch_size,
        mask_entity=args.mask_entity,
    )
    # Use model2 for NYT, model1 for others (as in original codebase)
    if "nyt" in args.query_jsonl.lower():
        model_ = model2.PassageAttention(passage_encoder, len(rel2id), rel2id)
    else:
        model_ = model1.PassageAttention(passage_encoder, len(rel2id), rel2id)

    tmp_pare_path, n_queries = _to_pare_instances(args.query_jsonl)

    try:
        from framework.data_loader import PassageRELoader

        test_loader = PassageRELoader(
            path=tmp_pare_path,
            rel2id=model_.rel2id,
            tokenizer=model_.passage_encoder.tokenize,
            batch_size=args.batch_size,
            shuffle=False,
        )

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        devices = [device.index] if device.type == "cuda" and device.index is not None else [0]

        framework_ = framework.PassageRE(
            model=model_,
            train_path=None,
            val_path=None,
            test_path=tmp_pare_path,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            max_epoch=1,
            lr=2e-5,
            weight_decay=1e-5,
            opt="adamw",
            warmup_step=0,
            devices=devices,
        )
        framework_.test_loader = test_loader

        ckpt_obj = torch.load(args.ckpt, map_location=device)
        framework_.model.module.load_state_dict(ckpt_obj["state_dict"])

        framework_.model.eval()

        id2rel = {v: k for k, v in rel2id.items()}
        na_id = rel2id.get("NA", None)

        os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
        with open(args.out_jsonl, "w") as f_out, torch.no_grad():
            for data in tqdm(framework_.test_loader, desc="PARE scoring"):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].to(framework_.device)
                        except Exception:
                            pass

                bag_name = data[1]
                token, mask = data[2].squeeze(1), data[3].squeeze(1)

                # PassageAttention model takes (token, mask)
                scores = framework_.model(token, mask)

                for b in range(scores.shape[0]):
                    hid, tid = bag_name[b][:2]
                    for relid in range(scores.shape[1]):
                        if na_id is not None and relid == na_id:
                            continue
                        rel = id2rel[relid]
                        if rel == "NA":
                            continue
                        f_out.write(
                            json.dumps(
                                {
                                    "entpair": [hid, tid],
                                    "relation": rel,
                                    "score": float(scores[b][relid].item()),
                                }
                            )
                            + "\n"
                        )

    finally:
        try:
            os.unlink(tmp_pare_path)
        except Exception:
            pass

    print(f"Wrote candidate relation scores for {n_queries} queries to {args.out_jsonl}")


if __name__ == "__main__":
    main()
