import argparse
import os
import shutil
import sys
import urllib.request

NYT10M_BASE_URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10m"
NYT10M_FILES = [
    "nyt10m_rel2id.json",
    "nyt10m_train.txt",
    "nyt10m_val.txt",
    "nyt10m_test.txt",
]


def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    tmp_path = dst_path + ".tmp"
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    def reporthook(blocknum, blocksize, totalsize):
        if totalsize <= 0:
            return
        downloaded = blocknum * blocksize
        pct = min(100.0, 100.0 * downloaded / totalsize)
        sys.stderr.write(f"\rDownloading {os.path.basename(dst_path)}: {pct:5.1f}%")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
    sys.stderr.write("\n")
    sys.stderr.flush()

    if os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"Downloaded empty file from {url}")

    os.replace(tmp_path, dst_path)


def _symlink_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYT10m benchmark files (DSRE/OpenNRE) into a local datasets folder")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "datasets", "nyt10m"),
        help="Output directory (default: <repo>/datasets/nyt10m)",
    )
    parser.add_argument(
        "--make_hydre_aliases",
        action="store_true",
        help="Create HYDRE-expected alias filenames (e.g. nyt10m_train_opennre_clean_hfmre256.jsonl -> nyt10m_train.txt)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.normpath(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for name in NYT10M_FILES:
        url = f"{NYT10M_BASE_URL}/{name}"
        dst = os.path.join(out_dir, name)
        _download(url, dst)

    if args.make_hydre_aliases:
        train_txt = os.path.join(out_dir, "nyt10m_train.txt")
        val_txt = os.path.join(out_dir, "nyt10m_val.txt")
        test_txt = os.path.join(out_dir, "nyt10m_test.txt")

        _symlink_or_copy(train_txt, os.path.join(out_dir, "nyt10m_train_opennre_clean_hfmre256.jsonl"))
        _symlink_or_copy(val_txt, os.path.join(out_dir, "nyt10m_val_opennre_clean_hfmre256.jsonl"))
        _symlink_or_copy(test_txt, os.path.join(out_dir, "nyt10m_test_opennre_clean_hfmre256.jsonl"))

    print("NYT10m download complete:")
    for name in NYT10M_FILES:
        p = os.path.join(out_dir, name)
        print(f"- {p} ({os.path.getsize(p)} bytes)")

    if args.make_hydre_aliases:
        print("HYDRE aliases created (symlink or copy):")
        print(f"- {os.path.join(out_dir, 'nyt10m_train_opennre_clean_hfmre256.jsonl')}")

    print("\nNext steps:")
    print(f"- Point HYDRE scripts to --train_path {os.path.join(out_dir, 'nyt10m_train.txt')} (or the alias if you created it)")
    print(f"- rel2id is at {os.path.join(out_dir, 'nyt10m_rel2id.json')}")


if __name__ == "__main__":
    main()
