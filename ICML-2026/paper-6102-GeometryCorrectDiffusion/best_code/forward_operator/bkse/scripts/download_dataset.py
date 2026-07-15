import argparse
import os
import os.path as osp

import requests


def download_file_from_google_drive(file_id, destination):
    os.makedirs(osp.dirname(destination), exist_ok=True)
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    dataset_ids = {
        "GOPRO_Large": "1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2",
        "train_sharp": "1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-",
        "train_blur": "1Be2cgzuuXibcqAuJekDgvHq4MLYkCgR8",
        "val_sharp": "1MGeObVQ1-Z29f-myDP7-8c3u0_xECKXq",
        "val_blur": "1N8z2yD0GDWmh6U4d4EADERtcUgDzGrHx",
        "test_blur": "1dr0--ZBKqr4P1M8lek6JKD1Vd6bhhrZT",
    }

    parser = argparse.ArgumentParser(
        description="Download REDS dataset from google drive to current folder", allow_abbrev=False
    )

    parser.add_argument("--REDS_train_sharp", action="store_true", help="download REDS train_sharp.zip")
    parser.add_argument("--REDS_train_blur", action="store_true", help="download REDS train_blur.zip")
    parser.add_argument("--REDS_val_sharp", action="store_true", help="download REDS val_sharp.zip")
    parser.add_argument("--REDS_val_blur", action="store_true", help="download REDS val_blur.zip")
    parser.add_argument("--GOPRO", action="store_true", help="download GOPRO_Large.zip")

    args = parser.parse_args()

    if args.REDS_train_sharp:
        download_file_from_google_drive(dataset_ids["train_sharp"], "REDS/train_sharp.zip")
    if args.REDS_train_blur:
        download_file_from_google_drive(dataset_ids["train_blur"], "REDS/train_blur.zip")
    if args.REDS_val_sharp:
        download_file_from_google_drive(dataset_ids["val_sharp"], "REDS/val_sharp.zip")
    if args.REDS_val_blur:
        download_file_from_google_drive(dataset_ids["val_blur"], "REDS/val_blur.zip")
    if args.GOPRO:
        download_file_from_google_drive(dataset_ids["GOPRO_Large"], "GOPRO/GOPRO.zip")
