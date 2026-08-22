# scheduler.py
import time
import requests
from pathlib import Path
from datetime import datetime
#import os
#import sys
#sys.path.append(os.path.abspath(".."))
from . import predict as pr
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler(timezone="Europe/Paris")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
path_to_data = PROJECT_ROOT / "data"
path_to_conso = path_to_data / "conso"
path_to_clean_conso = path_to_conso / "clean_conso"
path_to_real_time_conso = path_to_conso / "real_time_conso"

def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.rte-france.com/",
        "Accept": "application/zip, application/octet-stream, */*",
    }

    with requests.get(url, headers=headers, stream=True, timeout=30) as response:
        response.raise_for_status()

        # Vérifie que c'est bien un ZIP
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type:
            raise ValueError(f"Reçu du HTML au lieu d'un ZIP. Content-Type: {content_type}")

        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)



def download_monthly_data():
    url = "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip"
    current_year = datetime.now().year
    path = path_to_real_time_conso / f"conso_energie_{current_year}.zip"
    if path.exists():
        print(f"conso_energie_{current_year}.zip already exists.\nLets delete it.")
        path.unlink(missing_ok=True)
    assert not path.exists(), f"Error : {path.name} wasn't deleted."
    print(f"Download of conso_energie_{current_year}.zip has started")
    download_file(url, path)
    return path.exists()

def run_pipeline():
    print("\nDownloading new data")

    check = False
    while not check:
        check = download_monthly_data()

    print("The dataset has been saved successfully")

    pr.predict()

    print("The database has been updated with success")

if __name__ == "__main__":

    scheduler.add_job(
        run_pipeline,
        "cron",
        max_instances=1,
        minute="0,30,32"
    )

    scheduler.start()
    