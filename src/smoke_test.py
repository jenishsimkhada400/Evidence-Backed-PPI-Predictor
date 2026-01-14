from pathlib import Path
import yaml

def main():
    cfg_path = Path("configs/base.yaml")
    assert cfg_path.exists(), "configs/base.yaml not found"

    cfg = yaml.safe_load(cfg_path.read_text())
    print("Loaded config:", cfg["project"]["name"])

    # Check core dirs exist
    for p in ["data/raw", "data/processed", "data/embeddings", "reports"]:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("Smoke test OK ")

if __name__ == "__main__":
    main()
