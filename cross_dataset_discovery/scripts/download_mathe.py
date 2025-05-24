from huggingface_hub import snapshot_download

local_dir = "cross_dataset_discovery/assets/mathe/"

snapshot_download(
    repo_id="DARELab/MathE",
    local_dir=local_dir,
    repo_type="dataset",
)

print(f"Dataset downloaded to: {local_dir}")
