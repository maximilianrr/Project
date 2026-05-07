import modal

# Define the persistent volume
volume = modal.Volume.from_name("nanochat-weights", create_if_missing=True)

# 1. Attach your local files directly to the Image (The new Modal 1.0 way)
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("./models", remote_path="/root/models")
    .add_local_dir("../../nanochat/nanochat", remote_path="/root/nanochat")
    .add_local_dir("./utils", remote_path="/root/utils")
    .add_local_file("./config.py", remote_path="/root/config.py")
    .add_local_file("./train.py", remote_path="/root/train.py")
    .add_local_dir("./data", remote_path="/root/data")
)

app = modal.App("nanochat-training")

# 2. Define the remote function without the 'mounts' parameter
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    # Mount the volume to the 'data' directory to save your weights persistently
    volumes={"/root/output": volume}
)
def train_remote(load_data, init_tokenizer):
    # Import inside the function so it uses the container's torch
    from train import main

    main(load_data=load_data, init_tokenizer=init_tokenizer)
    
    # Crucial: Commit the changes to the volume so they are saved
    volume.commit()

# 3. Local entry point to trigger the remote run
@app.local_entrypoint()
def run():

    train_remote.remote(load_data=False, init_tokenizer=False)