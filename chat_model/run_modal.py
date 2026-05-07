import modal

# Define the persistent volume
volume = modal.Volume.from_name("nanochat-weights", create_if_missing=True)

# Attach local files to the iamge
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

# Define the remote functon
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    # Mount the output volume
    volumes={"/root/output": volume}
)
def train_remote(load_data, init_tokenizer):
    from train import main

    main(load_data=load_data, init_tokenizer=init_tokenizer)

    volume.commit()

@app.local_entrypoint()
def run():

    train_remote.remote(load_data=False, init_tokenizer=False)
