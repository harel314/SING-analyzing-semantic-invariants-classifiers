# Docker

Prebuilt Docker Hub images for SING development.

## CPU support (macOS/Windows/Linux without GPU)

- Image: `harel314/sing:cpu`
- Compose file: `docker/docker-compose.cpu.yml`
- Start script: `docker/up-cpu-dev.sh`

Start:

```bash
bash docker/up-cpu-dev.sh
```

Cursor attach target: `sing-dev-cpu`.

## Files

- `docker/docker-compose.gpu.yml`
- `docker/docker-compose.cpu.yml`
- `docker/up-gpu-dev.sh`
- `docker/up-cpu-dev.sh`

## GPU support

- Image: `harel314/sing:gpu-cuda12.4`
- Compose file: `docker/docker-compose.gpu.yml`
- Start script: `docker/up-gpu-dev.sh`

## Start dev container (bash)

```bash
bash docker/up-gpu-dev.sh
```

## GPU sanity check

```bash
docker compose -f docker/docker-compose.gpu.yml exec sing-dev nvidia-smi
```

## Connect with Cursor Remote Explorer

1. Start container: `bash docker/up-gpu-dev.sh`
2. In Cursor: Remote Explorer -> Containers -> attach to `sing-dev`.
3. Open `/workspace` inside the attached session.
4. Open `notebooks/01_single_image_quickstart.ipynb`.
5. Select Python kernel from the container environment.

If kernel is missing, run once inside container:

```bash
docker compose -f docker/docker-compose.gpu.yml exec sing-dev \
  python -m pip install ipykernel
```

Notebook note:
- Do not run install cells inside notebooks when using Docker images.
- Dependencies and package install are already baked into these images.

## Run SING CLI from the dev container

```bash
docker compose -f docker/docker-compose.gpu.yml exec sing-dev \
  python -m sing.cli \
    --image samples/border_collie_n02106166.jpeg \
    --model resnet \
    --seeds 42 1337 \
    --output-dir outputs/demo
```

Notes:
- First run downloads model weights from Hugging Face, so it can take longer.
- Output images and JSON are written under your local repo because the workspace is bind-mounted.
- For CPU-only flow, use `bash docker/up-cpu-dev.sh` and attach to `sing-dev-cpu`.

