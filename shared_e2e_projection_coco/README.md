# Shared-E2E Projection V2 — COCO Docker Trainer

이 폴더는 GT-Super 루트의 기존 Python/노트북 파일을 수정하거나 import하지 않는 독립 학습 패키지입니다. 학습 가능한 모델은 `shared_e2e_rep_projected` 하나뿐입니다.

## 포함된 기능

- COCO 2017 `train2017` / `val2017` 원본 split 사용
- `SenseTime/deformable-detr`의 COCO 91-slot classifier와 sparse category ID 유지
- Shared-E2E V2 representation gradient projection
- 단일 GPU와 `torchrun` 기반 단일 서버 다중 GPU DDP
- AMP, epoch 검증, CSV 기록, 자동 재시작 checkpoint
- COCO 데이터·checkpoint·Hugging Face cache를 Docker 이미지 밖에 보존

## 필요한 COCO 구조

```text
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Docker 이미지에는 데이터를 넣지 않습니다. 서버의 COCO 폴더를 read-only volume으로 연결합니다.

## 1. GPU 확인

GPU 서버에 NVIDIA driver, Docker, NVIDIA Container Toolkit이 설치되어 있어야 합니다.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

이 명령에서 GPU가 보이지 않으면 학습 컨테이너를 실행하기 전에 서버의 NVIDIA Container Toolkit 설정을 해결해야 합니다.

## 2. 이미지 빌드

GT-Super 루트에서 실행합니다.

```bash
docker build \
  -t gt-super-projection-v2:latest \
  ./shared_e2e_projection_coco
```

서버 driver와 기본 CUDA 12.8 이미지가 호환되지 않으면 빌드 인자로 다른 공식 PyTorch CUDA 이미지를 지정합니다.

```bash
docker build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:<서버와-호환되는-태그> \
  -t gt-super-projection-v2:latest \
  ./shared_e2e_projection_coco
```

## 3. 먼저 작은 smoke test

COCO 32장/16장으로 데이터, 모델 forward, projection backward, 평가, checkpoint 저장까지 확인합니다.

```bash
docker run --rm \
  --gpus '"device=0"' \
  --shm-size=8g \
  -v /srv/gt-super/data/coco:/workspace/data/coco:ro \
  -v /srv/gt-super/projection-v2-artifacts:/workspace/artifacts \
  -v /srv/gt-super/hf-cache:/workspace/hf-cache \
  gt-super-projection-v2:latest \
  python train.py \
    --data-root /workspace/data/coco \
    --output-root /workspace/artifacts \
    --epochs 1 \
    --batch-size 1 \
    --num-workers 2 \
    --train-limit 32 \
    --val-limit 16 \
    --seed 42
```

## 4. 단일 GPU 전체 학습

```bash
docker run -d \
  --name projection-v2-seed42 \
  --gpus '"device=0"' \
  --shm-size=16g \
  -v /srv/gt-super/data/coco:/workspace/data/coco:ro \
  -v /srv/gt-super/projection-v2-artifacts:/workspace/artifacts \
  -v /srv/gt-super/hf-cache:/workspace/hf-cache \
  gt-super-projection-v2:latest \
  python train.py \
    --data-root /workspace/data/coco \
    --output-root /workspace/artifacts \
    --epochs 50 \
    --batch-size 2 \
    --num-workers 8 \
    --seed 42 \
    --resume auto
```

`--batch-size`는 전체 batch가 아니라 GPU 한 장당 batch입니다. GPU 메모리가 부족하면 먼저 1로 낮춥니다.

## 5. 단일 서버 다중 GPU 학습

GPU 4장을 하나의 학습에 사용하는 예입니다.

```bash
docker run -d \
  --name projection-v2-ddp-seed42 \
  --gpus all \
  --shm-size=32g \
  -v /srv/gt-super/data/coco:/workspace/data/coco:ro \
  -v /srv/gt-super/projection-v2-artifacts:/workspace/artifacts \
  -v /srv/gt-super/hf-cache:/workspace/hf-cache \
  gt-super-projection-v2:latest \
  torchrun --standalone --nproc_per_node=4 train.py \
    --data-root /workspace/data/coco \
    --output-root /workspace/artifacts \
    --epochs 50 \
    --batch-size 2 \
    --num-workers 8 \
    --seed 42 \
    --resume auto
```

이 예의 global batch는 `2 × 4 = 8`입니다. 현재 설정은 learning rate를 자동으로 늘리지 않습니다. baseline과 공정하게 비교하려면 global batch와 learning rate 정책을 실험 간 동일하게 관리해야 합니다.

## Docker Compose

```bash
cd shared_e2e_projection_coco
cp .env.example .env
```

`.env`에서 서버 경로와 GPU 개수에 맞춰 `COCO_ROOT`, `OUTPUT_ROOT`, `HF_CACHE`, `NPROC_PER_NODE`를 수정한 뒤 실행합니다.

```bash
docker compose up --build -d
docker compose logs -f train
```

`NPROC_PER_NODE`는 컨테이너에 보이는 GPU 수를 넘으면 안 됩니다.

## 결과 위치

seed 42 기준으로 다음 위치에 저장됩니다.

```text
OUTPUT_ROOT/
└── shared_e2e_rep_projected/
    └── seed_42/
        ├── history.csv
        ├── projection_gradients.csv
        └── checkpoints/
            ├── latest.pt
            ├── epoch_005.pt
            └── ...
```

`latest.pt`는 매 epoch 갱신됩니다. `--save-every 5`이면 장기 보관 snapshot은 5 epoch마다 생성됩니다.

## 중단 후 재시작

동일한 명령에 `--resume auto`를 사용하면 `latest.pt`가 있을 때 이어서 학습하고, 없으면 새 학습을 시작합니다.

```bash
docker start projection-v2-seed42
docker logs -f projection-v2-seed42
```

컨테이너를 새로 만들 때도 같은 `OUTPUT_ROOT` volume과 동일한 학습 설정을 사용해야 합니다. DDP checkpoint는 저장할 때와 같은 GPU 프로세스 수로 재시작하도록 검사합니다.

## 주요 옵션

```bash
docker run --rm gt-super-projection-v2:latest python train.py --help
```

- `--train-limit`, `--val-limit`: smoke test용 고정 subset
- `--skip-initial-eval`: pretrained epoch-0 평가 생략
- `--offline`: 네트워크를 사용하지 않고 기존 Hugging Face cache만 사용
- `--no-amp`: AMP 비활성화
- `--no-disable-custom-kernels`: custom deformable-attention kernel 허용
- `--gradient-log-every`: projection step 상세 CSV 기록 주기

기본값은 호환성을 위해 custom kernel을 비활성화합니다. custom kernel이 준비된 서버 이미지에서만 `--no-disable-custom-kernels`를 사용하고 smoke test를 먼저 통과시킵니다.
