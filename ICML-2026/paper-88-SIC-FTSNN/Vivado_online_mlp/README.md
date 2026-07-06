# Spiker+-inspired Trainable MLP-SNN for Vivado

이 폴더는 **Spiker+의 계층 제어 철학** 을 참고해 만든, **Vivado 합성 가능 구조 + XSIM 시뮬레이션 중심** 의 MLP SNN 예제입니다.

중요한 점:
- **Spiker+ 원본 생성기와 동일한 코드가 아닙니다.**
- **Spiker+의 start/ready 스타일 제어, clock-driven LIF, 정적 이미지용 다단 MLP SNN** 을 참고했습니다.
- 학습은 원 논문의 offline BPTT를 그대로 HDL로 옮긴 것이 아니라, **하드웨어 친화적인 근사 online update** 로 단순화했습니다.
- 기본 구조는 **784 - Hidden - 10** fully-connected MLP SNN 입니다.
- **MNIST / Fashion-MNIST 중 선택** 해서 동일한 테스트벤치로 돌릴 수 있습니다.

## 폴더 구조

- `rtl/snn_pkg.vhd`  
  공통 타입/유틸리티 함수
- `rtl/mlp_snn_core.vhd`  
  실제 학습/추론 FSM, 가중치 메모리, rate coding, hidden/output layer 업데이트
- `rtl/mlp_snn_top.vhd`  
  상위 wrapper
- `tb/tb_mlp_snn.vhd`  
  text 파일 기반 데이터셋 로더를 포함한 시뮬레이션 testbench
- `tools/export_dataset_to_txt.py`  
  torchvision을 이용해 MNIST/F-MNIST를 text 형식으로 export
- `scripts/create_vivado_project.tcl`  
  Vivado 프로젝트 생성
- `scripts/run_xsim_batch.tcl`  
  XSIM batch 실행

## 구현 개요

### 1) 입력 인코딩
정적 28x28 grayscale 이미지를 10 step 기본값의 **deterministic rate coding** 으로 변환합니다.

- 픽셀 값이 높을수록 더 많은 time-step에서 spike 발생
- RNG가 없어 VHDL 시뮬레이션 재현성이 좋음

### 2) 뉴런 모델
clock-driven 1st-order LIF 근사입니다.

- hidden layer: subtractive reset
- output layer: 기본적으로 reset 없음
- leak은 `mem - mem / 2^LEAK_SHIFT` 형태로 구현

### 3) 학습 규칙
기본은 **sample-end online learning** 입니다.

- output layer:
  - `target spike count - observed spike count` 오차로 `W2` 업데이트
- hidden layer:
  - 고정 random feedback matrix를 이용한 간단한 direct-feedback-alignment 스타일 hidden error
  - `W1` 업데이트는 generic `G_ENABLE_W1_TRAINING` 으로 on/off 가능

즉, 정확한 BPTT 하드웨어화가 아니라 **Vivado에서 돌릴 수 있는 근사형 on-chip learning 예제** 입니다.

## 데이터셋 준비

### MNIST
```bash
python tools/export_dataset_to_txt.py --dataset mnist --outdir data --n-train 128 --n-test 32
```

### Fashion-MNIST
```bash
python tools/export_dataset_to_txt.py --dataset fmnist --outdir data --n-train 128 --n-test 32
```

생성 파일:
- `data/mnist_train_images.txt`
- `data/mnist_train_labels.txt`
- `data/mnist_test_images.txt`
- `data/mnist_test_labels.txt`

또는
- `data/fmnist_train_images.txt`
- `data/fmnist_train_labels.txt`
- `data/fmnist_test_images.txt`
- `data/fmnist_test_labels.txt`

각 image line은 `784`개의 정수(0~255), label line은 단일 정수입니다.

## Vivado 프로젝트 생성

```bash
vivado -mode batch -source scripts/create_vivado_project.tcl
```

다른 part를 쓰고 싶으면:
```bash
vivado -mode batch -source scripts/create_vivado_project.tcl -tclargs xc7a100tcsg324-1
```

## XSIM batch 실행

### MNIST
```bash
vivado -mode batch -source scripts/run_xsim_batch.tcl -tclargs mnist
```

### Fashion-MNIST
```bash
vivado -mode batch -source scripts/run_xsim_batch.tcl -tclargs fmnist
```

학습/테스트 개수 변경:
```bash
vivado -mode batch -source scripts/run_xsim_batch.tcl -tclargs mnist xc7z020clg400-1 256 64
```

## 자주 바꾸는 파라미터

`tb/tb_mlp_snn.vhd` 또는 `rtl/mlp_snn_top.vhd` / `rtl/mlp_snn_core.vhd` generic에서 조정하세요.

- `G_N_HIDDEN`
- `G_TIMESTEPS`
- `G_THRESHOLD_H`
- `G_THRESHOLD_O`
- `G_ENABLE_W1_TRAINING`
- `G_MAX_UPD`
- `G_W_MIN`, `G_W_MAX`

## 추천 시작점

처음에는 아래처럼 시작하는 것이 좋습니다.

- `G_N_HIDDEN = 32` 또는 `64`
- `G_TIMESTEPS = 10`
- `G_ENABLE_W1_TRAINING = false` 로 먼저 돌려보기
- 이후 `true` 로 바꿔 hidden update 활성화

## 한계

1. **BRAM 최적화가 되어 있지 않습니다.**  
   가중치 메모리가 inference accelerator 수준으로 최적화되어 있지 않아 자원 사용이 큽니다.

2. **simulation-first 구현입니다.**  
   Vivado 합성은 가능하도록 작성했지만, 실제 FPGA board에서 대규모 online training이 잘 돌아가도록 성능/배치 최적화한 코드는 아닙니다.

3. **정확도 보장 없음**  
   이 코드는 연구 시작점/검증용 뼈대입니다. 실제 정확도는 hidden size, threshold, learning rate clip, dataset subset 등에 크게 좌우됩니다.

4. **Spiker+ 원본과 동일한 training path가 아님**  
   원 Spiker+는 주로 offline training 후 parameter export 흐름입니다. 본 코드는 그 철학을 참고한 HDL-side 근사 online learner입니다.

## 다음에 바로 해볼 것

1. hidden layer를 `75`로 맞춰 public Spiker+ MNIST 예제와 유사하게 세팅
2. `G_ENABLE_W1_TRAINING=false` 로 먼저 smoke test
3. 동작 확인 후 `true`로 전환
4. 이후 weight storage를 1D RAM + address generator로 바꿔 BRAM inference 개선
