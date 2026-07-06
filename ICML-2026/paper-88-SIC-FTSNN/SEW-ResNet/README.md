# SEW-ResNet + Learnable Fragmentation Add-on

이 패키지는 두 논문 아이디어를 한 코드베이스로 묶습니다.

1. **SEW-ResNet (NeurIPS 2021)**
   - SEW residual block
   - `ADD / AND / IAND` spike-element-wise connection
   - downsample shortcut에 `SN` 포함
   - paper-consistent zero-initialization

2. **Learnable Fragmentation (uploaded ICML paper)**
   - `(h_k, v_k, r_k)`로 parameterize된 `T-1`개의 division lines
   - hard mask forward + straight-through soft gradient
   - candidate set 위에서 Gumbel-Softmax로 fragment 수 선택
   - temporal resampling to `T_max`
   - selector-weighted mixed fragment training
   - balance loss
   - entropy-based decoding

## Files

- `sew_resnet_paper.py`
  - SEW-ResNet 구현
- `learnable_fragmentation_addon.py`
  - fixed-T / dynamic-T learnable fragmentation 구현
- `sew_resnet_fragmentation_wrapper.py`
  - fragmentation -> optional encoder -> SEW-ResNet -> entropy decoder 를 한 번에 묶는 wrapper
- `examples/example_train_sew_fragmentation.py`
  - CIFAR-style usage example
- `examples/example_forward_only.py`
  - shape / API 확인용 간단 예시

## Core idea of the integration

당신이 올려준 MLP/VGG/ResNet 예시 코드들의 공통 패턴은 아래와 같습니다.

1. 입력 `x`를 fragmentation module에 넣어 `[B, T, C, H, W]` sequence 생성
2. 각 time step `t`마다 `x[:, t]`를 SNN backbone에 넣음
3. `[B, T, K]` logits를 모음
4. entropy-based decoding 또는 mean decoding 수행
5. main loss + balance loss로 학습

이 패키지의 `FragmentedSEWResNet`가 정확히 이 패턴을 SEW-ResNet에 맞춰 캡슐화합니다.

## Minimal usage

```python
from sew_resnet_fragmentation_wrapper import build_fragmented_sew_resnet

model = build_fragmented_sew_resnet(
    depth=18,
    num_classes=10,
    image_size=(32, 32),
    stem='cifar',
    cnf='ADD',
    dynamic_candidates=(2, 4, 8),
    init_direction='horizontal',
    decoder='entropy',
    entropy_gamma=1.0,
)

out = model(images, return_aux=True)
logits = out.logits
balance_loss = out.fragmentation.balance_loss
selector_probs = out.fragmentation.selector_probs
selected_t = out.fragmentation.selected_t
```

## Training loss example

```python
criterion = torch.nn.CrossEntropyLoss()
out = model(images, return_aux=True)
loss = criterion(out.logits, targets) + 0.01 * out.fragmentation.balance_loss
loss.backward()
optimizer.step()
```

## Notes

- `IAND`는 논문의 Table 1 정의인 `g(A, S) = (1 - A) * S`를 기준으로 구현했습니다.
- `zero_init_residual=True`일 때
  - `ADD`, `IAND`: residual branch의 마지막 BN을 0으로 초기화해서 `A=0`
  - `AND`: 마지막 BN bias를 `v_threshold`로 초기화해서 `A=1`
- `use_expected_poisson=True`를 켜면 differentiable expected-rate encoder를 wrapper가 붙입니다.
  - learnable fragmentation gradient를 살리기 위해 기본 샘플링 대신 expected mode를 선택할 수 있게 했습니다.
- exact ImageNet reproduction을 목표로 하면 원저자 repo / SpikingJelly old commit 설정을 따라가는 것이 가장 안전합니다.
