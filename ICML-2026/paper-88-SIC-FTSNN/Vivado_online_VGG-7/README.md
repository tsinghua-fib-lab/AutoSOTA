# VGG-7 CIFAR-10 trainer scaffold for Vivado

## What this is

This project is a **Vivado-oriented VHDL scaffold** for **single-sample training and inference** on a **VGG-7-style CNN** for **CIFAR-10**.

It is built in the same spirit as the earlier Spiker+-inspired MLP artifact:
- simple top-level start/ready/done control
- deterministic weight initialization
- a hardware-friendly update rule instead of exact floating-point backpropagation
- file-driven datasets for easy XSIM testing

## Important scope

This is **not** a faithful reproduction of:
- the original **Spiker+** flow, which is aimed at **SNN inference generation**
- the exact **NITI** integer-only optimizer
- exact gradient backpropagation for VGG-7

Instead, it provides a **simulation-first behavioral VHDL training scaffold** with:
- VGG-7 forward path:
  - Conv3x3
  - Conv3x3
  - MaxPool
  - Conv3x3
  - Conv3x3
  - MaxPool
  - Conv3x3
  - Conv3x3
  - MaxPool
  - FC(10)
- direct-feedback-style channel teaching signals
- sign-based local weight updates

## Default network width

The default profile is intentionally reduced to keep Vivado XSIM manageable:

- 8
- 8
- 16
- 16
- 32
- 32

So the default architecture is still **7 weight layers**, but it is **channel-reduced**.

If you want the exact `VGG-small-7` widths commonly used in CIFAR-10 integer-training papers, edit these constants in `rtl/vgg7_pkg.vhd`:

```vhdl
constant C_C1 : natural := 128;
constant C_C2 : natural := 128;
constant C_C3 : natural := 256;
constant C_C4 : natural := 256;
constant C_C5 : natural := 512;
constant C_C6 : natural := 512;
```

Be aware that this dramatically increases simulation time and synthesis cost.

## Files

### RTL

- `rtl/vgg7_pkg.vhd`
  - global constants
  - index helpers
  - numeric clipping helpers
  - deterministic weight initialization
  - deterministic feedback initialization
  - pack/unpack helpers for file-driven RGB input

- `rtl/vgg7_cifar10_train_top.vhd`
  - full VGG-7-style forward controller
  - direct-feedback teaching signal generation
  - sign-based classifier update
  - optional all-layer channelwise convolution updates

### Testbench

- `tb/tb_vgg7_cifar10.vhd`
  - text-file driven CIFAR-10 training/testing loop
  - reports per-sample predictions and aggregate test hits

### Dataset export script

- `scripts/export_cifar10_binary_to_txt.py`
  - reads the official CIFAR-10 binary files
  - writes `cifar10_train.txt` and `cifar10_test.txt`

### Vivado TCL

- `vivado/run_sim.tcl`
- `vivado/run_synth.tcl`

## CIFAR-10 text format expected by the testbench

Each line is:

```text
<label> <p0> <p1> ... <p3071>
```

where the pixel order is the official CIFAR-10 binary order:

- first 1024 bytes: red
- next 1024 bytes: green
- last 1024 bytes: blue

## Export dataset files

From `scripts/`:

```bash
python export_cifar10_binary_to_txt.py \
  --binary-root /path/to/cifar-10-batches-bin \
  --outdir ../data \
  --train-samples 1024 \
  --test-samples 256
```

Or, using the official tarball directly:

```bash
python export_cifar10_binary_to_txt.py \
  --tar /path/to/cifar-10-binary.tar.gz \
  --outdir ../data \
  --train-samples 1024 \
  --test-samples 256
```

The generated files are:

- `data/cifar10_train.txt`
- `data/cifar10_test.txt`

## Run simulation in Vivado

From `vivado/`:

```bash
vivado -mode batch -source run_sim.tcl
```

Or override train/test counts and epochs:

```bash
vivado -mode batch -source run_sim.tcl -tclargs 256 128 2
```

## Run synthesis

From `vivado/`:

```bash
vivado -mode batch -source run_synth.tcl
```

## Training rule used here

The output layer computes a sign teaching signal from a target score:

\[
e_k = \mathrm{sign}(t_k - s_k)
\]

where \(s_k\) is the class score and \(t_k\) is `C_TARGET_SCORE` for the true class and `0` otherwise.

The fully connected layer update is:

\[
\Delta W^{(7)}_{k,i} = \eta_{\mathrm{fc}} \, \mathrm{sign}(a_i) \, e_k
\]

Each convolutional block receives a fixed random-feedback teaching vector:

\[
h^{(\ell)} = \mathrm{sign}(B^{(\ell)} e)
\]

A coarse channelwise update is then applied:

\[
\Delta W^{(\ell)}_{oc,ic,ky,kx}
= \eta_{\mathrm{conv}}
\cdot \mathrm{sign}(u_{ic})
\cdot \mathrm{sign}(v_{oc})
\cdot h^{(\ell)}_{oc}
\]

where:
- \(u_{ic}\) is the spatial sum of the input channel
- \(v_{oc}\) is the spatial sum of the output channel

This is deliberately simple and hardware-friendly. It is **not exact backpropagation**.

## Practical notes

1. The forward path is VHDL and testbench-driven.
2. The code is written for **behavioral simulation first**.
3. The reduced-width default is much more realistic for XSIM than the full 128/256/512 profile.
4. The convolution update is intentionally coarse. Expect it to behave more like a proof-of-structure than a competitive CIFAR-10 trainer.
5. If you want a more faithful training engine next, the most useful upgrade is:
   - exact integer backprop for the FC layer
   - last-block-only fine-tuning for Conv5/Conv6
   - offline initialization of Conv1..Conv6 from PyTorch, with VHDL doing only adaptation

6. If you switch to the exact 128/128/256/256/512/512 profile, you may also want to widen internal accumulators or add stronger normalization in the forward path.

## Suggested first edits

The first file to change is `rtl/vgg7_pkg.vhd`.

Useful knobs:
- `C_C1` .. `C_C6`
- `C_ACT_W`
- `C_WEIGHT_W`
- `C_CONV_DIV_PER_IN_CH`
- `C_TARGET_SCORE`
- `C_LR_CONV`
- `C_LR_FC`
- `C_USE_ALL_LAYER_UPDATES`

## References to align the scaffold with the literature

- Spiker+ paper
- VGG paper
- NITI integer-training paper
- CIFAR-10 official dataset page
- Direct Feedback Alignment paper
