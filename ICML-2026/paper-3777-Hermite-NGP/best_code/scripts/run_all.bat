@echo off
::
:: Usage:
::   scripts\run_all.bat                 :: all experiments sequentially
::   scripts\run_all.bat helm2d_a10      :: one experiment
::   scripts\run_all.bat --list          :: list available names

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
if not exist results mkdir results
if not defined PY set "PY=python"

if /i "%~1"=="--list" goto LIST
if /i "%~1"=="-l"     goto LIST
set "ONLY=%~1"

:: ---------- 2D PDEs ----------
call :run helm2d_a10   "examples\helmholtz2d.py      --epochs 200000 --seed 456 --a1 10  --a2 10  --hash-size 14 --omega 0.5 --output results\helm2d_a10"
call :run helm2d_a20   "examples\helmholtz2d.py      --epochs 200000 --seed 456 --a1 20  --a2 20  --hash-size 16 --omega 0.5 --output results\helm2d_a20"
call :run helm2d_a100  "examples\helmholtz2d_a100.py --epochs 100000 --seed 456 --output results\helm2d_a100"

call :run conv1d_c30   "examples\convection1d.py --epochs 100000 --seed 456 --c 30 --lr 1e-3 --layers 1 --hidden 128 --omega 0.5 --n-levels 8 --no-causal --lr-patience 20000000 --output results\conv1d_c30"

:: ---------- 3D PDEs ----------
call :run helm3d_a3    "examples\helmholtz3d_a3.py --epochs 100000 --seed 42 --a1 3  --a2 3  --a3 3  --lr 1e-3 --layers 2 --output results\helm3d_a3"
call :run helm3d_a10   "examples\helmholtz3d.py    --epochs 150000 --seed 42 --a1 10 --a2 10 --a3 10 --lr 1e-3 --layers 2 --output results\helm3d_a10"
call :run taylor_green "examples\taylor_green.py --epochs 100000 --seed 456 --nu 0.01 --lr 1e-3 --layers 2 --cosine-scheduler"
call :run flow_mixing  "examples\flow_mixing.py  --epochs 200000 --seed 456 --lr 1e-3 --layers 2 --cosine-scheduler"

:: ---------- Complex geometry ----------
call :run poisson3d_bunny "examples\poisson3d_bunny.py --mesh %ROOT%\data\meshes\bunny.ply --epochs 150000 --seed 456 --lr 1e-3 --omega 0.2 --num_collocation 80000 --extra_bc_mode outside_only --extra_bc_gt_threshold 0.999 --n_extra_bc 10000 --extra_bc_from_gt %ROOT%\data\meshes\bunny_gt_volume_256.npy --output_prefix poisson_bunny --gt_dir %ROOT%\data\meshes --eval_interval 5000"
call :run sdf3d_bunny     "examples\sdf3d_bunny.py --mesh data\meshes\bunny.ply --epochs 100000 --seed 42 --gradient-weight 1 --near-weight 10 --offset-scale 0.02 --surface-ratio 0.50 --offset-ratio 0.25 --sdf-only-epochs 1000 --log2-hashmap-size 16 --output results\sdf3d_bunny"

:: ---------- Image reconstruction ----------
call :run image_recon_256 "examples\image_recon.py --out results\image_recon_256 --image-res 256 --loss grad --seed 7 --hs 16 --omega 2.0 --levels 8 --scale 2.0 --layers 2 --hidden 128 --epochs 40000 --lr 1e-3 --sched step --step-size 40000 --gamma 0.5 --bcw 10.0"

echo.
echo All requested experiments complete. Results in results\*\
exit /b 0


:LIST
echo Available experiments:
echo   helm2d_a10        helm2d_a20        helm2d_a100
echo   helm3d_a3         helm3d_a10
echo   conv1d_c30        taylor_green      flow_mixing
echo   poisson3d_bunny   sdf3d_bunny
echo   image_recon_256
exit /b 0


:: ----- helper -----
:run
set "NAME=%~1"
if defined ONLY if /i not "%ONLY%"=="!NAME!" exit /b 0
set "OUT=results\!NAME!"
if not exist "!OUT!" mkdir "!OUT!"
echo.
echo ============================================================
echo [!NAME!]
echo ============================================================
%PY% %~2
exit /b 0
