# 常用命令：
watch -n 1 nvidia-smi
source ~/hanyong/.venv/bin/activate

nvcc -arch=sm_80 program.cu -o program
nvcc -arch=sm_86 -O3 program.cu -o program

# 标准CMake编译流程
mkdir build && cd build
cmake ..
make -j$(nproc)



# 完整ncu内核分析
sudo -E $(which ncu) --metrics sm__cycles_elapsed.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum --export nsys_ ./
sudo -E $(which ncu) --metrics sm__cycles_elapsed.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum --export nsys_ $(which python) .py


sudo -E $(which ncu) --set full --target-processes all --replay-mode application --export ncu_ $(which python) .py

ncu --set full --target-processes all --replay-mode application --force-overwrite --export ncu_ $(which python) .py


ncu --set full --target-processes all --replay-mode kernel --force-overwrite --export ncu_ $(which python) .py
ncu --set full --target-processes all --replay-mode kernel --force-overwrite --export ncu_ ./


# 完整nsys系统分析
nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=nsys_ ./
nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=nsys_ python3 .py


# 端到端计算的分析
nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=e2e_all_old python3 generate_vllm_old_int2int8.py --dataset 3 --num_prompts 16 --min_length 3500 --max_length 4000 --max_tokens=256 --truncate_long
nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=e2e_all_new python3 generate_vllm.py --dataset 3 --num_prompts 256 --min_length 3500 --max_length 4000 --max_tokens=256 --truncate_long



# 基础ncu内核分析(export可选) ./后面是程序名
sudo -E $(which ncu) --export res ./
sudo -E $(which ncu) --export res $(which python) .py
# 基础nsys系统分析(不需要sudo)
nsys profile --output=res ./  
nsys profile --output=res python .py




## 1. 系统监视命令

### CPU & 内存监控
# 实时系统监控 (按CPU使用率排序，按q退出)
top
htop                                    # 更友好的界面，按F9可杀进程

# CPU使用率详情 (每2秒刷新，显示所有核心)
mpstat -P ALL 2                        # -P ALL显示所有CPU核心
vmstat 1                               # 每1秒显示系统统计信息

# 内存使用情况
free -h                                # 人类可读格式显示内存
watch -n 3 free -h                    # 每3秒刷新内存使用情况

# 查找占用资源最多的进程
ps aux --sort=-%cpu | head -10        # 按CPU使用率排序前10个进程
ps aux --sort=-%mem | head -10        # 按内存使用率排序前10个进程

### GPU监控 (CUDA开发重点)
# 基础GPU状态查看
nvidia-smi                             # 查看GPU基本信息和使用情况
nvidia-smi -l 1                       # 每1秒刷新一次(-l参数)
watch -n 1 nvidia-smi                 # 另一种每1秒刷新的方式

# 详细GPU监控 (CSV格式输出，便于脚本处理)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits

# 实时GPU性能监控 (专业版)
nvidia-smi dmon -i 0 -s pucvt -d 1    # -i指定GPU卡号，-s指定监控项目，-d刷新间隔
# 监控项目说明: p(power功耗) u(utilization利用率) c(clock频率) v(violations违规) t(temperature温度)

# GPU进程监控
nvidia-smi pmon -i 0 -d 1             # 监控GPU上运行的进程


### 磁盘监控
# 磁盘空间查看
df -h                                  # 查看各分区使用情况
du -sh /path/to/directory              # 查看指定目录大小
du -h --max-depth=1 /path              # 查看目录下各子目录大小

# 查找大文件 (清理磁盘空间时有用)
find /home/yanxia -type f -size +1G -exec ls -lh {} \; 2>/dev/null | sort -k5 -rh
find . -name "*.pt" -size +100M        # 查找大型模型文件
find . -name "*torch*" -size +100M     # 查找torch安装包


## 2. 文件操作命令

### 文件浏览与查找
# 文件列表 (按需求排序)
ls -la                                 # 详细列表，包含隐藏文件
ls -lS                                 # 按文件大小排序
ls -lt                                 # 按修改时间排序(最新在前)
ls -ltr                                # 按修改时间排序(最旧在前)

# 文件查找
find /path -name "*.py"                # 按文件名模式查找
find /path -type f -size +100M         # 查找大于100M的文件
find /path -mtime -7                   # 查找7天内修改的文件
find /path -name "*cuda*" -type f      # 查找包含cuda的文件

# 内容搜索
grep "search_text" file.txt            # 在文件中搜索文本
grep -r "search_text" /path            # 递归搜索目录
grep -i "text" file.txt                # 忽略大小写搜索
grep -n "text" file.txt                # 显示行号


### 文件操作
# 复制与移动
cp source.txt dest.txt                 # 复制文件
cp -r source_dir/ dest_dir/            # 递归复制目录
rsync -avh source/ dest/               # 更高效的同步复制
mv old_name.txt new_name.txt           # 移动/重命名

# 删除操作 (谨慎使用)
rm file.txt                            # 删除文件
rm -rf directory/                      # 强制删除目录(危险命令!)
rm -i *.txt                            # 交互式删除(会询问确认)

# 创建与链接
mkdir -p path/to/new/dir               # 递归创建目录
ln -s /path/to/target linkname         # 创建软链接
touch new_file.txt                     # 创建空文件或更新时间戳


### 文件查看
# 查看文件内容
cat file.txt                           # 完整显示文件内容
less file.txt                          # 分页查看(按q退出，按/搜索)
head -n 20 file.txt                    # 查看前20行
tail -n 20 file.txt                    # 查看后20行
tail -f log.txt                        # 实时监控文件更新

# 文件对比
diff file1.txt file2.txt               # 比较两个文件差异


## 3. 环境管理命令

### Python虚拟环境
# venv环境管理
python -m venv .venv                   # 创建虚拟环境
source .venv/bin/activate              # 激活环境(Linux/Mac)
deactivate                             # 退出环境

# conda环境管理  
conda create -n bitnet python=3.12     # 创建指定Python版本的环境
conda activate bitnet                  # 激活conda环境
conda deactivate                       # 退出conda环境
conda env list                         # 列出所有环境
conda env remove -n env_name           # 删除环境


### 包管理
# pip包管理
pip install package_name               # 安装包
pip install package==1.2.3            # 安装指定版本
pip install -r requirements.txt       # 从文件安装
pip list                               # 列出已安装包
pip freeze > requirements.txt         # 导出环境
pip uninstall package_name            # 卸载包
pip show package_name                  # 查看包详细信息

# conda包管理
conda install package_name             # 安装包
conda list                             # 列出已安装包
conda env export > environment.yml    # 导出环境
conda env create -f environment.yml   # 从文件创建环境


### 环境变量
# 查看环境变量
echo $PATH                             # 查看PATH变量
echo $CUDA_HOME                        # 查看CUDA路径
echo $VIRTUAL_ENV                      # 查看当前虚拟环境

# 设置环境变量 (临时)
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# 永久设置 (添加到~/.bashrc)
echo 'export CUDA_HOME=/usr/local/cuda-12.2' >> ~/.bashrc
source ~/.bashrc                       # 重新加载配置


## 4. 编译相关命令

### CUDA程序编译
# 基础编译
nvcc program.cu -o program             # 基本编译

# 指定GPU架构编译 (重要!)
nvcc -arch=sm_86 program.cu -o program # 为RTX 30/40系列GPU编译
nvcc -arch=sm_75 program.cu -o program # 为RTX 20系列GPU编译
nvcc -arch=sm_80 program.cu -o program # 为A100 GPU编译

# 优化编译
nvcc -arch=sm_86 -O3 program.cu -o program              # 最高优化
nvcc -arch=sm_86 -O3 -use_fast_math program.cu -o program # 快速数学库

# 调试编译
nvcc -g -G -O0 program.cu -o program   # 生成调试信息，关闭优化

# 生成PTX中间代码 (用于分析)
nvcc -ptx -arch=sm_86 program.cu       # 生成PTX汇编代码

# 链接外部库
nvcc -arch=sm_86 program.cu -o program -lcublas -lcurand -lcusolver

# 详细编译信息
nvcc -arch=sm_86 -Xptxas -v program.cu -o program       # 显示寄存器使用等信息


### CMake编译 (大型项目)
# 标准CMake编译流程
mkdir build && cd build               # 创建构建目录
cmake ..                              # 配置项目
make -j$(nproc)                       # 并行编译(使用所有CPU核心)

# 指定编译类型
cmake -DCMAKE_BUILD_TYPE=Release ..   # 发布版本
cmake -DCMAKE_BUILD_TYPE=Debug ..     # 调试版本

# 清理重新编译
make clean                            # 清理构建文件
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

### Python运行
# 基础运行
python script.py                      # 运行脚本
python -u script.py                   # 无缓冲输出(实时显示)

# 带参数运行
python script.py --arg1 value1 --arg2 value2

# 性能分析运行
python -m cProfile script.py          # 性能分析
python -m cProfile -o profile.stats script.py && python -m pstats profile.stats

# 调试运行
python -i script.py                   # 运行后进入交互模式
python -m pdb script.py               # 调试模式运行

## 5. CUDA性能分析命令 (专业开发重点)

### ncu - GPU内核分析 (Nsight Compute)
# 基础内核分析
ncu ./program                          # 分析所有CUDA内核
ncu python script.py                  # 分析Python中的CUDA代码

# 高级分析 (推荐用法)
ncu --profile-from-start off ./program    # 不自动开始分析，程序控制何时开始
ncu --kernel-regex ".*gemm.*|.*matmul.*" ./program  # 只分析特定内核(正则表达式)

# 详细性能指标分析
ncu --metrics sm__cycles_elapsed.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed ./program
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program  # 内存访问分析

# 保存分析结果
ncu --export analysis_report ./program    # 保存为.ncu-rep文件

# 完整分析命令 (生产环境推荐)
ncu --metrics sm__cycles_elapsed.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum --export detailed_analysis ./program


### nsys - 系统级分析 (Nsight Systems)
# 基础系统分析
nsys profile ./program                 # 基础分析
nsys profile python script.py         # 分析Python程序

# CUDA专项分析
nsys profile --trace=cuda ./program   # 只跟踪CUDA API调用
nsys profile --trace=cuda,nvtx ./program  # 跟踪CUDA + NVTX标记

# 完整分析 (推荐)
nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true ./program

# 保存和输出控制
nsys profile --output=analysis_result ./program  # 指定输出文件名
nsys profile --force-overwrite=true ./program    # 强制覆盖已存在的文件

# 时间控制分析
nsys profile --duration=60 ./program             # 只分析60秒
nsys profile --capture-range=cudaProfilerApi ./program  # 程序控制分析范围

# 特定用途分析
nsys profile --trace=cuda --sample=cpu ./program # 同时进行CPU采样


### 分析结果查看
# 图形界面查看 (推荐)
nsight-sys analysis_result.nsys-rep   # 打开Nsight Systems GUI
ncu-ui analysis_result.ncu-rep        # 打开Nsight Compute GUI

# 命令行查看统计
nsys stats analysis_result.nsys-rep   # 显示统计信息
nsys stats --report gputrace analysis_result.nsys-rep  # 只显示GPU跟踪信息

# 导出分析数据
nsys export --type=csv analysis_result.nsys-rep  # 导出为CSV格式




## 6. 进程与任务管理

### 后台任务管理
# 后台运行
command &                             # 后台运行命令
nohup command &                       # 后台运行，即使终端关闭也继续

# 任务控制
jobs                                  # 查看后台任务
fg %1                                 # 将任务1调到前台
bg %1                                 # 将任务1放到后台
kill %1                               # 杀死任务1

# 进程控制
ps aux | grep python                  # 查找Python进程
kill -9 PID                          # 强制杀死进程
killall python                       # 杀死所有Python进程

### Screen/Tmux会话管理
# Screen会话
screen -S session_name                # 创建命名会话
screen -ls                            # 列出所有会话
screen -r session_name                # 重新连接会话
# 在screen中按Ctrl+A然后按D分离会话

# Tmux会话 (推荐)
tmux new -s session_name              # 创建命名会话
tmux ls                               # 列出所有会话
tmux attach -t session_name           # 重新连接会话
# 在tmux中按Ctrl+B然后按D分离会话


## 7. Git版本控制

### 基础Git操作
# 仓库操作
git clone https://github.com/user/repo.git    # 克隆仓库
git init                                       # 初始化仓库

# 日常操作
git status                            # 查看状态
git add .                             # 添加所有更改
git add filename                      # 添加特定文件
git commit -m "commit message"        # 提交更改
git push origin main                  # 推送到远程main分支
git pull origin main                  # 拉取远程更新

# 分支操作
git branch                            # 查看分支
git checkout -b new_branch            # 创建并切换到新分支
git checkout main                     # 切换到main分支
git merge branch_name                 # 合并分支

# 查看历史
git log --oneline --graph             # 图形化显示提交历史
git diff                              # 查看更改差异


## 8. 网络与传输

### 文件传输
# SCP传输
scp file.txt user@host:/path/         # 上传文件
scp user@host:/path/file.txt ./       # 下载文件
scp -r directory/ user@host:/path/    # 上传目录

# Rsync同步 (推荐，支持断点续传)
rsync -avz --progress local/ user@host:remote/    # 同步到远程
rsync -avz --progress user@host:remote/ local/    # 从远程同步

# 下载工具
wget https://example.com/file.zip     # 下载文件
curl -O https://example.com/file.zip  # 另一种下载方式
curl -L -o filename.zip https://github.com/user/repo/archive/main.zip  # 下载GitHub仓库


### 网络调试
# 连通性测试
ping example.com                      # 测试连通性
traceroute example.com                # 跟踪路由
nslookup example.com                  # DNS查询

# 端口检查
telnet host port                      # 测试端口连通性
nc -zv host port                      # 另一种端口测试方式
lsof -i :8080                         # 查看端口8080的使用情况
netstat -tuln | grep 8080             # 查看端口监听状态


## 9. 系统维护

### 系统信息
# 基础信息
uname -a                              # 系统内核信息
cat /etc/*release                     # 操作系统版本
lscpu                                 # CPU详细信息
lspci | grep -i nvidia                # 查看NVIDIA GPU
nvidia-smi -q                         # GPU详细信息

# 运行状态
uptime                                # 系统运行时间和负载
who                                   # 当前登录用户
last                                  # 登录历史


### 服务管理 (systemd)
# 服务状态
systemctl status service_name         # 查看服务状态
systemctl list-units --type=service   # 列出所有服务

# 服务控制 (需要sudo权限)
sudo systemctl start service_name     # 启动服务
sudo systemctl stop service_name      # 停止服务
sudo systemctl restart service_name   # 重启服务
sudo systemctl enable service_name    # 开机自启
sudo systemctl disable service_name   # 禁用开机自启


## 10. 快捷操作技巧

### 命令行快捷键
# 光标移动
Ctrl + A                              # 移动到行首
Ctrl + E                              # 移动到行尾
Ctrl + U                              # 删除光标到行首的内容
Ctrl + K                              # 删除光标到行尾的内容

# 历史命令
Ctrl + R                              # 搜索历史命令
!!                                    # 重复上一条命令
!grep                                 # 重复最近的grep命令

# 任务控制
Ctrl + C                              # 终止当前命令
Ctrl + Z                              # 暂停当前命令
Ctrl + D                              # 退出当前shell


### 常用别名设置 (添加到~/.bashrc)
# 常用别名
alias ll='ls -la'
alias la='ls -la'
alias grep='grep --color=auto'
alias nvidia='nvidia-smi'
alias gpu='nvidia-smi'
alias activate='source ~/.venv/bin/activate'  # 根据你的虚拟环境路径调整

# CUDA相关别名
alias ncu-basic='ncu --profile-from-start off'
alias nsys-cuda='nsys profile --trace=cuda,nvtx'
alias nsys-full='nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true'