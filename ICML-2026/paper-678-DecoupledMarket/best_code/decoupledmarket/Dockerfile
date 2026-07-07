# Docker 容器化配置
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# 暴露端口（如果需要）
# EXPOSE 8000

# 默认命令
CMD ["python", "-m", "decoupledmarket.main_parallel", "--mode", "parallel", "--executor", "thread"]
