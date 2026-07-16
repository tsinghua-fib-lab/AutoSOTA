import nltk
from setuptools import find_packages, setup


def get_requirements(path: str):
    requirements = []
    for line in open(path):
        if not line.startswith("-r"):
            requirements.append(line.strip())
    return requirements


nltk.download("punkt")
nltk.download("omw-1.4")

setup(
    name="watermark-benchmark",
    version="1.0",
    description="Benchmark for LLM watermarks",
    long_description="Benchmark for LLM watermarkss",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.8",
    include_package_data=True,
    package_dir={"": "src"},
    package_data={
        "watermark_benchmark": ["low_entropy_tasks.json"],
    },
    entry_points={
        "console_scripts": [
            "watermark-benchmark-generate-huffman=watermark_benchmark.utils.bit_tokenizer:generate_huffman_coding",
            "watermark-benchmark-patch-generations=watermark_benchmark.utils.patch_verifiers:run",
            "watermark-benchmark-run=watermark_benchmark.pipeline.run_all:main",
            "watermark-benchmark-error-bars=watermark_benchmark.pipeline.run_error_bars:main",
            "watermark-benchmark-openai-eval=watermark_benchmark.utils.openai_quality:main",
            "watermark-benchmark-standalone-quality=watermark_benchmark.pipeline.quality:main",
            "watermark-benchmark-run-code=watermark_benchmark.pipeline.run_code:main",
            "watermark-benchmark-run-low-entropy=watermark_benchmark.pipeline.run_low_entropy:main",
        ]
    },
    zip_safe=False,
)
