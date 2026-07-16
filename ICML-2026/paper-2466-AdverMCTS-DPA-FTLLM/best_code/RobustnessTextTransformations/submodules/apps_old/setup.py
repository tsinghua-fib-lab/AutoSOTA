from setuptools import find_packages, setup

setup(
    name="apps",
    version="1.0",
    description="APPS code benchmark",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.8",
    include_package_data=True,
    package_dir={"": "src"},
    zip_safe=False,
)
