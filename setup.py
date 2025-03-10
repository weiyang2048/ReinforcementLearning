from setuptools import find_packages, setup

setup(
    name="reinforce",
    version="0.1.0",
    description="A reinforcement learning package",
    author="Wei Yang",
    author_email="weiyang2048@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "gymnasium>=0.28.1",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "tensorboard>=2.13.0",
        "optuna>=3.2.0",
        "tqdm>=4.65.0",
        "boto3>=1.35.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
)
