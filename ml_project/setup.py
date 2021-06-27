from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='ML project for ML in Production course.',
    author='Timur Kamaliev',
    install_requires=[
        'flake8==3.9.1',
        'hydra-core==1.0.6',
        'marshmallow-dataclass==8.4.1',
        'matplotlib==3.3.4',
        'numpy==1.19.5',
        'pandas==1.1.5',
        'python-dotenv>=0.5.1',
        'PyYAML==5.4.1',
        'scikit-learn==0.24.1',
        'seaborn==0.11.1',
    ],
    license='MIT',
)
