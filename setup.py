"""

"""
from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="feedback_prize",
        version="0.1.0",
        description="student writing feedback prize",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Michael Kingston",
        author_email="michael.kenneth.kingston@gmail.com",
        url="https://github.com/mkingopng/feedback_prize",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=["torch>=1.6.0",
                          "transformers",
                          "pandas",
                          "numpy",
                          "scipy",
                          "nltk",
                          "spacy",
                          "joblib",
                          "tqdm",
                          "argparse",
                          "sklearn",
                          "wandb",
                          "nltk",
                          ],
        platforms=["linux", "unix"],
        python_requires=">3.9.7",
    )
