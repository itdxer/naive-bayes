from setuptools import setup

setup(
    name="naive-bayes",
    version="0.1.1",
    author="Yurii Shevchuk",
    author_email="mail@itdxer.com",
    keywords="naive bayes text classification classifier",
    packages=["naivebayes"],
    description="Naive Bayes Text Classification",
    install_requires=[
        "scikit-learn>=0.15.2",
        "numpy>=1.9.0",
    ],
)
