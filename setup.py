from setuptools import setup, find_packages


import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("ert_gui/resources/")
logging_configuration = package_files("ert_logging/")

setup(
    name="ert",
    author="Equinor ASA",
    author_email="fg_sib-scout@equinor.com",
    use_scm_version={"root": ".", "write_to": "ert_shared/version.py"},
    entry_points={
        "console_scripts": [
            "ert3=ert3.console:main",
            "ert=ert_shared.main:main",
        ]
    },
    packages=find_packages(exclude=["tests*"]),
    package_data={"ert_gui": extra_files, "ert_logging": logging_configuration},
    include_package_data=True,
    license="Open Source",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "async-exit-stack; python_version < '3.7'",
        "async-generator; python_version < '3.7'",
        "aiofiles",
        "alembic",
        "ansicolors==1.1.8",
        "cloudevents",
        "cloudpickle",
        "console-progressbar==1.1.2",
        "decorator",
        "deprecation",
        "equinor-libres >= 10.0.0b1",
        "fastapi",
        "graphlib_backport; python_version < '3.9'",
        "jinja2",
        "matplotlib",
        "numpy",
        "pandas",
        "pluggy",
        "pydantic >= 1.8",
        "pyrsistent",
        "PyQt5",
        "pyyaml",
        "qtpy",
        "scipy",
        "semeio",
        "sqlalchemy",
        "typing-extensions; python_version < '3.8'",
        "websockets",
        "python-dateutil",
        "async_generator",
        "uvicorn",
        "prefect",
        "dask_jobqueue",
    ],
    extras_require={
        "storage": [
            "ert-storage==0.1.4",
        ],
    },
    zip_safe=False,
    tests_require=["pytest", "mock"],
    test_suite="tests",
    setup_requires=["pytest-runner", "setuptools_scm"],
)
