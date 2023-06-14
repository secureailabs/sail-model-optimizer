import os
import shutil
import subprocess

SAIL_PYPI_REPOSITORY_URL = os.environ["SAIL_PYPI_REPOSITORY_URL"]
SAIL_PYPI_USERNAME = os.environ["SAIL_PYPI_USERNAME"]
SAIL_PYPI_PASSWORD = os.environ["SAIL_PYPI_PASSWORD"]

# remove old build
shutil.rmtree("dist", ignore_errors=True)

# build dist
subprocess.run(f"python setup.py bdist_wheel")

# upload result
list_name_file = os.listdir("dist")
print(list_name_file)
for name_file in list_name_file:
    if name_file.endswith(".whl"):
        path_file = os.path.join("dist", name_file)
        subprocess.run(
            f"twine upload {path_file} --repository-url {SAIL_PYPI_REPOSITORY_URL} --username {SAIL_PYPI_USERNAME} --password {SAIL_PYPI_PASSWORD}"
        )
