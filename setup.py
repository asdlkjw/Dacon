try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from setuptools import find_packages
import os
import sys
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

package_name = "dacon_cars"


def get_package_root_path() -> str:
    """
    Returns
        package_root_path: package_name으로 찾은 해당 패키지의 root 경로
    """
    packages = list(filter(lambda x: package_name in x, sys.path))
    packages.remove(os.getcwd())
    package_root_path = packages[0]
    return package_root_path


def create_default_dotenv():
    with open(".env", "w") as f:
        print("개인설정 파일을 생성합니다 (.env)")
        f.write(f"PACKAGE_NAME = '{package_name}'\n")
        f.write(f"WEBHOOK_URL = ''\n")


if __name__ == "__main__":
    setup(
        name=package_name,
        version="0.1",
        packages=find_packages(where="src"),
        package_dir={"": "."},
        py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    )
    if not os.path.exists(".env"):
        create_default_dotenv()
