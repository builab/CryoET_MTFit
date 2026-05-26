# Install during development:
#   /Applications/ChimeraX-1.10.app/Contents/bin/ChimeraX --nogui --cmd "devel install /path/to/ChimeraX-MTFit; exit"

from setuptools import setup

setup(
    name="ChimeraX-MTFit",
    version="1.0.0",
    description="Microtubule curve fitting for cryo-ET template matching data",
    author="Bui Lab @ McGill",
    author_email="huy.bui@mcgill.ca",
    url="https://github.com/builab/CryoET_MTCurveFit",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"chimerax.mtfit": "src"},
    packages=["chimerax.mtfit"],
    include_package_data=True,
    package_data={"chimerax.mtfit": ["bundle_info.xml", "scripts/*.py", "utils/*.py"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "ChimeraX :: Bundle :: General :: 1,1 :: chimerax.mtfit :: ChimeraX-MTFit :: Microtubule curve fitting for cryo-ET",
        "ChimeraX :: Command :: mtfit :: General :: Run full MT fitting pipeline on a particle list model",
    ],
)
