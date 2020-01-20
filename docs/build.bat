@echo OFF

set SPHINXOPTS=" "
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build/html


if "%1"=="" (
    echo "Usage : build.bat html"
) else (
    %SPHINXBUILD% -b "%1" %SOURCEDIR% %BUILDDIR% 
)

