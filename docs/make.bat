@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

rem delete existing
rd /s /q _modules
rd /s /q _sources
rd /s /q _static
rd /s /q gps
rd /s /q utils
rd /s /q weather
del .buildinfo
del genindex.html
del index.html
del objects.inv
del py-modindex.html
del search.html
del searchindex.js

rem proceed with the build
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

rem move built html fils for github-pages
for %%i in (%BUILDDIR%\html\*) do move "%%i" .
for /d %%D in (%BUILDDIR%\html\*) do move "%%D" .

goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
