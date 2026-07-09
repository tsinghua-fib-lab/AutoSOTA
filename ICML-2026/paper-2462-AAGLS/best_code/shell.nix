# shell.nix
let
  pkgs = import <nixpkgs> { };
  pythonPackages = pkgs.python3Packages;

  metisShared = pkgs.metis.overrideAttrs (old: {
    configurePhase = ''
      runHook preConfigure
      make config shared=1 prefix=$out
      runHook postConfigure
    '';

    buildPhase = ''
      runHook preBuild
      make -j$NIX_BUILD_CORES
      runHook postBuild
    '';

    # Avoid `make install` (CMake RPATH_CHANGE on gpmetis fails on Nix)
    installPhase = ''
      runHook preInstall
      mkdir -p $out/include $out/lib

      cp -v include/metis.h $out/include/

      so="$(find build -type f -name 'libmetis.so*' | head -n1)"
      if [ -z "$so" ]; then
        echo "ERROR: could not find libmetis.so in build output" >&2
        find build -maxdepth 4 -type f -name 'libmetis*' >&2 || true
        exit 1
      fi

      install -m755 "$so" $out/lib/libmetis.so
      runHook postInstall
    '';

    NIX_CFLAGS_COMPILE = (old.NIX_CFLAGS_COMPILE or "") + " -fPIC";
  });
in

pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";

  buildInputs = [
    pythonPackages.python
    pythonPackages.numpy
    pythonPackages.scipy
    pythonPackages.matplotlib
    pythonPackages.pandas
    pythonPackages.graph-tool
    pkgs.graphviz

    metisShared

    pythonPackages.venvShellHook
  ];

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    export METIS_DLL=${metisShared}/lib/libmetis.so
    pip install -r requirements.txt
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export METIS_DLL=${metisShared}/lib/libmetis.so
    pip install -r requirements.txt
  '';
}
