{
  description = "Gmsh terrain-adaptive hybrid unstructured mesh generator";

  # AQWA の 01_mkMESH_INUN2DH と同じリビジョンに固定（geopandas/rasterio 等がキャッシュ済み）
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/421eebfd0ec7bccd4abe826ce62d7e6e83129493";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # nixpkgs の python3Packages.gmsh は opencascade-occt のソースビルドを伴い
      # darwin ではバイナリキャッシュが無いため、公式 PyPI wheel を使う。
      # wheel は py2.py3-none タグなので Python 3.14 でもそのまま動く。
      gmshVersion = "4.15.2";
      gmshWheels = {
        "aarch64-darwin" = {
          platform = "macosx_12_0_arm64";
          hash = "sha256-9mSbPln0knLn7oqyguy00abg1ifobPPjsag/0HQX5Pg=";
        };
        "x86_64-darwin" = {
          platform = "macosx_10_15_x86_64";
          hash = "sha256-n1qXtmM+QL4zHLd3q1cc75bm/1l5u3sgqtJDhyTKQXs=";
        };
        "x86_64-linux" = {
          platform = "manylinux_2_24_x86_64";
          hash = "sha256-QHapSM4iYlMw0UE9SYLiK1xp/C8PeVH132THeM9UEIw=";
        };
      };
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };

          wheel = gmshWheels.${system} or (throw
            "gmsh の PyPI wheel が ${system} 向けに定義されていません");

          gmshPkg = pkgs.python3Packages.buildPythonPackage {
            pname = "gmsh";
            version = gmshVersion;
            format = "wheel";
            src = pkgs.fetchPypi {
              pname = "gmsh";
              version = gmshVersion;
              format = "wheel";
              dist = "py2.py3";
              python = "py2.py3";
              abi = "none";
              inherit (wheel) platform hash;
            };
            doCheck = false;
            pythonImportsCheck = [ "gmsh" ];
          };

          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            gmshPkg
            geopandas
            shapely
            rasterio
            pyogrio
            pyproj
            numpy
            scipy
            # 盛り土の検出（トップハット変換・細線化）に使う
            scikit-image
            pandas
            meshio
            pyyaml
            matplotlib
            rtree
            numba
            tqdm
            pytest
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];

            shellHook = ''
              export PATH="${pythonEnv}/bin:$PATH"
              export PYTHONNOUSERSITE=1
              echo "=== gmsh terrain-adaptive mesh shell ==="
              echo "python: $(python3 --version)"
            '';
          };
        });
    };
}
