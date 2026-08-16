{
  description = "AQWA inundation mesh preprocessing environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          rasterstatsPkg = pkgs.python3Packages.buildPythonPackage rec {
            pname = "rasterstats";
            version = "0.21.0";
            pyproject = true;

            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-K5VfZ3W39kHICU0AvEOBSSGEuhYrXbobqmnfjbmO+l8=";
            };

            build-system = [ pkgs.python3Packages.hatchling ];
            dependencies = with pkgs.python3Packages; [
              affine
              click
              cligj
              numpy
              pyogrio
              rasterio
              shapely
              simplejson
            ];
            doCheck = false;
          };

          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            fiona
            geopandas
            matplotlib
            networkx
            numpy
            pandas
            pyogrio
            pyproj
            pyshp
            pyyaml
            rasterio
            rasterstatsPkg
            rtree
            scikit-learn
            scipy
            shapely
            tqdm
          ]);
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.clang
              pkgs.gnumake
              pythonEnv
            ];

            shellHook = ''
              export PATH="${pythonEnv}/bin:$PATH"
              echo "=== AQWA inundation mesh nix shell ==="
              echo "python: $(command -v python3)"
              echo "compiler: $(command -v cc)"
            '';
          };
        });
    };
}
