{ stdenv, makeRustPlatform, rustChannels, perf, wasm-pack, wasm-bindgen-cli, nodejs_latest }:

let
  rustPlatform = with rustChannels.stable; makeRustPlatform {
    cargo = cargo;
    rustc = rust.override { targets = ["wasm32-unknown-unknown"]; };
  };
in rustPlatform.buildRustPackage rec {
  pname = "pimostat";
  version = "0.1.0";

  src = ./.;
  nativeBuildInputs = [ wasm-pack wasm-bindgen-cli ];
  checkInputs = [ nodejs_latest ];

  cargoSha256 = "0q8wkv27fa8ly9q7vmyyp3qhr5ddq65qfyyxv2i5fcvf0nc6dpzd";

  meta = with stdenv.lib; {
    platforms = platforms.all;
  };
}
