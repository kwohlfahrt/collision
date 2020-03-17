{ stdenv, makeRustPlatform, rustChannels, wasm-pack, wasm-bindgen-cli, nodejs_latest }:

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

  cargoSha256 = "1zdfflk3vplvb2crj6h1lm4qq5fva9m7mx997z5mwm5w44a25h84";

  meta = with stdenv.lib; {
    platforms = platforms.all;
  };
}
