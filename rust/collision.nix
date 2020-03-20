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

  cargoSha256 = "1ml17s0hpl13vv1ad5amxs56r3wynj2ql7q0sxkzkawlm6a4z9ma";

  meta = with stdenv.lib; {
    platforms = platforms.all;
  };
}
