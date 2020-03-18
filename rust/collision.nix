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

  cargoSha256 = "0h184jcmmadf50mlhk6wivipf2nnjz2mmf81r870cikaai8xvvwz";

  meta = with stdenv.lib; {
    platforms = platforms.all;
  };
}
