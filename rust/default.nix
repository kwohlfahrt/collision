{ nixpkgs ? import <nixpkgs-unstable> {} }: with nixpkgs; callPackage ./collision.nix {}
