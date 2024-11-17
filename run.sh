zig build
nix develop --extra-experimental-features "nix-command flakes" --command sh -c '
    cargo build
    make vm-wasm
    make vm
    ./zig-out/bin/tyvm ./test/'"$1"'
  exit
'