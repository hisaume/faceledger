# Support mainstream glibc x86-64 Linux for version one

Faceledger version one supports CPU execution across mainstream glibc-based
x86-64 Linux rather than Ubuntu alone, validating representative
Debian/Ubuntu, Fedora, and Arch environments. This broadens practical desktop
reach without taking on the separate packaging and runtime burdens of GPU,
ARM64, musl, NixOS-native, or immutable-system support; the exact managed Python
and locked dependency baseline will be chosen through a packaging spike before
implementation.
