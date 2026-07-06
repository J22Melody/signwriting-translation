"""Clear the executable-stack flag on ELF shared objects.

torch 1.10 marks PT_GNU_STACK RWE, which qemu/Rosetta emulation refuses to map
("cannot enable executable stack"). This clears the X bit — what `execstack -c`
or patchelf 0.18's --clear-execstack do, neither of which is installable in the
python:3.9-slim image.
"""

import struct
import sys

PT_GNU_STACK = 0x6474E551

for path in sys.argv[1:]:
    with open(path, "r+b") as f:
        header = f.read(64)
        assert header[:4] == b"\x7fELF", path
        assert header[4] == 2, path  # 64-bit ELF only
        e_phoff = struct.unpack_from("<Q", header, 0x20)[0]
        e_phentsize, e_phnum = struct.unpack_from("<HH", header, 0x36)
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type == PT_GNU_STACK and p_flags & 1:
                f.seek(e_phoff + i * e_phentsize + 4)
                f.write(struct.pack("<I", p_flags & ~1))
                print(f"cleared execstack: {path}")
