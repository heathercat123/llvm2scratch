import llvm2scratch
import subprocess
import os

def main():
  llvm_prefix = os.environ.get("LLVM_PREFIX", "")
  llvm_suffix = os.environ.get("LLVM_SUFFIX", "")
  cc = os.environ.get("CC", f"{llvm_prefix}clang{llvm_suffix}")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)

  if not os.path.exists("./output"):
    os.mkdir("output")

  # --target=i386-none-elf will remove standard lib, preferable when adding own
  subprocess.run([cc, "-S", "-m32", "-Os", "-fno-vectorize", "-fno-slp-vectorize", "-emit-llvm", "-I", "sb3api.h", "demo.c", "-o", "main.ll"],
                 cwd=os.path.join(script_dir, "input"))

  with open("input/main.ll", "r") as file:
    proj, _ = llvm2scratch.compile(file.read(), llvm2scratch.Config(opti=True))
    proj.export("output/out.sprite3")

if __name__ == "__main__":
  main()
