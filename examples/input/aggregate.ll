%MyStruct = type { i32, float }
%Inner    = type { i32, i32 }
%Outer    = type { %Inner, float }
%Mixed    = type { i32, [2 x float] }

define void @main() {
entry:
  ; ============================================================
  ; Simple struct { i32, float }
  ; ============================================================

  %s0 = insertvalue %MyStruct undef, i32 10, 0
  %s1 = insertvalue %MyStruct %s0, float 0x3FF8000000000000, 1 ; 1.5

  %s1_i = extractvalue %MyStruct %s1, 0
  %s1_f = extractvalue %MyStruct %s1, 1

  ; ============================================================
  ; Nested struct { { i32, i32 }, float }
  ; ============================================================

  ; Build inner
  %i0 = insertvalue %Inner undef, i32 7, 0
  %i1 = insertvalue %Inner %i0, i32 9, 1

  ; Insert inner + float
  %o0 = insertvalue %Outer undef, %Inner %i1, 0
  %o1 = insertvalue %Outer %o0, float 0x40091EB860000000, 1 ; 3.14

  ; Extract nested pieces
  %inner_extracted = extractvalue %Outer %o1, 0
  %inner_a = extractvalue %Inner %inner_extracted, 0
  %inner_b = extractvalue %Inner %inner_extracted, 1
  %outer_f = extractvalue %Outer %o1, 1

  ; ============================================================
  ; Array [3 x i32]
  ; ============================================================

  %a0 = insertvalue [3 x i32] undef, i32 100, 0
  %a1 = insertvalue [3 x i32] %a0, i32 200, 1
  %a2 = insertvalue [3 x i32] %a1, i32 300, 2

  %a_e0 = extractvalue [3 x i32] %a2, 0
  %a_e1 = extractvalue [3 x i32] %a2, 1
  %a_e2 = extractvalue [3 x i32] %a2, 2

  %hi = shufflevector <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>, <2 x i32> <i32 0, i32 2>

  ; ============================================================
  ; Mixed struct { i32, [2 x float] }
  ; ============================================================

  %m0 = insertvalue %Mixed undef, i32 42, 0

  %fa0 = insertvalue [2 x float] undef, float 0x4014000000000000, 0 ; 5.0
  %fa1 = insertvalue [2 x float] %fa0, float 0x4018000000000000, 1 ; 6.0

  %m1 = insertvalue %Mixed %m0, [2 x float] %fa1, 1

  %m_arr = extractvalue %Mixed %m1, 1
  %m_f0 = extractvalue [2 x float] %m_arr, 0
  %m_f1 = extractvalue [2 x float] %m_arr, 1

  %hmm = load ptr addrspace(92), ptr addrspace(92) inttoptr( i32 0 to ptr addrspace(92) )
  %vec = add <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <i32 1, i32 2, i32 3, i32 4>

  %val = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 0, i32 1)
  %result = extractvalue { i32, i1 } %val, 0

  ret void
}
