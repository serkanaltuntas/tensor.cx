module {
  gpu.module @device {
    gpu.func @tensorcx_add(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr, %n: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %base = arith.muli %bid, %bdim : index
      %i = arith.addi %base, %tid : index
      %limit = arith.index_castui %n : i32 to index
      %active = arith.cmpi ult, %i, %limit : index
      scf.if %active {
        %idx = arith.index_cast %i : index to i64
        %ap = llvm.getelementptr %a[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %bp = llvm.getelementptr %b[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %av = llvm.load %ap : !llvm.ptr -> f32
        %bv = llvm.load %bp : !llvm.ptr -> f32
        %v = arith.addf %av, %bv : f32
        llvm.store %v, %op : f32, !llvm.ptr
      }
      gpu.return
    }
  }
}
