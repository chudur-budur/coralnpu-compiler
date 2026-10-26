# DoubleTilingExpert Extension Points for CoralNPU (DMA Phase 2, Part 2)

- **Author(s)**: AKM Khaled Talukder
- **Component Doc(s)**:
  [tiling_levels_explained.md](../tiling_levels_explained.md),
  [matrix_codegen.md](../matrix_codegen.md)
- **Bug(s)**: None. Buganizer component `564546727` was not searched: the
  `issues` CLI is not available on the machine this draft was written on.
- **Context**: DMA Phase 2, part 2: the step before keeping contraction tiles
  in DTCM. Checked against `311b1ac`, with IREE at `2b05c5db` plus patches
  0001-0010.

---

## Part 1: Design Specification

### 1.1 Objective & Scope

- **Problem Statement**: Phase 2 of the DMA roadmap keeps the cache-level
  tiles of a contraction in DTCM, and Phase 4 multi-buffers them. Both change
  IREE's `DoubleTilingExpert` in the middle. Phase 2 inserts a pass after
  cache-level tiling and allocates the promoted tiles differently during
  bufferization. Phase 4 inserts passes after bufferization. The expert is
  built per dispatch inside `LLVMCPULowerExecutableTarget`, by
  `addMultiTilingExpertPassPipeline`. CoralNPU can't change it for one
  dispatch without copying it or patching IREE.
- **Goals**:
  1. An IREE patch, 0011, that adds three optional extension points to
     `addMultiTilingExpertPassPipeline` (`MultiTilingExpertExtensions`). A
     unit attribute in a function's translation config turns them on for that
     function.
  2. A backend flag, `--coralnpu-dtcm-promotion` (default `false`), and a
     pass, `CoralNPUMarkDTCMPromotion`, that puts the attribute on the
     dispatches Phase 2 targets.
  3. CoralNPU registers its extensions. Both pass points get a probe pass that
     changes nothing, and a lit test shows where each one fires.
  4. **No change in generated code.** With the flag on, every opted-in
     dispatch lowers to the same ELF as before. With the flag off, nothing
     changes at all, IR included, and the patched expert adds the same passes
     as the unpatched one.
- **Non-Goals**:
  - Promoting any operand to DTCM (DMA Phase 2 parts 3 and 5), and the
    allocation callback that part 3 needs. The patch adds the `allocationFn`
    point; CoralNPU leaves it unset.
  - Phase 4's multi-buffering and software pipelining.
  - A CoralNPU copy of `addMultiTilingExpertPassPipeline`. It is needed only if
    a later phase has to reorder or replace an IREE pass (1.3, *When to
    switch*).
  - Convolutions, elementwise ops, `mmt4d`, vecmat, dot,
    `linalg.quantized_matmul`, and dynamic shapes. None of them get the
    attribute (see *The scope predicate*).

### 1.2 Background & Invariants

**How a dispatch gets its translation info today.**
`CoralNPUTargetBackend::buildConfigurationPassPipeline`
(`compiler/Target/CoralNPUTargetBackend.cpp:357-419`) runs the three CoralNPU
tile-size passes on each function. `CoralNPUTileSizeSelectionRegister` attaches
an `#iree_codegen.compilation_info` to the root op, and in it a
`translation_info` whose pipeline is `#iree_cpu.pipeline<DoubleTilingExpert>`
for every contraction and for many convolutions
(`CoralNPUTileSizeSelectionRegister.cpp:253`, `:1751-1795`).
`CoralNPUTileSizeSelectionDTCM` rewrites the lowering config and may add
`enable_loop_peeling` to the translation config (`:807-828`).
`CoralNPUTileSizeSelectionWorkgroup` rewrites the config again but keeps the
translation info (`CoralNPUTileSizeSelectionWorkgroup.cpp:119-122`). Then
`MaterializeUserConfigs` (`CoralNPUTargetBackend.cpp:416`) moves the
`translation_info` onto the function. It copies the whole
`TranslationInfoAttr`, configuration included (`setUserConfig`,
`Common/UserConfig.cpp:12-29`). `LLVMCPUSelectLoweringStrategy` (`:418`) keeps
it, because `initCPULaunchConfig` returns early for a function that already
has one (`KernelDispatch.cpp:4365-4368`). It then verifies `DoubleTilingExpert`
configs with `verifyMultiTilingExpertPassPipelineConfig`
(`LLVMCPUSelectLoweringStrategy.cpp:270-288`).

**How the pipeline is built.** In the translation pipeline,
`LLVMCPULowerExecutableTarget` builds `LLVMCPUPipelineOptions` from the target
and the function (`LLVMCPULowerExecutableTarget.cpp:123-174`). Two options
come from the function's translation config through `isOptEnabled`, which
only checks that a key is present (`Utils/CPUUtils.cpp:121-124`): pack/unpack
decomposition (`:142-143`) and peeling (`:174`). It then calls
`buildPipeline(pm, &cpuOpts)` on the pipeline attribute (`:179-199`). For
`DoubleTilingExpert`, `buildCPUPipeline` (`LLVMCPU/Passes.cpp:758-794`, case at
`:775-779`) calls `addMultiTilingExpertPassPipeline` (`:170-269`), which adds,
in order:

1. distribution, then one tile-and-fuse step per tiling level that the root's
   lowering config has, in enum order (`:175-212`; the levels are
   `IREECPUTypes.h:21-27`): the two cache levels, then the two vector levels;
2. inner-parallel vector tiling (`:218`), `ForallToFor` (`:223`) and peeling;
3. vectorization (`:233-252`);
4. bufferization, `addCPUBufferizePasses` (`:255`);
5. vector lowering (`:258-267`).

**Why promotion needs a bufferization change.** `addCPUBufferizePasses` uses
`cpuAllocationFn` (`Common/CPU/Passes.cpp:21-38`), which always emits a
hoisted `memref.alloca`. Workgroup local memory, which the CoralNPU runtime
allocates in DTCM, is assigned later by `LLVMCPUAssignWorkgroupLocalMemory`
(`Passes.cpp:587`), and that pass rejects any local-space result that isn't a
`memref.alloc` (`LLVMCPUAssignWorkgroupLocalMemory.cpp:191-201`).

**Invariants.**

- With the flag off, the configuration and translation pipelines are
  unchanged pass for pass. No function has the marker, so the patched expert
  adds the same passes as the unpatched one.
- With the flag on, an opted-in dispatch keeps its pipeline attribute,
  lowering config and `LLVMCPUPipelineOptions`, except for the new
  `enableMultiTilingExtensions` field. The only added passes are the probes,
  which change nothing, so the generated code is the same.
- The Zvt codegen runs in `beforeLowerToLLVMHook`, outside the per-dispatch
  pipeline (`CoralNPUTargetBackend.cpp:430-453`). It is unaffected.
- The only change to `third_party/` is patch 0011. It adds extension points
  and no CoralNPU logic.

### 1.3 Alternatives Considered

DMA Phases 2 and 4 need three changes to IREE's `DoubleTilingExpert`, at three
different stages of `addMultiTilingExpertPassPipeline`
(`LLVMCPU/Passes.cpp:170-269`):

| Need                                                       | Stage                                                                 | Who                                             |
| :--------------------------------------------------------- | :-------------------------------------------------------------------- | :---------------------------------------------- |
| Insert the promotion pass                                  | After cache-level tiling, before vector tiling; on tensors, in the workgroup `scf.forall` | Phase 2 parts 3 and 5                     |
| Allocate local-space tensors as a hoisted `memref.alloc`   | Bufferization (`:255`)                                                | Phase 2 part 3 (`part-3-accumulator.md`, *The allocation callback*) |
| Multi-buffer and software-pipeline the tile loop           | Expected after bufferization (`:255`), before vector lowering (`:258-267`); on memrefs, in `scf.for`. Not settled: see *When to switch* | Phase 4 (roadmap §3.4)  |

The second need exists because the stock callback, `cpuAllocationFn`
(`Common/CPU/Passes.cpp:21-38`), always emits `memref.alloca`, and
`LLVMCPUAssignWorkgroupLocalMemory` rejects any local-space result that isn't
a `memref.alloc` (`LLVMCPUAssignWorkgroupLocalMemory.cpp:191-201`). A pass
inserted before bufferization can't change that.

| Option                                                              | Pros                                                                                                                                             | Cons                                                                                                                                                                                                       | Est. LOC (Phase 2 parts 2-3 and Phase 4; code and tests, excluding the promotion pass and callback bodies, which every option needs) |
| :------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| **1 (Chosen)**: marker plus a registered extension struct           | Covers the three known needs in one patch; no dialect; no copy; config verification kept; the passes are rebuilt from IR, so reproducers and `iree-opt` replays match | One IREE patch (0011) to rebase; the struct is process-wide, so extension passes take their parameters from IR, not from session options                                                                 | +550 local; patch ~90 changed lines (~150-line `.patch` file) |
| **2**: extension struct carried in `LLVMCPUPipelineOptions`         | No global state; callbacks can capture session options, like the 0006 hook (`CoralNPUTargetBackend.cpp:430`); unreachable from `llvm-cpu`      | A `std::function` isn't part of the textual pipeline, so a crash reproducer or an `iree-opt` replay of `LowerExecutableTarget` silently drops the extension passes; needs a hand-written pass constructor, because `cpuOptions` is a textual option (`LLVMCPU/Passes.td:109`) | +550 local; patch ~110 changed lines |
| **3**: pipeline attribute that delegates now, copies in part 3      | No IREE patch in this CL                                                                                                                         | Part 3 still needs the copy, a patch (or a local `extern` decl) for `buildLLVMCPUVectorLoweringPipeline`, a re-created `addTileAndDistributePasses`, `clFailOnLargeVector` by name, and a drift test; verification is skipped for non-CPU pipelines (`LLVMCPUSelectLoweringStrategy.cpp:284-286`) | +500 now, ~+250 in part 3                                  |
| **4**: pipeline attribute plus a copy of the expert now             | Full control: can reorder or replace IREE passes, not only insert                                                                                | Same costs as 3, paid now, with a drift test guarding a copy no one runs differently yet                                                                                                                   | +800                                                                   |
| **5**: a refactor patch that splits the expert into exported phases | No global state, no copy, any insertion point                                                                                                    | A larger patch that moves IREE code, so it conflicts on most bumps; CoralNPU still has to compose the phases per dispatch, through the same marker plus a builder hook, or through the pipeline attribute of 3                       | +500 local, ~150-line patch                                            |
| **6**: `#iree_codegen.pass_pipeline<"...">` (text)                  | No new dialect                                                                                                                                   | Its `buildPipeline` ignores the options (`IREECodegenAttrs.cpp:142-148`). A CoralNPU pass would have to write the whole expert as a string per dispatch: an untyped copy with no drift guard, with the `cpuOpts` flags frozen into IR | ≥ option 4                                       |
| **7**: re-register `registerCPUPipelineBuilder`                     | No new attribute                                                                                                                                 | `buildCPUPipeline` is `static` (`Passes.cpp:758`), so the replacement re-implements its seven-case switch and still needs a copy to insert mid-expert; IREE registers first (`Passes.cpp:805`), so the second call asserts in debug builds and silently overwrites in release (`IREECPUAttrs.cpp:48-55`) | ≥ option 4                       |
| **8**: transform-dialect codegen                                    | Fully scriptable                                                                                                                                 | Re-expresses tiling, vectorization, bufferization and lowering as transform IR                                                                                                                            | large                                                                  |

**The chosen design.** One IREE patch,
`iree-v3.12.0-0011-llvmcpu-add-multi-tiling-expert-extensions.patch`, generated
on top of 0001-0010 (0008's canonicalize sits at `Passes.cpp:211`, next to the
first call site). It adds extension points and no logic (AGENTS.md *Third-party
code*: "restrict them to minimal extension points or hooks"). Three files:

- `LLVMCPU/Passes.h`, next to `addMultiTilingExpertPassPipeline` (`:133`).
  `<functional>` is already included (`:15`). For `AllocationFn` the patch adds
  `mlir/Dialect/Bufferization/IR/Bufferization.h`, which comes from
  `@llvm-project//mlir:BufferizationDialect`, already a direct dependency of
  `LLVMCPU` (`LLVMCPU/BUILD.bazel:146`). `BufferizableOpInterface.h` would
  break `layering_check` (`:16`), because its target isn't a direct
  dependency.

  ```cpp
  // Key of the unit attribute in a function's translation config that turns
  // the extensions on for that function.
  inline constexpr llvm::StringLiteral kMultiTilingExtensionsAttrName =
      "enable_multi_tiling_extensions";

  using MultiTilingExtensionFn = std::function<void(
      OpPassManager &, IREE::Codegen::LoweringConfigAttrInterface,
      const LLVMCPUPipelineOptions &)>;

  // Extension points of DoubleTilingExpert. Each one is optional. They are
  // called while building the pipeline, possibly for several functions at
  // once, so they must be stateless: they only add passes, and the passes
  // take their parameters from the IR.
  struct MultiTilingExpertExtensions {
    // After cache-level tiling, before vector-level tiling. On tensors.
    MultiTilingExtensionFn afterCacheTiling;
    // Replaces cpuAllocationFn in bufferization; the copy function is kept.
    // It runs for every marked function, so for anything not in workgroup
    // local memory it must do exactly what cpuAllocationFn does.
    bufferization::BufferizationOptions::AllocationFn allocationFn;
    // After bufferization, before vector lowering. On memrefs.
    MultiTilingExtensionFn afterBufferization;
  };

  // Called at most once per process, during plugin registration.
  void registerMultiTilingExpertExtensions(MultiTilingExpertExtensions ext);
  // Null if none were registered.
  const MultiTilingExpertExtensions *getMultiTilingExpertExtensions();
  ```

  plus `bool enableMultiTilingExtensions = false;` in `LLVMCPUPipelineOptions`
  (`:79-91`).
- `LLVMCPU/LLVMCPULowerExecutableTarget.cpp`, after the peeling line
  (`:174`): read `isOptEnabled(funcOp, kMultiTilingExtensionsAttrName)` into
  the new field. If it is set and nothing is registered, `emitWarning` that
  the extensions are skipped, for example for CoralNPU IR replayed in a stock
  `iree-opt`. The pipeline is still correct then, only not promoted.
- `LLVMCPU/Passes.cpp`: storage as in `registerCPUPipelineBuilder`
  (`IREECPUAttrs.cpp:43-55`), with a `report_fatal_error` instead of an
  `assert` on a second call, so that release builds can't overwrite it
  silently. In `addMultiTilingExpertPassPipeline`,
  `ext = pipelineOpt.enableMultiTilingExtensions ? getMultiTilingExpertExtensions() : nullptr`,
  then three call sites:
  1. `afterCacheTiling`: at the top of the tiling-level loop body (`:175`),
     when `i == VectorCommonParallelTiles`, before the `hasTilingLevel` check
     at `:176`. The levels are in enum order (`IREECPUTypes.h:21-27`), so this
     runs once for every config, after the last cache level and its
     pad-fusion and canonicalize (`:209-211`), and before any vector tiling,
     including `TileLastOpAndFuseProducerConsumer` (`:218`).
  2. `allocationFn`: when set, `:255` calls
     `addIREEComprehensiveBufferizePasses(funcPassManager, ext->allocationFn, copyFn)`
     (`Common/Passes.h:58-62`) instead of `addCPUBufferizePasses`. `copyFn`
     is a lambda around `createLinalgCopyOp` (`Utils/Utils.h:495`), the same
     as the `static` `cpuCopyFn` (`Common/CPU/Passes.cpp:40-44`).
  3. `afterBufferization`: between `:255` and `:258`.

  With `ext` null, the expert adds the same passes as today.

On the CoralNPU side:

- The marking pass, `CoralNPUMarkDTCMPromotion` (1.4), keeps `#iree_cpu.pipeline<DoubleTilingExpert>` and
  adds `kMultiTilingExtensionsAttrName` to the translation config of in-scope
  roots, as the DTCM pass adds `enable_loop_peeling`
  (`CoralNPUTileSizeSelectionDTCM.cpp:807-828`). The Workgroup pass,
  `MaterializeUserConfigs` and `LLVMCPUSelectLoweringStrategy` keep the
  configuration, and `verifyMultiTilingExpertPassPipelineConfig` still runs
  (`LLVMCPUSelectLoweringStrategy.cpp:270-288`).
- `CoralNPUSession::registerPasses` (`CoralNPUTarget.cpp:67`) registers the
  struct. `GlobalInit` calls it once per process
  (`API/Internal/CompilerDriver.cpp:428`), and so does `iree-opt`
  (`API/Internal/IREEOptToolEntryPoint.cpp:64`).
- **In this CL**, `afterCacheTiling` and `afterBufferization` each add
  `CoralNPUExtensionProbe`, a pass that changes nothing. A lit test prints the
  IR before it (`--mlir-print-ir-before=coralnpu-extension-probe`) and checks
  that the first probe sees tensors after cache tiling and before vector
  tiling, that the second sees memrefs before vector lowering, that
  each runs once per marked function, and that neither runs for unmarked
  functions. `allocationFn` stays unset. Part 3 replaces the first probe with
  the promotion pass and sets the callback; Phase 4 replaces the second.
- **Parameters come from the IR.** The registered struct can't see
  `CoralNPUOptions`. Part 4's `--coralnpu-dtcm-local-memory-budget-kb` must
  therefore reach the promotion pass through the executable target config,
  written in `getExecutableTarget` (`CoralNPUTargetBackend.cpp:303-343`), or
  through the translation config written by the marking pass. That also
  keeps the passes correct for per-session flags set through the C API.
- **Target check.** The marker is only written by the CoralNPU marking
  pass, but a hand-written `compilation_info` could put it on an `llvm-cpu`
  dispatch. Each extension pass returns early unless the function's
  executable target backend is `coralnpu`. The allocation callback can't
  check that, so its contract (above) is to behave exactly like
  `cpuAllocationFn` outside workgroup local memory.

**What this removes.** Compared with option 3, there is no `coralnpu_codegen`
dialect (three `.td` files, the dialect and attribute classes, their
generated targets and two registrations; about 250 lines). Part 3 has no copy
of the expert, no re-created `addTileAndDistributePasses` (`:71-86`), no
lookup of `clFailOnLargeVector` (`:47`) by name, no patch for
`buildLLVMCPUVectorLoweringPipeline` (`:92`), no drift test, and no follow-up
to restore config verification. Phase 4 probably needs no new patch, but
that isn't settled (*When to switch*).

**Decision Rationale**: The known needs are all insertions or a callback
swap, and option 1 covers them with one patch of about 90 changed lines.
Counted with tests, that is about 640 lines against about 750 for option 3,
which also still needs its own patch or `extern` declaration. Size is not
the main argument, though: option 1 keeps IREE's verification and leaves no
copy to drift. Option 2 is the same size, but it loses the extension
passes in reproducers and `iree-opt` replays, which are the main tools for
debugging these dispatches. Option 1's cost, that extension passes can't see
session options, is a constraint worth having: it puts their parameters in
the IR, where reproducers also see them. Options 4 to 8 are larger or need a
copy anyway.

**When to switch.** The hooks hold while every CoralNPU change to the expert
is an insertion or a callback. If a later phase has to reorder, replace or
drop an IREE pass (for example, bufferize before vectorization), switch to a
copied pipeline (option 4). Phase 4 is the likely test of this rule:

- At `afterBufferization` the copies into DTCM have already been vectorized
  into `transfer_read`/`transfer_write` pairs (`Passes.cpp:246`), and Phase 4
  has to turn them into `coralnpu_dma_start`/`wait`. Roadmap §3.4 asks for
  pipelining "soon after tiling".
- The Workgroup pass makes each workgroup exactly one cache tile
  (`CoralNPUTileSizeSelectionWorkgroup.cpp:95-114`), so the only loop inside a
  dispatch that Phase 4 can multi-buffer is the cache-reduction `scf.for`,
  which only RVV roots get, from part 5 on.

So Phase 4 may need another extension point, or a way to keep the copies out
of vectorization. Its design decides that; it isn't promised here.

**Signal on IREE bumps.** `scripts/patch-third_party.sh:36-46` reports a
patch that doesn't apply as `[SKIPPED]`, not as an error. The first hard
signal is then the compile error on `registerMultiTilingExpertExtensions`.
An upstream change elsewhere in the expert can also move what a call site
sees (tensors or memrefs, `scf.forall` or `scf.for`) while the patch still
applies. The probe lit test catches that.

**This reverses a recorded decision.** Roadmap §0.3 and §3.4 and the phase-2
README record "pipeline attribute" (here option 3). §3.4 rejected the hook
route because "`LowerExecutableTarget` rebuilds the pipeline options from
flags". The marker answers that: `LowerExecutableTarget` already reads
per-function options from the translation config (`:174`), so one patch is
enough. Those three documents are updated in `coral-work` to record this
decision.

**Approval.** The patch needs explicit approval under AGENTS.md *Third-party
code* before the `code` step starts. If it is refused, fall back to option 3.

### 1.4 Detailed Technical Design

**Pipeline / control flow** (new steps marked ►):

```
buildConfigurationPassPipeline (per function)
  CoralNPUTileSizeSelectionRegister   compilation_info, #iree_cpu.pipeline<DoubleTilingExpert>
  CoralNPUTileSizeSelectionDTCM       cache tiles, peeling
  CoralNPUTileSizeSelectionWorkgroup  distribution tiles
► CoralNPUMarkDTCMPromotion           only with --coralnpu-dtcm-promotion:
                                      in-scope roots get enable_multi_tiling_extensions
  MaterializeUserConfigs              translation_info moves to the function
  LLVMCPUSelectLoweringStrategy       keeps it; verifies the DoubleTilingExpert config

buildTranslationPassPipeline
  LLVMCPULowerExecutableTarget      ► reads the marker into enableMultiTilingExtensions
    addMultiTilingExpertPassPipeline
      distribution and cache-level tiling
►     afterCacheTiling                CoralNPUExtensionProbe   (part 3: the promotion pass)
      vector-level tiling, vectorization
      bufferization                 ► allocationFn: unset     (part 3: local space → hoisted alloc)
►     afterBufferization              CoralNPUExtensionProbe   (Phase 4: multi-buffering, pipelining)
      vector lowering
  beforeLowerToLLVMHook (Zvt codegen etc.)   unchanged
```

**The IREE patch.** 1.3 gives its API and call sites. Mechanics:

- Patches 0001-0010 are applied to the submodule as uncommitted changes, on
  top of HEAD `2b05c5db`, and 0006 and 0008 already touch `Passes.{h,cpp}`. So
  a commit of the three files would carry their hunks too. The procedure is:
  1. save any work in the submodule, then reset it to `2b05c5db`;
  2. `./scripts/patch-third_party.sh --commit iree` applies 0001-0010 as commits
     (`git am`; the patches are in mbox format);
  3. make the change, commit it, and export it with `git format-patch -1` as
     `iree-v3.12.0-0011-llvmcpu-add-multi-tiling-expert-extensions.patch` at
     the repo root;
  4. `git -C third_party/iree reset --hard 2b05c5db`, then
     `./scripts/patch-third_party.sh iree` without `--commit`, so the submodule is
     dirty again with 0001-0011, as AGENTS.md *Third-party code* expects;
  5. `git -C third_party/iree diff 2b05c5db --stat` lists the same files as
     before plus the 0011 hunks.

  None of the submodule commits are pushed.
- `scripts/patch-third_party.sh` applies root-level patches by glob
  (`patch_submodule`, `:50-68`), so the script doesn't change.
- The patch is one logical concern (AGENTS.md: "Keep `.patch` files small and
  scoped to a single upstreamable logical concern"): extension points of one
  IREE function, with no CoralNPU names in it.

**Library layering.** No new library. Every user of the new code is in
`//compiler/Transforms`: the marking pass, the probe, part 3's promotion pass
and part 5's tile selection. So the predicate, the extension registration and
both passes go there. `//compiler/Transforms:Transforms` gains a dependency on
`@iree_core//compiler/src/iree/compiler/Codegen/LLVMCPU`, for
`LLVMCPU/Passes.h`. `//compiler/Target:CoralNPUTarget` already links it, and
IREE's `LLVMCPU` doesn't depend on the plugin, so there is no cycle.

**Extension registration.** A new `compiler/Transforms/CoralNPUExpertExtensions.{h,cpp}`:

```cpp
// Registers CoralNPU's DoubleTilingExpert extensions with IREE. Called once
// per process, from CoralNPUSession::registerPasses.
void registerCoralNPUExpertExtensions();
```

The `.cpp` fills an `iree_compiler::MultiTilingExpertExtensions`. In this CL
`afterCacheTiling` and `afterBufferization` each add
`createCoralNPUExtensionProbePass()` to the pass manager they get, and
`allocationFn` stays unset. The lambdas capture nothing, as the contract in
`Passes.h` requires. `CoralNPUSession::registerPasses`
(`CoralNPUTarget.cpp:67`) calls it after `registerCoralNPUPasses()`. That also
covers `iree-opt`, which calls the same hook
(`API/Internal/IREEOptToolEntryPoint.cpp:64`).

**The probe pass.** Declared in `compiler/Transforms/Passes.td`, with a
constructor like the other passes (`Passes.td:28`):

```tablegen
def CoralNPUExtensionProbe : InterfacePass<
    "coralnpu-extension-probe", "mlir::FunctionOpInterface"> {
  let summary = "Marks a DoubleTilingExpert extension point; changes nothing";
  let description = [{
    Added at the extension points that have no CoralNPU pass yet. Tests print
    the IR before it to check where each point runs.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUExtensionProbePass()
  }];
}
```

`runOnOperation` only calls `markAllAnalysesPreserved()`. Part 3 and Phase 4
each replace one probe. When neither is left, the pass and its test go.

**The flag.** In `CoralNPUOptions` (`CoralNPUTargetBackend.h:47-61`) add
`bool dtcmPromotion = false;`, bound next to `coralnpu-dtcm-size-kb`
(`:109-111`):

```cpp
binder.opt<bool>(
    "coralnpu-dtcm-promotion", dtcmPromotion, llvm::cl::cat(category),
    llvm::cl::desc("Keep the cache-level tiles of contractions in DTCM "
                   "(experimental; default: false)"));
```

Until part 3 the flag only adds the marker, and through it the probes. The
description already states the intent, and `README.md` says the flag has no
effect on generated code yet.

**The marking pass.** Declared in `compiler/Transforms/Passes.td`:

```tablegen
def CoralNPUMarkDTCMPromotion : InterfacePass<
    "coralnpu-mark-dtcm-promotion", "mlir::FunctionOpInterface"> {
  let summary = "Turns on the CoralNPU DoubleTilingExpert extensions for DTCM promotion";
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUMarkDTCMPromotionPass()
  }];
}
```

- `compiler/Transforms/Passes.h` declares
  `createCoralNPUMarkDTCMPromotionPass()` and
  `createCoralNPUExtensionProbePass()`, both returning
  `std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>`, like the
  Workgroup pass (`Passes.h:63-64`). `Passes.h` needs no new include.
- Neither pass creates ops or attributes of a dialect that isn't already
  loaded, so neither has `dependentDialects`.
- `CoralNPUMarkDTCMPromotion.cpp` includes
  `compiler/Transforms/CoralNPUTileSizeSelectionUtils.h`,
  `iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h`,
  `iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h`
  (`getCompilationInfo` / `setCompilationInfo`, `:191-197`),
  `iree/compiler/Codegen/LLVMCPU/Passes.h` (`kMultiTilingExtensionsAttrName`),
  `mlir/IR/Remarks.h` and `llvm/Support/DebugLog.h`.
- Both passes are registered with the others by `registerCoralNPUPasses()`
  (`Passes.h:33`), so `iree-opt --coralnpu-mark-dtcm-promotion` works.
- The marking pass has no options. The backend adds it after
  `CoralNPUTileSizeSelectionWorkgroup` and before `MaterializeUserConfigs`
  (`CoralNPUTargetBackend.cpp:414-416`), **only when**
  `options_.dtcmPromotion` is set. With the flag off the pass manager is
  therefore unchanged.

The pass walks the function and, for each op where `getCompilationInfo(op)`
is non-null (one root per function, set by the Register pass), does the
following:

1. Skips the op unless its pipeline is `#iree_cpu.pipeline<DoubleTilingExpert>`.
   The extension points exist only in that expert.
2. Skips it unless `isDTCMPromotionCandidate(op)` holds (below).
3. Copies the translation config into a `NamedAttrList` (empty if there is
   none), sets `kMultiTilingExtensionsAttrName` to a `UnitAttr`, and builds a
   new `TranslationInfoAttr` with the same pipeline, codegen spec, workgroup
   size and subgroup size, as the DTCM pass does at `:816-819`. The config
   keeps `enable_loop_peeling`. Then
   `setCompilationInfo(op, CompilationInfoAttr::get(ctx, <same lowering
   config>, <new translation info>))`.

**Remark.** A root that is a contraction or a convolution
(`linalg::isaContractionOpInterface` or `linalg::isaConvolutionOpInterface`,
the same test as `CoralNPUTileSizeSelectionDTCM.cpp:684-686`) and fails step 2
is the only case where promotion was plausible but doesn't apply. The pass
reports it with the opt-in
`mlir::remark::missed(op->getLoc(), mlir::remark::RemarkOpts::name("DTCMPromotion").category("CoralNPU").subCategory("MarkDTCMPromotion"))`
(`mlir/IR/Remarks.h:726`), with a reason, plus `LDBG()`, so normal compilation
stays silent. Every other op (elementwise, transpose, reduce, and anything on
another pipeline) is skipped without a remark. `RemarkOpts::name`, `category`
and `subCategory` are the builders in `Remarks.h:105-124`.

**The scope predicate.** It goes in
`compiler/Transforms/CoralNPUTileSizeSelectionUtils.h`, next to
`isZvtMatrixContraction`, so that parts 3 and 5 use the same check (AGENTS.md,
*Keep cross-component predicates synchronized*):

```cpp
// Returns true if |op| is a root that DMA Phase 2 keeps in DTCM: a static
// contraction with M and K dimensions (matmul, batch_matmul,
// batch_reduce_matmul, matvec, batch_matvec and their generic forms) that
// is not in mmt4d layout. Peers: CoralNPUMarkDTCMPromotion, the promotion
// pass (part 3) and promotion-aware tile selection (part 5). The mmt4d test
// mirrors setContractVectorSizes in CoralNPUTileSizeSelectionRegister.cpp;
// keep the two in sync.
bool isDTCMPromotionCandidate(Operation *op);
```

Implementation: `linalg::isaContractionOpInterface`, `!hasDynamicShape()`, and
`linalg::inferContractionDims` with non-empty `m` and `k`, excluding the
`m`/`n`/`k` = 2/2/2 layout that `setContractVectorSizes` treats as mmt4d
(`CoralNPUTileSizeSelectionRegister.cpp:716-750`).

What falls in and out:

- **In:**
  - `matmul`, `batch_matmul`, `matvec`, `batch_matvec`, and generic
    contractions with that shape.
  - `batch_reduce_matmul`. Its batch dimension is a reduction, so
    `inferContractionDims` reports two `k` dims. It has one M and one N, so
    it isn't mistaken for mmt4d.
- **Out, convolutions.** They fail the first check. This matters because
  checking the pipeline alone wouldn't exclude them: not every convolution
  uses `ConvTileAndDecomposeExpert`, which has no cache-level tiling. CoralNPU
  gives several layouts `DoubleTilingExpert`, for example `conv_2d_nhwc_fhwc`
  (`:253`).
- **Out, `linalg.quantized_matmul` and `quantized_batch_matmul`.** They have
  four inputs, so `isaContractionOpInterface` rejects them
  (`mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp:536`), even though the
  Register pass sizes them like matmuls (`:1751-1755`). In practice IREE
  rewrites them to plain matmuls early, in
  `LinalgQuantizedMatmulToMatmulPass` (`GlobalOptimization/Passes.cpp:125`),
  so they rarely reach a dispatch in that form. Bringing them into scope is a
  follow-up (this doc's Part 3).
- **Out, vecmat and dot.** They have no `m`. Phase 2 covers matmul and matvec
  shapes only.

**IR example.** `--compile-to=executable-configurations` with the flag on,
for a 64x64x64 f32 matmul:

IREE prints `translation_info` through an alias (`IREECodegenDialect.cpp:27-29`),
so the marker appears in the alias definition at the top of the output. A
`DictionaryAttr` keeps its keys sorted, so the marker comes after
`enable_loop_peeling`:

```mlir
// Before (or with the flag off):
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
...
func.func @matmul_dispatch_0_matmul_64x64x64_f32() attributes {translation_info = #translation} {

// After:
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling, enable_multi_tiling_extensions}>
...
func.func @matmul_dispatch_0_matmul_64x64x64_f32() attributes {translation_info = #translation} {
```

The `linalg.matmul`'s `lowering_config` is identical in both. The alias and
function names here are illustrative.

**Edge cases.**

- *Dynamic shapes.* In the backend flow a dynamic root never reaches the new
  pass with a CoralNPU `compilation_info`:
  `CoralNPUTileSizeSelectionAnalysis.cpp:107-110` bails out first, and IREE's
  own defaults pick the pipeline later. The predicate still rejects dynamic
  shapes, so that it is correct for parts 3 and 5 and for direct `iree-opt`
  use, and the `iree-opt` test covers that.
- *Verification.* The pipeline stays `DoubleTilingExpert`, so
  `verifyMultiTilingExpertPassPipelineConfig` still runs on opted-in
  dispatches.
- *Other backends.* Only LLVMCPU's `LowerExecutableTarget` reads the key, and
  only the CoralNPU backend's configuration pipeline adds it. A hand-written
  `compilation_info` could still put it on an `llvm-cpu` dispatch, so every
  extension pass that changes IR returns early unless the function's
  executable target backend is `coralnpu`, the backend half of
  `hasCoralNPUTargetFeature` (`CoralNPUTileSizeSelectionUtils.cpp:71-75`). The probe changes
  nothing, so it needs no check.
- *Concurrency.* `LowerExecutableTarget` runs on functions in parallel and
  builds each function's pipeline there, so the extension callbacks can be
  called concurrently. CoralNPU's lambdas capture nothing and only add passes.
- *Linking.* `--coralnpu-link-executables` links per variant. Translation
  info is per function and isn't used during linking, so marked and unmarked
  functions in one executable are fine.
- *Plugin not loaded.* The marker is a plain unit attribute, so IR that
  contains it still parses in a stock `iree-opt`. `LowerExecutableTarget` then
  warns that the extensions are skipped and lowers the function without them.

### 1.5 Verification Plan

Two lit files, both picked up by the `glob` in `tests/transforms/BUILD.bazel`.
Neither uses `%iree_compile`. That substitution targets the host CPU as well as
CoralNPU (`tests/transforms/lit.cfg.py:44-49`), and the host variant also
prints `#iree_cpu.pipeline<DoubleTilingExpert>`.

**`tests/transforms/mark_dtcm_promotion.mlir`**: the pass alone, through
`iree-opt`. It follows `tile_size_selection_register.mlir`. The input is
hand-written IR: a `hal.executable` with a `coralnpu` target, whose ops carry
`compilation_info` with `#iree_cpu.pipeline<DoubleTilingExpert>`.

```
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-mark-dtcm-promotion)))))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --remarks-filter-missed=CoralNPU --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-mark-dtcm-promotion)))))" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
```

- **Marked:** static `linalg.matmul`, `linalg.batch_matmul`,
  `linalg.batch_reduce_matmul`, `linalg.matvec`, `linalg.batch_matvec`, and a
  `linalg.generic` matmul. Each keeps `DoubleTilingExpert` and gains
  `enable_multi_tiling_extensions`. A matmul whose translation config has
  `enable_loop_peeling` keeps it.
- **Not marked:**
  - `linalg.add`, `linalg.vecmat`, `linalg.dot`, and a
    `linalg.conv_2d_nhwc_fhwc` with a 3x3 filter.
  - `linalg.mmt4d`, and a `linalg.generic` in the same 2/2/2 layout, which
    guards the mmt4d test that mirrors `setContractVectorSizes`.
  - A dynamic-shape `linalg.matmul` with a hand-written `compilation_info`,
    which reaches the predicate only through `iree-opt` (see *Dynamic shapes*).
  - A `linalg.quantized_matmul`.
  - A matmul that already uses `ConvTileAndDecomposeExpert`, which is left alone
    because of step 1.
- **Checks:** each split is one function with a unique name, so `CHECK-LABEL`
  can't match a longer name. The aliases (`#translation`, `#compilation`) are
  captured with `CHECK-DAG: #[[$TI:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {{.*}}enable_multi_tiling_extensions`
  and matched on the op's `compilation_info`. Unmarked splits have
  `CHECK-NOT: enable_multi_tiling_extensions`.
- **Remark:** a second RUN line adds `--remarks-filter-missed=CoralNPU`
  (`MlirOptMain.cpp:272`; filtering by category as in
  `mlir/test/Pass/remarks.mlir:2`) with `2>&1` and checks the missed remark
  for the convolution and the dynamic matmul, and no remark for `linalg.add`.

**`tests/transforms/mark_dtcm_promotion_e2e.mlir`**: the backend flow. It has
only a few functions, so that unrelated lowering problems can't fail it. Each
function becomes its own dispatch.

```
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --coralnpu-dtcm-promotion --compile-to=executable-configurations %s \
// RUN:   | FileCheck %s
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s --check-prefix=OFF
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --coralnpu-dtcm-promotion \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f,+zfbfmin,+zvfbfmin,+zvfbfwma \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s --check-prefix=RVV
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --coralnpu-dtcm-promotion --compile-to=executable-targets \
// RUN:   --mlir-print-ir-before=coralnpu-extension-probe %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PROBE
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-print-ir-before=coralnpu-extension-probe %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=PROBE-OFF
```

- No `--coralnpu-roofline-speedup-threshold`: with no host device,
  `CoralNPUAffinityAnnotation.cpp:302-307` always picks CoralNPU.
- **Inputs:**
  - `@mm_f32`, a 64x64x64 f32 matmul;
  - `@mm_i8_n80`, an i8 matmul with M = 64, N = 80, K = 64. N = 80 is a partial
    64-column Zvt tile (the example in `matrix_codegen.md`). Like every Zvt
    root it gets `enable_loop_peeling` (`CoralNPUTileSizeSelectionDTCM.cpp:812-820`),
    and this one actually peels;
  - `@add_f32`, a static `linalg.add`.
- **`CHECK`:** `@mm_f32` and `@mm_i8_n80` carry the marker. The `@mm_i8_n80`
  alias also has `enable_loop_peeling`. `@add_f32` has no marker. All checks
  use `CHECK-DAG` alias captures, as above.
- **`OFF`:** `CHECK-NOT: enable_multi_tiling_extensions`.
- **`RVV`:** `@mm_f32` still carries the marker.
- **`PROBE`:** for each of `@mm_f32` and `@mm_i8_n80`, exactly two dumps headed
  `IR Dump Before CoralNPUExtensionProbe`, in this order:
  1. `afterCacheTiling`: the root works on tensors directly inside the
     `scf.forall` with `mapping = [#iree_codegen.workgroup_mapping`, and there
     is no `vector.` op yet. There is no separate cache-parallel loop: the
     Workgroup pass makes the distribution tiles equal to the cache-parallel
     tiles (`CoralNPUTileSizeSelectionWorkgroup.cpp:95-114`), so that loop has
     one iteration and the canonicalizer folds it, because it has no mapping
     (`mlir/lib/Dialect/SCF/IR/SCF.cpp:1618`);
  2. `afterBufferization`: the function works on memrefs, there is no
     `tensor.` op left, and the contraction is a `vector.contract` that
     hasn't been lowered.

  No dump names `@add_f32`'s function. `--mlir-disable-threading` keeps the
  dumps in order.
- **`PROBE-OFF`:** `CHECK-NOT: IR Dump Before CoralNPUExtensionProbe`, with
  `--allow-empty`, because without the flag nothing is printed.
- **`--mlir-print-ir-before` inside `LowerExecutableTarget`.** It works: the
  nested `runPipeline` shares the parent's instrumentation, and
  `tests/transforms/target_options.mlir` already prints after a pass in the
  translation pipeline's dynamic pipeline. Plugin passes are registered
  before the pass-manager options (`CompilerDriver.cpp:428`, then `:440`), so
  the pass name resolves.
- **To confirm on the first build:** that IREE leaves these ops as written
  before configuration (in particular that it doesn't rewrite the matmuls
  into another op).

  If an expectation turns out wrong, record it in 2.3 and adjust the input
  rather than the check.

**Identical code.**

- **Inputs:** every `*.mlir` under
  `tests/models/stablehlo/generated_{f32,bf16,i8,i16,i32}` and
  `tests/models/linalg/generated_{f32,bf16,i8,i16,i32}`, about 3,300 files. bf16
  is the case that exercises the native BF16 converts.
- **Configurations:** four per input, about 13,000 compiles in all:
  default features and RVV-only, each with and without the flag. Plus one
  baseline: default features without the flag, from a build of this CL's
  parent commit with patches 0001-0010 only, about 3,300 more compiles. This
  CL's code calls the 0011 API, so the baseline can't be this CL without the
  patch. Switching the submodule state rebuilds most of IREE.
- **Command,** the model tests' own target flag
  (`tests/models/stablehlo/defs.bzl:7`) plus the variable parts:

  ```
  iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
    [--coralnpu-dtcm-promotion] [--coralnpu-target-cpu-features=<rvv-only>] \
    --iree-hal-dump-executable-binaries-to=<out>/<features>/<flag>/<suite>/<input>/ \
    <input> -o <out>/<features>/<flag>/<suite>/<input>.vmfb
  ```

  `<features>` is `zvt` or `rvv`, and `<flag>` is `on`, `off`, or `base` for
  the parent-commit build. The `on` runs also pass
  `--iree-hal-dump-executable-configurations-to=<out>/<features>/on/<suite>/<input>/configs/`.
- **Pass criterion:** `diff -r` over the dumped ELFs reports no differences
  between `<out>/<features>/on` and `<out>/<features>/off` for each feature
  set, and between `<out>/zvt/off` and `<out>/zvt/base`. A `.vmfb` difference
  is reported but isn't a failure until the `.vmfb` output is shown to be
  deterministic.
- **Non-vacuous:** the script counts the dumped configurations that contain
  `enable_multi_tiling_extensions`, per suite and data type. Every suite with
  matmuls must have a non-zero count, or the check proves nothing.
- **Where it runs:** a script in `agent-temp/` (created if it doesn't exist),
  in parallel with `xargs -P "$(nproc)"`. No test files change.

**Model check tests.** Byte-identical ELFs give identical results.
`bazel test --config=dev //tests:ci` must still pass with the flag at its
default.

**Register & spill verification.** Not needed: no code changes.

---

## Part 2: Implementation Plan

### 2.1 Expected Affected Files

- `iree-v3.12.0-0011-llvmcpu-add-multi-tiling-expert-extensions.patch`: new.
  It changes `LLVMCPU/Passes.h`, `LLVMCPU/Passes.cpp` and
  `LLVMCPU/LLVMCPULowerExecutableTarget.cpp` under
  `third_party/iree/compiler/src/iree/compiler/Codegen/`.
- `compiler/Transforms/CoralNPUTileSizeSelectionUtils.{h,cpp}`:
  `isDTCMPromotionCandidate`.
- `compiler/Transforms/Passes.td`, `Passes.h`: declare
  `CoralNPUMarkDTCMPromotion` and `CoralNPUExtensionProbe`.
- `compiler/Transforms/CoralNPUMarkDTCMPromotion.cpp`: new; the marking pass.
- `compiler/Transforms/CoralNPUExtensionProbe.cpp`: new; the probe.
- `compiler/Transforms/CoralNPUExpertExtensions.{h,cpp}`: new;
  `registerCoralNPUExpertExtensions`.
- `compiler/Transforms/BUILD.bazel`, `CMakeLists.txt`: the new sources and
  header, and the IREE `LLVMCPU` dependency.
- `compiler/Target/CoralNPUTargetBackend.h`: the `dtcmPromotion` option.
- `compiler/Target/CoralNPUTargetBackend.cpp`: the conditional pass.
- `compiler/Target/CoralNPUTarget.cpp`: `registerPasses` calls
  `registerCoralNPUExpertExtensions`.
- `tests/transforms/mark_dtcm_promotion.mlir`: new lit test (the pass,
  through `iree-opt`).
- `tests/transforms/mark_dtcm_promotion_e2e.mlir`: new lit test (the backend
  flow and the probes).
- `tests/transforms/CMakeLists.txt`: both lit files, since its `SRCS` list is
  explicit (`:18-32`).
- `README.md`: document `--coralnpu-dtcm-promotion`.

### 2.2 Implementation Tasks (`code` step)

- [ ] **Gate**: explicit approval of patch 0011 under AGENTS.md *Third-party
  code*. Without it, stop and fall back to option 3 of 1.3, which needs this
  doc to be revised first.
- [ ] **Task 1**: Patch 0011 (1.3, *The chosen design*; 1.4, *The IREE
  patch*).
  - *Target*: the three IREE files, then the root-level `.patch` file
  - *Verify*: the five steps in 1.4, *The IREE patch*, end with the submodule
    dirty with 0001-0011 and nothing else; then
    `bazel build --config=dev @iree_core//tools:iree-compile`.
- [ ] **Task 2**: `isDTCMPromotionCandidate` in
  `CoralNPUTileSizeSelectionUtils`.
  - *Target*: `compiler/Transforms/CoralNPUTileSizeSelectionUtils.{h,cpp}`
  - *Verify*: `bazel build --config=dev //compiler/Transforms`
- [ ] **Task 3**: The probe pass, `CoralNPUExpertExtensions`, the call in
  `CoralNPUSession::registerPasses`, and the IREE `LLVMCPU` dependency of
  `//compiler/Transforms:Transforms`.
  - *Target*: `compiler/Transforms/*`, `compiler/Target/CoralNPUTarget.cpp`
  - *Verify*: `bazel build --config=dev @iree_core//tools:iree-compile
    @iree_core//tools:iree-opt`
- [ ] **Task 4**: The `CoralNPUMarkDTCMPromotion` pass. That covers its
  `Passes.td` entry, the `Passes.h` declaration (no new include), the `.cpp`
  with the includes listed in 1.4, and the missed-optimization remark for
  contraction and convolution roots.
  - *Target*: `compiler/Transforms/*`
  - *Verify*: `bazel build --config=dev //compiler/Transforms`
- [ ] **Task 5**: `--coralnpu-dtcm-promotion`, and the conditional pass in
  `buildConfigurationPassPipeline`.
  - *Target*: `compiler/Target/CoralNPUTargetBackend.{h,cpp}`
  - *Verify*: `bazel build --config=dev @iree_core//tools:iree-compile`
- [ ] **Task 6**: The two lit tests (1.5). Record in 2.3 any input that IREE
  rewrites before configuration, and whether the probe dumps appear.
  - *Target*: `tests/transforms/mark_dtcm_promotion.mlir`,
    `tests/transforms/mark_dtcm_promotion_e2e.mlir`
  - *Verify*: `bazel test --config=dev //tests/transforms:lit`
- [ ] **Task 7**: The identical-code check over all five data types in the
  stablehlo and linalg suites, with the default and RVV-only features, plus the
  parent-commit baseline: about 16,500 parallel compiles (1.5). Record the
  result and the marker counts in 2.3.
  - *Target*: an `agent-temp/` script (not committed)
  - *Verify*: `diff -r` over the dumped ELFs reports no differences, and
    every matmul suite has a non-zero marker count
- [ ] **Task 8**: The regression check.
  - *Verify*: `bazel test --config=dev //tests:ci`
- [ ] **Task 9 (Build & Doc Sync)**: after review, mirror the Bazel changes in
  `compiler/Transforms/CMakeLists.txt`: the new sources, the header, and
  `iree::compiler::Codegen::LLVMCPU` next to the other IREE dependencies
  (`:58-60`), and add the two lit files to `tests/transforms/CMakeLists.txt`
  (`:18-32`). Then document the flag in `README.md` next to
  `--coralnpu-fast-math` (`README.md:248`). `refresh_compile_commands`
  (`BUILD.bazel:23-37`) lists only top-level targets, so it doesn't change.

### 2.3 Implementation Notes & Adaptations

- *(None yet)*

### 2.4 Documentation Archive Checklist (`archive` step — Pre-Submit)

- [ ] Add a short "DoubleTilingExpert extension points" section to
      `docs/design/tiling_levels_explained.md`. It says which functions get
      `enable_multi_tiling_extensions`, where the three points are in the
      expert, why CoralNPU uses hooks rather than a copy, and when to switch.
      Update `docs/design/INDEX.md` with the new source paths and patch 0011.
- [ ] Move this document from `docs/design/changes/coralnpu_codegen_pipeline.md`
      to `docs/design/archive/coralnpu_codegen_pipeline.md` using `git mv`.
- [ ] Format modified code with `./scripts/format-code.sh`.

---

## Part 3: Follow-Up Items (For Reviewer Approval -> Buganizer `564546727`)

- [ ] **Promotion pass and allocation callback** (`P1` |
      `compiler/Transforms/CoralNPUExpertExtensions.cpp`): DMA Phase 2 part 3
      replaces the `afterCacheTiling` probe with the promotion pass, and sets
      `allocationFn` to the callback in `part-3-accumulator.md` (*The
      allocation callback*). The promotion pass returns early for non-CoralNPU
      targets (1.4, *Other backends*).
- [ ] **Where Phase 4 runs** (`P2` | Phase 4 design): decide whether
      `memref::multiBuffer`, `scf::pipelineForLoop` and the DMA rewrite can
      work at `afterBufferization`, where the copies are already vectorized
      and the only in-dispatch tile loop is the RVV cache-reduction `scf.for`.
      If not, add an extension point or keep the copies out of
      vectorization, or apply 1.3's *When to switch*. The acceptance test is a
      Phase 4 lit test of the chosen point.
- [ ] **DTCM budget through the IR** (`P2` | part 4): extension passes can't
      see `CoralNPUOptions`, so `--coralnpu-dtcm-local-memory-budget-kb` has to
      reach the promotion pass through the executable target config
      (`getExecutableTarget`, `CoralNPUTargetBackend.cpp:303-343`) or through
      the translation config written by the marking pass.
- [ ] **Upstream the extension points** (`P2` | IREE): propose
      `MultiTilingExpertExtensions` upstream, so that 0011 can be dropped at a
      later IREE bump. Upstream, the key should follow IREE's convention, a
      `getEnable...Str()` in `Utils/CPUUtils.h` like
      `getEnableLoopPeelingStr` (`CPUUtils.cpp:115-119`).
- [ ] **Quantized matmuls in scope** (`P3` |
      `compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp`):
      `linalg.quantized_matmul` and `quantized_batch_matmul` fail
      `isaContractionOpInterface`. If one is ever seen in a CoralNPU
      dispatch, extend the predicate (for example, with the Register pass's
      matmul classification) and add a lit case.
