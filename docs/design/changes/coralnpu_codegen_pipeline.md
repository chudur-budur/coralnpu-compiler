# A CoralNPU Codegen Pipeline Attribute (DMA Phase 2, Part 2)

- **Author(s)**: AKM Khaled Talukder
- **Component Doc(s)**:
  [tiling_levels_explained.md](../tiling_levels_explained.md),
  [matrix_codegen.md](../matrix_codegen.md)
- **Bug(s)**: None. Buganizer component `564546727` was not searched: the
  `issues` CLI is not available on the machine this draft was written on.
- **Context**: DMA Phase 2, part 2: the step before keeping contraction tiles
  in DTCM. Checked against `a30e7a2`.

---

## Part 1: Design Specification

### 1.1 Objective & Scope

- **Problem Statement**: Phase 2 of the DMA roadmap keeps the cache-level
  tiles of a contraction in DTCM. Its later parts insert a CoralNPU pass in
  the middle of IREE's `DoubleTilingExpert` pipeline, after cache-level tiling
  and before vector tiling. That pipeline is built per dispatch by
  `LLVMCPULowerExecutableTarget`, from the `#iree_cpu.pipeline<...>` in the
  function's `translation_info`. CoralNPU has no way to choose a different
  pipeline for one dispatch without patching IREE.
- **Goals**:
  1. A `coralnpu_codegen` dialect with one attribute,
     `#coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>`, that implements
     `IREE::Codegen::PipelineAttrInterface`.
  2. A backend flag, `--coralnpu-dtcm-promotion` (default `false`), and a pass
     that puts the attribute on the dispatches Phase 2 targets.
  3. **No change in generated code.** With the flag on, every opted-in
     dispatch lowers to the same ELF as before. With the flag off, nothing
     changes at all, IR included.
- **Non-Goals**:
  - Promoting any operand to DTCM (DMA Phase 2 parts 3 and 5). In this doc,
    "DTCM promotion" means only "the dispatch is lowered through the CoralNPU
    pipeline". What the pipeline does with the tiles is part 3's design.
  - A CoralNPU copy of `addMultiTilingExpertPassPipeline`, the IREE header
    patch it needs, and the test that guards the copy against drift. They move
    to part 3, the first change that needs them (Part 3 of this doc).
  - Convolutions, elementwise ops, `mmt4d`, vecmat, dot,
    `linalg.quantized_matmul`, and dynamic shapes. None of them get the
    attribute (see *The scope predicate*).

### 1.2 Background & Invariants

**How a dispatch picks its pipeline today.**
`CoralNPUTargetBackend::buildConfigurationPassPipeline`
(`compiler/Target/CoralNPUTargetBackend.cpp:357-419`) runs the three CoralNPU
tile-size passes on each function. `CoralNPUTileSizeSelectionRegister` attaches
an `#iree_codegen.compilation_info` to the root op, and in it a
`translation_info` whose pipeline is `#iree_cpu.pipeline<DoubleTilingExpert>`
for every contraction and for many convolutions
(`CoralNPUTileSizeSelectionRegister.cpp:253`, `:1751-1795`).
`CoralNPUTileSizeSelectionDTCM` rewrites the lowering config and may add
`enable_loop_peeling` to the translation config (`:810-828`).
`CoralNPUTileSizeSelectionWorkgroup` rewrites the config again but keeps the
translation info (`CoralNPUTileSizeSelectionWorkgroup.cpp:120`). Then
`MaterializeUserConfigs` (`CoralNPUTargetBackend.cpp:416`) moves the
`translation_info` onto the function. It copies any pipeline attribute
(`setUserConfig`, `Common/UserConfig.cpp:13-27`), and `TranslationInfoAttr`
only requires that the attribute implements `PipelineAttrInterface`
(`IREECodegenAttrs.cpp:173`). `LLVMCPUSelectLoweringStrategy` (`:418`) keeps
it, because `initCPULaunchConfig` returns early for a function that already
has one (`KernelDispatch.cpp:4365-4368`).

**How the pipeline is built.** In the translation pipeline,
`LLVMCPULowerExecutableTarget` builds `LLVMCPUPipelineOptions` from the target
and the function (`LLVMCPULowerExecutableTarget.cpp:123-174`; peeling comes
from the translation config at `:174`). It then calls
`buildPipeline(pm, &cpuOpts)` on any pass-pipeline attribute that implements
`PipelineAttrInterface` (`:179-199`). `cpuOpts` is a
`CPUCodegenPipelineOptions`: those options plus the root's lowering config
(`LLVMCPU/Passes.h:95-104`). IREE's own `#iree_cpu.pipeline` is one
implementation of that interface (`IREECPUAttrs.td:174-189`). For
`DoubleTilingExpert` it calls `addMultiTilingExpertPassPipeline`
(`buildCPUPipeline`, `LLVMCPU/Passes.cpp:758-794`, case at `:775-779`), which
is declared in `LLVMCPU/Passes.h:133`.

**Where else the pipeline attribute matters.** At this pin, three IREE sites
branch on `IREE::CPU::PipelineAttr` (searched over
`compiler/src/iree/compiler/Codegen` and `Dialect`): `buildCPUPipeline`,
`KernelDispatch.cpp:76` (which creates it), and
`LLVMCPUSelectLoweringStrategy.cpp:270-288`. The last one verifies
`DoubleTilingExpert` configs and **skips** verification for any other
`PipelineAttrInterface` (`:283-285`). Between `SelectLoweringStrategy` and
`LowerExecutableTarget`, the only other checks are for
`TransformDialectCodegenPipelineAttr` and `NoPipelineAttr`. No CoralNPU pass
reads the pipeline attribute after the DTCM pass, which only rebuilds it
(`CoralNPUTileSizeSelectionDTCM.cpp:817`).

**Invariants.**

- With the flag off, the configuration and translation pipelines are
  unchanged pass for pass.
- With the flag on, an opted-in dispatch gets the same `LLVMCPUPipelineOptions`
  and lowering config as before. The new attribute builds the same passes, so
  the generated code is the same.
- The Zvt codegen runs in `beforeLowerToLLVMHook`, outside the per-dispatch
  pipeline (`CoralNPUTargetBackend.cpp:430-453`). It is unaffected.
- No change to `third_party/`.

### 1.3 Alternatives Considered

| Option                                              | Pros                                                                 | Cons                                                                                                                 | Est. LOC  |
| :-------------------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :-------- |
| **1 (Chosen)**: attribute that delegates to IREE    | No IREE patch; identical code by construction; nothing to drift yet  | Part 3 still has to add the copy, the patch and the drift guard                                                      | +500 / -2 |
| **2**: attribute plus a copy of the expert now      | Part 3 becomes only "add one pass"                                   | Needs an IREE patch or a local `extern` decl; drift test guards code no one runs differently; `clFailOnLargeVector` only by name lookup | +800 / -2 |
| **3**: `#iree_codegen.pass_pipeline<"...">` (text)  | No new dialect                                                       | Its `buildPipeline` ignores the options (`IREECodegenAttrs.cpp:142-148`): loses peeling, masking, bf16 converts, lowering config | +40 / -0  |
| **4**: IREE hook inside `addMultiTilingExpert...`   | No copy ever                                                         | `LowerExecutableTarget` builds its own options from its pass option and the target config (`:123-174`), so a hook set by the backend never reaches it without a second patch | +80 / -0  |
| **5**: re-register `registerCPUPipelineBuilder`     | No new attribute                                                     | Process-wide and asserts on a second call (`IREECPUAttrs.cpp:48-55`); would also change `llvm-cpu` targets          | +60 / -0  |
| **6**: transform-dialect codegen                    | Fully scriptable                                                     | Re-expresses tiling, vectorization, bufferization and lowering as transform IR                                       | large     |

**Decision Rationale**: The attribute (options 1 and 2) is the only extension
point that receives the full CPU options and lowering config without an IREE
patch. Between the two, option 2 would add the copy now, but the copy only
pays off when part 3 inserts a pass. Until then it costs an IREE patch (or a
fragile local declaration of `buildLLVMCPUVectorLoweringPipeline`,
`Passes.cpp:92`), a drift test, and a re-creation of the `static`
`addTileAndDistributePasses` (`Passes.cpp:71-86`). It also has to read the
`static` `clFailOnLargeVector` option (`Passes.cpp:47`, read at `:250`) by name,
through `llvm::cl::getRegisteredOptions()["iree-llvmcpu-fail-on-large-vector"]`.
Option 1 makes this CL the seam that parts 3-6 plug into. Part 3 replaces one
function body.

### 1.4 Detailed Technical Design

**Pipeline / control flow** (new steps marked ►):

```
buildConfigurationPassPipeline (per function)
  CoralNPUTileSizeSelectionRegister   compilation_info, #iree_cpu.pipeline<DoubleTilingExpert>
  CoralNPUTileSizeSelectionDTCM       cache tiles, peeling
  CoralNPUTileSizeSelectionWorkgroup  distribution tiles
► CoralNPUSelectCodegenPipeline       only with --coralnpu-dtcm-promotion:
                                      in-scope roots get #coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>
  MaterializeUserConfigs              translation_info moves to the function
  LLVMCPUSelectLoweringStrategy       keeps it; skips verification for non-CPU pipelines

buildTranslationPassPipeline
  LLVMCPULowerExecutableTarget
►   PipelineAttr::buildPipeline       → addCoralNPUDTCMPipeline → addMultiTilingExpertPassPipeline
  beforeLowerToLLVMHook (Zvt codegen etc.)   unchanged
```

**Library layering.** One new library, `//compiler/Codegen:CoralNPUCodegen`
in `compiler/Codegen/`. It holds the dialect, the attribute, the pipeline
builder and the scope predicate, and it depends on IREE only, never on
`//compiler/Transforms`. `//compiler/Transforms` (the selection pass) depends
on it. Part 3's promotion pass also goes in `compiler/Codegen/`, so the
pipeline and the promotion pass can both use the predicate without a
dependency cycle.

It is separate from the existing `compiler/Pipelines/` library, which builds
the TOSA input-conversion pipeline (`compiler/Pipelines/Pipelines.h`). That
runs before dispatch formation and has nothing to do with per-dispatch
codegen.

**Namespaces and headers.** The dialect and attribute classes live in
`mlir::coralnpu_compiler::CoralNPUCodegen`. Free functions live in
`mlir::coralnpu_compiler`, like the rest of the plugin. Headers are included
as `compiler/Codegen/<File>.h`, the same way `compiler/Transforms/Passes.h`
is.

**TableGen files** (the layout of IREE's `Codegen/Dialect/CPU/IR`):

- `CoralNPUCodegenDialect.td`:

  ```tablegen
  include "mlir/IR/OpBase.td"

  def CoralNPUCodegen_Dialect : Dialect {
    let name = "coralnpu_codegen";
    let cppNamespace = "::mlir::coralnpu_compiler::CoralNPUCodegen";
    let useDefaultAttributePrinterParser = 1;
  }
  ```

- `CoralNPUCodegenEnums.td`:

  ```tablegen
  include "compiler/Codegen/CoralNPUCodegenDialect.td"
  include "mlir/IR/EnumAttr.td"

  def CoralNPU_LoweringPipeline : I32EnumAttr<"LoweringPipeline",
      "CoralNPU codegen pipeline", [
        // DoubleTilingExpert with the cache-level tiles of the root in DTCM.
        I32EnumAttrCase<"DTCMPromotedDoubleTiling", 0>]> {
    let cppNamespace = "::mlir::coralnpu_compiler::CoralNPUCodegen";
    let genSpecializedAttr = 0;
  }
  ```

- `CoralNPUCodegenAttrs.td`:

  ```tablegen
  include "compiler/Codegen/CoralNPUCodegenEnums.td"
  include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
  include "mlir/IR/AttrTypeBase.td"

  def CoralNPU_PipelineAttr : AttrDef<CoralNPUCodegen_Dialect, "Pipeline", [
      DeclareAttrInterfaceMethods<IREECodegen_PipelineAttrInterface>]> {
    let mnemonic = "pipeline";
    let summary = "CoralNPU codegen pipeline.";
    let parameters = (ins EnumParameter<CoralNPU_LoweringPipeline>:$value);
    let assemblyFormat = "`<` $value `>`";
  }
  ```

  As in `IREECPU_PipelineAttr`, no method list is needed: `buildPipeline` has
  no default, so it is always declared. `emitConstraints` and
  `materializeConfigurationAttr` keep their defaults (a no-op and an error;
  `IREECodegenInterfaces.td:755-815`). They are used only by the SMT tuner
  flow, which CoralNPU doesn't run.

**Generated files** (`gentbl_cc_library` from `@llvm-project//mlir:tblgen.bzl`,
as `compiler/Transforms/BUILD.bazel` uses):

| Target                          | TableGen flags                                              | Outputs                                                   | `td_file`                    |
| :------------------------------ | :---------------------------------------------------------- | :-------------------------------------------------------- | :--------------------------- |
| `CoralNPUCodegenDialectIncGen`  | `--gen-dialect-decls`, `--gen-dialect-defs`                 | `CoralNPUCodegenDialect.h.inc`, `CoralNPUCodegenDialect.cpp.inc` | `CoralNPUCodegenDialect.td` |
| `CoralNPUCodegenEnumsIncGen`    | `--gen-enum-decls`, `--gen-enum-defs` (`--dialect=coralnpu_codegen`) | `CoralNPUCodegenEnums.h.inc`, `CoralNPUCodegenEnums.cpp.inc` | `CoralNPUCodegenEnums.td` |
| `CoralNPUCodegenAttrsIncGen`    | `--gen-attrdef-decls`, `--gen-attrdef-defs`                 | `CoralNPUCodegenAttrs.h.inc`, `CoralNPUCodegenAttrs.cpp.inc` | `CoralNPUCodegenAttrs.td`   |

All three depend on a `td_library` named `CoralNPUCodegenTdFiles` (the three
`.td` files) with deps `@llvm-project//mlir:OpBaseTdFiles` and
`@iree_core//compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR:td_files`.

**C++ files.**

The headers follow `IREECPUDialect.h` / `IREECPUTypes.h`, so that each one
compiles on its own:

| File                         | Includes, in order                                                                                                                           |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `CoralNPUCodegenAttrs.h`     | `IREECodegenInterfaces.h`, `CoralNPUCodegenEnums.h.inc`, then `CoralNPUCodegenAttrs.h.inc` under `#define GET_ATTRDEF_CLASSES`              |
| `CoralNPUCodegenDialect.h`   | `CoralNPUCodegenAttrs.h`, then `CoralNPUCodegenDialect.h.inc`                                                                                |
| `CoralNPUCodegenDialect.cpp` | `CoralNPUCodegenDialect.cpp.inc`; in `initialize()`, `addAttributes<...>()` with `CoralNPUCodegenAttrs.cpp.inc` under `#define GET_ATTRDEF_LIST` |
| `CoralNPUCodegenAttrs.cpp`   | `CoralNPUCodegenDialect.h`, `mlir/IR/DialectImplementation.h`, `llvm/ADT/TypeSwitch.h`, `CoralNPUCodegenEnums.cpp.inc`, then `CoralNPUCodegenAttrs.cpp.inc` under `#define GET_ATTRDEF_CLASSES` (which defines `CoralNPUCodegenDialect::{parse,print}Attribute`) |

Header guards follow the existing style (`COMPILER_CODEGEN_CORALNPUCODEGENATTRS_H_`
and so on).

- `CoralNPUCodegenDialect.{h,cpp}`: the dialect class, as in the table.
- `CoralNPUCodegenAttrs.{h,cpp}`: the attribute. The `.cpp` defines

  ```cpp
  LogicalResult PipelineAttr::buildPipeline(
      OpPassManager &pm,
      const iree_compiler::CodegenPipelineOptions *options) const;
  ```

  It does what `buildCPUPipeline` does for `DoubleTilingExpert`
  (`Passes.cpp:760-779`): `dyn_cast_if_present<CPUCodegenPipelineOptions>`,
  default `LLVMCPUPipelineOptions` if that is null, `failure()` only if the
  lowering config is missing, else `addCoralNPUDTCMPipeline` and `success()`.
  `LLVMCPULowerExecutableTarget` turns a failure into "failed to build pass
  pipeline" (`:194-197`).
- `CoralNPUCodegenPipelines.{h,cpp}`:

  ```cpp
  // Builds the DTCMPromotedDoubleTiling pipeline. For now this is IREE's
  // DoubleTilingExpert; DMA Phase 2 part 3 replaces the body with a copy
  // that promotes the accumulator after cache-level tiling.
  void addCoralNPUDTCMPipeline(
      OpPassManager &funcPassManager,
      iree_compiler::IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
      const iree_compiler::LLVMCPUPipelineOptions &pipelineOpt);
  ```

  It has its own file so that part 3's copy and promotion pass don't grow the
  attribute file.
- `CoralNPUCodegenUtils.{h,cpp}`: the scope predicate (below).

**Bazel target.** `cc_library(name = "CoralNPUCodegen")` in
`compiler/Codegen/BUILD.bazel`, with `features = ["layering_check"]` like the
other packages. Each directly included header needs a direct dependency:

- the three `*IncGen` targets;
- `@iree_core//compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR:IREECodegenDialect`;
- `@iree_core//compiler/src/iree/compiler/Codegen/LLVMCPU`, already linked by
  `//compiler/Target:CoralNPUTarget`;
- `@iree_core//compiler/src/iree/compiler/Codegen/Utils`
  (`CodegenPipelineOptions.h`);
- `@llvm-project//llvm:Support`, `@llvm-project//mlir:IR`,
  `@llvm-project//mlir:LinalgDialect`, `@llvm-project//mlir:LinalgInterfaces`
  (both are listed by IREE's `Codegen/Dialect/CPU/IR:IREECPUDialect`),
  `@llvm-project//mlir:Pass` and `@llvm-project//mlir:Support`.

**Registration.**

- `CoralNPUSession::onRegisterDialects` (`CoralNPUTarget.cpp:82-84`, now a
  `// TODO`): `registry.insert<CoralNPUCodegen::CoralNPUCodegenDialect>()`.
  This lets `iree-compile --compile-from=...` and `iree-opt` parse the
  attribute.
- `CoralNPUTargetBackend::getDependentDialects`
  (`CoralNPUTargetBackend.cpp:346-354`): add the dialect next to
  `IREECPUDialect`.

**The flag.** In `CoralNPUOptions` (`CoralNPUTargetBackend.h:47-61`) add
`bool dtcmPromotion = false;`, bound next to `coralnpu-dtcm-size-kb`
(`:109-111`):

```cpp
binder.opt<bool>(
    "coralnpu-dtcm-promotion", dtcmPromotion, llvm::cl::cat(category),
    llvm::cl::desc("Keep the cache-level tiles of contractions in DTCM "
                   "(experimental; default: false)"));
```

Until part 3 the flag only switches the attribute. The description already
states the intent, and `README.md` says the flag has no effect on generated
code yet.

**The selection pass.** Declared in `compiler/Transforms/Passes.td`:

```tablegen
def CoralNPUSelectCodegenPipeline : InterfacePass<
    "coralnpu-select-codegen-pipeline", "mlir::FunctionOpInterface"> {
  let summary = "Selects the CoralNPU codegen pipeline for DTCM promotion";
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUSelectCodegenPipelinePass()
  }];
  let dependentDialects = [
    "::mlir::coralnpu_compiler::CoralNPUCodegen::CoralNPUCodegenDialect"
  ];
}
```

- `compiler/Transforms/Passes.h` declares
  `std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
  createCoralNPUSelectCodegenPipelinePass();`, like the Workgroup pass
  (`Passes.h:63-64`). `Passes.h` needs no new include. The dependent dialect
  is named only in the generated pass base class, so only
  `CoralNPUSelectCodegenPipeline.cpp` includes
  `compiler/Codegen/CoralNPUCodegenDialect.h`. That matches the Register pass:
  it depends on `IREECPUDialect`, and only its `.cpp` includes that header
  (`CoralNPUTileSizeSelectionRegister.cpp:29`).
- `CoralNPUSelectCodegenPipeline.cpp` also includes
  `compiler/Codegen/CoralNPUCodegenUtils.h`,
  `iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h`,
  `iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h`
  (`getCompilationInfo` / `setCompilationInfo`, `:191-197`),
  `mlir/IR/Remarks.h` and `llvm/Support/DebugLog.h`. The Transforms library
  already depends on everything except `//compiler/Codegen:CoralNPUCodegen`.
- The pass is registered with the others by `registerCoralNPUPasses()`
  (`Passes.h:33`), so `iree-opt --coralnpu-select-codegen-pipeline` works.
- `//compiler/Transforms:Transforms` gains the `//compiler/Codegen:CoralNPUCodegen`
  dependency.
- The pass has no options. The backend adds it after
  `CoralNPUTileSizeSelectionWorkgroup` and before `MaterializeUserConfigs`
  (`CoralNPUTargetBackend.cpp:414-416`), **only when**
  `options_.dtcmPromotion` is set. With the flag off the pass manager is
  therefore unchanged.

The pass walks the function and, for each op where `getCompilationInfo(op)`
is non-null (one root per function, set by the Register pass), does the
following:

1. Skips the op unless its pipeline is `#iree_cpu.pipeline<DoubleTilingExpert>`.
2. Skips it unless `isDTCMPromotionCandidate(op)` holds (below).
3. Builds a new `TranslationInfoAttr` with
   `#coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>` and the original
   codegen spec, workgroup size, subgroup size and **configuration
   dictionary**, as the DTCM pass does at `:816-819`
   (`TranslationInfoAttr::get(context, pipeline, codegenSpec, workgroupSize,
   subgroupSize, configuration)`). The dictionary carries
   `enable_loop_peeling`, which `LowerExecutableTarget` reads. Then
   `setCompilationInfo(op, CompilationInfoAttr::get(ctx, <same lowering
   config>, <new translation info>))`.

**Remark.** A root that is a contraction or a convolution
(`linalg::isaContractionOpInterface` or `linalg::isaConvolutionOpInterface`,
the same test as `CoralNPUTileSizeSelectionDTCM.cpp:684-686`) and fails step 2
is the only case where promotion was plausible but doesn't apply. The pass
reports it with the opt-in
`mlir::remark::missed(op->getLoc(), mlir::remark::RemarkOpts::name("DTCMPromotion").category("CoralNPU").subCategory("SelectCodegenPipeline"))`
(`mlir/IR/Remarks.h:726`), with a reason, plus `LDBG()`, so normal compilation
stays silent. Every other op (elementwise, transpose, reduce, and anything on
another pipeline) is skipped without a remark. `RemarkOpts::name`, `category`
and `subCategory` are the builders in `Remarks.h:105-124`.

**The scope predicate.** It lives in `compiler/Codegen/CoralNPUCodegenUtils.h`
so that parts 3 and 5 use the same check (AGENTS.md, *Keep cross-component
predicates synchronized*):

```cpp
// Returns true if |op| is a root that DMA Phase 2 keeps in DTCM: a static
// contraction with M and K dimensions (matmul, batch_matmul,
// batch_reduce_matmul, matvec, batch_matvec and their generic forms) that
// is not in mmt4d layout. Peers: CoralNPUSelectCodegenPipeline, the
// promotion pass (part 3) and promotion-aware tile selection (part 5). The
// mmt4d test mirrors setContractVectorSizes in
// CoralNPUTileSizeSelectionRegister.cpp; keep the two in sync.
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
  follow-up (Part 3).
- **Out, vecmat and dot.** They have no `m`. Phase 2 covers matmul and matvec
  shapes only.

**IR example.** `--compile-to=executable-configurations` with the flag on,
for a 64x64x64 f32 matmul:

IREE prints `translation_info` through an alias (`IREECodegenDialect.cpp:27-29`),
so the pipeline appears in the alias definition at the top of the output:

```mlir
// Before (or with the flag off):
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
...
func.func @matmul_dispatch_0_matmul_64x64x64_f32() attributes {translation_info = #translation} {

// After:
#translation = #iree_codegen.translation_info<pipeline = #coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>, {enable_loop_peeling}>
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
- *Lost verification.* For opted-in dispatches, `LLVMCPUSelectLoweringStrategy`
  no longer runs `verifyMultiTilingExpertPassPipelineConfig`, which is
  `static` (`LLVMCPUSelectLoweringStrategy.cpp:55`, `:283-285`). The verifier
  only rejects configs, so a config that passed before still lowers the same
  way, and code for valid configs can't change. An invalid config from a
  CoralNPU tile-pass bug would be caught later and less clearly. The flag is
  off by default; this is recorded in Part 3.
- *Other backends.* SPIR-V, VMVX and GPU call `buildPipeline` with their own
  option types. The attribute is only ever placed on CoralNPU functions, so
  that can't happen. If it did, the default options would apply, as in
  `buildCPUPipeline`.
- *Linking.* `--coralnpu-link-executables` links per variant. Translation
  info is per function and isn't used during linking, so mixing pipelines in
  one executable is fine.
- *Plugin not loaded.* If the plugin isn't loaded, nothing can produce the
  attribute, and printed IR that contains it fails to parse with "dialect
  `coralnpu_codegen` not found". That is expected.

### 1.5 Verification Plan

Two lit files, both picked up by the `glob` in `tests/transforms/BUILD.bazel`.
Neither uses `%iree_compile`. That substitution targets the host CPU as well as
CoralNPU (`tests/transforms/lit.cfg.py:44-49`), and the host variant also
prints `#iree_cpu.pipeline<DoubleTilingExpert>`.

**`tests/transforms/select_codegen_pipeline.mlir`**: the pass alone, through
`iree-opt`. It follows `tile_size_selection_register.mlir`. The input is
hand-written IR: a `hal.executable` with a `coralnpu` target, whose ops carry
`compilation_info` with `#iree_cpu.pipeline<DoubleTilingExpert>`. `iree-opt`
registers plugin dialects (`IREEOptToolEntryPoint.cpp:98`), so this file also
tests that `#coralnpu_codegen.pipeline` parses and prints.

```
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-select-codegen-pipeline)))))" %s | FileCheck %s
// RUN: iree-opt --split-input-file %s | FileCheck %s --check-prefix=PARSE
```

- **Swapped:** static `linalg.matmul`, `linalg.batch_matmul`,
  `linalg.batch_reduce_matmul`, `linalg.matvec`, `linalg.batch_matvec`, and a
  `linalg.generic` matmul. Each gets
  `#coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>`. A matmul whose
  translation config has `enable_loop_peeling` keeps it.
- **Not swapped:**
  - `linalg.add`, `linalg.vecmat`, and a `linalg.conv_2d_nhwc_fhwc` with a 3x3
    filter. They keep `DoubleTilingExpert`.
  - A dynamic-shape `linalg.matmul` with a hand-written `compilation_info`,
    which reaches the predicate only through `iree-opt` (see *Dynamic shapes*).
  - A `linalg.quantized_matmul`.
  - A matmul that already uses `ConvTileAndDecomposeExpert`, which is left alone
    because of step 1.
- **Parse round trip** (`PARSE`): one split already carries
  `#coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>`, and the check is that
  it prints back unchanged.
- **Checks:** each split is one function with a unique name, so `CHECK-LABEL`
  can't match a longer name. The aliases (`#translation`, `#compilation`) are
  captured with `CHECK-DAG: #[[$TI:.+]] = #iree_codegen.translation_info<pipeline = #coralnpu_codegen.pipeline<DTCMPromotedDoubleTiling>`
  and matched on the op's `compilation_info`.

**`tests/transforms/select_codegen_pipeline_e2e.mlir`**: the backend flow. It
has only a few functions, so that unrelated lowering problems can't fail it.
Each function becomes its own dispatch.

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
// RUN:   --mlir-print-ir-before=iree-llvmcpu-lower-executable-target %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LOWER
```

- No `--coralnpu-roofline-speedup-threshold`: with no host device,
  `CoralNPUAffinityAnnotation.cpp:302-307` always picks CoralNPU.
- **Inputs:**
  - `@mm_f32`, a 64x64x64 f32 matmul;
  - `@mm_i8_n80`, an i8 matmul with M = 64, N = 80, K = 64. N = 80 is a partial
    64-column Zvt tile (the example in `matrix_codegen.md`), so it gets
    `enable_loop_peeling`;
  - `@add_f32`, a static `linalg.add`.
- **`CHECK`:** `@mm_f32` and `@mm_i8_n80` carry the new pipeline. The
  `@mm_i8_n80` alias also has `enable_loop_peeling`. `@add_f32` keeps
  `DoubleTilingExpert`. All checks use `CHECK-DAG` alias captures, as above.
- **`OFF`:** `CHECK-NOT: coralnpu_codegen`.
- **`RVV`:** `@mm_f32` still carries the new pipeline.
- **`LOWER`:** the IR dumped before `LowerExecutableTarget` still has
  `#coralnpu_codegen.pipeline` on `@mm_f32`'s function. The compile also has
  to succeed, which exercises `buildPipeline`.
- **To confirm on the first build:** that IREE leaves these ops as written
  before configuration (in particular that it doesn't rewrite the matmuls
  into another op). If an expectation turns out wrong, record it in 2.3 and
  adjust the input rather than the check.

**Identical code.**

- **Inputs:** every `*.mlir` under
  `tests/models/stablehlo/generated_{f32,bf16,i8,i16,i32}` and
  `tests/models/linalg/generated_{f32,bf16,i8,i16,i32}`, about 3,300 files. bf16
  is the case that exercises the native BF16 converts.
- **Configurations:** four per input, about 13,000 compiles in all:
  default features and RVV-only, each with and without the flag.
- **Command,** the model tests' own target flag
  (`tests/models/stablehlo/defs.bzl:7`) plus the variable parts:

  ```
  iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
    [--coralnpu-dtcm-promotion] [--coralnpu-target-cpu-features=<rvv-only>] \
    --iree-hal-dump-executable-binaries-to=<out>/<features>/<flag>/<suite>/<input>/ \
    <input> -o <out>/<features>/<flag>/<suite>/<input>.vmfb
  ```

  `<features>` is `zvt` or `rvv`, and `<flag>` is `on` or `off`.
- **Pass criterion:** for each feature set, `diff -r` of
  `<out>/<features>/on` and `<out>/<features>/off` over the dumped ELFs reports
  no differences. A `.vmfb` difference is reported but isn't a failure until
  the `.vmfb` output is shown to be deterministic.
- **Where it runs:** a script in `agent-temp/` (created if it doesn't exist),
  in parallel with `xargs -P "$(nproc)"`. No test files change.

**Model check tests.** Byte-identical ELFs give identical results.
`bazel test --config=dev //tests:ci` must still pass with the flag at its
default.

**Register & spill verification.** Not needed: no code changes.

---

## Part 2: Implementation Plan

### 2.1 Expected Affected Files

- `compiler/Codegen/CoralNPUCodegenDialect.td`, `CoralNPUCodegenEnums.td`,
  `CoralNPUCodegenAttrs.td`: new; dialect, enum, attribute.
- `compiler/Codegen/CoralNPUCodegenDialect.{h,cpp}`: new; dialect class.
- `compiler/Codegen/CoralNPUCodegenAttrs.{h,cpp}`: new;
  `PipelineAttr::buildPipeline`.
- `compiler/Codegen/CoralNPUCodegenPipelines.{h,cpp}`: new;
  `addCoralNPUDTCMPipeline`.
- `compiler/Codegen/CoralNPUCodegenUtils.{h,cpp}`: new;
  `isDTCMPromotionCandidate`.
- `compiler/Codegen/BUILD.bazel`, `compiler/Codegen/CMakeLists.txt`: new.
- `compiler/CMakeLists.txt`: `add_subdirectory(Codegen)` before `Transforms`.
- `compiler/Transforms/Passes.td`, `Passes.h`: declare
  `CoralNPUSelectCodegenPipeline`.
- `compiler/Transforms/CoralNPUSelectCodegenPipeline.cpp`: new; the pass.
- `compiler/Transforms/BUILD.bazel`, `CMakeLists.txt`: the new source and the
  `//compiler/Codegen:CoralNPUCodegen` dependency.
- `compiler/Target/CoralNPUTargetBackend.h`: the `dtcmPromotion` option.
- `compiler/Target/CoralNPUTargetBackend.cpp`: the dependent dialect and the
  conditional pass.
- `compiler/Target/CoralNPUTarget.cpp`: `onRegisterDialects`.
- `compiler/Target/BUILD.bazel`, `CMakeLists.txt`: the
  `//compiler/Codegen:CoralNPUCodegen` dependency.
- `tests/transforms/select_codegen_pipeline.mlir`: new lit test (the pass,
  through `iree-opt`, plus parsing).
- `tests/transforms/select_codegen_pipeline_e2e.mlir`: new lit test (the
  backend flow).
- `README.md`: document `--coralnpu-dtcm-promotion`.

### 2.2 Implementation Tasks (`code` step)

- [ ] **Task 1**: The `compiler/Codegen` library: the three `.td` files, the
  generated targets, the dialect and attribute classes, `buildPipeline`, and
  `addCoralNPUDTCMPipeline` delegating to `addMultiTilingExpertPassPipeline`
  (1.4).
  - *Target*: `compiler/Codegen/*`
  - *Verify*: `bazel build --config=dev //compiler/Codegen:CoralNPUCodegen`
- [ ] **Task 2**: `isDTCMPromotionCandidate` in `CoralNPUCodegenUtils`.
  - *Target*: `compiler/Codegen/CoralNPUCodegenUtils.{h,cpp}`,
    `compiler/Codegen/BUILD.bazel`
  - *Verify*: `bazel build --config=dev //compiler/Codegen:CoralNPUCodegen`
- [ ] **Task 3**: Register the dialect in `onRegisterDialects` and
  `getDependentDialects`, and add the `//compiler/Codegen:CoralNPUCodegen`
  dependency to `//compiler/Target:CoralNPUTarget`.
  - *Target*: `compiler/Target/CoralNPUTarget.cpp`,
    `CoralNPUTargetBackend.cpp`, `compiler/Target/BUILD.bazel`
  - *Verify*: `bazel build --config=dev @iree_core//tools:iree-compile
    @iree_core//tools:iree-opt` (parsing is exercised in Task 6)
- [ ] **Task 4**: The `CoralNPUSelectCodegenPipeline` pass. That covers its
  `Passes.td` entry, the `Passes.h` declaration (no new include), the `.cpp`
  with the includes listed in 1.4, the `//compiler/Codegen:CoralNPUCodegen`
  dependency in `compiler/Transforms/BUILD.bazel`, and the
  missed-optimization remark for contraction and convolution roots.
  - *Target*: `compiler/Transforms/*`
  - *Verify*: `bazel build --config=dev //compiler/Transforms`
- [ ] **Task 5**: `--coralnpu-dtcm-promotion`, and the conditional pass in
  `buildConfigurationPassPipeline`.
  - *Target*: `compiler/Target/CoralNPUTargetBackend.{h,cpp}`
  - *Verify*: `bazel build --config=dev @iree_core//tools:iree-compile`
- [ ] **Task 6**: The two lit tests (1.5). Record in 2.3 any input that IREE
  rewrites before configuration.
  - *Target*: `tests/transforms/select_codegen_pipeline.mlir`,
    `tests/transforms/select_codegen_pipeline_e2e.mlir`
  - *Verify*: `bazel test --config=dev //tests/transforms:lit`
- [ ] **Task 7**: The identical-code check over all five data types in the
  stablehlo and linalg suites, with the default and RVV-only features: about
  13,000 parallel compiles (1.5). Record the result in 2.3.
  - *Target*: an `agent-temp/` script (not committed)
  - *Verify*: `diff -r` over the dumped ELFs reports no differences
- [ ] **Task 8**: The regression check.
  - *Verify*: `bazel test --config=dev //tests:ci`
- [ ] **Task 9 (Build & Doc Sync)**: after review, mirror the Bazel changes in
  CMake:
  - `compiler/CMakeLists.txt`;
  - `compiler/Codegen/CMakeLists.txt`, with one `iree_tablegen_library` per
    generated target and `iree_cc_library(NAME CoralNPUCodegen)`, which gives
    `coralnpu_compiler::Codegen::CoralNPUCodegen`. `OUTS` paths follow
    `compiler/Transforms/CMakeLists.txt:15-22`, for example
    `--gen-attrdef-decls compiler/Codegen/CoralNPUCodegenAttrs.h.inc`. The
    `.td` includes need the repo root (for `compiler/Codegen/...`) and IREE's
    `compiler/src` (for `iree/...`) on the TableGen include path; check that
    `iree_tablegen_library` already provides both, and add `TD_INCLUDES` if it
    doesn't;
  - `compiler/Transforms/CMakeLists.txt`;
  - `compiler/Target/CMakeLists.txt`.

  Then document the flag in `README.md` next to `--coralnpu-fast-math`
  (`README.md:248`). `refresh_compile_commands` (`BUILD.bazel:23-37`) lists
  only top-level targets, and `//compiler/Target:CoralNPUTarget` already
  reaches the new library, so it doesn't change.

### 2.3 Implementation Notes & Adaptations

- *(None yet)*

### 2.4 Documentation Archive Checklist (`archive` step — Pre-Submit)

- [ ] Add a short "Codegen pipeline selection" section to
      `docs/design/tiling_levels_explained.md`. It says where
      `translation_info` comes from and when it is `#coralnpu_codegen.pipeline`,
      and explains why the attribute delegates rather than copies. Update
      `docs/design/INDEX.md` with the new source paths.
- [ ] Move this document from `docs/design/changes/coralnpu_codegen_pipeline.md`
      to `docs/design/archive/coralnpu_codegen_pipeline.md` using `git mv`.
- [ ] Format modified code with `./scripts/format-code.sh`.

---

## Part 3: Follow-Up Items (For Reviewer Approval -> Buganizer `564546727`)

- [ ] **Copy the multi-tiling expert for DTCM promotion** (`P1` |
      `compiler/Codegen/CoralNPUCodegenPipelines.cpp`): DMA Phase 2 part 3
      replaces `addCoralNPUDTCMPipeline`'s body with a copy of
      `addMultiTilingExpertPassPipeline` (`LLVMCPU/Passes.cpp:170-269`) and
      inserts the promotion pass after `CacheReductionTiles`. It must
      re-create the `static` `addTileAndDistributePasses` (`:71-86`), and it
      has to read the `static` `clFailOnLargeVector` (`:47`) by name through
      `llvm::cl::getRegisteredOptions()`.
- [ ] **Declare `buildLLVMCPUVectorLoweringPipeline` in IREE** (`P1` |
      `iree-v3.12.0-0011-*.patch`): the copy needs it (`Passes.cpp:92`, no
      header declaration). The choice is a one-line, upstreamable patch to
      `LLVMCPU/Passes.h` or a local declaration; it needs explicit approval
      under AGENTS.md *Third-party code*.
- [ ] **Drift guard for the copied pipeline** (`P1` | new C++ test): build
      IREE's expert and the CoralNPU copy with the same options and config,
      compare `printAsTextualPipeline` after dropping CoralNPU-only passes, so
      that an IREE bump which changes the expert fails a test.
- [ ] **Quantized matmuls in scope** (`P3` |
      `compiler/Codegen/CoralNPUCodegenUtils.cpp`): `linalg.quantized_matmul`
      and `quantized_batch_matmul` fail `isaContractionOpInterface`. If one
      is ever seen in a CoralNPU dispatch, extend the predicate (for example,
      with the Register pass's matmul classification) and add a lit case.
- [ ] **Restore config verification for opted-in dispatches** (`P3` |
      `compiler/Transforms/CoralNPUSelectCodegenPipeline.cpp`):
      `verifyMultiTilingExpertPassPipelineConfig` is skipped for non-CPU
      pipelines and is `static`. Either upstream an export of it or check the
      same invariants before swapping the attribute.
