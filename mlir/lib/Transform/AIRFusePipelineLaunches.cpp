//===- AIRFusePipelineLaunches.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// air-fuse-pipeline-launches: co-locate a DECLARED group of air.launch ops
// into ONE air.launch holding ONE air.segment.
//
// WHY FUSING INTO ONE SEGMENT IS THE OPERATION
//   Each air.launch lowers to its own aie.device, so several launches are
//   time-multiplexed at segment granularity and on-chip residency holds only
//   WITHIN a segment. A pipeline whose stages hand off through L1->L1
//   air.channels is therefore not merely slower when its stages sit in
//   different launches -- the declared edges span devices. That is the
//   discriminator doc 23 sets for refuse-versus-skip: declining to fuse leaves
//   a program that is wrong, not one that is correct and unoptimized (unlike
//   air-label-scf-for-to-ping-pong, where declining leaves a correct
//   single-buffered loop). So this pass REFUSES a malformed group rather than
//   silently leaving it alone.
//
// DECLARED, NOT DERIVED
//   The pass never decides which launches belong together, never infers a
//   stage order, and never rewrites one staging construction into another. It
//   reads air.pipeline_group / air.pipeline_stage / air.staging (see
//   AIRDialect.h) and fuses exactly what was declared. Deriving the grouping
//   would need the unsound-without-H2 dataflow analysis that doc 16 sized H8
//   as "large"; declaring it costs a builder two attributes.
//
// WHAT air.staging DOES HERE
//   Nothing is rewritten from it -- picking "memtile" over "accum_in_place" is
//   picking a different loop construction, which the builder emits. The pass
//   CHECKS the claim against the IR, because both constructions compile and
//   compute the right numbers whether or not they actually stage anything:
//   an accumulator allocated at herd scope instead of inside the K loop still
//   produces correct output, at a DDR round trip per step, and no numeric gate
//   can see the difference (doc 22, "Clause 3 is the phase"). Two of the four
//   cells in that document's table would have shipped as working code.
//
// WHAT THIS PASS DOES NOT DO
//   It does not check the per-column shim MM2S budget. Fusing N stages into
//   one segment ADDS their per-column L3-facing demand (doc 23), and that is a
//   real constraint on the fused result -- but it is only countable on the
//   ROUTED design, long after this pass. It is counted by
//   ffn_resident_structure.py's census, which the pipeline-fusion gate reuses
//   rather than approximating here with a third, weaker counter.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRFusePipelineLaunches.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "air-fuse-pipeline-launches"

using namespace mlir;

namespace xilinx {
namespace air {

namespace {

// One declared stage: its launch, its stage index, and its staging claim.
struct PipelineMember {
  air::LaunchOp launch;
  int64_t stage;
  StringRef staging; // empty when undeclared
};

// The single air.segment a stage's launch body must hold. Returns nullptr and
// emits the reason when the body is not exactly {one air.segment, terminator}.
//
// The restriction is deliberate and narrow. Anything else in a launch body is
// L3-scope work whose ordering against the other stages this pass has no
// declared basis to choose, and guessing would be exactly the derivation H8
// was scoped away from.
static air::SegmentOp getSoleSegment(air::LaunchOp launch) {
  Block &body = launch.getBody().front();
  air::SegmentOp seg = nullptr;
  for (Operation &op : body) {
    if (isa<air::LaunchTerminatorOp>(op))
      continue;
    if (auto s = dyn_cast<air::SegmentOp>(op)) {
      if (seg) {
        launch.emitError("air.launch in pipeline group '")
            << launch->getAttrOfType<StringAttr>(attrs::PipelineGroup).getValue()
            << "' holds more than one air.segment; a pipeline stage must hold "
               "exactly one, since fusion co-locates the stages into a single "
               "segment";
        return nullptr;
      }
      seg = s;
      continue;
    }
    launch.emitError("air.launch in pipeline group '")
        << launch->getAttrOfType<StringAttr>(attrs::PipelineGroup).getValue()
        << "' holds '" << op.getName()
        << "' beside its air.segment; a pipeline stage's launch body must be "
           "exactly one air.segment, because this pass has no declared "
           "ordering for launch-scope work across stages";
    return nullptr;
  }
  if (!seg)
    launch.emitError("air.launch in pipeline group '")
        << launch->getAttrOfType<StringAttr>(attrs::PipelineGroup).getValue()
        << "' holds no air.segment; there is no stage body to fuse";
  return seg;
}

// Does this region hold the accumulator-ring shape air-hoist-dma-in-accum-
// pattern matches? That is: an scf.for whose body allocates a buffer that is
// BOTH the destination of one air.dma_memcpy_nd and the source of another.
//
// This encodes doc 22's two measured conditions in one predicate:
//   - the same memref on both sides of the round trip (the IN-PLACE kernel;
//     areSymmetricDmaOps requires it, and the two-buffer pAcc/C form never
//     matches at any alloc site), and
//   - the alloc INSIDE the loop (isIncomingDmaOp / isOutgoingDmaOp both
//     require the DMA to be tied to an in-loop alloc/dealloc).
// Allocate at herd scope -- the natural way -- and the ring does not form,
// measured 4 -> 4. This predicate is deliberately a NECESSARY condition on the
// declaration, not a promise the hoist will fire: it refuses the constructions
// that provably cannot ring, which is where the silent losses were.
static bool hasInLoopSymmetricDmaPair(Region &region) {
  bool found = false;
  region.walk([&](scf::ForOp forOp) {
    if (found)
      return WalkResult::interrupt();
    for (Operation &op : forOp.getBody()->getOperations()) {
      auto alloc = dyn_cast<memref::AllocOp>(&op);
      if (!alloc)
        continue;
      Value buf = alloc.getResult();
      bool isDmaDst = false, isDmaSrc = false;
      for (Operation *user : buf.getUsers()) {
        auto dma = dyn_cast<air::DmaMemcpyNdOp>(user);
        if (!dma)
          continue;
        // air.dma_memcpy_nd's first variadic operand group is the destination
        // and the second the source; compare by identity rather than position
        // so a shape change upstream cannot silently flip the test.
        if (dma.getDstMemref() == buf)
          isDmaDst = true;
        if (dma.getSrcMemref() == buf)
          isDmaSrc = true;
      }
      if (isDmaDst && isDmaSrc) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// Does this segment body allocate an L2 (memtile) buffer?
static bool allocatesL2(air::SegmentOp seg) {
  bool found = false;
  seg.getBody().walk([&](memref::AllocOp alloc) {
    auto memTy = llvm::dyn_cast<MemRefType>(alloc.getResult().getType());
    if (!memTy)
      return WalkResult::advance();
    auto space = llvm::dyn_cast_or_null<IntegerAttr>(memTy.getMemorySpace());
    if (space && space.getInt() == (int)air::MemorySpace::L2) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Check a stage's air.staging claim against what the builder actually emitted.
static LogicalResult verifyStagingClaim(PipelineMember &m, air::SegmentOp seg) {
  if (m.staging.empty() || m.staging == attrs::StagingL1)
    return success();
  if (m.staging == attrs::StagingMemtile) {
    if (!allocatesL2(seg))
      return m.launch.emitError("stage ")
             << m.stage << " declares air.staging = \"" << attrs::StagingMemtile
             << "\" but its segment allocates no L2 buffer, so nothing is "
                "staged through the memtile. The declaration describes the "
                "construction the builder emitted; it does not request one.";
    return success();
  }
  if (m.staging == attrs::StagingAccumInPlace) {
    if (!hasInLoopSymmetricDmaPair(seg.getBody()))
      return m.launch.emitError("stage ")
             << m.stage << " declares air.staging = \""
             << attrs::StagingAccumInPlace
             << "\" but no loop in its segment allocates a buffer that is both "
                "the destination and the source of an air.dma_memcpy_nd. "
                "air-hoist-dma-in-accum-pattern forms the ring only for the "
                "IN-PLACE kernel with the accumulator allocated INSIDE the "
                "loop; either condition missing leaves a DDR round trip per "
                "step that every numeric gate passes (doc 22).";
    return success();
  }
  // Unreachable while the dialect verifier owns the value set; kept so an added
  // value cannot silently be accepted here.
  return m.launch.emitError("unhandled air.staging value \"") << m.staging << "\"";
}

class AIRFusePipelineLaunchesPass
    : public air::impl::AIRFusePipelineLaunchesPassBase<
          AIRFusePipelineLaunchesPass> {

public:
  AIRFusePipelineLaunchesPass() = default;
  AIRFusePipelineLaunchesPass(const AIRFusePipelineLaunchesPass &pass){};

  void runOnOperation() override;

private:
  // Returns failure on a malformed group; the group is left untouched.
  LogicalResult fuseGroup(StringRef groupName,
                          SmallVectorImpl<PipelineMember> &members);
};

LogicalResult
AIRFusePipelineLaunchesPass::fuseGroup(StringRef groupName,
                                       SmallVectorImpl<PipelineMember> &members) {
  // --- 1. The group must describe a pipeline -------------------------------
  llvm::sort(members, [](const PipelineMember &a, const PipelineMember &b) {
    return a.stage < b.stage;
  });
  for (unsigned i = 0; i < members.size(); i++) {
    if (members[i].stage == (int64_t)i)
      continue;
    if (i && members[i].stage == members[i - 1].stage)
      return members[i].launch.emitError("pipeline group '")
             << groupName << "' declares air.pipeline_stage = "
             << members[i].stage
             << " more than once; a group must cover 0.." << members.size() - 1
             << " exactly once, so that the fused stage order is the declared "
                "one and not an artifact of IR order";
    return members[i].launch.emitError("pipeline group '")
           << groupName << "' has no air.pipeline_stage = " << i
           << "; stages must cover 0.." << members.size() - 1
           << " with no gap, and this group jumps to " << members[i].stage;
  }

  // Every member must sit in the same block: the fused launch replaces them in
  // place, and members in different funcs (or different regions) cannot be
  // co-located without moving code across a boundary this pass does not own.
  Block *block = members.front().launch->getBlock();
  for (auto &m : members)
    if (m.launch->getBlock() != block)
      return m.launch.emitError("pipeline group '")
             << groupName
             << "' spans more than one block; its stages cannot be co-located "
                "into one segment";

  // Grids and async tokens: refuse rather than guess.
  SmallVector<air::SegmentOp> segments;
  for (auto &m : members) {
    if (m.launch.getNumDims() != 0)
      return m.launch.emitError("stage ")
             << m.stage << " of pipeline group '" << groupName
             << "' declares a launch grid of " << m.launch.getNumDims()
             << " dimension(s); fusing stages with independent grids is not "
                "defined, since the fused launch has one iteration space";
    if (m.launch.getAsyncToken() || !m.launch.getAsyncDependencies().empty())
      return m.launch.emitError("stage ")
             << m.stage << " of pipeline group '" << groupName
             << "' is async; run air-fuse-pipeline-launches BEFORE "
                "air-dependency, so that dependency construction sees the "
                "fused form";
    air::SegmentOp seg = getSoleSegment(m.launch);
    if (!seg)
      return failure();
    if (seg.getNumDims() != 0)
      return seg.emitError("stage ")
             << m.stage << " of pipeline group '" << groupName
             << "' declares a segment grid; the fused segment has one";
    if (seg.getAsyncToken() || !seg.getAsyncDependencies().empty())
      return seg.emitError("stage ")
             << m.stage << " of pipeline group '" << groupName
             << "' has an async segment; run this pass before air-dependency";
    if (failed(verifyStagingClaim(m, seg)))
      return failure();
    segments.push_back(seg);
  }

  // --- 2. Union the operands ----------------------------------------------
  // A value fed to two stages becomes ONE operand of the fused launch: keeping
  // duplicates would make the fused module differ from a hand-written one that
  // passes each buffer once, and would grow the launch's shim-facing ABI.
  MLIRContext *ctx = &getContext();
  OpBuilder builder(members.front().launch);

  llvm::MapVector<Value, unsigned> launchOperandIdx;
  SmallVector<Value> launchOperands;
  for (auto &m : members)
    for (Value v : m.launch.getKernelOperands())
      if (launchOperandIdx.insert({v, launchOperands.size()}).second)
        launchOperands.push_back(v);

  // Segment operands, expressed as OUTER values. Every segment operand is a
  // block argument of its own launch (the launch body is exactly the segment),
  // so it ties back to an outer value which the fused launch also carries.
  llvm::MapVector<Value, unsigned> segOperandIdx;
  SmallVector<Value> segOuterOperands;
  for (auto [m, seg] : llvm::zip(members, segments)) {
    for (Value v : seg.getKernelOperands()) {
      auto ba = dyn_cast<BlockArgument>(v);
      if (!ba || ba.getOwner()->getParentOp() != m.launch.getOperation())
        return seg.emitError("stage ")
               << m.stage << " of pipeline group '" << groupName
               << "' passes its segment a value that is not one of its "
                  "launch's arguments; the fused segment cannot be given an "
                  "equivalent operand";
      Value outer = m.launch.getTiedKernelOperand(ba);
      if (!outer)
        return seg.emitError("stage ")
               << m.stage << " of pipeline group '" << groupName
               << "' passes its segment a launch block argument with no tied "
                  "launch operand";
      if (segOperandIdx.insert({outer, segOuterOperands.size()}).second)
        segOuterOperands.push_back(outer);
    }
  }

  // --- 3. Build the fused launch and segment ------------------------------
  Location loc = members.front().launch.getLoc();
  auto fusedLaunch =
      air::LaunchOp::create(builder, loc, ValueRange{}, launchOperands);
  // Carry the stage-0 launch's remaining attributes: the pipeline markers are
  // CONSUMED here (erased, as the ping-pong labels are), so what survives is
  // whatever else the builder put on the launch.
  for (NamedAttribute na : members.front().launch->getAttrs()) {
    StringRef n = na.getName().strref();
    if (n == attrs::PipelineGroup || n == attrs::PipelineStage ||
        n == attrs::Staging)
      continue;
    if (n == "operandSegmentSizes" || n == "sym_name")
      continue;
    fusedLaunch->setAttr(na.getName(), na.getValue());
  }

  builder.setInsertionPointToStart(&fusedLaunch.getBody().front());

  // Map each outer value to the fused launch's block argument for it.
  DenseMap<Value, Value> outerToLaunchArg;
  for (auto [i, v] : llvm::enumerate(launchOperands))
    outerToLaunchArg[v] = fusedLaunch.getKernelArgument(i);

  SmallVector<Value> fusedSegOperands;
  for (Value outer : segOuterOperands)
    fusedSegOperands.push_back(outerToLaunchArg.lookup(outer));

  auto fusedSegment =
      air::SegmentOp::create(builder, loc, ValueRange{}, fusedSegOperands);
  // The fused segment's name is the earliest-stage segment that declares one.
  // A rule rather than a synthesis, so that fusing a pipeline whose stages
  // agree on a segment name reproduces the hand-written module exactly.
  for (air::SegmentOp seg : segments)
    if (auto sym = seg.getSymNameAttr()) {
      fusedSegment.setSymNameAttr(sym);
      break;
    }

  DenseMap<Value, Value> outerToSegArg;
  for (auto [i, v] : llvm::enumerate(segOuterOperands))
    outerToSegArg[v] = fusedSegment.getKernelArgument(i);

  // --- 4. Move the stage bodies in, in DECLARED order ---------------------
  Block &fusedSegBlock = fusedSegment.getBody().front();
  Operation *fusedSegTerminator = fusedSegBlock.getTerminator();
  for (auto [m, seg] : llvm::zip(members, segments)) {
    // Re-express this segment's block arguments as the fused segment's.
    for (auto [i, constBa] : llvm::enumerate(seg.getKernelArguments())) {
      BlockArgument ba = constBa;
      Value outer = m.launch.getTiedKernelOperand(
          cast<BlockArgument>(seg.getKernelOperand(i)));
      ba.replaceAllUsesWith(outerToSegArg.lookup(outer));
    }
    // Splice the body in ahead of the terminator, preserving stage order.
    Block &src = seg.getBody().front();
    Operation *srcTerminator = src.getTerminator();
    fusedSegBlock.getOperations().splice(fusedSegTerminator->getIterator(),
                                         src.getOperations(), src.begin(),
                                         srcTerminator->getIterator());
  }

  for (auto &m : members)
    m.launch->erase();

  (void)ctx;
  return success();
}

void AIRFusePipelineLaunchesPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Collect declared groups. Keyed by (parent op, group name) so two funcs may
  // each declare a pipeline of the same name without being merged across the
  // boundary -- and insertion-ordered, so a malformed group reports at a
  // deterministic place.
  llvm::MapVector<std::pair<Operation *, StringRef>,
                  SmallVector<PipelineMember>>
      groups;

  WalkResult collect = module.walk([&](air::LaunchOp launch) {
    auto group = launch->getAttrOfType<StringAttr>(attrs::PipelineGroup);
    if (!group)
      return WalkResult::advance();
    auto stage = launch->getAttrOfType<IntegerAttr>(attrs::PipelineStage);
    if (!stage) {
      launch.emitError("air.launch declares air.pipeline_group '")
          << group.getValue()
          << "' but no air.pipeline_stage; the pass will not infer a stage "
             "order from IR order, because the fused order decides which "
             "stage's output feeds which";
      return WalkResult::interrupt();
    }
    StringRef staging;
    if (auto s = launch->getAttrOfType<StringAttr>(attrs::Staging))
      staging = s.getValue();
    Operation *parent = launch->getParentOp();
    groups[{parent, group.getValue()}].push_back(
        {launch, stage.getInt(), staging});
    return WalkResult::advance();
  });
  if (collect.wasInterrupted())
    return signalPassFailure();

  for (auto &entry : groups)
    if (failed(fuseGroup(entry.first.second, entry.second)))
      return signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> createAIRFusePipelineLaunchesPass() {
  return std::make_unique<AIRFusePipelineLaunchesPass>();
}

} // namespace air
} // namespace xilinx
